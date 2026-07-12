from __future__ import annotations

import json
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from slimx_rag.chunk import HeuristicTokenCounter
from slimx_rag.core.hashing import (
    STRUCTURED_CHUNK_CONFIG_VERSION,
    chunk_config_fingerprint,
    make_chunk_id,
    structured_chunk_config_fingerprint,
)
from slimx_rag.document import DocumentSource, ParsedDocument, ParserRegistry
from slimx_rag.index import (
    INDEX_BUILD_RECEIPT_FILENAME,
    INDEX_SCHEMA_VERSION,
    build_index_build_receipt,
    build_index_signature,
    get_or_create_index_instance_id,
    load_index_build_receipt,
    locked_index_instance,
    write_index_build_receipt,
)
from slimx_rag.settings import ChunkSettings, EmbedSettings, IndexSettings, StructuredChunkSettings


class _NamedParser:
    def __init__(self, name: str, version: str) -> None:
        self.name = name
        self.version = version

    def supports(self, source: DocumentSource) -> bool:
        return False

    def parse(self, source: DocumentSource) -> ParsedDocument:
        raise AssertionError("not used by signature tests")


class _ExtractionParser(_NamedParser):
    def __init__(self, name: str, version: str, extraction: dict[str, object]) -> None:
        super().__init__(name, version)
        self._extraction = extraction

    def extraction_signature(self) -> dict[str, object]:
        return dict(self._extraction)


def _registry(*versions: tuple[str, str]) -> ParserRegistry:
    registry = ParserRegistry()
    for name, version in versions or (("native-text", "1"),):
        registry.register(_NamedParser(name, version))
    return registry


def _signature(
    *,
    index: IndexSettings | None = None,
    embed: EmbedSettings | None = None,
    chunk: ChunkSettings | None = None,
    structured: StructuredChunkSettings | None = None,
    dimension: int | None = 32,
    registry: ParserRegistry | None = None,
    schema_version: int = INDEX_SCHEMA_VERSION,
    engine_version: str = "0.2.7",
    instance_id: str = "idx_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    token_counter: HeuristicTokenCounter | None = None,
):
    return build_index_signature(
        index_settings=index or IndexSettings(),
        embed_settings=embed or EmbedSettings(dim=32),
        chunk_settings=chunk or ChunkSettings(),
        structured_chunk_settings=structured or StructuredChunkSettings(),
        embedding_dimension=dimension,
        index_instance_id=instance_id,
        structured_token_counter=token_counter or HeuristicTokenCounter(max_tokens=256),
        parser_registry=registry or _registry(),
        index_schema_version=schema_version,
        engine_version=engine_version,
    )


def test_index_signature_is_deterministic_and_wire_safe() -> None:
    first = _signature()
    second = _signature()

    assert first == second
    assert first.to_dict() == second.to_dict()
    assert first.signature_version == "index-signature-v1"
    assert len(first.compatibility_fingerprint) == 16
    assert first.engine == "slimx-rag"
    assert first.embedding_model == "hash-blake2b-v1"
    assert first.index_instance_id == "idx_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert first.backend_corpus_namespace == "local:jsonl-index"
    assert first.backend_namespace_version == "backend-namespace-v1"
    assert first.chunk_config_version == "chunk-v1"
    assert first.file_chunk_config_version == "structured-v2"
    assert first.file_chunk_effective_max_tokens == 256
    assert first.file_chunk_token_counter_version == "heuristic-counter-v1"
    assert first.index_schema_version == 1


def test_compatibility_fingerprint_changes_for_semantic_inputs() -> None:
    baseline = _signature().compatibility_fingerprint
    changes = [
        _signature(index=IndexSettings(backend="faiss")),
        _signature(instance_id="idx_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
        _signature(embed=EmbedSettings(provider="openai", model="embed-v2"), dimension=1536),
        _signature(dimension=64),
        _signature(chunk=ChunkSettings(chunk_size=801)),
        _signature(structured=StructuredChunkSettings(target_tokens=255)),
        _signature(token_counter=HeuristicTokenCounter(max_tokens=128)),
        _signature(registry=_registry(("native-text", "2"))),
        _signature(schema_version=2),
    ]

    assert all(signature.compatibility_fingerprint != baseline for signature in changes)


def test_effective_parser_dependency_changes_compatibility() -> None:
    def registry(extraction: dict[str, object]) -> ParserRegistry:
        result = ParserRegistry()
        result.register(_ExtractionParser("native-pdf", "1", extraction))
        return result

    pypdf_v5 = _signature(registry=registry({"backend": "pypdf", "backend_version": "5.9.0", "available": True}))
    pypdf_v6 = _signature(registry=registry({"backend": "pypdf", "backend_version": "6.0.0", "available": True}))
    unavailable = _signature(registry=registry({"backend": "unavailable", "backend_version": None, "available": False}))

    assert pypdf_v5.parser_config_fingerprint != pypdf_v6.parser_config_fingerprint
    assert pypdf_v5.compatibility_fingerprint != pypdf_v6.compatibility_fingerprint
    assert pypdf_v5.compatibility_fingerprint != unavailable.compatibility_fingerprint


def test_operational_and_inactive_settings_do_not_change_compatibility() -> None:
    baseline = _signature()
    operational = _signature(
        index=IndexSettings(top_k=99),
        embed=replace(
            EmbedSettings(dim=32),
            model="inactive-openai-model",
            device="cuda",
            batch_size=1,
            retries=9,
        ),
        engine_version="99.0.0",
    )

    assert operational.compatibility_fingerprint == baseline.compatibility_fingerprint
    assert operational.engine_version != baseline.engine_version


def test_vector_affecting_embedding_config_changes_compatibility() -> None:
    base = EmbedSettings(provider="hf", hf_model="org/model", revision="r1")
    baseline = _signature(embed=base, dimension=768).compatibility_fingerprint

    for changed in (
        replace(base, hf_model="org/model-v2"),
        replace(base, revision="r2"),
        replace(base, normalize_text=False),
        replace(base, max_chars=1000),
        replace(base, normalize_embeddings=False),
        replace(base, query_prefix="query: "),
        replace(base, document_prefix="passage: "),
    ):
        assert _signature(embed=changed, dimension=768).compatibility_fingerprint != baseline


def test_hf_runtime_identity_is_an_embedding_compatibility_input() -> None:
    class _Counter(HeuristicTokenCounter):
        def __init__(self, identity: str) -> None:
            super().__init__()
            self.identity = identity

    first = _signature(
        embed=EmbedSettings(provider="hf", hf_model="org/model"),
        dimension=768,
        token_counter=_Counter("hf:org/model@model:commit-a:tokenizer:t@commit-a:vocab:a"),
    )
    second = _signature(
        embed=EmbedSettings(provider="hf", hf_model="org/model"),
        dimension=768,
        token_counter=_Counter("hf:org/model@model:commit-b:tokenizer:t@commit-b:vocab:b"),
    )

    assert first.embedding_runtime_identity != second.embedding_runtime_identity
    assert first.embedding_config_fingerprint != second.embedding_config_fingerprint
    assert first.compatibility_fingerprint != second.compatibility_fingerprint


def test_structured_chunk_fingerprint_matches_public_versioned_file_contract() -> None:
    settings = StructuredChunkSettings()
    counter = HeuristicTokenCounter(max_tokens=256)
    canonical = structured_chunk_config_fingerprint(
        max_tokens=settings.max_tokens,
        effective_max_tokens=256,
        force_split_overlap_tokens=settings.force_split_overlap_tokens,
        target_tokens=settings.target_tokens,
        include_identity_prefix=settings.include_identity_prefix,
        token_counter_name=counter.name,
        token_counter_version=counter.version,
        token_counter_identity=counter.identity,
    )

    signature = _signature()
    assert STRUCTURED_CHUNK_CONFIG_VERSION == "structured-v2"
    assert signature.file_chunk_config_version == STRUCTURED_CHUNK_CONFIG_VERSION
    assert canonical == signature.file_chunk_config_fingerprint


def test_structured_v2_does_not_reuse_v1_fingerprint_or_chunk_id() -> None:
    settings = StructuredChunkSettings()
    counter = HeuristicTokenCounter(max_tokens=256)
    legacy_v1 = chunk_config_fingerprint(
        chunk_size=settings.max_tokens,
        chunk_overlap=settings.force_split_overlap_tokens,
        separators=(
            "structured-v1",
            str(settings.target_tokens),
            str(int(settings.include_identity_prefix)),
        ),
    )
    current_v2 = structured_chunk_config_fingerprint(
        max_tokens=settings.max_tokens,
        effective_max_tokens=256,
        force_split_overlap_tokens=settings.force_split_overlap_tokens,
        target_tokens=settings.target_tokens,
        include_identity_prefix=settings.include_identity_prefix,
        token_counter_name=counter.name,
        token_counter_version=counter.version,
        token_counter_identity=counter.identity,
    )

    assert legacy_v1 != current_v2
    shared_inputs = {
        "parent_id": "doc#p1#g0",
        "content_hash_value": "content-hash",
        "chunk_index": 0,
    }
    assert make_chunk_id(**shared_inputs, chunk_cfg_hash=legacy_v1) != make_chunk_id(
        **shared_inputs,
        chunk_cfg_hash=current_v2,
    )


def test_index_instance_id_is_persistent_and_distinguishes_new_volumes(tmp_path) -> None:
    first_path = tmp_path / "volume-a" / "index_instance_id"
    second_path = tmp_path / "volume-b" / "index_instance_id"

    first = get_or_create_index_instance_id(first_path)
    assert get_or_create_index_instance_id(first_path) == first
    assert get_or_create_index_instance_id(second_path) != first
    assert first_path.read_text(encoding="utf-8").strip() == first


def test_concurrent_first_use_converges_on_one_index_instance_id(tmp_path) -> None:
    path = tmp_path / "shared" / "index_instance_id"
    workers = 16
    barrier = threading.Barrier(workers)

    def create() -> str:
        barrier.wait()
        return get_or_create_index_instance_id(path)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        values = list(executor.map(lambda _index: create(), range(workers)))

    assert len(set(values)) == 1
    assert path.read_text(encoding="utf-8").strip() == values[0]
    assert not path.with_name(f".{path.name}.lock").exists()


def test_signature_never_serializes_backend_secrets() -> None:
    first = _signature(
        index=IndexSettings(
            backend="qdrant",
            backend_config={
                "collection": "research",
                "url": "https://user:password@qdrant.internal:6333",
                "api_key": "secret-one",
            },
        )
    )
    rotated = _signature(
        index=IndexSettings(
            backend="qdrant",
            backend_config={
                "collection": "research",
                "url": "http://rotated:credentials@qdrant.internal:6333?token=new",
                "api_key": "secret-two",
            },
        )
    )

    serialized = json.dumps(first.to_dict())
    assert "secret-one" not in serialized
    assert "password" not in serialized
    assert first.compatibility_fingerprint == rotated.compatibility_fingerprint
    assert first.backend_corpus_namespace == "qdrant:cluster:qdrant.internal:6333:collection:research"
    changed_collection = _signature(index=IndexSettings(backend="qdrant", backend_config={"collection": "other"}))
    assert changed_collection.compatibility_fingerprint != first.compatibility_fingerprint
    changed_cluster = _signature(
        index=IndexSettings(
            backend="qdrant",
            backend_config={"collection": "research", "url": "https://other.internal:6333"},
        )
    )
    assert changed_cluster.compatibility_fingerprint != first.compatibility_fingerprint


def test_pgvector_namespace_ignores_dsn_but_binds_schema_and_table() -> None:
    first = _signature(
        index=IndexSettings(
            backend="pgvector",
            backend_config={
                "dsn": "postgresql://user:secret-a@db.internal:5432/app",
                "schema": "rag",
                "table": "chunks",
            },
        )
    )
    rotated = _signature(
        index=IndexSettings(
            backend="pgvector",
            backend_config={
                "dsn": "postgresql://other:secret-b@db.internal:5432/app",
                "schema": "rag",
                "table": "chunks",
            },
        )
    )
    changed_table = _signature(
        index=IndexSettings(
            backend="pgvector",
            backend_config={
                "dsn": "postgresql://other:secret-b@db.internal:5432/app",
                "schema": "rag",
                "table": "other",
            },
        )
    )

    assert first.backend_corpus_namespace == "pgvector:cluster:db.internal:5432/app:table:rag.chunks"
    assert first.compatibility_fingerprint == rotated.compatibility_fingerprint
    assert first.compatibility_fingerprint != changed_table.compatibility_fingerprint
    assert "postgres://" not in json.dumps(first.to_dict())


def test_explicit_remote_corpus_namespace_is_stable_across_endpoint_rotation() -> None:
    first = _signature(
        index=IndexSettings(
            backend="qdrant",
            backend_config={
                "corpus_namespace": "production-research",
                "collection": "one",
                "url": "https://cluster-a.internal",
            },
        )
    )
    moved = _signature(
        index=IndexSettings(
            backend="qdrant",
            backend_config={
                "corpus_namespace": "production-research",
                "collection": "two",
                "url": "https://cluster-b.internal",
            },
        )
    )

    assert first.backend_corpus_namespace == "qdrant:namespace:production-research"
    assert first.compatibility_fingerprint == moved.compatibility_fingerprint


def test_metadata_whitelist_is_a_normalized_compatibility_input() -> None:
    baseline = _signature(index=IndexSettings(metadata_whitelist=["page", "title"]))
    reordered = _signature(index=IndexSettings(metadata_whitelist=["title", "page", "page"]))
    changed = _signature(index=IndexSettings(metadata_whitelist=["page"]))

    assert baseline.metadata_whitelist == ("page", "title")
    assert baseline.compatibility_fingerprint == reordered.compatibility_fingerprint
    assert baseline.compatibility_fingerprint != changed.compatibility_fingerprint


def test_embedding_dimension_precedence_is_canonical() -> None:
    from slimx_rag.index import resolve_embedding_dimension

    assert (
        resolve_embedding_dimension(
            emitted_dimension=3,
            backend_dimension=4,
            persisted_actual_dimension=5,
            configured_dimension=6,
        )
        == 3
    )
    assert (
        resolve_embedding_dimension(
            backend_dimension=4,
            persisted_actual_dimension=5,
            configured_dimension=6,
        )
        == 4
    )
    assert resolve_embedding_dimension(persisted_actual_dimension=5, configured_dimension=6) == 5
    assert resolve_embedding_dimension(configured_dimension=6) == 6


def test_build_receipt_roundtrip_and_tamper_detection(tmp_path) -> None:
    signature = _signature()
    path = tmp_path / INDEX_BUILD_RECEIPT_FILENAME
    receipt = write_index_build_receipt(
        path,
        signature,
        document_pipeline={"ingest_mode": "text", "parser": None},
    )

    loaded = load_index_build_receipt(path)
    assert loaded == receipt
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["index_signature"]["file_chunk_config_version"] == "structured-v2"
    data["index_signature"]["file_chunk_config_version"] = "structured-v1"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RuntimeError, match="fingerprint"):
        load_index_build_receipt(path)

    write_index_build_receipt(
        path,
        signature,
        document_pipeline={"ingest_mode": "text", "parser": None},
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    data["index_signature"]["embedding_dimension"] = 999
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RuntimeError, match="fingerprint"):
        load_index_build_receipt(path)

    write_index_build_receipt(
        path,
        signature,
        document_pipeline={"ingest_mode": "text", "parser": None},
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    data["index_signature"]["engine_version"] = "tampered"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RuntimeError, match="receipt fingerprint"):
        load_index_build_receipt(path)


def test_build_receipt_rejects_partial_signature() -> None:
    partial = replace(_signature(), signature_complete=False)

    with pytest.raises(ValueError, match="complete signature"):
        build_index_build_receipt(partial, document_pipeline={"ingest_mode": "text"})


def test_stale_crashed_instance_lock_is_recovered(tmp_path) -> None:
    path = tmp_path / "index_instance_id"
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.write_text(
        json.dumps(
            {
                "token": "dead-owner",
                "pid": 2_147_483_647,
                "hostname": socket.gethostname(),
                "created_at": 0,
            }
        ),
        encoding="utf-8",
    )
    old = time.time() - 3600
    os.utime(lock_path, (old, old))

    instance_id = get_or_create_index_instance_id(path)

    assert instance_id.startswith("idx_")
    assert not lock_path.exists()


def test_current_protocol_crashed_lock_is_recovered_without_age_delay(tmp_path) -> None:
    path = tmp_path / "index_instance_id"
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.write_text(
        json.dumps(
            {
                "token": "dead-owner",
                "pid": 2_147_483_647,
                "hostname": socket.gethostname(),
                "created_at": time.time(),
                "advisory_lock": True,
            }
        ),
        encoding="utf-8",
    )

    instance_id = get_or_create_index_instance_id(path)

    assert instance_id.startswith("idx_")
    assert not lock_path.exists()


def test_crash_during_lock_owner_initialization_is_recovered(tmp_path) -> None:
    path = tmp_path / "index_instance_id"
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.touch()
    abandoned = time.time() - 2
    os.utime(lock_path, (abandoned, abandoned))

    instance_id = get_or_create_index_instance_id(path)

    assert instance_id.startswith("idx_")
    assert not lock_path.exists()


def test_old_timestamp_does_not_steal_a_live_instance_lock(tmp_path) -> None:
    from slimx_rag.index import signature as signature_module

    path = tmp_path / "index_instance_id"
    lock_path = path.with_name(f".{path.name}.lock")
    with locked_index_instance(path, create=True):
        old = time.time() - 3600
        os.utime(lock_path, (old, old))

        assert signature_module._recover_stale_instance_lock(lock_path) is False
        assert lock_path.exists()

    assert not lock_path.exists()
