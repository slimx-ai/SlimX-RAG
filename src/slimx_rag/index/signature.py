"""Canonical, non-secret compatibility signature for a SlimX-RAG index.

The signature describes the inputs that determine document/chunk identity and vector
compatibility. Operational knobs such as top-k, device, retries, paths, and credentials
are intentionally absent.
"""

from __future__ import annotations

import json
import os
import secrets
import shlex
import socket
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from urllib.parse import unquote, urlsplit

from slimx_rag.chunk.tokenizer import TokenCounter
from slimx_rag.core.hashing import (
    DEFAULT_HASH_POLICY,
    STRUCTURED_CHUNK_CONFIG_VERSION,
    chunk_config_fingerprint,
    hash_text,
    structured_chunk_config_fingerprint,
)
from slimx_rag.document import ParserRegistry, get_default_registry
from slimx_rag.settings import ChunkSettings, EmbedSettings, IndexSettings, StructuredChunkSettings
from slimx_rag.utils.commons import _atomic_write_text
from slimx_rag.version import get_engine_version

from .types import INDEX_SCHEMA_VERSION

try:  # POSIX advisory locking makes stale-file recovery race-free and crash-releasing.
    import fcntl
except ImportError:  # pragma: no cover - Windows falls back to owner metadata + O_EXCL.
    fcntl = None  # type: ignore[assignment]

INDEX_SIGNATURE_VERSION = "index-signature-v1"
EMBEDDING_CONFIG_VERSION = "embedding-config-v1"
PARSER_CONFIG_VERSION = "parser-registry-v1"
BACKEND_NAMESPACE_VERSION = "backend-namespace-v1"
INDEX_SHAPING_VERSION = "index-shaping-v1"
ENGINE_NAME = "slimx-rag"
INDEX_INSTANCE_ID_FILENAME = "index_instance_id"
INDEX_BUILD_RECEIPT_FILENAME = "index_build_receipt.json"
INDEX_BUILD_RECEIPT_VERSION = "index-build-receipt-v1"
_INDEX_INSTANCE_ID_PREFIX = "idx_"
_LOCK_STALE_SECONDS = 60.0
_LOCK_INITIALIZATION_GRACE_SECONDS = 1.0


def _fingerprint(value: object) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hash_text(
        canonical,
        digest_size=DEFAULT_HASH_POLICY.config_fingerprint_digest_size,
    )


def _effective_embedding_model(settings: EmbedSettings) -> str:
    if settings.provider == "hf":
        return settings.hf_model
    if settings.provider == "hash":
        # ``model`` is an inactive OpenAI setting for the deterministic hash provider.
        return "hash-blake2b-v1"
    return settings.model


def _positive_dimension(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        dimension = int(value)
    except (TypeError, ValueError):
        return None
    return dimension if dimension > 0 else None


def resolve_embedding_dimension(
    *,
    emitted_dimension: object = None,
    backend_dimension: object = None,
    persisted_actual_dimension: object = None,
    configured_dimension: object = None,
) -> int | None:
    """Resolve corpus dimension in canonical evidence order.

    An emitted vector is direct evidence, followed by the loaded backend, a previously
    persisted *actual* dimension, and finally the configured fallback.
    """
    for candidate in (
        emitted_dimension,
        backend_dimension,
        persisted_actual_dimension,
        configured_dimension,
    ):
        dimension = _positive_dimension(candidate)
        if dimension is not None:
            return dimension
    return None


def _safe_namespace_component(value: object, *, label: str) -> str:
    component = str(value or "").strip()
    if not component:
        raise ValueError(f"{label} must be non-empty for an index signature")
    if len(component) > 255 or any(ord(char) < 32 or ord(char) == 127 for char in component):
        raise ValueError(f"{label} contains unsupported characters")
    return component


def backend_corpus_namespace(settings: IndexSettings) -> str:
    """Return the non-secret corpus namespace selected by a backend configuration."""
    backend = (settings.backend or "local").lower().strip()
    config = settings.backend_config or {}
    explicit = config.get("corpus_namespace")
    if explicit:
        namespace = _safe_namespace_component(explicit, label="corpus namespace")
        return f"{backend}:namespace:{namespace}"
    if backend == "qdrant":
        collection = _safe_namespace_component(config.get("collection"), label="qdrant collection")
        raw_url = str(config.get("url") or "http://localhost:6333")
        parsed = urlsplit(raw_url if "://" in raw_url else f"//{raw_url}")
        host = _safe_namespace_component(parsed.hostname or "localhost", label="qdrant host")
        try:
            port = parsed.port or 6333
        except ValueError as exc:
            raise ValueError("qdrant url contains an invalid port") from exc
        return f"qdrant:cluster:{host}:{port}:collection:{collection}"
    if backend == "pgvector":
        schema = _safe_namespace_component(config.get("schema") or "public", label="pgvector schema")
        table = _safe_namespace_component(config.get("table") or "slimx_vectors", label="pgvector table")
        dsn = str(config.get("dsn") or "").strip()
        if "://" in dsn:
            parsed = urlsplit(dsn)
            host = parsed.hostname or "localhost"
            try:
                port = parsed.port or 5432
            except ValueError as exc:
                raise ValueError("pgvector dsn contains an invalid port") from exc
            database = unquote(parsed.path.lstrip("/"))
        else:
            parts: dict[str, str] = {}
            for item in shlex.split(dsn):
                key, separator, value = item.partition("=")
                if separator:
                    parts[key.strip().lower()] = value.strip()
            host = parts.get("host") or "localhost"
            port = int(parts.get("port") or "5432")
            database = parts.get("dbname") or parts.get("database") or ""
        safe_host = _safe_namespace_component(host, label="pgvector host")
        safe_database = _safe_namespace_component(database, label="pgvector database")
        return f"pgvector:cluster:{safe_host}:{port}/{safe_database}:table:{schema}.{table}"
    if backend == "faiss":
        return "faiss:local-index"
    if backend == "local":
        return "local:jsonl-index"
    return f"{_safe_namespace_component(backend, label='index backend')}:default"


def _embedding_config(
    settings: EmbedSettings,
    *,
    model: str,
    dimension: int | None,
    runtime_identity: str | None,
) -> dict[str, object]:
    config: dict[str, object] = {
        "version": EMBEDDING_CONFIG_VERSION,
        "provider": settings.provider,
        "model": model,
        "dimension": dimension,
        "configured_dimension": settings.dim if settings.provider == "hash" else None,
        "runtime_identity": runtime_identity,
        "normalize_text": settings.normalize_text,
        "max_chars": settings.max_chars,
    }
    if settings.provider == "hf":
        config.update(
            {
                "normalize_embeddings": settings.normalize_embeddings,
                "query_prefix": settings.query_prefix,
                "document_prefix": settings.document_prefix,
                "revision": settings.revision,
            }
        )
    return config


def _index_shaping_config(settings: IndexSettings) -> tuple[tuple[str, ...] | None, str]:
    whitelist = (
        tuple(sorted({str(key).strip() for key in settings.metadata_whitelist if str(key).strip()}))
        if settings.metadata_whitelist
        else None
    )
    fingerprint = _fingerprint(
        {
            "version": INDEX_SHAPING_VERSION,
            "metadata_whitelist": list(whitelist) if whitelist is not None else "all",
        }
    )
    return whitelist, fingerprint


def _parser_config(registry: ParserRegistry) -> list[dict[str, object]]:
    # Registry order is semantic: the first parser that supports a source wins.
    records: list[dict[str, object]] = []
    for parser in registry.parsers():
        extraction_fn = getattr(parser, "extraction_signature", None)
        extraction = (
            extraction_fn()
            if callable(extraction_fn)
            else {"backend": "builtin", "backend_version": parser.version, "available": True}
        )
        records.append({"name": parser.name, "version": parser.version, "extraction": extraction})
    return records


@dataclass(frozen=True, slots=True)
class IndexSignature:
    """Wire-safe index compatibility contract owned by the SlimX-RAG engine."""

    signature_version: str
    signature_complete: bool
    compatibility_fingerprint: str
    engine: str
    engine_version: str
    index_instance_id: str | None
    vector_backend: str
    backend_namespace_version: str
    backend_corpus_namespace: str
    index_shaping_version: str
    index_shaping_fingerprint: str
    metadata_whitelist: tuple[str, ...] | None
    embedding_config_version: str
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int | None
    embedding_configured_dimension: int | None
    embedding_runtime_identity: str | None
    embedding_config_fingerprint: str
    parser_config_version: str
    parser_config_fingerprint: str
    chunk_config_version: str
    chunk_config_fingerprint: str
    text_chunk_config_fingerprint: str
    file_chunk_config_version: str
    file_chunk_config_fingerprint: str
    file_chunk_effective_max_tokens: int | None
    file_chunk_token_counter_name: str | None
    file_chunk_token_counter_version: str | None
    file_chunk_token_counter_identity: str | None
    id_schema_version: str
    content_hash_version: str
    index_schema_version: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_index_signature(
    *,
    index_settings: IndexSettings,
    embed_settings: EmbedSettings,
    chunk_settings: ChunkSettings,
    structured_chunk_settings: StructuredChunkSettings,
    embedding_dimension: int | None,
    index_instance_id: str | None,
    structured_token_counter: TokenCounter | None,
    signature_complete: bool = True,
    parser_registry: ParserRegistry | None = None,
    index_schema_version: int = INDEX_SCHEMA_VERSION,
    engine_version: str | None = None,
) -> IndexSignature:
    """Build the deterministic signature from the engine's real active settings.

    ``embedding_dimension`` is explicit because callers that initialized an embedder or
    emitted vectors should pass the runtime dimension, while config-only callers pass the
    configured dimension. Package version is provenance and is deliberately excluded from
    ``compatibility_fingerprint``; compatibility-changing code must bump one of the
    explicit protocol/config versions below.
    """
    registry = parser_registry or get_default_registry()
    model = _effective_embedding_model(embed_settings)
    embedding_runtime_identity = (
        structured_token_counter.identity
        if embed_settings.provider == "hf" and structured_token_counter is not None
        else None
    )
    corpus_namespace = backend_corpus_namespace(index_settings)
    metadata_whitelist, index_shaping_fingerprint = _index_shaping_config(index_settings)
    embedding_config_fingerprint = _fingerprint(
        _embedding_config(
            embed_settings,
            model=model,
            dimension=embedding_dimension,
            runtime_identity=embedding_runtime_identity,
        )
    )
    parser_config_fingerprint = _fingerprint(_parser_config(registry))
    text_chunk_fingerprint = chunk_config_fingerprint(
        chunk_size=chunk_settings.chunk_size,
        chunk_overlap=chunk_settings.chunk_overlap,
        separators=chunk_settings.separators,
    )
    effective_max_tokens = (
        max(8, min(structured_chunk_settings.max_tokens, structured_token_counter.max_tokens))
        if structured_token_counter is not None
        else None
    )
    token_counter_name = structured_token_counter.name if structured_token_counter is not None else None
    token_counter_version = structured_token_counter.version if structured_token_counter is not None else None
    token_counter_identity = structured_token_counter.identity if structured_token_counter is not None else None
    file_chunk_fingerprint = structured_chunk_config_fingerprint(
        max_tokens=structured_chunk_settings.max_tokens,
        effective_max_tokens=effective_max_tokens or 0,
        force_split_overlap_tokens=structured_chunk_settings.force_split_overlap_tokens,
        target_tokens=structured_chunk_settings.target_tokens,
        include_identity_prefix=structured_chunk_settings.include_identity_prefix,
        token_counter_name=token_counter_name or "unknown",
        token_counter_version=token_counter_version or "unknown",
        token_counter_identity=token_counter_identity or "unknown",
    )
    combined_chunk_fingerprint = _fingerprint(
        {
            "text": text_chunk_fingerprint,
            "file": file_chunk_fingerprint,
        }
    )
    compatibility_fields: dict[str, object] = {
        "signature_version": INDEX_SIGNATURE_VERSION,
        "signature_complete": signature_complete,
        "engine": ENGINE_NAME,
        "index_instance_id": index_instance_id,
        "vector_backend": index_settings.backend,
        "backend_namespace_version": BACKEND_NAMESPACE_VERSION,
        "backend_corpus_namespace": corpus_namespace,
        "index_shaping_version": INDEX_SHAPING_VERSION,
        "index_shaping_fingerprint": index_shaping_fingerprint,
        "metadata_whitelist": metadata_whitelist,
        "embedding_config_version": EMBEDDING_CONFIG_VERSION,
        "embedding_provider": embed_settings.provider,
        "embedding_model": model,
        "embedding_dimension": embedding_dimension,
        "embedding_configured_dimension": (embed_settings.dim if embed_settings.provider == "hash" else None),
        "embedding_runtime_identity": embedding_runtime_identity,
        "embedding_config_fingerprint": embedding_config_fingerprint,
        "parser_config_version": PARSER_CONFIG_VERSION,
        "parser_config_fingerprint": parser_config_fingerprint,
        "chunk_config_version": DEFAULT_HASH_POLICY.chunk_id_version,
        "chunk_config_fingerprint": combined_chunk_fingerprint,
        "text_chunk_config_fingerprint": text_chunk_fingerprint,
        "file_chunk_config_version": STRUCTURED_CHUNK_CONFIG_VERSION,
        "file_chunk_config_fingerprint": file_chunk_fingerprint,
        "file_chunk_effective_max_tokens": effective_max_tokens,
        "file_chunk_token_counter_name": token_counter_name,
        "file_chunk_token_counter_version": token_counter_version,
        "file_chunk_token_counter_identity": token_counter_identity,
        "id_schema_version": DEFAULT_HASH_POLICY.id_schema_version,
        "content_hash_version": DEFAULT_HASH_POLICY.content_hash_version,
        "index_schema_version": index_schema_version,
    }
    return IndexSignature(
        signature_version=INDEX_SIGNATURE_VERSION,
        signature_complete=signature_complete,
        compatibility_fingerprint=_fingerprint(compatibility_fields),
        engine=ENGINE_NAME,
        engine_version=engine_version or get_engine_version(),
        index_instance_id=index_instance_id,
        vector_backend=index_settings.backend,
        backend_namespace_version=BACKEND_NAMESPACE_VERSION,
        backend_corpus_namespace=corpus_namespace,
        index_shaping_version=INDEX_SHAPING_VERSION,
        index_shaping_fingerprint=index_shaping_fingerprint,
        metadata_whitelist=metadata_whitelist,
        embedding_config_version=EMBEDDING_CONFIG_VERSION,
        embedding_provider=embed_settings.provider,
        embedding_model=model,
        embedding_dimension=embedding_dimension,
        embedding_configured_dimension=(embed_settings.dim if embed_settings.provider == "hash" else None),
        embedding_runtime_identity=embedding_runtime_identity,
        embedding_config_fingerprint=embedding_config_fingerprint,
        parser_config_version=PARSER_CONFIG_VERSION,
        parser_config_fingerprint=parser_config_fingerprint,
        chunk_config_version=DEFAULT_HASH_POLICY.chunk_id_version,
        chunk_config_fingerprint=combined_chunk_fingerprint,
        text_chunk_config_fingerprint=text_chunk_fingerprint,
        file_chunk_config_version=STRUCTURED_CHUNK_CONFIG_VERSION,
        file_chunk_config_fingerprint=file_chunk_fingerprint,
        file_chunk_effective_max_tokens=effective_max_tokens,
        file_chunk_token_counter_name=token_counter_name,
        file_chunk_token_counter_version=token_counter_version,
        file_chunk_token_counter_identity=token_counter_identity,
        id_schema_version=DEFAULT_HASH_POLICY.id_schema_version,
        content_hash_version=DEFAULT_HASH_POLICY.content_hash_version,
        index_schema_version=index_schema_version,
    )


def _signature_from_dict(data: object) -> IndexSignature:
    if not isinstance(data, dict):
        raise RuntimeError("Index build receipt signature must be an object")
    kwargs: dict[str, object] = {}
    for field in fields(IndexSignature):
        if field.name not in data:
            raise RuntimeError(f"Index build receipt signature is missing {field.name!r}")
        value = data[field.name]
        if field.name == "metadata_whitelist" and isinstance(value, list):
            value = tuple(str(item) for item in value)
        kwargs[field.name] = value
    signature = IndexSignature(**kwargs)  # type: ignore[arg-type]
    if signature.signature_version != INDEX_SIGNATURE_VERSION:
        raise RuntimeError(f"Unsupported index signature version: {signature.signature_version!r}")
    compatibility_fields = signature.to_dict()
    compatibility_fields.pop("compatibility_fingerprint")
    compatibility_fields.pop("engine_version")
    if _fingerprint(compatibility_fields) != signature.compatibility_fingerprint:
        raise RuntimeError("Index build receipt signature fingerprint is invalid")
    return signature


@dataclass(frozen=True, slots=True)
class IndexBuildReceipt:
    """Durable proof of the signature that actually produced the active corpus."""

    receipt_version: str
    receipt_fingerprint: str
    index_signature: IndexSignature
    document_pipeline: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "receipt_version": self.receipt_version,
            "receipt_fingerprint": self.receipt_fingerprint,
            "index_signature": self.index_signature.to_dict(),
            "document_pipeline": self.document_pipeline,
        }


def build_index_build_receipt(
    signature: IndexSignature, *, document_pipeline: Mapping[str, object]
) -> IndexBuildReceipt:
    if not signature.signature_complete or signature.index_instance_id is None:
        raise ValueError("An index build receipt requires a complete signature and instance id")
    payload = {
        "receipt_version": INDEX_BUILD_RECEIPT_VERSION,
        "index_signature": signature.to_dict(),
        "document_pipeline": document_pipeline,
    }
    return IndexBuildReceipt(
        receipt_version=INDEX_BUILD_RECEIPT_VERSION,
        receipt_fingerprint=_fingerprint(payload),
        index_signature=signature,
        document_pipeline=dict(document_pipeline),
    )


def write_index_build_receipt(
    path: Path, signature: IndexSignature, *, document_pipeline: Mapping[str, object]
) -> IndexBuildReceipt:
    receipt = build_index_build_receipt(signature, document_pipeline=document_pipeline)
    _atomic_write_text(Path(path), json.dumps(receipt.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return receipt


def load_index_build_receipt(path: Path) -> IndexBuildReceipt | None:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read index build receipt: {type(exc).__name__}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("Index build receipt must be an object")
    if data.get("receipt_version") != INDEX_BUILD_RECEIPT_VERSION:
        raise RuntimeError(f"Unsupported index build receipt version: {data.get('receipt_version')!r}")
    signature = _signature_from_dict(data.get("index_signature"))
    pipeline = data.get("document_pipeline")
    if not isinstance(pipeline, dict):
        raise RuntimeError("Index build receipt document_pipeline must be an object")
    receipt = build_index_build_receipt(signature, document_pipeline=dict(pipeline))
    if data.get("receipt_fingerprint") != receipt.receipt_fingerprint:
        raise RuntimeError("Index build receipt fingerprint is invalid")
    return receipt


def _read_index_instance_id(path: Path) -> str | None:
    try:
        existing = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    if not existing:
        raise RuntimeError(f"Empty index instance id in {path}")
    suffix = existing.removeprefix(_INDEX_INSTANCE_ID_PREFIX)
    if (
        not existing.startswith(_INDEX_INSTANCE_ID_PREFIX)
        or len(suffix) != 32
        or any(char not in "0123456789abcdef" for char in suffix)
    ):
        raise RuntimeError(f"Invalid index instance id in {path}")
    return existing


def _instance_lock_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.lock")


def _read_lock_owner(path: Path) -> dict[str, object] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _pid_is_alive(pid: object) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _lock_descriptor(descriptor: int, *, blocking: bool) -> bool:
    if fcntl is None:
        return True
    operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
    try:
        fcntl.flock(descriptor, operation)
    except BlockingIOError:
        return False
    return True


def _unlock_descriptor(descriptor: int) -> None:
    if fcntl is not None:
        fcntl.flock(descriptor, fcntl.LOCK_UN)


def _recover_stale_instance_lock(lock_path: Path) -> bool:
    try:
        stat = lock_path.stat()
    except FileNotFoundError:
        return True
    owner_hint = _read_lock_owner(lock_path)
    age = time.time() - stat.st_mtime
    # A current-protocol owner holds an advisory lock for its whole lease. If that lock is
    # already released, a crash is provable immediately; legacy/malformed owner files still
    # need the conservative age threshold before recovery.
    current_protocol = bool(owner_hint and owner_hint.get("advisory_lock"))
    abandoned_initialization = fcntl is not None and owner_hint is None and age > _LOCK_INITIALIZATION_GRACE_SECONDS
    if age <= _LOCK_STALE_SECONDS and not current_protocol and not abandoned_initialization:
        return False
    try:
        descriptor = os.open(lock_path, os.O_RDWR)
    except FileNotFoundError:
        return True
    try:
        # Every current owner retains an exclusive advisory lock on this inode. A crashed
        # owner releases it automatically, while a merely long-running owner cannot be
        # mistaken for stale even when the timestamp is old.
        if not _lock_descriptor(descriptor, blocking=False):
            return False
        opened_stat = os.fstat(descriptor)
        owner = _read_lock_owner(lock_path)
        opened_age = time.time() - opened_stat.st_mtime
        if (
            opened_age <= _LOCK_STALE_SECONDS
            and not (owner and owner.get("advisory_lock"))
            and not (fcntl is not None and owner is None and opened_age > _LOCK_INITIALIZATION_GRACE_SECONDS)
        ):
            return False
        if (
            owner
            and not owner.get("advisory_lock")
            and owner.get("hostname") == socket.gethostname()
            and _pid_is_alive(owner.get("pid"))
        ):
            return False
        try:
            current_stat = lock_path.stat()
        except FileNotFoundError:
            return True
        if (current_stat.st_dev, current_stat.st_ino) != (opened_stat.st_dev, opened_stat.st_ino):
            return True
        stale_path = lock_path.with_name(f".{lock_path.name}.{secrets.token_hex(8)}.stale")
        os.replace(lock_path, stale_path)
        stale_path.unlink(missing_ok=True)
        return True
    finally:
        _unlock_descriptor(descriptor)
        os.close(descriptor)


def _acquire_instance_lock(path: Path) -> tuple[Path, str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _instance_lock_path(path)
    token = secrets.token_hex(16)
    for _attempt in range(500):
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            _recover_stale_instance_lock(lock_path)
            time.sleep(0.01)
            continue
        try:
            # Establish the crash-releasing advisory lock immediately after O_EXCL, before
            # constructing/writing owner metadata, to minimize an unowned-init window.
            _lock_descriptor(descriptor, blocking=True)
            owner = {
                "token": token,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "created_at": time.time(),
                "advisory_lock": fcntl is not None,
            }
            os.write(descriptor, (json.dumps(owner, sort_keys=True) + "\n").encode("utf-8"))
            os.fsync(descriptor)
        except Exception:
            _unlock_descriptor(descriptor)
            os.close(descriptor)
            lock_path.unlink(missing_ok=True)
            raise
        return lock_path, token, descriptor
    raise TimeoutError(f"Timed out acquiring index instance lock for {path}")


def _release_instance_lock(lock_path: Path, token: str, descriptor: int) -> None:
    try:
        owner = _read_lock_owner(lock_path)
        if owner is not None and owner.get("token") == token:
            lock_path.unlink(missing_ok=True)
    finally:
        _unlock_descriptor(descriptor)
        os.close(descriptor)


@dataclass(slots=True)
class IndexInstanceLease:
    path: Path
    instance_id: str | None

    def publish_new(self) -> str:
        instance_id = f"{_INDEX_INSTANCE_ID_PREFIX}{secrets.token_hex(16)}"
        _atomic_write_text(self.path, instance_id + "\n")
        self.instance_id = instance_id
        return instance_id

    def remove(self) -> None:
        self.path.unlink(missing_ok=True)
        self.instance_id = None


@contextmanager
def locked_index_instance(path: Path, *, create: bool) -> Iterator[IndexInstanceLease]:
    """Hold the cross-process corpus identity lock while reading or mutating its receipt."""
    path = Path(path)
    lock_path, token, descriptor = _acquire_instance_lock(path)
    try:
        lease = IndexInstanceLease(path=path, instance_id=_read_index_instance_id(path))
        if create and lease.instance_id is None:
            lease.publish_new()
        yield lease
    finally:
        _release_instance_lock(lock_path, token, descriptor)


def get_or_create_index_instance_id(path: Path) -> str:
    """Return one opaque identity persisted with the index volume."""
    with locked_index_instance(path, create=True) as lease:
        assert lease.instance_id is not None
        return lease.instance_id


def read_index_instance_id(path: Path) -> str | None:
    """Read the persisted identity without creating one."""
    with locked_index_instance(path, create=False) as lease:
        return lease.instance_id


def delete_index_instance_id(path: Path) -> None:
    """Delete an instance identity while excluding concurrent readers/creators."""
    with locked_index_instance(path, create=False) as lease:
        lease.remove()


__all__ = [
    "BACKEND_NAMESPACE_VERSION",
    "EMBEDDING_CONFIG_VERSION",
    "ENGINE_NAME",
    "INDEX_INSTANCE_ID_FILENAME",
    "INDEX_SHAPING_VERSION",
    "INDEX_SIGNATURE_VERSION",
    "PARSER_CONFIG_VERSION",
    "STRUCTURED_CHUNK_CONFIG_VERSION",
    "IndexSignature",
    "IndexBuildReceipt",
    "IndexInstanceLease",
    "INDEX_BUILD_RECEIPT_FILENAME",
    "INDEX_BUILD_RECEIPT_VERSION",
    "build_index_signature",
    "build_index_build_receipt",
    "backend_corpus_namespace",
    "delete_index_instance_id",
    "get_or_create_index_instance_id",
    "load_index_build_receipt",
    "locked_index_instance",
    "read_index_instance_id",
    "resolve_embedding_dimension",
    "write_index_build_receipt",
]
