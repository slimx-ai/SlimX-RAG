from __future__ import annotations

import pytest

from slimx_rag.core.hashing import (
    DEFAULT_HASH_POLICY,
    HashPolicy,
    chunk_config_fingerprint,
    content_hash,
    fallback_doc_id,
    make_chunk_id,
    path_id,
)


def hex_len(digest_size: int) -> int:
    return digest_size * 2


def test_default_hash_policy_lengths_are_explicit() -> None:
    policy = DEFAULT_HASH_POLICY

    assert len(path_id("product/slimx-rag.md")) == hex_len(policy.path_id_digest_size)
    assert len(content_hash("hello")) == hex_len(policy.content_hash_digest_size)
    assert len(
        chunk_config_fingerprint(
            chunk_size=800,
            chunk_overlap=120,
            separators=("\n\n", "\n", " ", ""),
        )
    ) == hex_len(policy.config_fingerprint_digest_size)
    assert len(
        make_chunk_id(
            parent_id="doc-1",
            content_hash_value=content_hash("hello"),
            chunk_index=0,
            chunk_cfg_hash="cfg",
        )
    ) == hex_len(policy.chunk_id_digest_size)


def test_hashes_are_stable_and_content_sensitive() -> None:
    assert path_id("a/file.md") == path_id("a/file.md")
    assert path_id("a/file.md") != path_id("b/file.md")

    assert content_hash("hello") == content_hash("hello")
    assert content_hash("hello") != content_hash("hello!")

    cfg = chunk_config_fingerprint(chunk_size=800, chunk_overlap=120, separators=("\n", " "))
    cid1 = make_chunk_id(
        parent_id="doc-1",
        content_hash_value=content_hash("hello"),
        chunk_index=0,
        chunk_cfg_hash=cfg,
    )
    cid2 = make_chunk_id(
        parent_id="doc-1",
        content_hash_value=content_hash("hello"),
        chunk_index=1,
        chunk_cfg_hash=cfg,
    )

    assert cid1 != cid2


def test_chunk_config_fingerprint_changes_when_chunking_changes() -> None:
    cfg1 = chunk_config_fingerprint(chunk_size=800, chunk_overlap=120, separators=("\n", " "))
    cfg2 = chunk_config_fingerprint(chunk_size=801, chunk_overlap=120, separators=("\n", " "))
    cfg3 = chunk_config_fingerprint(chunk_size=800, chunk_overlap=120, separators=(" ", "\n"))

    assert cfg1 != cfg2
    assert cfg1 != cfg3


def test_fallback_doc_id_uses_source_and_content_hash() -> None:
    h = content_hash("hello")
    assert fallback_doc_id("source-a", h) == fallback_doc_id("source-a", h)
    assert fallback_doc_id("source-a", h) != fallback_doc_id("source-b", h)


def test_hash_policy_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        HashPolicy(algorithm="sha256").validate()

    with pytest.raises(ValueError):
        HashPolicy(content_hash_digest_size=0).validate()

    with pytest.raises(ValueError):
        HashPolicy(chunk_id_digest_size=65).validate()
