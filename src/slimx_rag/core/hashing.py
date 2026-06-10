from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class HashPolicy:
    """Versioned hashing policy for SlimX-RAG identities and fingerprints.

    Digest sizes are expressed in bytes because ``hashlib.blake2b`` uses byte
    sizes. The resulting hexadecimal strings are always twice as long.

    These values are internal protocol choices, not normal runtime settings:
    changing them changes document IDs, content hashes, chunk IDs, and index
    state compatibility.
    """

    algorithm: str = "blake2b"
    encoding: str = "utf-8"

    # Full document/path fingerprints: 32 bytes => 64 hex chars.
    path_id_digest_size: int = 32
    content_hash_digest_size: int = 32

    # Current project behavior uses full-length chunk IDs too.
    chunk_id_digest_size: int = 32

    # Small internal fingerprint: 8 bytes => 16 hex chars.
    config_fingerprint_digest_size: int = 8

    # Schema/version labels for future manifest compatibility checks.
    id_schema_version: str = "slimx-id-v1"
    content_hash_version: str = "content-v1"
    chunk_id_version: str = "chunk-v1"

    def validate(self) -> None:
        if self.algorithm != "blake2b":
            raise ValueError(f"Unsupported hash algorithm: {self.algorithm!r}")
        for name, value in (
            ("path_id_digest_size", self.path_id_digest_size),
            ("content_hash_digest_size", self.content_hash_digest_size),
            ("chunk_id_digest_size", self.chunk_id_digest_size),
            ("config_fingerprint_digest_size", self.config_fingerprint_digest_size),
        ):
            if not 1 <= value <= 64:
                raise ValueError(f"{name} must be between 1 and 64 bytes")

    def as_manifest_dict(self) -> dict[str, object]:
        """Return a JSON-serializable policy record for future manifests."""
        return asdict(self)


DEFAULT_HASH_POLICY = HashPolicy()


def hash_text(
    text: str,
    *,
    digest_size: int,
    policy: HashPolicy = DEFAULT_HASH_POLICY,
) -> str:
    """Hash text deterministically using the configured SlimX-RAG policy."""
    policy.validate()
    h = hashlib.blake2b(digest_size=digest_size)
    h.update((text or "").encode(policy.encoding, errors="ignore"))
    return h.hexdigest()


def path_id(kb_relpath: str, policy: HashPolicy = DEFAULT_HASH_POLICY) -> str:
    """Stable document identity derived from the knowledge-base relative path."""
    return hash_text(
        kb_relpath,
        digest_size=policy.path_id_digest_size,
        policy=policy,
    )


def content_hash(text: str, policy: HashPolicy = DEFAULT_HASH_POLICY) -> str:
    """Document version fingerprint; changes when loaded content changes."""
    return hash_text(
        text or "",
        digest_size=policy.content_hash_digest_size,
        policy=policy,
    )


def fallback_doc_id(
    source: str,
    content_hash_value: str,
    policy: HashPolicy = DEFAULT_HASH_POLICY,
) -> str:
    """Fallback identity for documents without a stable knowledge-base path."""
    payload = f"fallback\n{source}\n{content_hash_value}"
    return hash_text(payload, digest_size=policy.path_id_digest_size, policy=policy)


def chunk_config_fingerprint(
    *,
    chunk_size: int,
    chunk_overlap: int,
    separators: Sequence[str],
    policy: HashPolicy = DEFAULT_HASH_POLICY,
) -> str:
    """Fingerprint chunking parameters so chunk IDs change with config."""
    seps = "\u241E".join(separators)  # record separators unambiguously
    return hash_text(
        f"{chunk_size}|{chunk_overlap}|{seps}",
        digest_size=policy.config_fingerprint_digest_size,
        policy=policy,
    )


def make_chunk_id(
    *,
    parent_id: str,
    content_hash_value: str,
    chunk_index: int,
    chunk_cfg_hash: str,
    policy: HashPolicy = DEFAULT_HASH_POLICY,
) -> str:
    """Create a stable chunk ID from parent identity, version, config, and index."""
    payload = "\n".join(
        [
            parent_id,
            content_hash_value,
            chunk_cfg_hash,
            str(chunk_index),
        ]
    )
    return hash_text(payload, digest_size=policy.chunk_id_digest_size, policy=policy)


# Backwards-compatible private aliases used by older internal modules/tests.
def _hash_text(text: str, *, digest_size: int) -> str:
    return hash_text(text, digest_size=digest_size)


def _hash_path(kb_relpath: str) -> str:
    return path_id(kb_relpath)


def _content_hash(text: str) -> str:
    return content_hash(text)
