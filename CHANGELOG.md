# Changelog

## 0.2.8 — 2026-07-12

### Added

- Authenticated `POST /api/admin/index/reset` maintenance contract for explicit local/FAISS corpus reset without changing embedding settings.
- Exact destructive confirmation, required instance/fingerprint compare-and-swap preconditions, structured failure details, and previous/new signature provenance.
- Readiness diagnostics for invalid state/receipts and receipt-instance mismatch recovery.

### Fixed

- Text and file indexing now reject work that crossed an index-generation reset, including concurrent first-writer races.
- Pure index reset preserves the existing embedding override byte-for-byte and rolls back local artifacts transactionally.
- Every `/ready` response now includes `auth_enabled` and `engine_version`.

