from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def get_engine_version() -> str:
    """Return the installed SlimX-RAG package version without importing build tooling."""
    for distribution_name in ("slimx-rag", "slimx_rag"):
        try:
            return version(distribution_name)
        except PackageNotFoundError:
            continue
    return "unknown"


__all__ = ["get_engine_version"]
