"""Release-identity inputs of the CPU service image must be immutable (RAG-AUD-002/-014).

These are repository-file assertions (no Docker), so CI rejects a regression to mutable
inputs: base/tool images by digest, the committed lock honoured with --frozen, CPU torch from
the PyTorch index, SlimX from the exact reviewed archive, the embedding model baked at an exact
commit with an offline runtime, OCI labels, a non-root user, and a candidate publication path
that never builds GPU or moves `latest`.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
GPU_DOCKERFILE = (ROOT / "Dockerfile.gpu").read_text(encoding="utf-8")
PUBLISH = (ROOT / ".github/workflows/publish-image.yaml").read_text(encoding="utf-8")
CI = (ROOT / ".github/workflows/ci.yaml").read_text(encoding="utf-8")
PYPROJECT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
LOCK = (ROOT / "uv.lock").read_text(encoding="utf-8")

SLIMX_ARCHIVE = "https://github.com/slimx-ai/slimx/archive/e4b7ca30f9b528df477672cfa5911c69b722a556.tar.gz"
# The same sdist hash ControlRoom's apps/api/uv.lock records for this archive.
SLIMX_SDIST_SHA256 = "sha256:cfcd6c00277b764b6b39d8b0425296bbf2d95d42e0a1854d36503dfcfc2c341d"
DIGEST = re.compile(r"@sha256:[0-9a-f]{64}")


def _locked(name: str) -> list[str]:
    """The ``[[package]]`` blocks of ``uv.lock`` for ``name`` (text; tomllib needs 3.11+)."""
    return [block for block in LOCK.split("[[package]]") if re.search(rf'^name = "{re.escape(name)}"$', block, re.M)]


def _field(block: str, key: str) -> str | None:
    match = re.search(rf"^{key} = (.+)$", block, re.M)
    return match.group(1).strip() if match else None


def test_base_and_tool_images_are_pinned_by_digest() -> None:
    for dockerfile in (DOCKERFILE, GPU_DOCKERFILE):
        assert re.search(r"^ARG PYTHON_IMAGE=python:3\.12-slim@sha256:[0-9a-f]{64}$", dockerfile, re.M)
        assert re.search(r"^ARG UV_IMAGE=ghcr\.io/astral-sh/uv:\d+\.\d+\.\d+@sha256:[0-9a-f]{64}$", dockerfile, re.M)
        assert "pip install" not in dockerfile.replace("uv pip install", "")
        assert "git+" not in dockerfile and "apt-get install" not in dockerfile


def test_python_resolution_is_the_committed_lock() -> None:
    assert (ROOT / "uv.lock").is_file()
    ignored = [line.strip() for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()]
    assert "uv.lock" not in ignored
    assert "uv sync --frozen --no-dev --no-editable" in DOCKERFILE
    assert "uv sync --locked --group dev" in CI
    for extra in ("demo", "hf", "doc", "answer"):
        assert f"--extra {extra}" in DOCKERFILE


def test_torch_is_the_locked_cpu_wheel_and_slimx_is_the_reviewed_archive() -> None:
    torch_blocks = _locked("torch")
    assert torch_blocks
    assert {_field(block, "source") for block in torch_blocks} == {
        '{ registry = "https://download.pytorch.org/whl/cpu" }'
    }
    assert {_field(block, "version") for block in torch_blocks} <= {'"2.7.1"', '"2.7.1+cpu"'}
    assert 'torch = [{ index = "pytorch-cpu" }]' in PYPROJECT
    assert '"torch==2.7.1"' in PYPROJECT and '"sentence-transformers==3.4.1"' in PYPROJECT
    (slimx,) = _locked("slimx")
    assert _field(slimx, "source") == f'{{ url = "{SLIMX_ARCHIVE}" }}'
    assert _field(slimx, "sdist") == f'{{ hash = "{SLIMX_SDIST_SHA256}" }}'
    assert f'"slimx @ {SLIMX_ARCHIVE}"' in PYPROJECT
    (st,) = _locked("sentence-transformers")
    assert _field(st, "version") == '"3.4.1"'


def test_embedding_model_is_baked_at_an_exact_commit_and_runtime_is_offline() -> None:
    assert re.search(r"^ARG RAG_HF_REVISION=[0-9a-f]{40}$", DOCKERFILE, re.M)
    assert "SentenceTransformer(model, revision=revision" in DOCKERFILE
    assert "assert snapshots == [revision]" in DOCKERFILE
    offline = DOCKERFILE.index("HF_HUB_OFFLINE=1")
    assert offline > DOCKERFILE.index("SentenceTransformer(model, revision=revision")
    assert "TRANSFORMERS_OFFLINE=1" in DOCKERFILE
    assert "RAG_HF_REVISION:=" in (ROOT / "docker-entrypoint.sh").read_text(encoding="utf-8")


def test_base_image_label_reflects_the_build_argument() -> None:
    for dockerfile in (DOCKERFILE, GPU_DOCKERFILE):
        assert 'ai.slimx.rag.base_image="${PYTHON_IMAGE}"' in dockerfile
        assert not dockerfile.startswith("# syntax=")  # the BuildKit frontend is not a pinned input
    assert "uv lock --check" in DOCKERFILE and "uv lock --check" in PUBLISH
    assert "pip install uv==0.9.18" in CI and "--extra answer" in CI
    assert "--reinstall-package torch" in GPU_DOCKERFILE and "torch==2.7.1+cu124" in GPU_DOCKERFILE


def test_image_metadata_and_runtime_user() -> None:
    for label in (
        "org.opencontainers.image.source",
        "org.opencontainers.image.revision",
        "org.opencontainers.image.version",
        "ai.slimx.rag.embedding_revision",
    ):
        assert label in DOCKERFILE
    assert re.search(r"^USER 1000:1000$", DOCKERFILE, re.M)
    assert "HEALTHCHECK" in DOCKERFILE


def test_candidate_publication_is_cpu_only_and_never_moves_latest() -> None:
    candidate = PUBLISH[PUBLISH.index("  candidate-cpu:") : PUBLISH.index("  publish:")]
    assert "workflow_dispatch" in candidate and "inputs.mode == 'candidate-cpu'" in candidate
    assert "candidate-${{ github.sha }}" in candidate
    assert "latest" not in candidate.replace("ubuntu-latest", "")
    assert "Dockerfile.gpu" not in candidate and "file: Dockerfile\n" in candidate
    assert "sbom: true" in candidate and "provenance: mode=max" in candidate
    assert "SOURCE_REVISION=${{ github.sha }}" in candidate
    gpu = PUBLISH[PUBLISH.index("  publish-gpu:") :]
    assert "if: github.event_name == 'release'" in gpu
