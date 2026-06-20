"""Embedding `device` routing: the local `hf` embedder forwards the device, while the
deterministic `hash` embedder ignores it entirely (offline determinism never depends on
hardware)."""

from __future__ import annotations

import sys
import types

from slimx_rag.embed.embedder import HashEmbedder, make_embedder
from slimx_rag.settings import EmbedSettings


def test_hf_embedder_forwards_device(monkeypatch) -> None:
    recorded: dict[str, object] = {}

    class FakeST:
        def __init__(self, model, device=None):
            recorded["model"] = model
            recorded["device"] = device

        def encode(self, texts, **_kwargs):
            return [[0.0] for _ in texts]

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeST)
    )

    make_embedder(EmbedSettings(provider="hf", hf_model="some-model", device="cuda"))
    assert recorded == {"model": "some-model", "device": "cuda"}


def test_hash_embedder_ignores_device_and_stays_deterministic() -> None:
    # `device` set or not, the hash embedder produces identical vectors and never touches
    # torch/sentence-transformers (this test runs without those installed).
    on_gpu = make_embedder(EmbedSettings(provider="hash", dim=8, device="cuda"))
    on_cpu = make_embedder(EmbedSettings(provider="hash", dim=8, device=None))
    assert isinstance(on_gpu, HashEmbedder)
    assert on_gpu.embed_texts(["hello world"]) == on_cpu.embed_texts(["hello world"])
