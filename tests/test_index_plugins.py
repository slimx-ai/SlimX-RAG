from __future__ import annotations

from slimx_rag.index import make_index_backend
from slimx_rag.embed import EmbeddedChunk
from slimx_rag.settings import EmbedSettings, IndexSettings


def test_index_backend_local_roundtrip_and_query(tmp_path):
    idx_path = tmp_path / "index.jsonl"
    st_path = tmp_path / "index_state.json"

    settings = IndexSettings(
        backend="local",
        top_k=2,
        metadata_whitelist=["keep"],
        write_state=True,
        state_filename="index_state.json",
    )

    idx = make_index_backend(idx_path, settings=settings, state_path=st_path)
    idx.load()
    idx.set_embed_config(EmbedSettings(provider="hash", dim=2))

    # Initialize incremental state for two docs
    current_docs = {
        "doc1": ("h1", ["c1"]),
        "doc2": ("h2", ["c2"]),
    }
    assert idx.apply_incremental_plan(current_docs=current_docs) == 0

    # Upsert two chunks (vectors are orthogonal)
    written = idx.upsert(
        [
            EmbeddedChunk(chunk_id="c1", vector=[1.0, 0.0], text="A", metadata={"keep": 1, "drop": 2}),
            EmbeddedChunk(chunk_id="c2", vector=[0.0, 1.0], text="B", metadata={"keep": 3, "drop": 4}),
        ],
        skip_existing=True,
    )
    assert written == 2
    idx.save()

    # Reload and query
    idx2 = make_index_backend(idx_path, settings=settings, state_path=st_path)
    idx2.load()

    res = idx2.query([1.0, 0.0], top_k=1)
    assert len(res) == 1
    assert res[0].chunk_id == "c1"
    assert res[0].text == "A"
    # metadata whitelist applied
    assert res[0].metadata == {"keep": 1}

    # Remove doc2 and ensure incremental deletion removes c2
    deleted = idx2.apply_incremental_plan(current_docs={"doc1": ("h1", ["c1"])})
    assert deleted == 1
    idx2.save()

    idx3 = make_index_backend(idx_path, settings=settings, state_path=st_path)
    idx3.load()
    res2 = idx3.query([0.0, 1.0], top_k=2)
    assert all(r.chunk_id != "c2" for r in res2)
