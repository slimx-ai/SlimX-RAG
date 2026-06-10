from __future__ import annotations

import sys
import types

import pytest

from slimx_rag.embed import EmbeddedChunk
from slimx_rag.settings import EmbedSettings, IndexSettings


class FakeCursor:
    def __init__(self, conn: FakeConnection):
        self.conn = conn
        self.rowcount = 0
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql: str, params=None):
        self.conn.executed.append((sql, params))
        if "DELETE FROM" in sql:
            ids = list(params[0]) if params else []
            deleted = 0
            for cid in ids:
                if cid in self.conn.rows:
                    self.conn.rows.pop(cid)
                    deleted += 1
            self.rowcount = deleted
        elif "SELECT chunk_id" in sql:
            # Intentionally return reverse chunk_id order for equal scores to
            # prove PgVectorIndexBackend post-sorts deterministically.
            self._rows = [
                (cid, row["text"], row["metadata"], 1.0)
                for cid, row in sorted(self.conn.rows.items(), key=lambda kv: kv[0], reverse=True)
            ]

    def executemany(self, sql: str, rows):
        self.conn.executed.append((sql, rows))
        for chunk_id, embedding, text, metadata in rows:
            if "DO NOTHING" in sql and chunk_id in self.conn.rows:
                continue
            self.conn.rows[str(chunk_id)] = {
                "embedding": embedding,
                "text": text,
                "metadata": metadata,
            }
        self.rowcount = len(rows)

    def fetchall(self):
        return list(self._rows)


class FakeConnection:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.executed: list[tuple[str, object]] = []
        self.rows: dict[str, dict] = {}
        self.commits = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1


class FakePsycopg:
    connections: list[FakeConnection] = []
    rows: dict[str, dict] = {}

    @classmethod
    def connect(cls, dsn: str):
        conn = FakeConnection(dsn)
        conn.rows = cls.rows
        cls.connections.append(conn)
        return conn


def install_fake_psycopg(monkeypatch):
    FakePsycopg.connections.clear()
    FakePsycopg.rows.clear()
    fake = types.SimpleNamespace(connect=FakePsycopg.connect)
    monkeypatch.setitem(sys.modules, "psycopg", fake)


def all_sql() -> str:
    return "\n".join(sql for conn in FakePsycopg.connections for sql, _params in conn.executed)


def latest_connection() -> FakeConnection:
    return FakePsycopg.connections[-1]


def test_pgvector_first_upsert_creates_table_from_vector_dim(monkeypatch, tmp_path):
    install_fake_psycopg(monkeypatch)

    from slimx_rag.index.pgvector_backend import PgVectorIndexBackend

    idx = PgVectorIndexBackend(
        tmp_path / "unused.index",
        settings=IndexSettings(backend="pgvector", backend_config={"dsn": "postgresql://test/db"}),
        state_path=tmp_path / "index_state.json",
    )
    idx.load()
    idx.set_embed_config(EmbedSettings(provider="hash", dim=384))

    assert idx.dim is None
    assert "CREATE TABLE" not in all_sql()

    written = idx.upsert([
        EmbeddedChunk(chunk_id="c1", vector=[1.0, 0.0, 0.0], text="A", metadata={"keep": 1}),
    ])

    assert written == 1
    assert idx.dim == 3
    sql = all_sql()
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert "embedding vector(3)" in sql
    assert "INSERT INTO public.slimx_vectors" in sql


def test_pgvector_configured_dim_is_enforced_on_first_upsert(monkeypatch, tmp_path):
    install_fake_psycopg(monkeypatch)

    from slimx_rag.index.pgvector_backend import PgVectorIndexBackend

    idx = PgVectorIndexBackend(
        tmp_path / "unused.index",
        settings=IndexSettings(backend="pgvector", backend_config={"dsn": "postgresql://test/db", "dim": 2}),
        state_path=tmp_path / "index_state.json",
    )
    idx.load()

    assert idx.dim == 2
    assert "embedding vector(2)" in all_sql()

    with pytest.raises(RuntimeError, match="Vector dim mismatch"):
        idx.upsert([
            EmbeddedChunk(chunk_id="bad", vector=[1.0, 0.0, 0.0], text="bad", metadata={}),
        ])


def test_pgvector_applies_metadata_whitelist(monkeypatch, tmp_path):
    install_fake_psycopg(monkeypatch)

    from slimx_rag.index.pgvector_backend import PgVectorIndexBackend

    idx = PgVectorIndexBackend(
        tmp_path / "unused.index",
        settings=IndexSettings(
            backend="pgvector",
            backend_config={"dsn": "postgresql://test/db"},
            metadata_whitelist=["keep"],
        ),
        state_path=tmp_path / "index_state.json",
    )
    idx.load()

    idx.upsert([
        EmbeddedChunk(chunk_id="c1", vector=[1.0, 0.0], text="A", metadata={"keep": 1, "drop": 2}),
    ])

    assert latest_connection().rows["c1"]["metadata"] == {"keep": 1}


def test_pgvector_query_orders_equal_scores_by_chunk_id(monkeypatch, tmp_path):
    install_fake_psycopg(monkeypatch)

    from slimx_rag.index.pgvector_backend import PgVectorIndexBackend

    idx = PgVectorIndexBackend(
        tmp_path / "unused.index",
        settings=IndexSettings(backend="pgvector", backend_config={"dsn": "postgresql://test/db"}, top_k=3),
        state_path=tmp_path / "index_state.json",
    )
    idx.load()

    idx.upsert([
        EmbeddedChunk(chunk_id="c3", vector=[1.0, 0.0], text="C", metadata={}),
        EmbeddedChunk(chunk_id="c1", vector=[1.0, 0.0], text="A", metadata={}),
        EmbeddedChunk(chunk_id="c2", vector=[1.0, 0.0], text="B", metadata={}),
    ])

    assert [r.chunk_id for r in idx.query([1.0, 0.0], top_k=3)] == ["c1", "c2", "c3"]
    assert "ORDER BY embedding <=> %s::vector, chunk_id ASC" in all_sql()
