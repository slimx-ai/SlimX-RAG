from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from slimx_rag.embed import EmbeddedChunk
from slimx_rag.settings import IndexSettings

from .base import IndexBackend
from .types import SearchResult, IndexState


def _vector_literal(vec: List[float]) -> str:
    # pgvector input format: '[1,2,3]'
    return "[" + ",".join(f"{float(x):.8g}" for x in vec) + "]"


class PgVectorIndexBackend(IndexBackend):
    """Postgres + pgvector backend plugin (optional dependency).

    Required backend_config keys:
      - dsn: str (psycopg connection string)
    Optional keys:
      - table: str (default 'slimx_vectors')
      - create_table: bool (default True)
      - schema: str (default 'public')
    """

    def __init__(self, index_path: Path, *, settings: Optional[IndexSettings] = None, state_path: Optional[Path] = None):
        super().__init__(index_path, settings=settings, state_path=state_path)
        cfg = self.settings.backend_config or {}
        self.dsn = str(cfg.get("dsn") or "").strip()
        if not self.dsn:
            raise ValueError("pgvector backend requires settings.backend_config['dsn']")

        self.schema = str(cfg.get("schema") or "public")
        self.table = str(cfg.get("table") or "slimx_vectors")
        self.create_table = bool(cfg.get("create_table") if cfg.get("create_table") is not None else True)

        try:
            import psycopg  # type: ignore
        except ImportError as e:
            raise ImportError(
                "pgvector backend requires optional dependency. Install with: uv sync --extra pgvector"
            ) from e

        self._psycopg = psycopg
        self.state = IndexState.load(self.state_path)

    def _connect(self):
        return self._psycopg.connect(self.dsn)

    @property
    def _fqtn(self) -> str:
        return f"{self.schema}.{self.table}"

    def load(self) -> None:
        dim = self._dim or int((self.state.embed or {}).get("dim") or 0) or int(self.settings.backend_config.get("dim", 0) or 0)
        if dim <= 0 and self.create_table:
            # Defer until set_embed_config / first upsert
            return
        if dim > 0:
            self._dim = dim

        with self._connect() as conn:
            with conn.cursor() as cur:
                # Ensure extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                if self.create_table and self._dim:
                    cur.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS {self._fqtn} (
                            chunk_id TEXT PRIMARY KEY,
                            embedding vector({int(self._dim)}),
                            text TEXT,
                            metadata JSONB
                        );
                        """
                    )
            conn.commit()

        self.state = IndexState.load(self.state_path)

    def save(self) -> None:
        # Data is already persisted in DB; only state is local.
        self._save_state_if_enabled()

    def delete(self, chunk_ids: Iterable[str]) -> int:
        ids = [str(x) for x in chunk_ids if str(x)]
        if not ids:
            return 0
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self._fqtn} WHERE chunk_id = ANY(%s);", (ids,))
                deleted = cur.rowcount or 0
            conn.commit()
        return int(deleted)

    def upsert(self, items: Iterable[EmbeddedChunk], *, skip_existing: bool = True) -> int:
        dim = self._dim or int((self.state.embed or {}).get("dim") or 0) or int(self.settings.backend_config.get("dim", 0) or 0)
        if dim <= 0:
            raise ValueError("pgvector backend needs a known embedding dim. Call set_embed_config() or set backend_config['dim'].")
        self._dim = dim

        rows = []
        for it in items:
            if len(it.vector) != dim:
                raise RuntimeError(f"Vector dim mismatch: expected {dim}, got {len(it.vector)}")
            rows.append((str(it.chunk_id), _vector_literal(list(map(float, it.vector))), it.text, dict(it.metadata)))

        if not rows:
            return 0

        with self._connect() as conn:
            with conn.cursor() as cur:
                if skip_existing:
                    cur.executemany(
                        f"""
                        INSERT INTO {self._fqtn} (chunk_id, embedding, text, metadata)
                        VALUES (%s, %s::vector, %s, %s)
                        ON CONFLICT (chunk_id) DO NOTHING;
                        """,
                        rows,
                    )
                else:
                    cur.executemany(
                        f"""
                        INSERT INTO {self._fqtn} (chunk_id, embedding, text, metadata)
                        VALUES (%s, %s::vector, %s, %s)
                        ON CONFLICT (chunk_id)
                        DO UPDATE SET embedding = EXCLUDED.embedding, text = EXCLUDED.text, metadata = EXCLUDED.metadata;
                        """,
                        rows,
                    )
            conn.commit()

        return len(rows)

    def query(self, query_vector: List[float], *, top_k: Optional[int] = None) -> List[SearchResult]:
        dim = self._dim or int((self.state.embed or {}).get("dim") or 0) or int(self.settings.backend_config.get("dim", 0) or 0)
        if dim <= 0:
            return []
        if len(query_vector) != dim:
            raise RuntimeError(f"Query vector dim {len(query_vector)} does not match index dim {dim}")

        k = int(top_k or self.settings.top_k)
        qlit = _vector_literal(list(map(float, query_vector)))

        # pgvector cosine distance: (embedding <=> query)  (smaller is better)
        # Convert to a similarity-like score: score = 1 - distance
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT chunk_id, text, metadata, 1 - (embedding <=> %s::vector) AS score
                    FROM {self._fqtn}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (qlit, qlit, k),
                )
                rows = cur.fetchall() or []

        out: List[SearchResult] = []
        for chunk_id, text, metadata, score in rows:
            out.append(
                SearchResult(
                    chunk_id=str(chunk_id),
                    score=float(score),
                    text=str(text or ""),
                    metadata=dict(metadata or {}),
                )
            )
        return out
