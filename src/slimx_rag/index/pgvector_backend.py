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
      - dim: int (optional explicit storage constraint)
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

    def _configured_dim(self) -> int:
        return int((self.settings.backend_config or {}).get("dim", 0) or 0)

    def _ensure_table(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("pgvector table dimension must be > 0")

        self._dim = dim
        if not self.create_table:
            return

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._fqtn} (
                        chunk_id TEXT PRIMARY KEY,
                        embedding vector({int(dim)}),
                        text TEXT,
                        metadata JSONB
                    );
                    """
                )
            conn.commit()

    def load(self) -> None:
        # Only backend_config['dim'] is an explicit storage constraint. Do not use
        # state.embed['dim'] here because it is provider configuration, not proof
        # of the actual stored vector dimension.
        cfg_dim = self._configured_dim()
        if cfg_dim > 0:
            self._ensure_table(cfg_dim)

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
        cfg_dim = self._configured_dim()
        rows = []

        for it in items:
            vector = list(map(float, it.vector))
            actual_dim = len(vector)
            expected_dim = self._dim or cfg_dim or actual_dim

            if self._dim is None:
                self._ensure_table(expected_dim)

            if actual_dim != int(self._dim or expected_dim):
                raise RuntimeError(f"Vector dim mismatch: expected {self._dim or expected_dim}, got {actual_dim}")

            rows.append((str(it.chunk_id), _vector_literal(vector), it.text, self._apply_metadata_whitelist(dict(it.metadata))))

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
        dim = self._dim or self._configured_dim()
        if dim <= 0:
            return []
        if len(query_vector) != dim:
            raise RuntimeError(f"Query vector dim {len(query_vector)} does not match index dim {dim}")

        k = int(top_k or self.settings.top_k)
        candidate_k = max(k, k * 4)
        qlit = _vector_literal(list(map(float, query_vector)))

        # pgvector cosine distance: (embedding <=> query)  (smaller is better)
        # Convert to a similarity-like score: score = 1 - distance.
        # SQL includes chunk_id as a deterministic secondary tie-breaker; Python
        # post-sorting keeps the same contract across all backends.
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT chunk_id, text, metadata, 1 - (embedding <=> %s::vector) AS score
                    FROM {self._fqtn}
                    ORDER BY embedding <=> %s::vector, chunk_id ASC
                    LIMIT %s;
                    """,
                    (qlit, qlit, candidate_k),
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
        return self._sort_results(out, top_k=k)
