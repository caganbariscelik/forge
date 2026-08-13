import sqlite3
from contextlib import contextmanager
from pathlib import Path

from app.models.run import Run

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    data TEXT NOT NULL
);
"""


class RunStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        with self._conn() as conn:
            conn.execute(_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save(self, run: Run) -> None:
        run.touch()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO runs (id, status, created_at, updated_at, data) VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(id) DO UPDATE SET status=excluded.status, updated_at=excluded.updated_at, data=excluded.data",
                (run.id, run.status.value, run.created_at.isoformat(), run.updated_at.isoformat(), run.model_dump_json()),
            )

    def get(self, run_id: str) -> Run | None:
        with self._conn() as conn:
            row = conn.execute("SELECT data FROM runs WHERE id = ?", (run_id,)).fetchone()
        return Run.model_validate_json(row[0]) if row else None

    def list(self) -> list[Run]:
        with self._conn() as conn:
            rows = conn.execute("SELECT data FROM runs ORDER BY created_at DESC").fetchall()
        return [Run.model_validate_json(r[0]) for r in rows]

    def delete(self, run_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
