"""Run repository on SQLite (platform.md: SQLite first, PostgreSQL later)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from aether_core.domain.pipeline import PipelineRun, RunError, StepResult


class SqliteRunRepository:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, data TEXT NOT NULL)")

    def save(self, run: PipelineRun) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO runs (run_id, data) VALUES (?, ?)",
            (run.run_id, json.dumps(asdict(run), default=str)),
        )
        self._db.commit()

    def get(self, run_id: str) -> PipelineRun | None:
        row = self._db.execute("SELECT data FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        for field in ("started_at", "finished_at"):
            data[field] = datetime.fromisoformat(data[field]) if data[field] else None
        data["steps"] = tuple(StepResult(**s) for s in data["steps"])
        data["errors"] = tuple(RunError(**e) for e in data["errors"])
        data["items"] = tuple(data["items"])
        return PipelineRun(**data)
