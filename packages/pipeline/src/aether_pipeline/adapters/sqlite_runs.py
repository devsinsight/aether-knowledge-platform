"""Run repository and state store on SQLite (platform.md: SQLite first, PostgreSQL later)."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from aether_core.domain.links import LinkReport, UnresolvedLink
from aether_core.domain.pipeline import PipelineRun, RunError, StepResult
from aether_core.domain.sync import AffectedContext, IndexedChunk, IndexedVersion, RelatedItem


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    # The internal API serves requests from several threads; a lock serializes the writes.
    return sqlite3.connect(path, check_same_thread=False)


class SqliteRunRepository:
    def __init__(self, path: Path) -> None:
        self._db = _connect(path)
        self._lock = threading.Lock()
        self._db.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, data TEXT NOT NULL)")
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(runs)")}
        if "pipeline" not in columns:
            self._db.execute("ALTER TABLE runs ADD COLUMN pipeline TEXT")
            self._db.execute("ALTER TABLE runs ADD COLUMN started_at TEXT")

    def save(self, run: PipelineRun) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO runs (run_id, data, pipeline, started_at) VALUES (?, ?, ?, ?)",
                (
                    run.run_id,
                    json.dumps(asdict(run), default=str),
                    run.pipeline,
                    run.started_at.isoformat() if run.started_at else None,
                ),
            )
            self._db.commit()

    def get(self, run_id: str) -> PipelineRun | None:
        row = self._db.execute("SELECT data FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return _run_from_json(row[0]) if row else None

    def latest(self, pipeline: str) -> PipelineRun | None:
        row = self._db.execute(
            "SELECT data FROM runs WHERE pipeline = ? ORDER BY started_at DESC, rowid DESC LIMIT 1",
            (pipeline,),
        ).fetchone()
        return _run_from_json(row[0]) if row else None


def _run_from_json(raw: str) -> PipelineRun:
    data: dict[str, Any] = json.loads(raw)
    for field in ("started_at", "finished_at"):
        data[field] = datetime.fromisoformat(data[field]) if data.get(field) else None
    data["steps"] = tuple(StepResult(**s) for s in data["steps"])
    data["errors"] = tuple(RunError(**e) for e in data["errors"])
    data["items"] = tuple(data["items"])
    data["deleted"] = tuple(data.get("deleted", ()))
    data["affected"] = tuple(
        AffectedContext(
            item_id=a["item_id"],
            kind=a["kind"],
            key=a["key"],
            title=a["title"],
            terms=tuple(a["terms"]),
            related=tuple(RelatedItem(**{**r, "terms": tuple(r["terms"])}) for r in a["related"]),
        )
        for a in data.get("affected", ())
    )
    links = data.get("links") or {}
    data["links"] = LinkReport(
        tuple((str(h), int(n)) for h, n in links.get("discovered", ())),
        tuple(UnresolvedLink(**u) for u in links.get("unresolved", ())),
        tuple(UnresolvedLink(**u) for u in links.get("images_without_origin", ())),
    )
    return PipelineRun(**data)


class SqliteStateStore:
    """Last indexed version and chunk hashes per item; derived, rebuildable from the projection."""

    def __init__(self, path: Path) -> None:
        self._db = _connect(path)
        self._lock = threading.Lock()
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS indexed_items (pipeline TEXT NOT NULL, item_id TEXT NOT NULL,"
            " version TEXT NOT NULL, data TEXT NOT NULL, PRIMARY KEY (pipeline, item_id))"
        )

    def versions(self, pipeline: str) -> dict[str, str]:
        rows = self._db.execute("SELECT item_id, version FROM indexed_items WHERE pipeline = ?", (pipeline,))
        return dict(rows.fetchall())

    def get(self, pipeline: str, item_id: str) -> IndexedVersion | None:
        row = self._db.execute(
            "SELECT data FROM indexed_items WHERE pipeline = ? AND item_id = ?", (pipeline, item_id)
        ).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        data["chunks"] = tuple(IndexedChunk(**c) for c in data["chunks"])
        return IndexedVersion(**data)

    def save(self, pipeline: str, indexed: IndexedVersion) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO indexed_items (pipeline, item_id, version, data) VALUES (?, ?, ?, ?)",
                (pipeline, indexed.item_id, indexed.version, json.dumps(asdict(indexed))),
            )
            self._db.commit()

    def delete(self, pipeline: str, item_id: str) -> None:
        with self._lock:
            self._db.execute(
                "DELETE FROM indexed_items WHERE pipeline = ? AND item_id = ?", (pipeline, item_id)
            )
            self._db.commit()
