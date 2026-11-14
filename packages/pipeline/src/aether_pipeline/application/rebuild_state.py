"""Use case RebuildState: rebuild the state store from the projection.

The state store is derived data, like the projection itself. When it is lost, or when items were
indexed before it existed, it is rebuilt from the points' payloads (artifact, source version,
key, owner, chunk hashes), so the next sync does not re-ingest what is already current.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from aether_core.domain.sync import IndexedChunk, IndexedVersion, embedding_hash
from aether_core.ports import StateStore, VectorStore


class RebuildState:
    def __init__(self, store: VectorStore, state: StateStore) -> None:
        self._store, self._state = store, state

    def execute(self, pipeline: str, collection: str) -> int:
        """Return the number of items recorded."""
        by_artifact: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        for point_id, payload in self._store.list_points(collection):
            by_artifact[str(payload["artifact_id"])].append((point_id, payload))

        for artifact_id, points in by_artifact.items():
            points.sort(key=lambda p: (p[1].get("position", 0), p[0]))
            first = points[0][1]
            source = first["source"]
            self._state.save(
                pipeline,
                IndexedVersion(
                    item_id=str(source["provider_artifact_id"]),
                    artifact_id=artifact_id,
                    version=str(source["version"]),
                    key=first.get("key"),
                    title=str(first.get("title") or ""),
                    owner=(first.get("authority") or {}).get("owner"),
                    chunks=tuple(
                        IndexedChunk(
                            point_id,
                            str(p.get("content_hash")),
                            embedding_hash(
                                str(p.get("title") or ""),
                                p.get("heading_path") or [],
                                str(p.get("text") or ""),
                            ),
                        )
                        for point_id, p in points
                    ),
                ),
            )
        return len(by_artifact)
