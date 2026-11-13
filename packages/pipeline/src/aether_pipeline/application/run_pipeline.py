"""Use case RunPipeline: extract -> normalize -> chunk -> embed -> load (platform.md, walking skeleton)."""

from __future__ import annotations

import uuid
from collections.abc import Iterable

from aether_core.domain.pipeline import (
    PipelineDefinition,
    PipelineRun,
    RunError,
    RunMode,
    StepResult,
    is_excluded,
)
from aether_core.ports import Chunker, Embedder, RunRepository, SourceReader, VectorStore


class RunPipeline:
    def __init__(
        self,
        reader: SourceReader,
        chunker: Chunker,
        embedder: Embedder,
        store: VectorStore,
        runs: RunRepository,
    ) -> None:
        self._reader = reader
        self._chunker = chunker
        self._embedder = embedder
        self._store = store
        self._runs = runs

    def execute(
        self,
        definition: PipelineDefinition,
        items: Iterable[str],
        mode: RunMode = "single_item",
        trigger: str = "cli",
    ) -> PipelineRun:
        run = PipelineRun(str(uuid.uuid4()), definition.name, mode, trigger, items=tuple(items)).start()
        self._runs.save(run)
        self._store.ensure_collection(definition.collection, self._embedder.model_info)

        for item_id in run.items:
            try:
                run = self._ingest(definition, item_id, run)
            except Exception as error:  # an item failure is recorded, never hidden
                run = run.fail_item(RunError("extract", item_id, str(error), retryable=True))

        run = run.finish()
        self._runs.save(run)
        return run

    def _ingest(self, definition: PipelineDefinition, item_id: str, run: PipelineRun) -> PipelineRun:
        item = self._reader.read(item_id)
        run = run.record(StepResult("extract", processed=1))
        if is_excluded(item.ancestor_titles, definition):
            return run.record(StepResult("verify", processed=0))

        artifact = item.artifact
        chunks = self._chunker.chunk(artifact)
        run = run.record(StepResult("chunk", processed=len(chunks)))

        embeddings = self._embedder.embed_documents([c.embedding_text for c in chunks])
        run = run.record(StepResult("embed", processed=len(embeddings)))

        # Points of older source versions go; the same version is upserted with the same IDs.
        self._store.delete_artifact(
            definition.collection, artifact.artifact_id, keep_version=artifact.source.version
        )
        self._store.upsert(definition.collection, chunks, embeddings)
        return run.record(StepResult("load", processed=len(chunks)))
