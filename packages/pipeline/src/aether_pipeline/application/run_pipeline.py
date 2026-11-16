"""Use case RunPipeline: extract -> normalize -> chunk -> embed -> load, in three modes.

- ``single_item`` and ``full``: ingest the given items.
- ``incremental`` (sync): detect new, changed and deleted items from source versions (the same
  detection as PreviewSync), ingest new and changed items, delete the points of deleted ones, and
  report the affected context.

One pipeline in every mode (platform.md, "Three capabilities over one pipeline"): the chunks and
point IDs of a page are the same whichever mode indexed it. Embeddings of chunks whose embedded text
did not change are reused from the previous version's points, so only changed text is embedded.

Cross-source links: after chunking, ResolveLinks relates the item's embeds, links and images to their
targets; after loading or deleting an item, the links of other items that point to it are refreshed.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from aether_core.domain.contract import Chunk
from aether_core.domain.pipeline import (
    PipelineDefinition,
    PipelineRun,
    RunError,
    RunMode,
    StepResult,
    is_excluded,
)
from aether_core.domain.sync import Change, IndexedChunk, IndexedVersion, embedding_hash
from aether_core.ports import (
    Chunker,
    Embedder,
    Embedding,
    RunRepository,
    SourceReader,
    StateStore,
    UrlParser,
    VectorStore,
)
from aether_pipeline.application.preview_sync import detect_changes
from aether_pipeline.application.report_affected_context import ReportAffectedContext
from aether_pipeline.application.resolve_links import ResolveLinks


@dataclass(frozen=True)
class _Ingested:
    run: PipelineRun
    change: Change | None = None


class RunPipeline:
    def __init__(
        self,
        reader: SourceReader,
        chunker: Chunker,
        embedder: Embedder,
        store: VectorStore,
        runs: RunRepository,
        state: StateStore,
        url_parsers: Sequence[UrlParser] = (),
    ) -> None:
        self._reader = reader
        self._chunker = chunker
        self._embedder = embedder
        self._store = store
        self._runs = runs
        self._state = state
        self._affected = ReportAffectedContext(store)
        self._links = ResolveLinks(url_parsers, store)

    def execute(
        self,
        definition: PipelineDefinition,
        items: Iterable[str] = (),
        mode: RunMode = "single_item",
        trigger: str = "cli",
        run_id: str | None = None,
    ) -> PipelineRun:
        deleted: tuple[str, ...] = ()
        if mode == "incremental":
            plan = detect_changes(self._reader, self._state, definition)
            items, deleted = plan.to_ingest, plan.deleted
        run = PipelineRun(
            run_id or str(uuid.uuid4()), definition.name, mode, trigger, items=tuple(items), deleted=deleted
        ).start()
        self._runs.save(run)
        self._store.ensure_collection(definition.collection, self._embedder.model_info)

        changes: list[Change] = []
        for item_id in run.items:
            try:
                result = self._ingest(definition, item_id, run)
            except Exception as error:  # an item failure is recorded, never hidden
                run = run.fail_item(RunError("extract", item_id, str(error), retryable=True))
                continue
            run = result.run
            if result.change:
                changes.append(result.change)

        for item_id in run.deleted:
            try:
                change = self._delete(definition, item_id)
            except Exception as error:
                run = run.fail_item(RunError("delete", item_id, str(error), retryable=True))
                continue
            run = run.record(StepResult("delete", processed=1, item=item_id))
            if change:
                changes.append(change)

        if mode == "incremental" and changes:
            run = run.report(self._affected.execute(definition.collection, changes))
        run = run.finish()
        self._runs.save(run)
        return run

    def _ingest(self, definition: PipelineDefinition, item_id: str, run: PipelineRun) -> _Ingested:
        item = self._reader.read(item_id)
        artifact = item.artifact
        run = run.record(StepResult("extract", processed=1, item=item_id))
        if is_excluded(item.ancestor_titles, definition, artifact.title):
            return _Ingested(run.record(StepResult("verify", processed=0, item=item_id)))

        collection = definition.collection
        previous = self._state.get(definition.name, item_id)
        old_points = (
            self._store.get_points(collection, [c.point_id for c in previous.chunks]) if previous else []
        )
        cached: dict[str, Embedding] = {
            embedding_hash(
                str(p.payload.get("title") or ""),
                p.payload.get("heading_path") or [],
                str(p.payload.get("text") or ""),
            ): p.embedding
            for p in old_points
        }

        chunks = self._chunker.chunk(artifact)
        run = run.record(StepResult("chunk", processed=len(chunks), item=item_id))
        chunks, links = self._links.execute(collection, item, chunks)
        run = run.report_links(links).record(
            StepResult("enrich", processed=sum(len(c.links) for c in chunks), item=item_id)
        )

        hashes = [_embedding_hash(c) for c in chunks]
        to_embed = [c.embedding_text for c, h in zip(chunks, hashes, strict=True) if h not in cached]
        computed = iter(self._embedder.embed_documents(to_embed) if to_embed else [])
        embeddings = [cached[h] if h in cached else next(computed) for h in hashes]
        run = run.record(
            StepResult("embed", processed=len(to_embed), reused=len(chunks) - len(to_embed), item=item_id)
        )

        self._store.upsert(collection, chunks, embeddings)
        # Points of older source versions go; so do positions the current version no longer has.
        self._store.delete_artifact(collection, artifact.artifact_id, keep_version=artifact.source.version)
        current_ids = {c.point_id for c in chunks}
        stale = [c.point_id for c in previous.chunks if c.point_id not in current_ids] if previous else []
        if stale:
            self._store.delete_points(collection, stale)
        run = run.record(StepResult("load", processed=len(chunks), item=item_id))
        # Links of other items that were waiting for this one (or point to its old version) resolve now.
        self._links.refresh(collection, item.addresses)

        self._state.save(
            definition.name,
            IndexedVersion(
                item_id=item_id,
                artifact_id=artifact.artifact_id,
                version=artifact.source.version,
                key=artifact.key,
                title=artifact.title,
                owner=artifact.authority.owner,
                chunks=tuple(
                    IndexedChunk(c.point_id, c.content_hash, h) for c, h in zip(chunks, hashes, strict=True)
                ),
            ),
        )

        if previous is None:
            return _Ingested(run)
        old_hashes = {str(p.payload.get("content_hash")) for p in old_points}
        new_hashes = {c.content_hash for c in chunks}
        changed_texts = [c.text for c in chunks if c.content_hash not in old_hashes] + [
            str(p.payload.get("text") or "")
            for p in old_points
            if p.payload.get("content_hash") not in new_hashes
        ]
        if not changed_texts:
            return _Ingested(run)
        change = Change(
            item_id,
            artifact.artifact_id,
            "changed",
            artifact.key,
            artifact.title,
            tuple(changed_texts),
            item.addresses,
        )
        return _Ingested(run, change)

    def _delete(self, definition: PipelineDefinition, item_id: str) -> Change | None:
        previous = self._state.get(definition.name, item_id)
        if previous is None:
            return None
        old_points = self._store.get_points(definition.collection, [c.point_id for c in previous.chunks])
        self._store.delete_artifact(definition.collection, previous.artifact_id)
        self._state.delete(definition.name, item_id)
        addresses = tuple(str(a) for p in old_points[:1] for a in p.payload.get("addresses") or ())
        self._links.refresh(definition.collection, addresses)  # links to it are no longer resolved
        texts = tuple(str(p.payload.get("text") or "") for p in old_points)
        return Change(
            item_id, previous.artifact_id, "deleted", previous.key, previous.title, texts, addresses
        )


def _embedding_hash(chunk: Chunk) -> str:
    return embedding_hash(chunk.artifact.title, chunk.heading_path, chunk.text)
