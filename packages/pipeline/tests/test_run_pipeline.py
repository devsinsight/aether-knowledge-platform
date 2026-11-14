"""RunPipeline with in-memory fakes: idempotency, versions, exclusions, failures and incremental sync.

The sync tests follow the acceptance criteria of platform.md, "Increment: Sync".
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from aether_core.domain.contract import (
    Authority,
    Chunk,
    NormalizedKnowledgeArtifact,
    Source,
    content_hash,
    normalize,
)
from aether_core.domain.pipeline import PipelineDefinition, PipelineRun
from aether_core.domain.sync import IndexedVersion, ListedItem, mentions
from aether_core.ports import Embedding, SearchHit, SourceItem, SparseVector, StoredPoint
from aether_pipeline.application.preview_sync import PreviewSync
from aether_pipeline.application.rebuild_state import RebuildState
from aether_pipeline.application.run_pipeline import RunPipeline

DEFINITION = PipelineDefinition("confluence-aether", "confluence", {"space": "AETHER"}, "aether")

RULES = (
    "Rules\nR-10: a chain continues while kills are less than 2.0 seconds apart.\n"
    "R-11: the multiplier goes up to x8."
)
FEATURE = "Rules applied\nThe chain uses R-10 and R-11.\nThe HUD shows the chain timer."
DESIGN = "Overview\nAether is a vertical shooter."


def page(
    page_id: str,
    title: str,
    text: str,
    version: str = "1",
    key: str | None = None,
    owner: str | None = None,
    ancestors: tuple[str, ...] = ("Section 3 - Product",),
) -> SourceItem:
    artifact = normalize(
        artifact_id=f"confluence:{page_id}",
        title=title,
        source=Source("confluence", page_id, f"https://example/wiki/pages/{page_id}", version),
        authority=Authority("textual", owner=owner),
        text=text,
        key=key,
        artifact_type="feature",
        section="product",
        state="validated",
    )
    return SourceItem(artifact, ancestors)


class FakeReader:
    """A space: item ID -> page, or an exception to raise when read."""

    def __init__(self, *pages: SourceItem | tuple[str, Exception]) -> None:
        self.pages: dict[str, SourceItem | Exception] = {}
        for p in pages:
            if isinstance(p, tuple):
                self.pages[p[0]] = p[1]
            else:
                self.pages[p.artifact.source.provider_artifact_id] = p
        self.reads: list[str] = []

    def edit(self, item: SourceItem) -> None:
        self.pages[item.artifact.source.provider_artifact_id] = item

    def read(self, item_id: str) -> SourceItem:
        self.reads.append(item_id)
        item = self.pages[item_id]
        if isinstance(item, Exception):
            raise item
        return item

    def list_versions(self, scope: dict[str, str]) -> Iterable[ListedItem]:
        for item_id, item in self.pages.items():
            if isinstance(item, SourceItem):
                a = item.artifact
                yield ListedItem(item_id, a.source.version, a.title, item.ancestor_titles)


class LineChunker:
    def chunk(self, artifact: NormalizedKnowledgeArtifact) -> Sequence[Chunk]:
        return [Chunk(artifact, ("Body",), i, line) for i, line in enumerate(artifact.text.splitlines())]


class CountingEmbedder:
    model_info = {"dense_model": "fake", "dense_size": 2, "sparse_model": "fake"}

    def __init__(self) -> None:
        self.embedded: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Embedding]:
        self.embedded.extend(texts)
        return [Embedding((float(len(t)), 0.0), SparseVector((1,), (1.0,))) for t in texts]

    def embed_query(self, text: str) -> Embedding:
        return Embedding((1.0, 0.0), SparseVector((1,), (1.0,)))


class MemoryStore:
    def __init__(self) -> None:
        self.points: dict[str, tuple[dict[str, Any], Embedding]] = {}
        self.collections: dict[str, dict[str, Any]] = {}

    def payloads(self) -> list[dict[str, Any]]:
        return [p for p, _ in self.points.values()]

    def ensure_collection(self, name: str, model_info: dict[str, Any]) -> None:
        self.collections.setdefault(name, dict(model_info))

    def upsert(self, collection: str, chunks: Sequence[Chunk], embeddings: Sequence[Embedding]) -> None:
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            self.points[chunk.point_id] = (chunk.payload(), embedding)

    def delete_artifact(self, collection: str, artifact_id: str, keep_version: str | None = None) -> None:
        self.points = {
            pid: (p, e)
            for pid, (p, e) in self.points.items()
            if not (p["artifact_id"] == artifact_id and p["source"]["version"] != keep_version)
        }

    def delete_points(self, collection: str, point_ids: Sequence[str]) -> None:
        for pid in point_ids:
            self.points.pop(pid, None)

    def get_points(self, collection: str, point_ids: Sequence[str]) -> Sequence[StoredPoint]:
        return [
            StoredPoint(pid, self.points[pid][1], self.points[pid][0])
            for pid in point_ids
            if pid in self.points
        ]

    def find_mentions(self, collection: str, term: str, limit: int = 256) -> Sequence[dict[str, Any]]:
        return [p for p in self.payloads() if mentions(p["text"], term)][:limit]

    def list_points(self, collection: str) -> Iterable[tuple[str, dict[str, Any]]]:
        return [(pid, p) for pid, (p, _) in self.points.items()]

    def search(
        self, collection: str, query: Embedding, limit: int, filters: dict[str, Any] | None = None
    ) -> Sequence[SearchHit]:
        return []


class MemoryRuns:
    def __init__(self) -> None:
        self.runs: dict[str, PipelineRun] = {}

    def save(self, run: PipelineRun) -> None:
        self.runs[run.run_id] = run

    def get(self, run_id: str) -> PipelineRun | None:
        return self.runs.get(run_id)

    def latest(self, pipeline: str) -> PipelineRun | None:
        return list(self.runs.values())[-1] if self.runs else None


class MemoryState:
    def __init__(self) -> None:
        self.items: dict[tuple[str, str], IndexedVersion] = {}

    def versions(self, pipeline: str) -> dict[str, str]:
        return {i: v.version for (p, i), v in self.items.items() if p == pipeline}

    def get(self, pipeline: str, item_id: str) -> IndexedVersion | None:
        return self.items.get((pipeline, item_id))

    def save(self, pipeline: str, indexed: IndexedVersion) -> None:
        self.items[(pipeline, indexed.item_id)] = indexed

    def delete(self, pipeline: str, item_id: str) -> None:
        self.items.pop((pipeline, item_id), None)


class Platform:
    def __init__(self, reader: FakeReader) -> None:
        self.reader, self.store, self.runs, self.state = reader, MemoryStore(), MemoryRuns(), MemoryState()
        self.embedder = CountingEmbedder()
        self.pipeline = RunPipeline(reader, LineChunker(), self.embedder, self.store, self.runs, self.state)

    def run(self, *items: str) -> PipelineRun:
        return self.pipeline.execute(DEFINITION, items)

    def sync(self) -> PipelineRun:
        return self.pipeline.execute(DEFINITION, mode="incremental")

    def preview(self):  # noqa: ANN201
        return PreviewSync(self.reader, self.state).execute(DEFINITION)


def space() -> FakeReader:
    return FakeReader(
        page("1", "Game Rules", RULES, owner="Lucía Ferrer"),
        page("2", "Scoring & Chains", FEATURE, key="feature-scoring-and-chains", owner="Lucía Ferrer"),
        page("3", "Game Design", DESIGN, owner="Mei Tanaka"),
    )


# Walking skeleton behaviour


def test_same_page_twice_is_idempotent() -> None:
    p = Platform(space())
    assert p.run("2").status == "succeeded"
    first = set(p.store.points)
    p.run("2")
    assert set(p.store.points) == first and len(first) == 3


def test_rebuild_reproduces_collection_and_metadata() -> None:
    p = Platform(space())
    p.run("2")
    rebuilt = Platform(space())
    rebuilt.run("2")
    assert (rebuilt.store.points, rebuilt.store.collections) == (p.store.points, p.store.collections)


def test_new_version_replaces_old_points() -> None:
    reader = space()
    p = Platform(reader)
    p.run("2")
    reader.edit(page("2", "Scoring & Chains", FEATURE, version="2", key="feature-scoring-and-chains"))
    p.run("2")
    assert {pl["source"]["version"] for pl in p.store.payloads()} == {"2"}


def test_agent_workspace_is_not_indexed() -> None:
    reader = FakeReader(page("9", "Draft", DESIGN, ancestors=("Project Aether Home", "Agent Workspace")))
    p = Platform(reader)
    assert p.run("9").status == "succeeded"
    assert p.store.points == {}


def test_read_failure_is_recorded_and_fails_the_run() -> None:
    p = Platform(FakeReader(("1", RuntimeError("401 Unauthorized"))))
    result = p.run("1")
    assert result.status == "failed"
    assert result.errors[0].item == "1" and "401" in result.errors[0].message
    assert p.runs.get(result.run_id) == result


# Increment: Sync


def indexed_space() -> Platform:
    p = Platform(space())
    p.sync()
    p.embedder.embedded.clear()
    return p


def test_1_no_changes_gives_an_empty_plan_and_changes_nothing() -> None:
    p = indexed_space()
    before = dict(p.store.points)
    plan = p.preview()
    assert plan.is_empty and set(plan.unchanged) == {"1", "2", "3"}
    assert p.store.points == before


def test_2_one_edited_section_replaces_only_that_page_and_embeds_only_changed_chunks() -> None:
    p = indexed_space()
    others = {pid for pid, (pl, _) in p.store.points.items() if pl["artifact_id"] != "confluence:2"}
    edited = FEATURE.replace("The HUD shows the chain timer.", "The HUD shows the chain timer bar.")
    p.reader.edit(page("2", "Scoring & Chains", edited, version="2", key="feature-scoring-and-chains"))

    plan = p.preview()
    assert plan.changed == ("2",) and not plan.new and not plan.deleted

    run = p.sync()
    assert run.status == "succeeded" and run.items == ("2",)
    assert p.reader.reads[-1] == "2" and p.reader.reads.count("1") == 1  # page 1 was not read again
    assert [t.split("\n\n", 1)[1] for t in p.embedder.embedded] == ["The HUD shows the chain timer bar."]
    embed = next(s for s in run.steps if s.step == "embed")
    assert (embed.processed, embed.reused) == (1, 2)
    versions = {pl["source"]["version"] for pl in p.store.payloads() if pl["artifact_id"] == "confluence:2"}
    assert versions == {"2"}
    assert others <= set(p.store.points)  # other pages' points untouched


def test_3_a_page_deleted_or_archived_at_the_source_loses_its_points() -> None:
    p = indexed_space()
    del p.reader.pages["3"]  # archived pages are not listed by the reader either
    assert p.preview().deleted == ("3",)
    run = p.sync()
    assert run.deleted == ("3",) and run.status == "succeeded"
    assert not any(pl["artifact_id"] == "confluence:3" for pl in p.store.payloads())
    assert p.state.get(DEFINITION.name, "3") is None


def test_4_a_changed_rule_lists_the_items_that_reference_it_and_their_owners() -> None:
    p = indexed_space()
    edited = RULES.replace("2.0 seconds", "1.5 seconds")
    p.reader.edit(page("1", "Game Rules", edited, version="2", owner="Lucía Ferrer"))
    run = p.sync()

    (affected,) = run.affected
    assert affected.kind == "changed" and affected.title == "Game Rules"
    assert affected.terms == ("R-10",)  # only the identifiers in the text that changed
    (related,) = affected.related
    assert (related.key, related.owner, related.terms) == (
        "feature-scoring-and-chains",
        "Lucía Ferrer",
        ("R-10",),
    )


def test_4b_a_deleted_page_reports_who_referenced_it() -> None:
    p = indexed_space()
    del p.reader.pages["1"]
    (affected,) = p.sync().affected
    assert affected.kind == "deleted"
    assert [r.title for r in affected.related] == ["Scoring & Chains"]


def test_5_sync_again_with_no_change_processes_nothing() -> None:
    p = indexed_space()
    reads = len(p.reader.reads)
    run = p.sync()
    assert run.items == () and run.deleted == () and run.status == "succeeded"
    assert len(p.reader.reads) == reads and p.embedder.embedded == []


def test_excluded_pages_never_enter_the_plan() -> None:
    reader = space()
    reader.edit(page("8", "Agent Workspace", DESIGN, ancestors=("Project Aether Home",)))
    reader.edit(page("9", "Draft", DESIGN, ancestors=("Project Aether Home", "Agent Workspace")))
    plan = Platform(reader).preview()
    assert set(plan.new) == {"1", "2", "3"}


def test_state_store_is_rebuilt_from_the_projection() -> None:
    p = indexed_space()
    lost = MemoryState()
    assert RebuildState(p.store, lost).execute(DEFINITION.name, DEFINITION.collection) == 3
    assert lost.items == p.state.items


def test_single_item_run_keeps_the_state_store_current() -> None:
    p = Platform(space())
    p.run("1", "2", "3")
    assert p.preview().is_empty


def test_a_shorter_version_removes_stale_positions() -> None:
    reader = space()
    p = Platform(reader)
    p.run("1")
    reader.edit(page("1", "Game Rules", "Rules\nShorter."))
    p.run("1")  # same version, fewer chunks (for example after a chunking change)
    assert len([pl for pl in p.store.payloads() if pl["artifact_id"] == "confluence:1"]) == 2
    assert content_hash("Shorter.") in {pl["content_hash"] for pl in p.store.payloads()}
