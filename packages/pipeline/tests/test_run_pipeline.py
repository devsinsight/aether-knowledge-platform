"""RunPipeline with in-memory fakes: idempotency, version replacement, exclusions, failures."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from aether_core.domain.contract import Authority, Chunk, NormalizedKnowledgeArtifact, Source, normalize
from aether_core.domain.pipeline import PipelineDefinition, PipelineRun
from aether_core.ports import Embedding, SearchHit, SourceItem, SparseVector
from aether_pipeline.application.run_pipeline import RunPipeline

DEFINITION = PipelineDefinition("confluence-aether", "confluence", {"space": "AETHER"}, "aether")


def artifact(version: str = "1") -> NormalizedKnowledgeArtifact:
    return normalize(
        artifact_id="confluence:1",
        title="Scoring & Chains",
        source=Source("confluence", "1", "https://example/wiki/pages/1", version),
        authority=Authority("textual"),
        text="Chains\nA chain continues while kills are less than 2.0 seconds apart.",
        key="feature-scoring-and-chains",
        artifact_type="feature",
        section="product",
        state="draft",
    )


class FakeReader:
    def __init__(self, item: SourceItem | Exception) -> None:
        self.item = item

    def read(self, item_id: str) -> SourceItem:
        if isinstance(self.item, Exception):
            raise self.item
        return self.item

    def list_items(self, scope: dict[str, str]) -> Iterable[str]:
        return []


class LineChunker:
    def chunk(self, artifact: NormalizedKnowledgeArtifact) -> Sequence[Chunk]:
        return [Chunk(artifact, ("Rules",), i, line) for i, line in enumerate(artifact.text.splitlines())]


class FakeEmbedder:
    model_info = {"dense_model": "fake", "dense_size": 2, "sparse_model": "fake"}

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Embedding]:
        return [Embedding((1.0, 0.0), SparseVector((1,), (1.0,))) for _ in texts]

    def embed_query(self, text: str) -> Embedding:
        return Embedding((1.0, 0.0), SparseVector((1,), (1.0,)))


class MemoryStore:
    def __init__(self) -> None:
        self.points: dict[str, dict[str, Any]] = {}
        self.collections: dict[str, dict[str, Any]] = {}

    def ensure_collection(self, name: str, model_info: dict[str, Any]) -> None:
        self.collections.setdefault(name, dict(model_info))

    def upsert(self, collection: str, chunks: Sequence[Chunk], embeddings: Sequence[Embedding]) -> None:
        for chunk in chunks:
            self.points[chunk.point_id] = chunk.payload()

    def delete_artifact(self, collection: str, artifact_id: str, keep_version: str | None = None) -> None:
        self.points = {
            pid: p
            for pid, p in self.points.items()
            if not (p["artifact_id"] == artifact_id and p["source"]["version"] != keep_version)
        }

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


def make(reader: FakeReader, store: MemoryStore, runs: MemoryRuns | None = None) -> RunPipeline:
    return RunPipeline(reader, LineChunker(), FakeEmbedder(), store, runs or MemoryRuns())


def test_same_page_twice_is_idempotent() -> None:
    """Criterion 2: unchanged page -> same point IDs and count."""
    store = MemoryStore()
    use_case = make(FakeReader(SourceItem(artifact(), ("Section 3 - Product",))), store)
    assert use_case.execute(DEFINITION, ["1"]).status == "succeeded"
    first = set(store.points)
    use_case.execute(DEFINITION, ["1"])
    assert set(store.points) == first and len(first) == 2


def test_rebuild_reproduces_collection_and_metadata() -> None:
    """Criterion 3: deleting the collection and running again reproduces it."""
    store = MemoryStore()
    make(FakeReader(SourceItem(artifact())), store).execute(DEFINITION, ["1"])
    before = (dict(store.points), dict(store.collections))
    rebuilt = MemoryStore()
    make(FakeReader(SourceItem(artifact())), rebuilt).execute(DEFINITION, ["1"])
    assert (rebuilt.points, rebuilt.collections) == before


def test_new_version_replaces_old_points() -> None:
    store = MemoryStore()
    make(FakeReader(SourceItem(artifact("1"))), store).execute(DEFINITION, ["1"])
    make(FakeReader(SourceItem(artifact("2"))), store).execute(DEFINITION, ["1"])
    assert {p["source"]["version"] for p in store.points.values()} == {"2"}


def test_agent_workspace_is_not_indexed() -> None:
    store = MemoryStore()
    item = SourceItem(artifact(), ("Project Aether Home", "Agent Workspace"))
    assert make(FakeReader(item), store).execute(DEFINITION, ["1"]).status == "succeeded"
    assert store.points == {}


def test_read_failure_is_recorded_and_fails_the_run() -> None:
    runs = MemoryRuns()
    result = make(FakeReader(RuntimeError("401 Unauthorized")), MemoryStore(), runs).execute(
        DEFINITION, ["1"]
    )
    assert result.status == "failed"
    assert result.errors[0].item == "1" and "401" in result.errors[0].message
    assert runs.get(result.run_id) == result
