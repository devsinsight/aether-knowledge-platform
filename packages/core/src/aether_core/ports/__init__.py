"""Outbound ports (platform.md, Outbound ports). Adapters implement them; use cases depend on them."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from aether_core.domain.contract import Chunk, NormalizedKnowledgeArtifact
from aether_core.domain.links import LinkReference, SourceRef
from aether_core.domain.pipeline import PipelineRun
from aether_core.domain.sync import IndexedVersion, ListedItem


@dataclass(frozen=True)
class SourceItem:
    """One item read from a source: the normalized artifact and where it sits in the source tree."""

    artifact: NormalizedKnowledgeArtifact
    ancestor_titles: tuple[str, ...] = ()
    # Embeds, links and images found in the item, with their context (cross-source links).
    references: tuple[LinkReference, ...] = ()
    # How links from other items address this one: "<provider>:<id>" and "<provider>:<id>:<part>".
    addresses: tuple[str, ...] = ()


class UrlParser(Protocol):
    """One per adapter: recognizes only its own source's URLs and normalizes the ID (ingestion.md)."""

    @property
    def provider(self) -> str: ...

    @property
    def noun(self) -> str:
        """What the source holds, for the line an embed without text adds ("Embedded board: ...")."""
        ...

    def parse_url(self, url: str) -> SourceRef | None: ...


class SourceReader(Protocol):
    """Reads one provider, read-only, and maps each item to the contract."""

    def read(self, item_id: str) -> SourceItem: ...

    def list_versions(self, scope: dict[str, Any]) -> Iterable[ListedItem]:
        """Every item in scope with its current version, without reading bodies (wiki page version,
        work item revision, Miro modifiedAt per frame, file commit or hash)."""
        ...


class Chunker(Protocol):
    def chunk(self, artifact: NormalizedKnowledgeArtifact) -> Sequence[Chunk]: ...


@dataclass(frozen=True)
class SparseVector:
    indices: tuple[int, ...]
    values: tuple[float, ...]


@dataclass(frozen=True)
class Embedding:
    dense: tuple[float, ...]
    sparse: SparseVector


class Embedder(Protocol):
    @property
    def model_info(self) -> dict[str, Any]:
        """Dense and sparse model names and dense size, recorded in the collection metadata."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Embedding]: ...

    def embed_query(self, text: str) -> Embedding: ...


@dataclass(frozen=True)
class SearchHit:
    point_id: str
    score: float
    payload: dict[str, Any]


@dataclass(frozen=True)
class StoredPoint:
    point_id: str
    embedding: Embedding
    payload: dict[str, Any]


class VectorStore(Protocol):
    def ensure_collection(self, name: str, model_info: dict[str, Any]) -> None: ...

    def upsert(self, collection: str, chunks: Sequence[Chunk], embeddings: Sequence[Embedding]) -> None: ...

    def delete_artifact(self, collection: str, artifact_id: str, keep_version: str | None = None) -> None:
        """Delete an artifact's points, except those of keep_version (older versions go)."""
        ...

    def delete_points(self, collection: str, point_ids: Sequence[str]) -> None: ...

    def get_points(self, collection: str, point_ids: Sequence[str]) -> Sequence[StoredPoint]:
        """Points with their vectors and payload; missing IDs are skipped."""
        ...

    def find_mentions(self, collection: str, term: str, limit: int = 256) -> Sequence[dict[str, Any]]:
        """Payloads of the chunks whose text contains the term as a whole identifier."""
        ...

    def find_by_address(self, collection: str, addresses: Sequence[str]) -> Sequence[dict[str, Any]]:
        """Payloads (without text) of the chunks that declare any of these addresses."""
        ...

    def find_linking(self, collection: str, addresses: Sequence[str]) -> Sequence[tuple[str, dict[str, Any]]]:
        """Point IDs and payloads of the chunks whose links point to any of these addresses."""
        ...

    def set_links(self, collection: str, point_id: str, links: list[dict[str, Any]]) -> None:
        """Replace the ``links`` payload of one point (text and vectors stay as they are)."""
        ...

    def list_points(self, collection: str) -> Iterable[tuple[str, dict[str, Any]]]:
        """Every point ID with its payload (used to rebuild the state store from the projection)."""
        ...

    def search(
        self,
        collection: str,
        query: Embedding,
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> Sequence[SearchHit]: ...


class RunRepository(Protocol):
    def save(self, run: PipelineRun) -> None: ...

    def get(self, run_id: str) -> PipelineRun | None: ...

    def latest(self, pipeline: str) -> PipelineRun | None: ...


@dataclass(frozen=True)
class RawWorkRecord:
    front_matter: dict[str, Any]
    body: str


class WorkRecordFiles(Protocol):
    """Work record files in the folder mounted read-only into the publisher."""

    def read(self, path: str) -> RawWorkRecord:
        """Raises WorkRecordRejectedError for a path outside the folder or a file without front matter."""
        ...


@dataclass(frozen=True)
class PublishedPage:
    page_id: str
    version: int
    title: str
    url: str
    properties: dict[str, str]


class WikiPublisher(Protocol):
    """Writes work record pages through the wiki's REST API (LlamaIndex has no writers)."""

    def find_page(self, title: str) -> PublishedPage | None: ...

    def ensure_parent(self, parent_id: str, title: str) -> str:
        """The ID of the child page with this title under parent_id, created empty if missing."""
        ...

    def create_or_update(
        self,
        parent_id: str,
        title: str,
        body_markdown: str,
        properties: dict[str, str],
        expected_version: int | None,
    ) -> PublishedPage:
        """Create the page, or update it when expected_version (optimistic concurrency) still holds."""
        ...

    def set_properties(self, page: PublishedPage, properties: dict[str, str], note: str) -> PublishedPage:
        """Replace the Page Properties and add a note to the Validation section; body otherwise kept."""
        ...


class WorkItemLookup(Protocol):
    def exists(self, work_item_id: int) -> bool: ...


class SecretScanner(Protocol):
    def scan(self, text: str) -> list[str]:
        """Names of the secret patterns found (never the secret itself)."""
        ...


class SyncRequester(Protocol):
    def sync_items(self, item_ids: Sequence[str], trigger: str) -> str:
        """Ask the sync service to process these items; returns the run ID."""
        ...


class StateStore(Protocol):
    """Last indexed version and chunk hashes per item. Derived state: rebuildable from the projection."""

    def versions(self, pipeline: str) -> dict[str, str]:
        """Item ID -> last indexed version."""
        ...

    def get(self, pipeline: str, item_id: str) -> IndexedVersion | None: ...

    def save(self, pipeline: str, indexed: IndexedVersion) -> None: ...

    def delete(self, pipeline: str, item_id: str) -> None: ...
