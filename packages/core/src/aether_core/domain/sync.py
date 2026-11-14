"""Incremental sync model: listed and indexed versions, the sync plan and the affected context.

platform.md, "Increment: Sync". The plan is computed from versions, never guessed. The affected
context is reported to owners; related items are never re-embedded or changed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal

from aether_core.domain.contract import content_hash

ChangeKind = Literal["changed", "deleted"]

# Identifiers that pages use to reference each other in text: R-10, ADR-005, Q-03, QS-05.
_IDENTIFIER = re.compile(r"(?<![\w-])[A-Z]{1,4}-\d{1,4}(?![\w-])")


@dataclass(frozen=True)
class ListedItem:
    """An item as a source lists it, without its body: ID, current version and place in the tree."""

    item_id: str
    version: str
    title: str = ""
    ancestor_titles: tuple[str, ...] = ()


@dataclass(frozen=True)
class IndexedChunk:
    point_id: str
    content_hash: str
    embedding_hash: str


@dataclass(frozen=True)
class IndexedVersion:
    """What the state store knows about an item: the last indexed version and its chunks."""

    item_id: str
    artifact_id: str
    version: str
    key: str | None = None
    title: str = ""
    owner: str | None = None
    chunks: tuple[IndexedChunk, ...] = ()


@dataclass(frozen=True)
class SyncPlan:
    new: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()
    unchanged: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not (self.new or self.changed or self.deleted)

    @property
    def to_ingest(self) -> tuple[str, ...]:
        return (*self.new, *self.changed)


def compute_plan(listed: Iterable[ListedItem], indexed: Mapping[str, str]) -> SyncPlan:
    """Compare the versions a source lists with the last indexed ones (item ID -> version)."""
    new: list[str] = []
    changed: list[str] = []
    unchanged: list[str] = []
    seen: set[str] = set()
    for item in listed:
        seen.add(item.item_id)
        previous = indexed.get(item.item_id)
        if previous is None:
            new.append(item.item_id)
        elif previous != item.version:
            changed.append(item.item_id)
        else:
            unchanged.append(item.item_id)
    deleted = sorted(item_id for item_id in indexed if item_id not in seen)
    return SyncPlan(tuple(new), tuple(changed), tuple(unchanged), tuple(deleted))


def embedding_hash(title: str, heading_path: Iterable[str], text: str) -> str:
    """Hash of the exact text that is embedded (Chunk.embedding_text): title, heading path, chunk.

    Two chunks with the same hash get the same embedding, so it can be reused instead of computed.
    """
    context = " > ".join((title, *heading_path))
    return content_hash(f"{context}\n\n{text}")


def referenced_terms(key: str | None, texts: Iterable[str]) -> tuple[str, ...]:
    """Terms other items use to point at this change: its key and the identifiers in the changed text."""
    terms = {match.group(0) for text in texts for match in _IDENTIFIER.finditer(text)}
    if key:
        terms.add(key)
    return tuple(sorted(terms))


def mentions(text: str, term: str) -> bool:
    """True when the text contains the term as a whole identifier (R-10 does not match R-100)."""
    return re.search(rf"(?<![\w-]){re.escape(term)}(?![\w-])", text) is not None


@dataclass(frozen=True)
class Change:
    """A changed or deleted item and the text that changed (new and removed chunks)."""

    item_id: str
    artifact_id: str
    kind: ChangeKind
    key: str | None
    title: str
    texts: tuple[str, ...]


@dataclass(frozen=True)
class RelatedItem:
    artifact_id: str
    key: str | None
    title: str
    owner: str | None
    terms: tuple[str, ...]


@dataclass(frozen=True)
class AffectedContext:
    """For one changed or deleted item: the items in the index that reference it, and their owners."""

    item_id: str
    kind: ChangeKind
    key: str | None
    title: str
    terms: tuple[str, ...]
    related: tuple[RelatedItem, ...] = field(default_factory=tuple)
