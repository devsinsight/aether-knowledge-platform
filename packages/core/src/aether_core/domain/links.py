"""Cross-source links: embeds, links and images in an item that point to items of other sources.

platform.md, "Increment: Cross-source links"; ingestion.md, "Cross-source links". A link keeps the
context that surrounds it (heading path, link text and nearby paragraph, or the table row with its
headers); the target's content is never copied into the item.

Addresses are how items are found by link: ``<provider>:<id>`` for a whole item (a page, a board) and
``<provider>:<id>:<item>`` for a part of it (a frame, a node of a board). Each reader declares the
addresses of what it indexes; each adapter's ``parse_url`` turns a URL into a ``SourceRef``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlsplit

LinkKind = Literal["embed", "link", "image"]


def address(provider: str, target_id: str, item_id: str | None = None) -> str:
    return f"{provider}:{target_id}" + (f":{item_id}" if item_id else "")


@dataclass(frozen=True)
class SourceRef:
    """A normalized target: provider, ID and, optionally, an item inside it (a board and a widget)."""

    provider: str
    target_id: str
    item_id: str | None = None

    @property
    def address(self) -> str:
        return address(self.provider, self.target_id)

    @property
    def item_address(self) -> str | None:
        return address(self.provider, self.target_id, self.item_id) if self.item_id else None


@dataclass(frozen=True)
class LinkReference:
    """A reference as a reader found it. ``url`` is empty for an image without a link to its origin."""

    url: str
    kind: LinkKind
    heading_path: tuple[str, ...] = ()
    text: str = ""  # link text, or the image title
    context: str = ""  # nearby paragraph, or the row cells with their headers ("Ship: AX-1; Status: ...")


@dataclass(frozen=True)
class Link:
    """A reference after resolution: what it points to and whether that target is indexed."""

    reference: LinkReference
    target: SourceRef | None = None
    resolved: bool = False
    artifact_id: str | None = None  # the indexed frame or page; None for a whole board or an external link
    target_title: str | None = None
    target_key: str | None = None

    def payload(self) -> dict[str, Any]:
        ref, target = self.reference, self.target
        return {
            "kind": ref.kind,
            "url": ref.url,
            "text": ref.text,
            "context": ref.context,
            "heading_path": list(ref.heading_path),
            "target": (
                {"provider": target.provider, "id": target.target_id, "item": target.item_id}
                if target
                else None
            ),
            # Flat copies of the target addresses: indexed in Qdrant for lookups in both directions.
            "target_id": target.address if target else None,
            "target_item": target.item_address if target else None,
            "resolved": self.resolved,
            "artifact_id": self.artifact_id,
            "target_title": self.target_title,
            "target_key": self.target_key,
        }


@dataclass(frozen=True)
class UnresolvedLink:
    item: str
    url: str
    reason: str


@dataclass(frozen=True)
class LinkReport:
    """Run report of the links: discovered sources, unresolved links, images without origin."""

    discovered: tuple[tuple[str, int], ...] = ()  # host of targets no reader recognizes, and how many
    unresolved: tuple[UnresolvedLink, ...] = ()
    images_without_origin: tuple[UnresolvedLink, ...] = field(default_factory=tuple)

    def merge(self, other: LinkReport) -> LinkReport:
        hosts = Counter(dict(self.discovered))
        hosts.update(dict(other.discovered))
        return LinkReport(
            tuple(sorted(hosts.items())),
            (*self.unresolved, *other.unresolved),
            (*self.images_without_origin, *other.images_without_origin),
        )

    @property
    def is_empty(self) -> bool:
        return not (self.discovered or self.unresolved or self.images_without_origin)


def discovered_hosts(urls: Iterable[str]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(Counter(urlsplit(u).netloc.lower() or u for u in urls).items()))
