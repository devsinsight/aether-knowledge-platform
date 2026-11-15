"""Use case ResolveLinks: relate the embeds, links and images of an item to their targets.

platform.md, "Increment: Cross-source links". For each reference a reader found:
1. the adapter whose ``parse_url`` recognizes the URL gives the normalized target (readers never call
   each other: this use case holds one parser per adapter);
2. the target is looked up in the projection by address:
   - indexed: ``resolved``, linked to the frame when the URL names an item, otherwise to the whole
     item (a board relates to the board, never to a guessed frame);
   - in a source with a reader but not indexed yet: ``resolved: false`` until that source's own run
     completes it (``refresh``);
   - recognized by no adapter: an external link with its URL, reported as a discovered source;
3. the link is attached to the chunk whose heading path holds the reference, with its context. An
   embed without text adds one deterministic line to that chunk ("Embedded board: <title or URL>").

The target's content is never copied into the item. Nothing is written to any source.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from aether_core.domain.contract import Chunk
from aether_core.domain.links import (
    Link,
    LinkReference,
    LinkReport,
    SourceRef,
    UnresolvedLink,
    discovered_hosts,
)
from aether_core.ports import SourceItem, UrlParser, VectorStore


class ResolveLinks:
    def __init__(self, parsers: Sequence[UrlParser], store: VectorStore) -> None:
        self._parsers = tuple(parsers)
        self._store = store

    def parse(self, url: str) -> tuple[SourceRef | None, str]:
        """The normalized target and the noun of its source ("board", "page"); (None, "") if external."""
        for parser in self._parsers:
            target = parser.parse_url(url)
            if target is not None:
                return target, parser.noun
        return None, ""

    def execute(
        self, collection: str, item: SourceItem, chunks: Sequence[Chunk]
    ) -> tuple[list[Chunk], LinkReport]:
        item_id = item.artifact.source.provider_artifact_id
        attached: dict[int, list[Link]] = {}
        lines: dict[int, list[str]] = {}
        external: list[str] = []
        unresolved: list[UnresolvedLink] = []
        images: list[UnresolvedLink] = []
        for reference in item.references:
            if not reference.url:
                images.append(UnresolvedLink(item_id, "", f"image without origin: {reference.text or '-'}"))
                continue
            target, noun = self.parse(reference.url)
            if target is not None and {target.address, target.item_address} & set(item.addresses):
                continue  # a link of an item to itself relates nothing
            link = self._resolve(collection, reference, target)
            if target is None:
                external.append(reference.url)
            elif not link.resolved:
                unresolved.append(
                    UnresolvedLink(item_id, reference.url, f"{target.provider} target not indexed")
                )
            index = _chunk_for(reference, chunks)
            if index is None:
                continue
            attached.setdefault(index, []).append(link)
            if reference.kind == "embed" and reference.text.strip() in ("", reference.url.strip()):
                lines.setdefault(index, []).append(
                    f"Embedded {noun or 'content'}: {link.target_title or reference.url}"
                )

        result = []
        for index, chunk in enumerate(chunks):
            text = chunk.text
            for line in dict.fromkeys(lines.get(index, [])):
                if line not in text:
                    text = f"{text}\n{line}"
            result.append(
                replace(
                    chunk,
                    text=text,
                    links=tuple(attached.get(index, ())),
                    addresses=item.addresses,
                    ancestors=item.ancestor_titles,
                )
            )
        return result, LinkReport(discovered_hosts(external), tuple(unresolved), tuple(images))

    def refresh(self, collection: str, addresses: Sequence[str]) -> int:
        """Re-resolve the links of other items that point to these addresses, after the target was
        indexed or deleted. Only the ``links`` payload changes; text and vectors stay. Returns the
        number of points updated."""
        if not addresses:
            return 0
        wanted = set(addresses)
        updated = 0
        for point_id, payload in self._store.find_linking(collection, list(addresses)):
            links: list[dict[str, Any]] = [dict(entry) for entry in payload.get("links") or []]
            changed = False
            for position, entry in enumerate(links):
                if not ({entry.get("target_id"), entry.get("target_item")} & wanted):
                    continue
                target = entry.get("target") or {}
                ref = SourceRef(str(target["provider"]), str(target["id"]), target.get("item"))
                reference = LinkReference(
                    str(entry.get("url") or ""),
                    entry.get("kind") or "link",
                    tuple(entry.get("heading_path") or ()),
                    str(entry.get("text") or ""),
                    str(entry.get("context") or ""),
                )
                fresh = self._resolve(collection, reference, ref).payload()
                if fresh != entry:
                    links[position], changed = fresh, True
            if changed:
                self._store.set_links(collection, point_id, links)
                updated += 1
        return updated

    def _resolve(self, collection: str, reference: LinkReference, target: SourceRef | None) -> Link:
        if target is None:
            return Link(reference)
        wanted = target.item_address or target.address
        found = self._store.find_by_address(collection, [wanted])
        if not found:
            return Link(reference, target)
        by_artifact = {str(p.get("artifact_id")): p for p in found}
        # The artifact whose own address (its first one) is the target: a page, a frame, a board
        # without frames. A node address resolves to the frame that contains the node.
        own = [p for p in by_artifact.values() if (p.get("addresses") or [None])[0] == wanted]
        if target.item_id or own:
            chosen = own[0] if own else by_artifact[sorted(by_artifact)[0]]
            return Link(
                reference,
                target,
                True,
                str(chosen.get("artifact_id")),
                _str(chosen.get("title")),
                _str(chosen.get("key")),
            )
        # A whole board with frames: related to the board, never to one guessed frame.
        container = next(
            (p["ancestor_titles"][-1] for p in by_artifact.values() if p.get("ancestor_titles")), None
        )
        return Link(reference, target, True, None, container)


def _chunk_for(reference: LinkReference, chunks: Sequence[Chunk]) -> int | None:
    """The chunk whose heading path holds the reference; within it, the one that shows the link."""
    if not chunks:
        return None
    marks = [m for m in (reference.url, reference.text) if m]
    same = [i for i, c in enumerate(chunks) if c.heading_path == reference.heading_path]
    if not same:
        # The longest heading path that contains the reference's (Docling may drop a level).
        depth = max((_common(c.heading_path, reference.heading_path) for c in chunks), default=0)
        same = [i for i, c in enumerate(chunks) if _common(c.heading_path, reference.heading_path) == depth]
    showing = [i for i in same if any(m in chunks[i].text for m in marks)]
    return (showing or same)[0]


def _common(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def _str(value: object) -> str | None:
    return str(value) if value else None
