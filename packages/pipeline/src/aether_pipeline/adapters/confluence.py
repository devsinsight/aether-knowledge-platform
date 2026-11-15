"""Confluence Cloud reader (read-only). Maps a page to the contract.

- Key, type, state and owner come from the Page Properties macro (``details``) in the storage body.
- Section comes from the page tree ("Section N - ..." ancestors).
- The text is the rendered export view (HTML), parsed and chunked downstream.
- Cross-source links (embeds, links, images, macros with a URL) are read from the storage body, next to
  the text, with their heading path and context; ``ConfluenceUrls`` recognizes only this site's pages.
- Missing fields are never invented: ``normalize`` lists them (technology.md, "index and flag").
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

import httpx

from aether_core.domain.contract import Authority, Relationship, Source, normalize
from aether_core.domain.links import LinkKind, LinkReference, SourceRef, address
from aether_core.domain.sections import section_from_ancestors
from aether_core.domain.sync import ListedItem
from aether_core.ports import SourceItem

PROVIDER = "confluence"


class _PagePropertiesParser(HTMLParser):
    """Collects the header/value rows of the first ``details`` macro in a storage body."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.properties: dict[str, str] = {}
        self._macro_depth = 0  # nesting depth of structured macros once inside ``details``
        self._done = False
        self._cell: str | None = None
        self._buffer: list[str] = []
        self._row: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._done:
            return
        if tag == "ac:structured-macro":
            if self._macro_depth:
                self._macro_depth += 1
            elif dict(attrs).get("ac:name") == "details":
                self._macro_depth = 1
            return
        if not self._macro_depth:
            return
        if tag == "tr":
            self._row = []
        elif tag in ("th", "td"):
            self._cell, self._buffer = tag, []

    def handle_endtag(self, tag: str) -> None:
        if self._done or not self._macro_depth:
            return
        if tag == "ac:structured-macro":
            self._macro_depth -= 1
            self._done = self._macro_depth == 0
        elif tag in ("th", "td") and self._cell:
            self._row.append((self._cell, " ".join("".join(self._buffer).split())))
            self._cell = None
        elif tag == "tr":
            headers = [text for kind, text in self._row if kind == "th"]
            values = [text for kind, text in self._row if kind == "td"]
            if headers and values:
                self.properties[headers[0].lower()] = values[0]

    def handle_data(self, data: str) -> None:
        if self._cell:
            self._buffer.append(data)


def parse_page_properties(storage: str) -> dict[str, str]:
    """Return Page Properties as ``{lowercase header: value}``; empty when the page has none."""
    parser = _PagePropertiesParser()
    parser.feed(storage)
    return parser.properties


_HEADING = re.compile(r"^h([1-6])$")
_BLOCKS = ("p", "li", "blockquote")
_SKIPPED_MACROS = ("details", "code", "noformat")  # Page Properties are metadata; code is not prose


def _clean(text: str) -> str:
    return " ".join(text.split())


class _ReferenceParser(HTMLParser):
    """Embeds, links, images and URL macro parameters of a storage body, each with its context.

    Markup verified on the AETHER space (November 2026): a board embed is ``<a href=...
    data-card-appearance="embed">`` at the top level, after the paragraph that introduces it; links in
    tables sit in ``<td><p><a>``; the first row of a table holds its headers (``<th>``).
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.references: list[LinkReference] = []
        self._headings: list[tuple[int, str]] = []
        self._heading: tuple[int, list[str]] | None = None
        self._macros: list[str] = []
        self._skip = 0  # depth inside a skipped macro
        self._block: tuple[str, list[str]] | None = None  # outermost paragraph or list item
        self._last_block = ""
        self._pending_block: list[LinkReference] = []
        self._anchor: tuple[str, LinkKind, list[str]] | None = None
        self._image: list[str] | None = None
        self._parameter: str | None = None
        self._tables = 0
        self._header: list[str] = []
        self._row: list[tuple[str, str]] = []
        self._cell: list[str] | None = None
        self._pending_row: list[LinkReference] = []

    @property
    def _path(self) -> tuple[str, ...]:
        return tuple(text for _, text in self._headings)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: v or "" for k, v in attrs}
        if tag == "ac:structured-macro":
            name = a.get("ac:name", "")
            self._macros.append(name)
            if self._skip or name in _SKIPPED_MACROS:
                self._skip += 1
            return
        if self._skip:
            return
        if (match := _HEADING.match(tag)) is not None:
            self._heading = (int(match.group(1)), [])
        elif tag == "table":
            self._tables += 1
            if self._tables == 1:
                self._header = []
        elif tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []
        elif tag in _BLOCKS and self._block is None and not self._tables:
            self._block = (tag, [])
        elif tag == "a" and a.get("href", "").startswith(("http://", "https://")):
            kind: LinkKind = "embed" if a.get("data-card-appearance") in ("embed", "block") else "link"
            self._anchor = (a["href"], kind, [])
        elif tag == "ac:image":
            self._image = [a.get("ac:title") or a.get("ac:alt") or ""]
        elif tag == "ri:url" and self._image is not None:
            self._image.append(a.get("ri:value", ""))
        elif tag == "ac:parameter" and a.get("ac:name", "").lower() in ("url", "src"):
            self._parameter = ""

    def handle_endtag(self, tag: str) -> None:
        if tag == "ac:structured-macro":
            if self._macros:
                self._macros.pop()
            if self._skip:
                self._skip -= 1
            return
        if self._skip:
            return
        if _HEADING.match(tag) is not None and self._heading is not None:
            level, parts = self._heading
            self._headings = [h for h in self._headings if h[0] < level] + [(level, _clean("".join(parts)))]
            self._heading = None
        elif tag == "a" and self._anchor is not None:
            url, kind, parts = self._anchor
            self._anchor = None
            self._add(LinkReference(url, kind, self._path, _clean("".join(parts))))
        elif tag == "ac:image" and self._image is not None:
            title, *urls = self._image
            self._image = None
            self._add(LinkReference(urls[0] if urls else "", "image", self._path, _clean(title)))
        elif tag == "ac:parameter" and self._parameter is not None:
            url, self._parameter = self._parameter.strip(), None
            if url.startswith(("http://", "https://")):
                macro = self._macros[-1] if self._macros else ""
                self._add(LinkReference(url, "embed", self._path, f"{macro} macro" if macro else ""))
        elif tag in ("td", "th") and self._cell is not None:
            self._row.append((tag, _clean("".join(self._cell))))
            self._cell = None
        elif tag == "tr":
            if not self._header and self._row and all(kind == "th" for kind, _ in self._row):
                self._header = [text for _, text in self._row]
            cells = [
                f"{self._header[i]}: {text}" if i < len(self._header) and self._header[i] else text
                for i, (_, text) in enumerate(self._row)
            ]
            self.references.extend(
                _with_context(r, "; ".join(c for c in cells if c)) for r in self._pending_row
            )
            self._pending_row = []
        elif tag == "table":
            self._tables -= 1
        elif self._block is not None and tag == self._block[0]:
            text = _clean("".join(self._block[1]))
            self.references.extend(_with_context(r, text) for r in self._pending_block)
            self._pending_block, self._block = [], None
            self._last_block = text or self._last_block

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._parameter is not None:
            self._parameter += data
            return
        for buffer in (
            self._heading[1] if self._heading else None,
            self._block[1] if self._block else None,
            self._cell,
            self._anchor[2] if self._anchor else None,
        ):
            if buffer is not None:
                buffer.append(data)

    def _add(self, reference: LinkReference) -> None:
        if self._cell is not None:
            self._pending_row.append(reference)
        elif self._block is not None:
            self._pending_block.append(reference)
        else:  # a card or macro on its own: the paragraph before it introduces it
            self.references.append(_with_context(reference, self._last_block))


def _with_context(reference: LinkReference, context: str) -> LinkReference:
    return LinkReference(reference.url, reference.kind, reference.heading_path, reference.text, context)


def extract_references(storage: str) -> tuple[LinkReference, ...]:
    """Embeds, links, images and URL macro parameters of a page, with their heading path and context."""
    parser = _ReferenceParser()
    parser.feed(storage)
    parser.close()
    return tuple(parser.references)


class ConfluenceUrls:
    """``parse_url`` for Confluence: page URLs of this site only (``/wiki/spaces/<space>/pages/<id>``,
    ``/wiki/pages/viewpage.action?pageId=<id>``)."""

    provider = PROVIDER
    noun = "page"

    def __init__(self, base_url: str) -> None:
        self._host = urlsplit(base_url).netloc.lower()

    def parse_url(self, url: str) -> SourceRef | None:
        parts = urlsplit(url.strip())
        if not self._host or parts.netloc.lower() != self._host:
            return None
        match = re.search(r"/pages/(\d+)(?:/|$)", unquote(parts.path))
        page_id = match.group(1) if match else (parse_qs(parts.query).get("pageId") or [""])[0]
        return SourceRef(PROVIDER, page_id) if page_id.isdigit() else None


def relationships_from(props: dict[str, str]) -> tuple[Relationship, ...]:
    """Keys a page declares in its Page Properties (work records: Story and Feature)."""
    found: list[Relationship] = []
    story = props.get("story", "").split("(")[0].strip()
    if story:
        found.append(Relationship("work_record_of", story))
    if props.get("feature"):
        found.append(Relationship("part_of", props["feature"].strip()))
    return tuple(found)


def map_page(page: dict[str, Any], base_url: str) -> SourceItem:
    """Map a Confluence REST content response (with body, version and ancestors) to the contract."""
    storage = page["body"]["storage"]["value"]
    props = parse_page_properties(storage)
    ancestors = tuple(a["title"] for a in page.get("ancestors", []))
    page_id = str(page["id"])
    artifact = normalize(
        artifact_id=f"{PROVIDER}:{page_id}",
        title=page["title"],
        source=Source(
            provider=PROVIDER,
            provider_artifact_id=page_id,
            canonical_url=base_url.rstrip("/") + page["_links"]["webui"],
            version=str(page["version"]["number"]),
        ),
        authority=Authority("textual", owner=props.get("owner")),
        text=page["body"]["export_view"]["value"],
        key=props.get("key"),
        artifact_type=props.get("type"),
        section=section_from_ancestors(ancestors),
        state=props.get("state"),
        archived_at_source=page.get("status") == "archived",
        relationships=relationships_from(props),
    )
    return SourceItem(
        artifact=artifact,
        ancestor_titles=ancestors,
        references=extract_references(storage),
        addresses=(address(PROVIDER, page_id),),
    )


class ConfluenceReader:
    """SourceReader for Confluence Cloud. Credentials come from the environment, read-only account."""

    def __init__(self, base_url: str, email: str, api_token: str, client: httpx.Client | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(auth=(email, api_token), timeout=30.0)

    def read(self, item_id: str) -> SourceItem:
        response = self._client.get(
            f"{self._base_url}/rest/api/content/{item_id}",
            params={"expand": "body.storage,body.export_view,version,ancestors"},
        )
        response.raise_for_status()
        return map_page(response.json(), self._base_url)

    def list_versions(self, scope: dict[str, Any]) -> Iterable[ListedItem]:
        """Current pages of the space with their version and ancestors; bodies are not read.

        Archived and deleted pages are not listed, so the sync plan reports them as deleted.
        """
        start, limit = 0, 100
        while True:
            response = self._client.get(
                f"{self._base_url}/rest/api/content",
                params={
                    "spaceKey": scope["space"],
                    "type": "page",
                    "status": "current",
                    "expand": "version,ancestors",
                    "start": start,
                    "limit": limit,
                },
            )
            response.raise_for_status()
            data = response.json()
            for page in data.get("results", []):
                yield ListedItem(
                    item_id=str(page["id"]),
                    version=str(page["version"]["number"]),
                    title=page["title"],
                    ancestor_titles=tuple(a["title"] for a in page.get("ancestors", [])),
                )
            if data.get("size", 0) < limit or "next" not in data.get("_links", {}):
                return
            start += limit
