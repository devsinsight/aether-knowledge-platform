"""Confluence Cloud reader (read-only). Maps a page to the contract.

- Key, type, state and owner come from the Page Properties macro (``details``) in the storage body.
- Section comes from the page tree ("Section N - ..." ancestors).
- The text is the rendered export view (HTML), parsed and chunked downstream.
- Missing fields are never invented: ``normalize`` lists them (technology.md, "index and flag").
"""

from __future__ import annotations

from collections.abc import Iterable
from html.parser import HTMLParser
from typing import Any

import httpx

from aether_core.domain.contract import Authority, Source, normalize
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


def map_page(page: dict[str, Any], base_url: str) -> SourceItem:
    """Map a Confluence REST content response (with body, version and ancestors) to the contract."""
    props = parse_page_properties(page["body"]["storage"]["value"])
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
    )
    return SourceItem(artifact=artifact, ancestor_titles=ancestors)


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

    def list_versions(self, scope: dict[str, str]) -> Iterable[ListedItem]:
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
