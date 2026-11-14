"""Confluence Cloud writer for work records (REST API with an HTTP client; LlamaIndex has no writers).

- Markdown is converted to storage format (XHTML) with the ``markdown`` library.
- The common record goes into a Page Properties macro (``details``) at the end of the page, in the
  same shape the reader parses (Key, Type, State, Owner first).
- Updates use the page version for optimistic concurrency: Confluence rejects a stale version.
- Pages are found by their exact title in the space, which is unique per story.
"""

from __future__ import annotations

import re
from html import escape
from typing import Any

import httpx
import markdown

from aether_core.ports import PublishedPage
from aether_pipeline.adapters.confluence import parse_page_properties

_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_VALIDATION_HEADING = re.compile(r"(<h2[^>]*>\s*Validation\s*</h2>)", re.IGNORECASE)
_DETAILS = re.compile(r'<ac:structured-macro ac:name="details".*?</ac:structured-macro>\s*$', re.DOTALL)


def to_storage(body_markdown: str) -> str:
    """Markdown (without front matter) to Confluence storage XHTML."""
    text = _COMMENT.sub("", body_markdown)
    return markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"], output_format="xhtml")


def properties_macro(properties: dict[str, str]) -> str:
    rows = "".join(
        f"<tr><th><p>{escape(name)}</p></th><td><p>{escape(value)}</p></td></tr>"
        for name, value in properties.items()
    )
    return (
        '<ac:structured-macro ac:name="details" ac:schema-version="1"><ac:rich-text-body>'
        f"<table><tbody>{rows}</tbody></table></ac:rich-text-body></ac:structured-macro>"
    )


class ConfluenceWikiPublisher:
    def __init__(
        self, base_url: str, email: str, api_token: str, space_key: str, client: httpx.Client | None = None
    ) -> None:
        self._base = base_url.rstrip("/")
        self._space = space_key
        self._client = client or httpx.Client(auth=(email, api_token), timeout=30.0)

    def find_page(self, title: str) -> PublishedPage | None:
        page = self._find(title, expand="version,body.storage")
        return self._published(page) if page else None

    def ensure_parent(self, parent_id: str, title: str) -> str:
        existing = self._find(title)
        if existing:
            return str(existing["id"])
        intro = "<p>Work records of this feature, one page per user story (AIEF ADR-006).</p>"
        return str(self._create(parent_id, title, intro)["id"])

    def create_or_update(
        self,
        parent_id: str,
        title: str,
        body_markdown: str,
        properties: dict[str, str],
        expected_version: int | None,
    ) -> PublishedPage:
        storage = to_storage(body_markdown) + properties_macro(properties)
        if expected_version is None:
            page = self._create(parent_id, title, storage)
        else:
            page = self._update(self._find_id(title), title, storage, expected_version)
        return self._published(page, storage)

    def set_properties(self, page: PublishedPage, properties: dict[str, str], note: str) -> PublishedPage:
        current = self._get(page.page_id)
        storage = _DETAILS.sub("", current["body"]["storage"]["value"]).rstrip()
        paragraph = f"<p>{escape(note)}</p>"
        if _VALIDATION_HEADING.search(storage):
            storage = _VALIDATION_HEADING.sub(lambda m: m.group(1) + paragraph, storage, count=1)
        else:
            storage += f"<h2>Validation</h2>{paragraph}"
        storage += properties_macro(properties)
        updated = self._update(page.page_id, page.title, storage, int(current["version"]["number"]))
        return self._published(updated, storage)

    # REST calls

    def _find(self, title: str, expand: str = "version") -> dict[str, Any] | None:
        response = self._client.get(
            f"{self._base}/rest/api/content",
            params={"spaceKey": self._space, "title": title, "type": "page", "expand": expand},
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        return results[0] if results else None

    def _find_id(self, title: str) -> str:
        page = self._find(title)
        if page is None:
            raise RuntimeError(f"page {title!r} disappeared before it could be updated")
        return str(page["id"])

    def _get(self, page_id: str) -> dict[str, Any]:
        response = self._client.get(
            f"{self._base}/rest/api/content/{page_id}", params={"expand": "version,body.storage"}
        )
        response.raise_for_status()
        return response.json()

    def _create(self, parent_id: str, title: str, storage: str) -> dict[str, Any]:
        response = self._client.post(
            f"{self._base}/rest/api/content",
            json={
                "type": "page",
                "title": title,
                "space": {"key": self._space},
                "ancestors": [{"id": parent_id}],
                "body": {"storage": {"value": storage, "representation": "storage"}},
            },
        )
        response.raise_for_status()
        return response.json()

    def _update(self, page_id: str, title: str, storage: str, expected_version: int) -> dict[str, Any]:
        response = self._client.put(
            f"{self._base}/rest/api/content/{page_id}",
            json={
                "id": page_id,
                "type": "page",
                "title": title,
                "version": {"number": expected_version + 1, "message": "Work record published by publisher"},
                "body": {"storage": {"value": storage, "representation": "storage"}},
            },
        )
        if response.status_code == 409:
            raise RuntimeError(f"page {page_id} changed since version {expected_version}; publish again")
        response.raise_for_status()
        return response.json()

    def _published(self, page: dict[str, Any], storage: str | None = None) -> PublishedPage:
        body = storage if storage is not None else page.get("body", {}).get("storage", {}).get("value", "")
        return PublishedPage(
            page_id=str(page["id"]),
            version=int(page["version"]["number"]),
            title=page["title"],
            url=self._base + page.get("_links", {}).get("webui", f"/pages/{page['id']}"),
            properties=parse_page_properties(body),
        )
