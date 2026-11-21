"""Azure Boards reader (REST API 7.1, read-only): one item per work item.

ingestion.md, "Azure DevOps"; platform.md, "Example: work items (Azure DevOps)":
- Custom reader on the REST API: ``llama-index-readers-azure-devops`` (0.3.0) reads repository files,
  not work items, so this adapter is written by hand, like the Miro one.
- Scope: a WIQL query by project, work item types and, optionally, area (the pipeline definition).
- Readable text (embedded): type, ID and title; state, iteration, tags and revision; description;
  acceptance criteria numbered ``AC-n`` (the n-th bullet at this revision, 11 - Testing, Traceability).
- Version: the work item revision (``System.Rev``). An ID the query no longer returns was deleted or
  moved out of the scope, and Sync removes its points.

Lifecycle state (decided by the space owner on 21 Nov 2026):
- ``Removed`` -> ``archived``;
- tag ``ready`` (Definition of Ready, 06 - Engineering Standards), or state Active, Resolved or Closed
  -> ``validated``;
- anything else -> ``draft``.

Keys are not invented (Key Registry, Convention): a Feature or a knowledge gap carries its own key as a
tag (``feature-...``, ``gap-q-...``); a User Story carries the key of its feature, so it has no own key
and is indexed with ``metadata_complete: false``. Key tags of other items become relationships.
Section: ``knowledge-gaps`` for items tagged ``knowledge-gap``, ``product`` for the rest.

Cross-source links: parent, child, predecessor, successor, related and hyperlink relations, and the
links in the description and acceptance criteria, are references; ``AzureBoardsUrls`` recognizes work
item URLs (``_workitems/edit/<id>`` and API ``workItems/<id>``) so pages that link to a work item relate
to it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from html import unescape
from typing import Any
from urllib.parse import unquote, urlsplit

import httpx

from aether_core.domain.contract import Authority, Relationship, Source, normalize
from aether_core.domain.links import LinkReference, SourceRef, address
from aether_core.domain.sync import ListedItem
from aether_core.ports import SourceItem

PROVIDER = "azure-boards"
API_VERSION = "7.1"
BATCH = 200  # maximum IDs per workitemsbatch request
DEFAULT_TYPES = ("Epic", "Feature", "User Story")

READY_TAG = "ready"
GAP_TAG = "knowledge-gap"
VALIDATED_STATES = frozenset({"active", "resolved", "closed"})
ARCHIVED_STATE = "removed"

TYPES = {"epic": "epic", "feature": "feature", "user story": "user-story", "bug": "bug", "task": "task"}
OWN_KEY_PREFIX = {"feature": "feature-", "knowledge-gap": "gap-"}
_KEY = re.compile(r"^[a-z]+(?:-[a-z0-9]+)+$")

RELATIONS = {
    "System.LinkTypes.Hierarchy-Reverse": "Parent",
    "System.LinkTypes.Hierarchy-Forward": "Child",
    "System.LinkTypes.Dependency-Reverse": "Predecessor",
    "System.LinkTypes.Dependency-Forward": "Successor",
    "System.LinkTypes.Related": "Related",
    "Hyperlink": "Hyperlink",
}

_TAG = re.compile(r"<[^>]+>")
_BREAK = re.compile(r"</p>|<br\s*/?>|</div>|</h[1-6]>", re.IGNORECASE)
_ITEM = re.compile(r"<li[^>]*>", re.IGNORECASE)
_ANCHOR = re.compile(r"<a\s[^>]*href=\"(https?://[^\"]+)\"[^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)
_URL = re.compile(r"https?://[^\s<>\"')\]]+")
_MD_LINK = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+)\)")
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_EDIT_PATH = re.compile(r"/_workitems/edit/(\d+)")
_API_PATH = re.compile(r"/_apis/wit/workitems/(\d+)", re.IGNORECASE)


def is_html(text: str) -> bool:
    return bool(re.search(r"<(p|div|li|ul|ol|br|span|strong|em|b|i|a)\b", text, re.IGNORECASE))


def plain_text(value: str | None) -> str:
    """A rich text field as text, one line per paragraph and one ``- `` line per list item.
    Azure Boards stores rich text as HTML, or as Markdown when the field is in Markdown mode."""
    if not value:
        return ""
    if not is_html(value):
        return "\n".join(line.rstrip() for line in unescape(value).splitlines()).strip()
    text = _ITEM.sub("\n- ", _BREAK.sub("\n", value))
    text = unescape(_TAG.sub("", text))
    lines = (" ".join(line.split()) for line in text.splitlines())
    return "\n".join(line for line in lines if line and line != "-")


def criteria(value: str | None) -> list[str]:
    """Acceptance criteria as numbered bullets: ``AC-n`` is the n-th bullet (11 - Testing). Lines that
    are not bullets (a closing note, for example) are kept as they are, without a number."""
    lines, n = [], 0
    for line in plain_text(value).splitlines():
        if not line.strip():
            continue
        if _BULLET.match(line):
            n += 1
            lines.append(f"AC-{n}: {_BULLET.sub('', line).strip()}")
        else:
            lines.append(line.strip())
    return lines


def tags_of(fields: dict[str, Any]) -> list[str]:
    return [t.strip() for t in str(fields.get("System.Tags") or "").split(";") if t.strip()]


def state_of(fields: dict[str, Any], tags: Sequence[str]) -> tuple[str, bool]:
    """(lifecycle state, archived at source) under the rule in the module docstring."""
    source_state = str(fields.get("System.State") or "").strip().lower()
    if source_state == ARCHIVED_STATE:
        return "archived", True
    if READY_TAG in {t.lower() for t in tags} or source_state in VALIDATED_STATES:
        return "validated", False
    return "draft", False


def kind_of(fields: dict[str, Any], tags: Sequence[str]) -> str:
    if GAP_TAG in {t.lower() for t in tags}:
        return "knowledge-gap"
    work_item_type = str(fields.get("System.WorkItemType") or "").strip().lower()
    return TYPES.get(work_item_type, work_item_type.replace(" ", "-") or "work-item")


def keys_of(kind: str, tags: Sequence[str]) -> tuple[str | None, tuple[Relationship, ...]]:
    """The item's own key (a tag with its type's prefix) and the other key tags as relationships."""
    key_tags = [t for t in tags if _KEY.match(t) and t != GAP_TAG]
    prefix = OWN_KEY_PREFIX.get(kind)
    own = next((t for t in key_tags if prefix and t.startswith(prefix)), None)
    related = tuple(Relationship("tagged_with", t) for t in key_tags if t != own)
    return own, related


def person(value: Any) -> str | None:
    if isinstance(value, dict):
        return str(value.get("displayName") or "") or None
    return str(value) if value else None


def heading(fields: dict[str, Any], work_item_id: object) -> str:
    """``User Story 72: Submit run``: the title of a work item in the index and in its text."""
    kind = fields.get("System.WorkItemType", "Work item")
    return f"{kind} {work_item_id}: {fields.get('System.Title', '')}".strip()


def render(item: dict[str, Any]) -> str:
    """Deterministic readable text of a work item."""
    fields = item.get("fields") or {}
    tags = tags_of(fields)
    lines = [
        heading(fields, item["id"]),
        "; ".join(
            part
            for part in (
                f"State: {fields.get('System.State')}" if fields.get("System.State") else "",
                f"Iteration: {fields.get('System.IterationPath')}"
                if fields.get("System.IterationPath")
                else "",
                f"Tags: {'; '.join(tags)}" if tags else "",
                f"Story points: {fields['Microsoft.VSTS.Scheduling.StoryPoints']:g}"
                if isinstance(fields.get("Microsoft.VSTS.Scheduling.StoryPoints"), (int, float))
                else "",
                f"Revision: {item.get('rev', fields.get('System.Rev', ''))}",
            )
            if part
        ),
    ]
    relations = [f"{name}: {wid}" for name, wid, _ in related_items(item)]
    if relations:
        lines.append("; ".join(relations))
    description = plain_text(fields.get("System.Description"))
    if description:
        lines += ["Description:", description]
    acceptance = criteria(fields.get("Microsoft.VSTS.Common.AcceptanceCriteria"))
    if acceptance:
        lines += ["Acceptance criteria:", *acceptance]
    return "\n".join(lines)


def related_items(item: dict[str, Any]) -> list[tuple[str, str, str]]:
    """(relation name, work item ID or URL, URL) for the relations this reader keeps, in a fixed order."""
    found = []
    for relation in item.get("relations") or []:
        name = RELATIONS.get(str(relation.get("rel")))
        url = str(relation.get("url") or "")
        if name is None or not url:
            continue
        match = _API_PATH.search(urlsplit(url).path)
        found.append((name, match.group(1) if match else url, url))
    order = list(RELATIONS.values())
    return sorted(found, key=lambda r: (order.index(r[0]), r[1]))


def references(item: dict[str, Any]) -> tuple[LinkReference, ...]:
    fields = item.get("fields") or {}
    found = [
        LinkReference(url, "link", (), name, f"{name} of work item {item['id']}")
        for name, _, url in related_items(item)
    ]
    for field, label in (
        ("System.Description", "Description"),
        ("Microsoft.VSTS.Common.AcceptanceCriteria", "Acceptance criteria"),
    ):
        value = str(fields.get(field) or "")
        anchored = {url for url, _ in _ANCHOR.findall(value)} | {url for _, url in _MD_LINK.findall(value)}
        for url, text in _ANCHOR.findall(value):
            found.append(LinkReference(unescape(url), "link", (label,), plain_text(text), label))
        for text, url in _MD_LINK.findall(value):
            found.append(LinkReference(url, "link", (label,), text, label))
        for url in _URL.findall(_ANCHOR.sub("", value)):
            if url not in anchored:
                found.append(LinkReference(url.rstrip(".,;"), "link", (label,), "", label))
    return tuple(found)


class AzureBoardsUrls:
    """``parse_url`` for Azure Boards: work item URLs only (``/_workitems/edit/<id>`` and the API URL
    ``/_apis/wit/workItems/<id>``), on dev.azure.com or <org>.visualstudio.com."""

    provider = PROVIDER
    noun = "work item"

    def parse_url(self, url: str) -> SourceRef | None:
        parts = urlsplit(url.strip().rstrip(".,;)"))
        host = parts.netloc.lower()
        if host != "dev.azure.com" and not host.endswith(".visualstudio.com"):
            return None
        path = unquote(parts.path)
        match = _EDIT_PATH.search(path) or _API_PATH.search(path)
        return SourceRef(PROVIDER, match.group(1)) if match else None


def map_work_item(item: dict[str, Any], web_base: str, section: str | None = None) -> SourceItem:
    fields = item.get("fields") or {}
    tags = tags_of(fields)
    kind = kind_of(fields, tags)
    state, archived = state_of(fields, tags)
    key, relationships = keys_of(kind, tags)
    work_item_id = str(item["id"])
    artifact = normalize(
        artifact_id=f"{PROVIDER}:{work_item_id}",
        title=heading(fields, work_item_id),
        source=Source(
            PROVIDER,
            work_item_id,
            f"{web_base}/_workitems/edit/{work_item_id}",
            str(item.get("rev", fields.get("System.Rev", ""))),
        ),
        authority=Authority("work_item", owner=person(fields.get("System.AssignedTo"))),
        text=render(item),
        key=key,
        artifact_type=kind,
        section=section or ("knowledge-gaps" if kind == "knowledge-gap" else "product"),
        state=state,
        archived_at_source=archived,
        relationships=relationships,
    )
    ancestors = tuple(p for p in str(fields.get("System.AreaPath") or "").split("\\") if p)
    return SourceItem(
        artifact=artifact,
        ancestor_titles=ancestors,
        references=references(item),
        addresses=(address(PROVIDER, work_item_id),),
    )


def wiql(scope: dict[str, Any]) -> str:
    project = str(scope["project"]).replace("'", "''")
    types = ", ".join(
        "'" + str(t).replace("'", "''") + "'" for t in scope.get("work_item_types") or DEFAULT_TYPES
    )
    query = (
        "SELECT [System.Id] FROM WorkItems "
        f"WHERE [System.TeamProject] = '{project}' AND [System.WorkItemType] IN ({types})"
    )
    if scope.get("area_path"):
        area = str(scope["area_path"]).replace("'", "''")
        query += f" AND [System.AreaPath] UNDER '{area}'"
    return query + " ORDER BY [System.Id]"


class AzureBoardsAuthError(RuntimeError):
    """Azure DevOps rejected the token: it answers a redirect to the sign-in page (302) or a
    non-authoritative page (203) instead of 401 when the token is invalid, expired or lacks the scope."""


def checked(response: httpx.Response) -> httpx.Response:
    if response.status_code in (203, 302) or "_signin" in response.headers.get("location", ""):
        raise AzureBoardsAuthError(
            "Azure DevOps rejected ADO_READ_PAT (redirect to sign-in): use a personal access token of "
            "this organization with the Work Items (Read) scope, not expired"
        )
    response.raise_for_status()
    return response


class AzureBoardsReader:
    """SourceReader for Azure Boards. The token comes from the environment: a personal access token
    with the Work Items (Read) scope only, held by ``sync`` (platform.md, "Account")."""

    def __init__(self, token: str, scope: dict[str, Any], client: httpx.Client | None = None) -> None:
        organization = str(scope.get("organization_url") or "https://dev.azure.com/devsinsight").rstrip("/")
        self._project = str(scope["project"])
        self._api = f"{organization}/{self._project}/_apis/wit"
        self._web = f"{organization}/{self._project}"
        self._section = scope.get("section")
        self._client = client or httpx.Client(auth=("", token), timeout=30.0)

    def _ids(self, scope: dict[str, Any]) -> list[int]:
        response = self._client.post(
            f"{self._api}/wiql", params={"api-version": API_VERSION}, json={"query": wiql(scope)}
        )
        return [int(w["id"]) for w in checked(response).json().get("workItems") or []]

    def _batches(self, ids: Sequence[int], fields: Sequence[str]) -> Iterator[dict[str, Any]]:
        for start in range(0, len(ids), BATCH):
            response = self._client.post(
                f"{self._api}/workitemsbatch",
                params={"api-version": API_VERSION},
                json={"ids": list(ids[start : start + BATCH]), "fields": list(fields)},
            )
            yield from checked(response).json().get("value") or []

    def read(self, item_id: str) -> SourceItem:
        work_item_id = item_id.rpartition(":")[2]
        response = self._client.get(
            f"{self._api}/workitems/{work_item_id}",
            params={"$expand": "relations", "api-version": API_VERSION},
        )
        return map_work_item(checked(response).json(), self._web, self._section)

    def list_versions(self, scope: dict[str, Any]) -> Iterable[ListedItem]:
        fields = ("System.Id", "System.Rev", "System.Title", "System.WorkItemType", "System.AreaPath")
        for item in self._batches(self._ids(scope), fields):
            f = item.get("fields") or {}
            yield ListedItem(
                str(item["id"]),
                str(f.get("System.Rev", item.get("rev", ""))),
                heading(f, item["id"]),
                tuple(p for p in str(f.get("System.AreaPath") or "").split("\\") if p),
            )
