"""Miro reader (REST API v2, read-only): one item per frame, or the whole board when it has no frames.

ingestion.md, "Miro (REST API v2)":
- Readable text (embedded): frame title, nodes with text and type, relationships as "A → B: caption".
- Provenance: board ID, frame ID, link and modification dates.

LlamaIndex has no Miro reader, so this adapter is written by hand (platform.md, outbound adapters).

Version of a frame: Miro has no "modified since" filter, and editing a node does not change its frame's
``modifiedAt``. The version is therefore the latest ``modifiedAt`` among the frame, its nodes and its
connectors, plus a short hash of their IDs and dates, so a deleted node also yields a new version.

Item IDs are ``<board_id>:<frame_id>`` (``<board_id>:board`` for a board without frames). Key, owner and
lifecycle state are not invented: Miro has no Page Properties, so ``normalize`` lists what is missing.
A frame title written as a key (``feature-scoring-and-chains``) is taken as the frame's key.

Cross-source links: links in item text, embed items and image titles are references with the frame as
position; an image with no link to its origin is reported, never guessed. ``MiroUrls`` recognizes board
URLs only and normalizes the board ID (``%3D`` and ``=`` are the same board); ``moveToWidget`` names an
item. A frame is addressed by the board, the frame and every node in it, so a link to a node resolves
to its frame.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from html import unescape
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

import httpx

from aether_core.domain.contract import Authority, Source, normalize
from aether_core.domain.links import LinkReference, SourceRef, address
from aether_core.domain.sync import ListedItem
from aether_core.ports import SourceItem

PROVIDER = "miro"
API = "https://api.miro.com/v2"
WHOLE_BOARD = "board"
_KEY = re.compile(r"^[a-z]+(?:-[a-z0-9]+)+$")
_TAG = re.compile(r"<[^>]+>")
_BREAK = re.compile(r"</p>|<br\s*/?>|</li>", re.IGNORECASE)
_ANCHOR = re.compile(r"<a\s[^>]*href=\"(https?://[^\"]+)\"[^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)
_URL = re.compile(r"https?://[^\s<>\"']+")
_BOARD_PATH = re.compile(r"^/app/(?:board|live-embed)/([^/]+)")
_ITEM_PARAMS = ("moveToWidget", "focusWidget")


def plain_text(html: str | None) -> str:
    """Miro stores rich text as HTML: keep the text, one line per paragraph."""
    if not html:
        return ""
    text = unescape(_TAG.sub("", _BREAK.sub("\n", html)))
    return " / ".join(" ".join(line.split()) for line in text.splitlines() if line.strip())


def item_text(item: dict[str, Any]) -> str:
    data = item.get("data") or {}
    parts = [plain_text(data.get(field)) for field in ("title", "content", "description")]
    return " - ".join(part for part in parts if part)


@dataclass(frozen=True)
class Board:
    board_id: str
    name: str
    section: str | None
    items: tuple[dict[str, Any], ...]
    connectors: tuple[dict[str, Any], ...]

    def frames(self) -> list[dict[str, Any]]:
        return [i for i in self.items if i.get("type") == "frame"]

    def frame_of(self, item: dict[str, Any]) -> str | None:
        """The frame an item sits in, walking up nested groups; None for items outside every frame."""
        by_id = {str(i["id"]): i for i in self.items}
        seen: set[str] = set()
        current: dict[str, Any] | None = item
        while current is not None:
            parent = (current.get("parent") or {}).get("id")
            if parent is None or str(parent) in seen:
                return None
            seen.add(str(parent))
            current = by_id.get(str(parent))
            if current is not None and current.get("type") == "frame":
                return str(current["id"])
        return None

    def members(
        self, frame_id: str
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
        """The frame (None for the whole board), its nodes and the connectors whose ends are both in it."""
        if frame_id == WHOLE_BOARD:
            frame, nodes = None, [i for i in self.items if i.get("type") != "frame"]
        else:
            frame = next((i for i in self.items if str(i["id"]) == frame_id), None)
            if frame is None or frame.get("type") != "frame":
                raise KeyError(f"frame {frame_id} not found on board {self.board_id}")
            nodes = [i for i in self.items if i.get("type") != "frame" and self.frame_of(i) == frame_id]
        ids = {str(n["id"]) for n in nodes}
        connectors = [
            c
            for c in self.connectors
            if str((c.get("startItem") or {}).get("id")) in ids
            and str((c.get("endItem") or {}).get("id")) in ids
        ]
        return frame, nodes, connectors

    def item_ids(self) -> list[str]:
        frames = self.frames()
        return (
            [f"{self.board_id}:{f['id']}" for f in frames] if frames else [f"{self.board_id}:{WHOLE_BOARD}"]
        )

    def version(self, frame_id: str) -> str:
        frame, nodes, connectors = self.members(frame_id)
        parts = sorted(
            (str(x["id"]), str(x.get("modifiedAt") or ""))
            for x in ([frame] if frame else []) + nodes + connectors
        )
        latest = max((modified for _, modified in parts if modified), default="")
        digest = hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()[:8]
        return f"{latest}#{digest}"

    def title(self, frame_id: str) -> str:
        frame, _, _ = self.members(frame_id)
        frame_title = plain_text((frame or {}).get("data", {}).get("title")) if frame else ""
        return f"{self.name} - {frame_title}" if frame_title else self.name


def reading_order(board: Board) -> Callable[[dict[str, Any]], tuple[float, float, str]]:
    """Sort key top to bottom, then left to right. Miro positions are relative to the parent, so the
    positions up the parent chain are added; the item ID breaks ties deterministically."""
    by_id = {str(i["id"]): i for i in board.items}

    def key(item: dict[str, Any]) -> tuple[float, float, str]:
        y = x = 0.0
        current: dict[str, Any] | None = item
        seen: set[str] = set()
        while current is not None and str(current["id"]) not in seen:
            seen.add(str(current["id"]))
            position = current.get("position") or {}
            y, x = y + float(position.get("y") or 0.0), x + float(position.get("x") or 0.0)
            current = by_id.get(str((current.get("parent") or {}).get("id")))
        return y, x, str(item["id"])

    return key


def render(board: Board, frame_id: str) -> str:
    """Deterministic readable text of a frame: title, nodes in reading order, relationships."""
    frame, nodes, connectors = board.members(frame_id)
    names = {str(n["id"]): item_text(n) or n.get("type", "item") for n in nodes}
    lines = [f"Board: {board.name}"]
    if frame is not None:
        lines.append(f"Frame: {plain_text(frame.get('data', {}).get('title')) or frame['id']}")
    for node in sorted(nodes, key=reading_order(board)):
        text = item_text(node)
        if text:
            lines.append(f"- {str(node.get('type', 'item')).replace('_', ' ')}: {text}")
    relations = []
    for connector in connectors:
        start = names.get(str(connector["startItem"]["id"]), "?")
        end = names.get(str(connector["endItem"]["id"]), "?")
        captions = " / ".join(plain_text(c.get("content")) for c in connector.get("captions") or [])
        relations.append(f"- {start} → {end}" + (f": {captions}" if captions else ""))
    if relations:
        lines.append("Relationships:")
        lines.extend(sorted(relations))
    return "\n".join(lines)


def references(board: Board, frame_id: str) -> tuple[LinkReference, ...]:
    """Links in item text, embed items and image titles of a frame, in reading order."""
    _, nodes, _ = board.members(frame_id)
    found: list[LinkReference] = []
    for node in sorted(nodes, key=reading_order(board)):
        data = node.get("data") or {}
        context = item_text(node)
        kind = node.get("type")
        if kind == "embed" and str(data.get("url") or "").startswith("http"):
            found.append(LinkReference(str(data["url"]), "embed", (), "", context))
        elif kind == "image":
            title = plain_text(data.get("title"))
            urls = _URL.findall(title)
            found.extend(LinkReference(url, "image", (), title, context) for url in urls)
            if not urls:  # a display copy that does not name its origin: reported, never guessed
                found.append(LinkReference("", "image", (), title or str(node["id"]), context))
        for field in ("content", "title", "description"):
            for url, text in _ANCHOR.findall(str(data.get(field) or "")):
                found.append(LinkReference(unescape(url), "link", (), plain_text(text), context))
    return tuple(found)


def addresses(board: Board, frame_id: str) -> tuple[str, ...]:
    """The frame first (its own address), then the board and every node in the frame."""
    _, nodes, _ = board.members(frame_id)
    own = (address(PROVIDER, board.board_id, frame_id),) if frame_id != WHOLE_BOARD else ()
    nodes_addresses = tuple(address(PROVIDER, board.board_id, str(n["id"])) for n in nodes)
    return (*own, address(PROVIDER, board.board_id), *nodes_addresses)


class MiroUrls:
    """``parse_url`` for Miro: board URLs only; the item comes from ``moveToWidget`` or ``focusWidget``."""

    provider = PROVIDER
    noun = "board"

    def parse_url(self, url: str) -> SourceRef | None:
        parts = urlsplit(url.strip().rstrip(".,;)"))
        if parts.netloc.lower() not in ("miro.com", "www.miro.com"):
            return None
        match = _BOARD_PATH.match(parts.path)
        if match is None:
            return None
        board_id = unquote(match.group(1)).strip()
        query = parse_qs(parts.query)
        item = next((query[p][0] for p in _ITEM_PARAMS if query.get(p)), None)
        return SourceRef(PROVIDER, board_id, item.strip() if item else None) if board_id else None


def map_frame(board: Board, frame_id: str) -> SourceItem:
    frame, _, _ = board.members(frame_id)
    frame_title = plain_text(frame.get("data", {}).get("title")) if frame else ""
    url = f"https://miro.com/app/board/{board.board_id}/"
    if frame is not None:
        url += f"?moveToWidget={frame_id}"
    artifact = normalize(
        artifact_id=f"{PROVIDER}:{board.board_id}:{frame_id}",
        title=board.title(frame_id),
        source=Source(PROVIDER, f"{board.board_id}:{frame_id}", url, board.version(frame_id)),
        authority=Authority("editable_visual"),
        text=render(board, frame_id),
        key=frame_title if _KEY.match(frame_title) else None,
        artifact_type="diagram",
        section=board.section,
        state=None,
    )
    return SourceItem(
        artifact=artifact,
        ancestor_titles=(board.name,),
        references=references(board, frame_id),
        addresses=addresses(board, frame_id),
    )


def boards_in(scope: dict[str, Any]) -> list[dict[str, Any]]:
    """The boards a pipeline definition lists (scope.boards: [{id, section}])."""
    return [dict(b) for b in scope.get("boards") or []]


class MiroReader:
    """SourceReader for Miro. The token comes from the environment (a read-only ``boards:read`` token)."""

    def __init__(self, token: str, scope: dict[str, Any], client: httpx.Client | None = None) -> None:
        self._sections = {str(b["id"]): b.get("section") for b in boards_in(scope)}
        self._client = client or httpx.Client(headers={"Authorization": f"Bearer {token}"}, timeout=30.0)

    def _pages(self, path: str) -> Iterator[dict[str, Any]]:
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"limit": 50}
            if cursor:
                params["cursor"] = cursor
            response = self._client.get(f"{API}{path}", params=params)
            response.raise_for_status()
            data = response.json()
            yield from data.get("data") or []
            cursor = data.get("cursor")
            if not cursor:
                return

    def board(self, board_id: str) -> Board:
        response = self._client.get(f"{API}/boards/{board_id}")
        response.raise_for_status()
        return Board(
            board_id=board_id,
            name=str(response.json().get("name") or board_id),
            section=self._sections.get(board_id),
            items=tuple(self._pages(f"/boards/{board_id}/items")),
            connectors=tuple(self._pages(f"/boards/{board_id}/connectors")),
        )

    def read(self, item_id: str) -> SourceItem:
        board_id, _, frame_id = item_id.rpartition(":")
        return map_frame(self.board(board_id), frame_id)

    def list_versions(self, scope: dict[str, Any]) -> Iterable[ListedItem]:
        for entry in boards_in(scope):
            board = self.board(str(entry["id"]))
            for item_id in board.item_ids():
                frame_id = item_id.rpartition(":")[2]
                yield ListedItem(item_id, board.version(frame_id), board.title(frame_id), (board.name,))
