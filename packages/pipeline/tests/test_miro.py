"""Miro reader against a fake REST API v2, and the frame chunker."""

from __future__ import annotations

import copy
from typing import Any

import httpx

from aether_core.domain.contract import Authority, Source, normalize
from aether_pipeline.adapters.frame_chunker import FrameChunker
from aether_pipeline.adapters.miro import MiroReader, plain_text

SCOPE = {"boards": [{"id": "b1=", "section": "product"}]}


def item(
    item_id: str, kind: str, parent: str | None, modified: str, y: float = 0, **data: str
) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": kind,
        "data": data,
        "position": {"x": 0, "y": y},
        "parent": {"id": parent} if parent else None,
        "modifiedAt": modified,
    }


def board_items() -> list[dict[str, Any]]:
    return [
        item("f1", "frame", None, "2026-10-01T00:00:00Z", title="Ship states"),
        item("s1", "sticky_note", "f1", "2026-10-02T00:00:00Z", y=10, content="<p>Alive</p>"),
        item("s2", "shape", "f1", "2026-10-03T00:00:00Z", y=20, content="<p>Dead &amp; out</p>"),
        item("g1", "group", "f1", "2026-10-01T00:00:00Z", y=30),
        item("t1", "text", "g1", "2026-10-01T00:00:00Z", y=5, content="<p>Respawn</p><p>after 2 s</p>"),
        item("o1", "sticky_note", None, "2026-10-09T00:00:00Z", content="Outside every frame"),
        item("f2", "frame", None, "2026-10-01T00:00:00Z", title="feature-scoring-and-chains"),
    ]


class FakeMiro:
    def __init__(self) -> None:
        self.items = board_items()
        self.connectors = [
            {
                "id": "c1",
                "startItem": {"id": "s1"},
                "endItem": {"id": "s2"},
                "captions": [{"content": "hit"}],
                "modifiedAt": "2026-10-01T00:00:00Z",
            },
            {
                "id": "c2",
                "startItem": {"id": "s1"},
                "endItem": {"id": "o1"},
                "modifiedAt": "2026-10-01T00:00:00Z",
            },
        ]
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path.endswith("/items") or path.endswith("/connectors"):
            data = self.items if path.endswith("/items") else self.connectors
            page = int(request.url.params.get("cursor") or 0)
            body: dict[str, Any] = {"data": data[page * 4 : page * 4 + 4]}  # 4 per page: forces paging
            if page * 4 + 4 < len(data):
                body["cursor"] = str(page + 1)
            return httpx.Response(200, json=body)
        return httpx.Response(200, json={"id": "b1=", "name": "Aether - Player State Machine"})

    def reader(self) -> MiroReader:
        return MiroReader("unused", SCOPE, client=httpx.Client(transport=httpx.MockTransport(self.handler)))


def test_lists_one_item_per_frame_with_versions_and_no_bodies() -> None:
    listed = list(FakeMiro().reader().list_versions(SCOPE))
    assert [i.item_id for i in listed] == ["b1=:f1", "b1=:f2"]
    assert listed[0].title == "Aether - Player State Machine - Ship states"
    assert listed[0].ancestor_titles == ("Aether - Player State Machine",)
    assert listed[0].version.startswith("2026-10-03T00:00:00Z#")  # latest change inside the frame


def test_reads_a_frame_as_readable_text_with_provenance() -> None:
    artifact = FakeMiro().reader().read("b1=:f1").artifact
    assert artifact.text.splitlines() == [
        "Board: Aether - Player State Machine",
        "Frame: Ship states",
        "- sticky note: Alive",
        "- shape: Dead & out",
        "- text: Respawn / after 2 s",  # nested in a group, still in the frame
        "Relationships:",
        "- Alive → Dead & out: hit",  # c2 leaves the frame, so it is not part of it
    ]
    assert artifact.artifact_id == "miro:b1=:f1"
    assert artifact.source.canonical_url == "https://miro.com/app/board/b1=/?moveToWidget=f1"
    assert artifact.authority.artifact_class == "editable_visual"
    assert (artifact.section, artifact.artifact_type, artifact.lifecycle.state) == (
        "product",
        "diagram",
        "draft",
    )
    assert artifact.key is None and set(artifact.missing_fields) == {"key", "lifecycle.state"}


def test_a_frame_titled_with_a_key_carries_it() -> None:
    artifact = FakeMiro().reader().read("b1=:f2").artifact
    assert artifact.key == "feature-scoring-and-chains"


def test_version_changes_when_a_node_is_edited_or_deleted() -> None:
    fake = FakeMiro()
    original = fake.reader().read("b1=:f1").artifact.source.version
    fake.items[1] = {**fake.items[1], "modifiedAt": "2026-11-01T00:00:00Z"}
    edited = fake.reader().read("b1=:f1").artifact.source.version
    fake.items = [i for i in fake.items if i["id"] != "t1"]
    deleted = fake.reader().read("b1=:f1").artifact.source.version
    assert len({original, edited, deleted}) == 3
    fake.items = copy.deepcopy(board_items())
    fake.items[5] = {**fake.items[5], "modifiedAt": "2026-12-01T00:00:00Z"}  # outside the frame
    assert fake.reader().read("b1=:f1").artifact.source.version == original


def test_a_board_without_frames_is_one_item() -> None:
    fake = FakeMiro()
    fake.items = [i for i in board_items() if i["type"] != "frame" and i["parent"] is None]
    (listed,) = fake.reader().list_versions(SCOPE)
    assert listed.item_id == "b1=:board"
    artifact = fake.reader().read("b1=:board").artifact
    assert "Outside every frame" in artifact.text
    assert artifact.source.canonical_url == "https://miro.com/app/board/b1=/"


def test_plain_text_keeps_one_line_per_paragraph() -> None:
    assert plain_text("<p>A&nbsp;b</p><p></p><ul><li>c</li><li>d</li></ul>") == "A b / c / d"
    assert plain_text(None) == ""


def frame(text: str) -> Any:
    return normalize(
        artifact_id="miro:b:f",
        title="Board - Frame",
        source=Source("miro", "b:f", "https://miro.com/app/board/b/", "v"),
        authority=Authority("editable_visual"),
        text=text,
        key=None,
        artifact_type="diagram",
        section="product",
        state=None,
    )


def words(text: str) -> int:
    return len(text.split())


def test_a_frame_is_one_chunk_when_it_fits() -> None:
    (chunk,) = FrameChunker(words, max_tokens=50).chunk(frame("Board: B\nFrame: F\n- shape: A"))
    assert chunk.text == "Board: B\nFrame: F\n- shape: A" and chunk.heading_path == ()


def test_a_long_frame_is_split_by_lines_never_inside_a_line() -> None:
    lines = [f"- sticky note: idea number {n}" for n in range(6)]  # 6 words each
    chunks = FrameChunker(words, max_tokens=17).chunk(frame("\n".join(lines)))  # budget 17 - 3 - 2 = 12
    assert [c.text.splitlines() for c in chunks] == [lines[0:2], lines[2:4], lines[4:6]]
    assert [c.position for c in chunks] == [0, 1, 2]
    assert len({c.point_id for c in chunks}) == 3
