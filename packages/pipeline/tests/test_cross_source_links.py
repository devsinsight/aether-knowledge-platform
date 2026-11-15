"""Cross-source links (platform.md, "Increment: Cross-source links"), with the Ships page as the case.

The Ships page is a recorded response of the AETHER space (page 9076737, version 6): a whole-board Miro
embed under "Prototypes and Concept Art" and a table of Scenario assets under "Concept Art". The Miro
board is a fake REST API v2; the wiki page and the board share one in-memory projection.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
from test_run_pipeline import CountingEmbedder, MemoryRuns, MemoryState, MemoryStore

from aether_core.domain.contract import Chunk, NormalizedKnowledgeArtifact
from aether_core.domain.pipeline import PipelineDefinition, PipelineRun
from aether_core.ports import SourceItem
from aether_pipeline.adapters.confluence import ConfluenceUrls, extract_references, map_page
from aether_pipeline.adapters.frame_chunker import FrameChunker
from aether_pipeline.adapters.miro import MiroReader, MiroUrls
from aether_pipeline.application.run_pipeline import RunPipeline

BASE = "https://devsinsight.atlassian.net/wiki"
SHIPS = json.loads((Path(__file__).parent / "fixtures" / "confluence_ships_page.json").read_text("utf-8"))
BOARD = "uXjVEUtjxFw="
BOARD_URL = f"https://miro.com/app/board/{BOARD}/"
WIKI = PipelineDefinition("confluence-aether", "confluence", {"space": "AETHER"}, "aether")
BOARDS = PipelineDefinition(
    "miro-aether",
    "miro",
    {"boards": [{"id": BOARD, "section": "industry-extensions"}]},
    "aether",
    exclusions=(),
)
PARSERS = [ConfluenceUrls(BASE), MiroUrls()]


def ships(**changes: Any) -> dict[str, Any]:
    page = copy.deepcopy(SHIPS)
    if "storage" in changes:  # the export view follows the storage body in these small pages
        page["body"]["storage"]["value"] = page["body"]["export_view"]["value"] = changes["storage"]
    page["id"] = changes.get("page_id", page["id"])
    return page


# --- Readers: references and URLs ----------------------------------------------------------------


def test_the_ships_embed_and_table_links_are_read_with_their_context() -> None:
    """Criteria 1 and 2 at the reader: an embed with its heading path, a table row with its headers."""
    embed, *assets = extract_references(SHIPS["body"]["storage"]["value"])
    assert (embed.kind, embed.url, embed.heading_path) == (
        "embed",
        BOARD_URL,
        ("Ships", "Prototypes and Concept Art"),
    )
    assert embed.context.startswith("The Miro board holds the prototypes")
    assert [a.heading_path for a in assets] == [("Ships", "Concept Art")] * 3
    assert assets[0].text == "asset_2SLuzWyKg8Y19RBPSdoD9zyc"
    assert assets[0].context.startswith(
        "Ship: AX-1 Kestrel; Scenario asset ID: asset_2SLuzWyKg8Y19RBPSdoD9zyc;"
    )
    assert assets[0].context.endswith("Status: Approved")
    assert "&teamId=" in assets[0].url  # the entity is decoded, the URL is kept as written


def test_page_properties_and_code_are_not_references() -> None:
    storage = (
        '<h1>A</h1><ac:structured-macro ac:name="details"><ac:rich-text-body><table><tbody><tr><th><p>Key</p>'
        '</th><td><p><a href="https://miro.com/app/board/x=/">x</a></p></td></tr></tbody></table>'
        '</ac:rich-text-body></ac:structured-macro><ac:structured-macro ac:name="code"><ac:plain-text-body>'
        "<![CDATA[https://miro.com/app/board/y=/]]></ac:plain-text-body></ac:structured-macro>"
        '<ul><li><p>See <a href="https://dev.azure.com/devsinsight/Aether/_workitems/edit/59">#59</a></p></li></ul>'
        '<ac:image ac:title="Kestrel"><ri:attachment ri:filename="kestrel.png" /></ac:image>'
    )
    link, image = extract_references(storage)
    assert (link.kind, link.text, link.context, link.heading_path) == ("link", "#59", "See #59", ("A",))
    assert (image.kind, image.url, image.text) == ("image", "", "Kestrel")  # no link to its origin


def test_each_adapter_recognizes_only_its_own_urls_and_normalizes_ids() -> None:
    """Criterion 6 (second half): the same board written with ``=`` and ``%3D`` is one ID."""
    miro, wiki = MiroUrls(), ConfluenceUrls(BASE)
    plain = miro.parse_url(BOARD_URL)
    encoded = miro.parse_url("https://miro.com/app/board/uXjVEUtjxFw%3D/?share_link_id=1")
    assert plain == encoded and plain is not None and plain.address == f"miro:{BOARD}"
    item = miro.parse_url(f"{BOARD_URL}?moveToWidget=3458764612345678901&cot=14")
    assert item is not None and item.item_address == f"miro:{BOARD}:3458764612345678901"
    assert miro.parse_url(f"{BASE}/spaces/AETHER/pages/9076737/Ships") is None
    page = wiki.parse_url(f"{BASE}/spaces/AETHER/pages/9076737/Ships")
    assert page is not None and page.address == "confluence:9076737"
    viewpage = wiki.parse_url(f"{BASE}/pages/viewpage.action?pageId=9076737")
    assert viewpage == page
    assert wiki.parse_url(BOARD_URL) is None
    assert wiki.parse_url("https://app.scenario.com/assets?openAssetId=asset_X") is None


# --- Fakes: a section chunker for pages and a Miro board -----------------------------------------


class SectionChunker:
    """One chunk per heading section of the export view, with the heading path (like Docling's)."""

    def chunk(self, artifact: NormalizedKnowledgeArtifact) -> Sequence[Chunk]:
        chunks: list[Chunk] = []
        path: list[tuple[int, str]] = []
        for match in re.finditer(r"<h([1-6])[^>]*>(.*?)</h\1>(.*?)(?=<h[1-6]|$)", artifact.text, re.S):
            level, title, body = int(match.group(1)), _text(match.group(2)), match.group(3)
            path = [p for p in path if p[0] < level] + [(level, title)]
            body = re.sub(r'<a [^>]*href="([^"]+)"[^>]*>(.*?)</a>', r"[\2](\1)", body, flags=re.S)
            text = _text(body)
            if text:
                chunks.append(Chunk(artifact, tuple(t for _, t in path), len(chunks), text))
        return chunks


def _text(html: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", html).replace("&amp;", "&").split())


def node(item_id: str, kind: str, parent: str | None, y: float = 0, **data: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": kind,
        "data": data,
        "position": {"x": 0, "y": y},
        "parent": {"id": parent} if parent else None,
        "modifiedAt": "2026-11-01T00:00:00Z",
    }


class FakeBoard:
    """The Ship Prototypes board: a prototypes frame and a frame of display copies."""

    def __init__(self) -> None:
        self.items = [
            node("f1", "frame", None, title="Prototypes"),
            node("n1", "sticky_note", "f1", 1, content="<p>AX-1 Kestrel: balanced, hitbox 6 px</p>"),
            node("f2", "frame", None, title="Concept copies"),
            node("i1", "image", "f2", 1, title="Kestrel concept"),  # a copy that does not name its origin
            node(
                "n2",
                "text",
                "f2",
                2,
                content='<p>Spec: <a href="https://devsinsight.atlassian.net/wiki/spaces/AETHER/pages/9076737/Ships">'
                "Ships</a></p>",
            ),
        ]
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/items"):
            return httpx.Response(200, json={"data": self.items})
        if request.url.path.endswith("/connectors"):
            return httpx.Response(200, json={"data": []})
        return httpx.Response(200, json={"id": BOARD, "name": "Aether - Ship Prototypes"})

    def reader(self) -> MiroReader:
        client = httpx.Client(transport=httpx.MockTransport(self.handler))
        return MiroReader("unused", BOARDS.scope, client=client)


class PageReader:
    def __init__(self, *pages: dict[str, Any]) -> None:
        self.pages = {str(p["id"]): p for p in pages}

    def read(self, item_id: str) -> SourceItem:
        return map_page(self.pages[item_id], BASE)

    def list_versions(self, scope: dict[str, Any]):  # noqa: ANN201
        from aether_core.domain.sync import ListedItem

        return [ListedItem(i, str(p["version"]["number"]), p["title"]) for i, p in self.pages.items()]


class Platform:
    """The wiki and the board pipelines over one projection, as the sync service runs them."""

    def __init__(self, *pages: dict[str, Any]) -> None:
        self.store, self.runs, self.state = MemoryStore(), MemoryRuns(), MemoryState()
        self.board, self.pages = FakeBoard(), PageReader(*(pages or (ships(),)))
        embedder = CountingEmbedder()
        self.wiki = RunPipeline(
            self.pages, SectionChunker(), embedder, self.store, self.runs, self.state, PARSERS
        )
        self.miro = RunPipeline(
            self.board.reader(),
            FrameChunker(lambda t: len(t.split()), 512),
            embedder,
            self.store,
            self.runs,
            self.state,
            PARSERS,
        )

    def run_wiki(self) -> PipelineRun:
        return self.wiki.execute(WIKI, list(self.pages.pages), mode="full")

    def sync_boards(self, definition: PipelineDefinition = BOARDS) -> PipelineRun:
        return self.miro.execute(definition, mode="incremental")

    def chunk(self, artifact_id: str, *heading: str) -> dict[str, Any]:
        (found,) = [
            p
            for p in self.store.payloads()
            if p["artifact_id"] == artifact_id and tuple(p["heading_path"]) == heading
        ]
        return found


def test_an_embedded_board_waits_unresolved_then_resolves_when_the_board_is_read() -> None:
    """Criterion 1, and the case table: a target in a registered source not indexed yet."""
    platform = Platform()
    run = platform.run_wiki()
    section = platform.chunk("confluence:9076737", "Ships", "Prototypes and Concept Art")
    (link,) = section["links"]
    assert (link["kind"], link["target_id"], link["resolved"]) == ("embed", f"miro:{BOARD}", False)
    assert link["heading_path"] == ["Ships", "Prototypes and Concept Art"]
    assert link["context"].startswith("The Miro board holds the prototypes")
    assert section["text"].endswith(f"Embedded board: {BOARD_URL}")  # title unknown yet: the URL
    assert [(u.item, u.url) for u in run.links.unresolved] == [("9076737", BOARD_URL)]

    platform.sync_boards()
    (link,) = platform.chunk("confluence:9076737", "Ships", "Prototypes and Concept Art")["links"]
    # A whole-board embed relates to the board, never to one guessed frame.
    assert (link["resolved"], link["artifact_id"], link["target_title"]) == (
        True,
        None,
        "Aether - Ship Prototypes",
    )


def test_a_table_row_link_carries_the_row_and_an_unknown_source_is_discovered() -> None:
    """Criteria 2 and 4: Scenario has no reader, so its links stay external and are reported."""
    platform = Platform()
    run = platform.run_wiki()
    links = platform.chunk("confluence:9076737", "Ships", "Concept Art")["links"]
    assert [link["text"] for link in links] == [
        "asset_2SLuzWyKg8Y19RBPSdoD9zyc",
        "asset_UTACQ7DSbQWw7vMJsEs5wa26",
        "asset_ERnvvZDNp5xaCtzMG9D72sZq",
    ]
    assert links[1]["context"].startswith(
        "Ship: AX-2 Halberd; Scenario asset ID: asset_UTACQ7DSbQWw7vMJsEs5wa26"
    )
    assert all(link["target"] is None and link["resolved"] is False for link in links)
    assert run.links.discovered == (("app.scenario.com", 3),)


def test_a_link_that_names_a_board_item_resolves_to_its_frame() -> None:
    """Criterion 3."""
    storage = f'<h1>Kestrel</h1><p>See <a href="{BOARD_URL}?moveToWidget=n1">the prototype</a>.</p>'
    platform = Platform(ships(page_id="42", storage=storage))
    platform.sync_boards()
    platform.run_wiki()
    (link,) = platform.chunk("confluence:42", "Kestrel")["links"]
    assert (link["resolved"], link["artifact_id"]) == (True, f"miro:{BOARD}:f1")
    assert link["target_title"] == "Aether - Ship Prototypes - Prototypes"


def test_a_changed_frame_lists_the_embedding_page_as_affected_context() -> None:
    """Criterion 5: the page's version did not change, but it embeds the board."""
    platform = Platform()
    platform.sync_boards()
    platform.run_wiki()
    platform.board.items[1] = {
        **platform.board.items[1],
        "data": {"content": "<p>AX-1 Kestrel: balanced, hitbox 5 px</p>"},
        "modifiedAt": "2026-11-10T00:00:00Z",
    }
    run = platform.sync_boards()
    (affected,) = run.affected
    assert affected.item_id == f"{BOARD}:f1"
    ships_page = next(r for r in affected.related if r.artifact_id == "confluence:9076737")
    assert f"link:miro:{BOARD}" in ships_page.terms and ships_page.owner == "Lucía Ferrer"


def test_links_are_idempotent_and_both_spellings_resolve_to_one_target() -> None:
    """Criterion 6."""
    encoded = SHIPS["body"]["storage"]["value"].replace(
        BOARD_URL, "https://miro.com/app/board/uXjVEUtjxFw%3D/"
    )
    platform = Platform(ships(), ships(page_id="43", storage=encoded))
    platform.sync_boards()
    platform.run_wiki()
    before = sorted(json.dumps(p["links"], sort_keys=True) for p in platform.store.payloads())
    points = set(platform.store.points)
    platform.run_wiki()
    assert sorted(json.dumps(p["links"], sort_keys=True) for p in platform.store.payloads()) == before
    assert set(platform.store.points) == points
    targets = {
        platform.chunk(page, "Ships", "Prototypes and Concept Art")["links"][0]["target_id"]
        for page in ("confluence:9076737", "confluence:43")
    }
    assert targets == {f"miro:{BOARD}"}


def test_an_image_without_origin_is_a_finding_and_nothing_is_written() -> None:
    """Criterion 7, and a board text link back to the page (wiki links found on boards)."""
    platform = Platform()
    platform.run_wiki()
    run = platform.sync_boards()
    assert [(i.item, i.reason) for i in run.links.images_without_origin] == [
        (f"{BOARD}:f2", "image without origin: Kestrel concept")
    ]
    assert {r.method for r in platform.board.requests} == {"GET"}
    (frame,) = [p for p in platform.store.payloads() if p["artifact_id"] == f"miro:{BOARD}:f2"]
    (back,) = frame["links"]
    assert (back["target_id"], back["resolved"], back["artifact_id"]) == (
        "confluence:9076737",
        True,
        "confluence:9076737",
    )
    assert frame["addresses"][:2] == [f"miro:{BOARD}:f2", f"miro:{BOARD}"]


def test_deleting_the_board_unresolves_the_links_that_point_to_it() -> None:
    platform = Platform()
    platform.sync_boards()
    platform.run_wiki()
    removed = PipelineDefinition("miro-aether", "miro", {"boards": []}, "aether", exclusions=())
    platform.sync_boards(removed)  # the board left the registry: its frames are deleted
    (link,) = platform.chunk("confluence:9076737", "Ships", "Prototypes and Concept Art")["links"]
    assert link["resolved"] is False
