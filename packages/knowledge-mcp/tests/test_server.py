from __future__ import annotations

import inspect
from typing import Any

import httpx
import pytest

from aether_knowledge_mcp import server
from aether_knowledge_mcp.server import build_filter, to_result


def test_only_validated_by_default() -> None:
    f = build_filter(include_drafts=False, section=None, key=None)
    must = list(f.must or [])  # type: ignore[arg-type]
    assert must[0].match.any == ["validated"]  # type: ignore[union-attr]


def test_drafts_and_exact_filters_on_request() -> None:
    f = build_filter(include_drafts=True, section="product", key="feature-leaderboards")
    must = list(f.must or [])  # type: ignore[arg-type]
    assert len(must) == 3
    assert must[0].match.any == ["validated", "draft"]  # type: ignore[union-attr]


def test_result_carries_provenance_and_state() -> None:
    payload = {
        "title": "Leaderboards",
        "text": "Each difficulty has its own leaderboard.",
        "key": "feature-leaderboards",
        "section": "product",
        "artifact_type": "feature",
        "lifecycle": {"state": "draft"},
        "source": {
            "provider": "confluence",
            "provider_artifact_id": "8323155",
            "canonical_url": "u",
            "version": "2",
        },
        "metadata_complete": True,
    }
    result = to_result(0.5, payload)
    assert (
        result["source"]["version"] == "2"
        and result["state"] == "draft"
        and result["key"] == "feature-leaderboards"
    )


def test_sync_tools_only_call_the_internal_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sync criterion 6: the MCP asks the sync service; it never runs the pipeline itself."""
    calls: list[tuple[str, str, dict[str, Any]]] = []

    def fake_request(method: str, url: str, **kwargs: Any) -> httpx.Response:
        calls.append((method, url, dict(kwargs)))
        return httpx.Response(202, json={"run_id": "r1"}, request=httpx.Request(method, url))

    monkeypatch.setattr(httpx, "request", fake_request)
    assert server.sync_run() == {"run_id": "r1"}
    server.sync_preview()
    server.sync_status("r1")
    assert [(m, u.split("/")[-2:]) for m, u, _ in calls] == [
        ("POST", ["sync", "run"]),
        ("POST", ["sync", "preview"]),
        ("GET", ["sync", "status"]),
    ]
    assert calls[2][2]["params"] == {"run_id": "r1"}


def test_an_unreachable_service_is_reported_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    def down(method: str, url: str, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "request", down)
    assert "unreachable" in server.sync_preview()["error"]


def test_the_mcp_never_imports_the_pipeline_or_holds_source_credentials() -> None:
    source = inspect.getsource(server)
    assert "aether_pipeline" not in source and "CONFLUENCE" not in source


def test_publish_sends_only_the_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Publish criterion 8: only the path crosses the MCP; the MCP holds no wiki credential."""
    calls: list[tuple[str, str, dict[str, Any]]] = []

    def fake_request(method: str, url: str, **kwargs: Any) -> httpx.Response:
        calls.append((method, url, dict(kwargs)))
        return httpx.Response(200, json={"status": "published"}, request=httpx.Request(method, url))

    monkeypatch.setattr(httpx, "request", fake_request)
    assert server.work_records_publish("chain.md") == {"status": "published"}
    server.work_records_status("story-chain-window")
    assert calls[0][1].endswith("/work-records/publish")
    assert calls[0][2]["json"] == {"path": "chain.md", "trigger": "knowledge-mcp"}
    assert calls[1][:2] == ("GET", f"{server.PUBLISHER_URL}/work-records/story-chain-window")


def test_unscoped_search_skips_drafts_and_scoped_work_records_are_labeled() -> None:
    """Publish criterion 6: drafts stay out of search by default; a scoped query labels them."""
    must = list(build_filter(include_drafts=False, section=None, key=None).must or [])  # type: ignore[arg-type]
    assert must[0].match.any == ["validated"]  # type: ignore[union-attr]

    scoped = server.work_records_filter("story-chain-window")
    assert {c.key for c in scoped.should} == {"key", "relationships[].key"}  # type: ignore[union-attr]

    def chunk(position: int, text: str, state: str = "draft") -> dict[str, Any]:
        return {
            "artifact_id": "confluence:42",
            "title": "Work Record: story-chain-window",
            "lifecycle": {"state": state},
            "position": position,
            "text": text,
        }

    (record,) = server.group_work_records([chunk(1, "second"), chunk(0, "first")])
    assert record["text"] == "first\n\nsecond" and record["authoritative"] is False and record["note"]
    (validated,) = server.group_work_records([chunk(0, "x", "validated")])
    assert validated["authoritative"] is True and validated["note"] is None


def test_get_related_returns_linked_items_in_both_directions() -> None:
    """Cross-source links: a page that embeds a board and the board's frames find each other."""
    from qdrant_client import QdrantClient, models

    qdrant = QdrantClient(":memory:")
    qdrant.create_collection(server.COLLECTION, vectors_config={})
    embed = {
        "kind": "embed",
        "url": "https://miro.com/app/board/b=/",
        "text": "",
        "context": "The Miro board holds the prototypes.",
        "heading_path": ["Ships", "Prototypes and Concept Art"],
        "target": {"provider": "miro", "id": "b=", "item": None},
        "target_id": "miro:b=",
        "target_item": None,
        "resolved": True,
    }
    asset = {**embed, "kind": "link", "url": "https://app.scenario.com/assets?openAssetId=a", "target": None}
    asset.update(target_id=None, resolved=False)

    def point(
        n: int, artifact_id: str, state: str, addresses: list[str], links: list[Any]
    ) -> models.PointStruct:
        payload = {
            "artifact_id": artifact_id,
            "title": artifact_id,
            "key": None,
            "lifecycle": {"state": state},
            "addresses": addresses,
            "links": links,
        }
        return models.PointStruct(id=n, vector={}, payload=payload)

    qdrant.upsert(
        server.COLLECTION,
        [
            point(1, "confluence:9076737", "validated", ["confluence:9076737"], [embed, asset]),
            point(2, "miro:b=:f1", "draft", ["miro:b=:f1", "miro:b=", "miro:b=:n1"], []),
            point(3, "miro:b=:f2", "draft", ["miro:b=:f2", "miro:b="], []),
        ],
    )
    page = server.related_items(qdrant, None, "confluence:9076737", 50)
    frames = {i["artifact_id"]: i for i in page["items"] if i["artifact_id"].startswith("miro:")}
    assert set(frames) == {"miro:b=:f1", "miro:b=:f2"}
    assert frames["miro:b=:f1"]["relations"][0]["relation"] == "linked_from_this"
    assert frames["miro:b=:f1"]["authoritative"] is False  # drafts are labeled, not hidden
    assert [e["url"] for e in page["external_links"]] == [asset["url"]]

    frame = server.related_items(qdrant, None, "miro:b=:f1", 50)
    (embedding,) = [i for i in frame["items"] if i["artifact_id"] == "confluence:9076737"]
    assert embedding["relations"] == [
        {
            "relation": "links_to_this",
            "kind": "embed",
            "url": embed["url"],
            "text": "",
            "context": embed["context"],
            "heading_path": embed["heading_path"],
        }
    ]
    assert "error" in server.related_items(qdrant, None, None, 50)
