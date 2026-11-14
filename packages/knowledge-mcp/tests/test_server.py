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
