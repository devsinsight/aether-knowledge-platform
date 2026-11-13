from __future__ import annotations

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
