"""Contract tests for the Confluence adapter against a recorded-shape response (no network)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import httpx

from aether_pipeline.adapters.confluence import ConfluenceReader, map_page, parse_page_properties

BASE = "https://devsinsight.atlassian.net/wiki"
PAGE = json.loads(
    (Path(__file__).parent / "fixtures" / "confluence_page_with_properties.json").read_text("utf-8")
)


def test_page_properties_are_read_from_the_details_macro() -> None:
    props = parse_page_properties(PAGE["body"]["storage"]["value"])
    assert props == {
        "key": "feature-ship-selection",
        "type": "feature",
        "state": "draft",
        "owner": "Lucía Ferrer",
    }


def test_page_maps_to_contract_with_provenance() -> None:
    """Criterion 1: chunks carry key, section, type, state and source (provider, ID, URL, version)."""
    item = map_page(PAGE, BASE)
    a = item.artifact
    assert a.key == "feature-ship-selection"
    assert a.artifact_type == "feature"
    assert a.section == "product"
    assert a.lifecycle.state == "draft"
    assert a.authority.owner == "Lucía Ferrer"
    assert (a.source.provider, a.source.provider_artifact_id, a.source.version) == (
        "confluence",
        "8192018",
        "2",
    )
    assert a.source.canonical_url == BASE + "/spaces/AETHER/pages/8192018/Ship+Selection"
    assert a.metadata_complete


def test_page_without_properties_is_flagged_not_invented() -> None:
    """Criterion 4: no Page Properties -> metadata_complete false, fields listed, state draft, no key."""
    page = copy.deepcopy(PAGE)
    page["body"]["storage"]["value"] = "<h1>Ship Selection</h1><p>No properties.</p>"
    a = map_page(page, BASE).artifact
    assert not a.metadata_complete
    assert set(a.missing_fields) == {"key", "artifact_type", "lifecycle.state"}
    assert a.lifecycle.state == "draft"
    assert a.key is None
    assert a.section == "product"  # still derived from the page tree


def test_nested_macros_inside_details_do_not_end_it_early() -> None:
    storage = (
        '<ac:structured-macro ac:name="details"><ac:rich-text-body><table><tbody>'
        '<tr><th><p>Key</p></th><td><p><ac:structured-macro ac:name="status"></ac:structured-macro>'
        "decision-adr-019</p></td></tr>"
        "<tr><th><p>State</p></th><td><p>validated</p></td></tr>"
        "</tbody></table></ac:rich-text-body></ac:structured-macro>"
    )
    assert parse_page_properties(storage) == {"key": "decision-adr-019", "state": "validated"}


def test_list_versions_pages_through_the_space_without_bodies() -> None:
    def page_json(i: int) -> dict[str, object]:
        return {
            "id": str(i),
            "title": f"Page {i}",
            "version": {"number": i % 3 + 1},
            "ancestors": [{"title": "Home"}],
        }

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        start = int(request.url.params["start"])
        results = [page_json(i) for i in range(start, min(start + 100, 130))]
        links = {"next": "/rest/api/content?start=100"} if start == 0 else {}
        return httpx.Response(200, json={"results": results, "size": len(results), "_links": links})

    reader = ConfluenceReader(BASE, "u", "t", client=httpx.Client(transport=httpx.MockTransport(handler)))
    items = list(reader.list_versions({"space": "AETHER"}))
    assert len(items) == 130 and items[0].version == "1" and items[0].ancestor_titles == ("Home",)
    assert [r.url.params["start"] for r in requests] == ["0", "100"]
    assert "body" not in requests[0].url.params["expand"]
    assert requests[0].url.params["status"] == "current"
