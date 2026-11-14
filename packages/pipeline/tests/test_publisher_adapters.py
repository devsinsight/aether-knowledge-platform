"""Contract tests for the publisher adapters against recorded-shape responses (no network)."""

from __future__ import annotations

import json

import httpx

from aether_pipeline.adapters.azure_devops import AzureDevOpsWorkItems
from aether_pipeline.adapters.confluence import map_page
from aether_pipeline.adapters.confluence_publisher import (
    ConfluenceWikiPublisher,
    properties_macro,
    to_storage,
)

BASE = "https://devsinsight.atlassian.net/wiki"
PROPS = {
    "Key": "work-record-story-chain-window",
    "Type": "work-record",
    "State": "draft",
    "Owner": "Lucía Ferrer",
    "Story": "story-chain-window (ADO 59)",
    "Feature": "feature-scoring-and-chains",
}


def page_json(
    page_id: str, version: int, storage: str, title: str = "Work Record: story-chain-window"
) -> dict:
    return {
        "id": page_id,
        "title": title,
        "version": {"number": version},
        "body": {"storage": {"value": storage}, "export_view": {"value": storage}},
        "ancestors": [{"title": "Project Aether Home"}, {"title": "Work Records"}],
        "_links": {"webui": f"/spaces/AETHER/pages/{page_id}"},
    }


def test_sync_indexes_a_published_record_as_a_work_record_with_its_keys() -> None:
    storage = to_storage("# Chain\n\n| AC | How |\n| --- | --- |\n| AC-1 | done |\n") + properties_macro(
        PROPS
    )
    assert "<table>" in storage
    a = map_page(page_json("42", 1, storage), BASE).artifact
    assert (a.artifact_type, a.key, a.lifecycle.state) == ("work-record", PROPS["Key"], "draft")
    assert {(r.type, r.target_key) for r in a.relationships} == {
        ("work_record_of", "story-chain-window"),
        ("part_of", "feature-scoring-and-chains"),
    }
    assert a.source.provider_artifact_id == "42" and a.source.version == "1"


def test_create_then_update_with_optimistic_concurrency() -> None:
    sent: list[tuple[str, str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        sent.append((request.method, request.url.path, body))
        if request.method == "GET":
            return httpx.Response(200, json={"results": [page_json("42", 1, "")]})
        if request.method == "POST":
            return httpx.Response(200, json=page_json("42", 1, body["body"]["storage"]["value"]))
        if body["version"]["number"] != 2:
            return httpx.Response(409, json={"message": "version conflict"})
        return httpx.Response(200, json=page_json("42", 2, body["body"]["storage"]["value"]))

    wiki = ConfluenceWikiPublisher(
        BASE, "u", "t", "AETHER", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    created = wiki.create_or_update("WR", "Work Record: story-chain-window", "# Chain\n", PROPS, None)
    assert created.properties["state"] == "draft" and sent[0][2]["ancestors"] == [{"id": "WR"}]
    updated = wiki.create_or_update("WR", "Work Record: story-chain-window", "# Chain v2\n", PROPS, 1)
    assert updated.version == 2 and sent[-1][0] == "PUT"
    try:
        wiki.create_or_update("WR", "Work Record: story-chain-window", "# Chain v3\n", PROPS, 5)
    except RuntimeError as error:
        assert "changed since version 5" in str(error)
    else:
        raise AssertionError("a stale version must be refused")


def test_set_properties_keeps_the_body_and_fills_the_validation_section() -> None:
    storage = to_storage("# Chain\n\n## Validation\n") + properties_macro(PROPS)
    puts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/content"):
            return httpx.Response(200, json={"results": [page_json("42", 3, storage)]})
        if request.method == "GET":
            return httpx.Response(200, json=page_json("42", 3, storage))
        value = json.loads(request.content)["body"]["storage"]["value"]
        puts.append(value)
        return httpx.Response(200, json=page_json("42", 4, value))

    wiki = ConfluenceWikiPublisher(
        BASE, "u", "t", "AETHER", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    page = wiki.find_page("Work Record: story-chain-window")
    assert page is not None
    updated = wiki.set_properties(page, {**PROPS, "State": "validated"}, "Validated by Lucía Ferrer.")
    assert updated.properties["state"] == "validated" and updated.version == 4
    assert puts[0].count('ac:name="details"') == 1
    assert "<h2>Validation</h2><p>Validated by Lucía Ferrer.</p>" in puts[0] and "<h1>Chain</h1>" in puts[0]


def test_work_item_lookup_is_read_only_and_maps_404_to_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        return httpx.Response(200 if request.url.path.endswith("/59") else 404, json={})

    items = AzureDevOpsWorkItems(
        "https://dev.azure.com/devsinsight",
        "Aether",
        "pat",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert items.exists(59) and not items.exists(999)
