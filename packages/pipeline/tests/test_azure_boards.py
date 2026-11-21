"""Azure Boards reader against a fake REST API 7.1 (recorded response shapes), and its URL parser."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from aether_pipeline.adapters.azure_boards import (
    AzureBoardsAuthError,
    AzureBoardsReader,
    AzureBoardsUrls,
    criteria,
    plain_text,
    wiql,
)

ORG = "https://dev.azure.com/devsinsight"
PROJECT_GUID = "65337ef0-20f9-417e-8fd2-c3f0b3089eeb"
TESTING_PAGE = "https://devsinsight.atlassian.net/wiki/spaces/AETHER/pages/950273"
SCOPE = {"organization_url": ORG, "project": "Aether", "work_item_types": ["Feature", "User Story"]}


def relation(rel: str, target: int, name: str) -> dict[str, Any]:
    return {
        "rel": rel,
        "url": f"{ORG}/{PROJECT_GUID}/_apis/wit/workItems/{target}",
        "attributes": {"isLocked": False, "name": name},
    }


def work_items() -> dict[int, dict[str, Any]]:
    return {
        47: {
            "id": 47,
            "rev": 3,
            "fields": {
                "System.WorkItemType": "Feature",
                "System.Title": "Leaderboards",
                "System.State": "New",
                "System.AreaPath": "Aether",
                "System.IterationPath": "Aether\\M2 - Vertical slice",
                "System.Tags": "leaderboard; feature-leaderboards",
                "System.Description": "<div>Online leaderboards.</div>",
            },
            "relations": [relation("System.LinkTypes.Hierarchy-Forward", 72, "Child")],
        },
        72: {
            "id": 72,
            "rev": 4,
            "fields": {
                "System.WorkItemType": "User Story",
                "System.Title": "Submit run",
                "System.State": "Removed",
                "System.AreaPath": "Aether",
                "System.IterationPath": "Aether\\M2 - Vertical slice",
                "System.Tags": "api; leaderboard",
                "Microsoft.VSTS.Scheduling.StoryPoints": 5.0,
                "System.Description": "As a signed-in player, I want my finished run submitted.",
                "Microsoft.VSTS.Common.AcceptanceCriteria": (
                    "- **Given** a run ended by game over (R-22), **then** the API answers 202.\n"
                    "- **Given** the same run is submitted twice, **then** it is idempotent.\n"
                    "\n*Refined on 2026-11-21.*"
                ),
            },
            "relations": [
                relation("System.LinkTypes.Hierarchy-Reverse", 47, "Parent"),
                {
                    "rel": "ArtifactLink",
                    "url": "vstfs:///Git/Commit/x",
                    "attributes": {"name": "Fixed in Commit"},
                },
            ],
        },
        105: {
            "id": 105,
            "rev": 3,
            "fields": {
                "System.WorkItemType": "User Story",
                "System.Title": "Traceability check in the ci pipeline",
                "System.State": "New",
                "System.AreaPath": "Aether",
                "System.Tags": "infra; ready; feature-infrastructure-and-ci-cd; rule-r-22",
                "System.AssignedTo": {"displayName": "Daniel Okafor"},
                "Microsoft.VSTS.Common.AcceptanceCriteria": (
                    "<ul><li><p><b>Given</b> a pull request</p></li><li>See "
                    f'<a href="{TESTING_PAGE}">11 - Testing</a>'
                    "</li></ul>"
                ),
            },
            "relations": [relation("System.LinkTypes.Dependency-Reverse", 77, "Predecessor")],
        },
    }


class FakeBoards:
    def __init__(self) -> None:
        self.items = work_items()
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path.endswith("/_apis/wit/wiql"):
            return httpx.Response(200, json={"workItems": [{"id": i} for i in sorted(self.items)]})
        if path.endswith("/_apis/wit/workitemsbatch"):
            body = json.loads(request.content)
            value = []
            for i in body["ids"]:
                item = self.items[i]
                fields = {f: item["fields"][f] for f in body["fields"] if f in item["fields"]}
                fields["System.Id"], fields["System.Rev"] = item["id"], item["rev"]
                value.append({"id": item["id"], "rev": item["rev"], "fields": fields})
            return httpx.Response(200, json={"count": len(value), "value": value})
        work_item_id = int(path.rsplit("/", 1)[1])
        if work_item_id not in self.items:
            return httpx.Response(404, json={"message": "not found"})
        return httpx.Response(200, json=self.items[work_item_id])

    def reader(self) -> AzureBoardsReader:
        return AzureBoardsReader(
            "unused", SCOPE, client=httpx.Client(transport=httpx.MockTransport(self.handler))
        )


def test_lists_every_work_item_in_scope_with_its_revision_and_no_body() -> None:
    fake = FakeBoards()
    listed = list(fake.reader().list_versions(SCOPE))
    assert [(i.item_id, i.version) for i in listed] == [("47", "3"), ("72", "4"), ("105", "3")]
    assert listed[1].title == "User Story 72: Submit run"
    batch = json.loads(fake.requests[1].content)
    assert "System.Description" not in batch["fields"]  # versions only, never bodies


def test_wiql_scopes_project_types_and_area() -> None:
    query = wiql({"project": "Aether", "work_item_types": ["User Story"], "area_path": "Aether\\Game"})
    assert "[System.TeamProject] = 'Aether'" in query
    assert "[System.WorkItemType] IN ('User Story')" in query
    assert "[System.AreaPath] UNDER 'Aether\\Game'" in query


def test_reads_a_story_with_provenance_and_numbered_criteria() -> None:
    item = FakeBoards().reader().read("72")
    artifact = item.artifact
    assert artifact.artifact_id == "azure-boards:72"
    assert artifact.source.provider_artifact_id == "72"
    assert artifact.source.version == "4"
    assert artifact.source.canonical_url == f"{ORG}/Aether/_workitems/edit/72"
    assert artifact.authority.artifact_class == "work_item"
    lines = artifact.text.splitlines()
    assert lines[0] == "User Story 72: Submit run"
    assert "State: Removed" in lines[1] and "Revision: 4" in lines[1] and "Story points: 5" in lines[1]
    assert lines[2] == "Parent: 47"
    assert "AC-1: **Given** a run ended by game over (R-22), **then** the API answers 202." in lines
    assert "AC-2: **Given** the same run is submitted twice, **then** it is idempotent." in lines
    assert "*Refined on 2026-11-21.*" in lines  # not a bullet: kept without a number
    assert item.addresses == ("azure-boards:72",)


def test_state_rule() -> None:
    reader = FakeBoards().reader()
    assert reader.read("72").artifact.lifecycle.state == "archived"  # Removed
    assert reader.read("47").artifact.lifecycle.state == "draft"  # New, not ready
    assert reader.read("105").artifact.lifecycle.state == "validated"  # tagged ready


def test_keys_are_taken_from_tags_never_invented() -> None:
    reader = FakeBoards().reader()
    feature = reader.read("47").artifact
    assert feature.key == "feature-leaderboards" and feature.metadata_complete
    assert (feature.artifact_type, feature.section) == ("feature", "product")
    story = reader.read("72").artifact
    assert story.key is None and "key" in story.missing_fields  # a story carries its feature's key
    tagged = reader.read("105").artifact
    assert tagged.key is None
    assert {r.target_key for r in tagged.relationships} == {"feature-infrastructure-and-ci-cd", "rule-r-22"}
    assert tagged.authority.owner == "Daniel Okafor"


def test_a_knowledge_gap_goes_to_its_section() -> None:
    fake = FakeBoards()
    fake.items[72]["fields"]["System.Tags"] = "knowledge-gap; gap-q-10"
    fake.items[72]["fields"]["System.State"] = "New"
    artifact = fake.reader().read("72").artifact
    assert (artifact.artifact_type, artifact.section, artifact.key) == (
        "knowledge-gap",
        "knowledge-gaps",
        "gap-q-10",
    )


def test_relations_and_links_are_references() -> None:
    reader = FakeBoards().reader()
    parent = reader.read("72").references
    assert [(r.url.rsplit("/", 1)[1], r.text) for r in parent] == [("47", "Parent")]  # ArtifactLink dropped
    refs = reader.read("105").references
    assert [r.text for r in refs] == ["Predecessor", "11 - Testing"]
    assert refs[1].url == TESTING_PAGE


def test_a_new_revision_is_a_new_version() -> None:
    fake = FakeBoards()
    before = fake.reader().read("105").artifact
    fake.items[105] = {**fake.items[105], "rev": 4}
    after = fake.reader().read("105").artifact
    assert before.source.version != after.source.version


def test_html_criteria_are_text_and_numbered() -> None:
    assert criteria("<ul><li><p><b>Given</b> a pull request</p></li><li>See it</li></ul>") == [
        "AC-1: Given a pull request",
        "AC-2: See it",
    ]
    assert plain_text("<div>A&nbsp;b</div><div>c</div>") == "A b\nc"
    assert plain_text(None) == ""


def test_url_parser_recognizes_work_item_urls_only() -> None:
    urls = AzureBoardsUrls()
    assert urls.parse_url(f"{ORG}/Aether/_workitems/edit/72").target_id == "72"  # type: ignore[union-attr]
    assert urls.parse_url(f"{ORG}/{PROJECT_GUID}/_apis/wit/workItems/49").target_id == "49"  # type: ignore[union-attr]
    assert urls.parse_url("https://devsinsight.visualstudio.com/Aether/_workitems/edit/7") is not None
    assert urls.parse_url(f"{ORG}/Aether/_backlogs/backlog/") is None
    assert urls.parse_url("https://miro.com/app/board/x/") is None


def test_a_rejected_token_is_reported_clearly() -> None:
    def redirect(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://app.vssps.visualstudio.com/_signin?realm=x"})

    reader = AzureBoardsReader("bad", SCOPE, client=httpx.Client(transport=httpx.MockTransport(redirect)))
    with pytest.raises(AzureBoardsAuthError, match="ADO_READ_PAT"):
        list(reader.list_versions(SCOPE))
