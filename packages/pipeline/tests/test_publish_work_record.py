"""PublishWorkRecord and ValidateWorkRecord with fakes: platform.md, "Increment: Publish" criteria."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from aether_core.domain.pipeline import PipelineRun
from aether_core.domain.work_record import WorkRecordRejectedError
from aether_core.ports import PublishedPage
from aether_pipeline.adapters.confluence import parse_page_properties
from aether_pipeline.adapters.confluence_publisher import properties_macro, to_storage
from aether_pipeline.adapters.secret_scanner import RegexSecretScanner
from aether_pipeline.adapters.work_record_files import MountedWorkRecordFiles
from aether_pipeline.application.publish_work_record import PublishWorkRecord, ValidateWorkRecord

RECORD = """---
key: work-record-story-chain-window
type: work-record
story: story-chain-window (ADO 59)
story_revision: 4
feature: feature-scoring-and-chains
state: draft
author: developer agent
owner: Lucía Ferrer
---

<!-- template comment -->
# Chain and multiplier: work record

## Analysis

Rules R-10 and R-11 (Game Rules, v3).

## Validation
"""


class MemoryWiki:
    """Pages by title; storage is what the real adapter would write, so properties round-trip."""

    def __init__(self) -> None:
        self.pages: dict[str, dict[str, object]] = {}
        self.writes = 0

    def _page(self, title: str) -> PublishedPage | None:
        page = self.pages.get(title)
        if page is None:
            return None
        storage = str(page["storage"])
        return PublishedPage(
            str(page["id"]),
            int(str(page["version"])),
            title,
            f"u/{page['id']}",
            parse_page_properties(storage),
        )

    def find_page(self, title: str) -> PublishedPage | None:
        return self._page(title)

    def ensure_parent(self, parent_id: str, title: str) -> str:
        if title not in self.pages:
            self.pages[title] = {
                "id": f"p{len(self.pages)}",
                "version": 1,
                "parent": parent_id,
                "storage": "",
            }
        return str(self.pages[title]["id"])

    def create_or_update(
        self,
        parent_id: str,
        title: str,
        body_markdown: str,
        properties: dict[str, str],
        expected_version: int | None,
    ) -> PublishedPage:
        self.writes += 1
        storage = to_storage(body_markdown) + properties_macro(properties)
        if expected_version is None:
            assert title not in self.pages
            self.pages[title] = {
                "id": f"p{len(self.pages)}",
                "version": 1,
                "parent": parent_id,
                "storage": storage,
            }
        else:
            page = self.pages[title]
            assert page["version"] == expected_version, "optimistic concurrency"
            page.update(version=expected_version + 1, storage=storage)
        result = self._page(title)
        assert result is not None
        return result

    def set_properties(self, page: PublishedPage, properties: dict[str, str], note: str) -> PublishedPage:
        self.writes += 1
        stored = self.pages[page.title]
        stored.update(
            version=int(str(stored["version"])) + 1, storage=f"<p>{note}</p>" + properties_macro(properties)
        )
        result = self._page(page.title)
        assert result is not None
        return result


class WorkItems:
    def __init__(self, *existing: int) -> None:
        self.existing = set(existing)

    def exists(self, work_item_id: int) -> bool:
        return work_item_id in self.existing


class SyncCalls:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str]] = []

    def sync_items(self, item_ids: Sequence[str], trigger: str) -> str:
        self.calls.append((list(item_ids), trigger))
        return "sync-run"


class Runs:
    def __init__(self) -> None:
        self.runs: list[PipelineRun] = []

    def save(self, run: PipelineRun) -> None:
        self.runs.append(run)

    def get(self, run_id: str) -> PipelineRun | None:
        return next((r for r in self.runs if r.run_id == run_id), None)

    def latest(self, pipeline: str) -> PipelineRun | None:
        return self.runs[-1] if self.runs else None


class Setup:
    def __init__(self, folder: Path, work_items: WorkItems | None = None) -> None:
        self.folder, self.wiki, self.sync, self.runs = folder, MemoryWiki(), SyncCalls(), Runs()
        self.use_case = PublishWorkRecord(
            MountedWorkRecordFiles(folder),
            self.wiki,
            work_items or WorkItems(59),
            RegexSecretScanner(),
            self.sync,
            self.runs,
            parent_page_id="WR",
        )

    def write(self, name: str, text: str) -> str:
        (self.folder / name).write_text(text, encoding="utf-8")
        return name


@pytest.fixture
def setup(tmp_path: Path) -> Setup:
    folder = tmp_path / "work-records"
    folder.mkdir()
    return Setup(folder)


def test_1_a_valid_record_becomes_one_draft_page_under_its_feature_and_is_synced(setup: Setup) -> None:
    result = setup.use_case.execute(setup.write("chain.md", RECORD))
    assert result.status == "published" and result.created and result.page is not None
    page = setup.wiki.pages["Work Record: story-chain-window"]
    feature = setup.wiki.pages["Work Records: feature-scoring-and-chains"]
    assert page["parent"] == feature["id"] and feature["parent"] == "WR"
    props = result.page.properties
    assert props["type"] == "work-record" and props["state"] == "draft"
    assert props["key"] == "work-record-story-chain-window" and props["owner"] == "Lucía Ferrer"
    assert props["story"] == "story-chain-window (ADO 59)" and props["story revision"] == "4"
    assert "template comment" not in str(page["storage"])
    assert setup.sync.calls == [([result.page.page_id], "publisher")]
    assert setup.runs.runs[-1].status == "succeeded"


def test_2_publishing_the_same_story_again_adds_a_version_not_a_page(setup: Setup) -> None:
    first = setup.use_case.execute(setup.write("chain.md", RECORD))
    second = setup.use_case.execute(setup.write("chain.md", RECORD.replace("v3).", "v3). Updated.")))
    assert first.page and second.page and first.page.page_id == second.page.page_id
    assert (first.page.version, second.page.version) == (1, 2) and not second.created
    assert len([t for t in setup.wiki.pages if t.startswith("Work Record:")]) == 1


@pytest.mark.parametrize(
    ("change", "message"),
    [(("story_revision: 4\n", ""), "story_revision"), (("(ADO 59)", "(ADO 999)"), "999 does not exist")],
)
def test_3_missing_field_or_unknown_work_item_is_rejected_and_nothing_is_written(
    setup: Setup, change: tuple[str, str], message: str
) -> None:
    result = setup.use_case.execute(setup.write("bad.md", RECORD.replace(*change)))
    assert result.status == "rejected" and message in result.message
    assert setup.wiki.writes == 0 and setup.wiki.pages == {} and setup.sync.calls == []
    assert setup.runs.runs[-1].status == "failed"


@pytest.mark.parametrize("path", ["../outside.md", "/etc/passwd", "chain.txt"])
def test_4_a_path_outside_the_mounted_folder_is_rejected(setup: Setup, path: str) -> None:
    (setup.folder.parent / "outside.md").write_text(RECORD, encoding="utf-8")
    result = setup.use_case.execute(path)
    assert result.status == "rejected" and setup.wiki.writes == 0


def test_5_a_validated_record_cannot_be_overwritten_by_an_agent(setup: Setup) -> None:
    setup.use_case.execute(setup.write("chain.md", RECORD))
    ValidateWorkRecord(setup.wiki, setup.sync).execute(
        "story-chain-window", "https://github.com/x/pull/1", "Lucía Ferrer"
    )
    writes = setup.wiki.writes
    result = setup.use_case.execute("chain.md")
    assert result.status == "rejected" and "validated" in result.message
    assert setup.wiki.writes == writes


def test_7_validate_records_who_when_and_the_pull_request(setup: Setup) -> None:
    setup.use_case.execute(setup.write("chain.md", RECORD))
    page = ValidateWorkRecord(setup.wiki, setup.sync).execute(
        "story-chain-window", "https://github.com/x/pull/1", "Lucía Ferrer"
    )
    props = page.properties
    assert props["state"] == "validated" and props["validated by"] == "Lucía Ferrer"
    assert props["pull request"] == "https://github.com/x/pull/1" and props["validated at"].endswith("UTC")
    assert props["key"] == "work-record-story-chain-window"  # the rest of the record is kept
    assert page.version == 2 and setup.sync.calls[-1] == ([page.page_id], "publisher")


def test_validate_needs_the_pull_request_and_a_published_record(setup: Setup) -> None:
    validate = ValidateWorkRecord(setup.wiki, setup.sync)
    with pytest.raises(WorkRecordRejectedError):
        validate.execute("story-chain-window", "", "Lucía Ferrer")
    with pytest.raises(WorkRecordRejectedError, match="no work record"):
        validate.execute("story-chain-window", "https://github.com/x/pull/1", "Lucía Ferrer")


def test_a_file_with_a_secret_is_rejected(setup: Setup) -> None:
    leaked = RECORD + "\nconfig: api_key = 'abcdefghijklmnop1234'\n"
    result = setup.use_case.execute(setup.write("leak.md", leaked))
    assert result.status == "rejected" and "assigned secret" in result.message
    assert "abcdefghijklmnop1234" not in result.message and setup.wiki.writes == 0


def test_a_record_that_claims_to_be_validated_is_rejected(setup: Setup) -> None:
    result = setup.use_case.execute(setup.write("v.md", RECORD.replace("state: draft", "state: validated")))
    assert result.status == "rejected" and "starts as draft" in result.message


def test_a_sync_failure_does_not_undo_the_publication(setup: Setup) -> None:
    class Down:
        def sync_items(self, item_ids: Sequence[str], trigger: str) -> str:
            raise ConnectionError("sync is down")

    setup.use_case._sync = Down()  # noqa: SLF001
    result = setup.use_case.execute(setup.write("chain.md", RECORD))
    assert result.status == "published" and result.sync_run_id is None
