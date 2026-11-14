"""Use cases PublishWorkRecord and ValidateWorkRecord (platform.md, "Increment: Publish").

Publish: read the file by path from the mounted folder -> validate the front matter -> confirm the
anchored work item exists -> scan for secrets -> create or update the page under
Work Records / <feature> / <story> -> ask sync for that page -> record the run. Nothing is written when
any check fails. The projection is never written here: sync reads the page back from the wiki, so the
index cites the wiki page ID and version, never the file.
"""

from __future__ import annotations

import contextlib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from aether_core.domain.pipeline import PipelineRun, RunError, StepResult
from aether_core.domain.work_record import (
    WorkRecordRejectedError,
    check_overwrite,
    feature_page_title,
    page_title,
    parse_work_record,
)
from aether_core.ports import (
    PublishedPage,
    RunRepository,
    SecretScanner,
    SyncRequester,
    WikiPublisher,
    WorkItemLookup,
    WorkRecordFiles,
)

PUBLISHER = "publisher"


@dataclass(frozen=True)
class PublishResult:
    run_id: str
    status: str  # "published" or "rejected"
    message: str = ""
    page: PublishedPage | None = None
    created: bool = False
    sync_run_id: str | None = None


class PublishWorkRecord:
    def __init__(
        self,
        files: WorkRecordFiles,
        wiki: WikiPublisher,
        work_items: WorkItemLookup,
        scanner: SecretScanner,
        sync: SyncRequester,
        runs: RunRepository,
        parent_page_id: str,
    ) -> None:
        self._files, self._wiki, self._work_items = files, wiki, work_items
        self._scanner, self._sync, self._runs = scanner, sync, runs
        self._parent = parent_page_id

    def execute(self, path: str, trigger: str = "cli") -> PublishResult:
        run = PipelineRun(str(uuid.uuid4()), PUBLISHER, "single_item", trigger, items=(path,)).start()
        try:
            raw = self._files.read(path)
            record = parse_work_record(raw.front_matter, raw.body)
            if not self._work_items.exists(record.work_item_id):
                raise WorkRecordRejectedError(f"work item {record.work_item_id} does not exist")
            found = self._scanner.scan(raw.body + "\n" + "\n".join(map(str, raw.front_matter.values())))
            if found:
                raise WorkRecordRejectedError(f"the file looks like it contains secrets: {', '.join(found)}")
            existing = self._wiki.find_page(record.page_title)
            check_overwrite(existing.properties.get("state") if existing else None)
        except WorkRecordRejectedError as error:
            rejected = self._finish(run.fail_item(RunError("verify", path, str(error), retryable=False)))
            return PublishResult(rejected.run_id, "rejected", str(error))
        run = run.record(StepResult("verify", processed=1, item=path))

        feature_page = self._wiki.ensure_parent(self._parent, feature_page_title(record.feature))
        page = self._wiki.create_or_update(
            feature_page,
            record.page_title,
            record.body,
            record.properties(),
            existing.version if existing else None,
        )
        run = run.record(StepResult("load", processed=1, item=page.page_id))
        sync_run = self._request_sync(page.page_id)
        run = self._finish(run)
        return PublishResult(
            run.run_id, "published", page=page, created=existing is None, sync_run_id=sync_run
        )

    def _request_sync(self, page_id: str) -> str | None:
        try:
            return self._sync.sync_items([page_id], trigger=PUBLISHER)
        except Exception:  # the page is published; the next sync picks it up
            return None

    def _finish(self, run: PipelineRun) -> PipelineRun:
        finished = run.finish()
        self._runs.save(finished)
        return finished


class ValidateWorkRecord:
    """A person (or CI on merge) approves the work: the record becomes validated, with who, when and PR."""

    def __init__(self, wiki: WikiPublisher, sync: SyncRequester) -> None:
        self._wiki, self._sync = wiki, sync

    def execute(self, story_key: str, pull_request: str, validated_by: str) -> PublishedPage:
        if not pull_request.strip() or not validated_by.strip():
            raise WorkRecordRejectedError("validation needs the pull request and who validates")
        page = self._wiki.find_page(page_title(story_key))
        if page is None:
            raise WorkRecordRejectedError(f"no work record published for {story_key}")
        when = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        properties = {
            **_display(page.properties),
            "State": "validated",
            "Pull request": pull_request,
            "Validated by": validated_by,
            "Validated at": when,
        }
        note = f"Validated by {validated_by} on {when}, pull request {pull_request}."
        updated = self._wiki.set_properties(page, properties, note)
        with contextlib.suppress(Exception):  # the next sync picks it up
            self._sync.sync_items([updated.page_id], trigger=PUBLISHER)
        return updated


_LABELS = {
    "key": "Key",
    "type": "Type",
    "state": "State",
    "owner": "Owner",
    "story": "Story",
    "story revision": "Story revision",
    "feature": "Feature",
    "author": "Author",
    "pull request": "Pull request",
    "validated by": "Validated by",
    "validated at": "Validated at",
}


def _display(properties: dict[str, str]) -> dict[str, str]:
    """Page Properties are read back with lowercase headers; write them with their labels."""
    return {_LABELS.get(k, k.title()): v for k, v in properties.items()}
