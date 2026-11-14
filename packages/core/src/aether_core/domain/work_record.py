"""Work records (AIEF ADR-006; platform.md, "Increment: Publish").

A work record is the analysis and resolution of one user story, written by an agent as Markdown with
front matter (templates/work-record.md). The publisher is deterministic: it rejects a record with a
missing required field and never fills one; a published record starts as ``draft``; an agent never
overwrites a ``validated`` record.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

WORK_RECORD_TYPE = "work-record"
REQUIRED_FIELDS: tuple[str, ...] = ("key", "type", "story", "story_revision", "feature", "state", "author")

# "story-scoring-chain-window (ADO 123)": the story key and the work item ID it is anchored to.
_STORY = re.compile(r"^(?P<key>[A-Za-z0-9][\w.-]*)\s*\(\s*(?:ADO\s*)?#?(?P<id>\d+)\s*\)$")
_TITLE = re.compile(r"^#\s+(?P<title>.+?)\s*$", re.MULTILINE)


class WorkRecordRejectedError(ValueError):
    """The record cannot be published as it is; nothing was written."""


@dataclass(frozen=True)
class WorkRecord:
    key: str
    story_key: str
    work_item_id: int
    story_revision: str
    feature: str
    state: str
    author: str
    body: str
    title: str
    owner: str | None = None
    pull_request: str | None = None

    @property
    def page_title(self) -> str:
        """Deterministic and unique per story, so publishing again updates the same page."""
        return page_title(self.story_key)

    def properties(self) -> dict[str, str]:
        """The common record written into the Page Properties macro, in this order."""
        props = {
            "Key": self.key,
            "Type": WORK_RECORD_TYPE,
            "State": self.state,
            "Owner": self.owner or "",
            "Story": f"{self.story_key} (ADO {self.work_item_id})",
            "Story revision": self.story_revision,
            "Feature": self.feature,
            "Author": self.author,
            "Pull request": self.pull_request or "",
        }
        return {k: v for k, v in props.items() if v}


def page_title(story_key: str) -> str:
    return f"Work Record: {story_key}"


def feature_page_title(feature: str) -> str:
    return f"Work Records: {feature}"


def parse_work_record(front_matter: Mapping[str, object], body: str) -> WorkRecord:
    """Validate the front matter and build the record. Raises WorkRecordRejectedError, never fills."""
    values = {
        name: str(front_matter[name]).strip() for name in front_matter if front_matter[name] is not None
    }
    missing = [name for name in REQUIRED_FIELDS if not values.get(name)]
    if missing:
        raise WorkRecordRejectedError(f"missing required front matter: {', '.join(missing)}")
    if values["type"] != WORK_RECORD_TYPE:
        raise WorkRecordRejectedError(f"type must be {WORK_RECORD_TYPE!r}, got {values['type']!r}")
    if values["state"] != "draft":
        raise WorkRecordRejectedError("a published record starts as draft; only a person validates it")
    story = _STORY.match(values["story"])
    if story is None:
        raise WorkRecordRejectedError(
            f"story must be '<story key> (ADO <work item ID>)', got {values['story']!r}"
        )
    heading = _TITLE.search(body)
    return WorkRecord(
        key=values["key"],
        story_key=story.group("key"),
        work_item_id=int(story.group("id")),
        story_revision=values["story_revision"],
        feature=values["feature"],
        state="draft",
        author=values["author"],
        body=body,
        title=heading.group("title") if heading else story.group("key"),
        owner=values.get("owner") or None,
        pull_request=values.get("pull_request") or None,
    )


def check_overwrite(existing_state: str | None) -> None:
    """An agent never overwrites a validated record: a later change needs a person."""
    if existing_state == "validated":
        raise WorkRecordRejectedError(
            "the published record is validated; an agent cannot overwrite it (a person must reopen it)"
        )
