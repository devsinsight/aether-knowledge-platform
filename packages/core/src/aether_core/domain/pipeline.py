"""Pipeline domain model: definitions, runs, step results and errors (platform.md, Domain model)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Literal

RunMode = Literal["full", "incremental", "single_item"]
RunStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
Step = Literal["extract", "normalize", "parse", "chunk", "enrich", "embed", "load", "delete", "verify"]

# Pages under this title are drafts and reports; they are never indexed (AIEF 6651906, v3).
AGENT_WORKSPACE_TITLE = "Agent Workspace"


@dataclass(frozen=True)
class PipelineDefinition:
    """What to ingest. Loaded from YAML in Git; never holds credentials."""

    name: str
    source: str
    scope: dict[str, str]
    collection: str
    filters: dict[str, str] = field(default_factory=dict)
    exclusions: tuple[str, ...] = (AGENT_WORKSPACE_TITLE,)
    schedule: str | None = None


@dataclass(frozen=True)
class RunError:
    step: Step
    item: str
    message: str
    retryable: bool
    version: str | None = None


@dataclass(frozen=True)
class StepResult:
    step: Step
    processed: int = 0
    failed: int = 0


@dataclass(frozen=True)
class PipelineRun:
    run_id: str
    pipeline: str
    mode: RunMode
    trigger: str
    status: RunStatus = "queued"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    steps: tuple[StepResult, ...] = ()
    errors: tuple[RunError, ...] = ()
    items: tuple[str, ...] = ()

    def start(self) -> PipelineRun:
        return replace(self, status="running", started_at=datetime.now(UTC))

    def record(self, result: StepResult) -> PipelineRun:
        return replace(self, steps=(*self.steps, result))

    def fail_item(self, error: RunError) -> PipelineRun:
        """Failed items are recorded, never hidden."""
        return replace(self, errors=(*self.errors, error))

    def finish(self) -> PipelineRun:
        status: RunStatus = "failed" if self.errors else "succeeded"
        return replace(self, status=status, finished_at=datetime.now(UTC))


def is_excluded(ancestor_titles: tuple[str, ...], definition: PipelineDefinition) -> bool:
    """True when the item sits under an excluded subtree (for example the Agent Workspace)."""
    return any(title in definition.exclusions for title in ancestor_titles)
