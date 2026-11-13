from __future__ import annotations

from aether_core.domain.pipeline import (
    AGENT_WORKSPACE_TITLE,
    PipelineDefinition,
    PipelineRun,
    RunError,
    StepResult,
    is_excluded,
)

DEFINITION = PipelineDefinition(
    name="confluence-aether",
    source="confluence",
    scope={"space": "AETHER"},
    collection="aether",
)


def test_agent_workspace_is_excluded_by_default() -> None:
    assert is_excluded(("Project Aether Home", AGENT_WORKSPACE_TITLE), DEFINITION)
    assert not is_excluded(("Project Aether Home", "Section 2 - Business"), DEFINITION)


def test_run_without_errors_succeeds() -> None:
    run = PipelineRun("r1", "confluence-aether", "single_item", "cli").start()
    run = run.record(StepResult("extract", processed=1)).finish()
    assert run.status == "succeeded"
    assert run.started_at is not None and run.finished_at is not None
    assert run.steps[0].processed == 1


def test_failed_items_are_recorded_and_fail_the_run() -> None:
    run = PipelineRun("r1", "confluence-aether", "full", "cli").start()
    run = run.fail_item(RunError("extract", "655393", "timeout", retryable=True)).finish()
    assert run.status == "failed"
    assert run.errors[0].item == "655393"
