"""SQLite run repository and state store: round trips of the sync records."""

from __future__ import annotations

from pathlib import Path

from aether_core.domain.links import LinkReport, UnresolvedLink
from aether_core.domain.pipeline import PipelineRun, StepResult
from aether_core.domain.sync import AffectedContext, IndexedChunk, IndexedVersion, RelatedItem
from aether_pipeline.adapters.sqlite_runs import SqliteRunRepository, SqliteStateStore


def test_state_store_round_trip(tmp_path: Path) -> None:
    state = SqliteStateStore(tmp_path / "s.sqlite")
    item = IndexedVersion("1", "confluence:1", "3", "k", "Title", "Owner", (IndexedChunk("p", "c", "e"),))
    state.save("pipe", item)
    assert state.versions("pipe") == {"1": "3"} and state.versions("other") == {}
    assert state.get("pipe", "1") == item
    state.delete("pipe", "1")
    assert state.get("pipe", "1") is None


def test_runs_keep_affected_context_and_latest(tmp_path: Path) -> None:
    runs = SqliteRunRepository(tmp_path / "s.sqlite")
    related = RelatedItem("confluence:2", "feature-x", "Feature X", "Lucía Ferrer", ("R-10",))
    affected = AffectedContext("1", "changed", None, "Rules", ("R-10",), (related,))
    first = PipelineRun("a", "pipe", "incremental", "cli").start().finish()
    second = (
        PipelineRun("b", "pipe", "incremental", "cli", items=("1",), deleted=("3",))
        .start()
        .record(StepResult("embed", processed=1, reused=2, item="1"))
        .report((affected,))
        .finish()
    )
    runs.save(first)
    runs.save(second)
    assert runs.get("b") == second
    latest = runs.latest("pipe")
    assert latest is not None and latest.run_id == "b"


def test_a_run_keeps_its_link_report(tmp_path: Path) -> None:
    runs = SqliteRunRepository(tmp_path / "r.sqlite")
    report = LinkReport(
        (("app.scenario.com", 3),),
        (UnresolvedLink("9076737", "https://miro.com/app/board/b=/", "miro target not indexed"),),
        (UnresolvedLink("b=:f2", "", "image without origin: Kestrel concept"),),
    )
    run = PipelineRun("r1", "confluence-aether", "full", "cli").start().report_links(report).finish()
    runs.save(run)
    assert runs.get("r1") == run
