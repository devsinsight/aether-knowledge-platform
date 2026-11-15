"""Internal sync API over HTTP with fake use cases: preview, run (one at a time) and status, per source."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from typing import Any

from aether_core.domain.pipeline import PipelineDefinition, PipelineRun, RunMode
from aether_core.domain.sync import SyncPlan
from aether_pipeline.entrypoints.sync_api import SyncService, make_handler

WIKI = PipelineDefinition("confluence-aether", "confluence", {"space": "AETHER"}, "aether")
BOARDS = PipelineDefinition("miro-aether", "miro", {"boards": [{"id": "b1", "section": "product"}]}, "aether")


class Runs:
    def __init__(self) -> None:
        self.runs: dict[str, PipelineRun] = {}

    def save(self, run: PipelineRun) -> None:
        self.runs[run.run_id] = run

    def get(self, run_id: str) -> PipelineRun | None:
        return self.runs.get(run_id)

    def latest(self, pipeline: str) -> PipelineRun | None:
        found = [r for r in self.runs.values() if r.pipeline == pipeline]
        return found[-1] if found else None


class Preview:
    def __init__(self, definition: PipelineDefinition) -> None:
        if definition.source == "miro" and "broken" in definition.scope:
            raise RuntimeError("Missing environment variable MIRO_ACCESS_TOKEN")

    def execute(self, definition: PipelineDefinition) -> SyncPlan:
        if definition.source == "miro":
            return SyncPlan(new=("b1:f1",))
        return SyncPlan(changed=("2",), unchanged=("1",))


class Pipeline:
    def __init__(self, runs: Runs, release: threading.Event) -> None:
        self.runs, self.release = runs, release
        self.calls: list[tuple[str, tuple[str, ...], str, str]] = []

    def for_definition(self, definition: PipelineDefinition) -> Pipeline:
        if "broken" in definition.scope:
            raise RuntimeError("Missing environment variable MIRO_ACCESS_TOKEN")
        return self

    def execute(
        self,
        definition: PipelineDefinition,
        items: Any = (),
        mode: RunMode = "single_item",
        trigger: str = "cli",
        run_id: str | None = None,
    ) -> PipelineRun:
        self.calls.append((definition.name, tuple(items), mode, trigger))
        self.release.wait(5)
        run = PipelineRun(run_id or "x", definition.name, mode, trigger, items=tuple(items)).start().finish()
        self.runs.save(run)
        return run


def call(base: str, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(base + path, data=data, method=method)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def wait_until_done(base: str, run_id: str) -> dict[str, Any]:
    for _ in range(100):
        status, run = call(base, "GET", f"/sync/status?run_id={run_id}")
        if status == 200 and not run["active"]:
            return run
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def serve(
    definitions: list[PipelineDefinition],
) -> tuple[str, Pipeline, threading.Event, ThreadingHTTPServer]:
    runs, release = Runs(), threading.Event()
    pipeline = Pipeline(runs, release)
    service = SyncService(definitions, Preview, pipeline.for_definition, runs)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}", pipeline, release, server


def test_preview_run_status_per_source_and_one_sync_at_a_time() -> None:
    base, pipeline, release, server = serve([WIKI, BOARDS])
    try:
        status, preview = call(base, "POST", "/sync/preview")
        assert status == 200 and preview["is_empty"] is False
        assert [(p["pipeline"], p["new"], p["changed"]) for p in preview["pipelines"]] == [
            ("confluence-aether", [], ["2"]),
            ("miro-aether", ["b1:f1"], []),
        ]
        assert call(base, "POST", "/sync/preview", {"pipeline": "miro-aether"})[1]["pipelines"][0]["new"] == [
            "b1:f1"
        ]
        assert call(base, "POST", "/sync/preview", {"pipeline": "nope"})[0] == 400

        status, started = call(base, "POST", "/sync/run", {"trigger": "knowledge-mcp"})
        assert status == 202 and set(started["run_ids"]) == {"confluence-aether", "miro-aether"}
        assert call(base, "POST", "/sync/run", {})[0] == 409  # one sync at a time
        release.set()
        for run_id in started["run_ids"].values():
            assert wait_until_done(base, run_id)["status"] == "succeeded"
        assert [(c[0], c[2]) for c in pipeline.calls] == [
            ("confluence-aether", "incremental"),
            ("miro-aether", "incremental"),
        ]
        latest = call(base, "GET", "/sync/status")[1]["runs"]
        assert [r["pipeline"] for r in latest] == ["confluence-aether", "miro-aether"]

        # The publisher syncs one page: items go to the wiki pipeline, and the response keeps run_id.
        _, single = call(base, "POST", "/sync/run", {"items": ["9076781"], "trigger": "publisher"})
        wait_until_done(base, single["run_id"])
        assert pipeline.calls[-1] == ("confluence-aether", ("9076781",), "single_item", "publisher")
    finally:
        server.shutdown()


def test_a_failing_source_is_reported_and_never_stops_the_others() -> None:
    broken = PipelineDefinition("miro-aether", "miro", {"broken": True}, "aether")
    base, pipeline, release, server = serve([WIKI, broken])
    release.set()
    try:
        plans = call(base, "POST", "/sync/preview")[1]["pipelines"]
        assert "changed" in plans[0] and "MIRO_ACCESS_TOKEN" in plans[1]["error"]
        _, started = call(base, "POST", "/sync/run", {})
        wiki = wait_until_done(base, started["run_ids"]["confluence-aether"])
        boards = wait_until_done(base, started["run_ids"]["miro-aether"])
        assert wiki["status"] == "succeeded"
        assert boards["status"] == "failed" and "MIRO_ACCESS_TOKEN" in boards["errors"][0]["message"]
    finally:
        server.shutdown()
