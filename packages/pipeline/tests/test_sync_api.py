"""Internal sync API over HTTP with fake use cases: preview, run (one at a time) and status."""

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

DEFINITION = PipelineDefinition("confluence-aether", "confluence", {"space": "AETHER"}, "aether")


class Runs:
    def __init__(self) -> None:
        self.runs: dict[str, PipelineRun] = {}

    def save(self, run: PipelineRun) -> None:
        self.runs[run.run_id] = run

    def get(self, run_id: str) -> PipelineRun | None:
        return self.runs.get(run_id)

    def latest(self, pipeline: str) -> PipelineRun | None:
        return list(self.runs.values())[-1] if self.runs else None


class Preview:
    def execute(self, definition: PipelineDefinition) -> SyncPlan:
        return SyncPlan(changed=("2",), unchanged=("1",))


class Pipeline:
    def __init__(self, runs: Runs, release: threading.Event) -> None:
        self.runs, self.release = runs, release
        self.calls: list[tuple[tuple[str, ...], str, str]] = []

    def execute(
        self,
        definition: PipelineDefinition,
        items: Any = (),
        mode: RunMode = "single_item",
        trigger: str = "cli",
        run_id: str | None = None,
    ) -> PipelineRun:
        self.calls.append((tuple(items), mode, trigger))
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


def test_preview_run_status_and_one_run_at_a_time() -> None:
    runs, release = Runs(), threading.Event()
    pipeline = Pipeline(runs, release)
    service = SyncService(DEFINITION, lambda: Preview(), lambda: pipeline, runs)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert call(base, "POST", "/sync/preview") == (
            200,
            {"new": [], "changed": ["2"], "unchanged": ["1"], "deleted": [], "is_empty": False},
        )
        status, started = call(base, "POST", "/sync/run", {"trigger": "knowledge-mcp"})
        assert status == 202
        assert call(base, "POST", "/sync/run", {})[0] == 409  # one run at a time
        release.set()
        run = wait_until_done(base, started["run_id"])
        assert run["status"] == "succeeded" and run["mode"] == "incremental"
        assert pipeline.calls[0] == ((), "incremental", "knowledge-mcp")

        _, single = call(base, "POST", "/sync/run", {"items": ["9076781"], "trigger": "publisher"})
        wait_until_done(base, single["run_id"])
        assert pipeline.calls[1] == (("9076781",), "single_item", "publisher")
    finally:
        server.shutdown()
