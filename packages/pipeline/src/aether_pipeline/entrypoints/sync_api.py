"""Internal API of the ``sync`` service, reachable only on the compose network (no host port).

    POST /sync/preview {"pipeline"?}                    -> the SyncPlan of each source; changes nothing
    POST /sync/run {"items"?, "pipeline"?, "trigger"?}  -> an incremental sync of every source (or one),
                                                           or a run of the given items of one source
    GET  /sync/status[?run_id=]                         -> the run, or the latest run of each source

One sync at a time: a second run while one is active gets 409. Sources run one after the other in a
background thread, one PipelineRun each, so callers (the Knowledge MCP, the publisher) poll
``/sync/status``. A source that fails (for example, a missing credential) is reported as a failed run
and never stops the others.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol
from urllib.parse import parse_qs, urlparse

from aether_core.domain.pipeline import PipelineDefinition, PipelineRun, RunError, RunMode
from aether_core.domain.sync import SyncPlan
from aether_core.ports import RunRepository


class _Preview(Protocol):
    def execute(self, definition: PipelineDefinition) -> SyncPlan: ...


class _Pipeline(Protocol):
    def execute(
        self,
        definition: PipelineDefinition,
        items: Any = (),
        mode: RunMode = "single_item",
        trigger: str = "cli",
        run_id: str | None = None,
    ) -> PipelineRun: ...


class SyncBusyError(RuntimeError):
    pass


class UnknownPipelineError(LookupError):
    pass


def plan_json(plan: SyncPlan) -> dict[str, Any]:
    return {**asdict(plan), "is_empty": plan.is_empty}


def run_json(run: PipelineRun) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(run), default=str))


class SyncService:
    """Sync of every pipeline definition the service was started with; the first one is the wiki."""

    def __init__(
        self,
        definitions: Sequence[PipelineDefinition],
        preview: Callable[[PipelineDefinition], _Preview],
        pipeline: Callable[[PipelineDefinition], _Pipeline],
        runs: RunRepository,
    ) -> None:
        if not definitions:
            raise ValueError("at least one pipeline definition is required")
        self._definitions = tuple(definitions)
        self._preview, self._pipeline, self._runs = preview, pipeline, runs
        self._lock = threading.Lock()
        self._pending: dict[str, str] = {}  # run ID -> pipeline, for runs queued or running

    def _select(self, pipeline: str | None) -> tuple[PipelineDefinition, ...]:
        if pipeline is None:
            return self._definitions
        found = tuple(d for d in self._definitions if d.name == pipeline)
        if not found:
            raise UnknownPipelineError(f"unknown pipeline {pipeline!r}")
        return found

    def preview(self, pipeline: str | None = None) -> dict[str, Any]:
        plans: list[dict[str, Any]] = []
        for definition in self._select(pipeline):
            try:
                plan = self._preview(definition).execute(definition)
                plans.append({"pipeline": definition.name, **plan_json(plan)})
            except Exception as error:  # one failing source is reported, never hides the others
                plans.append({"pipeline": definition.name, "error": str(error)})
        return {"pipelines": plans, "is_empty": all(p.get("is_empty", False) for p in plans)}

    def start(self, items: list[str] | None, trigger: str, pipeline: str | None = None) -> dict[str, Any]:
        selected = self._select(pipeline)
        if items:
            selected = selected[:1]  # items belong to one source: the named one, or the wiki
        if not self._lock.acquire(blocking=False):
            raise SyncBusyError(f"runs {sorted(self._pending)} are still running")
        run_ids = {d.name: str(uuid.uuid4()) for d in selected}
        self._pending = {run_id: name for name, run_id in run_ids.items()}
        mode: RunMode = "single_item" if items else "incremental"
        jobs = [(d, run_ids[d.name]) for d in selected]
        thread = threading.Thread(target=self._run, args=(jobs, items or [], mode, trigger), daemon=True)
        thread.start()
        result: dict[str, Any] = {"run_ids": run_ids}
        if len(run_ids) == 1:
            result["run_id"] = next(iter(run_ids.values()))
        return result

    def _run(
        self, jobs: list[tuple[PipelineDefinition, str]], items: list[str], mode: RunMode, trigger: str
    ) -> None:
        try:
            for definition, run_id in jobs:
                try:
                    self._pipeline(definition).execute(
                        definition, items, mode=mode, trigger=trigger, run_id=run_id
                    )
                except Exception as error:  # e.g. the source cannot be built: record a failed run
                    failed = PipelineRun(run_id, definition.name, mode, trigger, items=tuple(items)).start()
                    self._runs.save(failed.fail_item(RunError("extract", "*", str(error), False)).finish())
                finally:
                    self._pending.pop(run_id, None)
        finally:
            self._pending = {}
            self._lock.release()

    def status(self, run_id: str | None) -> dict[str, Any] | None:
        if run_id:
            run = self._runs.get(run_id)
            if run is None and run_id in self._pending:
                return {
                    "run_id": run_id,
                    "pipeline": self._pending[run_id],
                    "status": "queued",
                    "active": True,
                }
            return {**run_json(run), "active": run_id in self._pending} if run else None
        latest = [self._runs.latest(d.name) for d in self._definitions]
        found = [{**run_json(r), "active": r.run_id in self._pending} for r in latest if r]
        return {"runs": found} if found else None


def make_handler(service: SyncService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - http.server naming
            path = urlparse(self.path).path
            body = self._body()
            pipeline = str(body["pipeline"]) if body.get("pipeline") else None
            try:
                if path == "/sync/preview":
                    return self._send(HTTPStatus.OK, service.preview(pipeline))
                if path == "/sync/run":
                    items = [str(i) for i in body.get("items") or []]
                    started = service.start(items or None, str(body.get("trigger") or "api"), pipeline)
                    return self._send(HTTPStatus.ACCEPTED, started)
            except SyncBusyError as error:
                return self._send(HTTPStatus.CONFLICT, {"error": str(error)})
            except UnknownPipelineError as error:
                return self._send(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except Exception as error:  # reported to the caller, never hidden
                return self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})
            self._send(HTTPStatus.NOT_FOUND, {"error": f"unknown path {path}"})

        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            if url.path != "/sync/status":
                return self._send(HTTPStatus.NOT_FOUND, {"error": f"unknown path {url.path}"})
            run_id = parse_qs(url.query).get("run_id", [None])[0]
            status = service.status(run_id)
            if status is None:
                return self._send(HTTPStatus.NOT_FOUND, {"error": "no run found"})
            self._send(HTTPStatus.OK, status)

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            data = json.loads(self.rfile.read(length) or b"{}")
            return data if isinstance(data, dict) else {}

        def _send(self, status: HTTPStatus, data: dict[str, Any]) -> None:
            raw = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            print(f"sync-api {self.address_string()} {format % args}", flush=True)

    return Handler


def serve(service: SyncService, host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(service))
    print(f"sync internal API on {host}:{port}", flush=True)
    server.serve_forever()
