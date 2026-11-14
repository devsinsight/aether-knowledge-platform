"""Internal API of the ``sync`` service, reachable only on the compose network (no host port).

    POST /sync/preview            -> the SyncPlan; changes nothing
    POST /sync/run   {"items": [...]}  -> starts an incremental sync, or a run of the given items
    GET  /sync/status[?run_id=]   -> the run (the latest when no ID is given)

One run at a time: a second run while one is active gets 409. Runs execute in a background thread,
so callers (the Knowledge MCP, the publisher) poll ``/sync/status``.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol
from urllib.parse import parse_qs, urlparse

from aether_core.domain.pipeline import PipelineDefinition, PipelineRun, RunMode
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


def plan_json(plan: SyncPlan) -> dict[str, Any]:
    return {**asdict(plan), "is_empty": plan.is_empty}


def run_json(run: PipelineRun) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(run), default=str))


class SyncService:
    def __init__(
        self,
        definition: PipelineDefinition,
        preview: Callable[[], _Preview],
        pipeline: Callable[[], _Pipeline],
        runs: RunRepository,
    ) -> None:
        self._definition = definition
        self._preview, self._pipeline, self._runs = preview, pipeline, runs
        self._lock = threading.Lock()
        self._active: str | None = None

    def preview(self) -> dict[str, Any]:
        return plan_json(self._preview().execute(self._definition))

    def start(self, items: list[str] | None, trigger: str) -> str:
        if not self._lock.acquire(blocking=False):
            raise SyncBusyError(f"run {self._active} is still running")
        run_id = str(uuid.uuid4())
        self._active = run_id
        mode: RunMode = "single_item" if items else "incremental"
        thread = threading.Thread(target=self._run, args=(run_id, items or [], mode, trigger), daemon=True)
        thread.start()
        return run_id

    def _run(self, run_id: str, items: list[str], mode: RunMode, trigger: str) -> None:
        try:
            self._pipeline().execute(self._definition, items, mode=mode, trigger=trigger, run_id=run_id)
        finally:
            self._active = None
            self._lock.release()

    def status(self, run_id: str | None) -> dict[str, Any] | None:
        run = self._runs.get(run_id) if run_id else self._runs.latest(self._definition.name)
        return {**run_json(run), "active": run.run_id == self._active} if run else None


def make_handler(service: SyncService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - http.server naming
            path = urlparse(self.path).path
            body = self._body()
            try:
                if path == "/sync/preview":
                    return self._send(HTTPStatus.OK, service.preview())
                if path == "/sync/run":
                    items = [str(i) for i in body.get("items") or []]
                    run_id = service.start(items or None, str(body.get("trigger") or "api"))
                    return self._send(HTTPStatus.ACCEPTED, {"run_id": run_id})
            except SyncBusyError as error:
                return self._send(HTTPStatus.CONFLICT, {"error": str(error)})
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
