"""Internal publisher API over HTTP with fake use cases: publish by path, validate, status."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from typing import Any

from aether_core.domain.work_record import WorkRecordRejectedError
from aether_core.ports import PublishedPage
from aether_pipeline.application.publish_work_record import PublishResult
from aether_pipeline.entrypoints.publisher_api import PublisherService, make_handler

PAGE = PublishedPage("42", 1, "Work Record: story-chain-window", "u/42", {"state": "draft"})


class Publish:
    def __init__(self) -> None:
        self.paths: list[str] = []

    def execute(self, path: str, trigger: str = "cli") -> PublishResult:
        self.paths.append(path)
        if path == "bad.md":
            return PublishResult("r2", "rejected", "missing required front matter: story_revision")
        return PublishResult("r1", "published", page=PAGE, created=True, sync_run_id="s1")


class Validate:
    def execute(self, story_key: str, pull_request: str, validated_by: str) -> PublishedPage:
        if not pull_request:
            raise WorkRecordRejectedError("validation needs the pull request and who validates")
        return PAGE


class Wiki:
    def find_page(self, title: str) -> PublishedPage | None:
        return PAGE if title == PAGE.title else None


def call(base: str, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    data = json.dumps(body).encode() if body is not None else None
    try:
        with urllib.request.urlopen(
            urllib.request.Request(base + path, data=data, method=method), timeout=5
        ) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_publish_validate_and_status() -> None:
    publish = Publish()
    service = PublisherService(lambda: publish, lambda: Validate(), lambda: Wiki())  # type: ignore[arg-type,return-value]
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        status, result = call(base, "POST", "/work-records/publish", {"path": "chain.md"})
        assert status == 200 and result["status"] == "published" and result["page"]["page_id"] == "42"
        assert publish.paths == ["chain.md"]
        assert call(base, "POST", "/work-records/publish", {"path": "bad.md"})[0] == 422

        ok = {
            "story": "story-chain-window",
            "pull_request": "https://github.com/x/pull/1",
            "validated_by": "L",
        }
        assert call(base, "POST", "/work-records/validate", ok)[0] == 200
        assert call(base, "POST", "/work-records/validate", {**ok, "pull_request": ""})[0] == 422

        status, found = call(base, "GET", "/work-records/story-chain-window")
        assert status == 200 and found["state"] == "draft"
        assert call(base, "GET", "/work-records/unknown")[0] == 404
    finally:
        server.shutdown()
