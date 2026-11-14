"""Internal API of the ``publisher`` service, reachable only on the compose network (no host port).

POST /work-records/publish   {"path": "<file in the mounted folder>"}
POST /work-records/validate  {"story": "<story key>", "pull_request": "<url>", "validated_by": "<name>"}
GET  /work-records/<story key>
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote, urlparse

from aether_core.domain.work_record import WorkRecordRejectedError, page_title
from aether_core.ports import WikiPublisher
from aether_pipeline.application.publish_work_record import PublishWorkRecord, ValidateWorkRecord


class PublisherService:
    def __init__(
        self,
        publish: Callable[[], PublishWorkRecord],
        validate: Callable[[], ValidateWorkRecord],
        wiki: Callable[[], WikiPublisher],
    ) -> None:
        self._publish, self._validate, self._wiki = publish, validate, wiki

    def publish(self, path: str, trigger: str) -> tuple[HTTPStatus, dict[str, Any]]:
        result = self._publish().execute(path, trigger=trigger)
        status = HTTPStatus.OK if result.status == "published" else HTTPStatus.UNPROCESSABLE_ENTITY
        return status, asdict(result)

    def validate(self, story: str, pull_request: str, by: str) -> tuple[HTTPStatus, dict[str, Any]]:
        try:
            page = self._validate().execute(story, pull_request, by)
        except WorkRecordRejectedError as error:
            return HTTPStatus.UNPROCESSABLE_ENTITY, {"status": "rejected", "message": str(error)}
        return HTTPStatus.OK, {"status": "validated", "page": asdict(page)}

    def status(self, story: str) -> tuple[HTTPStatus, dict[str, Any]]:
        page = self._wiki().find_page(page_title(story))
        if page is None:
            return HTTPStatus.NOT_FOUND, {"error": f"no work record published for {story}"}
        return HTTPStatus.OK, {"page": asdict(page), "state": page.properties.get("state")}


def make_handler(service: PublisherService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - http.server naming
            path = urlparse(self.path).path
            body = self._body()
            try:
                if path == "/work-records/publish":
                    return self._send(
                        *service.publish(str(body.get("path") or ""), str(body.get("trigger") or "api"))
                    )
                if path == "/work-records/validate":
                    return self._send(
                        *service.validate(
                            str(body.get("story") or ""),
                            str(body.get("pull_request") or ""),
                            str(body.get("validated_by") or ""),
                        )
                    )
            except Exception as error:  # reported to the caller, never hidden
                return self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})
            self._send(HTTPStatus.NOT_FOUND, {"error": f"unknown path {path}"})

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            prefix = "/work-records/"
            if not path.startswith(prefix) or len(path) == len(prefix):
                return self._send(HTTPStatus.NOT_FOUND, {"error": f"unknown path {path}"})
            try:
                self._send(*service.status(unquote(path[len(prefix) :])))
            except Exception as error:
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(length) or b"{}") if length else {}
            return data if isinstance(data, dict) else {}

        def _send(self, status: HTTPStatus, data: dict[str, Any]) -> None:
            raw = json.dumps(data, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            print(f"publisher-api {self.address_string()} {format % args}", flush=True)

    return Handler


def serve(service: PublisherService, host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(service))
    print(f"publisher internal API on {host}:{port}", flush=True)
    server.serve_forever()
