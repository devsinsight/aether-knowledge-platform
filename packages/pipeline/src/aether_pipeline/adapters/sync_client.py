"""Client of the sync service's internal API, used by the publisher to sync the page it wrote."""

from __future__ import annotations

from collections.abc import Sequence

import httpx


class HttpSyncRequester:
    def __init__(self, sync_url: str, client: httpx.Client | None = None) -> None:
        self._url = sync_url.rstrip("/")
        self._client = client or httpx.Client(timeout=30.0)

    def sync_items(self, item_ids: Sequence[str], trigger: str) -> str:
        response = self._client.post(
            f"{self._url}/sync/run", json={"items": list(item_ids), "trigger": trigger}
        )
        response.raise_for_status()
        return str(response.json()["run_id"])
