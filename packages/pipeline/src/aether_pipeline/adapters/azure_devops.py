"""Azure Boards work item lookup (read-only): confirms that a work record's story exists."""

from __future__ import annotations

import httpx


class AzureDevOpsWorkItems:
    def __init__(self, org_url: str, project: str, pat: str, client: httpx.Client | None = None) -> None:
        self._base = f"{org_url.rstrip('/')}/{project}/_apis/wit/workitems"
        self._client = client or httpx.Client(auth=("", pat), timeout=30.0)

    def exists(self, work_item_id: int) -> bool:
        response = self._client.get(
            f"{self._base}/{work_item_id}", params={"fields": "System.Id", "api-version": "7.1"}
        )
        if response.status_code == 404:
            return False
        response.raise_for_status()
        return True
