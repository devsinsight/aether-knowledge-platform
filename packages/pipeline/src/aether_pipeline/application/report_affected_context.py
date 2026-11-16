"""Use case ReportAffectedContext: who else is touched by a change (platform.md, "Synchronization").

For each changed or deleted item, the projection is queried for the items that reference it: by its
key, and by the identifiers in the text that changed (rule, decision and gap IDs such as R-10 or
ADR-005, which is how the Aether pages reference each other), and by cross-source links: the items
whose ``links`` point to it (a page that embeds a board is affected when a frame of the board changes,
although the page's own version did not change). Related items are only reported, with their owners;
they are never re-embedded or changed.
"""

from __future__ import annotations

from collections.abc import Sequence

from aether_core.domain.sync import AffectedContext, Change, RelatedItem, mentions, referenced_terms
from aether_core.ports import VectorStore


class ReportAffectedContext:
    def __init__(self, store: VectorStore) -> None:
        self._store = store

    def execute(self, collection: str, changes: Sequence[Change]) -> tuple[AffectedContext, ...]:
        return tuple(self._one(collection, change) for change in changes)

    def _one(self, collection: str, change: Change) -> AffectedContext:
        terms = referenced_terms(change.key, change.texts)
        found: dict[str, tuple[dict[str, object], set[str]]] = {}
        for term in terms:
            for payload in self._store.find_mentions(collection, term):
                artifact_id = str(payload.get("artifact_id") or "")
                if not artifact_id or artifact_id == change.artifact_id:
                    continue
                if not mentions(str(payload.get("text") or ""), term):
                    continue
                found.setdefault(artifact_id, (payload, set()))[1].add(term)
        for _, payload in self._store.find_linking(collection, list(change.addresses)):
            artifact_id = str(payload.get("artifact_id") or "")
            if not artifact_id or artifact_id == change.artifact_id:
                continue
            targets = {
                str(link.get("target_item") or link.get("target_id"))
                for link in payload.get("links") or []
                if {link.get("target_id"), link.get("target_item")} & set(change.addresses)
            }
            found.setdefault(artifact_id, (payload, set()))[1].update(f"link:{t}" for t in targets)
        related = tuple(
            RelatedItem(
                artifact_id=artifact_id,
                key=_str(payload.get("key")),
                title=str(payload.get("title") or ""),
                owner=_owner(payload),
                terms=tuple(sorted(matched)),
            )
            for artifact_id, (payload, matched) in sorted(found.items())
        )
        return AffectedContext(change.item_id, change.kind, change.key, change.title, terms, related)


def _str(value: object) -> str | None:
    return str(value) if value else None


def _owner(payload: dict[str, object]) -> str | None:
    authority = payload.get("authority")
    return _str(authority.get("owner")) if isinstance(authority, dict) else None
