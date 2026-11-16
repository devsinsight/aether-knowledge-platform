"""Read-only queries of the Knowledge MCP over the projection: documents, decisions, gaps, context.

Every function takes a Qdrant client (the read-only key) and returns plain data with provenance and
state. Only ``validated`` knowledge is returned by default; a draft is named, labeled non-authoritative
and returned as a link, never as text, unless the caller asks for drafts. Nothing here writes.

Decisions and gaps are read as they are in the sources: the status of a decision comes from its page
properties, and a gap is a row of the gap register. Nothing is inferred; a field the source does not
hold is ``None``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

from qdrant_client import QdrantClient, models

DECISIONS = "decisions"
GAPS = "knowledge-gaps"
GAP_TYPES = ["knowledge-gap", "knowledge-gap-register"]

# Page properties as the wiki reader serializes them: "Status, 1 = Accepted. Date, 1 = 25 Sep 2026".
_PROPERTY = re.compile(
    r"(?:^|(?<=\. ))([A-Z][\w /-]*), \d+ = (.*?)(?=\.\s+[A-Z][\w /-]*, \d+ = |\.?\s*$)", re.M | re.S
)
# A table row whose first cell is an ID: "**Q-01**, Question = ... **Q-01**, Owner = ...".
_ROW = re.compile(
    r"\*{0,2}([A-Z]+-\d+)\*{0,2}, ([A-Za-z][\w /-]*?) = (.*?)(?=\.\s+\*{0,2}[A-Z]+-\d+\*{0,2}, |\.?\s*$)",
    re.S,
)
_SHORT_ID = re.compile(r"^([A-Z]+-\d+)\b")


def _state(payload: dict[str, Any]) -> str | None:
    return (payload.get("lifecycle") or {}).get("state")


def _allowed(include_drafts: bool) -> set[str]:
    return {"validated", "draft"} if include_drafts else {"validated"}


def _match(field: str, value: Any) -> models.FieldCondition:
    return models.FieldCondition(key=field, match=models.MatchValue(value=value))


def scroll_all(
    qdrant: QdrantClient, collection: str, query_filter: models.Filter, limit: int = 2000
) -> list[dict[str, Any]]:
    points, _ = qdrant.scroll(
        collection, scroll_filter=query_filter, limit=limit, with_payload=True, with_vectors=False
    )
    return [dict(p.payload or {}) for p in points]


def group_items(payloads: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Chunks per artifact, in document order."""
    items: dict[str, list[dict[str, Any]]] = {}
    for payload in payloads:
        items.setdefault(str(payload.get("artifact_id")), []).append(payload)
    for chunks in items.values():
        chunks.sort(key=lambda p: p.get("position", 0))
    return items


def summary(payload: dict[str, Any]) -> dict[str, Any]:
    """What identifies an item: key, type, section, owner, state and provenance."""
    state = _state(payload)
    return {
        "artifact_id": payload.get("artifact_id"),
        "title": payload.get("title"),
        "key": payload.get("key"),
        "section": payload.get("section"),
        "type": payload.get("artifact_type"),
        "owner": (payload.get("authority") or {}).get("owner"),
        "state": state,
        "authoritative": state == "validated",
        "source": payload.get("source"),
    }


def as_link(payload: dict[str, Any]) -> dict[str, Any]:
    """A draft, an image or a diagram: named and linked, its text not returned."""
    return {
        **summary(payload),
        "url": (payload.get("source") or {}).get("canonical_url"),
        "note": None if _state(payload) == "validated" else "Not validated: non-authoritative.",
    }


def document_text(chunks: Sequence[dict[str, Any]]) -> str:
    """The chunks joined in order, with a heading line wherever the heading path changes."""
    parts: list[str] = []
    previous: tuple[str, ...] | None = None
    for chunk in chunks:
        path = tuple(chunk.get("heading_path") or ())
        if path != previous and path:
            parts.append("#" * min(len(path), 6) + " " + path[-1])
        previous = path
        if chunk.get("text"):
            parts.append(str(chunk["text"]))
    return "\n\n".join(parts)


def properties(text: str) -> dict[str, str]:
    """Page properties found in a text, by lowercase name ("status", "date", "deciders")."""
    return {m.group(1).strip().lower(): m.group(2).strip() for m in _PROPERTY.finditer(text or "")}


def get_document(
    qdrant: QdrantClient,
    collection: str,
    artifact_id: str | None = None,
    key: str | None = None,
    include_drafts: bool = False,
) -> dict[str, Any]:
    if not artifact_id and not key:
        return {"error": "pass an artifact_id or a key"}
    condition = _match("artifact_id", artifact_id) if artifact_id else _match("key", key)
    items = group_items(scroll_all(qdrant, collection, models.Filter(must=[condition])))
    if not items:
        return {"error": f"not found: {artifact_id or key}"}
    documents = []
    for chunks in items.values():
        head = chunks[0]
        if _state(head) not in _allowed(include_drafts):
            documents.append({**as_link(head), "note": "Not validated: pass include_drafts to read it."})
            continue
        links = [
            {k: link.get(k) for k in ("kind", "url", "text", "heading_path", "target_id", "resolved")}
            for chunk in chunks
            for link in chunk.get("links") or []
        ]
        documents.append(
            {
                **summary(head),
                "note": None if _state(head) == "validated" else "Draft: non-authoritative.",
                "ancestor_titles": head.get("ancestor_titles") or [],
                "relationships": head.get("relationships") or [],
                "metadata_complete": head.get("metadata_complete"),
                "missing_fields": head.get("missing_fields") or [],
                "links": links,
                "text": document_text(chunks),
            }
        )
    return {"documents": documents}


def terms(key: str | None, title: str | None) -> set[str]:
    """How other items name this one: its key and the ID that starts its title (ADR-015, R-10)."""
    found = {key} if key else set()
    match = _SHORT_ID.match(title or "")
    if match:
        found.add(match.group(1))
    return found


def _mentions(text: str, wanted: set[str]) -> bool:
    return any(re.search(rf"(?<![\w-]){re.escape(t)}(?![\w-])", text) for t in wanted)


def decisions(
    qdrant: QdrantClient,
    collection: str,
    key: str | None = None,
    ranked: Sequence[str] = (),
    include_drafts: bool = False,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """The decisions of the Decisions section with their status, date and deciders as the page
    properties state them. ``key`` keeps the decisions with that key, declaring it or mentioning it;
    ``ranked`` (artifact IDs from a search) keeps and orders by those."""
    query_filter = models.Filter(must=[_match("section", DECISIONS), _match("artifact_type", "decision")])
    items = group_items(scroll_all(qdrant, collection, query_filter))
    wanted = terms(key, None) if key else set()
    result = []
    for artifact_id, chunks in items.items():
        head = chunks[0]
        if _state(head) not in _allowed(include_drafts):
            continue
        text = "\n".join(str(c.get("text") or "") for c in chunks)
        declared = {r.get("key") for r in head.get("relationships") or []}
        if key and not (head.get("key") == key or key in declared or _mentions(text, wanted)):
            continue
        if ranked and artifact_id not in ranked:
            continue
        props = properties(text)
        own = terms(None, head.get("title"))
        result.append(
            {
                **summary(head),
                "status": props.get("status"),
                "date": props.get("date"),
                "deciders": props.get("deciders"),
                # IDs the decision names (other ADRs, rules, questions), as written in its text.
                "mentions": sorted(set(re.findall(r"\b[A-Z]+-\d+\b", text)) - own),
            }
        )
    order = {a: i for i, a in enumerate(ranked)}
    result.sort(key=lambda d: (order.get(str(d["artifact_id"]), 0), str(d.get("key") or d["title"])))
    return result[:limit]


def _rows(text: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for match in _ROW.finditer(text):
        column = match.group(2).strip().lower().replace(" ", "_")
        rows.setdefault(match.group(1), {})[column] = match.group(3).strip().rstrip("|").strip()
    return rows


def gaps(
    qdrant: QdrantClient,
    collection: str,
    key: str | None = None,
    include_drafts: bool = False,
    mention: set[str] | None = None,
) -> dict[str, Any]:
    """Knowledge gaps: the rows of each gap register (status from the heading they sit under) and
    the items typed knowledge-gap. ``key`` (or ``mention``) keeps the gaps that name it."""
    query_filter = models.Filter(
        should=[_match("section", GAPS), *[_match("artifact_type", t) for t in GAP_TYPES]]
    )
    wanted = (mention or set()) | (terms(key, None) if key else set())
    registers, result = [], []
    for chunks in group_items(scroll_all(qdrant, collection, query_filter)).values():
        head = chunks[0]
        if _state(head) not in _allowed(include_drafts):
            continue
        registers.append(summary(head))
        by_status: dict[str, list[str]] = {}
        for chunk in chunks:
            path = list(chunk.get("heading_path") or [])
            status = path[-1] if len(path) > 1 else None
            text = "\n".join(
                line for line in str(chunk.get("text") or "").splitlines() if line.strip() != "| - |"
            )
            by_status.setdefault(status or "", []).append(text)
        rows_found = False
        for status, texts in by_status.items():
            for gap_id, fields in _rows("".join(texts)).items():
                rows_found = True
                gap = {
                    "id": gap_id,
                    "status": status or None,
                    **fields,
                    "register": head.get("title"),
                    "source": head.get("source"),
                    "state": _state(head),
                }
                if wanted and not (gap_id in wanted or _mentions(" ".join(fields.values()), wanted)):
                    continue
                result.append(gap)
        if not rows_found and head.get("artifact_type") == "knowledge-gap":
            text = document_text(chunks)
            if not wanted or _mentions(text, wanted | terms(head.get("key"), head.get("title"))):
                result.append({**summary(head), "status": properties(text).get("status"), "text": text})
    return {"gaps": result, "registers": registers}


def context(
    qdrant: QdrantClient,
    collection: str,
    hits: Sequence[dict[str, Any]],
    related: Any,
    limit: int = 5,
    include_drafts: bool = False,
) -> dict[str, Any]:
    """What a person or an agent needs before working on a topic: the best items for the query with
    their matching passages, the items linked to them (as links), and the decisions and gaps that
    name them. ``hits`` are search payloads in rank order; ``related(artifact_id)`` returns the
    ``knowledge.get_related`` result of one item."""
    items: dict[str, dict[str, Any]] = {}
    for payload in hits:
        artifact_id = str(payload.get("artifact_id"))
        if artifact_id not in items:
            if len(items) >= limit:
                continue
            items[artifact_id] = {**summary(payload), "passages": []}
        items[artifact_id]["passages"].append(
            {"heading_path": payload.get("heading_path") or [], "text": payload.get("text")}
        )
    names: set[str] = set()
    for item in items.values():
        names |= terms(item.get("key"), item.get("title"))

    linked: dict[str, dict[str, Any]] = {}
    for artifact_id in items:
        for entry in related(artifact_id).get("items", []):
            other = str(entry.get("artifact_id"))
            if other in items or other in linked:
                continue
            relation = next((r for r in entry.get("relations", []) if r.get("relation") != "self"), None)
            if relation is None:
                continue
            linked[other] = {
                "artifact_id": other,
                "title": entry.get("title"),
                "state": entry.get("state"),
                "authoritative": entry.get("authoritative"),
                "url": (entry.get("source") or {}).get("canonical_url"),
                "relation": relation.get("relation"),
                "from": artifact_id,
                "context": relation.get("context"),
            }

    found_decisions = [
        d
        for d in decisions(qdrant, collection, include_drafts=include_drafts, limit=1000)
        if str(d["artifact_id"]) in items or (names and (names & set(d["mentions"]) or d.get("key") in names))
    ]
    found_gaps = (
        gaps(qdrant, collection, include_drafts=include_drafts, mention=names)["gaps"] if names else []
    )
    return {
        "items": list(items.values()),
        "related": list(linked.values()),
        "decisions": found_decisions,
        "gaps": found_gaps,
    }
