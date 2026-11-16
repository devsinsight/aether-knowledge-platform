"""Knowledge MCP queries over an in-memory Qdrant: documents, decisions, gaps and context.

Payloads follow what the pipeline stores, with texts as Docling serializes the real pages (page
properties as "Status, 1 = Accepted", table rows as "**Q-07**, Owner = ...").
"""

from __future__ import annotations

from typing import Any

import pytest
from qdrant_client import QdrantClient, models

from aether_knowledge_mcp import queries

C = "aether"


def chunk(
    artifact_id: str,
    position: int,
    text: str,
    heading_path: list[str],
    *,
    title: str,
    key: str | None = None,
    section: str = "technical",
    artifact_type: str = "architecture",
    state: str = "validated",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "title": title,
        "key": key,
        "section": section,
        "artifact_type": artifact_type,
        "lifecycle": {"state": state},
        "authority": {"owner": "Daniel Okafor"},
        "source": {
            "provider": artifact_id.split(":")[0],
            "canonical_url": f"https://x/{artifact_id}",
            "version": "3",
        },
        "position": position,
        "heading_path": heading_path,
        "text": text,
        **extra,
    }


ADR15 = "ADR-015 Local Development with Docker Compose and LocalStack"
ADR11 = "ADR-011 AWS with CDK Replaces Azure Hosting"
GAPS = "10 - Knowledge Gaps"
PAYLOADS = [
    chunk(
        "confluence:15",
        1,
        "Every developer must run the whole game locally.",
        [ADR15, "Context"],
        title=ADR15,
        key="decision-adr-015",
        section="decisions",
        artifact_type="decision",
    ),
    chunk(
        "confluence:15",
        0,
        "Status, 1 = Accepted. Date, 1 = 01 Nov 2026. Deciders, 1 = Daniel Okafor, Mei Tanaka",
        [ADR15],
        title=ADR15,
        key="decision-adr-015",
        section="decisions",
        artifact_type="decision",
    ),
    chunk(
        "confluence:11",
        0,
        "Status, 1 = Accepted. Supersedes ADR-004. Amends ADR-006 (the validation queue is SQS).. "
        "Date, 1 = 01 Nov 2026",
        [ADR11],
        title=ADR11,
        key="decision-adr-011",
        section="decisions",
        artifact_type="decision",
    ),
    chunk(
        "confluence:19",
        0,
        "Status, 1 = Proposed",
        ["ADR-019 Tier 2"],
        title="ADR-019 Tier 2",
        key="decision-adr-019",
        section="decisions",
        artifact_type="decision",
        state="draft",
    ),
    chunk(
        "confluence:10",
        1,
        "| - |\n**Q-06**, Question = What does prod cost per month on AWS?. **Q-06**, Owner = Daniel Okafor. "
        "**Q-07**, Question = Does Nebula Forge need a paid LocalStack plan? If not affordable, do we switch",
        [GAPS, "Open"],
        title=GAPS,
        section="knowledge-gaps",
        artifact_type="knowledge-gap-register",
    ),
    chunk(
        "confluence:10",
        2,
        "| - |\n to Floci?. **Q-07**, Why it matters = Every developer depends on it (ADR-015). "
        "**Q-07**, Owner = Daniel Okafor. **Q-07**, Needed by = M1 |",
        [GAPS, "Open"],
        title=GAPS,
        section="knowledge-gaps",
        artifact_type="knowledge-gap-register",
    ),
    chunk(
        "confluence:10",
        3,
        "None yet.",
        [GAPS, "Resolved"],
        title=GAPS,
        section="knowledge-gaps",
        artifact_type="knowledge-gap-register",
    ),
    chunk(
        "miro:b=:f1",
        0,
        "Local stack diagram",
        [],
        title="Board - Local Stack",
        section="technical",
        artifact_type="diagram",
        state="draft",
        links=[],
        addresses=["miro:b=:f1", "miro:b="],
    ),
    chunk(
        "confluence:7",
        0,
        "The local stack runs in Docker Compose.",
        ["Local Development"],
        title="Local Development",
        key="technical-local-dev",
        addresses=["confluence:7"],
        links=[
            {
                "kind": "embed",
                "url": "https://miro.com/app/board/b=/",
                "text": "",
                "heading_path": ["Local Development"],
                "target_id": "miro:b=",
                "target_item": None,
                "resolved": True,
                "context": "The diagram.",
            }
        ],
    ),
]


@pytest.fixture
def qdrant() -> QdrantClient:
    client = QdrantClient(":memory:")
    client.create_collection(C, vectors_config={})
    client.upsert(C, [models.PointStruct(id=n, vector={}, payload=p) for n, p in enumerate(PAYLOADS)])
    return client


def test_get_document_returns_the_text_in_order_with_its_provenance(qdrant: QdrantClient) -> None:
    (doc,) = queries.get_document(qdrant, C, key="decision-adr-015")["documents"]
    assert doc["text"].index("Status, 1 = Accepted") < doc["text"].index("## Context")
    assert doc["owner"] == "Daniel Okafor" and doc["source"]["version"] == "3" and doc["authoritative"]


def test_get_document_names_a_draft_without_its_text_unless_asked(qdrant: QdrantClient) -> None:
    (frame,) = queries.get_document(qdrant, C, artifact_id="miro:b=:f1")["documents"]
    assert "text" not in frame and frame["url"] == "https://x/miro:b=:f1" and frame["authoritative"] is False
    (read,) = queries.get_document(qdrant, C, artifact_id="miro:b=:f1", include_drafts=True)["documents"]
    assert read["text"] == "Local stack diagram" and read["note"]
    assert "error" in queries.get_document(qdrant, C) and "error" in queries.get_document(
        qdrant, C, key="nope"
    )


def test_decisions_carry_status_date_and_deciders_from_their_page_properties(qdrant: QdrantClient) -> None:
    by_key = {d["key"]: d for d in queries.decisions(qdrant, C)}
    assert set(by_key) == {"decision-adr-011", "decision-adr-015"}  # the draft ADR-019 stays out
    assert by_key["decision-adr-015"]["status"] == "Accepted"
    assert by_key["decision-adr-015"]["deciders"] == "Daniel Okafor, Mei Tanaka"
    assert by_key["decision-adr-011"]["status"].startswith("Accepted. Supersedes ADR-004")
    assert by_key["decision-adr-011"]["mentions"] == ["ADR-004", "ADR-006"]
    assert [d["key"] for d in queries.decisions(qdrant, C, include_drafts=True)][-1] == "decision-adr-019"


def test_decisions_by_key_and_by_rank(qdrant: QdrantClient) -> None:
    assert [d["key"] for d in queries.decisions(qdrant, C, key="ADR-004")] == ["decision-adr-011"]
    ranked = queries.decisions(qdrant, C, ranked=["confluence:15", "confluence:11"])
    assert [d["key"] for d in ranked] == ["decision-adr-015", "decision-adr-011"]


def test_gaps_are_the_register_rows_even_across_chunks(qdrant: QdrantClient) -> None:
    result = queries.gaps(qdrant, C)
    by_id = {g["id"]: g for g in result["gaps"]}
    assert set(by_id) == {"Q-06", "Q-07"}
    assert by_id["Q-07"]["question"].endswith("do we switch to Floci?")
    assert by_id["Q-07"]["needed_by"] == "M1"  # the table's closing pipe is not part of the value
    assert by_id["Q-07"]["status"] == "Open" and by_id["Q-07"]["register"] == GAPS
    assert [r["title"] for r in result["registers"]] == [GAPS]
    assert [g["id"] for g in queries.gaps(qdrant, C, key="ADR-015")["gaps"]] == ["Q-07"]
    assert [g["id"] for g in queries.gaps(qdrant, C, key="Q-06")["gaps"]] == ["Q-06"]


def test_context_joins_passages_links_decisions_and_gaps(qdrant: QdrantClient) -> None:
    hits = [PAYLOADS[0], PAYLOADS[8], PAYLOADS[1]]

    def related(artifact_id: str) -> dict[str, Any]:
        if artifact_id != "confluence:7":
            return {"items": []}
        return {
            "items": [
                {
                    "artifact_id": "miro:b=:f1",
                    "title": "Board - Local Stack",
                    "state": "draft",
                    "authoritative": False,
                    "source": PAYLOADS[7]["source"],
                    "relations": [{"relation": "linked_from_this", "context": "The diagram."}],
                }
            ]
        }

    result = queries.context(qdrant, C, hits, related, limit=5)
    assert [i["artifact_id"] for i in result["items"]] == ["confluence:15", "confluence:7"]
    assert len(result["items"][0]["passages"]) == 2
    (link,) = result["related"]
    assert link["url"] == "https://x/miro:b=:f1" and "text" not in link and link["from"] == "confluence:7"
    assert [d["key"] for d in result["decisions"]] == ["decision-adr-015"]
    assert [g["id"] for g in result["gaps"]] == ["Q-07"]  # it names ADR-015
