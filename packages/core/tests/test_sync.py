"""Sync domain: the plan is computed from versions; terms and hashes are deterministic."""

from __future__ import annotations

from aether_core.domain.contract import Authority, Chunk, Source, content_hash, normalize
from aether_core.domain.sync import ListedItem, compute_plan, embedding_hash, mentions, referenced_terms


def test_plan_classifies_new_changed_unchanged_and_deleted() -> None:
    listed = [ListedItem("1", "3"), ListedItem("2", "1"), ListedItem("4", "1")]
    plan = compute_plan(listed, {"1": "2", "2": "1", "3": "5"})
    assert (plan.new, plan.changed, plan.unchanged, plan.deleted) == (("4",), ("1",), ("2",), ("3",))
    assert plan.to_ingest == ("4", "1") and not plan.is_empty


def test_same_versions_give_an_empty_plan() -> None:
    plan = compute_plan([ListedItem("1", "2")], {"1": "2"})
    assert plan.is_empty and plan.unchanged == ("1",)


def test_referenced_terms_are_the_key_and_the_identifiers_in_the_text() -> None:
    texts = ["R-10: chain window. See ADR-005 and Q-03.", "QS-05 budget; UTF-8x is not an ID"]
    assert referenced_terms("feature-x", texts) == ("ADR-005", "Q-03", "QS-05", "R-10", "feature-x")
    assert referenced_terms(None, ["no identifiers here"]) == ()


def test_mentions_matches_whole_identifiers_only() -> None:
    assert mentions("rules R-10, R-11", "R-10")
    assert not mentions("rule R-100", "R-10")
    assert not mentions("XR-10", "R-10")


def test_embedding_hash_is_the_hash_of_the_embedded_text() -> None:
    artifact = normalize(
        artifact_id="confluence:1",
        title="Scoring",
        source=Source("confluence", "1", "https://example/1", "1"),
        authority=Authority("textual"),
        text="x",
        key=None,
        artifact_type=None,
        section=None,
        state=None,
    )
    chunk = Chunk(artifact, ("Rules", "Chains"), 0, "A chain lasts 2.0 seconds.")
    assert embedding_hash("Scoring", ("Rules", "Chains"), chunk.text) == content_hash(chunk.embedding_text)
