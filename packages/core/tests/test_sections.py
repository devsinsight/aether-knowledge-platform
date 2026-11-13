from __future__ import annotations

from aether_core.domain.sections import section_from_ancestors


def test_nearest_section_ancestor_wins() -> None:
    titles = ("Project Aether Home", "Section 3 - Product", "04 - Features")
    assert section_from_ancestors(titles) == "product"


def test_every_section_number_maps() -> None:
    assert section_from_ancestors(("Section 0 - Catalog",)) == "catalog"
    assert section_from_ancestors(("Section 8 - Industry Extensions",)) == "industry-extensions"


def test_no_section_ancestor_is_missing() -> None:
    assert section_from_ancestors(("Project Aether Home", "Agent Workspace")) is None
    assert section_from_ancestors(()) is None
