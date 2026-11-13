"""Section from the page tree (technology.md, metadata default): "Section N - ..." pages map to sections."""

from __future__ import annotations

import re

from aether_core.domain.contract import SECTIONS

_SECTION_TITLE = re.compile(r"^Section ([0-8]) - ")


def section_from_ancestors(titles: tuple[str, ...]) -> str | None:
    """Return the section of the nearest "Section N - ..." ancestor, or None when there is none."""
    for title in reversed(titles):
        match = _SECTION_TITLE.match(title)
        if match:
            return SECTIONS[int(match.group(1))]
    return None
