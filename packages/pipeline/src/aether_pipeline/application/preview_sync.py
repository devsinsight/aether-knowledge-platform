"""Use case PreviewSync: what an incremental sync would do, per source. Changes nothing.

RunPipeline in incremental mode uses the same detection (``detect_changes``), so the preview and the
run can never disagree.
"""

from __future__ import annotations

from aether_core.domain.pipeline import PipelineDefinition, is_excluded
from aether_core.domain.sync import SyncPlan, compute_plan
from aether_core.ports import SourceReader, StateStore


def detect_changes(reader: SourceReader, state: StateStore, definition: PipelineDefinition) -> SyncPlan:
    """List the source's versions (excluded subtrees left out) and compare them with the state store."""
    listed = [
        item
        for item in reader.list_versions(definition.scope)
        if not is_excluded(item.ancestor_titles, definition, item.title)
    ]
    return compute_plan(listed, state.versions(definition.name))


class PreviewSync:
    def __init__(self, reader: SourceReader, state: StateStore) -> None:
        self._reader, self._state = reader, state

    def execute(self, definition: PipelineDefinition) -> SyncPlan:
        return detect_changes(self._reader, self._state, definition)
