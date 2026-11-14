"""Composition root: builds the use cases from the environment. Wiring happens only in entry points.

Credentials come from the environment (or a local ``.env``), never from the pipeline definition.
Heavy adapters (Docling, FastEmbed) are imported only when a use case that needs them is built.
"""

from __future__ import annotations

import os
import sys
from functools import cache
from pathlib import Path

import yaml

from aether_core.domain.pipeline import AGENT_WORKSPACE_TITLE, PipelineDefinition
from aether_pipeline.adapters.sqlite_runs import SqliteRunRepository, SqliteStateStore

DEFAULT_DENSE = "BAAI/bge-base-en-v1.5"
DEFAULT_SPARSE = "Qdrant/bm25"


def load_env(path: Path) -> None:
    """Load KEY=VALUE lines from a local .env without overriding the real environment."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def load_definition(path: Path) -> PipelineDefinition:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PipelineDefinition(
        name=data["name"],
        source=data["source"],
        scope=dict(data.get("scope", {})),
        collection=data["collection"],
        filters=dict(data.get("filters", {})),
        exclusions=tuple(data.get("exclusions", [AGENT_WORKSPACE_TITLE])),
        schedule=data.get("schedule"),
    )


def require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"Missing environment variable {name} (see .env.example)")
    return value


def state_path() -> Path:
    return Path(os.environ.get("STATE_DB", "state/runs.sqlite"))


@cache
def runs() -> SqliteRunRepository:
    return SqliteRunRepository(state_path())


@cache
def state() -> SqliteStateStore:
    return SqliteStateStore(state_path())


def dense_model() -> str:
    return os.environ.get("DENSE_MODEL", DEFAULT_DENSE)


@cache
def store():
    from aether_pipeline.adapters.llamaindex_qdrant_store import LlamaIndexQdrantStore

    return LlamaIndexQdrantStore(
        os.environ.get("QDRANT_URL", "http://localhost:6333"), api_key=os.environ.get("QDRANT_API_KEY")
    )


@cache
def embedder():
    from aether_pipeline.adapters.llamaindex_embedder import LlamaIndexFastEmbedEmbedder

    return LlamaIndexFastEmbedEmbedder(dense_model(), os.environ.get("SPARSE_MODEL", DEFAULT_SPARSE))


@cache
def reader():
    from aether_pipeline.adapters.confluence import ConfluenceReader

    return ConfluenceReader(
        require("CONFLUENCE_BASE_URL"), require("CONFLUENCE_EMAIL"), require("CONFLUENCE_API_TOKEN")
    )


def run_pipeline(definition: PipelineDefinition):
    from aether_pipeline.adapters.llamaindex_chunker import LlamaIndexDoclingChunker
    from aether_pipeline.application.run_pipeline import RunPipeline

    chunker = LlamaIndexDoclingChunker(
        tokenizer_model=dense_model(), max_tokens=int(definition.filters.get("max_tokens", 512))
    )
    return RunPipeline(reader(), chunker, embedder(), store(), runs(), state())


def preview_sync():
    from aether_pipeline.application.preview_sync import PreviewSync

    return PreviewSync(reader(), state())


# Publisher: the only service with wiki write access (an account limited to the Work Records tree).


@cache
def wiki_publisher():
    from aether_pipeline.adapters.confluence_publisher import ConfluenceWikiPublisher

    return ConfluenceWikiPublisher(
        require("CONFLUENCE_BASE_URL"),
        require("CONFLUENCE_WRITE_EMAIL"),
        require("CONFLUENCE_WRITE_API_TOKEN"),
        os.environ.get("CONFLUENCE_SPACE", "AETHER"),
    )


@cache
def sync_requester():
    from aether_pipeline.adapters.sync_client import HttpSyncRequester

    return HttpSyncRequester(os.environ.get("SYNC_URL", "http://sync:8081"))


def publish_work_record():
    from aether_pipeline.adapters.azure_devops import AzureDevOpsWorkItems
    from aether_pipeline.adapters.secret_scanner import RegexSecretScanner
    from aether_pipeline.adapters.work_record_files import MountedWorkRecordFiles
    from aether_pipeline.application.publish_work_record import PublishWorkRecord

    return PublishWorkRecord(
        files=MountedWorkRecordFiles(Path(os.environ.get("WORK_RECORDS_DIR", "work-records"))),
        wiki=wiki_publisher(),
        work_items=AzureDevOpsWorkItems(
            os.environ.get("ADO_ORG_URL", "https://dev.azure.com/devsinsight"),
            os.environ.get("ADO_PROJECT", "Aether"),
            require("ADO_READ_PAT"),
        ),
        scanner=RegexSecretScanner(),
        sync=sync_requester(),
        runs=runs(),
        parent_page_id=require("WORK_RECORDS_PARENT_PAGE_ID"),
    )


def validate_work_record():
    from aether_pipeline.application.publish_work_record import ValidateWorkRecord

    return ValidateWorkRecord(wiki_publisher(), sync_requester())
