"""CLI and composition root: ``aether-pipeline run <pipeline.yaml> --item <page-id> [--item ...]``.

Wiring happens only here. Credentials come from the environment (or a local ``.env``), never from
the pipeline definition.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

from aether_core.domain.pipeline import AGENT_WORKSPACE_TITLE, PipelineDefinition
from aether_pipeline.application.evaluate_retrieval import EvaluateRetrieval, GoldenQuestion


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


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"Missing environment variable {name} (see .env.example)")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aether-pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run a pipeline for one or more items")
    run.add_argument("pipeline", type=Path)
    run.add_argument("--item", action="append", required=True, help="Source item ID (repeatable)")
    evaluate = sub.add_parser("eval", help="Measure retrieval with a golden set (recall@k, MRR)")
    evaluate.add_argument("pipeline", type=Path)
    evaluate.add_argument("golden_set", type=Path)
    evaluate.add_argument("--k", type=int, default=5)
    evaluate.add_argument("--include-drafts", action="store_true", help="Search drafts too")
    args = parser.parse_args(argv)

    load_env(Path(".env"))
    definition = load_definition(args.pipeline)
    if args.command == "eval":
        return _evaluate(definition, args.golden_set, args.k, args.include_drafts)

    # Heavy adapters are imported only when a run is wired.
    from aether_pipeline.adapters.confluence import ConfluenceReader
    from aether_pipeline.adapters.llamaindex_chunker import LlamaIndexDoclingChunker
    from aether_pipeline.adapters.llamaindex_embedder import LlamaIndexFastEmbedEmbedder
    from aether_pipeline.adapters.llamaindex_qdrant_store import LlamaIndexQdrantStore
    from aether_pipeline.adapters.sqlite_runs import SqliteRunRepository
    from aether_pipeline.application.run_pipeline import RunPipeline

    settings = definition.filters
    dense = os.environ.get("DENSE_MODEL", "BAAI/bge-base-en-v1.5")
    use_case = RunPipeline(
        reader=ConfluenceReader(
            _require("CONFLUENCE_BASE_URL"), _require("CONFLUENCE_EMAIL"), _require("CONFLUENCE_API_TOKEN")
        ),
        chunker=LlamaIndexDoclingChunker(
            tokenizer_model=dense, max_tokens=int(settings.get("max_tokens", 512))
        ),
        embedder=LlamaIndexFastEmbedEmbedder(dense, os.environ.get("SPARSE_MODEL", "Qdrant/bm25")),
        store=LlamaIndexQdrantStore(os.environ.get("QDRANT_URL", "http://localhost:6333")),
        runs=SqliteRunRepository(Path(os.environ.get("STATE_DB", "state/runs.sqlite"))),
    )
    result = use_case.execute(definition, args.item)

    print(f"run {result.run_id}: {result.status}")
    for step in result.steps:
        print(f"  {step.step:<8} processed={step.processed} failed={step.failed}")
    for error in result.errors:
        print(f"  ERROR {error.step} item={error.item}: {error.message}")
    return 0 if result.status == "succeeded" else 1


def load_golden_set(path: Path) -> list[GoldenQuestion]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        GoldenQuestion(q["question"], tuple(q["expected"]), q.get("kind", "semantic"))
        for q in data["questions"]
    ]


def _evaluate(definition: PipelineDefinition, golden_set: Path, k: int, include_drafts: bool) -> int:
    from aether_pipeline.adapters.llamaindex_embedder import LlamaIndexFastEmbedEmbedder
    from aether_pipeline.adapters.llamaindex_qdrant_store import LlamaIndexQdrantStore

    dense = os.environ.get("DENSE_MODEL", "BAAI/bge-base-en-v1.5")
    use_case = EvaluateRetrieval(
        LlamaIndexFastEmbedEmbedder(dense, os.environ.get("SPARSE_MODEL", "Qdrant/bm25")),
        LlamaIndexQdrantStore(os.environ.get("QDRANT_URL", "http://localhost:6333")),
    )
    filters = None if include_drafts else {"lifecycle.state": ["validated"]}
    report = use_case.execute(definition.collection, load_golden_set(golden_set), k, filters)
    for r in report.results:
        mark = f"#{r.rank}" if r.rank else "MISS"
        print(f"{mark:<5} [{r.question.kind}] {r.question.question}")
        if not r.rank:
            print(f"      expected {list(r.question.expected)}; got {list(r.returned)}")
    print(f"questions={len(report.results)} recall@{k}={report.recall_at_k:.3f} MRR={report.mrr:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
