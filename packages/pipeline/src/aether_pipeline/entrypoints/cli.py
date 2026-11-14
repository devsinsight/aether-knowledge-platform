"""CLI ``aether-pipeline``. Wiring is in ``wiring``; credentials come from the environment.

run <pipeline.yaml> --item <id> [--item ...]   ingest the given items
sync <pipeline.yaml> --preview                  new, changed and deleted items; changes nothing
sync <pipeline.yaml>                            apply them (incremental mode) and report affected context
state rebuild <pipeline.yaml>                   rebuild the state store from the projection
serve <pipeline.yaml> [--port 8081]             internal sync API (compose network only)
eval <pipeline.yaml> <golden-set.yaml>          recall@k and MRR
publisher publish <path>                        publish a work record file (by path) to the wiki
publisher validate <story> --pr <url> --by <name>   a person or CI approves the record
publisher status <story>                        the published record of a story
publisher serve [--port 8090]                   internal publisher API (compose network only)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from aether_core.domain.pipeline import PipelineDefinition, PipelineRun
from aether_core.domain.sync import SyncPlan
from aether_pipeline.application.evaluate_retrieval import EvaluateRetrieval, GoldenQuestion
from aether_pipeline.entrypoints import wiring


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aether-pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a pipeline for one or more items")
    run.add_argument("pipeline", type=Path)
    run.add_argument("--item", action="append", required=True, help="Source item ID (repeatable)")

    sync = sub.add_parser("sync", help="Incremental sync: preview, then run")
    sync.add_argument("pipeline", type=Path)
    sync.add_argument("--preview", action="store_true", help="Show the plan; change nothing")

    state = sub.add_parser("state", help="State store maintenance")
    state.add_argument("action", choices=["rebuild"])
    state.add_argument("pipeline", type=Path)

    serve = sub.add_parser("serve", help="Internal sync API for the compose network")
    serve.add_argument("pipeline", type=Path)
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8081)

    evaluate = sub.add_parser("eval", help="Measure retrieval with a golden set (recall@k, MRR)")
    evaluate.add_argument("pipeline", type=Path)
    evaluate.add_argument("golden_set", type=Path)
    evaluate.add_argument("--k", type=int, default=5)
    evaluate.add_argument("--include-drafts", action="store_true", help="Search drafts too")

    publisher = sub.add_parser("publisher", help="Work records: publish to the wiki, validate, serve")
    actions = publisher.add_subparsers(dest="action", required=True)
    publish = actions.add_parser("publish", help="Publish a work record file by path")
    publish.add_argument("path")
    validate = actions.add_parser("validate", help="Mark a story's work record as validated (a person or CI)")
    validate.add_argument("story", help="Story key")
    validate.add_argument("--pr", required=True, help="URL of the merged pull request")
    validate.add_argument("--by", required=True, help="Who validates")
    status = actions.add_parser("status", help="The published work record of a story")
    status.add_argument("story")
    publisher_serve = actions.add_parser("serve", help="Internal publisher API for the compose network")
    publisher_serve.add_argument("--host", default="0.0.0.0")
    publisher_serve.add_argument("--port", type=int, default=8090)

    args = parser.parse_args(argv)
    wiring.load_env(Path(".env"))
    if args.command == "publisher":
        return _publisher(args)
    definition = wiring.load_definition(args.pipeline)

    if args.command == "eval":
        return _evaluate(definition, args.golden_set, args.k, args.include_drafts)
    if args.command == "state":
        from aether_pipeline.application.rebuild_state import RebuildState

        count = RebuildState(wiring.store(), wiring.state()).execute(definition.name, definition.collection)
        print(f"state store rebuilt from collection {definition.collection!r}: {count} items")
        return 0
    if args.command == "serve":
        from aether_pipeline.entrypoints.sync_api import SyncService
        from aether_pipeline.entrypoints.sync_api import serve as serve_api

        service = SyncService(
            definition, wiring.preview_sync, lambda: wiring.run_pipeline(definition), wiring.runs()
        )
        serve_api(service, args.host, args.port)
        return 0
    if args.command == "sync" and args.preview:
        print_plan(wiring.preview_sync().execute(definition))
        return 0

    pipeline = wiring.run_pipeline(definition)
    if args.command == "sync":
        result = pipeline.execute(definition, mode="incremental", trigger="cli")
    else:
        result = pipeline.execute(definition, args.item)
    print_run(result)
    return 0 if result.status == "succeeded" else 1


def print_plan(plan: SyncPlan) -> None:
    print(
        f"plan: new={len(plan.new)} changed={len(plan.changed)} deleted={len(plan.deleted)}"
        f" unchanged={len(plan.unchanged)}"
    )
    for label, items in (("new", plan.new), ("changed", plan.changed), ("deleted", plan.deleted)):
        for item in items:
            print(f"  {label:<8} {item}")
    if plan.is_empty:
        print("  nothing to do")


def print_run(run: PipelineRun) -> None:
    print(f"run {run.run_id} ({run.mode}): {run.status}")
    totals: dict[str, list[int]] = {}
    for step in run.steps:
        total = totals.setdefault(step.step, [0, 0, 0])
        total[0] += step.processed
        total[1] += step.failed
        total[2] += step.reused
    print(f"  items={len(run.items)} deleted={len(run.deleted)}")
    for name, (processed, failed, reused) in totals.items():
        extra = f" reused={reused}" if name == "embed" else ""
        print(f"  {name:<8} processed={processed} failed={failed}{extra}")
    for error in run.errors:
        print(f"  ERROR {error.step} item={error.item}: {error.message}")
    for affected in run.affected:
        label = affected.key or affected.title
        print(f"  affected by {affected.kind} {label} (terms: {', '.join(affected.terms) or '-'}):")
        for related in affected.related:
            owner = related.owner or "no owner"
            print(
                f"    {related.key or related.artifact_id}  {related.title}  [{owner}]"
                f"  via {', '.join(related.terms)}"
            )
        if not affected.related:
            print("    no related items found")


def _publisher(args: argparse.Namespace) -> int:
    from aether_pipeline.entrypoints.publisher_api import PublisherService

    service = PublisherService(wiring.publish_work_record, wiring.validate_work_record, wiring.wiki_publisher)
    if args.action == "serve":
        from aether_pipeline.entrypoints.publisher_api import serve as serve_api

        serve_api(service, args.host, args.port)
        return 0
    if args.action == "publish":
        status, result = service.publish(args.path, trigger="cli")
    elif args.action == "validate":
        status, result = service.validate(args.story, args.pr, args.by)
    else:
        status, result = service.status(args.story)
    print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
    return 0 if status < 400 else 1


def load_golden_set(path: Path) -> list[GoldenQuestion]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        GoldenQuestion(q["question"], tuple(q["expected"]), q.get("kind", "semantic"))
        for q in data["questions"]
    ]


def _evaluate(definition: PipelineDefinition, golden_set: Path, k: int, include_drafts: bool) -> int:
    use_case = EvaluateRetrieval(wiring.embedder(), wiring.store())
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
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
