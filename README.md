# Aether knowledge platform (Tier 2)

The semantic access platform for the Project Aether knowledge base. It follows the AI Engineering Framework (AIEF): authority stays in the sources (Confluence, Azure Boards, Miro, Scenario), and this platform builds a **rebuildable** projection of them behind a read-only Knowledge MCP and an Ops MCP.

Decision: ADR-019 *Tier 2 Knowledge Platform with the Framework Defaults* (Confluence, 08 - Architecture Decisions).

## Layout

| Path | Contents |
| --- | --- |
| `packages/core` (`aether_core`) | Domain: contract v2 (`domain/contract.py`, copied from the aief asset), pipeline model, ports. No infrastructure |
| `packages/pipeline` (`aether_pipeline`) | `application/` use cases, `adapters/` outbound adapters (LlamaIndex is the ingestion engine: DoclingNodeParser, FastEmbedEmbedding, QdrantVectorStore), `entrypoints/` CLI, webhooks, Ops MCP and the composition root |
| `packages/knowledge-mcp` (`aether_knowledge_mcp`) | Read-only Knowledge MCP; never imports the pipeline |
| `pipelines/` | Pipeline definitions (YAML); no credentials |
| `golden-set/` | Retrieval questions and expected keys |
| `work-records/` | Work record files written by agents, published by path (mounted read-only into `publisher`) |
| `deploy/` | docker compose (Qdrant and services) |

To inspect the projection, open the Qdrant Web UI at <http://localhost:6333/dashboard>. It is bundled in the Qdrant image, so `docker compose -f deploy/compose.yaml up -d qdrant` is enough.

Layer rules are enforced by import-linter (`[tool.importlinter]` in `pyproject.toml`).

## Develop

```
uv sync
uv run ruff check . && uv run ruff format --check .
uv run pyright          # on Windows with Application Control: uv run python -m pyright
uv run lint-imports
uv run pytest -q
```

Credentials come from the environment (`.env`, see `.env.example`) and use read-only accounts. Never commit them.

## Operate

One compose file runs the platform: `qdrant` (with its Web UI), `sync` (the pipeline, with an internal API), `publisher` (work records) and, on demand, `knowledge-mcp` and the `pipeline` CLI. Each service reads only its own git-ignored env file (`.env`, `.env.qdrant`, `.env.knowledge-mcp`, `.env.publisher`; see the `*.example` files): read-only source accounts in `sync`, the Qdrant read-only key in `knowledge-mcp`, wiki write access only in `publisher`.

```
docker compose -f deploy/compose.yaml up -d qdrant sync publisher
```

### Sync (manual: preview, then run)

```
docker compose -f deploy/compose.yaml exec sync aether-pipeline sync pipelines/confluence-aether.yaml --preview
docker compose -f deploy/compose.yaml exec sync aether-pipeline sync pipelines/confluence-aether.yaml
```

The preview lists new, changed and deleted pages and changes nothing. The run ingests new and changed pages with the same pipeline as a full run (same chunks and point IDs), reuses the embeddings of chunks whose embedded text did not change, deletes the points of pages deleted or archived at the source, and reports the **affected context**: for each changed or deleted page, the indexed items that reference it, by its key or by the identifiers in the text that changed (R-10, ADR-005, Q-03), with their owners. Related items are only reported, never re-embedded. From an agent: `sync.preview`, `sync.run` (after the user confirms) and `sync.status` in the Knowledge MCP.

The state store (`state/runs.sqlite`) records the last indexed version and chunk hashes of each page. It is derived data: `docker compose -f deploy/compose.yaml --profile tools run --rm pipeline state rebuild pipelines/confluence-aether.yaml` rebuilds it from the projection.

### Publish (work records)

An agent writes the work record of a story as Markdown with front matter ([template](work-records/README.md): key, type `work-record`, `story: <story key> (ADO <id>)`, story_revision, feature, state `draft`, author) into `work-records/`, then publishes it **by path**:

```
docker compose -f deploy/compose.yaml exec publisher aether-pipeline publisher publish <file>.md
docker compose -f deploy/compose.yaml exec publisher aether-pipeline publisher validate <story key> --pr <url> --by "<name>"
```

The publisher rejects a file with a missing field, an anchored work item that does not exist, a secret, or a path outside the folder, and never fills a field. It writes `Work Records / Work Records: <feature> / Work Record: <story>` as `draft` with the common record in Page Properties (publishing again adds a version to the same page), then asks `sync` for that page, so the index cites the wiki page and version. A `validated` record is never overwritten by an agent. From an agent: `work_records.publish(path)` (after the user confirms), `work_records.status(story)` and `knowledge.get_work_records(key)`, which labels drafts non-authoritative.

### Cross-source links

A page often points to knowledge held by another tool: an embedded Miro board, a link to a work item, a Scenario asset. The Confluence reader reads these references from the storage body (next to the text), with their heading path and context: the paragraph that introduces an embed, or the table row with its headers (`Ship: AX-1 Kestrel; ...; Status: Approved`). The Miro reader reads links in item text, embed items and image titles, with the frame as position. Each adapter recognizes only its own URLs (`=` and `%3D` are the same board; `moveToWidget` names an item), and the `ResolveLinks` use case relates them; readers never import each other (import-linter checks it).

Each chunk carries a `links` payload (kind, target, URL, `resolved`, context, heading path) and the `addresses` other items use to point at it. A whole-board embed relates to the board, a link that names an item to the frame that contains it. A target not indexed yet stays `resolved: false` until its source's own run completes it; a URL no adapter recognizes stays an external link and the run reports its host as a discovered source. An image without a link to its origin is reported as a finding. Nothing is copied from the target, and no source is edited. When a linked frame changes, the pages that embed its board appear in the affected context; `knowledge.get_related` returns linked items in both directions.

After adding links to readers that already existed, run a full ingest once (page versions did not change, so a sync would skip them; embeddings are reused when the text did not change):

```
docker compose -f deploy/compose.yaml exec sync aether-pipeline run pipelines/confluence-aether.yaml --all
```

Prerequisites for publishing: a **Work Records** page in the space, created by a person (its ID goes in `WORK_RECORDS_PARENT_PAGE_ID`), and a wiki account with write access limited to that tree where Confluence allows it.

## Retrieval quality

```
docker compose -f deploy/compose.yaml --profile tools run --rm pipeline eval pipelines/confluence-aether.yaml golden-set/aether.yaml --k 5
```

`golden-set/aether.yaml` is **validated** (34 questions written by an AI agent from page titles and headings, approved on 2026-11-13 by the project owner on behalf of the item owners, Azure Boards #102–#104). Results count distinct items, and search only validated content unless `--include-drafts` is given.

| Date | Corpus | Models and settings | recall@5 | MRR |
| --- | --- | --- | --- | --- |
| 2026-11-13 | 75 Confluence pages, 446 chunks | bge-base-en-v1.5 + Qdrant/bm25, RRF, HybridChunker max 512 tokens, no reranker | 1.000 | 0.897 (0.882 in the first run) |

Full results and findings: [golden-set/report-2026-11-13.md](golden-set/report-2026-11-13.md). The baseline is likely optimistic, because the questions reuse the words of the titles and headings. Add questions phrased the way people really ask, before tuning anything.

## Status

- [x] Increment 1: scaffold (workspace, contract, domain model, ports, layer contracts, CI)
- [x] Increment 2: walking skeleton (one Confluence page -> contract -> Docling chunks -> FastEmbed -> Qdrant -> `knowledge.search`). Live run on page 9076781 passed ingestion, idempotency, rebuild and validated-only search (2026-11-13), also after moving the adapters to LlamaIndex
- [x] Increment: Sync (incremental mode, preview and run, affected context, internal API, sync tools in the Knowledge MCP). Live on the AETHER space (2026-11-14): criteria 1, 2, 5 and 6 passed (ADR-019 v4: 1 of 4 chunks embedded, 3 reused); 3 and 4 are covered by tests and wait for a page archived and a rule edited at the source; Qdrant API keys not set yet
- [ ] Increment: Publish (publisher service, work record tools). Implemented; the 8 criteria pass with fakes and recorded responses. Live run waits for the Work Records page, the wiki write account and a read-only Azure Boards PAT
- [ ] Increment: Miro reader (REST API v2, one item per frame, versions for Sync, frame chunker; the sync service serves one definition per source). Implemented with a fake API; live run waits for a `boards:read` token
- [ ] Increment: Cross-source links (embeds, links and images related to their targets with their context; `knowledge.get_related`). The 8 criteria pass with fakes and the recorded Ships page; a dry run on the live Ships page attaches the board embed and the three asset rows to the right chunks. Full ingest pending
