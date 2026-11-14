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
- [ ] Increment 2: walking skeleton (one Confluence page -> contract -> Docling chunks -> FastEmbed -> Qdrant -> `knowledge.search`). Live run on page 9076781 passed ingestion, idempotency, rebuild and validated-only search (2026-11-13), also after moving the adapters to LlamaIndex
