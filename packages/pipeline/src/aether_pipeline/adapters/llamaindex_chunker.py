"""Chunking by structure through LlamaIndex: DoclingNodeParser with Docling HybridChunker.

LlamaIndex is the ingestion engine (platform.md) and lives only in outbound adapters. Docling converts the
page HTML into a DoclingDocument; LlamaIndex's DoclingNodeParser chunks it with HybridChunker, which gets
the tokenizer of the dense model so chunks fit its token limit. The heading path of each chunk is kept as
deterministic context; there is no overlap. Nodes are mapped back to contract chunks, so point IDs stay
deterministic (contract v2) and never come from LlamaIndex's random node IDs.

Docling's HTML backend drops everything before the first heading when the page has no ``<h1>`` (a
status table or an owner line at the top of a page). The page title is added as the ``<h1>`` in that
case, so nothing is lost and every chunk's heading path starts with the title.
"""

from __future__ import annotations

import html
import re
from collections.abc import Sequence
from io import BytesIO

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from llama_index.core import Document
from llama_index.node_parser.docling import DoclingNodeParser

from aether_core.domain.contract import Chunk, NormalizedKnowledgeArtifact

_H1 = re.compile(r"<h1[\s>]", re.I)


def with_title(body: str, title: str) -> str:
    """The page body with its title as the ``<h1>`` when it has none."""
    if _H1.search(body) or not title:
        return body
    return f"<h1>{html.escape(title)}</h1>{body}"


class LlamaIndexDoclingChunker:
    def __init__(self, tokenizer_model: str, max_tokens: int) -> None:
        tokenizer = HuggingFaceTokenizer.from_pretrained(tokenizer_model, max_tokens=max_tokens)
        self._parser = DoclingNodeParser(chunker=HybridChunker(tokenizer=tokenizer))
        self._converter = DocumentConverter(allowed_formats=[InputFormat.HTML])

    def chunk(self, artifact: NormalizedKnowledgeArtifact) -> Sequence[Chunk]:
        stream = DocumentStream(
            name=f"{artifact.source.provider_artifact_id}.html",
            stream=BytesIO(with_title(artifact.text, artifact.title).encode("utf-8")),
        )
        dl_doc = self._converter.convert(stream).document
        nodes = self._parser.get_nodes_from_documents([Document(text=dl_doc.model_dump_json())])
        chunks: list[Chunk] = []
        for position, node in enumerate(nodes):
            text = node.get_content().strip()
            if text:
                headings = tuple(node.metadata.get("headings") or ())
                chunks.append(Chunk(artifact, headings, position, text))
        return chunks
