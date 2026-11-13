"""Chunking by structure with Docling HybridChunker (technology.md default).

HybridChunker is passed explicitly with the tokenizer of the dense model, so chunks fit its token limit.
The heading path of each chunk is kept as deterministic context; there is no overlap.
"""

from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

from aether_core.domain.contract import Chunk, NormalizedKnowledgeArtifact


class DoclingChunker:
    def __init__(self, tokenizer_model: str, max_tokens: int) -> None:
        tokenizer = HuggingFaceTokenizer.from_pretrained(tokenizer_model, max_tokens=max_tokens)
        self._chunker = HybridChunker(tokenizer=tokenizer)
        self._converter = DocumentConverter(allowed_formats=[InputFormat.HTML])

    def chunk(self, artifact: NormalizedKnowledgeArtifact) -> Sequence[Chunk]:
        stream = DocumentStream(
            name=f"{artifact.source.provider_artifact_id}.html", stream=BytesIO(artifact.text.encode("utf-8"))
        )
        document = self._converter.convert(stream).document
        chunks: list[Chunk] = []
        for position, doc_chunk in enumerate(self._chunker.chunk(dl_doc=document)):
            headings = tuple(getattr(doc_chunk.meta, "headings", None) or ())
            text = doc_chunk.text.strip()
            if text:
                chunks.append(Chunk(artifact, headings, position, text))
        return chunks
