"""Chunker for visual boards: one chunk per frame (ingestion.md, chunking rules).

A frame's text is already one unit (title, nodes, relationships), so it is not parsed again. When it
exceeds the dense model's token limit it is split by lines into consecutive groups, never inside a
line. The heading path is empty: the title (board and frame) already prefixes the embedded text.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from aether_core.domain.contract import Chunk, NormalizedKnowledgeArtifact


def huggingface_counter(model: str) -> Callable[[str], int]:
    """Token counter of the dense model's tokenizer (the same one the Docling chunker uses)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    return lambda text: len(tokenizer.tokenize(text))


class FrameChunker:
    def __init__(self, count_tokens: Callable[[str], int], max_tokens: int) -> None:
        self._count, self._max = count_tokens, max_tokens

    def chunk(self, artifact: NormalizedKnowledgeArtifact) -> Sequence[Chunk]:
        budget = self._max - self._count(artifact.title) - 2  # room for the title prefix
        groups: list[list[str]] = [[]]
        for line in (line for line in artifact.text.splitlines() if line.strip()):
            candidate = "\n".join([*groups[-1], line])
            if groups[-1] and self._count(candidate) > budget:
                groups.append([line])
            else:
                groups[-1].append(line)
        texts = ["\n".join(group) for group in groups if group]
        return [Chunk(artifact, (), position, text) for position, text in enumerate(texts)]
