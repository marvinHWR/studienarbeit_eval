"""Chunking + embedding + top-k retrieval.

The per-question corpus is tiny (HotpotQA: 10 paragraphs), so an in-memory cosine search
over a normalized embedding matrix is enough and fully deterministic. The same code handles
the pooled PubMedQA corpus. `source_idx` maps each chunk back to its paragraph index, which
the retrieval diagnostics need.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass
class Chunk:
    text: str
    source_idx: int          # index into Record.paragraphs
    score: float = 0.0


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _chunks_from_paragraphs(paragraphs: list[str], chunk_level: str) -> list[Chunk]:
    if chunk_level == "paragraph":
        return [Chunk(text=p, source_idx=i) for i, p in enumerate(paragraphs)]
    if chunk_level == "sentence":
        out: list[Chunk] = []
        for i, p in enumerate(paragraphs):
            for s in _SENT_SPLIT.split(p.strip()):
                if s.strip():
                    out.append(Chunk(text=s.strip(), source_idx=i))
        return out
    raise ValueError(f"Unknown chunk_level: {chunk_level!r}")


@lru_cache(maxsize=4)
def _load_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


class Retriever:
    def __init__(self, embedding_model: str, chunk_level: str = "paragraph"):
        self.embedder = _load_embedder(embedding_model)
        self.chunk_level = chunk_level
        # Corpus embeddings keyed by the chunk texts. Retrieval is deterministic across the
        # N samples (and identical for the RAG arm on the shared PubMedQA pool), so embedding
        # each corpus once instead of per call removes ~2/3 of the redundant embedding passes.
        self._corpus_cache: dict[tuple[str, ...], np.ndarray] = {}

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    def retrieve(self, question: str, paragraphs: list[str], k: int) -> list[Chunk]:
        chunks = _chunks_from_paragraphs(paragraphs, self.chunk_level)
        if not chunks:
            return []
        key = tuple(c.text for c in chunks)
        mat = self._corpus_cache.get(key)
        if mat is None:
            mat = self._embed([c.text for c in chunks])      # (N, d), L2-normalized
            self._corpus_cache[key] = mat
        q = self._embed([question])[0]                        # (d,)
        sims = mat @ q                                        # cosine similarity
        order = np.argsort(-sims)[: min(k, len(chunks))]
        out = []
        for j in order:
            c = chunks[int(j)]
            out.append(Chunk(text=c.text, source_idx=c.source_idx, score=float(sims[j])))
        return out
