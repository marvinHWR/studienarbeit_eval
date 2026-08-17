"""Context diagnostics, computed from gold paragraph indices — for EVERY arm, not just RAG.

Deterministic set-overlap versions of context precision/recall against the supporting-fact gold,
over R = whatever ended up in the arm's context block (not "whatever a retriever fetched"). They
are therefore defined for all three arms, but only informative for RAG: Full-Context has |R| =
kb_size by construction (precision = |G|/kb_size, recall = 1), and No-Context has R = empty, so
precision is 0/0 — reported as 0.0 by convention, which the thesis states explicitly next to the
formulas. Called "Kontextreinheit"/"Kontextabdeckung" in the write-up to keep them apart from the
LLM-judged RAGAS metrics of the same name.
"""
from __future__ import annotations


def context_precision(retrieved_idx: list[int], gold_para_idx: list[int]) -> float:
    if not retrieved_idx:
        return 0.0
    gold = set(gold_para_idx)
    hits = sum(1 for i in retrieved_idx if i in gold)
    return hits / len(retrieved_idx)


def context_recall(retrieved_idx: list[int], gold_para_idx: list[int]) -> float:
    gold = set(gold_para_idx)
    if not gold:
        return 1.0
    retrieved = set(retrieved_idx)
    return len(gold & retrieved) / len(gold)
