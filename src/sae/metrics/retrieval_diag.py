"""Retrieval diagnostics (RAG arm only), computed from gold paragraph indices.

Deterministic set-overlap versions of context precision/recall against the supporting-fact
gold. For Full-Context these are trivially precision = gold/all, recall = 1 by construction.
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
