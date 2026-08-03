"""Deterministic, judge-independent metrics: HotpotQA official EM and token-F1.

These need no LLM, are fully reproducible, and are contamination-diagnostic (a No-Context arm
scoring near-perfect EM signals memorized answers). They are the second yardstick that keeps
the cross-arm verdict from resting on a single noisy LLM metric.
Normalization follows the official SQuAD/HotpotQA scorer.
"""
from __future__ import annotations

import re
import string
from collections import Counter


def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(_normalize(prediction) == _normalize(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_toks = _normalize(prediction).split()
    gold_toks = _normalize(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)
