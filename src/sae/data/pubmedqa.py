"""PubMedQA (pqa_labeled) loader -> unified Record schema.

Medical-domain robustness set that reconnects to the ECG product story. Abstracts are
POOLED into one shared corpus so retrieval is non-trivial while Full-Context still fits a
128k window. Answers are yes/no/maybe labels -> scored with ACCURACY (not answer_correctness).

`pool_size` abstracts are shared across all questions as the corpus; each question's own
context is guaranteed to be in the pool.
"""
from __future__ import annotations

import random

from datasets import load_dataset as hf_load

from .schema import Record


def load_pubmedqa(
    n: int | None = None,
    seed: int = 42,
    pool_size: int = 120,
) -> list[Record]:
    ds = hf_load("qiaojin/PubMedQA", "pqa_labeled", split="train")

    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    q_idx = idx[: (n or len(idx))]

    # Build the shared pool: every selected question's context + fillers up to pool_size.
    def ctx_paras(row) -> list[str]:
        return list(row["context"]["contexts"])

    pool: list[str] = []
    para_to_pos: dict[str, int] = {}

    def add(text: str) -> int:
        if text not in para_to_pos:
            para_to_pos[text] = len(pool)
            pool.append(text)
        return para_to_pos[text]

    records_raw = []
    for i in q_idx:
        row = ds[i]
        gold_positions = [add(p) for p in ctx_paras(row)]
        records_raw.append((row, gold_positions))

    # Pad the pool with additional abstracts until pool_size is reached.
    for i in idx:
        if len(pool) >= pool_size:
            break
        for p in ctx_paras(ds[i]):
            if len(pool) >= pool_size:
                break
            add(p)

    records: list[Record] = []
    for row, gold_positions in records_raw:
        records.append(
            Record(
                id=str(row["pubid"]),
                question=row["question"],
                gold_answer=row["long_answer"],
                paragraphs=pool,                       # shared pooled corpus
                gold_para_idx=sorted(set(gold_positions)),
                gold_label=row["final_decision"],      # yes / no / maybe
            )
        )
    return records
