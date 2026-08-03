"""Experiment orchestration: run all arms over all questions, with the KB-size sweep.

Produces one long-format dataframe (one row per question x arm x sample x kb_size) carrying
answers, retrieved indices, and cost/latency. Quality scoring (deterministic + RAGAS) is
applied afterwards so the expensive generation pass is done once and cached.
"""
from __future__ import annotations

import random

import pandas as pd
from tqdm import tqdm

from ..arms import run_arm
from ..arms.arms import system_prompt_for
from ..config import Config
from ..data.schema import Record
from ..llm.client import LLMClient
from ..metrics.deterministic import exact_match, token_f1
from ..metrics.retrieval_diag import context_precision, context_recall
from ..retrieval.retriever import Retriever


def resize_kb(record: Record, kb_size: int, seed: int) -> Record:
    """Return a copy of `record` whose corpus is trimmed/kept to kb_size paragraphs,
    always preserving the gold paragraphs. Distractors are sampled deterministically.
    Used to make KB size an independent variable for H3/H4."""
    gold = set(record.gold_para_idx)
    gold_paras = [record.paragraphs[i] for i in record.gold_para_idx]
    distractor_idx = [i for i in range(len(record.paragraphs)) if i not in gold]

    rng = random.Random(f"{record.id}-{kb_size}-{seed}")
    rng.shuffle(distractor_idx)
    n_dist = max(0, kb_size - len(gold_paras))
    kept_dist = [record.paragraphs[i] for i in distractor_idx[:n_dist]]

    # Tag each paragraph with whether it's gold, then shuffle to avoid gold-always-first
    # positional bias. Tracking gold by identity through the shuffle (not by text lookup)
    # is robust to duplicate paragraph texts — a distractor equal to a gold, or two identical
    # golds — which `new_paras.index(p)` would mislocate.
    items = [(p, True) for p in gold_paras] + [(p, False) for p in kept_dist]
    rng.shuffle(items)
    new_paras = [text for text, _ in items]
    new_gold_idx = [i for i, (_, is_gold) in enumerate(items) if is_gold]

    return Record(
        id=record.id,
        question=record.question,
        gold_answer=record.gold_answer,
        paragraphs=new_paras,
        gold_para_idx=sorted(new_gold_idx),
        gold_label=record.gold_label,
    )


def run_experiment(
    records: list[Record],
    cfg: Config,
    kb_sizes: list[int] | None = None,
    dataset: str | None = None,
) -> pd.DataFrame:
    client = LLMClient(
        cfg.generator.model, cfg.generator.temperature, cfg.generator.max_tokens
    )
    retriever = Retriever(cfg.embedding_model, cfg.chunk_level)
    kb_sizes = kb_sizes if kb_sizes is not None else [None]  # None = native corpus size
    # The answer-format prompt is dataset-dependent (label-leading for yes/no/maybe sets), so it
    # is resolved from the *actual* dataset being run — which the script can override past
    # cfg.dataset — not from cfg alone. Identical across arms, so the control is preserved.
    system = system_prompt_for(dataset or cfg.dataset)

    out_rows = []
    for record in tqdm(records, desc="questions"):
        for kb_size in kb_sizes:
            rec = record if kb_size is None else resize_kb(record, kb_size, cfg.seed)
            for arm in cfg.arms:
                for s in range(cfg.generator.n_samples):
                    seed = cfg.seed + s
                    o = run_arm(arm, rec, client, retriever, cfg.k, seed=seed, system=system)
                    r = o.result
                    row = {
                        "id": rec.id,
                        "question": rec.question,
                        "gold_answer": rec.gold_answer,
                        "gold_label": rec.gold_label,
                        "arm": arm,
                        "sample": s,
                        "kb_size": kb_size if kb_size is not None else len(rec.paragraphs),
                        "answer": o.answer,
                        "contexts": o.contexts,
                        "retrieved_idx": o.retrieved_idx,
                        "gold_para_idx": rec.gold_para_idx,
                        "full_corpus": rec.paragraphs,
                        # deterministic quality (cheap, judge-independent)
                        "em": exact_match(o.answer, rec.gold_answer),
                        "f1": token_f1(o.answer, rec.gold_answer),
                        # retrieval diagnostics
                        "ctx_precision": context_precision(o.retrieved_idx, rec.gold_para_idx),
                        "ctx_recall": context_recall(o.retrieved_idx, rec.gold_para_idx),
                        # cost / latency
                        "prompt_tokens": r.prompt_tokens if r else None,
                        "completion_tokens": r.completion_tokens if r else None,
                        # thinking-model split of completion_tokens (hidden reasoning vs. the
                        # visible answer) + a truncation flag, so a chopped answer is auditable.
                        "reasoning_tokens": r.reasoning_tokens if r else None,
                        "text_tokens": r.text_tokens if r else None,
                        "truncated": r.truncated if r else None,
                        "ttft_s": r.ttft_s if r else None,
                        "total_latency_s": r.total_latency_s if r else None,
                    }
                    out_rows.append(row)
    return pd.DataFrame(out_rows)
