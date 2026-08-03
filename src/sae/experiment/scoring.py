"""Post-generation scoring + run persistence, shared by the run and rescore scripts.

The generation pass is dataset-independent; only the *scoring* differs (RAGAS answer
quality vs. PubMedQA label accuracy). Which one to use is a property of the dataset, not a
per-script `if` branch — so it lives in one registry here and every script consumes it. The
save plumbing (parquet + Excel report) is identical across scripts and lives here too.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..data.schema import LABEL_DATASETS
from ..metrics.accuracy import extract_label, is_abstention, label_accuracy
from ..metrics.ragas_eval import score_ragas
from ..reporting import write_excel_report

# Datasets scored by classification accuracy (yes/no/maybe labels) instead of, or in addition
# to, RAGAS, live in `data.schema.LABEL_DATASETS` (single source of truth — the arms module
# reads the same set to pick the label-leading prompt). Adding a label-style dataset = one entry
# there, not a new script branch.

# RAGAS metric subset per dataset. Label datasets keep faithfulness/answer_relevancy (they
# judge grounding/on-topic-ness independent of answer format) but drop answer_correctness,
# which compares free-text semantic similarity against `ground_truth` and mismatches a
# categorical (yes/no/maybe) gold. Every other dataset gets the full default set.
DEFAULT_RAGAS_METRICS = ["answer_correctness", "answer_relevancy", "faithfulness"]
RAGAS_METRICS_BY_DATASET = {
    "pubmedqa": ["answer_relevancy", "faithfulness"],
}


def scoring_strategy(dataset: str) -> str:
    """"label" for label-accuracy datasets, "ragas" otherwise."""
    return "label" if dataset.lower() in LABEL_DATASETS else "ragas"


def _strict_faithfulness(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Faithfulness scored against each row's ACTUAL context (`df['contexts']`), not the full
    corpus. Aligned to `df`'s (0..n-1) index, with NaN where the arm had no context (no_context,
    whose empty context makes groundedness undefined). This is the strict "grounded in what it
    retrieved" measure; for full_context it equals the full-corpus faithfulness by construction.
    """
    result = pd.Series(np.nan, index=df.index, name="faithfulness_retrieved", dtype="float64")
    has_ctx = df["contexts"].apply(lambda c: c is not None and len(c) > 0)
    sub = df[has_ctx]
    if sub.empty:
        return result
    rows = [
        {"question": r.question, "answer": r.answer,
         "ground_truth": r.gold_answer, "contexts": r.contexts}
        for r in sub.itertuples()
    ]
    strict = score_ragas(
        rows, cfg.judge.model, cfg.judge.temperature, cfg.embedding_model,
        metric_names=["faithfulness"], contexts_key="contexts",
    )
    result.loc[sub.index] = strict["faithfulness"].to_numpy()
    return result


def score_and_attach_ragas(
    df: pd.DataFrame, cfg: Config, metric_names: list[str] | None = None
) -> pd.DataFrame:
    """Score each answer with RAGAS (judge != generator) and append the metric columns.

    Groundedness (`faithfulness`) is scored against the FULL corpus (identical across arms), so
    No-Context is comparable — see metrics.ragas_eval. When faithfulness is in the metric set,
    a second `faithfulness_retrieved` column is added, scored against each arm's actual context
    (strict groundedness; NaN for the context-less no_context arm).
    """
    rag_rows = [
        {"question": r.question, "answer": r.answer,
         "ground_truth": r.gold_answer, "full_corpus": r.full_corpus}
        for r in df.itertuples()
    ]
    ragas_df = score_ragas(
        rag_rows, cfg.judge.model, cfg.judge.temperature, cfg.embedding_model, metric_names
    )
    out = pd.concat([df.reset_index(drop=True), ragas_df.reset_index(drop=True)], axis=1)

    effective = metric_names or DEFAULT_RAGAS_METRICS
    if (getattr(cfg, "strict_faithfulness", True)
            and "faithfulness" in effective and "contexts" in df.columns):
        out["faithfulness_retrieved"] = _strict_faithfulness(df.reset_index(drop=True), cfg).to_numpy()
    return out


def attach_label_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row predicted label + abstention flag + label accuracy, so accuracy, macro-F1,
    per-class recall and abstention rate can all be recomputed downstream (analyze.py) from the
    saved frame. Judge-free and idempotent, so it doubles as the backfill for an improved
    `extract_label` (see scripts/rescore.py)."""
    df = df.copy()
    df["pred_label"] = [extract_label(a) for a in df["answer"]]
    df["abstained"] = [is_abstention(a) for a in df["answer"]]
    df["label_acc"] = [label_accuracy(a, g) for a, g in zip(df["answer"], df["gold_label"])]
    return df


def attach_faithfulness_retrieved(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Append the strict per-arm `faithfulness_retrieved` column (judge pass; faithfulness only).
    Used by the rescore backfill to add the column without re-running the other RAGAS metrics."""
    df = df.copy()
    df["faithfulness_retrieved"] = _strict_faithfulness(df.reset_index(drop=True), cfg).to_numpy()
    return df


def score_run(df: pd.DataFrame, dataset: str, cfg: Config) -> pd.DataFrame:
    """Attach the quality metric(s) appropriate for `dataset` and return the scored frame."""
    metric_names = RAGAS_METRICS_BY_DATASET.get(dataset.lower(), DEFAULT_RAGAS_METRICS)
    if scoring_strategy(dataset) == "label":
        df = attach_label_metrics(df)
    return score_and_attach_ragas(df, cfg, metric_names)


def save_run(df: pd.DataFrame, cfg: Config, parquet_stem: str, xlsx_stem: str) -> Path:
    """Write the scored frame to Parquet (raw source of truth) + an Excel report, and return
    the parquet path. Filenames: `{parquet_stem}.parquet` and `answers_{xlsx_stem}.xlsx`."""
    out = Path(cfg.paths.get("results_dir", "results")) / f"{parquet_stem}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"Scored dataframe -> {out}  ({len(df)} rows)")

    write_excel_report(df, out.with_name(f"answers_{xlsx_stem}.xlsx"))
    return out
