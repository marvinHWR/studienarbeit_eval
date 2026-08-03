"""Backfill improved metrics onto an already-generated run WITHOUT regenerating answers.

Reads an existing scored Parquet and rewrites it with two refreshed metrics, cheaply:
  - the PubMedQA label columns (`pred_label`/`abstained`/`label_acc`) — judge-free, so this picks
    up any improvement to `extract_label` at zero cost;
  - the strict `faithfulness_retrieved` column — scored against each arm's actual context, the
    only new judge cost (one faithfulness eval per non-no_context row).
Existing RAGAS columns (full-corpus `faithfulness`, `answer_relevancy`, ...) are left untouched.

Output is written to a NEW `*_rescored` Parquet + Excel so the original run is never overwritten.

Usage:
  python scripts/rescore.py --run results/run_pubmedqa.parquet --dataset pubmedqa
  python scripts/rescore.py --run results/run_pubmedqa.parquet --dataset pubmedqa --sample 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sae.config import load_config
from sae.experiment import (
    attach_faithfulness_retrieved,
    attach_label_metrics,
    save_run,
    scoring_strategy,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="path to an existing scored Parquet")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--dataset", default=None, help="dataset name (defaults to cfg.dataset)")
    ap.add_argument("--sample", type=int, default=None,
                    help="cap to the first N rows for a cheap validation pass")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset = args.dataset or cfg.dataset

    df = pd.read_parquet(args.run)
    if args.sample is not None:
        df = df.head(args.sample).copy()
    print(f"[rescore] {len(df)} rows from {args.run} (dataset={dataset})")

    if scoring_strategy(dataset) == "label":
        df = attach_label_metrics(df)            # free: hardened label extraction
    df = attach_faithfulness_retrieved(df, cfg)  # judge: strict faithfulness only

    stem = Path(args.run).stem
    save_run(df, cfg, f"{stem}_rescored", f"{dataset}_rescored")


if __name__ == "__main__":
    main()
