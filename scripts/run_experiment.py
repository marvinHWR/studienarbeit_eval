"""Full experiment run: N questions x 3 arms x N samples x KB-size sweep, then score.

Usage:
  python scripts/run_experiment.py --config config/pubmedqa_run.yaml   # the full PubMedQA run
  python scripts/run_experiment.py --config config/pubmedqa_test.yaml --no-sweep   # smoke test
"""
from __future__ import annotations

import argparse

from sae.config import load_config
from sae.data import load_dataset
from sae.experiment import run_experiment, save_run, score_run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--dataset", default=None, help="override cfg.dataset")
    ap.add_argument("--no-sweep", action="store_true", help="skip KB-size sweep")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset = args.dataset or cfg.dataset
    records = load_dataset(dataset, n=cfg.n_questions, seed=cfg.seed)
    print(f"[{dataset}] {len(records)} questions | arms={cfg.arms} "
          f"| samples={cfg.generator.n_samples}")

    kb_sizes = [None] if args.no_sweep else cfg.kb_size_sweep
    df = run_experiment(records, cfg, kb_sizes=kb_sizes, dataset=dataset)

    # Scoring strategy (RAGAS vs. label accuracy) is a property of the dataset, not this script.
    df = score_run(df, dataset, cfg)
    save_run(df, cfg, f"run_{dataset}", dataset)


if __name__ == "__main__":
    main()
