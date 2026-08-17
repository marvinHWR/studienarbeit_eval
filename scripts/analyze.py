"""Analysis: quality tables, the three confirmatory tests, cost/latency + KB-size plots.

Aggregates the N samples per question to one value per question per arm (majority vote for the
categorical label, mean for the continuous metrics), runs the three predefined hypothesis tests
(Bonferroni-corrected over the family of three), and renders the figures that feed Chapter 4.
Everything else is reported descriptively, without a p-value.

Usage:  python scripts/analyze.py --run results/run_pubmedqa.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sae.metrics.accuracy import macro_f1, per_class_recall
from sae.stats import bonferroni_adjust, mcnemar_exact, paired_t

_LABELS = ("yes", "no", "maybe")
# Metrics carried through the descriptive tables. `faithfulness_retrieved` (strict, scored against
# each arm's actual context) is NaN for no_context by design and absent from the shipped run.
# `ctx_precision`/`ctx_recall` are deterministic retrieval diagnostics (judge-free): they explain
# *why* the arms differ (RAG trades a little recall for a lot of precision), but their differences
# follow from how the arms were constructed, so they are reported without a significance test.
QUALITY_METRICS = ["em", "f1", "label_acc", "answer_correctness", "answer_relevancy",
                   "faithfulness", "faithfulness_retrieved", "ctx_precision", "ctx_recall"]
COST_METRICS = ["prompt_tokens", "completion_tokens", "ttft_s", "total_latency_s"]
# Metrics plotted across the KB-size sweep (quality + the two cost drivers that scale with prompt
# size). Filtered to those present in the run; the shipped run has a single KB size, so no plots.
SWEEP_METRICS = ["prompt_tokens", "total_latency_s"]


def _question_table(df_arm: pd.DataFrame) -> pd.DataFrame:
    """One row per question for an arm: majority-vote predicted label across the N samples, the
    constant gold label, and whether the vote is correct. This question-as-unit aggregation is
    what both the classification report and the McNemar tests operate on."""
    def mode(s: pd.Series):
        m = s.dropna().mode()
        return m.iloc[0] if len(m) else None

    g = df_arm.groupby("id").agg(pred=("pred_label", mode), gold=("gold_label", "first"))
    g["gold"] = g["gold"].astype(str).str.strip().str.lower()
    g["correct"] = g["pred"].astype(str).str.strip().str.lower() == g["gold"]
    return g


def _confusion_figure(preds: list, golds: list, arm: str, outdir: Path) -> None:
    cats = list(_LABELS) + ["none"]          # 'none' column = abstentions (no extractable label)
    idx = {c: i for i, c in enumerate(cats)}
    mat = np.zeros((3, len(cats)), dtype=int)   # rows: gold yes/no/maybe; cols: predicted (+none)
    for p, g in zip(preds, golds):
        if g in idx and idx[g] < 3:
            mat[idx[g], idx[p] if p in idx else idx["none"]] += 1
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats)
    ax.set_yticks(range(3))
    ax.set_yticklabels(list(_LABELS))
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    ax.set_title(f"Confusion — {arm}")
    hi = mat.max() or 1
    for i in range(3):
        for j in range(len(cats)):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > hi / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(outdir / f"confusion_{arm}.png", dpi=150)
    plt.close(fig)


def report_label_metrics(df: pd.DataFrame, arms: list[str], outdir: Path) -> pd.DataFrame:
    """Classification report for label datasets: accuracy, macro-F1, per-class recall and
    abstention rate per arm, plus a confusion-matrix figure per arm. These expose the
    minority-class ("maybe") behavior and unparseable answers that a single accuracy number hides.
    """
    rows = []
    for arm in arms:
        sub = df[df["arm"] == arm]
        q = _question_table(sub)
        preds, golds = q["pred"].tolist(), q["gold"].tolist()
        n = len(golds)
        acc = float(q["correct"].mean()) if n else float("nan")
        rec = per_class_recall(preds, golds)
        rows.append({
            "arm": arm, "n_questions": n, "accuracy": acc,
            "macro_f1": macro_f1(preds, golds),
            "recall_yes": rec["yes"], "recall_no": rec["no"], "recall_maybe": rec["maybe"],
            "abstention_rate": float(sub["abstained"].mean()) if "abstained" in sub else float("nan"),
        })
        _confusion_figure(preds, golds, arm, outdir)
    tbl = pd.DataFrame(rows)
    tbl.to_csv(outdir / "label_metrics.csv", index=False)
    print("\n=== Label metrics by arm (question-level, majority vote across samples) ===")
    print(tbl.round(4).to_string(index=False))
    return tbl


def confirmatory_tests(df: pd.DataFrame, per_q: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    """The three predefined hypothesis tests, Bonferroni-corrected as one family.

    T1/T3 compare binary per-question correctness (majority vote) with the exact McNemar test;
    T2 compares the per-question mean faithfulness with a paired t-test. No other metric or arm
    pair is tested — those are reported descriptively.

    The tests are always run at ONE fixed kb_size (the largest present): corpus size is an
    independent variable of the sweep, and pooling across sweep points would fold it into the arm
    comparison. The shipped run has a single size, so this is a no-op there.
    """
    rows: list[dict] = []
    kb = max(df["kb_size"].unique())
    if df["kb_size"].nunique() > 1:
        print(f"[confirmatory_tests] mehrere kb_size-Werte gefunden — getestet wird bei kb_size={kb}")
    df = df[df["kb_size"] == kb]
    per_q = per_q[per_q["kb_size"] == kb]
    arms = set(df["arm"].unique())

    def mcnemar_row(test_id: str, hypothesis: str, arm_a: str, arm_b: str) -> None:
        if not {arm_a, arm_b} <= arms:
            return
        qa = _question_table(df[df["arm"] == arm_a])
        qb = _question_table(df[df["arm"] == arm_b])
        common = qa.index.intersection(qb.index)
        res = mcnemar_exact(qa.loc[common, "correct"].to_numpy(),
                            qb.loc[common, "correct"].to_numpy(),
                            metric="label_acc", arm_a=arm_a, arm_b=arm_b)
        rows.append({
            "test_id": test_id, "hypothesis": hypothesis, "metric": res.metric,
            "arm_a": res.arm_a, "arm_b": res.arm_b, "method": "McNemar (exakt)",
            "n": res.n_pairs, "b_only_a": res.b, "c_only_b": res.c,
            "n_discordant": res.n_discordant, "statistic": np.nan, "df": np.nan,
            "p_value": res.p_value,
        })

    def ttest_row(test_id: str, hypothesis: str, metric: str, arm_a: str, arm_b: str) -> None:
        if not {arm_a, arm_b} <= arms or metric not in per_q.columns:
            return
        wide = per_q.set_index(["arm", "id"])[metric]
        common = wide.loc[arm_a].index.intersection(wide.loc[arm_b].index)
        res = paired_t(wide.loc[arm_a].loc[common].to_numpy(),
                       wide.loc[arm_b].loc[common].to_numpy(),
                       metric=metric, arm_a=arm_a, arm_b=arm_b)
        rows.append({
            "test_id": test_id, "hypothesis": hypothesis, "metric": res.metric,
            "arm_a": res.arm_a, "arm_b": res.arm_b, "method": "gepaarter t-Test",
            "n": res.n, "b_only_a": np.nan, "c_only_b": np.nan, "n_discordant": np.nan,
            "statistic": res.statistic, "df": res.df, "p_value": res.p_value,
        })

    mcnemar_row("T1", "H1", "rag", "no_context")
    ttest_row("T2", "H2", "faithfulness", "rag", "no_context")
    mcnemar_row("T3", "H3", "full_context", "rag")

    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_bonferroni"] = bonferroni_adjust(out["p_value"].to_numpy())
        out.to_csv(outdir / "confirmatory_tests.csv", index=False)
    print("\n=== Konfirmatorische Tests (je Hypothese einer, Bonferroni ueber die Familie) ===")
    print(out.to_string(index=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--outdir", default="results/figures")
    args = ap.parse_args()

    df = pd.read_parquet(args.run)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = [m for m in QUALITY_METRICS if m in df.columns]
    # Label datasets (PubMedQA): EM/F1 compare a free-text answer against the long conclusion and
    # are structurally ~0, so drop them from the report — label_acc + the classification report
    # (report_label_metrics) carry the quality verdict instead.
    is_label = "label_acc" in df.columns
    if is_label:
        metrics = [m for m in metrics if m not in ("em", "f1")]

    # Aggregate samples -> one value per (id, arm, kb_size). kb_size is held fixed (it's an
    # independent variable of the sweep, not something to average over).
    per_q = df.groupby(["id", "arm", "kb_size"], as_index=False)[metrics + COST_METRICS].mean()

    # Quality + cost summary table.
    print("\n=== Mean by arm ===")
    print(df.groupby("arm")[metrics + COST_METRICS].mean().round(3))

    # Efficiency view: primary quality per 1k prompt tokens — the grounded cost/quality tradeoff
    # at the heart of the study (uses token counts, no LLM judge). label_acc for label datasets.
    prim = "label_acc" if is_label else ("answer_correctness" if "answer_correctness" in df.columns else "f1")
    if prim in df.columns and "prompt_tokens" in df.columns:
        by_arm = df.groupby("arm")
        eff = pd.DataFrame({prim: by_arm[prim].mean(), "prompt_tokens": by_arm["prompt_tokens"].mean()})
        eff["quality_per_1k_ptok"] = eff[prim] / eff["prompt_tokens"] * 1000
        eff.to_csv(outdir / "efficiency.csv")
        print(f"\n=== Efficiency: {prim} per 1k prompt tokens ===")
        print(eff.round(4))

    # Classification report (accuracy / macro-F1 / per-class recall / abstention) for label
    # datasets (needs the per-row pred_label column added by scoring since 2026-07).
    if is_label and "pred_label" in df.columns:
        report_label_metrics(df, sorted(df["arm"].unique()), outdir)
        confirmatory_tests(df, per_q, outdir)

    # Descriptive trend plots when the run actually swept KB size (one line per arm).
    if df["kb_size"].nunique() > 1:
        sweep_metrics = [m for m in metrics + SWEEP_METRICS if m in df.columns]
        agg = df.groupby(["kb_size", "arm"])[sweep_metrics].mean()
        for metric in sweep_metrics:
            fig, ax = plt.subplots(figsize=(6, 4))
            for arm in sorted(df["arm"].unique()):
                sub = agg.xs(arm, level="arm")[metric]
                ax.plot(sub.index, sub.values, marker="o", label=arm)
            ax.set_xlabel("KB size (paragraphs)")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} vs. KB size")
            ax.legend()
            fig.tight_layout()
            fig.savefig(outdir / f"sweep_{metric}.png", dpi=150)
            plt.close(fig)
        print(f"\nKB-size sweep plots -> {outdir}")

    # Quality bars by arm.
    fig, ax = plt.subplots(figsize=(7, 4))
    df.groupby("arm")[metrics].mean().plot.bar(ax=ax)
    ax.set_ylabel("score")
    ax.set_title("Quality by arm")
    fig.tight_layout()
    fig.savefig(outdir / "quality_by_arm.png", dpi=150)
    plt.close(fig)
    print(f"Figures + stats -> {outdir}")


if __name__ == "__main__":
    main()
