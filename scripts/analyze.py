"""Analysis: quality tables, paired stats, KB-size H3/H4 tests, cost/latency + KB-size plots.

Aggregates the N samples per question to a per-question mean (so paired tests operate on one
value per question per arm), runs the predefined paired comparisons (per fixed KB size), tests
the KB-size hypotheses inferentially (per-question slope trend for H3, arm x size interaction
for H4; Holm-corrected), and renders the figures that feed Chapter 5 (Ergebnisse).

Usage:  python scripts/analyze.py --run results/run_pubmedqa.parquet
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sae.metrics.accuracy import macro_f1, per_class_recall
from sae.stats import compare_arms, holm_adjust, slope, trend_test

_LABELS = ("yes", "no", "maybe")
# `faithfulness_retrieved` (strict, scored against each arm's actual context) is NaN for
# no_context by design, so its paired comparisons involving no_context come back degenerate
# (n=0) — expected, not a bug. compare_arms drops the NaN pairs.
# `ctx_precision`/`ctx_recall` are deterministic retrieval diagnostics (judge-free): they explain
# *why* the arms differ (RAG trades a little recall for a lot of precision), so they are tested
# alongside the answer-quality metrics rather than only reported descriptively. Note this widens
# the Holm family below — the correction is applied over all metric x arm-pair tests at one kb_size.
QUALITY_METRICS = ["em", "f1", "label_acc", "answer_correctness", "answer_relevancy",
                   "faithfulness", "faithfulness_retrieved", "ctx_precision", "ctx_recall"]
COST_METRICS = ["prompt_tokens", "completion_tokens", "ttft_s", "total_latency_s"]
# Metrics whose trend across the KB-size sweep is tested for H3/H4 (quality + the two cost
# drivers that scale with prompt size). Filtered to those present in the run.
SWEEP_METRICS = ["prompt_tokens", "total_latency_s"]


def per_question_slopes(df: pd.DataFrame, arm: str, metric: str) -> dict[str, float]:
    """Per-question OLS slope of `metric` over kb_size for one arm.

    Samples are averaged to one value per (id, kb_size) first, then a single slope summarizes
    how the question responds to corpus size. Questions seen at only one KB size are skipped.
    """
    per = df[df["arm"] == arm].groupby(["id", "kb_size"])[metric].mean()
    out: dict[str, float] = {}
    for qid, g in per.groupby(level="id"):
        s = g.dropna()
        xs = s.index.get_level_values("kb_size").to_numpy(float)
        if np.unique(xs).size < 2:
            continue
        out[qid] = slope(xs, s.to_numpy(float))
    return out


def _question_predictions(df_arm: pd.DataFrame) -> tuple[list, list]:
    """One (pred, gold) per question for an arm: majority-vote predicted label across the N
    samples (the same question-as-unit aggregation the paired tests use), gold constant per id."""
    def mode(s: pd.Series):
        m = s.dropna().mode()
        return m.iloc[0] if len(m) else None

    g = df_arm.groupby("id").agg(pred=("pred_label", mode), gold=("gold_label", "first"))
    golds = [str(x).strip().lower() for x in g["gold"].tolist()]
    return g["pred"].tolist(), golds


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
        preds, golds = _question_predictions(sub)
        n = len(golds)
        acc = float(np.mean([p == g for p, g in zip(preds, golds)])) if n else float("nan")
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
    # are structurally ~0, so drop them from the arm comparison — label_acc + the classification
    # report (report_label_metrics) carry the quality verdict instead.
    is_label = "label_acc" in df.columns
    if is_label:
        metrics = [m for m in metrics if m not in ("em", "f1")]

    # Aggregate samples -> one value per (id, arm, kb_size) for paired tests. kb_size is held
    # fixed (it's an independent variable of the sweep, not something to average over), so the
    # paired comparison never mixes different KB sizes into one arm value.
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

    # Predefined paired comparisons across every arm pair, every quality metric, per kb_size.
    print("\n=== Paired Wilcoxon (per-question means, held at fixed kb_size) ===")
    rows = []
    arms = sorted(per_q["arm"].unique())
    for kb in sorted(per_q["kb_size"].unique()):
        at_kb = per_q[per_q["kb_size"] == kb]
        wide = {a: at_kb[at_kb["arm"] == a].set_index("id") for a in arms}
        for m in metrics:
            for a, b in itertools.combinations(arms, 2):
                common = wide[a].index.intersection(wide[b].index)
                res = compare_arms(
                    wide[a].loc[common, m].to_numpy(),
                    wide[b].loc[common, m].to_numpy(),
                    metric=m, arm_a=a, arm_b=b,
                )
                rows.append(res.__dict__ | {"kb_size": kb})
    stats_df = pd.DataFrame(rows)
    # Holm-correct within each fixed-kb family: all metric x arm-pair tests at one kb_size form a
    # single confirmatory family, so the main arm table carries p_holm too (multiplicity control).
    if not stats_df.empty:
        stats_df["p_holm"] = np.nan
        for _, grp in stats_df.groupby("kb_size"):
            stats_df.loc[grp.index, "p_holm"] = holm_adjust(grp["p_value"].to_numpy())
    stats_df.to_csv(outdir / "paired_stats.csv", index=False)
    print(stats_df.round(4).to_string(index=False))

    # KB-size hypotheses (H3/H4). Only meaningful when the run actually swept KB size.
    if df["kb_size"].nunique() > 1:
        sweep_metrics = [m for m in metrics + SWEEP_METRICS if m in df.columns]

        # H3: within-arm trend — does the per-question slope of the metric over KB size differ
        # systematically from zero? (e.g. Full-Context prompt_tokens should rise; No-Context flat.)
        print("\n=== KB-size trend within arm (H3): one-sample Wilcoxon on per-question slopes ===")
        trend_rows = [
            trend_test(np.array(list(per_question_slopes(df, a, m).values())), metric=m, arm=a).__dict__
            for m in sweep_metrics for a in arms
        ]
        trend_df = pd.DataFrame(trend_rows)
        trend_df["p_holm"] = holm_adjust(trend_df["p_value"].to_numpy())
        trend_df.to_csv(outdir / "kb_trend_stats.csv", index=False)
        print(trend_df.round(4).to_string(index=False))

        # H4: arm x KB-size interaction — do two arms respond differently to KB size? Compare the
        # per-question slopes between arms (paired on question id) with the same Wilcoxon machinery.
        print("\n=== KB-size x arm interaction (H4): paired Wilcoxon on slope differences ===")
        inter_rows = []
        for m in sweep_metrics:
            slopes_by_arm = {a: per_question_slopes(df, a, m) for a in arms}
            for a, b in itertools.combinations(arms, 2):
                common = sorted(set(slopes_by_arm[a]) & set(slopes_by_arm[b]))
                res = compare_arms(
                    np.array([slopes_by_arm[a][q] for q in common], float),
                    np.array([slopes_by_arm[b][q] for q in common], float),
                    metric=m, arm_a=a, arm_b=b,
                )
                inter_rows.append(res.__dict__)
        inter_df = pd.DataFrame(inter_rows)
        inter_df["p_holm"] = holm_adjust(inter_df["p_value"].to_numpy())
        inter_df.to_csv(outdir / "kb_interaction_stats.csv", index=False)
        print(inter_df.round(4).to_string(index=False))

        # Trend plots for every swept metric (quality + cost), one line per arm.
        agg = df.groupby(["kb_size", "arm"])[sweep_metrics].mean()
        for metric in sweep_metrics:
            fig, ax = plt.subplots(figsize=(6, 4))
            for arm in arms:
                sub = agg.xs(arm, level="arm")[metric]
                ax.plot(sub.index, sub.values, marker="o", label=arm)
            ax.set_xlabel("KB size (paragraphs)")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} vs. KB size")
            ax.legend()
            fig.tight_layout()
            fig.savefig(outdir / f"sweep_{metric}.png", dpi=150)
            plt.close(fig)
        print(f"\nKB-size sweep plots + H3/H4 stats -> {outdir}")

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
