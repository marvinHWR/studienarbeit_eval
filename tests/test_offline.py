"""Offline unit tests: run without any API key, model download, or network.

Covers the deterministic, judge-independent parts of the harness so the core logic can be
verified before wiring up a provider.
    pytest -q
"""
from __future__ import annotations

import numpy as np
import pytest

from sae.config import load_config
from sae.data.schema import Record
from sae.experiment.runner import resize_kb
from sae.metrics.accuracy import (
    extract_label,
    is_abstention,
    label_accuracy,
    macro_f1,
    per_class_recall,
)
from sae.metrics.deterministic import exact_match, token_f1
from sae.metrics.retrieval_diag import context_precision, context_recall
from sae.stats import bonferroni_adjust, mcnemar_exact, paired_t


def test_exact_match_normalization():
    assert exact_match("The Beatles", "beatles") == 1.0
    assert exact_match("A cat.", "cat") == 1.0
    assert exact_match("dog", "cat") == 0.0


def test_token_f1_partial():
    assert token_f1("New York City", "New York City") == 1.0
    assert 0.0 < token_f1("New York", "New York City") < 1.0
    assert token_f1("", "cat") == 0.0


def test_retrieval_diag():
    assert context_recall([1, 3], [1, 3]) == 1.0
    assert context_recall([1], [1, 3]) == 0.5
    assert context_precision([1, 2], [1, 3]) == 0.5
    assert context_recall([], []) == 1.0  # no gold -> vacuously complete


def test_label_extraction():
    assert extract_label("Yes, the study supports it.") == "yes"
    assert extract_label("No effect was found.") == "no"
    assert extract_label("It is unclear.") is None
    assert label_accuracy("Yes.", "yes") == 1.0
    assert label_accuracy("Maybe.", "no") == 0.0
    # Label-leading answers (the new PubMedQA prompt): the leading decision wins over later tokens.
    assert extract_label("Maybe. The evidence is mixed and inconclusive.") == "maybe"
    assert extract_label("No, there was no significant association found.") == "no"
    # Tolerate leading formatting / lead-in phrases while still anchoring to the verdict.
    assert extract_label("**Yes.** The trial was positive.") == "yes"
    assert extract_label("Answer: maybe") == "maybe"
    assert extract_label("The answer is yes.") == "yes"
    # A label word buried in a non-committal answer must NOT be read as a verdict (was a bug: the
    # old first-match-anywhere regex scored this a confident "no" and hid a genuine abstention).
    assert extract_label("There is no clear consensus in the literature.") is None
    assert extract_label("The evidence does not clearly support a conclusion here.") is None


def test_is_abstention():
    assert is_abstention("It is unclear.") is True          # no extractable label
    assert is_abstention("I do not know.") is True
    assert is_abstention("Maybe, the evidence is mixed.") is False   # a valid decision
    assert is_abstention("There is no clear consensus.") is True     # non-leading "no" != verdict


def test_strict_faithfulness_nan_for_empty_context(monkeypatch):
    """`faithfulness_retrieved` is scored per-arm against the actual context, and must be NaN for
    the context-less no_context arm while keeping the judge scores for arms that had context —
    aligned back to the original row order. Judge call is stubbed so this stays offline."""
    import pandas as pd
    from types import SimpleNamespace

    import sae.experiment.scoring as scoring

    df = pd.DataFrame({
        "question": ["q1", "q2", "q3"],
        "answer": ["a1", "a2", "a3"],
        "gold_answer": ["g1", "g2", "g3"],
        "contexts": [["c"], [], ["c1", "c2"]],   # row 2 = no_context (empty) -> NaN
    })
    # Stub the judge: one faithfulness score per non-empty-context row, in order.
    monkeypatch.setattr(scoring, "score_ragas",
                        lambda rows, *a, **k: pd.DataFrame({"faithfulness": [0.5, 0.9]}))
    cfg = SimpleNamespace(judge=SimpleNamespace(model="m", temperature=0.0), embedding_model="e")

    s = scoring._strict_faithfulness(df, cfg)
    assert s.name == "faithfulness_retrieved"
    assert s.iloc[0] == 0.5
    assert np.isnan(s.iloc[1])          # no_context -> undefined
    assert s.iloc[2] == 0.9


def test_per_class_recall_and_macro_f1():
    preds = ["yes", "no", "maybe", "yes", None]     # last answer abstained (no label)
    golds = ["yes", "no", "maybe", "no", "maybe"]
    rec = per_class_recall(preds, golds)
    assert rec["yes"] == 1.0        # 1/1 gold-yes labelled yes
    assert rec["no"] == 0.5         # 1/2 gold-no correct (one predicted yes)
    assert rec["maybe"] == 0.5      # 1/2 gold-maybe correct (one abstained)
    assert macro_f1(["yes", "no", "maybe"], ["yes", "no", "maybe"]) == pytest.approx(1.0)
    # Never predicting the minority class 'maybe' tanks macro-F1 below what accuracy alone shows.
    assert macro_f1(["yes", "yes", "yes"], ["yes", "yes", "maybe"]) < 1.0


def test_resize_kb_preserves_gold_and_size():
    rec = Record(
        id="q1", question="?", gold_answer="a",
        paragraphs=[f"p{i}" for i in range(10)],
        gold_para_idx=[2, 5],
    )
    small = resize_kb(rec, kb_size=4, seed=42)
    assert len(small.paragraphs) == 4
    kept = {small.paragraphs[i] for i in small.gold_para_idx}
    assert kept == {"p2", "p5"}                 # gold always retained
    assert context_recall(small.gold_para_idx, small.gold_para_idx) == 1.0


def test_resize_kb_locates_gold_with_duplicate_texts():
    # A distractor whose text equals a gold paragraph must not steal the gold's index:
    # resize_kb tracks gold by identity through the shuffle, not by first text match.
    rec = Record(
        id="dup", question="?", gold_answer="a",
        paragraphs=["gold", "gold", "d1", "d2", "d3"],  # p1 duplicates the gold text at p0
        gold_para_idx=[0],
    )
    small = resize_kb(rec, kb_size=4, seed=1)
    assert len(small.gold_para_idx) == 1                 # exactly one gold retained
    assert small.paragraphs[small.gold_para_idx[0]] == "gold"


def test_mcnemar_counts_only_discordant_pairs():
    # Concordant questions (both right / both wrong) carry no information; only b and c do.
    a = np.array([True, True, True, False, False, True, False])
    b = np.array([True, False, False, True, False, True, False])
    res = mcnemar_exact(a, b, metric="label_acc", arm_a="rag", arm_b="no_context")
    assert res.n_pairs == 7
    assert (res.b, res.c) == (2, 1)          # 2x only A right, 1x only B right
    assert res.n_discordant == 3


def test_mcnemar_symmetric_gives_p_one():
    # The thesis' H3 case in miniature: the arms disagree, but symmetrically -> no evidence.
    a = np.array([True] * 21 + [False] * 21)
    b = ~a
    res = mcnemar_exact(a, b, metric="label_acc", arm_a="full_context", arm_b="rag")
    assert res.b == res.c == 21
    assert res.p_value == pytest.approx(1.0)
    # Arms that never disagree are also p = 1.0, not a division by zero.
    same = mcnemar_exact(a, a, metric="label_acc", arm_a="x", arm_b="y")
    assert same.n_discordant == 0 and same.p_value == 1.0


def test_mcnemar_detects_asymmetry():
    a = np.array([True] * 20 + [False] * 2)     # 20 only-A vs. 2 only-B
    b = ~a
    res = mcnemar_exact(a, b, metric="label_acc", arm_a="rag", arm_b="no_context")
    assert res.p_value < 0.001


def test_paired_t_drops_nan_pairs():
    a = np.array([0.9, 0.8, np.nan, 0.7, 0.6])
    b = np.array([0.5, 0.3, 0.9, np.nan, 0.2])           # two pairs contain a NaN
    res = paired_t(a, b, metric="faithfulness", arm_a="rag", arm_b="no_context")
    assert res.n == 3 and res.df == 2                    # post-drop count, not 5
    assert not np.isnan(res.p_value)
    assert not np.isnan(res.statistic)


def test_paired_t_detects_difference():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.6, 1.0, 40)
    b = a - rng.uniform(0.1, 0.3, 40)           # a strictly better
    res = paired_t(a, b, metric="faithfulness", arm_a="rag", arm_b="no_context")
    assert res.p_value < 0.05
    assert res.statistic > 0
    assert res.mean_a > res.mean_b


def test_bonferroni_adjust_bounded():
    p = np.array([0.01, 0.04, 0.30])
    adj = bonferroni_adjust(p)
    assert np.all(adj >= p)          # adjustment never lowers a p-value
    assert np.all(adj <= 1.0)
    assert adj[0] == pytest.approx(0.03)     # 0.01 x family size 3
    assert adj[2] == pytest.approx(0.90)


def test_config_rejects_same_judge_and_generator(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "seed: 1\n"
        "generator: {model: 'x/m', temperature: 0.2, max_tokens: 8, n_samples: 1}\n"
        "judge: {model: 'x/m', temperature: 0.0, max_tokens: 8}\n"
        "embedding: {model: 'e'}\n"
        "retrieval: {k: 2, chunk_level: paragraph}\n"
        "kb_size_sweep: [2]\n"
        "experiment: {dataset: pubmedqa, n_questions: 2, arms: [no_context]}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="judge.model must differ"):
        load_config(cfg)
