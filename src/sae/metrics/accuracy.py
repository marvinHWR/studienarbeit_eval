"""PubMedQA label metrics.

pqa_labeled answers are yes/no/maybe -> the correct metric is classification accuracy, not
RAGAS answer_correctness (which mismatches a 3-way label). The predicted label is extracted
from the free-text answer by keyword; the label-leading prompt (`arms.LABEL_SYSTEM`) puts the
decision first, so the first-token match is the model's actual decision.

Accuracy alone hides two failure modes, so this module also exposes:
  - `macro_f1` / `per_class_recall` — reveal minority-class ("maybe") collapse that overall
    accuracy masks (the majority class can carry a high accuracy while "maybe" is never predicted);
  - `is_abstention` — the unparseable/hedged answers that accuracy silently scores 0.
"""
from __future__ import annotations

import re

_LABELS = ("yes", "no", "maybe")

# The label-leading prompt (`arms.LABEL_SYSTEM`) forces the model to *begin* its answer with
# exactly one of yes/no/maybe, so the verdict is honored only in leading position — not scanned
# for anywhere. A bare `re.search` for the first token would misfire on a non-committal answer
# that merely *contains* a label word ("There is no clear consensus." -> spurious "no"), scoring
# it as a confident label and hiding a genuine abstention. We therefore match the verdict only at
# the start, tolerating leading whitespace / markdown / quotes / brackets and a short optional
# lead-in phrase ("Answer:", "The final answer is", "Verdict -"). Anything else -> abstention.
_WRAP = r'[\s>*_~#().\[\]:=\-"]*'
_PREFIX = r'(?:the\s+)?(?:final\s+)?(?:answer(?:\s+is)?|verdict|conclusion|response)'
_LEADING_LABEL = re.compile(rf"^{_WRAP}(?:{_PREFIX}{_WRAP})?(yes|no|maybe)\b")


def extract_label(answer: str) -> str | None:
    """The leading yes/no/maybe verdict, or None when the answer doesn't commit to one.

    Only a verdict in the *leading* position (after optional formatting/lead-in) counts — see the
    note above. Returns None (an abstention) for hedged or off-format answers.
    """
    m = _LEADING_LABEL.match(answer.strip().lower())
    return m.group(1) if m else None


def label_accuracy(prediction: str, gold_label: str) -> float:
    pred = extract_label(prediction)
    if pred is None:
        return 0.0
    return float(pred == gold_label.strip().lower())


def is_abstention(answer: str) -> bool:
    """True when no yes/no/maybe decision can be read from `answer`.

    This is exactly the set of answers `label_accuracy` auto-scores 0 (unparseable / hedged /
    "I do not know"). Reported as its own metric so a low accuracy driven by the model *declining
    to commit* is not confused with a low accuracy driven by confident wrong labels.
    """
    return extract_label(answer) is None


def _norm(preds: list, golds: list) -> tuple[list, list]:
    p = [None if x is None else str(x).strip().lower() for x in preds]
    g = [str(x).strip().lower() for x in golds]
    return p, g


def per_class_recall(preds: list, golds: list, labels: tuple = _LABELS) -> dict[str, float]:
    """Recall per class = fraction of gold-`c` items the model labelled `c`. NaN when class `c`
    is absent from the gold. `preds` are extracted labels and may be None (abstentions), which
    never match a class."""
    p, g = _norm(preds, golds)
    out: dict[str, float] = {}
    for c in labels:
        idx = [i for i, gt in enumerate(g) if gt == c]
        out[c] = float(sum(1 for i in idx if p[i] == c) / len(idx)) if idx else float("nan")
    return out


def macro_f1(preds: list, golds: list, labels: tuple = _LABELS) -> float:
    """Unweighted mean per-class F1 over the classes present in the gold. Abstentions (None
    predictions) count as false negatives for their gold class."""
    p, g = _norm(preds, golds)
    f1s = []
    for c in [c for c in labels if c in g]:
        tp = sum(1 for pi, gi in zip(p, g) if pi == c and gi == c)
        fp = sum(1 for pi, gi in zip(p, g) if pi == c and gi != c)
        fn = sum(1 for pi, gi in zip(p, g) if pi != c and gi == c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(sum(f1s) / len(f1s)) if f1s else 0.0
