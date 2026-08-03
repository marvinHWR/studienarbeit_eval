"""Unified record schema + dataset dispatch.

Every dataset normalizes to `Record`, so the arms, retrieval, and metrics code is
dataset-independent. This repository ships the PubMedQA loader (the experiment's dataset);
adding another dataset means adding a loader plus one line in `load_dataset`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Datasets whose gold is a categorical yes/no/maybe label. These are scored by classification
# accuracy (not EM/F1 or RAGAS answer_correctness) AND prompted to lead with an extractable
# label. Single source of truth, imported by arms + experiment.scoring so the two never drift.
LABEL_DATASETS = {"pubmedqa"}


@dataclass
class Record:
    id: str
    question: str
    gold_answer: str
    paragraphs: list[str]                       # the bounded per-question / pooled corpus
    gold_para_idx: list[int] = field(default_factory=list)   # indices of supporting paragraphs
    gold_label: str | None = None               # PubMedQA yes/no/maybe; None otherwise

    def full_context(self) -> str:
        """All paragraphs concatenated — the Full-Context-Stuffing arm's context."""
        return "\n\n".join(f"[{i}] {p}" for i, p in enumerate(self.paragraphs))


def load_dataset(name: str, n: int | None = None, seed: int = 42, **kwargs) -> list[Record]:
    """Dispatch to a concrete loader by name."""
    name = name.lower()
    if name == "pubmedqa":
        from .pubmedqa import load_pubmedqa
        return load_pubmedqa(n=n, seed=seed, **kwargs)
    raise ValueError(f"Unknown dataset: {name!r}")
