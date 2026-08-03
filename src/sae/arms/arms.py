"""The three context strategies.

The experimental control: identical model, identical prompt template, identical questions —
ONLY the text placed in the {context} slot differs across arms.
  - no_context   -> empty context (parametric knowledge only)
  - full_context -> all paragraphs (Full-Context-Stuffing, no retrieval)
  - rag          -> retrieved top-k chunks only
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..data.schema import Record, LABEL_DATASETS
from ..llm.client import LLMClient, LLMResult
from ..retrieval.retriever import Retriever, Chunk

ARMS = ("no_context", "full_context", "rag")

SYSTEM = (
    "You are a precise question-answering assistant. Answer the question as briefly as "
    "possible. If context is provided, rely on it; if the answer is not present, say you "
    "do not know. Do not add explanations unless asked."
)

# Categorical (yes/no/maybe) datasets get their own system prompt: the model must LEAD with an
# extractable label AND justify it. The label makes `metrics.accuracy.extract_label` reliable
# (first token wins); the justification keeps the RAGAS answer-quality metrics meaningful. This
# prompt is identical across all three arms, so the experimental control still holds — only the
# {context_block} differs per arm. (Replaces the generic "say you do not know", which otherwise
# turns hedged answers into unparseable auto-zeros and suppresses the "maybe" class entirely.)
LABEL_SYSTEM = (
    "You are a precise question-answering assistant. If context is provided, rely on it. "
    "This question has a yes/no/maybe answer: begin your response with exactly one word — "
    "yes, no, or maybe — then give a brief one- to two-sentence justification. If the "
    "evidence is mixed or insufficient to decide, answer maybe."
)


def system_prompt_for(dataset: str | None) -> str:
    """The system prompt for `dataset`: the label-leading prompt for yes/no/maybe datasets,
    otherwise the generic brief-answer prompt."""
    if dataset and dataset.lower() in LABEL_DATASETS:
        return LABEL_SYSTEM
    return SYSTEM

# Single template shared by all arms. The context block is the only thing that varies.
USER_TEMPLATE = "{context_block}Question: {question}\nAnswer:"


@dataclass
class ArmOutput:
    arm: str
    answer: str
    contexts: list[str]              # what was actually in the prompt (for RAGAS)
    retrieved_idx: list[int] = field(default_factory=list)
    result: LLMResult | None = None


def _context_block(arm: str, record: Record, retriever: Retriever | None, k: int):
    if arm == "no_context":
        return "", [], []
    if arm == "full_context":
        ctx = record.paragraphs
        block = "Context:\n" + record.full_context() + "\n\n"
        return block, ctx, list(range(len(record.paragraphs)))
    if arm == "rag":
        assert retriever is not None, "rag arm needs a retriever"
        chunks: list[Chunk] = retriever.retrieve(record.question, record.paragraphs, k)
        ctx = [c.text for c in chunks]
        block = "Context:\n" + "\n\n".join(f"[{i}] {t}" for i, t in enumerate(ctx)) + "\n\n"
        return block, ctx, [c.source_idx for c in chunks]
    raise ValueError(f"Unknown arm: {arm!r}")


def run_arm(
    arm: str,
    record: Record,
    client: LLMClient,
    retriever: Retriever | None = None,
    k: int = 4,
    seed: int | None = None,
    system: str = SYSTEM,
) -> ArmOutput:
    block, contexts, retrieved_idx = _context_block(arm, record, retriever, k)
    user = USER_TEMPLATE.format(context_block=block, question=record.question)
    result = client.generate(system=system, user=user, seed=seed)
    return ArmOutput(
        arm=arm,
        answer=result.text,
        contexts=contexts,
        retrieved_idx=retrieved_idx,
        result=result,
    )
