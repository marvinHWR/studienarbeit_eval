"""RAGAS metrics wrapper.

Cross-arm quality metrics scored against fixed gold. Two design decisions from the concept
plan are enforced here:
  1. The JUDGE LLM is passed in explicitly and must differ from the generator (guaranteed by
     config.load_config). RAGAS uses it for answer_correctness / answer_relevancy / faithfulness.
  2. GROUNDEDNESS (faithfulness) is scored against the FULL SOURCE CORPUS, not the per-arm
     context. This makes No-Context scorable and puts all three arms on one yardstick.

RAGAS's public API shifts between minor versions; pin the version in pyproject and adjust the
metric imports here if needed. This targets the 0.2.x `evaluate()` interface.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from ..llm.retry import with_retries


@lru_cache(maxsize=None)
def _rejects_temperature(model: str) -> bool:
    """Probe whether `model` rejects the `temperature` param outright (e.g. claude-sonnet-5).

    Cached per model (probed once) and retried on transient provider errors via the shared
    `with_retries` helper, so a one-off 429/503 doesn't crash the end-of-run scoring pass.
    """
    import litellm

    try:
        with_retries(
            lambda: litellm.completion(
                model=model,
                messages=[{"role": "user", "content": "ok"}],
                max_tokens=1,
                temperature=0.0,
            ),
            litellm=litellm,
            label=model,
        )
        return False
    except litellm.BadRequestError as e:
        if "temperature" in str(e).lower():
            return True
        raise


def _build_judge(model: str, temperature: float, embedding_model: str):
    """Wrap a litellm model + local embeddings as RAGAS judge components."""
    from langchain_community.chat_models import ChatLiteLLM
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    llm_cls = ChatLiteLLM
    # Some newer models (e.g. claude-sonnet-5) reject the temperature parameter outright.
    # RAGAS's LangchainLLMWrapper mutates `.temperature` on the wrapped model directly before
    # every call (see ragas.llms.LangchainLLMWrapper.agenerate_text), bypassing whatever we
    # configure here -- so a subclass that drops `temperature` from the outgoing request
    # params is the only way to keep it from ever reaching the provider.
    if _rejects_temperature(model):
        print(f"[ragas] {model} rejects the temperature parameter; omitting it from all calls.")

        class _NoTemperatureChatLiteLLM(ChatLiteLLM):
            @property
            def _default_params(self) -> dict:
                params = dict(super()._default_params)
                params.pop("temperature", None)
                return params

        llm_cls = _NoTemperatureChatLiteLLM
        temperature = None

    judge_llm = LangchainLLMWrapper(llm_cls(model=model, temperature=temperature))
    judge_emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=embedding_model))
    return judge_llm, judge_emb


def score_ragas(
    rows: list[dict[str, Any]],
    judge_model: str,
    judge_temperature: float,
    embedding_model: str,
    metric_names: list[str] | None = None,
    contexts_key: str = "full_corpus",
) -> pd.DataFrame:
    """Score a batch of answers.

    Each row must provide:
      question, answer, ground_truth (gold answer), and `contexts_key` (list[str] reference).
    `metric_names` selects a subset of {"answer_correctness", "answer_relevancy",
    "faithfulness"}; defaults to all three. Datasets whose gold is a categorical label
    (PubMedQA yes/no/maybe) should drop "answer_correctness" — it compares free-text
    semantic similarity against `ground_truth` and doesn't apply to a label gold.

    `contexts_key` names the row field used as the groundedness reference. It defaults to
    `"full_corpus"` (the fixed full corpus, identical across arms — the standard cross-arm
    faithfulness). Pass `"contexts"` to score faithfulness against each arm's *actual* context
    instead (the strict "grounded in what it retrieved" measure).
    Returns a per-row dataframe with the requested metric columns.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_correctness, answer_relevancy, faithfulness

    registry = {
        "answer_correctness": answer_correctness,
        "answer_relevancy": answer_relevancy,
        "faithfulness": faithfulness,
    }
    metrics = [registry[m] for m in (metric_names or registry.keys())]

    judge_llm, judge_emb = _build_judge(judge_model, judge_temperature, embedding_model)

    ds = Dataset.from_dict(
        {
            "question": [r["question"] for r in rows],
            "answer": [r["answer"] for r in rows],
            "ground_truth": [r["ground_truth"] for r in rows],
            # Groundedness reference, selected by `contexts_key`: the fixed full corpus
            # (default, identical across arms) or each arm's actual retrieved context.
            "contexts": [list(r[contexts_key]) for r in rows],
        }
    )

    result = evaluate(
        ds,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_emb,
    )
    return result.to_pandas()
