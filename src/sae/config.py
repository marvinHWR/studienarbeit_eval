"""Config loading with the hard judge != generator guardrail."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelCfg:
    model: str
    temperature: float = 0.0
    max_tokens: int = 512
    n_samples: int = 1


@dataclass
class Config:
    seed: int
    generator: ModelCfg
    judge: ModelCfg
    embedding_model: str
    k: int
    chunk_level: str
    kb_size_sweep: list[int]
    dataset: str
    n_questions: int
    arms: list[str]
    paths: dict[str, str] = field(default_factory=dict)
    # Compute the strict per-arm `faithfulness_retrieved` column during scoring. Off on the full
    # run to save judge cost (an extra faithfulness pass over the two context arms ≈ +65% of the
    # faithfulness calls); scripts/rescore.py can still backfill it on demand. Defaults on so
    # existing/other configs keep their behavior.
    strict_faithfulness: bool = True


def load_config(path: str | Path = "config/default.yaml") -> Config:
    # Populate os.environ from .env so litellm (generator + judge) finds provider keys.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    gen = ModelCfg(**data["generator"])
    # `strict_faithfulness` is a scoring cost knob that lives next to the judge in YAML; pop it
    # before building ModelCfg (which doesn't accept the extra key).
    judge_data = dict(data["judge"])
    strict_faithfulness = bool(judge_data.pop("strict_faithfulness", True))
    judge = ModelCfg(**judge_data)

    # HARD RULE from the study design: the judge must not be the generator,
    # otherwise answer_correctness is contaminated by self-preference bias.
    if gen.model.strip().lower() == judge.model.strip().lower():
        raise ValueError(
            f"judge.model must differ from generator.model (both are '{gen.model}'). "
            "See concept plan: judge != generator is a validity requirement."
        )
    if "REPLACE" in gen.model or "REPLACE" in judge.model:
        raise ValueError(
            "Pin concrete, versioned models in config before running "
            "(generator/judge still contain the 'REPLACE' placeholder)."
        )

    return Config(
        seed=data["seed"],
        generator=gen,
        judge=judge,
        embedding_model=data["embedding"]["model"],
        k=data["retrieval"]["k"],
        chunk_level=data["retrieval"]["chunk_level"],
        kb_size_sweep=data["kb_size_sweep"],
        dataset=data["experiment"]["dataset"],
        n_questions=data["experiment"]["n_questions"],
        arms=data["experiment"]["arms"],
        paths=data.get("paths", {}),
        strict_faithfulness=strict_faithfulness,
    )
