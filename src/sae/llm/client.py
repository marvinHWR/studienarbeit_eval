"""Provider-agnostic LLM client with cost/latency instrumentation.

No eval framework measures cost/latency, so we instrument at the call boundary. Streaming is
enabled to capture time-to-first-token (TTFT) separately from total latency. `litellm` gives
one interface across OpenAI / Anthropic / local models, so generator and judge can be
different providers without code changes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .retry import with_retries


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float           # time to first streamed token
    total_latency_s: float
    model: str
    # `completion_tokens` for a thinking model (e.g. gemini-3.5-flash) counts hidden reasoning
    # tokens that are billed but never streamed back as content — so a small `max_tokens` can be
    # exhausted by reasoning, chopping the visible answer mid-sentence. We keep the reasoning/text
    # split for honest cost reporting and `finish_reason` to detect that truncation.
    reasoning_tokens: int = 0
    text_tokens: int = 0
    finish_reason: str | None = None

    @property
    def truncated(self) -> bool:
        """The provider stopped at the token cap, not at a natural stop — the visible answer is
        very likely cut off. Raise the generator's max_tokens (thinking models need headroom for
        reasoning *plus* the answer)."""
        return self.finish_reason == "length"


class LLMClient:
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 512):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, system: str, user: str, seed: int | None = None) -> LLMResult:
        import litellm  # lazy: keeps the core harness importable without the LLM stack

        # Generic safety net: let litellm drop params a model is known not to support
        # rather than erroring. (Note: models that merely *lock* temperature to a default,
        # like claude-sonnet-5, aren't covered by this — the judge path handles that.)
        litellm.drop_params = True

        # litellm's built-in num_retries can't cover us here: with stream=True it defers the
        # HTTP connection to the first chunk iteration, so a 503/429 raised while *consuming*
        # the stream escapes the retry wrapper entirely. We retry the whole attempt ourselves
        # via the shared `with_retries` helper (backoff 2, 4, 8, 16, 30, 30 s).
        # Gemini's free tier returns 503 "high demand" on transient capacity spikes that are
        # usually short-lived, and long runs make thousands of calls — so back off and retry.
        return with_retries(
            lambda: self._stream_once(litellm, system, user, seed),
            litellm=litellm,
            label=self.model,
        )

    def _stream_once(self, litellm, system: str, user: str, seed: int | None) -> LLMResult:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        t0 = time.perf_counter()
        ttft: float | None = None
        parts: list[str] = []

        stream = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=seed,
            stream=True,
            stream_options={"include_usage": True},
        )

        prompt_tokens = completion_tokens = 0
        reasoning_tokens = text_tokens = 0
        finish_reason: str | None = None
        for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            delta = choice.delta.content if choice else None
            if delta:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                parts.append(delta)
            if choice is not None and getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason
            usage = getattr(chunk, "usage", None)
            if usage:
                prompt_tokens = usage.prompt_tokens or prompt_tokens
                completion_tokens = usage.completion_tokens or completion_tokens
                details = getattr(usage, "completion_tokens_details", None)
                if details:
                    reasoning_tokens = getattr(details, "reasoning_tokens", None) or reasoning_tokens
                    text_tokens = getattr(details, "text_tokens", None) or text_tokens

        total = time.perf_counter() - t0
        text = "".join(parts)

        # A thinking model can burn the whole max_tokens budget on hidden reasoning and get cut
        # off mid-answer (finish_reason='length'). Surface it loudly — a silently truncated answer
        # corrupts every downstream metric — so a long run can't quietly produce chopped answers.
        if finish_reason == "length":
            print(
                f"[llm] WARNING: {self.model} hit max_tokens={self.max_tokens} "
                f"(reasoning={reasoning_tokens}, visible text={text_tokens} tok) — the answer is "
                f"truncated. Raise generator.max_tokens."
            )

        # Fallback token counts if the provider omitted usage on the stream.
        if prompt_tokens == 0:
            prompt_tokens = litellm.token_counter(model=self.model, messages=messages)
        if completion_tokens == 0:
            completion_tokens = litellm.token_counter(
                model=self.model, text=text
            )

        return LLMResult(
            text=text.strip(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_s=ttft if ttft is not None else total,
            total_latency_s=total,
            model=self.model,
            reasoning_tokens=reasoning_tokens,
            text_tokens=text_tokens,
            finish_reason=finish_reason,
        )
