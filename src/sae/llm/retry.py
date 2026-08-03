"""Transient-error retry with exponential backoff, shared across every litellm call site.

Long runs make thousands of provider calls; free/low tiers return short-lived 429/503/500s.
Both the generation path (`LLMClient`) and the RAGAS judge probe (`metrics.ragas_eval`) wrap
their calls in `with_retries` so a transient hiccup backs off and retries instead of crashing
a paid run. litellm is passed in (imported lazily at each call site) to keep the core harness
importable without the LLM stack.
"""
from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")


def transient_error_types(litellm) -> tuple[type[Exception], ...]:
    """The provider errors worth retrying — capacity/quota/connection hiccups, not bad requests."""
    return (
        litellm.ServiceUnavailableError,   # 503 high-demand (free-tier deprioritization)
        litellm.RateLimitError,            # 429 quota / rate limit
        litellm.InternalServerError,       # 500 provider hiccup
        litellm.APIConnectionError,        # dropped connection / DNS
        litellm.Timeout,
    )


def with_retries(
    fn: Callable[[], T], *, litellm, label: str, max_attempts: int = 6
) -> T:
    """Call `fn`, retrying transient provider errors with capped exponential backoff.

    Backoff schedule: 2, 4, 8, 16, 30, 30 s. Non-transient errors propagate immediately.
    `label` (e.g. the model name) is only used in the retry log line.
    """
    transient = transient_error_types(litellm)
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except transient as e:
            last_exc = e
            if attempt == max_attempts - 1:
                break
            delay = min(2 ** (attempt + 1), 30)
            print(
                f"[llm] {type(e).__name__} from {label} "
                f"(attempt {attempt + 1}/{max_attempts}); retrying in {delay}s..."
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]
