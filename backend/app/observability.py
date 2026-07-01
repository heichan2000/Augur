"""Structured token/cost observability for completed chat turns.

Emits one structured INFO log record per successfully completed turn via
the ``"augur.observability"`` logger. Cost computation is pure arithmetic
and pricing lookup degrades gracefully for unknown models — neither may
crash the request that triggered it.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("augur.observability")

# USD per 1,000,000 tokens, per model. Anthropic list prices — maintained by
# hand; edit when the configured model or its pricing changes. Source of truth
# is Anthropic's pricing page.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # model id: (input_usd_per_mtok, output_usd_per_mtok)
    "claude-sonnet-4-6": (3.0, 15.0),
}


def compute_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    input_price_per_mtok: float,
    output_price_per_mtok: float,
) -> float:
    """USD cost for a turn given per-million-token prices. Pure arithmetic."""
    return (input_tokens / 1_000_000) * input_price_per_mtok + (
        output_tokens / 1_000_000
    ) * output_price_per_mtok


def price_for(model: str) -> tuple[float, float] | None:
    """Return (input, output) per-Mtok prices for `model`, or None if unknown."""
    return MODEL_PRICING.get(model)


def log_turn_usage(
    *, session_id: str, model: str, input_tokens: int, output_tokens: int
) -> None:
    """Emit one structured INFO record of a completed turn's token usage + cost.

    Looks up `model` in MODEL_PRICING; if found, includes computed `cost_usd`,
    else `cost_usd` is None (unknown-model pricing must NOT crash the request).
    Attaches fields via logging `extra=` so structured-log consumers and tests
    (caplog) can read them off the record.
    """
    prices = price_for(model)
    cost_usd: float | None = None
    if prices is not None:
        input_price, output_price = prices
        cost_usd = compute_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_price_per_mtok=input_price,
            output_price_per_mtok=output_price,
        )

    logger.info(
        "turn usage",
        extra={
            "session_id": session_id,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
        },
    )
