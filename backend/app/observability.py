"""Structured token/cost observability for chat turns.

Emits one structured INFO log record per successfully completed turn (with
computed cost) and one structured WARNING record per failed turn, both via
the ``"augur.observability"`` logger. Cost computation is pure arithmetic
and pricing lookup degrades gracefully for unknown models — neither may
crash the request that triggered it. `configure_logging` wires a minimal
JSON-line handler onto the parent ``"augur"`` logger so these records are
actually emitted (and readable) in a default deployment.
"""
from __future__ import annotations

import json
import logging

logger = logging.getLogger("augur.observability")

# The only record attributes a structured-log consumer may see, beyond the
# standard level/logger/message fields. Keeps secrets and log-record noise
# (pathname, args, exc_info, ...) out of the emitted JSON.
_STRUCTURED_FIELDS = (
    "session_id",
    "model",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "error_type",
)


class StructuredFormatter(logging.Formatter):
    """Renders a LogRecord as a single JSON line.

    Includes `level`, `logger`, `message`, plus whichever of the whitelisted
    observability fields (`_STRUCTURED_FIELDS`) are present on the record.
    Fields absent on a given record are omitted — no arbitrary record
    attributes are ever included.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in _STRUCTURED_FIELDS:
            if field in record.__dict__:
                payload[field] = record.__dict__[field]
        return json.dumps(payload)


def configure_logging(level: int = logging.INFO) -> None:
    """Attach a single structured stderr handler to the "augur" logger.

    Idempotent: calling this more than once (e.g. because `create_app()`
    runs at import time and tests build many apps) does not stack duplicate
    handlers. Sets `propagate = False` so the root logger — which a default
    deployment leaves unconfigured at WARNING — doesn't also see (and drop,
    or double-emit) these records.
    """
    target = logging.getLogger("augur")
    target.setLevel(level)
    target.propagate = False

    already_configured = any(
        isinstance(handler.formatter, StructuredFormatter) for handler in target.handlers
    )
    if already_configured:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(StructuredFormatter())
    target.addHandler(handler)


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


def log_persist_timeout(*, session_id: str, model: str) -> None:
    """Emit one structured ERROR record for a turn whose write did not land.

    Reached only from the stopped path in ``app.chat``, when the shielded
    persist does not finish inside its timeout. The turn's content is gone
    at that point — the session id and model are what is left to identify
    which turn was lost, so both go on the record.
    """
    logger.error(
        "turn persist timeout",
        extra={
            "session_id": session_id,
            "model": model,
            "error_type": "persist_timeout",
        },
    )


def log_turn_error(*, session_id: str, error_type: str) -> None:
    """Emit one structured WARNING record for a chat turn that failed.

    No message content or traceback text is logged — only the session id and
    the machine-readable `error_type` (matching the SSE `error` event's
    `type` field), attached via logging `extra=`.
    """
    logger.warning(
        "turn error",
        extra={
            "session_id": session_id,
            "error_type": error_type,
        },
    )
