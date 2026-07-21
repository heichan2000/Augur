"""Tests for app.observability — pricing lookup, cost arithmetic, the
structured usage-logging record emitted on `"augur.observability"`, and the
`configure_logging`/`StructuredFormatter` handler wiring.

TDD: written BEFORE app/observability.py exists.
"""
from __future__ import annotations

import json
import logging

import pytest

from app.observability import (
    StructuredFormatter,
    compute_cost,
    configure_logging,
    log_turn_usage,
    price_for,
)


# ---------------------------------------------------------------------------
# Behavior 1: compute_cost arithmetic is exact
# ---------------------------------------------------------------------------


def test_compute_cost_round_million_tokens():
    cost = compute_cost(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        input_price_per_mtok=3.0,
        output_price_per_mtok=15.0,
    )
    assert cost == pytest.approx(18.0)


def test_compute_cost_non_round_counts():
    cost = compute_cost(
        input_tokens=500,
        output_tokens=250,
        input_price_per_mtok=3.0,
        output_price_per_mtok=15.0,
    )
    # (500/1e6)*3.0 + (250/1e6)*15.0 == 0.0015 + 0.00375 == 0.00525
    assert cost == pytest.approx(0.00525)


# ---------------------------------------------------------------------------
# Behavior 2: price_for returns configured prices, None for unknown models
# ---------------------------------------------------------------------------


def test_price_for_known_model():
    assert price_for("claude-sonnet-4-6") == (3.0, 15.0)


def test_price_for_unknown_model_returns_none():
    assert price_for("nope") is None


# ---------------------------------------------------------------------------
# Behavior 3: log_turn_usage emits one structured record with cost
# ---------------------------------------------------------------------------


def test_log_turn_usage_emits_record_with_cost_for_known_model(caplog):
    with caplog.at_level(logging.INFO, logger="augur.observability"):
        log_turn_usage(
            session_id="s1",
            model="claude-sonnet-4-6",
            input_tokens=500,
            output_tokens=250,
        )

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.session_id == "s1"
    assert record.model == "claude-sonnet-4-6"
    assert record.input_tokens == 500
    assert record.output_tokens == 250
    assert record.cost_usd == pytest.approx(
        compute_cost(
            input_tokens=500,
            output_tokens=250,
            input_price_per_mtok=3.0,
            output_price_per_mtok=15.0,
        )
    )


# ---------------------------------------------------------------------------
# Behavior 4: log_turn_usage degrades to cost_usd=None for unknown models
# ---------------------------------------------------------------------------


def test_log_turn_usage_unknown_model_does_not_crash(caplog):
    with caplog.at_level(logging.INFO, logger="augur.observability"):
        log_turn_usage(
            session_id="s1",
            model="unpriced-model",
            input_tokens=10,
            output_tokens=20,
        )

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.cost_usd is None
    assert record.input_tokens == 10
    assert record.output_tokens == 20


# ---------------------------------------------------------------------------
# Fixture: snapshot/restore the "augur" logger around configure_logging tests
# so this test module doesn't leak handlers/level/propagate into the rest of
# the suite (other tests rely on caplog against "augur.observability").
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_augur_logger():
    logger = logging.getLogger("augur")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    try:
        yield logger
    finally:
        logger.handlers = original_handlers
        logger.level = original_level
        logger.propagate = original_propagate


def _structured_handlers(logger: logging.Logger) -> list[logging.Handler]:
    """Handlers configure_logging is responsible for (its own formatter type).

    Filters out anything else that may be attached to the "augur" logger —
    notably pytest's own log-capturing plugin, which (see
    `_pytest.logging.catching_logs`) deliberately attaches its transient
    per-test handlers to any *non-propagating* logger, and `configure_logging`
    intentionally makes "augur" non-propagating. Those handlers are pytest's
    business, not ours, so the idempotency guarantee is scoped to our own
    handler type rather than the full (environment-dependent) handler count.
    """
    return [h for h in logger.handlers if isinstance(h.formatter, StructuredFormatter)]


# ---------------------------------------------------------------------------
# Behavior 5: configure_logging attaches exactly one handler, idempotently
# ---------------------------------------------------------------------------


def test_configure_logging_attaches_one_handler_at_info_level(clean_augur_logger):
    configure_logging()

    assert len(_structured_handlers(clean_augur_logger)) == 1
    assert clean_augur_logger.level == logging.INFO
    assert clean_augur_logger.propagate is False


def test_configure_logging_is_idempotent(clean_augur_logger):
    configure_logging()
    configure_logging()
    configure_logging()

    assert len(_structured_handlers(clean_augur_logger)) == 1


# ---------------------------------------------------------------------------
# Behavior 6: StructuredFormatter renders whitelisted extra= fields as JSON
# ---------------------------------------------------------------------------


def _make_record(*, msg: str, level: int, extra: dict) -> logging.LogRecord:
    record = logging.LogRecord(
        name="augur.observability",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=None,
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_formatter_renders_usage_fields_as_json():
    formatter = StructuredFormatter()
    record = _make_record(
        msg="turn usage",
        level=logging.INFO,
        extra={
            "session_id": "s1",
            "model": "claude-sonnet-4-6",
            "input_tokens": 500,
            "output_tokens": 250,
            "cost_usd": 0.00525,
        },
    )

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "augur.observability"
    assert payload["message"] == "turn usage"
    assert payload["session_id"] == "s1"
    assert payload["model"] == "claude-sonnet-4-6"
    assert payload["input_tokens"] == 500
    assert payload["output_tokens"] == 250
    assert payload["cost_usd"] == pytest.approx(0.00525)


def test_formatter_renders_error_fields_only_as_json():
    formatter = StructuredFormatter()
    record = _make_record(
        msg="turn error",
        level=logging.WARNING,
        extra={"session_id": "s1", "error_type": "rate_limit"},
    )

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "WARNING"
    assert payload["logger"] == "augur.observability"
    assert payload["message"] == "turn error"
    assert payload["session_id"] == "s1"
    assert payload["error_type"] == "rate_limit"
    assert "input_tokens" not in payload
    assert "output_tokens" not in payload
    assert "model" not in payload
    assert "cost_usd" not in payload


# ---------------------------------------------------------------------------
# Behavior 7: StructuredFormatter does not leak non-whitelisted attributes
# ---------------------------------------------------------------------------


def test_formatter_does_not_leak_unwhitelisted_attributes():
    formatter = StructuredFormatter()
    record = _make_record(
        msg="turn error",
        level=logging.WARNING,
        extra={"session_id": "s1", "error_type": "internal", "secret": "x"},
    )

    payload = json.loads(formatter.format(record))

    assert "secret" not in payload
    assert "pathname" not in payload
    assert "lineno" not in payload
    assert "args" not in payload
    assert "exc_info" not in payload
    # Only the expected keys are present.
    assert set(payload.keys()) == {
        "level",
        "logger",
        "message",
        "session_id",
        "error_type",
    }


# ---------------------------------------------------------------------------
# Behavior 8: StructuredFormatter renders tool_name
# ---------------------------------------------------------------------------


def test_formatter_renders_tool_name_field():
    formatter = StructuredFormatter()
    record = _make_record(
        msg="unknown tool requested",
        level=logging.WARNING,
        extra={"tool_name": "nonexistent"},
    )

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "WARNING"
    assert payload["message"] == "unknown tool requested"
    assert payload["tool_name"] == "nonexistent"
    assert set(payload.keys()) == {"level", "logger", "message", "tool_name"}
