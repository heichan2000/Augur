"""Tests for app.observability — pricing lookup, cost arithmetic, and the
structured usage-logging record emitted on `"augur.observability"`.

TDD: written BEFORE app/observability.py exists.
"""
from __future__ import annotations

import logging

import pytest

from app.observability import compute_cost, log_turn_usage, price_for


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
