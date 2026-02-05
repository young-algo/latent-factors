"""Tests for walk-forward backtest engine correctness."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.signal_backtest import SignalBacktester


@dataclass
class _MockConsensus:
    consensus_direction: SimpleNamespace
    confidence: float
    consensus_score: float


class _MockAggregator:
    def __init__(self, signal_map: dict[pd.Timestamp, dict[str, tuple[str, float]]]):
        self.signal_map = signal_map
        self.called_dates: list[pd.Timestamp] = []

    def aggregate_signals(self, date=None):
        ts = pd.Timestamp(date)
        self.called_dates.append(ts)
        payload = self.signal_map.get(ts, {})
        out = {}
        for entity, (direction, confidence) in payload.items():
            out[entity] = _MockConsensus(
                consensus_direction=SimpleNamespace(value=direction),
                confidence=confidence,
                consensus_score=100.0,
            )
        return out


def test_backtester_uses_t_plus_1_entity_returns_and_as_of_dates():
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    returns = pd.DataFrame(
        {
            "A": [0.0, 0.0, 0.01, 0.02, -0.03, 0.01],
            "B": [0.0, 0.0, -0.01, 0.00, 0.01, -0.02],
        },
        index=dates,
    )

    # Test window will be dates[2:5], so tradable signal dates are dates[2], dates[3].
    signal_map = {
        dates[2]: {"A": ("buy", 90.0)},
        dates[3]: {"A": ("sell", 90.0)},
    }
    aggregator = _MockAggregator(signal_map)

    backtester = SignalBacktester(aggregator, returns, transaction_costs=0.0)
    summary = backtester.run_backtest(train_size=2, test_size=3, n_walks=1, min_confidence=0.0)

    assert summary["num_walks"] == 1.0
    assert aggregator.called_dates == [dates[2], dates[3]]

    walk = backtester.results[0]
    # Signal at dates[2] is applied to return at dates[3]: +0.02
    # Signal at dates[3] is applied to return at dates[4]: short A => +0.03
    expected = pd.Series([0.02, 0.03], index=[dates[3], dates[4]])
    realized = walk.equity_curve.pct_change()
    realized.iloc[0] = walk.equity_curve.iloc[0] - 1.0
    pd.testing.assert_series_equal(realized, expected, check_names=False)


def test_turnover_costs_are_applied_from_weight_changes():
    dates = pd.date_range("2024-02-01", periods=4, freq="B")
    test_returns = pd.DataFrame(
        {"A": [0.0, 0.01, 0.01, 0.01]},
        index=dates,
    )

    aggregator = _MockAggregator({})
    backtester = SignalBacktester(aggregator, test_returns, transaction_costs=0.001)

    # Two signal dates: dates[0], dates[1].
    signals = {
        dates[0]: [{"entity": "A", "direction": "buy", "confidence": 100.0, "score": 100.0}],
        dates[1]: [{"entity": "A", "direction": "sell", "confidence": 100.0, "score": 100.0}],
    }

    strategy_returns, _ = backtester._calculate_strategy_returns(test_returns.iloc[:3], signals)

    # Day 1: flat -> +1 weight, turnover 1.0 => 10 bps cost.
    # Net return: 0.01 - 0.001 = 0.009
    # Day 2: +1 -> -1 weight, turnover 2.0 => 20 bps cost.
    # Net return: (-0.01) - 0.002 = -0.012
    np.testing.assert_allclose(strategy_returns.values, np.array([0.009, -0.012]), rtol=1e-10, atol=1e-10)
