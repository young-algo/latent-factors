"""Leakage guard tests for as-of signal/regime boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.cross_sectional import SignalDirection as CrossSignalDirection
from src.cross_sectional import StockSignal
from src.regime_detection import MarketRegime, RegimeAllocation, RegimeDetector
from src.signal_aggregator import SignalAggregator


class _MockMomentumAnalyzer:
    def __init__(self):
        self.signal_dates: list[pd.Timestamp] = []
        self.alert_dates: list[pd.Timestamp] = []
        self.factor_returns = pd.DataFrame(columns=["F1"])

    def get_all_signals(self, date=None):
        self.signal_dates.append(pd.Timestamp(date))
        return {
            "F1": {
                "combined_signal": "buy",
                "rsi": 55.0,
                "rsi_signal": "neutral",
                "macd_signal": "bullish",
                "adx": 25.0,
                "regime": "trending_up",
            }
        }

    def get_all_extreme_alerts(self, date=None):
        self.alert_dates.append(pd.Timestamp(date))
        return []


class _MockCrossSectionalAnalyzer:
    def __init__(self):
        self.calls: list[pd.Timestamp] = []
        self.factor_loadings = pd.DataFrame({"F1": [1.0]}, index=["AAA"])

    def generate_long_short_signals(self, top_pct=0.1, bottom_pct=0.1, scores=None, min_confidence=0.0, as_of=None):
        self.calls.append(pd.Timestamp(as_of))
        return [
            StockSignal(
                ticker="AAA",
                direction=CrossSignalDirection.LONG,
                composite_score=1.2,
                decile=1,
                rank=1,
                total_stocks=1,
                factor_breakdown={"F1": 1.0},
                confidence=0.9,
            )
        ]


class _MockRegimeDetector:
    def __init__(self):
        self.calls: list[pd.Timestamp] = []

    def generate_regime_signals(self, as_of=None):
        self.calls.append(pd.Timestamp(as_of))
        return RegimeAllocation(
            regime=MarketRegime.LOW_VOL_BULL,
            factor_weights={"F1": 1.0},
            risk_on_score=0.8,
            defensive_tilt=False,
            recommended_action="Risk on",
        )


def test_signal_aggregator_propagates_as_of_date_to_all_sources():
    aggregator = SignalAggregator(factor_research_system=SimpleNamespace())
    momentum = _MockMomentumAnalyzer()
    cross = _MockCrossSectionalAnalyzer()
    regime = _MockRegimeDetector()

    aggregator.add_momentum_signals(momentum)
    aggregator.add_cross_sectional_signals(cross)
    aggregator.add_regime_signals(regime)

    as_of = pd.Timestamp("2024-06-03")
    consensus = aggregator.aggregate_signals(date=as_of)

    assert momentum.signal_dates == [as_of]
    assert momentum.alert_dates == [as_of]
    assert cross.calls == [as_of]
    assert regime.calls == [as_of]
    assert "AAA" in consensus
    assert "F1" in consensus

    # Leakage guard: all signal timestamps should match the requested as-of date.
    for entry in consensus.values():
        for signal in entry.contributing_signals:
            assert pd.Timestamp(signal.timestamp) == as_of


class _IdentityScaler:
    def transform(self, X):
        return X


class _ThresholdHMM:
    """Simple HMM stub: positive latest return -> state 0, else state 1."""

    n_components = 2
    means_ = np.array([[0.0], [0.0]])
    covars_ = np.array([[[1.0]], [[1.0]]])
    transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])

    def predict_proba(self, X):
        latest = float(X[0, 0])
        if latest >= 0:
            return np.array([[0.9, 0.1]])
        return np.array([[0.1, 0.9]])


def test_regime_detector_respects_as_of_boundaries():
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    returns = pd.DataFrame({"F1": [0.01, 0.02, 0.015, 0.01, 0.005, -0.40]}, index=dates)

    detector = RegimeDetector(returns)
    detector.hmm_model = _ThresholdHMM()
    detector.scaler = _IdentityScaler()
    detector.regime_labels = {
        0: MarketRegime.LOW_VOL_BULL,
        1: MarketRegime.LOW_VOL_BEAR,
    }

    early = detector.detect_current_regime(as_of=dates[3])
    late = detector.detect_current_regime(as_of=dates[-1])

    assert early.regime == MarketRegime.LOW_VOL_BULL
    assert late.regime == MarketRegime.LOW_VOL_BEAR
    assert early.trend > 0
    assert late.trend < 0

    early_probs = detector.get_regime_probabilities(as_of=dates[3])
    late_probs = detector.get_regime_probabilities(as_of=dates[-1])
    assert early_probs[MarketRegime.LOW_VOL_BULL] > early_probs[MarketRegime.LOW_VOL_BEAR]
    assert late_probs[MarketRegime.LOW_VOL_BEAR] > late_probs[MarketRegime.LOW_VOL_BULL]


def test_regime_detector_as_of_before_history_raises():
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    returns = pd.DataFrame({"F1": [0.01, 0.02, 0.03]}, index=dates)

    detector = RegimeDetector(returns)
    detector.hmm_model = _ThresholdHMM()
    detector.scaler = _IdentityScaler()
    detector.regime_labels = {0: MarketRegime.LOW_VOL_BULL, 1: MarketRegime.LOW_VOL_BEAR}

    with np.testing.assert_raises(ValueError):
        detector.detect_current_regime(as_of=dates[0] - pd.Timedelta(days=1))
