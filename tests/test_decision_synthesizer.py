"""Tests for Decision Synthesizer module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.decision_synthesizer import (
    SignalState, RegimeState, FactorMomentum,
    Recommendation, TradeExpression, ConvictionLevel, ActionCategory,
    ConvictionScorer
)


class TestSignalState:
    """Tests for SignalState data structure."""

    def test_signal_state_creation(self):
        """SignalState should hold all signal components."""
        regime = RegimeState(
            name="Low-Vol Bull",
            confidence=0.78,
            days_in_regime=12,
            trend="strengthening"
        )

        momentum = [
            FactorMomentum(factor="F1", name="Tech-Momentum", return_7d=0.021, strength="strong"),
            FactorMomentum(factor="F2", name="Quality", return_7d=0.008, strength="moderate"),
        ]

        state = SignalState(
            date=datetime(2026, 2, 4),
            regime=regime,
            factor_momentum=momentum,
            extremes_detected=[],
            cross_sectional_spread=1.2
        )

        assert state.regime.name == "Low-Vol Bull"
        assert state.regime.confidence == 0.78
        assert len(state.factor_momentum) == 2
        assert state.factor_momentum[0].strength == "strong"


class TestRecommendation:
    """Tests for Recommendation data structure."""

    def test_recommendation_creation(self):
        """Recommendation should capture action with full context."""
        rec = Recommendation(
            action="Increase Tech-Momentum exposure",
            conviction=ConvictionLevel.HIGH,
            conviction_score=8.2,
            category=ActionCategory.OPPORTUNISTIC,
            reasons=[
                "Regime: Low-Vol Bull favors momentum factors",
                "Signal: Tech-Momentum +2.1% over 7 days (z-score: 1.8)",
                "Confirmation: Cross-sectional ranks show tech in top decile"
            ],
            conflicts=[],
            expressions=[
                TradeExpression(description="Simple", trade="Buy QQQ", size_pct=0.05),
                TradeExpression(description="Targeted", trade="NVDA, MSFT, AAPL, AMD, AVGO", size_pct=0.01),
            ],
            exit_trigger="Regime flips to High-Vol or Bear"
        )

        assert rec.conviction == ConvictionLevel.HIGH
        assert rec.conviction_score == 8.2
        assert rec.category == ActionCategory.OPPORTUNISTIC
        assert len(rec.reasons) == 3
        assert len(rec.expressions) == 2


class TestConvictionScoring:
    """Tests for conviction scoring logic."""

    def test_high_conviction_all_aligned(self):
        """All signals aligned should produce HIGH conviction."""
        scorer = ConvictionScorer()

        score = scorer.calculate(
            signal_strength=1.8,  # z-score
            signal_agreement=3,   # 3 of 3 signals agree
            total_signals=3,
            regime_fit=True
        )

        assert score >= 8.0
        assert scorer.to_level(score) == ConvictionLevel.HIGH

    def test_low_conviction_conflicting(self):
        """Conflicting signals should produce LOW or CONFLICTED."""
        scorer = ConvictionScorer()

        score = scorer.calculate(
            signal_strength=1.5,
            signal_agreement=1,  # Only 1 of 3 agree
            total_signals=3,
            regime_fit=False
        )

        assert score < 5.0
        assert scorer.to_level(score) in [ConvictionLevel.LOW, ConvictionLevel.CONFLICTED]

    def test_medium_conviction_partial_agreement(self):
        """Partial agreement should produce MEDIUM conviction."""
        scorer = ConvictionScorer()

        score = scorer.calculate(
            signal_strength=1.2,
            signal_agreement=2,  # 2 of 3 agree
            total_signals=3,
            regime_fit=True
        )

        assert 5.0 <= score < 8.0
        assert scorer.to_level(score) == ConvictionLevel.MEDIUM
