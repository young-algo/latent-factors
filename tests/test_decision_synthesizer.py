"""Tests for Decision Synthesizer module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.decision_synthesizer import (
    SignalState, RegimeState, FactorMomentum,
    Recommendation, TradeExpression, ConvictionLevel, ActionCategory,
    ConvictionScorer, DecisionSynthesizer
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


class TestSignalAlignment:
    """Tests for signal alignment scoring."""

    def test_perfect_alignment(self):
        """All bullish signals should give 9-10 alignment."""
        scorer = ConvictionScorer()

        alignment = scorer.calculate_alignment(
            regime_bullish=True,
            momentum_bullish=[True, True, True],  # 3 factors all bullish
            cross_section_bullish=True
        )

        assert alignment >= 9

    def test_mixed_signals(self):
        """Mixed signals should give 4-6 alignment."""
        scorer = ConvictionScorer()

        alignment = scorer.calculate_alignment(
            regime_bullish=True,
            momentum_bullish=[True, False, True],  # 2 of 3 bullish
            cross_section_bullish=False
        )

        assert 4 <= alignment <= 6

    def test_conflicting_signals(self):
        """Regime vs other signals conflicting should give 1-3."""
        scorer = ConvictionScorer()

        alignment = scorer.calculate_alignment(
            regime_bullish=False,  # Regime says bearish
            momentum_bullish=[True, True, True],  # But momentum bullish
            cross_section_bullish=True
        )

        assert alignment <= 4  # Regime conflict penalized heavily


class TestSignalCollection:
    """Tests for collecting signals from existing modules."""

    def test_collect_signals_returns_signal_state(self):
        """collect_all_signals should return complete SignalState."""
        synthesizer = DecisionSynthesizer()

        # Create mock factor returns
        dates = pd.date_range('2025-01-01', periods=100, freq='D')
        factor_returns = pd.DataFrame({
            'F1': np.random.randn(100) * 0.02,
            'F2': np.random.randn(100) * 0.02,
        }, index=dates)

        # Create mock factor loadings
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META']
        factor_loadings = pd.DataFrame({
            'F1': np.random.randn(5),
            'F2': np.random.randn(5),
        }, index=tickers)

        state = synthesizer.collect_all_signals(
            factor_returns=factor_returns,
            factor_loadings=factor_loadings,
            factor_names={'F1': 'Tech-Momentum', 'F2': 'Quality'}
        )

        assert isinstance(state, SignalState)
        assert state.regime is not None
        assert len(state.factor_momentum) == 2
        assert state.date is not None


class TestRecommendationGeneration:
    """Tests for generating recommendations from signals."""

    def test_generates_opportunistic_on_regime_shift(self):
        """Regime shift with high confidence should trigger OPPORTUNISTIC."""
        synthesizer = DecisionSynthesizer()

        state = SignalState(
            date=datetime.now(),
            regime=RegimeState(
                name="Low-Vol Bull",
                confidence=0.85,
                days_in_regime=3,
                trend="strengthening"
            ),
            factor_momentum=[
                FactorMomentum("F1", "Tech-Momentum", 0.025, "strong"),
                FactorMomentum("F2", "Quality", 0.010, "moderate"),
            ],
            extremes_detected=[],
            cross_sectional_spread=1.5
        )

        recommendations = synthesizer.generate_recommendations(state)

        # Should have at least one recommendation
        assert len(recommendations) >= 1

        # High confidence regime + strong momentum should produce OPPORTUNISTIC
        opportunistic = [r for r in recommendations if r.category == ActionCategory.OPPORTUNISTIC]
        assert len(opportunistic) >= 1

    def test_generates_watch_on_low_conviction(self):
        """Low conviction signals should produce WATCH items."""
        synthesizer = DecisionSynthesizer()

        state = SignalState(
            date=datetime.now(),
            regime=RegimeState(
                name="Transition",
                confidence=0.45,
                days_in_regime=1,
                trend="unknown"
            ),
            factor_momentum=[
                FactorMomentum("F1", "Tech-Momentum", 0.003, "flat"),
            ],
            extremes_detected=[],
            cross_sectional_spread=0.5
        )

        recommendations = synthesizer.generate_recommendations(state)

        # Low conviction should result in WATCH, not action
        for rec in recommendations:
            assert rec.category == ActionCategory.WATCH or rec.conviction_score < 5.0


class TestBriefingRenderer:
    """Tests for morning briefing rendering."""

    def test_render_briefing_contains_sections(self):
        """Rendered briefing should contain all required sections."""
        synthesizer = DecisionSynthesizer()

        state = SignalState(
            date=datetime(2026, 2, 4),
            regime=RegimeState("Low-Vol Bull", 0.78, 12, "strengthening"),
            factor_momentum=[
                FactorMomentum("F1", "Tech-Momentum", 0.021, "strong"),
                FactorMomentum("F2", "Quality", 0.008, "moderate"),
            ],
            extremes_detected=[],
            cross_sectional_spread=1.2
        )

        recommendations = synthesizer.generate_recommendations(state)
        briefing = synthesizer.render_briefing(state, recommendations)

        # Check required sections exist
        assert "MORNING BRIEFING" in briefing
        assert "REGIME:" in briefing
        assert "FACTOR MOMENTUM" in briefing
        assert "SIGNAL ALIGNMENT" in briefing
        assert "Tech-Momentum" in briefing

    def test_render_briefing_formats_recommendations(self):
        """Recommendations should be formatted with conviction and reasons."""
        synthesizer = DecisionSynthesizer()

        state = SignalState(
            date=datetime(2026, 2, 4),
            regime=RegimeState("Low-Vol Bull", 0.85, 5, "stable"),
            factor_momentum=[
                FactorMomentum("F1", "Tech-Momentum", 0.025, "strong"),
            ],
            extremes_detected=[],
            cross_sectional_spread=1.5
        )

        recommendations = synthesizer.generate_recommendations(state)
        briefing = synthesizer.render_briefing(state, recommendations)

        # Should contain recommendation formatting
        assert "Conviction:" in briefing or "ACTION:" in briefing
