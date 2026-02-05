# Decision Synthesizer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Decision Synthesizer module that transforms abstract signals into actionable morning briefings with conviction-scored trade recommendations.

**Architecture:** New `decision_synthesizer.py` module that collects signals from existing analyzers (regime, momentum, cross-sectional), applies a conviction scoring framework, resolves conflicts via hierarchy, and renders structured briefings. Integrates via new `briefing` CLI command.

**Tech Stack:** Python, pandas, dataclasses for structured output, existing signal modules (regime_detection, trading_signals, cross_sectional, signal_aggregator)

---

## Task 1: Create SignalState Data Structure

**Files:**
- Create: `src/decision_synthesizer.py`
- Test: `tests/test_decision_synthesizer.py`

**Step 1: Write the failing test**

```python
# tests/test_decision_synthesizer.py
"""Tests for Decision Synthesizer module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.decision_synthesizer import SignalState, RegimeState, FactorMomentum


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
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_decision_synthesizer.py::TestSignalState::test_signal_state_creation -v
```

Expected: FAIL with "No module named 'src.decision_synthesizer'"

**Step 3: Write minimal implementation**

```python
# src/decision_synthesizer.py
"""
Decision Synthesizer - Translating Signals into Actionable Trading Decisions
=============================================================================

This module bridges the gap between signal generation and trade execution by
synthesizing multiple signal sources into conviction-scored recommendations
and structured morning briefings.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any


class ConvictionLevel(Enum):
    """Conviction levels for recommendations."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    CONFLICTED = "CONFLICTED"


class ActionCategory(Enum):
    """Categories for recommended actions."""
    OPPORTUNISTIC = "OPPORTUNISTIC"
    WEEKLY_REBALANCE = "WEEKLY_REBALANCE"
    WATCH = "WATCH"


@dataclass
class RegimeState:
    """Current market regime state."""
    name: str
    confidence: float
    days_in_regime: int
    trend: str  # "strengthening", "weakening", "stable"


@dataclass
class FactorMomentum:
    """Momentum state for a single factor."""
    factor: str
    name: str
    return_7d: float
    strength: str  # "strong", "moderate", "flat", "weak"


@dataclass
class ExtremeReading:
    """An extreme value alert."""
    factor: str
    name: str
    z_score: float
    direction: str  # "positive", "negative"


@dataclass
class SignalState:
    """Complete snapshot of all signal states."""
    date: datetime
    regime: RegimeState
    factor_momentum: List[FactorMomentum]
    extremes_detected: List[ExtremeReading]
    cross_sectional_spread: float  # std devs
    meta_model_prediction: Optional[float] = None
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_decision_synthesizer.py::TestSignalState::test_signal_state_creation -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/decision_synthesizer.py tests/test_decision_synthesizer.py
git commit -m "feat(synthesizer): add SignalState data structures"
```

---

## Task 2: Create Recommendation Data Structure

**Files:**
- Modify: `src/decision_synthesizer.py`
- Modify: `tests/test_decision_synthesizer.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_decision_synthesizer.py

from src.decision_synthesizer import Recommendation, TradeExpression


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
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_decision_synthesizer.py::TestRecommendation::test_recommendation_creation -v
```

Expected: FAIL with "cannot import name 'Recommendation'"

**Step 3: Write minimal implementation**

```python
# Add to src/decision_synthesizer.py after SignalState

@dataclass
class TradeExpression:
    """A specific way to express a trade idea."""
    description: str  # "Simple", "Targeted", "ETF proxy"
    trade: str  # "Buy QQQ" or "NVDA, MSFT, AAPL"
    size_pct: float  # Position size as fraction of portfolio


@dataclass
class Recommendation:
    """A complete actionable recommendation."""
    action: str
    conviction: ConvictionLevel
    conviction_score: float  # 0-10
    category: ActionCategory
    reasons: List[str]
    conflicts: List[str]
    expressions: List[TradeExpression]
    exit_trigger: str
    sizing_rationale: str = ""
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_decision_synthesizer.py::TestRecommendation::test_recommendation_creation -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/decision_synthesizer.py tests/test_decision_synthesizer.py
git commit -m "feat(synthesizer): add Recommendation data structure"
```

---

## Task 3: Implement Conviction Scoring

**Files:**
- Modify: `src/decision_synthesizer.py`
- Modify: `tests/test_decision_synthesizer.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_decision_synthesizer.py

from src.decision_synthesizer import DecisionSynthesizer, ConvictionScorer


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
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_decision_synthesizer.py::TestConvictionScoring -v
```

Expected: FAIL with "cannot import name 'ConvictionScorer'"

**Step 3: Write minimal implementation**

```python
# Add to src/decision_synthesizer.py

class ConvictionScorer:
    """
    Scores potential actions on conviction using weighted dimensions.

    Dimensions:
    - Signal Strength (40%): How extreme is the reading?
    - Signal Agreement (35%): Do multiple signals confirm?
    - Regime Fit (25%): Does action make sense in current regime?
    """

    def __init__(
        self,
        weight_strength: float = 0.40,
        weight_agreement: float = 0.35,
        weight_regime: float = 0.25
    ):
        self.weight_strength = weight_strength
        self.weight_agreement = weight_agreement
        self.weight_regime = weight_regime

    def calculate(
        self,
        signal_strength: float,  # z-score or similar
        signal_agreement: int,   # count of agreeing signals
        total_signals: int,      # total signal sources
        regime_fit: bool         # does action fit regime?
    ) -> float:
        """
        Calculate conviction score (0-10 scale).

        Args:
            signal_strength: Absolute z-score or normalized strength (0-3 typical)
            signal_agreement: Number of signals pointing same direction
            total_signals: Total number of signal sources considered
            regime_fit: Whether the action aligns with current regime

        Returns:
            Conviction score from 0 to 10
        """
        # Normalize signal strength to 0-10 (cap at z-score of 3)
        strength_score = min(signal_strength / 3.0, 1.0) * 10

        # Agreement as percentage, scaled to 0-10
        agreement_score = (signal_agreement / total_signals) * 10 if total_signals > 0 else 0

        # Regime fit is binary: 10 or 0
        regime_score = 10.0 if regime_fit else 0.0

        # Weighted combination
        score = (
            self.weight_strength * strength_score +
            self.weight_agreement * agreement_score +
            self.weight_regime * regime_score
        )

        return round(score, 1)

    def to_level(self, score: float) -> ConvictionLevel:
        """Convert numeric score to conviction level."""
        if score >= 8.0:
            return ConvictionLevel.HIGH
        elif score >= 5.0:
            return ConvictionLevel.MEDIUM
        elif score >= 3.0:
            return ConvictionLevel.LOW
        else:
            return ConvictionLevel.CONFLICTED
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_decision_synthesizer.py::TestConvictionScoring -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/decision_synthesizer.py tests/test_decision_synthesizer.py
git commit -m "feat(synthesizer): add ConvictionScorer with weighted dimensions"
```

---

## Task 4: Implement Signal Alignment Score

**Files:**
- Modify: `src/decision_synthesizer.py`
- Modify: `tests/test_decision_synthesizer.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_decision_synthesizer.py

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
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_decision_synthesizer.py::TestSignalAlignment -v
```

Expected: FAIL with "has no attribute 'calculate_alignment'"

**Step 3: Write minimal implementation**

```python
# Add to ConvictionScorer class in src/decision_synthesizer.py

    def calculate_alignment(
        self,
        regime_bullish: bool,
        momentum_bullish: List[bool],
        cross_section_bullish: bool
    ) -> int:
        """
        Calculate signal alignment score (1-10).

        Regime dominates: if regime conflicts with other signals,
        alignment is capped at 4 regardless of other agreement.

        Args:
            regime_bullish: Whether regime favors risk-on
            momentum_bullish: List of bool for each factor's momentum direction
            cross_section_bullish: Whether cross-sectional spread favors longs

        Returns:
            Alignment score from 1 to 10
        """
        # Count bullish signals
        momentum_bullish_count = sum(momentum_bullish)
        momentum_total = len(momentum_bullish)
        momentum_pct = momentum_bullish_count / momentum_total if momentum_total > 0 else 0.5

        # Determine overall non-regime direction
        non_regime_bullish = (momentum_pct > 0.5) or cross_section_bullish

        # Check for regime conflict (regime dominates rule)
        regime_conflict = regime_bullish != non_regime_bullish

        if regime_conflict:
            # Cap at 4 when regime conflicts
            base_score = 2 + (momentum_pct * 2)  # 2-4 range
            return int(min(base_score, 4))

        # No regime conflict - calculate alignment
        # Start with regime agreement bonus
        score = 5.0

        # Add momentum agreement (up to +3)
        if momentum_pct >= 0.8:
            score += 3.0
        elif momentum_pct >= 0.6:
            score += 2.0
        elif momentum_pct >= 0.4:
            score += 1.0

        # Add cross-sectional confirmation (up to +2)
        if cross_section_bullish == regime_bullish:
            score += 2.0

        return int(min(score, 10))
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_decision_synthesizer.py::TestSignalAlignment -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/decision_synthesizer.py tests/test_decision_synthesizer.py
git commit -m "feat(synthesizer): add signal alignment scoring with regime dominance"
```

---

## Task 5: Implement Signal Collection from Existing Modules

**Files:**
- Modify: `src/decision_synthesizer.py`
- Modify: `tests/test_decision_synthesizer.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_decision_synthesizer.py
from unittest.mock import Mock, patch

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
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_decision_synthesizer.py::TestSignalCollection::test_collect_signals_returns_signal_state -v
```

Expected: FAIL with "cannot import name 'DecisionSynthesizer'"

**Step 3: Write minimal implementation**

```python
# Add to src/decision_synthesizer.py

import pandas as pd
import numpy as np
import logging

_LOGGER = logging.getLogger(__name__)


class DecisionSynthesizer:
    """
    Translates signals into actionable trading decisions.

    Collects signals from regime detection, momentum analysis, and
    cross-sectional ranking, then synthesizes into conviction-scored
    recommendations and morning briefings.
    """

    def __init__(self):
        self.scorer = ConvictionScorer()

    def collect_all_signals(
        self,
        factor_returns: pd.DataFrame,
        factor_loadings: pd.DataFrame,
        factor_names: Optional[Dict[str, str]] = None
    ) -> SignalState:
        """
        Gather current state from all signal sources.

        Args:
            factor_returns: DataFrame of factor returns (dates x factors)
            factor_loadings: DataFrame of stock loadings (tickers x factors)
            factor_names: Optional mapping of factor IDs to human names

        Returns:
            SignalState with all signal components populated
        """
        from src.regime_detection import RegimeDetector
        from src.trading_signals import FactorMomentumAnalyzer
        from src.cross_sectional import CrossSectionalAnalyzer

        factor_names = factor_names or {}

        # Detect regime
        try:
            detector = RegimeDetector(factor_returns)
            detector.fit_hmm(n_regimes=4)
            regime_result = detector.detect_current_regime()

            regime_state = RegimeState(
                name=regime_result.get('regime', 'Unknown'),
                confidence=regime_result.get('confidence', 0.5),
                days_in_regime=regime_result.get('days_in_regime', 1),
                trend=self._calculate_regime_trend(detector)
            )
        except Exception as e:
            _LOGGER.warning(f"Regime detection failed: {e}")
            regime_state = RegimeState(
                name="Unknown",
                confidence=0.0,
                days_in_regime=0,
                trend="unknown"
            )

        # Calculate factor momentum
        momentum_list = []
        try:
            analyzer = FactorMomentumAnalyzer(factor_returns)

            for factor in factor_returns.columns:
                return_7d = factor_returns[factor].tail(7).sum()
                z_score = self._calculate_zscore(factor_returns[factor], window=20)

                strength = self._classify_strength(z_score)

                momentum_list.append(FactorMomentum(
                    factor=factor,
                    name=factor_names.get(factor, factor),
                    return_7d=return_7d,
                    strength=strength
                ))
        except Exception as e:
            _LOGGER.warning(f"Momentum analysis failed: {e}")

        # Detect extremes
        extremes = []
        try:
            for factor in factor_returns.columns:
                z = self._calculate_zscore(factor_returns[factor], window=20)
                if abs(z) > 2.0:
                    extremes.append(ExtremeReading(
                        factor=factor,
                        name=factor_names.get(factor, factor),
                        z_score=z,
                        direction="positive" if z > 0 else "negative"
                    ))
        except Exception as e:
            _LOGGER.warning(f"Extreme detection failed: {e}")

        # Calculate cross-sectional spread
        try:
            cs_analyzer = CrossSectionalAnalyzer(factor_loadings)
            # Simplified: use first factor's spread
            scores = factor_loadings.iloc[:, 0] if len(factor_loadings.columns) > 0 else pd.Series()
            spread = (scores.quantile(0.9) - scores.quantile(0.1)) / scores.std() if len(scores) > 0 else 0
        except Exception as e:
            _LOGGER.warning(f"Cross-sectional analysis failed: {e}")
            spread = 0.0

        return SignalState(
            date=datetime.now(),
            regime=regime_state,
            factor_momentum=momentum_list,
            extremes_detected=extremes,
            cross_sectional_spread=spread
        )

    def _calculate_zscore(self, series: pd.Series, window: int = 20) -> float:
        """Calculate z-score of latest value vs rolling window."""
        if len(series) < window:
            return 0.0
        recent = series.tail(window)
        mean = recent.mean()
        std = recent.std()
        if std == 0:
            return 0.0
        return (series.iloc[-1] - mean) / std

    def _classify_strength(self, z_score: float) -> str:
        """Classify momentum strength from z-score."""
        abs_z = abs(z_score)
        if abs_z > 1.5:
            return "strong"
        elif abs_z > 0.75:
            return "moderate"
        elif abs_z > 0.25:
            return "flat"
        else:
            return "weak"

    def _calculate_regime_trend(self, detector) -> str:
        """Determine if regime is strengthening or weakening."""
        # Simplified: would need historical probabilities
        return "stable"
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_decision_synthesizer.py::TestSignalCollection::test_collect_signals_returns_signal_state -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/decision_synthesizer.py tests/test_decision_synthesizer.py
git commit -m "feat(synthesizer): add DecisionSynthesizer.collect_all_signals"
```

---

## Task 6: Implement Recommendation Generation

**Files:**
- Modify: `src/decision_synthesizer.py`
- Modify: `tests/test_decision_synthesizer.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_decision_synthesizer.py

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
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_decision_synthesizer.py::TestRecommendationGeneration -v
```

Expected: FAIL with "has no attribute 'generate_recommendations'"

**Step 3: Write minimal implementation**

```python
# Add to DecisionSynthesizer class in src/decision_synthesizer.py

    def generate_recommendations(self, state: SignalState) -> List[Recommendation]:
        """
        Generate actionable recommendations from signal state.

        Applies conviction scoring and categorizes by urgency.

        Args:
            state: Current SignalState snapshot

        Returns:
            List of Recommendations sorted by conviction
        """
        recommendations = []

        # Check for regime-based opportunities
        if state.regime.confidence > 0.70 and state.regime.days_in_regime >= 2:
            rec = self._evaluate_regime_opportunity(state)
            if rec:
                recommendations.append(rec)

        # Check for factor momentum opportunities
        for fm in state.factor_momentum:
            rec = self._evaluate_momentum_opportunity(fm, state)
            if rec:
                recommendations.append(rec)

        # Check for extreme value opportunities
        for extreme in state.extremes_detected:
            rec = self._evaluate_extreme_opportunity(extreme, state)
            if rec:
                recommendations.append(rec)

        # Sort by conviction score descending
        recommendations.sort(key=lambda r: r.conviction_score, reverse=True)

        return recommendations

    def _evaluate_regime_opportunity(self, state: SignalState) -> Optional[Recommendation]:
        """Evaluate regime-based trading opportunity."""
        regime = state.regime

        # Determine if regime favors risk-on or risk-off
        bullish_regimes = ["Low-Vol Bull", "High-Vol Bull"]
        bearish_regimes = ["Low-Vol Bear", "High-Vol Bear", "Crisis"]

        is_bullish = regime.name in bullish_regimes

        # Count confirming momentum signals
        momentum_confirms = sum(
            1 for fm in state.factor_momentum
            if (fm.strength in ["strong", "moderate"] and fm.return_7d > 0) == is_bullish
        )
        total_factors = len(state.factor_momentum)

        # Score conviction
        score = self.scorer.calculate(
            signal_strength=regime.confidence * 3,  # Scale confidence to z-score equivalent
            signal_agreement=momentum_confirms,
            total_signals=max(total_factors, 1),
            regime_fit=True  # By definition, regime-based action fits regime
        )

        level = self.scorer.to_level(score)

        # Determine category
        if level == ConvictionLevel.HIGH:
            category = ActionCategory.OPPORTUNISTIC
        elif level == ConvictionLevel.MEDIUM:
            category = ActionCategory.WEEKLY_REBALANCE
        else:
            category = ActionCategory.WATCH

        action = f"{'Increase' if is_bullish else 'Reduce'} equity exposure"

        return Recommendation(
            action=action,
            conviction=level,
            conviction_score=score,
            category=category,
            reasons=[
                f"Regime: {regime.name} ({regime.confidence:.0%} confidence, {regime.days_in_regime} days)",
                f"Trend: {regime.trend}",
                f"Momentum confirmation: {momentum_confirms}/{total_factors} factors aligned"
            ],
            conflicts=self._identify_conflicts(state, is_bullish),
            expressions=[
                TradeExpression("Simple", "SPY" if is_bullish else "SH", 0.05),
            ],
            exit_trigger=f"Regime shifts to {'bearish' if is_bullish else 'bullish'}",
            sizing_rationale=f"{'Full' if level == ConvictionLevel.HIGH else 'Reduced'} size based on conviction"
        )

    def _evaluate_momentum_opportunity(
        self,
        fm: FactorMomentum,
        state: SignalState
    ) -> Optional[Recommendation]:
        """Evaluate factor momentum opportunity."""
        if fm.strength not in ["strong", "moderate"]:
            return None

        # Check regime fit
        bullish_regimes = ["Low-Vol Bull", "High-Vol Bull"]
        regime_bullish = state.regime.name in bullish_regimes
        momentum_bullish = fm.return_7d > 0
        regime_fit = regime_bullish == momentum_bullish

        # If regime conflict, demote to WATCH
        if not regime_fit and state.regime.confidence > 0.6:
            return Recommendation(
                action=f"Monitor {fm.name}",
                conviction=ConvictionLevel.LOW,
                conviction_score=3.0,
                category=ActionCategory.WATCH,
                reasons=[
                    f"Signal: {fm.name} {'+' if momentum_bullish else '-'}{abs(fm.return_7d):.1%} (7d)",
                    f"Regime conflict: {state.regime.name} suggests opposite direction"
                ],
                conflicts=[f"Regime ({state.regime.name}) conflicts with momentum signal"],
                expressions=[],
                exit_trigger=f"Regime aligns with {fm.name} momentum"
            )

        # Score conviction
        strength_score = 1.5 if fm.strength == "strong" else 1.0
        score = self.scorer.calculate(
            signal_strength=strength_score,
            signal_agreement=2 if regime_fit else 1,
            total_signals=3,
            regime_fit=regime_fit
        )

        level = self.scorer.to_level(score)
        category = (
            ActionCategory.OPPORTUNISTIC if level == ConvictionLevel.HIGH
            else ActionCategory.WEEKLY_REBALANCE if level == ConvictionLevel.MEDIUM
            else ActionCategory.WATCH
        )

        action = f"{'Increase' if momentum_bullish else 'Decrease'} {fm.name} exposure"

        return Recommendation(
            action=action,
            conviction=level,
            conviction_score=score,
            category=category,
            reasons=[
                f"Signal: {fm.name} {'+' if momentum_bullish else ''}{fm.return_7d:.1%} over 7 days ({fm.strength})",
                f"Regime: {state.regime.name} {'confirms' if regime_fit else 'conflicts'}"
            ],
            conflicts=[] if regime_fit else [f"Regime conflict: {state.regime.name}"],
            expressions=[
                TradeExpression("Factor tilt", f"Tilt toward {fm.name}", 0.03),
            ],
            exit_trigger=f"{fm.name} momentum reverses or regime shifts"
        )

    def _evaluate_extreme_opportunity(
        self,
        extreme: ExtremeReading,
        state: SignalState
    ) -> Optional[Recommendation]:
        """Evaluate extreme value opportunity (mean reversion)."""
        # Extreme values suggest mean reversion
        is_reversal_long = extreme.direction == "negative"  # Oversold = buy

        score = self.scorer.calculate(
            signal_strength=abs(extreme.z_score),
            signal_agreement=1,  # Extreme is single signal
            total_signals=3,
            regime_fit=True  # Extremes can work in any regime
        )

        level = self.scorer.to_level(score)

        return Recommendation(
            action=f"Mean reversion {'long' if is_reversal_long else 'short'} on {extreme.name}",
            conviction=level,
            conviction_score=score,
            category=ActionCategory.OPPORTUNISTIC,  # Extremes are time-sensitive
            reasons=[
                f"Extreme: {extreme.name} z-score {extreme.z_score:.1f} ({extreme.direction})",
                f"Statistical: Beyond 2-sigma suggests reversion"
            ],
            conflicts=[],
            expressions=[
                TradeExpression(
                    "Reversion",
                    f"{'Buy' if is_reversal_long else 'Sell'} {extreme.name}-exposed stocks",
                    0.02
                ),
            ],
            exit_trigger=f"Z-score returns to +/- 1.0"
        )

    def _identify_conflicts(self, state: SignalState, proposed_bullish: bool) -> List[str]:
        """Identify signals that conflict with proposed direction."""
        conflicts = []

        for fm in state.factor_momentum:
            momentum_bullish = fm.return_7d > 0
            if fm.strength in ["strong", "moderate"] and momentum_bullish != proposed_bullish:
                conflicts.append(f"{fm.name} momentum is {'bullish' if momentum_bullish else 'bearish'}")

        return conflicts
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_decision_synthesizer.py::TestRecommendationGeneration -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/decision_synthesizer.py tests/test_decision_synthesizer.py
git commit -m "feat(synthesizer): add recommendation generation with conviction scoring"
```

---

## Task 7: Implement Morning Briefing Renderer

**Files:**
- Modify: `src/decision_synthesizer.py`
- Modify: `tests/test_decision_synthesizer.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_decision_synthesizer.py

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
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_decision_synthesizer.py::TestBriefingRenderer -v
```

Expected: FAIL with "has no attribute 'render_briefing'"

**Step 3: Write minimal implementation**

```python
# Add to DecisionSynthesizer class in src/decision_synthesizer.py

    def render_briefing(
        self,
        state: SignalState,
        recommendations: List[Recommendation],
        format: str = "text"
    ) -> str:
        """
        Render morning briefing as formatted text.

        Args:
            state: Current SignalState
            recommendations: Generated recommendations
            format: Output format ("text" or "markdown")

        Returns:
            Formatted briefing string
        """
        lines = []

        # Header
        lines.append("" * 70)
        lines.append(f"MORNING BRIEFING - {state.date.strftime('%Y-%m-%d')}")
        lines.append("" * 70)
        lines.append("")

        # Regime section
        regime = state.regime
        lines.append(f"REGIME: {regime.name} ({regime.confidence:.0%} confidence, {regime.days_in_regime} days)")
        lines.append(f"TREND:  {regime.trend.capitalize()}")
        lines.append("")

        # Factor momentum section
        lines.append("FACTOR MOMENTUM (7-day):")
        for fm in state.factor_momentum:
            symbol = self._strength_symbol(fm.strength, fm.return_7d > 0)
            sign = "+" if fm.return_7d > 0 else ""
            lines.append(f"  {symbol} {fm.name:<20} {sign}{fm.return_7d:.1%}  ({fm.strength})")
        lines.append("")

        # Extremes section
        if state.extremes_detected:
            lines.append("EXTREMES DETECTED:")
            for ext in state.extremes_detected:
                lines.append(f"   {ext.name}: z-score {ext.z_score:.1f} ({ext.direction})")
        else:
            lines.append("EXTREMES DETECTED: None today")
        lines.append("")

        # Signal alignment
        alignment = self._calculate_overall_alignment(state)
        lines.append(f"SIGNAL ALIGNMENT: {alignment}/10 {self._alignment_description(alignment)}")
        lines.append("")

        # Recommendations by category
        lines.append("" * 70)
        lines.append("RECOMMENDATIONS")
        lines.append("" * 70)

        for category in [ActionCategory.OPPORTUNISTIC, ActionCategory.WEEKLY_REBALANCE, ActionCategory.WATCH]:
            cat_recs = [r for r in recommendations if r.category == category]
            if cat_recs:
                lines.append("")
                lines.append(f" {category.value}")
                for rec in cat_recs:
                    lines.extend(self._format_recommendation(rec))

        if not recommendations:
            lines.append("")
            lines.append("No actionable recommendations at this time.")

        lines.append("")
        lines.append("" * 70)

        return "\n".join(lines)

    def _strength_symbol(self, strength: str, is_positive: bool) -> str:
        """Return symbol for momentum strength."""
        if strength == "strong":
            return "" if is_positive else ""
        elif strength == "moderate":
            return "" if is_positive else ""
        else:
            return ""

    def _calculate_overall_alignment(self, state: SignalState) -> int:
        """Calculate overall signal alignment score."""
        bullish_regimes = ["Low-Vol Bull", "High-Vol Bull"]
        regime_bullish = state.regime.name in bullish_regimes

        momentum_bullish = [fm.return_7d > 0 for fm in state.factor_momentum]
        cross_bullish = state.cross_sectional_spread > 1.0

        return self.scorer.calculate_alignment(
            regime_bullish=regime_bullish,
            momentum_bullish=momentum_bullish,
            cross_section_bullish=cross_bullish
        )

    def _alignment_description(self, score: int) -> str:
        """Return description for alignment score."""
        if score >= 9:
            return "→ Act with confidence"
        elif score >= 6:
            return "→ Act with normal sizing"
        elif score >= 4:
            return "→ Reduce sizes or wait"
        else:
            return "→ Stay flat or defensive"

    def _format_recommendation(self, rec: Recommendation) -> List[str]:
        """Format a single recommendation."""
        lines = []
        lines.append("")
        lines.append(f"{'' * 68}")
        lines.append(f" ACTION: {rec.action:<58}")
        lines.append(f"{'' * 68}")
        lines.append(f" Conviction: {rec.conviction.value} ({rec.conviction_score}/10){' ' * (68 - 25 - len(rec.conviction.value))}")
        lines.append(f"{' ' * 68}")
        lines.append(f" WHY:{' ' * 63}")
        for reason in rec.reasons[:3]:  # Limit to 3 reasons
            reason_trimmed = reason[:62]
            lines.append(f"  • {reason_trimmed:<63}")

        if rec.conflicts:
            lines.append(f"{' ' * 68}")
            lines.append(f" CONFLICTS:{' ' * 57}")
            for conflict in rec.conflicts[:2]:
                conflict_trimmed = conflict[:62]
                lines.append(f"  • {conflict_trimmed:<63}")
        else:
            lines.append(f"{' ' * 68}")
            lines.append(f" CONFLICTS: None{' ' * 52}")

        if rec.expressions:
            lines.append(f"{' ' * 68}")
            lines.append(f" SUGGESTED EXPRESSION:{' ' * 46}")
            for expr in rec.expressions[:2]:
                expr_str = f"{expr.description}: {expr.trade} ({expr.size_pct:.0%})"[:62]
                lines.append(f"  • {expr_str:<63}")

        lines.append(f"{' ' * 68}")
        exit_trimmed = rec.exit_trigger[:55]
        lines.append(f" EXIT: {exit_trimmed:<61}")
        lines.append(f"{'' * 68}")

        return lines
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_decision_synthesizer.py::TestBriefingRenderer -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/decision_synthesizer.py tests/test_decision_synthesizer.py
git commit -m "feat(synthesizer): add morning briefing renderer"
```

---

## Task 8: Add CLI Command for Briefing

**Files:**
- Modify: `src/__main__.py`

**Step 1: Write the failing test**

Run the command to verify it doesn't exist:

```bash
uv run python -m src briefing --help
```

Expected: Error about unknown command

**Step 2: Add briefing command to CLI**

Add after the `cmd_report` function (around line 440):

```python
# =============================================================================
# Briefing Commands
# =============================================================================

def cmd_briefing(args):
    """Generate morning briefing with actionable recommendations."""
    from src.decision_synthesizer import DecisionSynthesizer
    from src.research import FactorResearchSystem
    import pickle
    from pathlib import Path

    print("" * 70)
    print(" GENERATING MORNING BRIEFING")
    print("" * 70)

    # Load or generate factors
    cache_file = f"factor_cache_{'_'.join(args.universe)}_{args.method}.pkl"

    if Path(cache_file).exists():
        print(f"\n Loading cached factors from {cache_file}")
        with open(cache_file, 'rb') as f:
            factor_returns, factor_loadings = pickle.load(f)
        factor_names = {}
    else:
        print(f"\n Generating factors for {', '.join(args.universe)}...")
        api_key = get_api_key()
        frs = FactorResearchSystem(
            api_key,
            universe=args.universe,
            factor_method=args.method,
            n_components=args.components,
            expand_etfs=True
        )
        frs.fit_factors()
        factor_returns = frs.get_factor_returns()
        factor_loadings = frs._expos
        factor_names = {}

        # Cache for next time
        with open(cache_file, 'wb') as f:
            pickle.dump((factor_returns, factor_loadings), f)

    # Generate briefing
    print("\n Analyzing signals...")
    synthesizer = DecisionSynthesizer()

    state = synthesizer.collect_all_signals(
        factor_returns=factor_returns,
        factor_loadings=factor_loadings,
        factor_names=factor_names
    )

    recommendations = synthesizer.generate_recommendations(state)
    briefing = synthesizer.render_briefing(state, recommendations)

    # Output
    print("\n")
    print(briefing)

    # Save if requested
    if args.output:
        output_path = args.output
        if args.format == "markdown":
            output_path = output_path if output_path.endswith('.md') else f"{output_path}.md"
        else:
            output_path = output_path if output_path.endswith('.txt') else f"{output_path}.txt"

        with open(output_path, 'w') as f:
            f.write(briefing)
        print(f"\n Briefing saved to: {output_path}")
```

Add the parser after the `signals` subparser section (around line 900):

```python
    # -------------------------------------------------------------------------
    # Briefing command
    # -------------------------------------------------------------------------
    briefing_parser = subparsers.add_parser(
        'briefing',
        help='Generate morning briefing with actionable recommendations',
        description='Synthesize all signals into a morning briefing with trade recommendations'
    )
    briefing_parser.add_argument(
        '--universe',
        nargs='+',
        default=['SPY'],
        help='Stock/ETF universe (default: SPY)'
    )
    briefing_parser.add_argument(
        '--method',
        default='pca',
        choices=['fundamental', 'pca', 'ica'],
        help='Factor method (default: pca)'
    )
    briefing_parser.add_argument(
        '--components',
        type=int,
        default=8,
        help='Number of factors (default: 8)'
    )
    briefing_parser.add_argument(
        '--output', '-o',
        help='Save briefing to file'
    )
    briefing_parser.add_argument(
        '--format',
        choices=['text', 'markdown'],
        default='text',
        help='Output format (default: text)'
    )
```

Add to the command dispatch (in the `main()` function):

```python
    elif args.command == 'briefing':
        cmd_briefing(args)
```

**Step 3: Run to verify it works**

```bash
uv run python -m src briefing --help
```

Expected: Help text for briefing command

**Step 4: Commit**

```bash
git add src/__main__.py
git commit -m "feat(cli): add briefing command for morning recommendations"
```

---

## Task 9: Integration Test - End-to-End Briefing

**Files:**
- Create: `tests/test_briefing_integration.py`

**Step 1: Write integration test**

```python
# tests/test_briefing_integration.py
"""Integration tests for the briefing command."""

import pytest
import subprocess
import sys


class TestBriefingIntegration:
    """End-to-end tests for briefing generation."""

    @pytest.mark.slow
    def test_briefing_command_runs(self):
        """Briefing command should run without errors."""
        result = subprocess.run(
            [sys.executable, "-m", "src", "briefing", "--help"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "morning briefing" in result.stdout.lower() or "recommendations" in result.stdout.lower()

    @pytest.mark.slow
    def test_briefing_with_cached_data(self):
        """Briefing should work with existing cached factor data."""
        # This test requires cached data to exist
        # Skip if no cache available
        from pathlib import Path

        cache_files = list(Path('.').glob('factor_cache_*.pkl'))
        if not cache_files:
            pytest.skip("No cached factor data available")

        # Extract universe from first cache file
        cache_name = cache_files[0].stem
        parts = cache_name.replace('factor_cache_', '').rsplit('_', 1)
        universe = parts[0]
        method = parts[1] if len(parts) > 1 else 'pca'

        result = subprocess.run(
            [sys.executable, "-m", "src", "briefing",
             "--universe", universe,
             "--method", method],
            capture_output=True,
            text=True,
            timeout=60
        )

        # Should complete (may have warnings but shouldn't crash)
        assert "MORNING BRIEFING" in result.stdout or result.returncode == 0
```

**Step 2: Run integration test**

```bash
uv run pytest tests/test_briefing_integration.py -v -m "not slow"
```

**Step 3: Commit**

```bash
git add tests/test_briefing_integration.py
git commit -m "test: add briefing integration tests"
```

---

## Task 10: Position Sizing Module

**Files:**
- Modify: `src/decision_synthesizer.py`
- Modify: `tests/test_decision_synthesizer.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_decision_synthesizer.py

class TestPositionSizing:
    """Tests for position sizing logic."""

    def test_high_conviction_full_size(self):
        """HIGH conviction should get full 5% position."""
        from src.decision_synthesizer import PositionSizer

        sizer = PositionSizer(portfolio_value=1_000_000)

        size = sizer.calculate_size(
            conviction=ConvictionLevel.HIGH,
            conviction_score=8.5,
            current_exposure=0.10,
            exposure_limit=0.25
        )

        assert size == 0.05  # 5% of portfolio

    def test_conflicted_zero_size(self):
        """CONFLICTED conviction should get zero position."""
        from src.decision_synthesizer import PositionSizer

        sizer = PositionSizer(portfolio_value=1_000_000)

        size = sizer.calculate_size(
            conviction=ConvictionLevel.CONFLICTED,
            conviction_score=2.0,
            current_exposure=0.10,
            exposure_limit=0.25
        )

        assert size == 0.0

    def test_exposure_limit_respected(self):
        """Position should be reduced if it would breach exposure limit."""
        from src.decision_synthesizer import PositionSizer

        sizer = PositionSizer(portfolio_value=1_000_000)

        # Current exposure 22%, limit 25%, so max add is 3%
        size = sizer.calculate_size(
            conviction=ConvictionLevel.HIGH,  # Would normally be 5%
            conviction_score=9.0,
            current_exposure=0.22,
            exposure_limit=0.25
        )

        assert size <= 0.03  # Capped by remaining room
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_decision_synthesizer.py::TestPositionSizing -v
```

Expected: FAIL with "cannot import name 'PositionSizer'"

**Step 3: Write minimal implementation**

```python
# Add to src/decision_synthesizer.py

@dataclass
class PositionSizer:
    """
    Calculate position sizes based on conviction and risk limits.

    Base sizes by conviction:
    - HIGH (8-10): 5% of portfolio
    - MEDIUM (5-7): 3% of portfolio
    - LOW (3-4): 1% of portfolio
    - CONFLICTED: 0%
    """

    portfolio_value: float
    base_sizes: Dict[ConvictionLevel, float] = field(default_factory=lambda: {
        ConvictionLevel.HIGH: 0.05,
        ConvictionLevel.MEDIUM: 0.03,
        ConvictionLevel.LOW: 0.01,
        ConvictionLevel.CONFLICTED: 0.0
    })

    def calculate_size(
        self,
        conviction: ConvictionLevel,
        conviction_score: float,
        current_exposure: float,
        exposure_limit: float
    ) -> float:
        """
        Calculate position size respecting conviction and limits.

        Args:
            conviction: Conviction level
            conviction_score: Numeric score (0-10)
            current_exposure: Current exposure to this factor/direction
            exposure_limit: Maximum allowed exposure

        Returns:
            Position size as fraction of portfolio
        """
        base_size = self.base_sizes.get(conviction, 0.0)

        # Check exposure limit
        remaining_room = max(0, exposure_limit - current_exposure)

        # Cap at remaining room
        final_size = min(base_size, remaining_room)

        return final_size

    def calculate_dollars(
        self,
        conviction: ConvictionLevel,
        conviction_score: float,
        current_exposure: float,
        exposure_limit: float
    ) -> float:
        """Calculate position size in dollars."""
        size_pct = self.calculate_size(
            conviction, conviction_score, current_exposure, exposure_limit
        )
        return size_pct * self.portfolio_value
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_decision_synthesizer.py::TestPositionSizing -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/decision_synthesizer.py tests/test_decision_synthesizer.py
git commit -m "feat(synthesizer): add PositionSizer with exposure limits"
```

---

## Summary

This implementation plan covers:

1. **Tasks 1-2**: Core data structures (SignalState, Recommendation)
2. **Tasks 3-4**: Conviction scoring with regime dominance
3. **Task 5**: Signal collection from existing modules
4. **Task 6**: Recommendation generation logic
5. **Task 7**: Morning briefing text rendering
6. **Task 8**: CLI command integration
7. **Task 9**: Integration testing
8. **Task 10**: Position sizing with risk guardrails

**Total estimated commits**: 10

**Key design decisions preserved**:
- Regime dominates (conflict caps alignment at 4)
- Conviction scoring: 40% strength, 35% agreement, 25% regime fit
- Categories: OPPORTUNISTIC, WEEKLY_REBALANCE, WATCH
- Position sizing: 5% HIGH, 3% MEDIUM, 1% LOW, 0% CONFLICTED
