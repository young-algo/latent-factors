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
import logging

import pandas as pd
import numpy as np

_LOGGER = logging.getLogger(__name__)


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

        # Determine overall non-regime direction (majority vote)
        non_regime_bullish = (momentum_pct > 0.5) or cross_section_bullish

        # Check for regime conflict (regime dominates rule)
        regime_conflict = regime_bullish != non_regime_bullish

        if regime_conflict:
            # Cap at 4 when regime conflicts
            base_score = 2 + (momentum_pct * 2)  # 2-4 range
            return int(min(base_score, 4))

        # No regime conflict - calculate alignment
        # Start with base score
        score = 4.0

        # Add momentum agreement (up to +3)
        if momentum_pct >= 0.8:
            score += 3.0
        elif momentum_pct >= 0.6:
            score += 1.0  # Partial agreement gets less bonus

        # Add cross-sectional confirmation (up to +3)
        if cross_section_bullish == regime_bullish:
            score += 3.0

        return int(min(score, 10))


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
