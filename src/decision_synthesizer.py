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
