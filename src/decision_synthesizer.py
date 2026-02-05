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
