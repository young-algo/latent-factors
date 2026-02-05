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
        lines.append("═" * 70)
        lines.append(f"MORNING BRIEFING - {state.date.strftime('%Y-%m-%d')}")
        lines.append("═" * 70)
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
                lines.append(f"  ⚠ {ext.name}: z-score {ext.z_score:.1f} ({ext.direction})")
        else:
            lines.append("EXTREMES DETECTED: None today")
        lines.append("")

        # Signal alignment
        alignment = self._calculate_overall_alignment(state)
        lines.append(f"SIGNAL ALIGNMENT: {alignment}/10 {self._alignment_description(alignment)}")
        lines.append("")

        # Recommendations by category
        lines.append("─" * 70)
        lines.append("RECOMMENDATIONS")
        lines.append("─" * 70)

        for category in [ActionCategory.OPPORTUNISTIC, ActionCategory.WEEKLY_REBALANCE, ActionCategory.WATCH]:
            cat_recs = [r for r in recommendations if r.category == category]
            if cat_recs:
                lines.append("")
                lines.append(f"▸ {category.value}")
                for rec in cat_recs:
                    lines.extend(self._format_recommendation(rec))

        if not recommendations:
            lines.append("")
            lines.append("No actionable recommendations at this time.")

        lines.append("")
        lines.append("═" * 70)

        return "\n".join(lines)

    def _strength_symbol(self, strength: str, is_positive: bool) -> str:
        """Return symbol for momentum strength."""
        if strength == "strong":
            return "✓" if is_positive else "✗"
        elif strength == "moderate":
            return "✓" if is_positive else "✗"
        else:
            return "○"

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
        lines.append(f"┌{'─' * 68}┐")
        lines.append(f"│ ACTION: {rec.action:<58}│")
        lines.append(f"├{'─' * 68}┤")
        lines.append(f"│ Conviction: {rec.conviction.value} ({rec.conviction_score}/10){' ' * (68 - 25 - len(rec.conviction.value))}│")
        lines.append(f"│{' ' * 68}│")
        lines.append(f"│ WHY:{' ' * 63}│")
        for reason in rec.reasons[:3]:  # Limit to 3 reasons
            reason_trimmed = reason[:62]
            lines.append(f"│  • {reason_trimmed:<63}│")

        if rec.conflicts:
            lines.append(f"│{' ' * 68}│")
            lines.append(f"│ CONFLICTS:{' ' * 57}│")
            for conflict in rec.conflicts[:2]:
                conflict_trimmed = conflict[:62]
                lines.append(f"│  • {conflict_trimmed:<63}│")
        else:
            lines.append(f"│{' ' * 68}│")
            lines.append(f"│ CONFLICTS: None{' ' * 52}│")

        if rec.expressions:
            lines.append(f"│{' ' * 68}│")
            lines.append(f"│ SUGGESTED EXPRESSION:{' ' * 46}│")
            for expr in rec.expressions[:2]:
                expr_str = f"{expr.description}: {expr.trade} ({expr.size_pct:.0%})"[:62]
                lines.append(f"│  • {expr_str:<63}│")

        lines.append(f"│{' ' * 68}│")
        exit_trimmed = rec.exit_trigger[:55]
        lines.append(f"│ EXIT: {exit_trimmed:<61}│")
        lines.append(f"└{'─' * 68}┘")

        return lines
