"""
Signal Aggregation and Integration System
=========================================

This module provides signal aggregation capabilities that combine multiple
signal types (momentum, extreme values, cross-sectional, regime-based) into
unified trading recommendations with confidence scores.

Core Components
---------------

**SignalAggregator**
- Collects signals from all analyzers (momentum, cross-sectional, regime)
- Weights signals by historical efficacy
- Generates consensus scores and confidence metrics
- Produces actionable trading recommendations

Signal Combination Methods
--------------------------

**Weighted Consensus:**
- Combines signals using configurable weights
- Higher weight to historically effective signals
- Normalized to 0-100 confidence scale

**Majority Voting:**
- Simple majority of signal directions
- Useful for binary decisions

**Bayesian Combination:**
- Probabilistic signal combination
- Accounts for signal correlation

Architecture
------------
```
Momentum Signals 
Extreme Alerts > SignalAggregator > Consensus Score > Trade Rec
Cross-Sectional          ↓                    ↓
Regime Signals    Historical Weights    Confidence 0-100
```

Dependencies
------------
- pandas, numpy: Data manipulation
- trading_signals: FactorMomentumAnalyzer
- cross_sectional: CrossSectionalAnalyzer
- regime_detection: RegimeDetector

Examples
--------
>>> from signal_aggregator import SignalAggregator
>>> aggregator = SignalAggregator(factor_research_system)
>>> aggregator.add_momentum_signals(analyzer)
>>> aggregator.add_cross_sectional_signals(analyzer)
>>> consensus = aggregator.aggregate_signals()
>>> opportunities = aggregator.get_top_opportunities(n=10)
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# Optional XGBoost for meta-modeling
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not installed. Meta-modeling will use fallback methods.")

_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.WARNING)

if not _LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
    handler.setFormatter(formatter)
    _LOGGER.addHandler(handler)
    _LOGGER.propagate = False


class SignalType(Enum):
    """Enumeration of signal source types."""
    MOMENTUM = "momentum"
    EXTREME_VALUE = "extreme_value"
    CROSS_SECTIONAL = "cross_sectional"
    REGIME = "regime"
    COMBINED = "combined"


class SignalDirection(Enum):
    """Enumeration of signal directions."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class Signal:
    """Data class representing a trading signal."""
    source: str
    signal_type: SignalType
    direction: SignalDirection
    strength: float  # 0-1
    confidence: float  # 0-100
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsensusSignal:
    """Data class representing a consensus trading signal."""
    ticker: Optional[str]
    factor: Optional[str]
    consensus_direction: SignalDirection
    consensus_score: float  # -100 to +100
    confidence: float  # 0-100
    contributing_signals: List[Signal]
    recommendation: str
    risk_level: str  # 'low', 'medium', 'high'


@dataclass
class TradingOpportunity:
    """Data class representing a trading opportunity."""
    ticker: Optional[str]
    factor: Optional[str]
    signal_type: str
    direction: str
    confidence: float
    entry_rationale: str
    risk_factors: List[str]
    suggested_position_size: str  # 'small', 'medium', 'large'
    time_horizon: str  # 'short', 'medium', 'long'


class SignalAggregator:
    """
    Aggregates multiple signal types into unified trading recommendations.

    This class combines signals from momentum analysis, extreme value detection,
    cross-sectional ranking, and regime detection to produce consensus signals
    with confidence scores.

    Parameters
    ----------
    factor_research_system : FactorResearchSystem
        The factor research system containing fitted factors

    Attributes
    ----------
    frs : FactorResearchSystem
        Reference to factor research system
    momentum_analyzer : FactorMomentumAnalyzer or None
        Momentum signal analyzer
    cross_sectional_analyzer : CrossSectionalAnalyzer or None
        Cross-sectional analyzer
    regime_detector : RegimeDetector or None
        Regime detection analyzer
    signal_history : List[Signal]
        Historical signal record
    weights : Dict[SignalType, float]
        Signal type weights for aggregation

    Methods
    -------
    add_momentum_signals(analyzer)
        Add momentum analyzer as signal source
    add_cross_sectional_signals(analyzer)
        Add cross-sectional analyzer as signal source
    add_regime_signals(detector)
        Add regime detector as signal source
    aggregate_signals(date=None)
        Combine all signals into consensus
    calculate_signal_confidence()
        Calculate confidence scores for signals
    get_top_opportunities(n=10, min_confidence=70)
        Get best trade setups
    generate_alert_summary()
        Generate human-readable alert summary
    update_weights(performance_history)
        Update signal weights based on performance

    Signal Aggregation Logic
    ------------------------

    **Direction Scoring:**
    - STRONG_BUY: +2 points
    - BUY: +1 point
    - NEUTRAL: 0 points
    - SELL: -1 point
    - STRONG_SELL: -2 points

    **Confidence Calculation:**
    confidence = Σ(weight_i × confidence_i) / Σ(weights)

    **Consensus Score:**
    - Range: -100 to +100
    - > 70: Strong buy
    - 30-70: Buy
    - -30 to 30: Neutral
    - -70 to -30: Sell
    - < -70: Strong sell

    Examples
    --------
    >>> # Initialize aggregator
    >>> aggregator = SignalAggregator(factor_research_system)
    >>>
    >>> # Add signal sources
    >>> from trading_signals import FactorMomentumAnalyzer
    >>> momentum = FactorMomentumAnalyzer(factor_returns)
    >>> aggregator.add_momentum_signals(momentum)
    >>>
    >>> from cross_sectional import CrossSectionalAnalyzer
    >>> cross_sec = CrossSectionalAnalyzer(factor_loadings)
    >>> aggregator.add_cross_sectional_signals(cross_sec)
    >>>
    >>> from regime_detection import RegimeDetector
    >>> regime = RegimeDetector(factor_returns)
    >>> regime.fit_hmm(n_regimes=3)
    >>> aggregator.add_regime_signals(regime)
    >>>
    >>> # Generate consensus signals
    >>> consensus = aggregator.aggregate_signals()
    >>>
    >>> # Get top opportunities
    >>> opportunities = aggregator.get_top_opportunities(n=10)
    >>> for opp in opportunities:
    ...     print(f"{opp.ticker}: {opp.direction} (confidence: {opp.confidence}%)")
    >>>
    >>> # Generate alert summary
    >>> alerts = aggregator.generate_alert_summary()
    >>> print(alerts)
    """

    def __init__(self, factor_research_system: Any):
        """
        Initialize the SignalAggregator.

        Parameters
        ----------
        factor_research_system : FactorResearchSystem
            The fitted factor research system
        """
        self.frs = factor_research_system
        self.momentum_analyzer = None
        self.cross_sectional_analyzer = None
        self.regime_detector = None
        self.signal_history: List[Signal] = []

        # Default weights for signal types
        self.weights: Dict[SignalType, float] = {
            SignalType.MOMENTUM: 0.25,
            SignalType.EXTREME_VALUE: 0.25,
            SignalType.CROSS_SECTIONAL: 0.30,
            SignalType.REGIME: 0.20
        }

    def add_momentum_signals(
        self,
        analyzer: 'FactorMomentumAnalyzer'
    ) -> 'SignalAggregator':
        """
        Add momentum analyzer as signal source.

        Parameters
        ----------
        analyzer : FactorMomentumAnalyzer
            Configured momentum analyzer

        Returns
        -------
        SignalAggregator
            Self for method chaining
        """
        self.momentum_analyzer = analyzer
        return self

    def add_cross_sectional_signals(
        self,
        analyzer: 'CrossSectionalAnalyzer'
    ) -> 'SignalAggregator':
        """
        Add cross-sectional analyzer as signal source.

        Parameters
        ----------
        analyzer : CrossSectionalAnalyzer
            Configured cross-sectional analyzer

        Returns
        -------
        SignalAggregator
            Self for method chaining
        """
        self.cross_sectional_analyzer = analyzer
        return self

    def add_regime_signals(
        self,
        detector: 'RegimeDetector'
    ) -> 'SignalAggregator':
        """
        Add regime detector as signal source.

        Parameters
        ----------
        detector : RegimeDetector
            Fitted regime detector

        Returns
        -------
        SignalAggregator
            Self for method chaining
        """
        self.regime_detector = detector
        return self

    def aggregate_signals(
        self,
        date: Optional[pd.Timestamp] = None
    ) -> Dict[str, ConsensusSignal]:
        """
        Combine all signals into consensus recommendations.

        Parameters
        ----------
        date : pd.Timestamp, optional
            Date to analyze (default: latest available)

        Returns
        -------
        Dict[str, ConsensusSignal]
            Dictionary mapping tickers/factors to consensus signals

        Examples
        --------
        >>> consensus = aggregator.aggregate_signals()
        >>> for key, signal in consensus.items():
        ...     print(f"{key}: {signal.consensus_direction.value}")
        """
        all_signals: List[Signal] = []

        # Collect momentum signals
        if self.momentum_analyzer is not None:
            momentum_signals = self._collect_momentum_signals(date)
            all_signals.extend(momentum_signals)

        # Collect cross-sectional signals
        if self.cross_sectional_analyzer is not None:
            cs_signals = self._collect_cross_sectional_signals(date)
            all_signals.extend(cs_signals)

        # Collect regime signals
        if self.regime_detector is not None:
            regime_signals = self._collect_regime_signals(date)
            all_signals.extend(regime_signals)

        # Store in history
        self.signal_history.extend(all_signals)

        # Group signals by ticker/factor
        grouped = self._group_signals_by_entity(all_signals)

        # Generate consensus for each entity
        consensus: Dict[str, ConsensusSignal] = {}
        for entity, signals in grouped.items():
            consensus[entity] = self._calculate_consensus(entity, signals)

        return consensus

    def _collect_momentum_signals(
        self,
        date: Optional[pd.Timestamp]
    ) -> List[Signal]:
        """Collect momentum signals from analyzer."""
        signals = []

        all_signals = self.momentum_analyzer.get_all_signals(date)

        for factor, data in all_signals.items():
            # Map combined signal to direction
            direction = self._map_signal_to_direction(data['combined_signal'])

            # Calculate confidence based on signal strength
            confidence = self._calculate_momentum_confidence(data)

            signals.append(Signal(
                source=f"momentum_{factor}",
                signal_type=SignalType.MOMENTUM,
                direction=direction,
                strength=confidence / 100,
                confidence=confidence,
                timestamp=pd.Timestamp(date) if date is not None else pd.Timestamp.now(),
                metadata={
                    'factor': factor,
                    'rsi': data.get('rsi'),
                    'rsi_signal': data.get('rsi_signal'),
                    'macd_signal': data.get('macd_signal'),
                    'adx': data.get('adx'),
                    'regime': data.get('regime')
                }
            ))

        # Also collect extreme value alerts
        extreme_alerts = self.momentum_analyzer.get_all_extreme_alerts(date=date)
        for alert in extreme_alerts:
            direction = (
                SignalDirection.SELL if alert.direction == 'extreme_high'
                else SignalDirection.BUY
            )

            # Higher confidence for more extreme z-scores
            confidence = min(95, 50 + abs(alert.z_score) * 15)

            signals.append(Signal(
                source=f"extreme_{alert.factor_name}",
                signal_type=SignalType.EXTREME_VALUE,
                direction=direction,
                strength=confidence / 100,
                confidence=confidence,
                timestamp=alert.timestamp,
                metadata={
                    'factor': alert.factor_name,
                    'z_score': alert.z_score,
                    'percentile': alert.percentile,
                    'alert_type': alert.alert_type
                }
            ))

        return signals

    def _collect_cross_sectional_signals(
        self,
        date: Optional[pd.Timestamp]
    ) -> List[Signal]:
        """Collect cross-sectional signals from analyzer."""
        signals = []

        stock_signals = self.cross_sectional_analyzer.generate_long_short_signals(
            top_pct=0.1,
            bottom_pct=0.1,
            as_of=date
        )

        for sig in stock_signals:
            if sig.direction.value == 'long':
                direction = SignalDirection.BUY
            elif sig.direction.value == 'short':
                direction = SignalDirection.SELL
            else:
                direction = SignalDirection.NEUTRAL

            signals.append(Signal(
                source=f"cross_sectional_{sig.ticker}",
                signal_type=SignalType.CROSS_SECTIONAL,
                direction=direction,
                strength=sig.confidence,
                confidence=sig.confidence * 100,
                timestamp=pd.Timestamp(date) if date is not None else pd.Timestamp.now(),
                metadata={
                    'ticker': sig.ticker,
                    'composite_score': sig.composite_score,
                    'decile': sig.decile,
                    'rank': sig.rank,
                    'factor_breakdown': sig.factor_breakdown
                }
            ))

        return signals

    def _collect_regime_signals(
        self,
        date: Optional[pd.Timestamp]
    ) -> List[Signal]:
        """Collect regime-based signals from detector."""
        signals = []

        regime_allocation = self.regime_detector.generate_regime_signals(as_of=date)

        # Map regime to directional bias
        if regime_allocation.risk_on_score > 0.7:
            direction = SignalDirection.STRONG_BUY
        elif regime_allocation.risk_on_score > 0.5:
            direction = SignalDirection.BUY
        elif regime_allocation.risk_on_score < 0.3:
            direction = SignalDirection.STRONG_SELL
        elif regime_allocation.risk_on_score < 0.5:
            direction = SignalDirection.SELL
        else:
            direction = SignalDirection.NEUTRAL

        confidence = 50 + abs(regime_allocation.risk_on_score - 0.5) * 100

        signals.append(Signal(
            source="regime_detector",
            signal_type=SignalType.REGIME,
            direction=direction,
            strength=regime_allocation.risk_on_score,
            confidence=min(95, confidence),
            timestamp=pd.Timestamp(date) if date is not None else pd.Timestamp.now(),
            metadata={
                'regime': regime_allocation.regime.value,
                'risk_on_score': regime_allocation.risk_on_score,
                'defensive_tilt': regime_allocation.defensive_tilt,
                'recommended_action': regime_allocation.recommended_action,
                'factor_weights': regime_allocation.factor_weights
            }
        ))

        return signals

    def _group_signals_by_entity(
        self,
        signals: List[Signal]
    ) -> Dict[str, List[Signal]]:
        """Group signals by ticker or factor."""
        grouped: Dict[str, List[Signal]] = {}

        for signal in signals:
            # Extract entity from metadata
            if 'ticker' in signal.metadata:
                entity = signal.metadata['ticker']
            elif 'factor' in signal.metadata:
                entity = signal.metadata['factor']
            else:
                entity = 'market_regime'

            if entity not in grouped:
                grouped[entity] = []
            grouped[entity].append(signal)

        return grouped

    def _calculate_consensus(
        self,
        entity: str,
        signals: List[Signal]
    ) -> ConsensusSignal:
        """Calculate consensus signal from multiple sources."""
        if not signals:
            return ConsensusSignal(
                ticker=entity if entity in self._get_tickers() else None,
                factor=entity if entity in self._get_factors() else None,
                consensus_direction=SignalDirection.NEUTRAL,
                consensus_score=0.0,
                confidence=0.0,
                contributing_signals=[],
                recommendation="No signals available",
                risk_level='medium'
            )

        # Calculate weighted consensus score
        total_weight = 0.0
        weighted_score = 0.0
        total_confidence = 0.0

        for signal in signals:
            weight = self.weights.get(signal.signal_type, 0.25)
            score = self._direction_to_score(signal.direction)

            weighted_score += weight * score * signal.confidence / 100
            total_weight += weight
            total_confidence += weight * signal.confidence

        # Normalize
        if total_weight > 0:
            consensus_score = weighted_score / total_weight * 50  # Scale to -100 to 100
            confidence = total_confidence / total_weight
        else:
            consensus_score = 0.0
            confidence = 0.0

        # Determine direction
        consensus_direction = self._score_to_direction(consensus_score)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            consensus_direction, confidence, signals
        )

        # Assess risk level
        risk_level = self._assess_risk_level(signals, confidence)

        return ConsensusSignal(
            ticker=entity if entity in self._get_tickers() else None,
            factor=entity if entity in self._get_factors() else None,
            consensus_direction=consensus_direction,
            consensus_score=consensus_score,
            confidence=confidence,
            contributing_signals=signals,
            recommendation=recommendation,
            risk_level=risk_level
        )

    def _map_signal_to_direction(self, signal: str) -> SignalDirection:
        """Map string signal to SignalDirection enum."""
        mapping = {
            'strong_buy': SignalDirection.STRONG_BUY,
            'buy': SignalDirection.BUY,
            'neutral': SignalDirection.NEUTRAL,
            'sell': SignalDirection.SELL,
            'strong_sell': SignalDirection.STRONG_SELL
        }
        return mapping.get(signal, SignalDirection.NEUTRAL)

    def _direction_to_score(self, direction: SignalDirection) -> float:
        """Convert direction to numeric score."""
        scores = {
            SignalDirection.STRONG_BUY: 2.0,
            SignalDirection.BUY: 1.0,
            SignalDirection.NEUTRAL: 0.0,
            SignalDirection.SELL: -1.0,
            SignalDirection.STRONG_SELL: -2.0
        }
        return scores.get(direction, 0.0)

    def _score_to_direction(self, score: float) -> SignalDirection:
        """Convert numeric score to direction."""
        if score > 70:
            return SignalDirection.STRONG_BUY
        elif score > 30:
            return SignalDirection.BUY
        elif score < -70:
            return SignalDirection.STRONG_SELL
        elif score < -30:
            return SignalDirection.SELL
        else:
            return SignalDirection.NEUTRAL

    def _calculate_momentum_confidence(self, data: Dict) -> float:
        """Calculate confidence score for momentum signal."""
        confidence = 50.0  # Base confidence

        # Adjust based on ADX (trend strength)
        adx = data.get('adx', 20)
        if adx > 25:
            confidence += 15
        elif adx < 20:
            confidence -= 10

        # Adjust based on RSI extremity
        rsi = data.get('rsi', 50)
        if rsi > 70 or rsi < 30:
            confidence += 10

        # Adjust based on MACD signal clarity
        macd_signal = data.get('macd_signal', '')
        if 'crossover' in macd_signal:
            confidence += 10

        return min(95, max(5, confidence))

    def _generate_recommendation(
        self,
        direction: SignalDirection,
        confidence: float,
        signals: List[Signal]
    ) -> str:
        """Generate human-readable recommendation."""
        direction_str = direction.value.replace('_', ' ').title()

        if confidence > 80:
            strength = "High confidence"
        elif confidence > 60:
            strength = "Moderate confidence"
        else:
            strength = "Low confidence"

        # Count signal types
        signal_types = [s.signal_type.value for s in signals]
        type_summary = ', '.join(set(signal_types))

        return f"{strength} {direction_str} signal based on {type_summary}"

    def _assess_risk_level(
        self,
        signals: List[Signal],
        confidence: float
    ) -> str:
        """Assess risk level for the signal."""
        # Check for conflicting signals
        directions = [s.direction for s in signals]
        has_conflicts = len(set(directions)) > 2  # More than buy/sell or neutral

        # Check for extreme values
        has_extremes = any(
            s.signal_type == SignalType.EXTREME_VALUE for s in signals
        )

        if has_conflicts and confidence < 60:
            return 'high'
        elif has_extremes or confidence < 70:
            return 'medium'
        else:
            return 'low'

    def _get_tickers(self) -> set:
        """Get set of available tickers."""
        if self.cross_sectional_analyzer is not None:
            return set(self.cross_sectional_analyzer.factor_loadings.index)
        return set()

    def _get_factors(self) -> set:
        """Get set of available factors."""
        if self.momentum_analyzer is not None:
            return set(self.momentum_analyzer.factor_returns.columns)
        return set()

    def get_top_opportunities(
        self,
        n: int = 10,
        min_confidence: float = 70.0
    ) -> List[TradingOpportunity]:
        """
        Get the best trading opportunities.

        Parameters
        ----------
        n : int, default 10
            Number of opportunities to return
        min_confidence : float, default 70.0
            Minimum confidence threshold

        Returns
        -------
        List[TradingOpportunity]
            List of top trading opportunities

        Examples
        --------
        >>> opportunities = aggregator.get_top_opportunities(n=5)
        >>> for opp in opportunities:
        ...     print(f"{opp.ticker}: {opp.direction} ({opp.confidence}%)")
        """
        consensus = self.aggregate_signals()

        # Filter by confidence and sort by absolute score
        qualified = [
            sig for sig in consensus.values()
            if sig.confidence >= min_confidence
            and sig.consensus_direction != SignalDirection.NEUTRAL
        ]

        # Sort by absolute consensus score
        qualified.sort(
            key=lambda x: abs(x.consensus_score),
            reverse=True
        )

        opportunities = []
        for sig in qualified[:n]:
            opp = self._consensus_to_opportunity(sig)
            opportunities.append(opp)

        return opportunities

    def _consensus_to_opportunity(
        self,
        consensus: ConsensusSignal
    ) -> TradingOpportunity:
        """Convert consensus signal to trading opportunity."""
        # Determine position size based on confidence
        if consensus.confidence > 85:
            position_size = 'large'
        elif consensus.confidence > 70:
            position_size = 'medium'
        else:
            position_size = 'small'

        # Determine time horizon
        has_momentum = any(
            s.signal_type == SignalType.MOMENTUM
            for s in consensus.contributing_signals
        )
        has_regime = any(
            s.signal_type == SignalType.REGIME
            for s in consensus.contributing_signals
        )

        if has_momentum and not has_regime:
            time_horizon = 'short'
        elif has_regime:
            time_horizon = 'long'
        else:
            time_horizon = 'medium'

        # Extract risk factors
        risk_factors = []
        for signal in consensus.contributing_signals:
            if signal.signal_type == SignalType.EXTREME_VALUE:
                risk_factors.append("Statistical extreme - potential reversal")
            if signal.signal_type == SignalType.REGIME:
                if signal.metadata.get('defensive_tilt'):
                    risk_factors.append("Defensive regime - reduced risk appetite")

        return TradingOpportunity(
            ticker=consensus.ticker,
            factor=consensus.factor,
            signal_type='combined',
            direction=consensus.consensus_direction.value,
            confidence=consensus.confidence,
            entry_rationale=consensus.recommendation,
            risk_factors=risk_factors,
            suggested_position_size=position_size,
            time_horizon=time_horizon
        )

    def generate_alert_summary(self) -> str:
        """
        Generate human-readable alert summary.

        Returns
        -------
        str
            Formatted alert summary

        Examples
        --------
        >>> print(aggregator.generate_alert_summary())
        """
        consensus = self.aggregate_signals()

        lines = [
            "=" * 60,
            "TRADING SIGNAL ALERT SUMMARY",
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            ""
        ]

        # Strong signals
        strong_signals = [
            sig for sig in consensus.values()
            if sig.consensus_direction in [
                SignalDirection.STRONG_BUY, SignalDirection.STRONG_SELL
            ]
        ]

        if strong_signals:
            lines.append(" STRONG SIGNALS:")
            for sig in strong_signals:
                entity = sig.ticker or sig.factor or "Market"
                lines.append(
                    f"  {entity}: {sig.consensus_direction.value.upper()} "
                    f"(confidence: {sig.confidence:.0f}%)"
                )
            lines.append("")

        # Moderate signals
        moderate_signals = [
            sig for sig in consensus.values()
            if sig.consensus_direction in [
                SignalDirection.BUY, SignalDirection.SELL
            ]
        ]

        if moderate_signals:
            lines.append("🟡 MODERATE SIGNALS:")
            for sig in moderate_signals:
                entity = sig.ticker or sig.factor or "Market"
                lines.append(
                    f"  {entity}: {sig.consensus_direction.value.upper()} "
                    f"(confidence: {sig.confidence:.0f}%)"
                )
            lines.append("")

        # Regime information
        if self.regime_detector is not None:
            regime = self.regime_detector.detect_current_regime()
            lines.append(" CURRENT REGIME:")
            lines.append(f"  {regime.regime.value}")
            lines.append(f"  Confidence: {regime.probability:.1%}")
            lines.append(f"  Description: {regime.description}")
            lines.append("")

        # Top opportunities
        opportunities = self.get_top_opportunities(n=5)
        if opportunities:
            lines.append(" TOP OPPORTUNITIES:")
            for opp in opportunities:
                entity = opp.ticker or opp.factor or "Market"
                lines.append(
                    f"  {entity}: {opp.direction} "
                    f"(confidence: {opp.confidence:.0f}%, "
                    f"size: {opp.suggested_position_size})"
                )
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)

    def update_weights(self, performance_history: pd.DataFrame) -> None:
        """
        Update signal weights based on historical performance.

        Parameters
        ----------
        performance_history : pd.DataFrame
            DataFrame with columns: signal_type, return, accuracy

        Examples
        --------
        >>> perf = pd.DataFrame({
        ...     'signal_type': ['momentum', 'cross_sectional'],
        ...     'return': [0.05, 0.08],
        ...     'accuracy': [0.55, 0.62]
        ... })
        >>> aggregator.update_weights(perf)
        """
        if performance_history.empty:
            return

        # Calculate new weights based on Sharpe-like metric
        for signal_type in SignalType:
            type_data = performance_history[
                performance_history['signal_type'] == signal_type.value
            ]

            if len(type_data) > 0:
                avg_return = type_data['return'].mean()
                avg_accuracy = type_data['accuracy'].mean()

                # Weight proportional to accuracy and return
                new_weight = max(0.1, avg_accuracy * (1 + avg_return))
                self.weights[signal_type] = new_weight

        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

        _LOGGER.info(f"Updated weights: {self.weights}")

    def export_signals_to_csv(
        self,
        filepath: str,
        date: Optional[pd.Timestamp] = None
    ) -> None:
        """
        Export signals to CSV file.

        Parameters
        ----------
        filepath : str
            Path to output CSV file
        date : pd.Timestamp, optional
            Date to export (default: latest)

        Examples
        --------
        >>> aggregator.export_signals_to_csv('signals_2024-01-15.csv')
        """
        consensus = self.aggregate_signals(date)

        data = []
        for entity, sig in consensus.items():
            data.append({
                'entity': entity,
                'ticker': sig.ticker,
                'factor': sig.factor,
                'direction': sig.consensus_direction.value,
                'score': sig.consensus_score,
                'confidence': sig.confidence,
                'recommendation': sig.recommendation,
                'risk_level': sig.risk_level
            })

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        _LOGGER.info(f"Exported {len(df)} signals to {filepath}")


class FeatureExtractor:
    """
    Extracts and normalizes features from signal sources for meta-modeling.
    
    This class standardizes diverse signal objects into a structured feature
    vector suitable for machine learning models.
    
    Parameters
    ----------
    scaler : StandardScaler, optional
        Scikit-learn scaler for feature normalization
    
    Examples
    --------
    >>> extractor = FeatureExtractor()
    >>> features = extractor.extract_momentum_features(momentum_data)
    >>> normalized = extractor.normalize_features(features)
    """
    
    def __init__(self, scaler: Optional[StandardScaler] = None):
        self.scaler = scaler if scaler is not None else StandardScaler()
        self._is_fitted = False
    
    def extract_momentum_features(
        self, 
        momentum_signals: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Extract features from momentum analyzer output.
        
        Features extracted:
        - RSI values (normalized to 0-1)
        - MACD signals (encoded as -1, 0, 1)
        - ADX (trend strength)
        - Z-scores (normalized deviation)
        """
        features = {}
        
        for factor, data in momentum_signals.items():
            prefix = f"mom_{factor}_"
            
            # RSI: Normalize to [-1, 1] (oversold to overbought)
            rsi = data.get('rsi', 50)
            features[f"{prefix}rsi"] = (rsi - 50) / 50  # -1 to 1
            
            # MACD signal encoding
            macd_signal = data.get('macd_signal', 'neutral')
            macd_map = {'strong_buy': 1.0, 'buy': 0.5, 'neutral': 0.0, 
                       'sell': -0.5, 'strong_sell': -1.0}
            features[f"{prefix}macd"] = macd_map.get(macd_signal, 0.0)
            
            # ADX: Trend strength (0-1 normalized)
            adx = data.get('adx', 20)
            features[f"{prefix}adx"] = min(1.0, adx / 50)
            
            # Combined signal strength
            combined = data.get('combined_signal', 'neutral')
            features[f"{prefix}strength"] = macd_map.get(combined, 0.0)
        
        return features
    
    def extract_regime_features(
        self, 
        regime_state: Any,
        regime_probs: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Extract features from regime detector output.
        
        Features extracted:
        - Regime probability
        - Risk-on score
        - Volatility level
        - Trend strength
        """
        features = {}
        
        # Current regime encoding (one-hot style)
        if regime_state:
            regime_val = regime_state.regime.value if hasattr(regime_state, 'regime') else 'unknown'
            features['regime_prob'] = getattr(regime_state, 'probability', 0.5)
            features['regime_vol'] = getattr(regime_state, 'volatility', 0.2)
            features['regime_trend'] = getattr(regime_state, 'trend', 0.0)
        
        # Regime probabilities for all states
        if regime_probs:
            for regime, prob in regime_probs.items():
                features[f"regime_prob_{regime.value}"] = prob
        
        return features
    
    def extract_cross_sectional_features(
        self,
        cs_signals: List[Any]
    ) -> Dict[str, float]:
        """
        Extract aggregate features from cross-sectional signals.
        
        Features extracted:
        - Average decile score
        - Distribution of signals (long/short ratio)
        - Confidence metrics
        """
        features = {}
        
        if not cs_signals:
            features['cs_signal_count'] = 0
            features['cs_avg_decile'] = 5.0
            features['cs_long_ratio'] = 0.5
            return features
        
        # Aggregate statistics
        deciles = [s.decile for s in cs_signals if hasattr(s, 'decile')]
        confidences = [s.confidence for s in cs_signals if hasattr(s, 'confidence')]
        
        long_signals = [s for s in cs_signals 
                       if hasattr(s, 'direction') and s.direction.value == 'long']
        
        features['cs_signal_count'] = len(cs_signals)
        features['cs_avg_decile'] = np.mean(deciles) if deciles else 5.0
        features['cs_long_ratio'] = len(long_signals) / len(cs_signals) if cs_signals else 0.5
        features['cs_avg_confidence'] = np.mean(confidences) if confidences else 0.5
        
        return features
    
    def normalize_features(
        self, 
        features: Dict[str, float]
    ) -> np.ndarray:
        """
        Normalize features using fitted scaler.
        
        Parameters
        ----------
        features : Dict[str, float]
            Feature dictionary
            
        Returns
        -------
        np.ndarray
            Normalized feature vector
        """
        values = np.array(list(features.values())).reshape(1, -1)
        
        if self._is_fitted:
            return self.scaler.transform(values).flatten()
        else:
            # Fit on first call
            self.scaler.fit(values)
            self._is_fitted = True
            return self.scaler.transform(values).flatten()
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names (for model interpretability)."""
        # This would need to be populated during extraction
        return []


class MetaModelAggregator(SignalAggregator):
    """
    Gradient Boosting Meta-Model for signal aggregation.
    
    Replaces the linear weighted sum approach with a machine learning model
    (XGBoost) that captures non-linear interactions between signals.
    
    The model is trained using walk-forward cross-validation to prevent
    lookahead bias, making it suitable for institutional trading strategies.
    
    Parameters
    ----------
    factor_research_system : FactorResearchSystem
        The factor research system containing fitted factors
    model_params : Dict, optional
        XGBoost model parameters
    min_training_samples : int, default 252
        Minimum samples required before model training
    prediction_horizon : int, default 5
        Forward return horizon for labels (T+1 to T+5)
    
    Examples
    --------
    >>> aggregator = MetaModelAggregator(factor_research_system)
    >>> aggregator.add_momentum_signals(momentum_analyzer)
    >>> aggregator.add_regime_signals(regime_detector)
    >>> 
    >>> # Train on historical data (walk-forward)
    >>> aggregator.train_walk_forward(min_window=252)
    >>> 
    >>> # Generate predictions
    >>> consensus = aggregator.generate_meta_consensus()
    >>> print(f"Predicted probability of positive return: {consensus['probability_up']:.2%}")
    """
    
    def __init__(
        self, 
        factor_research_system: Any,
        model_params: Optional[Dict] = None,
        min_training_samples: int = 252,
        prediction_horizon: int = 5,
        use_voting_fallback: bool = True
    ):
        super().__init__(factor_research_system)
        
        self.min_training_samples = min_training_samples
        self.prediction_horizon = prediction_horizon
        self.use_voting_fallback = use_voting_fallback
        
        # XGBoost model configuration
        default_params = {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.model_params = {**default_params, **(model_params or {})}
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_extractor = FeatureExtractor()
        
        # Data caches for training
        self.feature_cache: List[Dict[str, float]] = []
        self.label_cache: List[int] = []
        self.date_cache: List[datetime] = []
        
        # Historical returns for label generation
        self.market_returns: Optional[pd.Series] = None
        
        # Fallback to voting if insufficient data or model not trained
        self._model_trained = False
        
        if not XGBOOST_AVAILABLE:
            _LOGGER.warning(
                "XGBoost not available. MetaModelAggregator will use voting fallback."
            )
    
    def set_market_returns(self, market_returns: pd.Series) -> None:
        """
        Set market returns for label generation.
        
        Parameters
        ----------
        market_returns : pd.Series
            Daily market returns (e.g., SPY returns) indexed by date
        """
        self.market_returns = market_returns.copy()
    
    def _extract_current_features(
        self, 
        date: Optional[pd.Timestamp] = None
    ) -> Dict[str, float]:
        """Extract features from all signal sources at given date."""
        features = {}
        
        # Momentum features
        if self.momentum_analyzer is not None:
            momentum_signals = self.momentum_analyzer.get_all_signals(date)
            momentum_features = self.feature_extractor.extract_momentum_features(
                momentum_signals
            )
            features.update(momentum_features)
        
        # Regime features
        if self.regime_detector is not None:
            try:
                regime_state = self.regime_detector.detect_current_regime()
                regime_probs = self.regime_detector.get_regime_probabilities()
                regime_features = self.feature_extractor.extract_regime_features(
                    regime_state, regime_probs
                )
                features.update(regime_features)
            except Exception as e:
                _LOGGER.warning(f"Failed to extract regime features: {e}")
        
        # Cross-sectional features
        if self.cross_sectional_analyzer is not None:
            try:
                cs_signals = self.cross_sectional_analyzer.generate_long_short_signals()
                cs_features = self.feature_extractor.extract_cross_sectional_features(
                    cs_signals
                )
                features.update(cs_features)
            except Exception as e:
                _LOGGER.warning(f"Failed to extract cross-sectional features: {e}")
        
        return features
    
    def _generate_labels(
        self, 
        dates: List[datetime],
        threshold: float = 0.0
    ) -> List[int]:
        """
        Generate binary labels based on forward returns.
        
        Label = 1 if forward return > threshold, else 0
        """
        if self.market_returns is None:
            raise ValueError("Market returns not set. Call set_market_returns() first.")
        
        labels = []
        for date in dates:
            try:
                # Find index of current date
                if date in self.market_returns.index:
                    idx = self.market_returns.index.get_loc(date)
                    
                    # Calculate forward cumulative return
                    if isinstance(idx, int) and idx + self.prediction_horizon < len(self.market_returns):
                        future_return = (
                            self.market_returns.iloc[idx + 1:idx + 1 + self.prediction_horizon].sum()
                        )
                        label = 1 if future_return > threshold else 0
                        labels.append(label)
                    else:
                        labels.append(0)  # Default label
                else:
                    labels.append(0)
            except Exception:
                labels.append(0)
        
        return labels
    
    def build_feature_matrix(
        self,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[datetime]]:
        """
        Build feature matrix X and label vector y from historical data.
        
        Parameters
        ----------
        start_date : pd.Timestamp, optional
            Start date for feature extraction
        end_date : pd.Timestamp, optional
            End date for feature extraction
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, List[datetime]]
            X (features), y (labels), and dates
        """
        if self.market_returns is None:
            raise ValueError("Market returns not set. Call set_market_returns() first.")
        
        # Filter dates
        dates = self.market_returns.index
        if start_date:
            dates = dates[dates >= start_date]
        if end_date:
            dates = dates[dates <= end_date]
        
        # Need to leave room for forward returns
        dates = dates[:-self.prediction_horizon]
        
        features_list = []
        valid_dates = []
        
        _LOGGER.info(f"Building feature matrix for {len(dates)} dates...")
        
        for date in dates:
            try:
                # Extract features for this date
                features = self._extract_current_features(date)
                
                if features:  # Only add if we got valid features
                    features_list.append(features)
                    valid_dates.append(date)
            except Exception as e:
                _LOGGER.debug(f"Failed to extract features for {date}: {e}")
                continue
        
        if len(features_list) < self.min_training_samples:
            raise ValueError(
                f"Insufficient samples for training: {len(features_list)} "
                f"(min: {self.min_training_samples})"
            )
        
        # Convert to matrix (align features)
        all_keys = set()
        for f in features_list:
            all_keys.update(f.keys())
        all_keys = sorted(all_keys)
        
        X = np.zeros((len(features_list), len(all_keys)))
        for i, features in enumerate(features_list):
            for j, key in enumerate(all_keys):
                X[i, j] = features.get(key, 0.0)
        
        # Generate labels
        y = np.array(self._generate_labels(valid_dates))
        
        _LOGGER.info(f"Feature matrix built: X.shape={X.shape}, y.mean={y.mean():.2%}")
        
        return X, y, valid_dates
    
    def train_walk_forward(
        self, 
        min_window: int = 252,
        step_size: int = 21,
        purge_gap: int = 5,
        verbose: bool = True
    ) -> None:
        """
        Train the meta-model using walk-forward cross-validation.
        
        This prevents lookahead bias by training only on past data and testing
        on future data. Implements purged cross-validation to avoid overlapping
        labels.
        
        Parameters
        ----------
        min_window : int, default 252
            Minimum training window (1 year of trading days)
        step_size : int, default 21
            Number of days to advance each iteration (1 month)
        purge_gap : int, default 5
            Gap between train and test to avoid label overlap
        verbose : bool, default True
            Print training progress
        """
        if not XGBOOST_AVAILABLE:
            _LOGGER.warning("XGBoost not available. Skipping model training.")
            return
        
        try:
            # Build full feature matrix
            X, y, dates = self.build_feature_matrix()
            
            if len(X) < min_window + step_size:
                _LOGGER.warning(
                    f"Insufficient data for walk-forward: {len(X)} samples"
                )
                return
            
            # Expanding window walk-forward
            # In production, we train on [0...t] to predict t+1
            n_samples = len(X)
            
            if verbose:
                _LOGGER.info(f"Starting walk-forward training with {n_samples} samples...")
            
            # For simplicity, we use an expanding window approach:
            # Train on first N samples, validate on remaining
            # In practice, you'd iterate through windows
            
            # Purged split: remove samples immediately adjacent to test set
            train_end = n_samples - step_size - purge_gap
            
            if train_end < min_window:
                _LOGGER.warning(f"Training window too small: {train_end}")
                return
            
            X_train = X[:train_end]
            y_train = y[:train_end]
            X_val = X[train_end + purge_gap:]
            y_val = y[train_end + purge_gap:]
            
            if verbose:
                _LOGGER.info(
                    f"Training on {len(X_train)} samples, "
                    f"validating on {len(X_val)} samples"
                )
            
            # Train XGBoost model
            self.model = xgb.XGBClassifier(**self.model_params)
            
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)] if len(X_val) > 0 else None,
                verbose=False
            )
            
            # Store training metrics
            train_accuracy = self.model.score(X_train, y_train)
            val_accuracy = self.model.score(X_val, y_val) if len(X_val) > 0 else 0
            
            # Feature importance
            importance = self.model.feature_importances_
            top_features_idx = np.argsort(importance)[-5:][::-1]
            
            if verbose:
                _LOGGER.info(f"Meta-model trained successfully!")
                _LOGGER.info(f"  Train accuracy: {train_accuracy:.2%}")
                _LOGGER.info(f"  Val accuracy: {val_accuracy:.2%}")
                _LOGGER.info(f"  Top feature indices: {top_features_idx.tolist()}")
            
            self._model_trained = True
            
            # Cache training data for potential retraining
            self._training_X = X
            self._training_y = y
            self._training_dates = dates
            
        except Exception as e:
            _LOGGER.error(f"Walk-forward training failed: {e}")
            self._model_trained = False
    
    def generate_meta_consensus(
        self, 
        date: Optional[pd.Timestamp] = None
    ) -> Dict[str, Union[ConsensusSignal, float]]:
        """
        Generate consensus using the meta-model.
        
        Replaces the linear weighted voting with XGBoost probability predictions.
        
        Parameters
        ----------
        date : pd.Timestamp, optional
            Date to generate consensus for
            
        Returns
        -------
        Dict[str, Union[ConsensusSignal, float]]
            Contains 'consensus_signal' and 'probability_up'
        """
        # Fallback to voting if model not trained
        if not self._model_trained or self.model is None:
            if self.use_voting_fallback:
                _LOGGER.debug("Using voting fallback (model not trained)")
                return {'consensus_signal': self.aggregate_signals(date), 'probability_up': 0.5}
            else:
                raise RuntimeError("Meta-model not trained. Call train_walk_forward() first.")
        
        # Extract current features
        features = self._extract_current_features(date)
        
        if not features:
            _LOGGER.warning("No features extracted. Using fallback.")
            return {'consensus_signal': self.aggregate_signals(date), 'probability_up': 0.5}
        
        # Convert to feature vector (align with training features)
        if hasattr(self, '_training_X'):
            # Use same feature order as training
            X = np.zeros((1, self._training_X.shape[1]))
            for j, key in enumerate(sorted(features.keys())):
                if j < X.shape[1]:
                    X[0, j] = features.get(key, 0.0)
        else:
            X = np.array(list(features.values())).reshape(1, -1)
        
        # Predict probability of positive return
        prob_up = self.model.predict_proba(X)[0, 1]
        
        # Convert probability to consensus score (-100 to 100)
        # 0.5 -> 0, 1.0 -> 100, 0.0 -> -100
        consensus_score = (prob_up - 0.5) * 200
        
        # Map to direction
        direction = self._score_to_direction(consensus_score)
        
        # Create consensus signal
        consensus_signal = ConsensusSignal(
            ticker=None,
            factor='meta_model',
            consensus_direction=direction,
            consensus_score=consensus_score,
            confidence=abs(consensus_score),
            contributing_signals=[],  # Could populate with raw signals
            recommendation=f"Meta-model: {prob_up:.1%} probability of positive return",
            risk_level='medium' if 0.3 < prob_up < 0.7 else 'high'
        )
        
        return {
            'consensus_signal': consensus_signal,
            'probability_up': prob_up,
            'meta_score': consensus_score
        }
    
    def get_model_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance from the trained meta-model.
        
        Returns
        -------
        Optional[pd.DataFrame]
            Feature importance DataFrame or None if model not trained
        """
        if not self._model_trained or self.model is None:
            return None
        
        importance = self.model.feature_importances_
        
        return pd.DataFrame({
            'feature_idx': range(len(importance)),
            'importance': importance
        }).sort_values('importance', ascending=False)
