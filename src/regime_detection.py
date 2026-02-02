"""
Market Regime Detection using Hidden Markov Models
===================================================

This module implements regime detection for factor-based trading strategies
using Hidden Markov Models (HMM). It identifies market regimes (bull, bear,
volatile, calm) and provides regime-based factor allocation recommendations.

Core Components
---------------

**RegimeDetector**
- Hidden Markov Model training and inference
- Real-time regime probability estimation
- Regime-based factor performance analysis
- Optimal factor recommendations per regime

Regime Types
------------

**Low Volatility Bull:**
- Characteristics: Rising prices, low volatility
- Optimal factors: Momentum, Growth
- Strategy: Aggressive long exposure

**High Volatility Bear:**
- Characteristics: Falling prices, high volatility
- Optimal factors: Quality, Value, Low Volatility
- Strategy: Defensive positioning

**Transition:**
- Characteristics: Mixed signals, regime uncertainty
- Optimal factors: Diversified mix
- Strategy: Reduce factor exposure

**Crisis:**
- Characteristics: Extreme volatility, correlations → 1
- Optimal factors: Minimum volatility, Quality
- Strategy: Maximum defensive positioning

Architecture
------------
```
Factor Returns → HMM Training → Regime Inference → Allocation Recommendation
       ↓              ↓                ↓                    ↓
   [T x K]      [Baum-Welch]    [Forward-Backward]    [Factor Weights]
```

Dependencies
------------
- pandas, numpy: Data manipulation
- hmmlearn: Hidden Markov Model implementation
- scikit-learn: Preprocessing and model selection

Examples
--------
>>> from regime_detection import RegimeDetector
>>> detector = RegimeDetector(factor_returns)
>>> detector.fit_hmm(n_regimes=3)
>>> current_regime = detector.detect_current_regime()
>>> optimal_factors = detector.get_regime_optimal_factors(current_regime)
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Optional hmmlearn import with fallback
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    warnings.warn("hmmlearn not installed. Regime detection will use simplified methods.")

_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.WARNING)

if not _LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
    handler.setFormatter(formatter)
    _LOGGER.addHandler(handler)
    _LOGGER.propagate = False


class MarketRegime(Enum):
    """Enumeration of market regime classifications."""
    LOW_VOL_BULL = "low_volatility_bull"
    HIGH_VOL_BULL = "high_volatility_bull"
    LOW_VOL_BEAR = "low_volatility_bear"
    HIGH_VOL_BEAR = "high_volatility_bear"
    TRANSITION = "transition"
    CRISIS = "crisis"
    UNKNOWN = "unknown"


@dataclass
class RegimeState:
    """Data class representing a detected regime state."""
    regime: MarketRegime
    probability: float
    volatility: float
    trend: float
    description: str


@dataclass
class RegimeAllocation:
    """Data class representing regime-based factor allocation."""
    regime: MarketRegime
    factor_weights: Dict[str, float]
    risk_on_score: float  # 0-1, higher = more aggressive
    defensive_tilt: bool
    recommended_action: str


class RegimeDetector:
    """
    Market regime detection using Hidden Markov Models.

    This class implements regime detection for factor-based trading strategies.
    It uses HMM to identify hidden market states and provides regime-based
    factor allocation recommendations.

    Parameters
    ----------
    factor_returns : pd.DataFrame
        Factor returns DataFrame with shape (T, K) where:
        - Index: Trading dates (datetime)
        - Columns: Factor identifiers
        - Values: Daily factor returns

    Attributes
    ----------
    factor_returns : pd.DataFrame
        Stored factor returns
    n_factors : int
        Number of factors
    scaler : StandardScaler
        Feature scaler for HMM input
    hmm_model : GaussianHMM or None
        Fitted HMM model
    regime_labels : Dict[int, MarketRegime]
        Mapping from HMM states to regime labels
    regime_history : pd.DataFrame
        Historical regime classifications

    Methods
    -------
    fit_hmm(n_regimes=3, covariance_type='full')
        Train HMM on factor returns
    detect_current_regime()
        Detect current market regime
    get_regime_probabilities()
        Get probability distribution over regimes
    get_regime_optimal_factors(regime)
        Get best performing factors for a regime
    generate_regime_signals()
        Generate allocation adjustment signals
    analyze_regime_transitions()
        Analyze historical regime transitions
    predict_regime(duration=5)
        Predict regime for next N periods

    Regime Detection Logic
    ----------------------

    **HMM State Characterization:**
    After fitting, states are characterized by:
    - Mean return: Positive = bull, negative = bear
    - Volatility: Standard deviation of returns
    - Persistence: State transition probabilities

    **State Labeling:**
    States are automatically labeled based on:
    1. Mean return sign and magnitude
    2. Volatility level (vs historical average)
    3. Transition matrix patterns

    **Regime-Based Recommendations:**
    - Low Vol Bull: Overweight momentum, growth
    - High Vol Bull: Reduce size, maintain momentum
    - Low Vol Bear: Overweight quality, value
    - High Vol Bear: Maximum defensive, min volatility
    - Crisis: Risk-off, cash/quality focus

    Examples
    --------
    >>> # Initialize detector
    >>> detector = RegimeDetector(factor_returns)
    >>>
    >>> # Fit HMM with 3 regimes
    >>> detector.fit_hmm(n_regimes=3)
    >>>
    >>> # Detect current regime
    >>> regime = detector.detect_current_regime()
    >>> print(f"Current regime: {regime.regime.value}")
    >>> print(f"Confidence: {regime.probability:.1%}")
    >>>
    >>> # Get optimal factors for current regime
    >>> optimal = detector.get_regime_optimal_factors(regime.regime)
    >>> print(f"Recommended factors: {optimal}")
    >>>
    >>> # Generate allocation signals
    >>> signals = detector.generate_regime_signals()
    >>> print(f"Risk-on score: {signals.risk_on_score:.2f}")
    >>> print(f"Recommended action: {signals.recommended_action}")
    """

    def __init__(self, factor_returns: pd.DataFrame):
        """
        Initialize the RegimeDetector.

        Parameters
        ----------
        factor_returns : pd.DataFrame
            Factor returns with datetime index
        """
        self.factor_returns = factor_returns.copy()
        self.n_factors = factor_returns.shape[1]
        self.scaler = StandardScaler()
        self.hmm_model = None
        self.regime_labels: Dict[int, MarketRegime] = {}
        self.regime_history: Optional[pd.DataFrame] = None
        self._validate_data()

    def _validate_data(self) -> None:
        """Validate input factor returns data."""
        if self.factor_returns.empty:
            raise ValueError("Factor returns DataFrame is empty")

        if not isinstance(self.factor_returns.index, pd.DatetimeIndex):
            _LOGGER.warning("Factor returns index is not DatetimeIndex, converting...")
            try:
                self.factor_returns.index = pd.to_datetime(self.factor_returns.index)
            except Exception as e:
                raise ValueError(f"Could not convert index to datetime: {e}")

        # Check for minimum data
        min_required = 252  # Minimum for reliable regime detection
        if len(self.factor_returns) < min_required:
            _LOGGER.warning(
                f"Factor returns only has {len(self.factor_returns)} observations, "
                f"minimum recommended is {min_required} for reliable regime detection"
            )

        if not HMM_AVAILABLE:
            _LOGGER.warning(
                "hmmlearn not available. Install with: uv add hmmlearn"
            )

    def fit_hmm(
        self,
        n_regimes: int = 3,
        covariance_type: str = 'full',
        n_iter: int = 100,
        random_state: int = 42
    ) -> 'RegimeDetector':
        """
        Train Hidden Markov Model on factor returns.

        Parameters
        ----------
        n_regimes : int, default 3
            Number of hidden states (2-4 recommended)
        covariance_type : str, default 'full'
            Covariance matrix type: 'full', 'tied', 'diag', 'spherical'
        n_iter : int, default 100
            Maximum number of EM iterations
        random_state : int, default 42
            Random seed for reproducibility

        Returns
        -------
        RegimeDetector
            Self for method chaining

        HMM Architecture
        ----------------
        **State Space:** n_regimes hidden states
        **Observations:** Factor returns (multivariate Gaussian)
        **Emissions:** Gaussian distributions per state
        **Transitions:** Learned transition matrix

        Training Process
        ----------------
        1. Initialize parameters randomly
        2. E-step: Compute posterior probabilities (forward-backward)
        3. M-step: Update parameters to maximize likelihood
        4. Repeat until convergence

        Examples
        --------
        >>> detector.fit_hmm(n_regimes=3)
        >>> detector.fit_hmm(n_regimes=4, covariance_type='diag')
        """
        if not HMM_AVAILABLE:
            raise ImportError(
                "hmmlearn is required for HMM regime detection. "
                "Install with: uv add hmmlearn"
            )

        # Prepare data
        X = self.factor_returns.values
        X_scaled = self.scaler.fit_transform(X)

        # Remove any NaN values
        valid_mask = ~np.isnan(X_scaled).any(axis=1)
        X_clean = X_scaled[valid_mask]

        if len(X_clean) < 100:
            raise ValueError(
                f"Insufficient valid data after cleaning: {len(X_clean)} observations"
            )

        _LOGGER.info(f"Fitting HMM with {n_regimes} regimes on {len(X_clean)} observations")

        # Fit HMM
        self.hmm_model = GaussianHMM(
            n_components=n_regimes,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
            verbose=False
        )

        self.hmm_model.fit(X_clean)

        _LOGGER.info(f"HMM converged: {self.hmm_model.monitor_.converged}")
        _LOGGER.info(f"Log likelihood: {self.hmm_model.score(X_clean):.2f}")

        # Label regimes based on learned parameters
        self._label_regimes()

        # Generate regime history
        self._generate_regime_history(X_scaled, valid_mask)

        return self

    def _label_regimes(self) -> None:
        """Label HMM states with regime classifications."""
        if self.hmm_model is None:
            return

        n_regimes = self.hmm_model.n_components
        means = self.hmm_model.means_

        # Calculate portfolio return for each state (equal-weighted factors)
        state_returns = means.mean(axis=1)

        # Calculate volatility for each state
        state_volatilities = np.array([
            np.sqrt(np.diag(self.hmm_model.covars_[i])).mean()
            if self.hmm_model.covariance_type == 'full'
            else np.sqrt(self.hmm_model.covars_[i]).mean()
            for i in range(n_regimes)
        ])

        # Historical volatility for comparison
        hist_vol = self.factor_returns.std().mean()

        # Label each state
        for i in range(n_regimes):
            ret = state_returns[i]
            vol = state_volatilities[i]

            if vol > hist_vol * 1.5:
                # High volatility
                if ret > 0:
                    self.regime_labels[i] = MarketRegime.HIGH_VOL_BULL
                else:
                    self.regime_labels[i] = MarketRegime.HIGH_VOL_BEAR
            elif vol > hist_vol * 1.2:
                # Moderate-high volatility
                if ret > 0:
                    self.regime_labels[i] = MarketRegime.HIGH_VOL_BULL
                elif ret < -0.001:
                    self.regime_labels[i] = MarketRegime.HIGH_VOL_BEAR
                else:
                    self.regime_labels[i] = MarketRegime.TRANSITION
            else:
                # Low volatility
                if ret > 0.001:
                    self.regime_labels[i] = MarketRegime.LOW_VOL_BULL
                elif ret < -0.001:
                    self.regime_labels[i] = MarketRegime.LOW_VOL_BEAR
                else:
                    self.regime_labels[i] = MarketRegime.TRANSITION

        # Check for crisis regime (very high volatility)
        max_vol_idx = np.argmax(state_volatilities)
        if state_volatilities[max_vol_idx] > hist_vol * 2:
            self.regime_labels[max_vol_idx] = MarketRegime.CRISIS

        _LOGGER.info(f"Regime labels: {self.regime_labels}")

    def _generate_regime_history(
        self,
        X_scaled: np.ndarray,
        valid_mask: np.ndarray
    ) -> None:
        """Generate historical regime classifications."""
        if self.hmm_model is None:
            return

        # Predict states for all observations
        X_clean = X_scaled[valid_mask]
        states = self.hmm_model.predict(X_clean)

        # Create regime history DataFrame
        dates = self.factor_returns.index[valid_mask]
        self.regime_history = pd.DataFrame({
            'state': states,
            'regime': [self.regime_labels.get(s, MarketRegime.UNKNOWN) for s in states]
        }, index=dates)

        # Add factor returns
        for col in self.factor_returns.columns:
            self.regime_history[col] = self.factor_returns.loc[dates, col].values

    def detect_current_regime(self) -> RegimeState:
        """
        Detect the current market regime.

        Returns
        -------
        RegimeState
            Current regime classification with probability

        Examples
        --------
        >>> regime = detector.detect_current_regime()
        >>> print(f"Current: {regime.regime.value} ({regime.probability:.1%})")
        """
        if self.hmm_model is None:
            raise RuntimeError("HMM not fitted. Call fit_hmm() first.")

        # Get latest observation
        latest = self.factor_returns.iloc[-1:].values
        latest_scaled = self.scaler.transform(latest)

        # Predict state probabilities using predict_proba for normalized probabilities
        probs = self.hmm_model.predict_proba(latest_scaled)[0]

        # Get most likely state
        most_likely_state = np.argmax(probs)
        max_prob = probs[most_likely_state]

        regime = self.regime_labels.get(most_likely_state, MarketRegime.UNKNOWN)

        # Calculate recent volatility and trend
        recent_returns = self.factor_returns.iloc[-63:]  # Last 3 months
        volatility = recent_returns.std().mean()
        trend = recent_returns.mean().mean()

        # Generate description
        descriptions = {
            MarketRegime.LOW_VOL_BULL: "Low volatility bull market - favor momentum",
            MarketRegime.HIGH_VOL_BULL: "High volatility bull - reduce exposure",
            MarketRegime.LOW_VOL_BEAR: "Low volatility bear - defensive positioning",
            MarketRegime.HIGH_VOL_BEAR: "High volatility bear - maximum defense",
            MarketRegime.TRANSITION: "Regime transition - maintain diversification",
            MarketRegime.CRISIS: "Crisis regime - risk off, preserve capital",
            MarketRegime.UNKNOWN: "Unknown regime - exercise caution"
        }

        return RegimeState(
            regime=regime,
            probability=max_prob,
            volatility=volatility,
            trend=trend,
            description=descriptions.get(regime, "Unknown")
        )

    def get_regime_probabilities(self) -> Dict[MarketRegime, float]:
        """
        Get probability distribution over all regimes.

        Returns
        -------
        Dict[MarketRegime, float]
            Probability for each regime

        Examples
        --------
        >>> probs = detector.get_regime_probabilities()
        >>> for regime, prob in probs.items():
        ...     print(f"{regime.value}: {prob:.1%}")
        """
        if self.hmm_model is None:
            raise RuntimeError("HMM not fitted. Call fit_hmm() first.")

        # Get latest observation
        latest = self.factor_returns.iloc[-1:].values
        latest_scaled = self.scaler.transform(latest)

        # Get state probabilities using predict_proba
        probs = self.hmm_model.predict_proba(latest_scaled)[0]

        # Aggregate by regime
        regime_probs: Dict[MarketRegime, float] = {}
        for state, prob in enumerate(probs):
            regime = self.regime_labels.get(state, MarketRegime.UNKNOWN)
            regime_probs[regime] = regime_probs.get(regime, 0.0) + prob

        return regime_probs

    def get_regime_optimal_factors(
        self,
        regime: MarketRegime,
        lookback: int = 63
    ) -> Dict[str, float]:
        """
        Get optimal factor weights for a given regime.

        Parameters
        ----------
        regime : MarketRegime
            Target regime
        lookback : int, default 63
            Lookback period for performance calculation

        Returns
        -------
        Dict[str, float]
            Recommended factor weights

        Examples
        --------
        >>> current = detector.detect_current_regime()
        >>> optimal = detector.get_regime_optimal_factors(current.regime)
        >>> print(f"Recommended weights: {optimal}")
        """
        if self.regime_history is None:
            raise RuntimeError("HMM not fitted. Call fit_hmm() first.")

        # Get periods in target regime
        regime_periods = self.regime_history[
            self.regime_history['regime'] == regime
        ]

        if len(regime_periods) < 10:
            _LOGGER.warning(
                f"Only {len(regime_periods)} periods in {regime.value} regime, "
                "using default weights"
            )
            return self._get_default_factor_weights(regime)

        # Calculate factor performance in this regime
        factor_cols = [c for c in self.regime_history.columns
                      if c not in ['state', 'regime']]

        regime_returns = regime_periods[factor_cols].mean()
        regime_sharpe = (
            regime_periods[factor_cols].mean() /
            regime_periods[factor_cols].std().replace(0, np.nan)
        )

        # Weight by Sharpe ratio (positive only)
        positive_sharpe = regime_sharpe[regime_sharpe > 0]
        if len(positive_sharpe) > 0:
            weights = positive_sharpe / positive_sharpe.sum()
        else:
            # All negative Sharpe - use minimum volatility
            weights = 1.0 / len(factor_cols)
            weights = pd.Series(weights, index=factor_cols)

        return weights.to_dict()

    def _get_default_factor_weights(self, regime: MarketRegime) -> Dict[str, float]:
        """Get default factor weights based on regime type."""
        factor_cols = [c for c in self.regime_history.columns
                      if c not in ['state', 'regime']]

        weights = {}

        if regime == MarketRegime.LOW_VOL_BULL:
            # Favor momentum and growth
            for f in factor_cols:
                if 'momentum' in f.lower() or 'growth' in f.lower():
                    weights[f] = 0.3
                else:
                    weights[f] = 0.1

        elif regime == MarketRegime.HIGH_VOL_BULL:
            # Balanced with quality tilt
            for f in factor_cols:
                if 'quality' in f.lower():
                    weights[f] = 0.25
                else:
                    weights[f] = 0.15

        elif regime in [MarketRegime.LOW_VOL_BEAR, MarketRegime.HIGH_VOL_BEAR]:
            # Defensive - value and quality
            for f in factor_cols:
                if 'value' in f.lower() or 'quality' in f.lower():
                    weights[f] = 0.3
                elif 'momentum' in f.lower():
                    weights[f] = 0.05
                else:
                    weights[f] = 0.15

        elif regime == MarketRegime.CRISIS:
            # Maximum defensive
            for f in factor_cols:
                if 'min_vol' in f.lower() or 'quality' in f.lower():
                    weights[f] = 0.4
                else:
                    weights[f] = 0.05

        else:  # TRANSITION or UNKNOWN
            # Equal weight
            n = len(factor_cols)
            weights = {f: 1.0/n for f in factor_cols}

        # Normalize to sum to 1
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}

    def generate_regime_signals(self) -> RegimeAllocation:
        """
        Generate allocation adjustment signals based on current regime.

        Returns
        -------
        RegimeAllocation
            Regime-based allocation recommendation

        Examples
        --------
        >>> allocation = detector.generate_regime_signals()
        >>> print(f"Risk-on score: {allocation.risk_on_score:.2f}")
        >>> print(f"Action: {allocation.recommended_action}")
        """
        current = self.detect_current_regime()
        optimal_factors = self.get_regime_optimal_factors(current.regime)

        # Calculate risk-on score
        risk_scores = {
            MarketRegime.LOW_VOL_BULL: 0.9,
            MarketRegime.HIGH_VOL_BULL: 0.7,
            MarketRegime.LOW_VOL_BEAR: 0.3,
            MarketRegime.HIGH_VOL_BEAR: 0.1,
            MarketRegime.TRANSITION: 0.5,
            MarketRegime.CRISIS: 0.0,
            MarketRegime.UNKNOWN: 0.5
        }
        risk_on_score = risk_scores.get(current.regime, 0.5)

        # Determine defensive tilt
        defensive_regimes = [
            MarketRegime.LOW_VOL_BEAR,
            MarketRegime.HIGH_VOL_BEAR,
            MarketRegime.CRISIS
        ]
        defensive_tilt = current.regime in defensive_regimes

        # Generate recommendation
        recommendations = {
            MarketRegime.LOW_VOL_BULL: "Increase factor exposure, overweight momentum",
            MarketRegime.HIGH_VOL_BULL: "Maintain exposure, add quality tilt",
            MarketRegime.LOW_VOL_BEAR: "Reduce exposure, favor value and quality",
            MarketRegime.HIGH_VOL_BEAR: "Significant de-risking required",
            MarketRegime.TRANSITION: "Maintain diversification, await clarity",
            MarketRegime.CRISIS: "Maximum defensive positioning, preserve capital",
            MarketRegime.UNKNOWN: "Exercise caution, reduce factor exposure"
        }

        return RegimeAllocation(
            regime=current.regime,
            factor_weights=optimal_factors,
            risk_on_score=risk_on_score,
            defensive_tilt=defensive_tilt,
            recommended_action=recommendations.get(
                current.regime,
                "Maintain current allocation"
            )
        )

    def analyze_regime_transitions(self) -> pd.DataFrame:
        """
        Analyze historical regime transition probabilities.

        Returns
        -------
        pd.DataFrame
            Transition matrix with regime labels

        Examples
        --------
        >>> transitions = detector.analyze_regime_transitions()
        >>> print(transitions)
        """
        if self.regime_history is None:
            raise RuntimeError("HMM not fitted. Call fit_hmm() first.")

        if self.hmm_model is None:
            return pd.DataFrame()

        # Get transition matrix from HMM
        transmat = self.hmm_model.transmat_

        # Create labeled DataFrame
        states = list(self.regime_labels.keys())
        labels = [self.regime_labels[s].value for s in states]

        transition_df = pd.DataFrame(
            transmat[np.ix_(states, states)],
            index=labels,
            columns=labels
        )

        return transition_df

    def predict_regime(self, duration: int = 5) -> List[RegimeState]:
        """
        Predict regime for the next N periods.

        Parameters
        ----------
        duration : int, default 5
            Number of periods to predict

        Returns
        -------
        List[RegimeState]
            Predicted regime states

        Examples
        --------
        >>> predictions = detector.predict_regime(duration=5)
        >>> for i, pred in enumerate(predictions):
        ...     print(f"Day {i+1}: {pred.regime.value}")
        """
        if self.hmm_model is None:
            raise RuntimeError("HMM not fitted. Call fit_hmm() first.")

        # Get current state probabilities
        current = self.detect_current_regime()
        current_state = [
            s for s, r in self.regime_labels.items()
            if r == current.regime
        ][0]

        # Propagate forward
        transmat = self.hmm_model.transmat_
        state_probs = np.zeros(self.hmm_model.n_components)
        state_probs[current_state] = 1.0

        predictions = []
        for _ in range(duration):
            # Step forward
            state_probs = state_probs @ transmat

            # Get most likely regime
            most_likely = np.argmax(state_probs)
            regime = self.regime_labels.get(most_likely, MarketRegime.UNKNOWN)

            # Get regime characteristics
            mean = self.hmm_model.means_[most_likely].mean()
            vol = np.sqrt(np.diag(self.hmm_model.covars_[most_likely])).mean()

            predictions.append(RegimeState(
                regime=regime,
                probability=state_probs[most_likely],
                volatility=vol,
                trend=mean,
                description=f"Predicted: {regime.value}"
            ))

        return predictions

    def get_regime_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for each detected regime.

        Returns
        -------
        pd.DataFrame
            Regime summary statistics
        """
        if self.regime_history is None:
            raise RuntimeError("HMM not fitted. Call fit_hmm() first.")

        factor_cols = [c for c in self.regime_history.columns
                      if c not in ['state', 'regime']]

        summary = []
        for regime in self.regime_history['regime'].unique():
            regime_data = self.regime_history[
                self.regime_history['regime'] == regime
            ]

            summary.append({
                'regime': regime.value,
                'count': len(regime_data),
                'pct_time': len(regime_data) / len(self.regime_history),
                'avg_return': regime_data[factor_cols].mean().mean(),
                'avg_volatility': regime_data[factor_cols].std().mean(),
                'sharpe': (
                    regime_data[factor_cols].mean().mean() /
                    regime_data[factor_cols].std().mean()
                ) if regime_data[factor_cols].std().mean() > 0 else 0
            })

        return pd.DataFrame(summary)


class SimpleRegimeDetector:
    """
    Simplified regime detector without HMM dependency.

    Uses rule-based classification when hmmlearn is not available.
    """

    def __init__(self, factor_returns: pd.DataFrame):
        self.factor_returns = factor_returns

    def detect_current_regime(self) -> RegimeState:
        """Detect regime using simple rules."""
        recent = self.factor_returns.iloc[-63:]  # Last 3 months

        mean_return = recent.mean().mean()
        volatility = recent.std().mean()
        hist_vol = self.factor_returns.std().mean()

        # Simple classification
        if volatility > hist_vol * 1.5:
            if mean_return > 0:
                regime = MarketRegime.HIGH_VOL_BULL
                desc = "High volatility bull market"
            else:
                regime = MarketRegime.HIGH_VOL_BEAR
                desc = "High volatility bear market"
        elif mean_return > 0.001:
            regime = MarketRegime.LOW_VOL_BULL
            desc = "Low volatility bull market"
        elif mean_return < -0.001:
            regime = MarketRegime.LOW_VOL_BEAR
            desc = "Low volatility bear market"
        else:
            regime = MarketRegime.TRANSITION
            desc = "Transition regime"

        return RegimeState(
            regime=regime,
            probability=0.7,  # Lower confidence for rule-based
            volatility=volatility,
            trend=mean_return,
            description=desc
        )

    def get_regime_optimal_factors(self, regime: MarketRegime) -> Dict[str, float]:
        """Get default factor weights."""
        n = len(self.factor_returns.columns)
        return {col: 1.0/n for col in self.factor_returns.columns}
