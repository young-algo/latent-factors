"""
Trading Signal Generation System - Factor Momentum and Extreme Value Analysis
============================================================================

This module provides comprehensive trading signal generation capabilities for
factor-based investment strategies. It implements technical analysis indicators
(RSI, MACD, ADX) and statistical extreme value detection (z-scores, Bollinger
Bands, percentiles) to identify actionable trading opportunities from factor
returns.

Core Components
---------------

**FactorMomentumAnalyzer**
- Technical momentum indicators (RSI, MACD, ROC, ADX)
- Momentum regime detection (trending vs mean-reverting)
- Statistical extreme value alerts (z-scores, Bollinger Bands)
- Percentile-based ranking within historical distributions

Signal Types
------------

**Momentum Signals:**
- RSI > 70: Overbought (potential short)
- RSI < 30: Oversold (potential long)
- MACD crossover: Momentum shift signal
- ADX > 25: Strong trend confirmed

**Extreme Value Signals:**
- Z-score > +2.0: Extreme positive (reversal short candidate)
- Z-score < -2.0: Extreme negative (reversal long candidate)
- Beyond 95th/5th percentile: Statistical extreme alert
- 3-sigma events: Black swan detection

Architecture
------------
```
Factor Returns → Technical Indicators → Signal Detection → Alert Generation
     ↓                ↓                      ↓                    ↓
   [F1, F2]      [RSI, MACD]          [Threshold Check]    [Signal Dict]
```

Dependencies
------------
- pandas, numpy: Data manipulation and calculations
- pandas-ta: Technical analysis indicators (optional)

Examples
--------
>>> from trading_signals import FactorMomentumAnalyzer
>>> analyzer = FactorMomentumAnalyzer(factor_returns)
>>> rsi = analyzer.calculate_rsi('F1')
>>> alerts = analyzer.get_extreme_alerts()
>>> regime = analyzer.detect_momentum_regime('F1')
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.WARNING)

if not _LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
    handler.setFormatter(formatter)
    _LOGGER.addHandler(handler)
    _LOGGER.propagate = False


class MomentumRegime(Enum):
    """Enumeration of momentum regime classifications."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    NEUTRAL = "neutral"


class SignalStrength(Enum):
    """Enumeration of signal strength levels."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NONE = "none"


@dataclass
class TradingSignal:
    """Data class representing a trading signal."""
    factor_name: str
    signal_type: str
    direction: str  # 'long', 'short', 'neutral'
    strength: SignalStrength
    confidence: float  # 0-100
    value: float
    threshold: float
    timestamp: pd.Timestamp
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ExtremeAlert:
    """Data class representing an extreme value alert."""
    factor_name: str
    alert_type: str
    z_score: float
    percentile: float
    current_value: float
    threshold: float
    direction: str  # 'extreme_high', 'extreme_low'
    timestamp: pd.Timestamp


class FactorMomentumAnalyzer:
    """
    Technical and statistical analysis engine for factor returns.

    This class provides comprehensive momentum and extreme value analysis
    for factor returns, generating actionable trading signals through
    technical indicators (RSI, MACD, ADX) and statistical measures
    (z-scores, Bollinger Bands, percentiles).

    Parameters
    ----------
    factor_returns : pd.DataFrame
        Factor returns DataFrame with shape (T, K) where:
        - Index: Trading dates (datetime)
        - Columns: Factor identifiers (e.g., 'F1', 'F2', ...)
        - Values: Daily factor returns

    Attributes
    ----------
    factor_returns : pd.DataFrame
        Stored factor returns for analysis
    n_factors : int
        Number of factors in the returns matrix
    indicators : Dict[str, pd.DataFrame]
        Cache of calculated technical indicators

    Methods
    -------
    calculate_rsi(factor_name, period=14)
        Calculate Relative Strength Index for a factor
    calculate_macd(factor_name, fast=12, slow=26, signal=9)
        Calculate MACD with signal line and histogram
    calculate_adx(factor_name, period=14)
        Calculate Average Directional Index for trend strength
    calculate_roc(factor_name, periods=[21, 63, 126])
        Calculate Rate of Change over multiple lookbacks
    detect_momentum_regime(factor_name)
        Classify current momentum regime
    calculate_zscore(factor_name, window=20)
        Calculate rolling z-scores for extreme detection
    calculate_bollinger_bands(factor_name, window=20, num_std=2)
        Calculate Bollinger Bands for factor returns
    get_percentile_rank(factor_name, lookback=252)
        Calculate percentile rank within historical distribution
    check_extreme_levels(factor_name, z_threshold=2.0)
        Check for statistical extremes and generate alerts
    get_all_signals(date=None)
        Generate all trading signals for a given date

    Signal Generation Logic
    -----------------------
    **RSI Signals:**
    - RSI > 70: Overbought signal (potential mean reversion short)
    - RSI < 30: Oversold signal (potential mean reversion long)
    - RSI 30-70: Neutral zone

    **MACD Signals:**
    - MACD crosses above signal: Bullish momentum
    - MACD crosses below signal: Bearish momentum
    - Histogram direction: Momentum acceleration/deceleration

    **ADX Signals:**
    - ADX > 25: Strong trend (follow trend)
    - ADX < 20: Weak trend (mean reversion possible)
    - ADX 20-25: Transition zone

    **Z-Score Signals:**
    - |z| > 2.0: Statistical extreme (potential reversal)
    - |z| > 3.0: Rare event (black swan detection)

    Examples
    --------
    >>> # Initialize analyzer with factor returns
    >>> analyzer = FactorMomentumAnalyzer(factor_returns)
    >>>
    >>> # Calculate RSI for factor F1
    >>> rsi = analyzer.calculate_rsi('F1', period=14)
    >>>
    >>> # Get MACD signals
    >>> macd_line, signal_line, histogram = analyzer.calculate_macd('F1')
    >>>
    >>> # Detect current regime
    >>> regime = analyzer.detect_momentum_regime('F1')
    >>> print(f"Current regime: {regime.value}")
    >>>
    >>> # Check for extremes
    >>> alerts = analyzer.check_extreme_levels('F1', z_threshold=2.0)
    >>>
    >>> # Get all signals for today
    >>> signals = analyzer.get_all_signals()
    """

    def __init__(self, factor_returns: pd.DataFrame):
        """
        Initialize the FactorMomentumAnalyzer.

        Parameters
        ----------
        factor_returns : pd.DataFrame
            Factor returns with datetime index and factor columns
        """
        self.factor_returns = factor_returns.copy()
        self.n_factors = factor_returns.shape[1]
        self._indicators_cache: Dict[str, Dict] = {}
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

        # Check for required data length
        min_required = 63  # Minimum for meaningful analysis
        if len(self.factor_returns) < min_required:
            _LOGGER.warning(
                f"Factor returns only has {len(self.factor_returns)} observations, "
                f"minimum recommended is {min_required}"
            )

    def _resolve_date_index(self, date: Optional[pd.Timestamp]) -> int:
        """
        Resolve an as-of date to the nearest available index position <= date.

        Parameters
        ----------
        date : pd.Timestamp | None
            Target as-of date. If None, returns the latest index.

        Returns
        -------
        int
            Integer position in `self.factor_returns.index`.
        """
        if date is None:
            return len(self.factor_returns.index) - 1

        ts = pd.Timestamp(date)
        loc = self.factor_returns.index.searchsorted(ts, side="right") - 1
        if loc < 0:
            raise ValueError(
                f"As-of date {ts.date()} is before first observation "
                f"{self.factor_returns.index[0].date()}"
            )
        return int(loc)

    def calculate_rsi(self, factor_name: str, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI) for a factor.

        RSI is a momentum oscillator that measures the speed and magnitude
        of recent price changes. It oscillates between 0 and 100, with
        traditional overbought/oversold levels at 70/30.

        Parameters
        ----------
        factor_name : str
            Name of the factor column to analyze
        period : int, default 14
            Lookback period for RSI calculation (standard is 14 days)

        Returns
        -------
        pd.Series
            RSI values indexed by date (0-100 scale)

        Calculation
        -----------
        RS = Average Gain / Average Loss
        RSI = 100 - (100 / (1 + RS))

        Interpretation
        --------------
        - RSI > 70: Overbought (potential short/reversal)
        - RSI < 30: Oversold (potential long/reversal)
        - RSI 50: Neutral momentum
        - Divergences: Price makes new high but RSI doesn't = weakness

        Examples
        --------
        >>> rsi = analyzer.calculate_rsi('F1', period=14)
        >>> current_rsi = rsi.iloc[-1]
        >>> if current_rsi > 70:
        ...     print("Factor is overbought")
        """
        if factor_name not in self.factor_returns.columns:
            raise ValueError(f"Factor '{factor_name}' not found in returns")

        returns = self.factor_returns[factor_name]

        # Calculate gains and losses
        gains = returns.where(returns > 0, 0)
        losses = -returns.where(returns < 0, 0)

        # Calculate average gains and losses using exponential moving average
        avg_gains = gains.ewm(alpha=1/period, min_periods=period).mean()
        avg_losses = losses.ewm(alpha=1/period, min_periods=period).mean()

        # Calculate RS and RSI
        rs = avg_gains / avg_losses.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_macd(
        self,
        factor_name: str,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence) for a factor.

        MACD is a trend-following momentum indicator that shows the relationship
        between two moving averages of prices. It consists of the MACD line,
        signal line, and histogram.

        Parameters
        ----------
        factor_name : str
            Name of the factor column to analyze
        fast : int, default 12
            Fast EMA period
        slow : int, default 26
            Slow EMA period
        signal : int, default 9
            Signal line EMA period

        Returns
        -------
        tuple of pd.Series
            (macd_line, signal_line, histogram)
            - macd_line: Fast EMA - Slow EMA
            - signal_line: EMA of MACD line
            - histogram: MACD line - Signal line

        Interpretation
        --------------
        - MACD crosses above signal: Bullish (buy signal)
        - MACD crosses below signal: Bearish (sell signal)
        - Histogram positive and growing: Increasing bullish momentum
        - Histogram negative and growing: Increasing bearish momentum
        - Divergence: Price makes new high but MACD doesn't = potential reversal

        Examples
        --------
        >>> macd, signal_line, hist = analyzer.calculate_macd('F1')
        >>> if macd.iloc[-1] > signal_line.iloc[-1] and macd.iloc[-2] <= signal_line.iloc[-2]:
        ...     print("Bullish MACD crossover detected")
        """
        if factor_name not in self.factor_returns.columns:
            raise ValueError(f"Factor '{factor_name}' not found in returns")

        # Calculate cumulative returns (price-like series)
        prices = (1 + self.factor_returns[factor_name]).cumprod()

        # Calculate EMAs
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()

        # MACD line
        macd_line = ema_fast - ema_slow

        # Signal line
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()

        # Histogram
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def calculate_adx(self, factor_name: str, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index (ADX) for trend strength.

        ADX measures the strength of a trend regardless of direction.
        It ranges from 0 to 100, with higher values indicating stronger trends.

        Parameters
        ----------
        factor_name : str
            Name of the factor column to analyze
        period : int, default 14
            Lookback period for ADX calculation

        Returns
        -------
        pd.Series
            ADX values indexed by date (0-100 scale)

        Interpretation
        --------------
        - ADX < 20: Weak trend (range-bound, mean reversion strategies)
        - ADX 20-25: Transition zone
        - ADX 25-50: Strong trend (trend following strategies)
        - ADX 50-75: Very strong trend
        - ADX > 75: Extremely strong trend (rare, potential exhaustion)

        Calculation
        -----------
        1. Calculate +DM and -DM (Directional Movement)
        2. Calculate TR (True Range)
        3. Smooth +DM, -DM, and TR
        4. Calculate +DI and -DI
        5. Calculate DX = |+DI - -DI| / |+DI + -DI| * 100
        6. ADX = smoothed moving average of DX

        Examples
        --------
        >>> adx = analyzer.calculate_adx('F1')
        >>> if adx.iloc[-1] > 25:
        ...     print("Strong trend confirmed - use trend following")
        ... else:
        ...     print("Weak trend - consider mean reversion")
        """
        if factor_name not in self.factor_returns.columns:
            raise ValueError(f"Factor '{factor_name}' not found in returns")

        # Use cumulative returns as price proxy
        prices = (1 + self.factor_returns[factor_name]).cumprod()

        # Calculate True Range components
        high = prices.rolling(window=2).max()
        low = prices.rolling(window=2).min()
        close_prev = prices.shift(1)

        # True Range
        tr1 = high - low
        tr2 = abs(high - close_prev)
        tr3 = abs(low - close_prev)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        # Smoothed averages
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * plus_dm.rolling(window=period).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.rolling(window=period).mean() / atr.replace(0, np.nan)

        # Directional Index and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.rolling(window=period).mean()

        return adx

    def calculate_roc(
        self,
        factor_name: str,
        periods: List[int] = None
    ) -> pd.DataFrame:
        """
        Calculate Rate of Change (ROC) over multiple lookback periods.

        ROC measures the percentage change in price over a specified period,
        indicating momentum strength and direction.

        Parameters
        ----------
        factor_name : str
            Name of the factor column to analyze
        periods : List[int], optional
            List of lookback periods in days
            Default: [21, 63, 126] (1 month, 3 months, 6 months)

        Returns
        -------
        pd.DataFrame
            ROC values for each period, columns named 'ROC_{period}'

        Interpretation
        --------------
        - ROC > 0: Positive momentum (price increased)
        - ROC < 0: Negative momentum (price decreased)
        - ROC crossing zero: Momentum direction change
        - Multiple period alignment: Stronger signal

        Examples
        --------
        >>> roc = analyzer.calculate_roc('F1', periods=[21, 63, 126])
        >>> if all(roc.iloc[-1] > 0):
        ...     print("Positive momentum across all timeframes")
        """
        if factor_name not in self.factor_returns.columns:
            raise ValueError(f"Factor '{factor_name}' not found in returns")

        if periods is None:
            periods = [21, 63, 126]

        prices = (1 + self.factor_returns[factor_name]).cumprod()

        roc_data = {}
        for period in periods:
            roc = 100 * (prices / prices.shift(period) - 1)
            roc_data[f'ROC_{period}'] = roc

        return pd.DataFrame(roc_data)

    def detect_momentum_regime(
        self,
        factor_name: str,
        adx_threshold: float = 25.0,
        rsi_period: int = 14,
        date: Optional[pd.Timestamp] = None
    ) -> MomentumRegime:
        """
        Detect the current momentum regime for a factor.

        Classifies the factor's momentum state into one of four regimes:
        trending_up, trending_down, mean_reverting, or neutral.

        Parameters
        ----------
        factor_name : str
            Name of the factor column to analyze
        adx_threshold : float, default 25.0
            ADX level to distinguish trending from mean-reverting
        rsi_period : int, default 14
            Period for RSI calculation
        date : pd.Timestamp, optional
            As-of date for regime detection. Uses data up to and including date.

        Returns
        -------
        MomentumRegime
            Enum value indicating current regime classification

        Regime Classification
        ---------------------
        - **TRENDING_UP**: ADX > threshold and positive momentum
        - **TRENDING_DOWN**: ADX > threshold and negative momentum
        - **MEAN_REVERTING**: ADX < threshold (weak trend)
        - **NEUTRAL**: Indeterminate state

        Trading Implications
        --------------------
        - Trending regimes: Use trend-following strategies
        - Mean-reverting: Use contrarian/reversal strategies
        - Neutral: Reduce exposure or wait for clarity

        Examples
        --------
        >>> regime = analyzer.detect_momentum_regime('F1')
        >>> if regime == MomentumRegime.TRENDING_UP:
        ...     print("Go long - strong upward trend")
        ... elif regime == MomentumRegime.MEAN_REVERTING:
        ...     print("Consider mean reversion strategy")
        """
        if factor_name not in self.factor_returns.columns:
            raise ValueError(f"Factor '{factor_name}' not found in returns")

        idx = self._resolve_date_index(date)

        # Get as-of values
        adx = self.calculate_adx(factor_name).iloc[idx]
        rsi = self.calculate_rsi(factor_name, period=rsi_period).iloc[idx]

        # Get recent momentum direction from ROC
        roc = self.calculate_roc(factor_name, periods=[21]).iloc[idx, 0]

        if pd.isna(adx) or pd.isna(rsi):
            return MomentumRegime.NEUTRAL

        # Classify regime
        if adx > adx_threshold:
            # Strong trend
            if rsi > 55 or roc > 0:
                return MomentumRegime.TRENDING_UP
            elif rsi < 45 or roc < 0:
                return MomentumRegime.TRENDING_DOWN
            else:
                return MomentumRegime.NEUTRAL
        else:
            # Weak trend - mean reverting environment
            return MomentumRegime.MEAN_REVERTING

    def calculate_zscore(
        self,
        factor_name: str,
        window: int = 20
    ) -> pd.Series:
        """
        Calculate rolling z-scores for extreme value detection.

        Z-scores measure how many standard deviations a value is from
        the mean, useful for identifying statistical extremes.

        Parameters
        ----------
        factor_name : str
            Name of the factor column to analyze
        window : int, default 20
            Rolling window size for mean/std calculation

        Returns
        -------
        pd.Series
            Z-scores indexed by date

        Interpretation
        --------------
        - |z| < 1: Within normal range (~68% of data)
        - 1 < |z| < 2: Moderate deviation (~27% of data)
        - 2 < |z| < 3: Extreme deviation (~4.5% of data)
        - |z| > 3: Very extreme (<0.3% of data, potential reversal)

        Examples
        --------
        >>> zscore = analyzer.calculate_zscore('F1', window=20)
        >>> if abs(zscore.iloc[-1]) > 2:
        ...     print("Factor at statistical extreme")
        """
        if factor_name not in self.factor_returns.columns:
            raise ValueError(f"Factor '{factor_name}' not found in returns")

        returns = self.factor_returns[factor_name]

        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()

        zscore = (returns - rolling_mean) / rolling_std.replace(0, np.nan)

        return zscore

    def calculate_bollinger_bands(
        self,
        factor_name: str,
        window: int = 20,
        num_std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands for factor returns.

        Bollinger Bands consist of a middle band (moving average) and
        upper/lower bands (moving average +/- standard deviations).

        Parameters
        ----------
        factor_name : str
            Name of the factor column to analyze
        window : int, default 20
            Rolling window for moving average and standard deviation
        num_std : float, default 2.0
            Number of standard deviations for bands

        Returns
        -------
        tuple of pd.Series
            (upper_band, middle_band, lower_band)

        Interpretation
        --------------
        - Price near upper band: Potentially overbought
        - Price near lower band: Potentially oversold
        - Band squeeze: Low volatility (potential breakout imminent)
        - Band expansion: High volatility
        - %B indicator: (Price - Lower) / (Upper - Lower)

        Examples
        --------
        >>> upper, middle, lower = analyzer.calculate_bollinger_bands('F1')
        >>> current_return = analyzer.factor_returns['F1'].iloc[-1]
        >>> if current_return > upper.iloc[-1]:
        ...     print("Return above upper band - potential reversal")
        """
        if factor_name not in self.factor_returns.columns:
            raise ValueError(f"Factor '{factor_name}' not found in returns")

        returns = self.factor_returns[factor_name]

        middle_band = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()

        upper_band = middle_band + (rolling_std * num_std)
        lower_band = middle_band - (rolling_std * num_std)

        return upper_band, middle_band, lower_band

    def get_percentile_rank(
        self,
        factor_name: str,
        lookback: int = 252
    ) -> pd.Series:
        """
        Calculate percentile rank within historical distribution.

        Shows where the current value ranks relative to recent history
        (0 = lowest, 100 = highest).

        Parameters
        ----------
        factor_name : str
            Name of the factor column to analyze
        lookback : int, default 252
            Number of periods to include in historical distribution
            (252 ≈ 1 year of trading days)

        Returns
        -------
        pd.Series
            Percentile ranks (0-100) indexed by date

        Interpretation
        --------------
        - 0-5: Extreme low (potential long opportunity)
        - 5-25: Lower quartile
        - 25-75: Middle 50% (normal range)
        - 75-95: Upper quartile
        - 95-100: Extreme high (potential short opportunity)

        Examples
        --------
        >>> pct_rank = analyzer.get_percentile_rank('F1', lookback=252)
        >>> if pct_rank.iloc[-1] > 95:
        ...     print("Factor at 95th+ percentile - potential reversal")
        """
        if factor_name not in self.factor_returns.columns:
            raise ValueError(f"Factor '{factor_name}' not found in returns")

        returns = self.factor_returns[factor_name]

        def rolling_percentile(x):
            if len(x) < 2:
                return np.nan
            current = x[-1]
            return 100 * (x < current).mean()

        percentile = returns.rolling(window=lookback).apply(
            rolling_percentile, raw=True
        )

        return percentile

    def check_extreme_levels(
        self,
        factor_name: str,
        z_threshold: float = 2.0,
        percentile_threshold: float = 95.0,
        date: Optional[pd.Timestamp] = None
    ) -> Optional[ExtremeAlert]:
        """
        Check if a factor is at extreme levels and generate alert.

        Combines z-score and percentile analysis to identify statistical
        extremes that may indicate reversal opportunities.

        Parameters
        ----------
        factor_name : str
            Name of the factor column to analyze
        z_threshold : float, default 2.0
            Z-score threshold for extreme detection
        percentile_threshold : float, default 95.0
            Percentile threshold for extreme detection
        date : pd.Timestamp, optional
            As-of date for extreme detection. Uses data up to date.

        Returns
        -------
        ExtremeAlert or None
            Alert object if extreme detected, None otherwise

        Alert Types
        -----------
        - 'zscore_extreme': Z-score beyond threshold
        - 'percentile_extreme': Percentile beyond threshold
        - 'combined_extreme': Both z-score and percentile extreme

        Examples
        --------
        >>> alert = analyzer.check_extreme_levels('F1', z_threshold=2.0)
        >>> if alert:
        ...     print(f"Extreme detected: {alert.direction}, z-score: {alert.z_score:.2f}")
        """
        if factor_name not in self.factor_returns.columns:
            raise ValueError(f"Factor '{factor_name}' not found in returns")

        idx = self._resolve_date_index(date)

        # Calculate metrics
        zscore = self.calculate_zscore(factor_name).iloc[idx]
        percentile = self.get_percentile_rank(factor_name).iloc[idx]
        current_value = self.factor_returns[factor_name].iloc[idx]

        if pd.isna(zscore) or pd.isna(percentile):
            return None

        # Check for extremes
        is_zscore_extreme = abs(zscore) > z_threshold
        is_percentile_extreme = (
            percentile > percentile_threshold or percentile < (100 - percentile_threshold)
        )

        if not is_zscore_extreme and not is_percentile_extreme:
            return None

        # Determine direction
        if zscore > 0 or percentile > 50:
            direction = 'extreme_high'
        else:
            direction = 'extreme_low'

        # Determine alert type
        if is_zscore_extreme and is_percentile_extreme:
            alert_type = 'combined_extreme'
        elif is_zscore_extreme:
            alert_type = 'zscore_extreme'
        else:
            alert_type = 'percentile_extreme'

        return ExtremeAlert(
            factor_name=factor_name,
            alert_type=alert_type,
            z_score=zscore,
            percentile=percentile,
            current_value=current_value,
            threshold=z_threshold,
            direction=direction,
            timestamp=self.factor_returns.index[idx]
        )

    def get_all_extreme_alerts(
        self,
        z_threshold: float = 2.0,
        percentile_threshold: float = 95.0,
        date: Optional[pd.Timestamp] = None
    ) -> List[ExtremeAlert]:
        """
        Get extreme alerts for all factors.

        Parameters
        ----------
        z_threshold : float, default 2.0
            Z-score threshold for extreme detection
        percentile_threshold : float, default 95.0
            Percentile threshold for extreme detection
        date : pd.Timestamp, optional
            As-of date for alert generation.

        Returns
        -------
        List[ExtremeAlert]
            List of all active extreme alerts
        """
        alerts = []
        for factor_name in self.factor_returns.columns:
            alert = self.check_extreme_levels(
                factor_name,
                date=date,
                z_threshold=z_threshold,
                percentile_threshold=percentile_threshold
            )
            if alert:
                alerts.append(alert)
        return alerts

    def get_momentum_signals(
        self,
        factor_name: str,
        date: Optional[pd.Timestamp] = None
    ) -> Dict[str, Union[str, float]]:
        """
        Get comprehensive momentum signals for a factor.

        Parameters
        ----------
        factor_name : str
            Name of the factor column to analyze
        date : pd.Timestamp, optional
            Date to analyze (default: latest available)

        Returns
        -------
        Dict
            Dictionary containing all momentum indicators and signals
        """
        if factor_name not in self.factor_returns.columns:
            raise ValueError(f"Factor '{factor_name}' not found in returns")

        idx = self._resolve_date_index(date)
        as_of_date = self.factor_returns.index[idx]

        # Calculate all indicators
        rsi = self.calculate_rsi(factor_name)
        macd_line, signal_line, histogram = self.calculate_macd(factor_name)
        adx = self.calculate_adx(factor_name)
        roc = self.calculate_roc(factor_name)

        # RSI signal
        rsi_val = rsi.iloc[idx]
        if rsi_val > 70:
            rsi_signal = 'overbought'
        elif rsi_val < 30:
            rsi_signal = 'oversold'
        else:
            rsi_signal = 'neutral'

        # MACD signal
        macd_val = macd_line.iloc[idx]
        signal_val = signal_line.iloc[idx]
        hist_val = histogram.iloc[idx]

        if idx > 0:
            macd_prev = macd_line.iloc[idx - 1]
            signal_prev = signal_line.iloc[idx - 1]

            if macd_val > signal_val and macd_prev <= signal_prev:
                macd_signal = 'bullish_crossover'
            elif macd_val < signal_val and macd_prev >= signal_prev:
                macd_signal = 'bearish_crossover'
            elif macd_val > signal_val:
                macd_signal = 'bullish'
            else:
                macd_signal = 'bearish'
        else:
            macd_signal = 'neutral'

        # ADX signal
        adx_val = adx.iloc[idx]
        if adx_val > 25:
            adx_signal = 'strong_trend'
        elif adx_val < 20:
            adx_signal = 'weak_trend'
        else:
            adx_signal = 'transition'

        # Regime
        regime = self.detect_momentum_regime(factor_name, date=as_of_date)

        return {
            'factor': factor_name,
            'date': as_of_date,
            'rsi': rsi_val,
            'rsi_signal': rsi_signal,
            'macd': macd_val,
            'macd_signal': macd_signal,
            'macd_histogram': hist_val,
            'adx': adx_val,
            'adx_signal': adx_signal,
            'regime': regime.value,
            'roc_1m': roc.iloc[idx, 0] if not roc.empty else np.nan,
            'combined_signal': self._combine_momentum_signals(
                rsi_signal, macd_signal, adx_signal, regime
            )
        }

    def _combine_momentum_signals(
        self,
        rsi_signal: str,
        macd_signal: str,
        adx_signal: str,
        regime: MomentumRegime
    ) -> str:
        """Combine individual momentum signals into overall signal."""
        # Count bullish/bearish signals
        bullish_count = 0
        bearish_count = 0

        if rsi_signal == 'oversold':
            bullish_count += 1
        elif rsi_signal == 'overbought':
            bearish_count += 1

        if 'bullish' in macd_signal:
            bullish_count += 1
        elif 'bearish' in macd_signal:
            bearish_count += 1

        if regime == MomentumRegime.TRENDING_UP:
            bullish_count += 1
        elif regime == MomentumRegime.TRENDING_DOWN:
            bearish_count += 1

        # Determine combined signal
        if bullish_count >= 2 and bearish_count == 0:
            return 'strong_buy'
        elif bullish_count > bearish_count:
            return 'buy'
        elif bearish_count >= 2 and bullish_count == 0:
            return 'strong_sell'
        elif bearish_count > bullish_count:
            return 'sell'
        else:
            return 'neutral'

    def get_all_signals(
        self,
        date: Optional[pd.Timestamp] = None
    ) -> Dict[str, Dict]:
        """
        Generate all trading signals for all factors on a given date.

        Parameters
        ----------
        date : pd.Timestamp, optional
            Date to analyze (default: latest available)

        Returns
        -------
        Dict[str, Dict]
            Dictionary mapping factor names to their signal dictionaries
        """
        all_signals = {}
        for factor_name in self.factor_returns.columns:
            all_signals[factor_name] = self.get_momentum_signals(factor_name, date)
        return all_signals

    def get_signal_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of all current signals.

        Returns
        -------
        pd.DataFrame
            Summary table with key metrics for all factors
        """
        summary_data = []

        for factor_name in self.factor_returns.columns:
            signals = self.get_momentum_signals(factor_name)
            alert = self.check_extreme_levels(factor_name)

            summary_data.append({
                'factor': factor_name,
                'rsi': signals['rsi'],
                'rsi_signal': signals['rsi_signal'],
                'macd_signal': signals['macd_signal'],
                'adx': signals['adx'],
                'regime': signals['regime'],
                'combined_signal': signals['combined_signal'],
                'extreme_alert': alert.alert_type if alert else None,
                'z_score': alert.z_score if alert else np.nan
            })

        return pd.DataFrame(summary_data)
