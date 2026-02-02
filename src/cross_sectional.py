"""
Cross-Sectional Factor Analysis and Ranking System
==================================================

This module provides cross-sectional analysis capabilities for ranking stocks
based on their factor exposures. It supports composite score construction,
decile-based rankings, and long/short portfolio signal generation.

Core Components
---------------

**CrossSectionalAnalyzer**
- Composite factor score calculation with custom weighting
- Decile-based universe ranking
- Long/short candidate identification
- Factor exposure reporting and attribution

Ranking Methodologies
---------------------

**Composite Scoring:**
- Weighted sum of factor exposures
- Support for custom factor weighting schemes
- Z-score normalization for comparability

**Decile Rankings:**
- Universe segmentation into 10 equal buckets
- Top decile: Long candidates
- Bottom decile: Short candidates
- Middle deciles: Neutral/hold

**Style Rotation:**
- Value vs Growth indicators
- Large vs Small cap signals
- Sector rotation detection

Architecture
------------
```
Factor Loadings → Composite Scoring → Decile Ranking → Long/Short Signals
       ↓                ↓                  ↓                  ↓
   [N x K]         [Weighted Sum]    [Percentiles]    [Top/Bottom 10%]
```

Dependencies
------------
- pandas, numpy: Data manipulation and calculations
- scikit-learn: Standardization and preprocessing

Examples
--------
>>> from cross_sectional import CrossSectionalAnalyzer
>>> analyzer = CrossSectionalAnalyzer(factor_loadings)
>>> scores = analyzer.calculate_factor_scores(weights={'F1': 0.5, 'F2': 0.5})
>>> rankings = analyzer.rank_universe(scores)
>>> signals = analyzer.generate_long_short_signals(top_pct=0.1, bottom_pct=0.1)
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.WARNING)

if not _LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
    handler.setFormatter(formatter)
    _LOGGER.addHandler(handler)
    _LOGGER.propagate = False


class SignalDirection(Enum):
    """Enumeration of signal directions."""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class StyleFactor(Enum):
    """Enumeration of common style factors."""
    VALUE = "value"
    GROWTH = "growth"
    MOMENTUM = "momentum"
    QUALITY = "quality"
    SIZE = "size"
    VOLATILITY = "volatility"


@dataclass
class StockSignal:
    """Data class representing a stock-level trading signal."""
    ticker: str
    direction: SignalDirection
    composite_score: float
    decile: int
    rank: int
    total_stocks: int
    factor_breakdown: Dict[str, float]
    confidence: float


@dataclass
class StyleRotation:
    """Data class representing style rotation indicators."""
    value_vs_growth: float  # Positive = value favored, negative = growth
    large_vs_small: float   # Positive = large cap favored
    momentum_strength: float
    quality_bias: float
    timestamp: pd.Timestamp


class CrossSectionalAnalyzer:
    """
    Cross-sectional factor analysis and ranking engine.

    This class provides comprehensive cross-sectional analysis capabilities
    for ranking stocks based on their factor exposures. It supports
    composite score construction, decile rankings, and long/short signal
    generation.

    Parameters
    ----------
    factor_loadings : pd.DataFrame
        Factor exposure/loadings DataFrame with shape (N, K) where:
        - Index: Stock tickers
        - Columns: Factor identifiers (e.g., 'F1', 'F2', ...)
        - Values: Factor exposures (can be positive or negative)

    Attributes
    ----------
    factor_loadings : pd.DataFrame
        Stored factor loadings for analysis
    n_stocks : int
        Number of stocks in the universe
    n_factors : int
        Number of factors
    scaler : StandardScaler
        Scaler for normalization

    Methods
    -------
    calculate_factor_scores(weights=None, method='weighted_sum')
        Calculate composite factor scores per stock
    rank_universe(scores, n_buckets=10)
        Generate decile rankings from scores
    generate_long_short_signals(top_pct=0.1, bottom_pct=0.1, scores=None)
        Generate long/short candidates
    get_factor_exposure_report(ticker)
        Get detailed factor breakdown for a stock
    detect_style_rotation()
        Detect style rotation trends
    get_sector_exposure(sector_mapping)
        Calculate sector factor exposures
    optimize_factor_weights(target_returns, method='mean_variance')
        Optimize factor weights for target returns

    Scoring Methodologies
    ---------------------

    **Weighted Sum (default):**
    score_i = Σ(w_k × exposure_ik)
    where w_k is the weight for factor k

    **Z-Score Standardization:**
    All exposures are z-scored before weighting to ensure:
    - Equal contribution from each factor
    - Normalized scale across factors
    - Robustness to factor volatility differences

    Ranking Interpretation
    ----------------------
    - Decile 1 (top 10%): Highest scores, long candidates
    - Decile 2-5: Above average, potential longs
    - Decile 6-9: Below average, potential shorts
    - Decile 10 (bottom 10%): Lowest scores, short candidates

    Examples
    --------
    >>> # Initialize analyzer
    >>> analyzer = CrossSectionalAnalyzer(factor_loadings)
    >>>
    >>> # Calculate equal-weighted composite scores
    >>> scores = analyzer.calculate_factor_scores()
    >>>
    >>> # Custom factor weighting
    >>> weights = {'F1': 0.4, 'F2': 0.3, 'F3': 0.3}
    >>> scores = analyzer.calculate_factor_scores(weights=weights)
    >>>
    >>> # Generate rankings
    >>> rankings = analyzer.rank_universe(scores)
    >>> top_stocks = rankings[rankings['decile'] == 1]
    >>>
    >>> # Get long/short signals
    >>> signals = analyzer.generate_long_short_signals(top_pct=0.1, bottom_pct=0.1)
    >>> longs = [s for s in signals if s.direction == SignalDirection.LONG]
    >>> shorts = [s for s in signals if s.direction == SignalDirection.SHORT]
    >>>
    >>> # Factor exposure report
    >>> report = analyzer.get_factor_exposure_report('AAPL')
    """

    def __init__(self, factor_loadings: pd.DataFrame):
        """
        Initialize the CrossSectionalAnalyzer.

        Parameters
        ----------
        factor_loadings : pd.DataFrame
            Factor exposures with stock tickers as index
        """
        self.factor_loadings = factor_loadings.copy()
        self.n_stocks = factor_loadings.shape[0]
        self.n_factors = factor_loadings.shape[1]
        self.scaler = StandardScaler()
        self._validate_data()

    def _validate_data(self) -> None:
        """Validate input factor loadings data."""
        if self.factor_loadings.empty:
            raise ValueError("Factor loadings DataFrame is empty")

        if self.n_stocks < 20:
            _LOGGER.warning(
                f"Only {self.n_stocks} stocks in universe, "
                "minimum recommended is 20 for reliable rankings"
            )

        # Check for NaN values
        nan_pct = self.factor_loadings.isna().mean().mean()
        if nan_pct > 0.1:
            _LOGGER.warning(
                f"Factor loadings have {nan_pct:.1%} NaN values, "
                "consider imputation"
            )

    def calculate_factor_scores(
        self,
        weights: Optional[Dict[str, float]] = None,
        method: str = 'weighted_sum',
        standardize: bool = True
    ) -> pd.Series:
        """
        Calculate composite factor scores for each stock.

        Parameters
        ----------
        weights : Dict[str, float], optional
            Factor weights. If None, uses equal weights.
            Keys should match factor column names.
        method : str, default 'weighted_sum'
            Scoring method: 'weighted_sum', 'product', or 'rank'
        standardize : bool, default True
            Whether to z-score standardize factor exposures before scoring

        Returns
        -------
        pd.Series
            Composite scores indexed by stock ticker

        Scoring Methods
        ---------------
        **weighted_sum:** Linear combination of factor exposures
        score = Σ(weight_k × exposure_k)

        **product:** Multiplicative combination (for orthogonal factors)
        score = Π(exposure_k ^ weight_k)

        **rank:** Rank-based scoring (robust to outliers)
        score = Σ(weight_k × rank(exposure_k))

        Examples
        --------
        >>> # Equal-weighted scores
        >>> scores = analyzer.calculate_factor_scores()
        >>>
        >>> # Custom weights
        >>> weights = {'Value': 0.5, 'Momentum': 0.3, 'Quality': 0.2}
        >>> scores = analyzer.calculate_factor_scores(weights=weights)
        >>>
        >>> # Rank-based scoring (more robust)
        >>> scores = analyzer.calculate_factor_scores(method='rank')
        """
        loadings = self.factor_loadings.copy()

        # Handle NaN values
        loadings = loadings.fillna(0)

        # Standardize if requested
        if standardize:
            loadings = pd.DataFrame(
                self.scaler.fit_transform(loadings),
                index=loadings.index,
                columns=loadings.columns
            )

        # Set default weights if not provided
        if weights is None:
            weights = {col: 1.0 / self.n_factors for col in loadings.columns}

        # Validate weights
        for factor in weights.keys():
            if factor not in loadings.columns:
                raise ValueError(f"Weight factor '{factor}' not found in loadings")

        # Calculate scores based on method
        if method == 'weighted_sum':
            scores = pd.Series(0.0, index=loadings.index)
            for factor, weight in weights.items():
                scores += weight * loadings[factor]

        elif method == 'product':
            scores = pd.Series(1.0, index=loadings.index)
            for factor, weight in weights.items():
                scores *= (loadings[factor] ** weight)

        elif method == 'rank':
            ranks = loadings.rank(pct=True)
            scores = pd.Series(0.0, index=loadings.index)
            for factor, weight in weights.items():
                scores += weight * ranks[factor]

        else:
            raise ValueError(f"Unknown scoring method: {method}")

        return scores

    def rank_universe(
        self,
        scores: Optional[pd.Series] = None,
        n_buckets: int = 10
    ) -> pd.DataFrame:
        """
        Generate decile rankings from composite scores.

        Parameters
        ----------
        scores : pd.Series, optional
            Composite scores. If None, calculates equal-weighted scores.
        n_buckets : int, default 10
            Number of ranking buckets (10 = deciles, 5 = quintiles, etc.)

        Returns
        -------
        pd.DataFrame
            Ranking DataFrame with columns:
            - 'score': Composite score
            - 'rank': Absolute rank (1 = highest)
            - 'decile': Bucket assignment (1 = top bucket)
            - 'percentile': Percentile rank (0-100)

        Examples
        --------
        >>> rankings = analyzer.rank_universe(scores, n_buckets=10)
        >>> top_decile = rankings[rankings['decile'] == 1]
        >>> bottom_decile = rankings[rankings['decile'] == 10]
        """
        if scores is None:
            scores = self.calculate_factor_scores()

        # Create ranking DataFrame
        rankings = pd.DataFrame({
            'score': scores
        })

        # Calculate ranks (1 = highest score)
        rankings['rank'] = scores.rank(ascending=False, method='min').astype(int)

        # Calculate deciles (1 = top bucket with highest scores)
        rankings['decile'] = pd.qcut(
            scores,
            q=n_buckets,
            labels=range(n_buckets, 0, -1),  # Reverse: highest scores get decile 1
            duplicates='drop'
        ).astype(int)

        # Calculate percentile (0-100)
        rankings['percentile'] = scores.rank(pct=True) * 100

        return rankings.sort_values('rank')

    def generate_long_short_signals(
        self,
        top_pct: float = 0.1,
        bottom_pct: float = 0.1,
        scores: Optional[pd.Series] = None,
        min_confidence: float = 0.0
    ) -> List[StockSignal]:
        """
        Generate long/short trading signals based on factor rankings.

        Parameters
        ----------
        top_pct : float, default 0.1
            Percentage of universe for long signals (0.1 = top 10%)
        bottom_pct : float, default 0.1
            Percentage of universe for short signals (0.1 = bottom 10%)
        scores : pd.Series, optional
            Composite scores. If None, calculates equal-weighted scores.
        min_confidence : float, default 0.0
            Minimum confidence threshold for signals

        Returns
        -------
        List[StockSignal]
            List of stock signals with direction and metadata

        Examples
        --------
        >>> signals = analyzer.generate_long_short_signals(top_pct=0.1, bottom_pct=0.1)
        >>> longs = [s for s in signals if s.direction == SignalDirection.LONG]
        >>> shorts = [s for s in signals if s.direction == SignalDirection.SHORT]
        >>> print(f"Long {len(longs)}, Short {len(shorts)}")
        """
        if scores is None:
            scores = self.calculate_factor_scores()

        rankings = self.rank_universe(scores)

        signals = []
        n_stocks = len(rankings)

        # Calculate thresholds
        long_threshold = int(n_stocks * top_pct)
        short_threshold = int(n_stocks * (1 - bottom_pct))

        for ticker in rankings.index:
            row = rankings.loc[ticker]
            rank = row['rank']
            decile = row['decile']
            score = row['score']

            # Determine direction
            if rank <= long_threshold:
                direction = SignalDirection.LONG
                confidence = 1.0 - (rank / long_threshold) * 0.5  # Higher rank = higher confidence
            elif rank > short_threshold:
                direction = SignalDirection.SHORT
                confidence = 0.5 + ((rank - short_threshold) / (n_stocks - short_threshold)) * 0.5
            else:
                direction = SignalDirection.NEUTRAL
                confidence = 0.0

            # Skip neutral signals and those below confidence threshold
            if direction == SignalDirection.NEUTRAL or confidence < min_confidence:
                continue

            # Get factor breakdown
            factor_breakdown = self.factor_loadings.loc[ticker].to_dict()

            signals.append(StockSignal(
                ticker=ticker,
                direction=direction,
                composite_score=score,
                decile=decile,
                rank=rank,
                total_stocks=n_stocks,
                factor_breakdown=factor_breakdown,
                confidence=confidence
            ))

        return signals

    def get_factor_exposure_report(
        self,
        ticker: str,
        standardized: bool = True
    ) -> Dict:
        """
        Generate detailed factor exposure report for a stock.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        standardized : bool, default True
            Whether to show z-scored exposures

        Returns
        -------
        Dict
            Factor exposure report containing:
            - 'ticker': Stock symbol
            - 'raw_exposures': Raw factor loadings
            - 'standardized_exposures': Z-scored exposures
            - 'percentile_ranks': Percentile within universe
            - 'dominant_factor': Factor with highest absolute exposure
            - 'style_classification': Style-based classification

        Examples
        --------
        >>> report = analyzer.get_factor_exposure_report('AAPL')
        >>> print(f"Dominant factor: {report['dominant_factor']}")
        >>> print(f"Style: {report['style_classification']}")
        """
        if ticker not in self.factor_loadings.index:
            raise ValueError(f"Ticker '{ticker}' not found in factor loadings")

        # Get raw exposures
        raw_exposures = self.factor_loadings.loc[ticker].to_dict()

        # Calculate standardized exposures
        standardized_exposures = {}
        for factor in self.factor_loadings.columns:
            mean = self.factor_loadings[factor].mean()
            std = self.factor_loadings[factor].std()
            if std > 0:
                standardized_exposures[factor] = (raw_exposures[factor] - mean) / std
            else:
                standardized_exposures[factor] = 0.0

        # Calculate percentile ranks
        percentile_ranks = {}
        for factor in self.factor_loadings.columns:
            values = self.factor_loadings[factor]
            percentile_ranks[factor] = 100 * (values < raw_exposures[factor]).mean()

        # Find dominant factor
        abs_exposures = {k: abs(v) for k, v in standardized_exposures.items()}
        dominant_factor = max(abs_exposures, key=abs_exposures.get)

        # Style classification
        style_classification = self._classify_style(standardized_exposures)

        return {
            'ticker': ticker,
            'raw_exposures': raw_exposures,
            'standardized_exposures': standardized_exposures,
            'percentile_ranks': percentile_ranks,
            'dominant_factor': dominant_factor,
            'style_classification': style_classification
        }

    def _classify_style(self, exposures: Dict[str, float]) -> Dict[str, str]:
        """Classify stock style based on factor exposures."""
        classification = {}

        # Value vs Growth classification
        value_factors = [f for f in exposures.keys() if 'value' in f.lower()]
        growth_factors = [f for f in exposures.keys() if 'growth' in f.lower()]

        if value_factors and growth_factors:
            value_score = np.mean([exposures[f] for f in value_factors])
            growth_score = np.mean([exposures[f] for f in growth_factors])

            if value_score > growth_score:
                classification['value_growth'] = 'value'
            else:
                classification['value_growth'] = 'growth'
        elif value_factors:
            classification['value_growth'] = 'value' if np.mean([exposures[f] for f in value_factors]) > 0 else 'neutral'
        elif growth_factors:
            classification['value_growth'] = 'growth' if np.mean([exposures[f] for f in growth_factors]) > 0 else 'neutral'
        else:
            classification['value_growth'] = 'neutral'

        # Size classification
        size_factors = [f for f in exposures.keys() if 'size' in f.lower() or 'cap' in f.lower()]
        if size_factors:
            size_score = np.mean([exposures[f] for f in size_factors])
            classification['size'] = 'large' if size_score > 0 else 'small'
        else:
            classification['size'] = 'neutral'

        # Momentum classification
        momentum_factors = [f for f in exposures.keys() if 'momentum' in f.lower() or 'mom' in f.lower()]
        if momentum_factors:
            mom_score = np.mean([exposures[f] for f in momentum_factors])
            classification['momentum'] = 'high' if mom_score > 0 else 'low'
        else:
            classification['momentum'] = 'neutral'

        # Quality classification
        quality_factors = [f for f in exposures.keys() if 'quality' in f.lower() or 'profit' in f.lower()]
        if quality_factors:
            quality_score = np.mean([exposures[f] for f in quality_factors])
            classification['quality'] = 'high' if quality_score > 0 else 'low'
        else:
            classification['quality'] = 'neutral'

        return classification

    def detect_style_rotation(
        self,
        lookback: int = 63
    ) -> StyleRotation:
        """
        Detect style rotation trends from factor performance.

        Parameters
        ----------
        lookback : int, default 63
            Number of periods to analyze (63 ≈ 3 months)

        Returns
        -------
        StyleRotation
            Style rotation indicators

        Examples
        --------
        >>> rotation = analyzer.detect_style_rotation()
        >>> if rotation.value_vs_growth > 0.5:
        ...     print("Value factors outperforming")
        """
        # This is a placeholder implementation
        # In practice, you would need factor returns over time

        # Calculate value vs growth spread
        value_factors = [f for f in self.factor_loadings.columns if 'value' in f.lower()]
        growth_factors = [f for f in self.factor_loadings.columns if 'growth' in f.lower()]

        if value_factors and growth_factors:
            value_mean = self.factor_loadings[value_factors].mean().mean()
            growth_mean = self.factor_loadings[growth_factors].mean().mean()
            value_vs_growth = value_mean - growth_mean
        else:
            value_vs_growth = 0.0

        # Size spread
        size_factors = [f for f in self.factor_loadings.columns if 'size' in f.lower()]
        if size_factors:
            large_vs_small = self.factor_loadings[size_factors].mean().mean()
        else:
            large_vs_small = 0.0

        # Momentum strength
        momentum_factors = [f for f in self.factor_loadings.columns if 'momentum' in f.lower()]
        if momentum_factors:
            momentum_strength = self.factor_loadings[momentum_factors].mean().mean()
        else:
            momentum_strength = 0.0

        # Quality bias
        quality_factors = [f for f in self.factor_loadings.columns if 'quality' in f.lower()]
        if quality_factors:
            quality_bias = self.factor_loadings[quality_factors].mean().mean()
        else:
            quality_bias = 0.0

        return StyleRotation(
            value_vs_growth=value_vs_growth,
            large_vs_small=large_vs_small,
            momentum_strength=momentum_strength,
            quality_bias=quality_bias,
            timestamp=pd.Timestamp.now()
        )

    def get_sector_exposure(
        self,
        sector_mapping: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Calculate average factor exposures by sector.

        Parameters
        ----------
        sector_mapping : Dict[str, str]
            Mapping of tickers to sector names

        Returns
        -------
        pd.DataFrame
            Sector factor exposure matrix

        Examples
        --------
        >>> sectors = {'AAPL': 'Technology', 'JPM': 'Financials', ...}
        >>> sector_exposure = analyzer.get_sector_exposure(sectors)
        """
        # Add sector column
        df = self.factor_loadings.copy()
        df['sector'] = df.index.map(sector_mapping)

        # Group by sector and calculate mean
        sector_exposure = df.groupby('sector').mean()

        return sector_exposure

    def optimize_factor_weights(
        self,
        target_returns: pd.Series,
        method: str = 'mean_variance',
        risk_aversion: float = 1.0
    ) -> Dict[str, float]:
        """
        Optimize factor weights to maximize risk-adjusted returns.

        Parameters
        ----------
        target_returns : pd.Series
            Target returns to optimize against (e.g., future returns)
        method : str, default 'mean_variance'
            Optimization method: 'mean_variance', 'max_sharpe', or 'max_ic'
        risk_aversion : float, default 1.0
            Risk aversion parameter for mean-variance optimization

        Returns
        -------
        Dict[str, float]
            Optimized factor weights

        Examples
        --------
        >>> weights = analyzer.optimize_factor_weights(
        ...     target_returns=future_returns,
        ...     method='mean_variance',
        ...     risk_aversion=2.0
        ... )
        """
        # Align data
        aligned_loadings = self.factor_loadings.loc[
            self.factor_loadings.index.intersection(target_returns.index)
        ]
        aligned_returns = target_returns.loc[aligned_loadings.index]

        if method == 'max_ic':
            # Maximize information coefficient
            ics = {}
            for factor in aligned_loadings.columns:
                ic = aligned_loadings[factor].corr(aligned_returns)
                ics[factor] = abs(ic) if not pd.isna(ic) else 0

            # Normalize to sum to 1
            total_ic = sum(ics.values())
            if total_ic > 0:
                weights = {k: v / total_ic for k, v in ics.items()}
            else:
                weights = {k: 1.0 / len(ics) for k in ics}

        elif method == 'mean_variance':
            # Simplified mean-variance optimization
            # In practice, use cvxpy for proper quadratic programming
            expected_returns = {}
            for factor in aligned_loadings.columns:
                expected_returns[factor] = aligned_loadings[factor].corr(aligned_returns)

            # Simple weighting by expected return
            total = sum(abs(v) for v in expected_returns.values())
            if total > 0:
                weights = {k: max(0, v) / total for k, v in expected_returns.items()}
            else:
                weights = {k: 1.0 / len(expected_returns) for k in expected_returns}

        else:
            raise ValueError(f"Unknown optimization method: {method}")

        return weights

    def get_portfolio_construction_weights(
        self,
        scores: Optional[pd.Series] = None,
        long_pct: float = 0.1,
        short_pct: float = 0.1,
        neutral_weight: float = 0.0
    ) -> pd.Series:
        """
        Generate portfolio construction weights from factor scores.

        Parameters
        ----------
        scores : pd.Series, optional
            Composite scores. If None, calculates equal-weighted scores.
        long_pct : float, default 0.1
            Percentage of universe to go long
        short_pct : float, default 0.1
            Percentage of universe to go short
        neutral_weight : float, default 0.0
            Weight for neutral positions

        Returns
        -------
        pd.Series
            Portfolio weights (sum to 0 for market neutral)

        Examples
        --------
        >>> weights = analyzer.get_portfolio_construction_weights(
        ...     long_pct=0.1, short_pct=0.1
        ... )
        >>> long_exposure = weights[weights > 0].sum()
        >>> short_exposure = weights[weights < 0].sum()
        """
        if scores is None:
            scores = self.calculate_factor_scores()

        rankings = self.rank_universe(scores)
        n_stocks = len(rankings)

        weights = pd.Series(neutral_weight, index=rankings.index)

        # Long positions (top decile)
        long_threshold = int(n_stocks * long_pct)
        long_stocks = rankings[rankings['rank'] <= long_threshold].index
        weights.loc[long_stocks] = 1.0 / len(long_stocks) if len(long_stocks) > 0 else 0

        # Short positions (bottom decile)
        short_threshold = int(n_stocks * (1 - short_pct))
        short_stocks = rankings[rankings['rank'] > short_threshold].index
        weights.loc[short_stocks] = -1.0 / len(short_stocks) if len(short_stocks) > 0 else 0

        return weights
