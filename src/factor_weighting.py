"""
Optimal Factor Weighting Methods for Cross-Sectional Portfolios
===============================================================

This module provides principled methods for determining factor weights based on
factor characteristics, risk metrics, and performance measures.

Available Methods
-----------------
1. **Sharpe Ratio Weighting** - Weight by risk-adjusted returns
2. **Information Coefficient (IC) Weighting** - Weight by predictive power
3. **Factor Momentum Weighting** - Dynamically adjust based on factor trends
4. **Risk Parity Weighting** - Equal risk contribution from each factor
5. **Principal Component Weighting** - PCA-based variance decomposition
6. **Minimum Variance Weighting** - Minimize portfolio variance
7. **Maximum Diversification** - Maximize diversification ratio
8. **Regime-Dependent Weighting** - Adaptive weights based on market regime

Mathematical Foundation
-----------------------
All methods assume factor loadings matrix B (N×K) and optional factor returns F (T×K).
The goal is to find optimal weight vector w (K×1) for composite scoring:

    score = B @ w

Where w is determined by factor characteristics rather than ad hoc assumptions.

Usage
-----
>>> from factor_weighting import OptimalFactorWeighter
>>> weighter = OptimalFactorWeighter(factor_loadings, factor_returns)

>>> # Sharpe-weighted (risk-adjusted)
>>> weights = weighter.sharpe_weights(lookback=63)

>>> # IC-weighted (predictive power)
>>> weights = weighter.ic_weights(forward_returns)

>>> # Factor momentum (trend-following)
>>> weights = weighter.momentum_weights(method='absolute')

>>> # Risk parity (equal risk contribution)
>>> weights = weighter.risk_parity_weights()

>>> # Combine multiple methods
>>> weights = weighter.blend_weights({
...     'sharpe': 0.4,
...     'momentum': 0.3,
...     'ic': 0.3
}, forward_returns)
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.optimize import minimize

from .covariance import CovarianceMethod, estimate_covariance

_LOGGER = logging.getLogger(__name__)


class WeightingMethod(Enum):
    """Enumeration of available weighting methods."""
    EQUAL = "equal"
    SHARPE = "sharpe"
    INFORMATION_COEFFICIENT = "ic"
    FACTOR_MOMENTUM = "momentum"
    RISK_PARITY = "risk_parity"
    MINIMUM_VARIANCE = "min_variance"
    MAXIMUM_DIVERSIFICATION = "max_diversification"
    PRINCIPAL_COMPONENT = "pca"
    REGIME_DEPENDENT = "regime"


@dataclass
class FactorCharacteristics:
    """Container for factor characteristics used in weighting."""
    factor_name: str
    sharpe_ratio: float
    volatility: float
    mean_return: float
    information_coefficient: Optional[float] = None
    momentum_score: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class OptimalFactorWeighter:
    """
    Optimal factor weighting engine based on factor characteristics.
    
    This class provides multiple principled methods for determining factor weights
    in cross-sectional portfolios, moving beyond ad hoc weighting to approaches
    grounded in risk metrics, performance measures, and statistical properties.
    
    Parameters
    ----------
    factor_loadings : pd.DataFrame
        Factor loadings matrix with shape (N, K) where:
        - N = number of stocks
        - K = number of factors
        - Values = factor exposures
    factor_returns : pd.DataFrame, optional
        Factor returns matrix with shape (T, K) where:
        - T = number of time periods
        - K = number of factors
        - Values = daily/periodic factor returns
        
    Attributes
    ----------
    loadings : pd.DataFrame
        Stored factor loadings
    returns : pd.DataFrame | None
        Stored factor returns
    n_stocks : int
        Number of stocks
    n_factors : int
        Number of factors
        
    Methods
    -------
    equal_weights()
        Simple equal weighting (baseline)
    sharpe_weights(lookback=63, annualize=True)
        Weight by Sharpe ratio (risk-adjusted returns)
    ic_weights(forward_returns, method='spearman')
        Weight by Information Coefficient (predictive power)
    momentum_weights(lookback=21, method='absolute')
        Weight by factor momentum (trend-following)
    risk_parity_weights(target_risk=None)
        Equal risk contribution weighting
    min_variance_weights(constraints=None)
        Minimum variance portfolio weighting
    max_diversification_weights()
        Maximum diversification ratio weighting
    pca_weights(variance_threshold=0.95)
        PCA-based variance weighting
    regime_weights(regime_detector, current_regime)
        Regime-dependent adaptive weighting
    blend_weights(method_weights, **kwargs)
        Blend multiple weighting methods
    get_factor_characteristics(lookback=63)
        Calculate comprehensive factor characteristics
        
    Mathematical Details
    --------------------
    **Sharpe Weighting:**
    w_i = Sharpe_i / Σ|Sharpe_j|
    
    Factors with higher risk-adjusted returns get higher weights.
    Negative Sharpe ratios are set to zero (no shorting factors).
    
    **IC Weighting:**
    w_i = |IC_i| / Σ|IC_j|
    
    Factors with higher predictive power (correlation with forward returns)
    get higher weights. Uses Spearman rank correlation by default.
    
    **Factor Momentum:**
    w_i = max(0, momentum_i) / Σmax(0, momentum_j)
    
    Factors with positive momentum get higher weights.
    Momentum can be absolute (recent returns) or relative (vs peers).
    
    **Risk Parity:**
    Solve: w_i × σ_i × ρ_ij × w_j = constant for all i
    
    Each factor contributes equally to portfolio risk.
    Requires iterative optimization (Newton-Raphson).
    
    **Minimum Variance:**
    min: w' Σ w
    s.t.: Σw_i = 1, w_i ≥ 0
    
    Minimizes portfolio variance using factor covariance matrix.
    
    Examples
    --------
    >>> # Initialize with factor data
    >>> weighter = OptimalFactorWeighter(loadings, returns)
    
    >>> # Method 1: Sharpe-weighted (risk-adjusted)
    >>> sharpe_weights = weighter.sharpe_weights(lookback=63)
    >>> print(f"Sharpe weights: {sharpe_weights}")
    
    >>> # Method 2: IC-weighted (requires forward returns)
    >>> ic_weights = weighter.ic_weights(forward_returns)
    
    >>> # Method 3: Factor momentum
    >>> mom_weights = weighter.momentum_weights(lookback=21)
    
    >>> # Method 4: Risk parity
    >>> rp_weights = weighter.risk_parity_weights()
    
    >>> # Blend methods for robustness
    >>> blended = weighter.blend_weights(
    ...     {'sharpe': 0.4, 'momentum': 0.3, 'ic': 0.3},
    ...     forward_returns=forward_returns
    ... )
    """
    
    def __init__(
        self,
        factor_loadings: pd.DataFrame,
        factor_returns: Optional[pd.DataFrame] = None,
        cov_method: CovarianceMethod = CovarianceMethod.LEDOIT_WOLF,
        cov_halflife: Optional[int] = None,
    ):
        """
        Initialize the OptimalFactorWeighter.

        Parameters
        ----------
        factor_loadings : pd.DataFrame
            Factor loadings matrix (N×K)
        factor_returns : pd.DataFrame, optional
            Factor returns matrix (T×K)
        cov_method : CovarianceMethod
            Covariance estimation method (default: Ledoit-Wolf shrinkage)
        cov_halflife : int, optional
            Half-life for EWMA covariance (required when method is EWMA)
        """
        self.loadings = factor_loadings.copy()
        self.returns = factor_returns.copy() if factor_returns is not None else None
        self.n_stocks = factor_loadings.shape[0]
        self.n_factors = factor_loadings.shape[1]
        self.factor_names = list(factor_loadings.columns)
        self.cov_method = cov_method
        self.cov_halflife = cov_halflife
        
    def equal_weights(self) -> Dict[str, float]:
        """
        Equal weighting baseline.
        
        Returns
        -------
        Dict[str, float]
            Equal weights for all factors
        """
        return {f: 1.0 / self.n_factors for f in self.factor_names}
    
    def sharpe_weights(
        self,
        lookback: int = 63,
        annualize: bool = True,
        risk_free_rate: float = 0.0,
        min_sharpe: float = 0.0
    ) -> Dict[str, float]:
        """
        Weight factors by their Sharpe ratio (risk-adjusted returns).
        
        This approach gives higher weights to factors that have delivered
        higher returns per unit of risk, penalizing volatile underperformers.
        
        Parameters
        ----------
        lookback : int, default 63
            Lookback period for calculating Sharpe (63 ≈ 3 months)
        annualize : bool, default True
            Whether to annualize the Sharpe ratio
        risk_free_rate : float, default 0.0
            Risk-free rate for Sharpe calculation
        min_sharpe : float, default 0.0
            Minimum Sharpe threshold (factors below get zero weight)
            
        Returns
        -------
        Dict[str, float]
            Sharpe-weighted factor weights
            
        Formula
        -------
        Sharpe_i = (mean_return_i - risk_free) / std_i
        w_i = max(0, Sharpe_i - min_sharpe) / Σmax(0, Sharpe_j - min_sharpe)
        
        Examples
        --------
        >>> weighter = OptimalFactorWeighter(loadings, returns)
        >>> weights = weighter.sharpe_weights(lookback=126)  # 6 months
        >>> 
        >>> # Require positive Sharpe only
        >>> weights = weighter.sharpe_weights(min_sharpe=0.5)
        """
        if self.returns is None:
            raise ValueError("Factor returns required for Sharpe weighting")
        
        # Calculate Sharpe ratios
        sharpes = {}
        recent_returns = self.returns.tail(lookback)
        
        for factor in self.factor_names:
            rets = recent_returns[factor]
            mean_ret = rets.mean()
            std = rets.std()
            
            if std > 0:
                sharpe = (mean_ret - risk_free_rate) / std
                if annualize:
                    sharpe *= np.sqrt(252)
            else:
                sharpe = 0.0
            
            sharpes[factor] = sharpe
        
        # Apply minimum threshold and normalize
        adjusted_sharpes = {k: max(0, v - min_sharpe) for k, v in sharpes.items()}
        total = sum(adjusted_sharpes.values())
        
        if total > 0:
            weights = {k: v / total for k, v in adjusted_sharpes.items()}
        else:
            # All factors had negative Sharpe, fall back to equal
            weights = self.equal_weights()
        
        _LOGGER.info(f"Sharpe weights: {weights}")
        return weights
    
    def ic_weights(
        self,
        forward_returns: pd.Series,
        method: Literal['spearman', 'pearson'] = 'spearman'
    ) -> Dict[str, float]:
        """
        Weight factors by their Information Coefficient (predictive power).
        
        IC measures how well factor exposures predict future returns.
        Higher |IC| = better predictive power = higher weight.
        
        Parameters
        ----------
        forward_returns : pd.Series
            Forward returns for each stock (index = tickers)
        method : str, default 'spearman'
            Correlation method: 'spearman' (rank) or 'pearson' (linear)
            
        Returns
        -------
        Dict[str, float]
            IC-weighted factor weights
            
        Formula
        -------
        IC_i = Corr(exposure_i, forward_returns)
        w_i = |IC_i| / Σ|IC_j|
        
        Notes
        -----
        - Spearman (rank) correlation is more robust to outliers
        - Pearson (linear) correlation assumes linear relationships
        - IC is calculated cross-sectionally (one period)
        - For multi-period IC, average the absolute ICs
        
        Examples
        --------
        >>> weighter = OptimalFactorWeighter(loadings, returns)
        >>> 
        >>> # Get forward returns (e.g., next month returns)
        >>> forward_rets = get_forward_returns(universe, horizon=21)
        >>> 
        >>> # IC-weighted
        >>> weights = weighter.ic_weights(forward_rets, method='spearman')
        """
        # Align data
        aligned_loadings = self.loadings.loc[
            self.loadings.index.intersection(forward_returns.index)
        ]
        aligned_fwd = forward_returns.loc[aligned_loadings.index]
        
        # Calculate ICs
        ics = {}
        for factor in self.factor_names:
            if method == 'spearman':
                ic = aligned_loadings[factor].corr(aligned_fwd, method='spearman')
            else:
                ic = aligned_loadings[factor].corr(aligned_fwd, method='pearson')
            
            ics[factor] = abs(ic) if not pd.isna(ic) else 0.0
        
        # Normalize
        total_ic = sum(ics.values())
        if total_ic > 0:
            weights = {k: v / total_ic for k, v in ics.items()}
        else:
            weights = self.equal_weights()
        
        _LOGGER.info(f"IC weights: {weights}")
        return weights
    
    def momentum_weights(
        self,
        lookback: int = 21,
        method: Literal['absolute', 'relative', 'vol_adjusted'] = 'absolute'
    ) -> Dict[str, float]:
        """
        Weight factors by their recent momentum (trend-following).
        
        This approach overweights factors that have been performing well
        recently, based on the principle of factor momentum.
        
        Parameters
        ----------
        lookback : int, default 21
            Lookback period for momentum (21 ≈ 1 month)
        method : str, default 'absolute'
            Momentum calculation method:
            - 'absolute': Simple cumulative return
            - 'relative': Return relative to other factors
            - 'vol_adjusted': Return / volatility
            
        Returns
        -------
        Dict[str, float]
            Momentum-weighted factor weights
            
        Formula
        -------
        Absolute: momentum_i = Π(1 + r_i,t) - 1
        Relative: momentum_i = absolute_momentum_i - mean(momentum)
        Vol-adjusted: momentum_i = absolute_momentum_i / σ_i
        
        w_i = max(0, momentum_i) / Σmax(0, momentum_j)
        
        Examples
        --------
        >>> weighter = OptimalFactorWeighter(loadings, returns)
        >>> 
        >>> # Absolute momentum
        >>> weights = weighter.momentum_weights(lookback=21, method='absolute')
        >>> 
        >>> # Volatility-adjusted momentum
        >>> weights = weighter.momentum_weights(lookback=63, method='vol_adjusted')
        """
        if self.returns is None:
            raise ValueError("Factor returns required for momentum weighting")
        
        recent_returns = self.returns.tail(lookback)
        momentums = {}
        
        for factor in self.factor_names:
            rets = recent_returns[factor]
            
            if method == 'absolute':
                momentum = (1 + rets).prod() - 1
            elif method == 'relative':
                abs_mom = (1 + rets).prod() - 1
                all_moms = [(1 + recent_returns[f]).prod() - 1 for f in self.factor_names]
                momentum = abs_mom - np.mean(all_moms)
            elif method == 'vol_adjusted':
                abs_mom = (1 + rets).prod() - 1
                vol = rets.std() * np.sqrt(lookback)
                momentum = abs_mom / vol if vol > 0 else 0
            else:
                raise ValueError(f"Unknown momentum method: {method}")
            
            momentums[factor] = momentum
        
        # Only positive momentum gets weight
        positive_moms = {k: max(0, v) for k, v in momentums.items()}
        total = sum(positive_moms.values())
        
        if total > 0:
            weights = {k: v / total for k, v in positive_moms.items()}
        else:
            weights = self.equal_weights()
        
        _LOGGER.info(f"Momentum weights: {weights}")
        return weights
    
    def risk_parity_weights(
        self,
        lookback: int = 63,
        target_risk: Optional[float] = None,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> Dict[str, float]:
        """
        Risk parity weighting - equal risk contribution from each factor.
        
        Each factor contributes equally to portfolio risk, preventing
        concentration in high-volatility factors.
        
        Parameters
        ----------
        lookback : int, default 63
            Lookback for covariance calculation
        target_risk : float, optional
            Target portfolio volatility (if None, not constrained)
        max_iter : int, default 100
            Maximum iterations for optimization
        tol : float, default 1e-6
            Convergence tolerance
            
        Returns
        -------
        Dict[str, float]
            Risk parity factor weights
            
        Mathematical Formulation
        ------------------------
        Risk contribution: RC_i = w_i × (Σw)_i / √(w'Σw)
        Objective: RC_i = RC_j for all i,j
        
        Solved via Newton-Raphson iteration:
        w_new = 0.5 × (w_old + (Σ^{-1} × 1) / (1' × Σ^{-1} × 1))
        
        Examples
        --------
        >>> weighter = OptimalFactorWeighter(loadings, returns)
        >>> 
        >>> # Basic risk parity
        >>> weights = weighter.risk_parity_weights()
        >>> 
        >>> # With 10% target volatility
        >>> weights = weighter.risk_parity_weights(target_risk=0.10)
        """
        if self.returns is None:
            raise ValueError("Factor returns required for risk parity")
        
        # Calculate covariance matrix
        recent_returns = self.returns.tail(lookback)
        cov = estimate_covariance(
            recent_returns[self.factor_names],
            method=self.cov_method, halflife=self.cov_halflife,
        )

        # Initialize with inverse volatility weights
        inv_vols = 1.0 / np.sqrt(np.diag(cov))
        w = inv_vols / inv_vols.sum()
        
        # Newton-Raphson iteration
        for _ in range(max_iter):
            # Portfolio volatility
            port_var = w @ cov @ w
            
            # Gradient
            grad = cov @ w
            
            # Update
            w_new = 0.5 * (w + grad / grad.sum())
            
            # Check convergence
            if np.sum((w_new - w) ** 2) < tol:
                break
            
            w = w_new
        
        # Normalize
        w = w / w.sum()
        
        weights = {f: w[i] for i, f in enumerate(self.factor_names)}
        _LOGGER.info(f"Risk parity weights: {weights}")
        return weights
    
    def min_variance_weights(
        self,
        lookback: int = 63,
        allow_short: bool = False
    ) -> Dict[str, float]:
        """
        Minimum variance portfolio weighting.
        
        Minimizes portfolio variance subject to constraints.
        Tends to overweight low-volatility factors.
        
        Parameters
        ----------
        lookback : int, default 63
            Lookback for covariance calculation
        allow_short : bool, default False
            Whether to allow short positions
            
        Returns
        -------
        Dict[str, float]
            Minimum variance factor weights
            
        Optimization Problem
        --------------------
        min: w' Σ w
        s.t.: Σw_i = 1
              w_i ≥ 0 (if not allow_short)
        """
        if self.returns is None:
            raise ValueError("Factor returns required for min variance")
        
        # Calculate covariance matrix
        recent_returns = self.returns.tail(lookback)
        cov = estimate_covariance(
            recent_returns[self.factor_names],
            method=self.cov_method, halflife=self.cov_halflife,
        )

        # Objective function
        def objective(w):
            return w @ cov @ w
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        
        # Bounds
        bounds = [(None, None) if allow_short else (0, None) for _ in self.factor_names]
        
        # Initial guess
        w0 = np.ones(self.n_factors) / self.n_factors
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            _LOGGER.warning(f"Min variance optimization failed: {result.message}")
            return self.equal_weights()
        
        w = result.x
        weights = {f: w[i] for i, f in enumerate(self.factor_names)}
        _LOGGER.info(f"Min variance weights: {weights}")
        return weights
    
    def max_diversification_weights(
        self,
        lookback: int = 63
    ) -> Dict[str, float]:
        """
        Maximum diversification weighting.
        
        Maximizes the diversification ratio, which measures how much
        risk is reduced by combining imperfectly correlated factors.
        
        Parameters
        ----------
        lookback : int, default 63
            Lookback for covariance calculation
            
        Returns
        -------
        Dict[str, float]
            Maximum diversification factor weights
            
        Diversification Ratio
        --------------------
        DR = (w'σ) / √(w'Σw)
        
        Where σ is the vector of volatilities. Higher DR = better diversification.
        """
        if self.returns is None:
            raise ValueError("Factor returns required for max diversification")
        
        # Calculate covariance and volatilities
        recent_returns = self.returns.tail(lookback)
        cov = estimate_covariance(
            recent_returns[self.factor_names],
            method=self.cov_method, halflife=self.cov_halflife,
        )
        vols = np.sqrt(np.diag(cov))
        
        # Objective: maximize DR = (w'vols) / sqrt(w'Sw)
        # Equivalent to minimizing -DR
        def neg_diversification_ratio(w):
            port_var = w @ cov @ w
            if port_var <= 0:
                return 0
            weighted_vols = w @ vols
            return -weighted_vols / np.sqrt(port_var)
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0, None) for _ in self.factor_names]
        
        # Initial guess
        w0 = np.ones(self.n_factors) / self.n_factors
        
        # Optimize
        result = minimize(
            neg_diversification_ratio,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            _LOGGER.warning(f"Max diversification optimization failed: {result.message}")
            return self.equal_weights()
        
        w = result.x
        weights = {f: w[i] for i, f in enumerate(self.factor_names)}
        _LOGGER.info(f"Max diversification weights: {weights}")
        return weights
    
    def pca_weights(
        self,
        variance_threshold: float = 0.95
    ) -> Dict[str, float]:
        """
        PCA-based factor weighting.
        
        Uses Principal Component Analysis to identify the most important
        sources of variance and weights factors by their contribution.
        
        Parameters
        ----------
        variance_threshold : float, default 0.95
            Cumulative variance threshold for selecting principal components
            
        Returns
        -------
        Dict[str, float]
            PCA-based factor weights
            
        Methodology
        -----------
        1. Run PCA on factor loadings
        2. Select PCs explaining variance_threshold of variance
        3. Weight = sum of squared loadings on selected PCs
        """
        # Standardize loadings
        loadings_std = (self.loadings - self.loadings.mean()) / self.loadings.std()
        loadings_std = loadings_std.fillna(0)
        
        # Run PCA
        pca = PCA()
        pca.fit(loadings_std)
        
        # Select components explaining threshold variance
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumvar >= variance_threshold) + 1
        
        # Calculate factor importance (sum of squared loadings)
        components = pca.components_[:n_components]
        importance = np.sum(components ** 2, axis=0)
        
        # Normalize to weights
        weights = importance / importance.sum()
        
        result = {f: weights[i] for i, f in enumerate(self.factor_names)}
        _LOGGER.info(f"PCA weights (using {n_components} components): {result}")
        return result
    
    def blend_weights(
        self,
        method_weights: Dict[str, float],
        **kwargs
    ) -> Dict[str, float]:
        """
        Blend multiple weighting methods for robustness.
        
        Combines weights from different methods using a weighted average.
        This can provide more stable weights than any single method.
        
        Parameters
        ----------
        method_weights : Dict[str, float]
            Map of method name to weight in blend (must sum to 1.0)
            Available methods: 'equal', 'sharpe', 'momentum', 'ic', 
            'risk_parity', 'min_variance', 'max_diversification', 'pca'
        **kwargs
            Arguments to pass to individual methods (e.g., forward_returns for IC)
            
        Returns
        -------
        Dict[str, float]
            Blended factor weights
            
        Examples
        --------
        >>> weighter = OptimalFactorWeighter(loadings, returns)
        >>> 
        >>> # Blend Sharpe and momentum
        >>> weights = weighter.blend_weights({
        ...     'sharpe': 0.5,
        ...     'momentum': 0.5
        ... })
        >>> 
        >>> # Complex blend with IC
        >>> weights = weighter.blend_weights(
        ...     {'sharpe': 0.3, 'ic': 0.4, 'momentum': 0.3},
        ...     forward_returns=fwd_rets
        ... )
        """
        # Normalize method weights
        total = sum(method_weights.values())
        method_weights = {k: v / total for k, v in method_weights.items()}
        
        # Collect weights from each method
        all_weights = {}
        
        for method, weight in method_weights.items():
            if weight <= 0:
                continue
                
            if method == 'equal':
                w = self.equal_weights()
            elif method == 'sharpe':
                w = self.sharpe_weights(**{k: v for k, v in kwargs.items() 
                                           if k in ['lookback', 'annualize', 'min_sharpe']})
            elif method == 'ic':
                if 'forward_returns' not in kwargs:
                    raise ValueError("forward_returns required for IC weighting")
                w = self.ic_weights(kwargs['forward_returns'])
            elif method == 'momentum':
                w = self.momentum_weights(**{k: v for k, v in kwargs.items()
                                            if k in ['lookback', 'method']})
            elif method == 'risk_parity':
                w = self.risk_parity_weights(**{k: v for k, v in kwargs.items()
                                               if k in ['lookback', 'target_risk']})
            elif method == 'min_variance':
                w = self.min_variance_weights(**{k: v for k, v in kwargs.items()
                                                if k in ['lookback', 'allow_short']})
            elif method == 'max_diversification':
                w = self.max_diversification_weights(**{k: v for k, v in kwargs.items()
                                                      if k in ['lookback']})
            elif method == 'pca':
                w = self.pca_weights(**{k: v for k, v in kwargs.items()
                                       if k in ['variance_threshold']})
            else:
                raise ValueError(f"Unknown weighting method: {method}")
            
            all_weights[method] = w
        
        # Blend
        blended = {f: 0.0 for f in self.factor_names}
        for method, w in all_weights.items():
            for f in self.factor_names:
                blended[f] += w[f] * method_weights[method]
        
        # Renormalize
        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}
        
        _LOGGER.info(f"Blended weights: {blended}")
        return blended
    
    def get_factor_characteristics(
        self,
        lookback: int = 63,
        forward_returns: Optional[pd.Series] = None
    ) -> Dict[str, FactorCharacteristics]:
        """
        Calculate comprehensive characteristics for each factor.
        
        Parameters
        ----------
        lookback : int, default 63
            Lookback period for calculations
        forward_returns : pd.Series, optional
            Forward returns for IC calculation
            
        Returns
        -------
        Dict[str, FactorCharacteristics]
            Characteristics for each factor
        """
        if self.returns is None:
            raise ValueError("Factor returns required")
        
        recent = self.returns.tail(lookback)
        characteristics = {}
        
        for factor in self.factor_names:
            rets = recent[factor]
            
            # Basic stats
            mean_ret = rets.mean()
            vol = rets.std()
            sharpe = (mean_ret / vol * np.sqrt(252)) if vol > 0 else 0
            
            # Drawdown
            cumret = (1 + rets).cumprod()
            running_max = cumret.expanding().max()
            drawdown = (cumret - running_max) / running_max
            max_dd = drawdown.min()
            
            # Win rate
            win_rate = (rets > 0).mean()
            
            # IC if forward returns provided
            ic = None
            if forward_returns is not None:
                aligned = self.loadings.loc[
                    self.loadings.index.intersection(forward_returns.index), factor
                ]
                fwd = forward_returns.loc[aligned.index]
                ic = aligned.corr(fwd, method='spearman')
            
            characteristics[factor] = FactorCharacteristics(
                factor_name=factor,
                sharpe_ratio=sharpe,
                volatility=vol,
                mean_return=mean_ret,
                information_coefficient=ic,
                max_drawdown=max_dd,
                win_rate=win_rate
            )
        
        return characteristics
