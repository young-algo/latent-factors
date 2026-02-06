"""
Latent Factor Discovery: Statistical and Deep Learning Methods
============================================================

This module implements multiple approaches for discovering latent factors from
stock return data, supporting both classical statistical methods and modern
deep learning techniques for quantitative finance research.

Factor Discovery Methods
-----------------------

**Statistical Methods (Fast, Interpretable)**
- **PCA (Principal Component Analysis)**: Orthogonal linear factors maximizing variance
- **ICA (Independent Component Analysis)**: Statistically independent components
- **NMF (Non-negative Matrix Factorization)**: Parts-based decomposition for positive factors

**Deep Learning Methods (Non-linear, Expressive)**  
- **Autoencoder**: Neural network for non-linear latent factor discovery
- **Rolling Window Support**: Time-varying factor analysis

Mathematical Foundation
----------------------
All methods decompose the return matrix R (T×N) into:
- **Factor Returns**: F (T×K) - time series of factor performance  
- **Factor Loadings**: B (N×K) - asset exposures to each factor

Where: R ≈ F @ B.T (matrix approximation)

Residualization (Beta Removal)
------------------------------
**CRITICAL**: All factor discovery methods now automatically residualize returns
against market (SPY) and sector ETFs BEFORE factor extraction. This prevents
the "Beta in Disguise" problem where PCA/ICA rediscovers market beta and sector
exposures instead of true alpha factors.

The residualization process:
1. Estimate time-varying stock beta against SPY (market beta removal)
2. Estimate time-varying sector betas on residuals (sector exposure removal)
3. Run factor discovery on pure idiosyncratic returns

Factor Orthogonality
--------------------
Non-orthogonal methods (ICA, autoencoder) automatically apply symmetric
orthogonalization post-discovery to ensure zero cross-correlation between
factors. This prevents optimizers from double-counting the same signal.

Performance Characteristics
--------------------------
- **PCA/ICA**: O(N²T) complexity, seconds for 100+ assets
- **NMF**: O(N²K*iter) complexity, slower for large universes  
- **Autoencoder**: O(epochs*batch_size) complexity, GPU recommended
- **Memory**: O(NT + NK) for data storage and factor matrices

Dependencies
-----------
- **Required**: scikit-learn, numpy, pandas
- **Optional**: PyTorch (for autoencoder methods)
- **Calls**: AlphaVantage system (for benchmark data fetching)
- **Called by**: discover_and_label.py, research.py

Factor Validation
----------------
Includes comprehensive validation framework to ensure:
- Statistical distinctiveness (correlation analysis)
- Realistic return characteristics  
- Meaningful factor loadings distribution
- Performance monitoring and quality checks

Examples
--------
>>> returns = get_stock_returns()  # T×N DataFrame
>>> 
>>> # PCA - orthogonal by construction, auto-residualized
>>> factors, loadings = statistical_factors(returns, n_components=10, method=StatMethod.PCA)
>>> 
>>> # ICA - auto-residualized AND auto-orthogonalized
>>> factors, loadings = statistical_factors(returns, n_components=10, method=StatMethod.ICA)
>>> 
>>> # Skip residualization (NOT recommended - you get Beta in Disguise)
>>> factors, loadings = statistical_factors(returns, n_components=10, skip_residualization=True)
"""

from __future__ import annotations
import logging, math
from enum import Enum, auto
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FastICA, NMF, SparsePCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler

# Lazy imports for data fetching - only loaded when needed
def _get_data_backend():
    """
    Lazy import to avoid circular dependencies.

    Returns a `DataBackend` instance only when `ALPHAVANTAGE_API_KEY` is
    configured. Otherwise returns None and benchmark residualization will
    be skipped.

    Note
    ----
    Callers should prefer passing an existing backend via the `cache_backend`
    parameter to avoid repeated initialization and duplicate API calls.
    """
    try:
        from .config import config
    except ImportError:
        try:
            from src.config import config  # type: ignore
        except ImportError:
            return None

    api_key = getattr(config, "ALPHAVANTAGE_API_KEY", None)
    if not api_key:
        return None

    try:
        from .alphavantage_system import DataBackend
    except ImportError:
        try:
            from src.alphavantage_system import DataBackend  # type: ignore
        except ImportError:
            return None

    try:
        return DataBackend(api_key)
    except Exception as e:
        _LOG.warning("Failed to initialize DataBackend for benchmark returns: %s", e)
        return None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

_LOG = logging.getLogger(__name__)

# Standard sector ETFs for residualization
# Uses Select Sector SPDRs + key industry ETFs
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLI", "XLP", "XLY", "XLB", "XLU", "XLRE", "XLC"]
MARKET_ETF = "SPY"


# --------------------------------------------------------------------- #
# ‑‑‑ Residualization (Beta Removal) ‑‑‑
# --------------------------------------------------------------------- #
def _fetch_benchmark_returns(
    dates: pd.DatetimeIndex,
    cache_backend=None
) -> Tuple[Optional[pd.Series], Optional[pd.DataFrame]]:
    """
    Fetch market and sector ETF returns for residualization.
    
    Automatically fetches SPY and sector ETF data aligned to the input dates.
    Uses the AlphaVantage cache system if available, otherwise returns None.
    
    Parameters
    ----------
    dates : pd.DatetimeIndex
        Dates to align benchmark returns to
    cache_backend : DataBackend-compatible, optional
        Pre-initialized backend instance (for efficiency in batch calls)
        
    Returns
    -------
    Tuple[Optional[pd.Series], Optional[pd.DataFrame]]
        - Market returns (SPY) or None if unavailable
        - Sector returns DataFrame or None if unavailable
    """
    backend = cache_backend or _get_data_backend()
    
    if backend is None:
        _LOG.warning("AlphaVantage backend unavailable. Cannot fetch benchmark data.")
        return None, None
    
    try:
        # Fetch all benchmarks in one call
        all_etfs = [MARKET_ETF] + SECTOR_ETFS
        end_date = dates.max().strftime('%Y-%m-%d')
        
        _LOG.debug("Fetching benchmark data for %d ETFs", len(all_etfs))
        prices = backend.get_prices(all_etfs, end=end_date)
        
        if prices.empty:
            _LOG.warning("No benchmark data retrieved")
            return None, None
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Align to input dates (intersection)
        common_dates = returns.index.intersection(dates)
        if len(common_dates) < 30:
            _LOG.warning(
                "Insufficient overlap between stock data (%d days) and benchmarks (%d days)",
                len(dates), len(returns)
            )
            return None, None
        
        returns = returns.loc[common_dates]
        
        # Extract market and sector returns
        if MARKET_ETF not in returns.columns:
            _LOG.warning("SPY data not available in benchmark fetch")
            return None, None
            
        market_returns = returns[MARKET_ETF]
        
        # Get available sector ETFs
        available_sectors = [s for s in SECTOR_ETFS if s in returns.columns]
        if not available_sectors:
            _LOG.warning("No sector ETF data available")
            sector_returns = None
        else:
            sector_returns = returns[available_sectors]
            _LOG.info(
                "Using %d/%d sector ETFs for residualization: %s",
                len(available_sectors), len(SECTOR_ETFS), available_sectors
            )
        
        return market_returns, sector_returns
        
    except Exception as e:
        _LOG.warning("Failed to fetch benchmark returns: %s", e)
        return None, None


def _time_varying_residuals(
    y: pd.Series,
    X: pd.DataFrame,
    method: str,
    window: int,
    min_observations: int,
    ewm_halflife: float,
    kalman_process_variance: float,
    kalman_observation_variance: Optional[float],
    kalman_initial_covariance: float
) -> pd.Series:
    """
    Estimate residuals with time-varying betas using only past information.

    For timestamp t, coefficients are fit on [t-window, t) and applied to x_t.
    This avoids look-ahead bias from full-sample regressions.
    """
    if method not in {"rolling", "ewm", "kalman"}:
        raise ValueError(f"Unknown residualization method: {method}")
    if method in {"rolling", "ewm"} and window < min_observations:
        raise ValueError(
            f"window ({window}) must be >= min_observations ({min_observations})"
        )
    if method == "ewm" and ewm_halflife <= 0:
        raise ValueError("ewm_halflife must be > 0 for ewm residualization")
    if method == "kalman" and kalman_process_variance <= 0:
        raise ValueError(
            "kalman_process_variance must be > 0 for kalman residualization"
        )
    if (
        method == "kalman"
        and kalman_observation_variance is not None
        and kalman_observation_variance <= 0
    ):
        raise ValueError("kalman_observation_variance must be > 0 when provided")
    if method == "kalman" and kalman_initial_covariance <= 0:
        raise ValueError(
            "kalman_initial_covariance must be > 0 for kalman residualization"
        )

    aligned = pd.concat([y.rename("y"), X], axis=1).dropna()
    residuals = pd.Series(np.nan, index=y.index, dtype=float)
    if aligned.empty:
        return residuals

    y_vals = aligned.iloc[:, 0].to_numpy(dtype=float)
    X_vals = aligned.iloc[:, 1:].to_numpy(dtype=float)
    idx = aligned.index

    if method == "kalman":
        n_obs = len(aligned)
        n_features = X_vals.shape[1] + 1  # intercept + exposures
        kalman_residuals = np.full(n_obs, np.nan, dtype=float)
        if n_obs < min_observations:
            return residuals

        # Initialize state from a pre-sample OLS fit.
        X_init = np.column_stack(
            [np.ones(min_observations, dtype=float), X_vals[:min_observations]]
        )
        y_init = y_vals[:min_observations]
        theta = np.linalg.lstsq(X_init, y_init, rcond=None)[0]

        if kalman_observation_variance is None:
            init_residuals = y_init - (X_init @ theta)
            obs_var = float(np.var(init_residuals))
            obs_var = max(obs_var, 1e-8)
        else:
            obs_var = float(kalman_observation_variance)

        # Random-walk state covariance with small process drift.
        q_scale = kalman_process_variance * obs_var
        Q = np.eye(n_features, dtype=float) * q_scale
        P = np.eye(n_features, dtype=float) * (kalman_initial_covariance * obs_var)
        R = obs_var

        for t in range(min_observations, n_obs):
            # Predict with information up to t-1 only.
            theta_pred = theta
            P_pred = P + Q
            h_t = np.concatenate(([1.0], X_vals[t]))

            y_hat = float(h_t @ theta_pred)
            innovation = y_vals[t] - y_hat
            kalman_residuals[t] = innovation

            s_t = float(h_t @ P_pred @ h_t + R)
            s_t = max(s_t, 1e-10)
            k_t = (P_pred @ h_t) / s_t

            theta = theta_pred + k_t * innovation
            P = P_pred - np.outer(k_t, h_t) @ P_pred
            P = 0.5 * (P + P.T)

        residuals.loc[idx] = kalman_residuals
        return residuals

    for t in range(len(aligned)):
        if t < min_observations:
            continue

        start = max(0, t - window)
        y_hist = y_vals[start:t]
        X_hist = X_vals[start:t]
        if len(y_hist) < min_observations:
            continue

        design = np.column_stack([np.ones(len(y_hist), dtype=float), X_hist])
        if method == "ewm":
            # Recent observations receive higher weight (WLS via row scaling).
            ages = np.arange(len(y_hist) - 1, -1, -1, dtype=float)
            weights = np.power(0.5, ages / ewm_halflife)
            sqrt_w = np.sqrt(weights)
            design = design * sqrt_w[:, None]
            y_fit = y_hist * sqrt_w
        else:
            y_fit = y_hist

        coeffs = np.linalg.lstsq(design, y_fit, rcond=None)[0]
        x_t = np.concatenate(([1.0], X_vals[t]))
        y_hat = float(np.dot(x_t, coeffs))
        residuals.loc[idx[t]] = y_vals[t] - y_hat

    return residuals


def _residual_r2(y: pd.Series, residuals: pd.Series) -> float:
    """Compute R² implied by residual series on overlapping non-NaN timestamps."""
    aligned = pd.concat([y.rename("y"), residuals.rename("resid")], axis=1).dropna()
    if aligned.empty:
        return 0.0

    resid_vals = aligned["resid"].to_numpy(dtype=float)
    y_vals = aligned["y"].to_numpy(dtype=float)
    ss_res = float(np.sum(resid_vals ** 2))
    ss_tot = float(np.sum((y_vals - y_vals.mean()) ** 2))
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def residualize_returns(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    sector_returns: Optional[pd.DataFrame] = None,
    min_observations: int = 30,
    method: str = "rolling",
    rolling_window: int = 126,
    ewm_halflife: float = 63.0,
    kalman_process_variance: float = 1e-5,
    kalman_observation_variance: Optional[float] = None,
    kalman_initial_covariance: float = 10.0,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Residualize stock returns against market and sector factors.
    
    This is the CORE FIX for the "Beta in Disguise" problem. Running PCA/ICA
    on raw returns mathematically guarantees rediscovering market beta (F1) 
    and sector exposures (F2-F5). This function removes that systematic 
    exposure, leaving only idiosyncratic (alpha) returns.
    
    Mathematical Process:
    --------------------
    1. Fit time-varying market beta (rolling/EWM/Kalman): R_i,t = α_i,t + β_i,t × R_mkt,t + ε_i,t
    2. Fit time-varying sector betas on ε_i,t: ε_i,t = α'_i,t + Σ γ_ij,t × R_sector_j,t + η_i,t
    3. Return η_i - the pure stock-specific returns
    
    Parameters
    ----------
    returns : pd.DataFrame
        Stock return matrix (T×N) with dates as index, tickers as columns
    market_returns : pd.Series
        Market benchmark returns (e.g., SPY), aligned to returns index
    sector_returns : pd.DataFrame, optional
        Sector ETF returns for additional cleaning
    min_observations : int, default 30
        Minimum historical observations required before first beta estimate
    method : str, default "rolling"
        Time-varying beta estimator:
        - "rolling": rolling OLS with fixed lookback
        - "ewm": exponentially-weighted OLS
        - "kalman": state-space random-walk beta filter
    rolling_window : int, default 126
        Lookback window used for rolling/EWM regressions
    ewm_halflife : float, default 63.0
        Half-life for exponentially-weighted regression (used when method="ewm")
    kalman_process_variance : float, default 1e-5
        Process noise scale for beta drift when method="kalman"
    kalman_observation_variance : float | None, default None
        Observation noise variance for method="kalman". If None, estimated
        from initial in-sample residuals.
    kalman_initial_covariance : float, default 10.0
        Initial state covariance scale for method="kalman"
    verbose : bool, default True
        Log R² statistics for diagnostics
        
    Returns
    -------
    pd.DataFrame
        Residualized returns (idiosyncratic component only)
        
    Notes
    -----
    - Coefficients at time t are estimated only from data before t (no look-ahead)
    - Mean market R² is typically 0.15-0.35 for diversified stocks
    - Additional sector R² is typically 0.05-0.15
    """
    if min_observations < 2:
        raise ValueError("min_observations must be >= 2")
    if method not in {"rolling", "ewm", "kalman"}:
        raise ValueError(f"Unknown residualization method: {method}")
    if method == "kalman" and kalman_process_variance <= 0:
        raise ValueError("kalman_process_variance must be > 0")
    if (
        method == "kalman"
        and kalman_observation_variance is not None
        and kalman_observation_variance <= 0
    ):
        raise ValueError("kalman_observation_variance must be > 0 when provided")
    if method == "kalman" and kalman_initial_covariance <= 0:
        raise ValueError("kalman_initial_covariance must be > 0")

    effective_window = rolling_window
    if method in {"rolling", "ewm"}:
        effective_window = max(rolling_window, min_observations)
    if method in {"rolling", "ewm"} and rolling_window < min_observations and verbose:
        _LOG.info(
            "rolling_window=%d < min_observations=%d; using window=%d",
            rolling_window, min_observations, effective_window
        )

    residual_returns = returns.copy()
    market_r2_list = []
    sector_r2_list = []
    insufficient_data = []
    
    # Ensure market returns align
    market_aligned = market_returns.reindex(returns.index).dropna()
    common_idx = market_aligned.index
    
    if len(common_idx) < min_observations:
        _LOG.error("Insufficient market data overlap for residualization")
        return returns

    sector_aligned = None
    if sector_returns is not None and len(sector_returns.columns) > 0:
        sector_aligned = sector_returns.reindex(common_idx)
    
    for stock in returns.columns:
        stock_data = returns.loc[common_idx, stock].dropna()
        if len(stock_data) < min_observations:
            insufficient_data.append(stock)
            continue

        stock_residual = stock_data.copy()

        # Step 1: Time-varying market beta removal
        market_resid = _time_varying_residuals(
            y=stock_data,
            X=market_aligned.loc[stock_data.index].to_frame(name=MARKET_ETF),
            method=method,
            window=effective_window,
            min_observations=min_observations,
            ewm_halflife=ewm_halflife,
            kalman_process_variance=kalman_process_variance,
            kalman_observation_variance=kalman_observation_variance,
            kalman_initial_covariance=kalman_initial_covariance
        )

        market_pred_idx = market_resid.dropna().index
        if len(market_pred_idx) == 0:
            insufficient_data.append(stock)
            continue

        stock_residual.loc[market_pred_idx] = market_resid.loc[market_pred_idx]
        market_r2_list.append(_residual_r2(stock_data, market_resid))

        # Step 2: Time-varying sector beta removal on market-residualized series
        if sector_aligned is not None:
            stage1_resid = market_resid.dropna()
            if len(stage1_resid) >= min_observations:
                sect_data = sector_aligned.loc[stage1_resid.index].dropna()

                if len(sect_data) >= min_observations:
                    stage1_aligned = stage1_resid.loc[sect_data.index]
                    sector_resid = _time_varying_residuals(
                        y=stage1_aligned,
                        X=sect_data,
                        method=method,
                        window=effective_window,
                        min_observations=min_observations,
                        ewm_halflife=ewm_halflife,
                        kalman_process_variance=kalman_process_variance,
                        kalman_observation_variance=kalman_observation_variance,
                        kalman_initial_covariance=kalman_initial_covariance
                    )
                    sector_pred_idx = sector_resid.dropna().index
                    if len(sector_pred_idx) > 0:
                        stock_residual.loc[sector_pred_idx] = sector_resid.loc[sector_pred_idx]
                        sector_r2_list.append(_residual_r2(stage1_aligned, sector_resid))

        residual_returns.loc[stock_residual.index, stock] = stock_residual
    
    if verbose:
        if market_r2_list:
            _LOG.info(
                "Residualization complete | Market R²: %.3f (%.3f) | "
                "Sector R²: %.3f (%.3f) | Failed: %d/%d",
                np.mean(market_r2_list), np.std(market_r2_list),
                np.mean(sector_r2_list) if sector_r2_list else 0,
                np.std(sector_r2_list) if sector_r2_list else 0,
                len(insufficient_data), len(returns.columns)
            )
        if insufficient_data:
            _LOG.debug("Stocks with insufficient data: %s", insufficient_data[:10])
    
    return residual_returns


# --------------------------------------------------------------------- #
# ‑‑‑ Factor Orthogonalization ‑‑‑
# --------------------------------------------------------------------- #
def orthogonalize_factors(
    factor_returns: pd.DataFrame,
    method: str = "symmetric"
) -> pd.DataFrame:
    """
    Orthogonalize factor returns to ensure zero cross-correlation.
    
    While PCA produces orthogonal components by construction, other methods
    (ICA, autoencoders) may produce correlated factors. This function enforces
    strict orthogonality, preventing optimizers from double-counting signals.
    
    Parameters
    ----------
    factor_returns : pd.DataFrame
        Factor returns matrix (T×K)
    method : str, default "symmetric"
        - "symmetric": SVD-based, order-independent (recommended)
        - "gram_schmidt": Sequential orthogonalization
        - "modified_gs": Numerically stable Gram-Schmidt
        
    Returns
    -------
    pd.DataFrame
        Orthogonalized factors with identity correlation matrix
    """
    F = factor_returns.values.copy()
    T, K = F.shape
    
    # Center factors
    F_centered = F - F.mean(axis=0, keepdims=True)
    
    if method == "symmetric":
        # SVD: F = U @ Σ @ V^T => F_orth = U @ Σ
        U, S, Vt = np.linalg.svd(F_centered, full_matrices=False)
        F_orth = U @ np.diag(S)
        
    elif method == "gram_schmidt":
        F_orth = np.zeros_like(F_centered)
        F_orth[:, 0] = F_centered[:, 0]
        for k in range(1, K):
            f_k = F_centered[:, k].copy()
            for j in range(k):
                proj = np.dot(F_centered[:, k], F_orth[:, j]) / np.dot(F_orth[:, j], F_orth[:, j])
                f_k -= proj * F_orth[:, j]
            F_orth[:, k] = f_k
            
    elif method == "modified_gs":
        F_orth = F_centered.copy()
        for k in range(K):
            norm = np.linalg.norm(F_orth[:, k])
            if norm > 1e-10:
                F_orth[:, k] /= norm
            for j in range(k + 1, K):
                proj = np.dot(F_orth[:, j], F_orth[:, k])
                F_orth[:, j] -= proj * F_orth[:, k]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Preserve original standard deviations
    orig_std = factor_returns.std(axis=0).values
    orth_std = F_orth.std(axis=0)
    scale_factors = np.where(orth_std > 1e-10, orig_std / orth_std, 1.0)
    F_scaled = F_orth * scale_factors
    
    # Restore means
    F_final = F_scaled + factor_returns.mean(axis=0).values
    
    return pd.DataFrame(F_final, index=factor_returns.index, columns=factor_returns.columns)


# --------------------------------------------------------------------- #
# ‑‑‑ Classical statistical methods ‑‑‑
# --------------------------------------------------------------------- #
class StatMethod(Enum):
    """Enumeration of supported statistical factor discovery methods."""
    PCA = auto()   # Orthogonal by construction
    ICA = auto()   # Independent but NOT orthogonal - requires post-processing
    NMF = auto()   # Non-negative, not orthogonal
    SPARSE_PCA = auto() # Sparse loadings
    FACTOR_ANALYSIS = auto() # Classical factor analysis


def statistical_factors(
    returns: pd.DataFrame,
    n_components: int = 10,
    method: StatMethod = StatMethod.PCA,
    whiten: bool = True,
    skip_residualization: bool = False,
    skip_orthogonalization: bool = False,
    cache_backend=None,
    residualization_method: str = "rolling",
    residualization_window: int = 126,
    residualization_halflife: float = 63.0,
    residualization_min_observations: int = 30,
    residualization_kalman_process_variance: float = 1e-5,
    residualization_kalman_observation_variance: Optional[float] = None,
    residualization_kalman_initial_covariance: float = 10.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Discover latent factors using classical statistical methods.
    
    **IMPORTANT**: This function automatically residualizes returns against
    market (SPY) and sector ETFs BEFORE factor extraction. This is the correct
    approach - running PCA/ICA on raw returns just rediscovers market beta.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Stock return matrix (T×N). Will be residualized automatically.
    n_components : int, default 10
        Number of factors to extract. Use ~10% of asset count.
    method : StatMethod, default StatMethod.PCA
        - PCA: Fast, orthogonal, variance-maximizing
        - ICA: Independent components (auto-orthogonalized)
        - NMF: Non-negative decomposition
        - SPARSE_PCA: Sparse loadings
        - FACTOR_ANALYSIS: Classical factor analysis
    whiten : bool, default True
        Standardize returns before factor extraction
    skip_residualization : bool, default False
        **DANGEROUS**: Skip market/sector residualization. Only use if your
        returns are already residualized or you specifically want to discover
        market beta (rarely the goal).
    skip_orthogonalization : bool, default False
        Skip orthogonalization for non-orthogonal methods (ICA, NMF).
        Not recommended - leads to correlated factors.
    cache_backend : DataBackend-compatible, optional
        Pre-initialized backend for fetching benchmark data efficiently.
        This can be a `DataBackend` or any object implementing `get_prices(...)`,
        including `FactorResearchSystem`.
    residualization_method : str, default "rolling"
        Time-varying beta method used during benchmark residualization:
        "rolling", "ewm", or "kalman".
    residualization_window : int, default 126
        Lookback window (trading days) for time-varying beta estimation.
    residualization_halflife : float, default 63.0
        Half-life for exponentially weighted residualization when
        `residualization_method="ewm"`.
    residualization_min_observations : int, default 30
        Minimum lookback observations required before residualizing a timestamp.
    residualization_kalman_process_variance : float, default 1e-5
        Process noise scale for Kalman residualization.
    residualization_kalman_observation_variance : float | None, default None
        Observation noise variance for Kalman residualization.
    residualization_kalman_initial_covariance : float, default 10.0
        Initial state covariance scale for Kalman residualization.
        
    Returns
    -------
    fac_ret : pd.DataFrame
        Factor returns (T×K) - time series of factor performance
    loadings : pd.DataFrame
        Factor loadings (N×K) - asset exposures to each factor
        
    Examples
    --------
    >>> # Standard usage - auto-residualized, auto-orthogonalized
    >>> factors, loadings = statistical_factors(returns, n_components=10)
    
    >>> # Skip residualization (NOT recommended - you get Beta in Disguise)
    >>> factors, loadings = statistical_factors(returns, skip_residualization=True)
    """
    # Step 1: Automatic residualization (THE FIX for Beta in Disguise)
    if not skip_residualization:
        market_returns, sector_returns = _fetch_benchmark_returns(
            returns.index, cache_backend
        )
        
        if market_returns is not None:
            returns = residualize_returns(
                returns, 
                market_returns=market_returns,
                sector_returns=sector_returns,
                min_observations=residualization_min_observations,
                method=residualization_method,
                rolling_window=residualization_window,
                ewm_halflife=residualization_halflife,
                kalman_process_variance=residualization_kalman_process_variance,
                kalman_observation_variance=residualization_kalman_observation_variance,
                kalman_initial_covariance=residualization_kalman_initial_covariance,
                verbose=True
            )
        else:
            _LOG.warning(
                "Could not fetch benchmark data. Running on raw returns - "
                "factors will contain market beta (may not be desired)."
            )
    else:
        _LOG.warning(
            "SKIP_RESIDUALIZATION=True: You are running factor discovery on raw returns. "
            "The first factor will be ~SPY, factors 2-5 will be sector exposures. "
            "This is usually NOT what you want for alpha generation."
        )
    
    # Step 2: Factor extraction
    scaler = StandardScaler(with_mean=True, with_std=whiten)
    X = scaler.fit_transform(returns.to_numpy())

    if method is StatMethod.PCA:
        model = PCA(n_components=n_components, random_state=0)
        needs_orthogonalization = False  # PCA is orthogonal by construction
    elif method is StatMethod.ICA:
        whiten_param = 'unit-variance' if whiten else False
        model = FastICA(n_components=n_components, random_state=0, whiten=whiten_param)
        needs_orthogonalization = True   # ICA factors are independent but correlated
    elif method is StatMethod.SPARSE_PCA:
        model = SparsePCA(n_components=n_components, random_state=0)
        needs_orthogonalization = True
    elif method is StatMethod.FACTOR_ANALYSIS:
        model = FactorAnalysis(n_components=n_components, random_state=0)
        needs_orthogonalization = True
    else:  # NMF
        X = returns.clip(lower=0).to_numpy()
        model = NMF(n_components=n_components, init="nndsvda", random_state=0,
                    max_iter=2000, tol=1e-3)
        needs_orthogonalization = True   # NMF factors are not orthogonal

    F = model.fit_transform(X)           # obs × k
    B = model.components_.T              # asset × k

    # Build DataFrames
    loadings_df = pd.DataFrame(
        B, 
        index=returns.columns,
        columns=[f"F{k}" for k in range(1, B.shape[1] + 1)]
    )

    # Calculate factor returns via weighted regression
    B_normalized = B / np.abs(B).sum(axis=0, keepdims=True)
    factor_returns_array = returns.values @ B_normalized

    fac_ret = pd.DataFrame(
        factor_returns_array, 
        index=returns.index,
        columns=loadings_df.columns
    )
    
    # Step 3: Automatic orthogonalization for non-orthogonal methods
    if needs_orthogonalization and not skip_orthogonalization:
        _LOG.info("Applying orthogonalization to %s factors...", method.name)
        fac_ret = orthogonalize_factors(fac_ret, method="symmetric")
        
        # Recalculate loadings for orthogonalized factors
        F = fac_ret.values
        R = returns.values
        B_new = np.linalg.lstsq(F, R, rcond=None)[0].T
        loadings_df = pd.DataFrame(
            B_new, 
            index=returns.columns,
            columns=fac_ret.columns
        )
        
        max_corr = fac_ret.corr().values[np.triu_indices_from(fac_ret.corr().values, k=1)].max()
        _LOG.info("Orthogonalization complete. Max off-diagonal correlation: %.4f", max_corr)
    
    return fac_ret, loadings_df


# --------------------------------------------------------------------- #
# ‑‑‑ Deep autoencoder (PyTorch) ‑‑‑
# --------------------------------------------------------------------- #
if TORCH_AVAILABLE:
    class _AutoEncoder(nn.Module):
        """Neural network autoencoder for non-linear factor discovery."""
        
        def __init__(self, n_assets: int, k: int, hidden: int):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(n_assets, hidden),
                nn.ReLU(),
                nn.Linear(hidden, k)
            )
            self.dec = nn.Sequential(
                nn.Linear(k, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_assets)
            )

        def forward(self, x):
            z = self.enc(x)
            out = self.dec(z)
            return out, z


def autoencoder_factors(
    returns: pd.DataFrame,
    k: int = 10,
    hidden: int = 128,
    lr: float = 1e-3,
    epochs: int = 200,
    batch: int = 64,
    device: str | None = None,
    skip_residualization: bool = False,
    cache_backend=None,
    residualization_method: str = "rolling",
    residualization_window: int = 126,
    residualization_halflife: float = 63.0,
    residualization_min_observations: int = 30,
    residualization_kalman_process_variance: float = 1e-5,
    residualization_kalman_observation_variance: Optional[float] = None,
    residualization_kalman_initial_covariance: float = 10.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Discover latent factors using neural network autoencoder.
    
    **IMPORTANT**: Automatically residualizes returns against market/sectors
    BEFORE training, and orthogonalizes factors AFTER extraction.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Stock return matrix (T×N). Auto-residualized.
    k : int, default 10
        Number of latent factors
    hidden : int, default 128
        Hidden layer dimension
    lr : float, default 1e-3
        Learning rate
    epochs : int, default 200
        Training epochs
    batch : int, default 64
        Batch size
    device : str | None, default None
        "cuda" or "cpu" (auto-detected if None)
    skip_residualization : bool, default False
        Skip market/sector residualization (not recommended)
    cache_backend : DataBackend-compatible, optional
        Pre-initialized backend for benchmark fetching
    residualization_method : str, default "rolling"
        Time-varying beta method used during benchmark residualization:
        "rolling", "ewm", or "kalman".
    residualization_window : int, default 126
        Lookback window (trading days) for time-varying beta estimation.
    residualization_halflife : float, default 63.0
        Half-life for exponentially weighted residualization when
        `residualization_method="ewm"`.
    residualization_min_observations : int, default 30
        Minimum lookback observations required before residualizing a timestamp.
    residualization_kalman_process_variance : float, default 1e-5
        Process noise scale for Kalman residualization.
    residualization_kalman_observation_variance : float | None, default None
        Observation noise variance for Kalman residualization.
    residualization_kalman_initial_covariance : float, default 10.0
        Initial state covariance scale for Kalman residualization.
        
    Returns
    -------
    fac_ret : pd.DataFrame
        Factor returns (T×K) - orthogonalized
    loadings : pd.DataFrame
        Factor loadings (N×K)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required. Install with: pip install torch")
    
    # Step 1: Automatic residualization
    if not skip_residualization:
        market_returns, sector_returns = _fetch_benchmark_returns(
            returns.index, cache_backend
        )
        if market_returns is not None:
            returns = residualize_returns(
                returns, 
                market_returns=market_returns,
                sector_returns=sector_returns,
                min_observations=residualization_min_observations,
                method=residualization_method,
                rolling_window=residualization_window,
                ewm_halflife=residualization_halflife,
                kalman_process_variance=residualization_kalman_process_variance,
                kalman_observation_variance=residualization_kalman_observation_variance,
                kalman_initial_covariance=residualization_kalman_initial_covariance,
                verbose=True
            )
        else:
            _LOG.warning("Could not fetch benchmark data. Using raw returns.")
    else:
        _LOG.warning("SKIP_RESIDUALIZATION=True: Using raw returns (includes market beta)")
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(returns.to_numpy(), dtype=torch.float32)
    ds = TensorDataset(X)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True)

    model = _AutoEncoder(n_assets=X.shape[1], k=k, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        for (x,) in dl:
            x = x.to(device)
            opt.zero_grad()
            x_hat, _ = model(x)
            loss = loss_fn(x_hat, x)
            loss.backward()
            opt.step()
        if (epoch + 1) % 50 == 0:
            _LOG.info("AE epoch %d/%d  loss %.5f", epoch + 1, epochs, loss.item())

    # Extract factor returns
    with torch.no_grad():
        _, Z = model(X.to(device))
    fac_ret = pd.DataFrame(
        Z.cpu().numpy(), 
        index=returns.index,
        columns=[f"F{i}" for i in range(1, k + 1)]
    )

    # Orthogonalize autoencoder factors (they're not orthogonal by construction)
    _LOG.info("Orthogonalizing autoencoder factors...")
    fac_ret = orthogonalize_factors(fac_ret, method="symmetric")
    
    # Recalculate loadings
    loadings = pd.DataFrame(index=returns.columns, columns=fac_ret.columns)
    for i, sym in enumerate(returns.columns):
        beta = np.linalg.lstsq(fac_ret, returns.iloc[:, i], rcond=None)[0]
        loadings.loc[sym] = beta
    
    return fac_ret, loadings.astype(float)


# --------------------------------------------------------------------- #
# ‑‑‑ Validation ‑‑‑
# --------------------------------------------------------------------- #
def validate_factor_distinctiveness(
    factor_returns: pd.DataFrame, 
    factor_loadings: pd.DataFrame,
    corr_threshold: float = 0.8
) -> Dict[str, any]:
    """
    Comprehensive validation framework for factor quality and distinctiveness.
    
    This function performs extensive validation to ensure discovered factors are
    statistically meaningful, economically interpretable, and not redundant.
    
    Parameters
    ----------
    factor_returns : pd.DataFrame
        Factor returns matrix (T×K)
    factor_loadings : pd.DataFrame
        Factor loadings matrix (N×K)
    corr_threshold : float, default 0.8
        Correlation threshold for flagging redundant factors
        
    Returns
    -------
    Dict[str, any]
        Validation results including:
        - is_valid: Overall validation status
        - warnings: List of issues found
        - correlations: Factor correlation analysis
        - recommendations: Actionable suggestions
    """
    validation_results = {
        "is_valid": True,
        "warnings": [],
        "correlations": {},
        "recommendations": []
    }
    
    # Check factor return correlations
    factor_corr = factor_returns.corr()
    high_corr_pairs = []
    
    for i in range(len(factor_corr.columns)):
        for j in range(i+1, len(factor_corr.columns)):
            corr_val = abs(factor_corr.iloc[i, j])
            if corr_val > corr_threshold:
                pair = (factor_corr.columns[i], factor_corr.columns[j])
                high_corr_pairs.append((pair, corr_val))
    
    if high_corr_pairs:
        validation_results["is_valid"] = False
        validation_results["warnings"].append(
            f"Found {len(high_corr_pairs)} factor pairs with correlation > {corr_threshold}"
        )
        validation_results["correlations"]["high_pairs"] = high_corr_pairs
        validation_results["recommendations"].append(
            "Consider reducing the number of factors or using regularization"
        )
    
    # Check loading distribution
    for factor in factor_loadings.columns:
        loadings = factor_loadings[factor]
        loading_std = loadings.std()
        
        if loading_std < 0.01:
            validation_results["warnings"].append(
                f"Factor {factor} has uniform loadings (std={loading_std:.4f})"
            )
        
        # Check concentration
        abs_loadings = loadings.abs()
        top_5_weight = abs_loadings.nlargest(5).sum() / abs_loadings.sum()
        if top_5_weight > 0.8:
            validation_results["warnings"].append(
                f"Factor {factor} is concentrated in few assets ({top_5_weight:.1%} in top 5)"
            )
    
    # Check return statistics
    for factor in factor_returns.columns:
        returns = factor_returns[factor]
        
        daily_vol = returns.std()
        annualized_vol = daily_vol * np.sqrt(252) * 100
        
        if annualized_vol > 100:
            validation_results["warnings"].append(
                f"Factor {factor} has high volatility ({annualized_vol:.1f}% annualized)"
            )
        
        if (returns > 0).all():
            validation_results["warnings"].append(f"Factor {factor} has only positive returns")
        elif (returns < 0).all():
            validation_results["warnings"].append(f"Factor {factor} has only negative returns")
    
    if not validation_results["warnings"]:
        validation_results["recommendations"].append("Factors pass basic validation checks")
    
    _LOG.info("Factor validation: %s", "PASSED" if validation_results["is_valid"] else "FAILED")
    
    return validation_results
