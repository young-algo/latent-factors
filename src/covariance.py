"""
Robust Covariance Estimation for Factor Returns
================================================

Provides a single entry point — ``estimate_covariance()`` — that replaces raw
``pd.DataFrame.cov()`` everywhere in the system.  Four estimators are available:

* **SAMPLE** — pandas sample covariance (baseline / backward-compat)
* **LEDOIT_WOLF** — Ledoit-Wolf shrinkage (default)
* **OAS** — Oracle Approximating Shrinkage (better when p/n > 0.5)
* **EWMA** — Exponentially Weighted Moving Average (RiskMetrics-style)

All methods return a K x K ``np.ndarray`` that is guaranteed PSD.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, OAS

_LOGGER = logging.getLogger(__name__)


class CovarianceMethod(Enum):
    """Available covariance estimation methods."""
    SAMPLE = "sample"
    LEDOIT_WOLF = "ledoit_wolf"
    OAS = "oas"
    EWMA = "ewma"


def estimate_covariance(
    returns: pd.DataFrame,
    method: CovarianceMethod = CovarianceMethod.LEDOIT_WOLF,
    halflife: Optional[int] = None,
    annualize: bool = False,
    annualization_factor: int = 252,
) -> np.ndarray:
    """
    Estimate a K x K covariance matrix from factor returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Factor returns matrix (T x K).
    method : CovarianceMethod
        Estimation method to use.
    halflife : int, optional
        Half-life in periods for EWMA.  Required when *method* is EWMA.
    annualize : bool
        If True, multiply the result by *annualization_factor*.
    annualization_factor : int
        Scaling factor (default 252 for daily data).

    Returns
    -------
    np.ndarray
        K x K covariance matrix (always PSD).

    Raises
    ------
    ValueError
        If EWMA is requested without a halflife, or returns are empty.
    """
    if returns.empty or len(returns) < 2:
        raise ValueError(
            f"Need at least 2 observations, got {len(returns)}"
        )

    if method == CovarianceMethod.SAMPLE:
        cov = returns.cov().values

    elif method == CovarianceMethod.LEDOIT_WOLF:
        cov = LedoitWolf().fit(returns.values).covariance_

    elif method == CovarianceMethod.OAS:
        cov = OAS().fit(returns.values).covariance_

    elif method == CovarianceMethod.EWMA:
        if halflife is None:
            raise ValueError("halflife is required for EWMA covariance")
        cov = _ewma_covariance(returns.values, halflife)

    else:
        raise ValueError(f"Unknown method: {method}")

    if annualize:
        cov = cov * annualization_factor

    return cov


def auto_select_method(
    n_observations: int, n_features: int
) -> CovarianceMethod:
    """
    Pick the best shrinkage estimator based on data dimensions.

    Uses OAS when observations are scarce relative to features
    (n_obs < 2 * n_features), otherwise Ledoit-Wolf.  Never auto-selects
    EWMA — that requires an explicit half-life choice.

    Parameters
    ----------
    n_observations : int
        Number of rows (T).
    n_features : int
        Number of columns (K).

    Returns
    -------
    CovarianceMethod
    """
    if n_observations < 2 * n_features:
        return CovarianceMethod.OAS
    return CovarianceMethod.LEDOIT_WOLF


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ewma_covariance(data: np.ndarray, halflife: int) -> np.ndarray:
    """
    RiskMetrics-style exponentially weighted covariance.

    Σ_t = λ * Σ_{t-1} + (1 - λ) * r_t * r_t'

    where λ = 1 - ln(2) / halflife.

    If the resulting matrix is ill-conditioned (condition number > 1e10),
    applies Ledoit-Wolf shrinkage as a post-hoc stabiliser.
    """
    T, K = data.shape
    lam = 1.0 - np.log(2) / halflife

    # Demean
    mu = data.mean(axis=0)
    centered = data - mu

    # Initialise with first observation outer product
    cov = np.outer(centered[0], centered[0])

    for t in range(1, T):
        cov = lam * cov + (1 - lam) * np.outer(centered[t], centered[t])

    # Condition-number guard
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.max() / max(eigvals.min(), 1e-15) > 1e10:
        _LOGGER.info("EWMA covariance ill-conditioned; applying LW shrinkage")
        cov = LedoitWolf().fit(data).covariance_

    return cov
