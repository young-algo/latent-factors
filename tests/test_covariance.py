"""
Tests for src.covariance — robust covariance estimation.
"""

import numpy as np
import pandas as pd
import pytest

from src.covariance import (
    CovarianceMethod,
    auto_select_method,
    estimate_covariance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_returns(T: int, K: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    cols = [f"F{i}" for i in range(K)]
    return pd.DataFrame(rng.randn(T, K) * 0.01, columns=cols)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestSample:
    def test_matches_pandas(self):
        """SAMPLE must exactly reproduce pd.DataFrame.cov()."""
        df = _random_returns(100, 5)
        expected = df.cov().values
        result = estimate_covariance(df, method=CovarianceMethod.SAMPLE)
        np.testing.assert_allclose(result, expected, rtol=1e-12)


class TestLedoitWolf:
    def test_positive_definite_near_singular(self):
        """Ledoit-Wolf should produce a PD matrix even when T ~ K."""
        df = _random_returns(50, 20, seed=7)
        cov = estimate_covariance(df, method=CovarianceMethod.LEDOIT_WOLF)
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals > 0), f"Non-PD eigenvalues: {eigvals[:3]}"

    def test_shape(self):
        df = _random_returns(100, 8)
        cov = estimate_covariance(df, method=CovarianceMethod.LEDOIT_WOLF)
        assert cov.shape == (8, 8)

    def test_symmetric(self):
        df = _random_returns(100, 5)
        cov = estimate_covariance(df, method=CovarianceMethod.LEDOIT_WOLF)
        np.testing.assert_allclose(cov, cov.T, atol=1e-14)


class TestOAS:
    def test_high_dimensional(self):
        """OAS should handle p/n > 0.5 gracefully."""
        df = _random_returns(30, 25, seed=99)
        cov = estimate_covariance(df, method=CovarianceMethod.OAS)
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals > 0), f"Non-PD eigenvalues: {eigvals[:3]}"


class TestEWMA:
    def test_requires_halflife(self):
        df = _random_returns(100, 3)
        with pytest.raises(ValueError, match="halflife"):
            estimate_covariance(df, method=CovarianceMethod.EWMA)

    def test_recency_bias(self):
        """Short halflife should weight recent high-vol regime more."""
        rng = np.random.RandomState(0)
        # Low-vol regime then high-vol regime
        low = rng.randn(200, 3) * 0.005
        high = rng.randn(50, 3) * 0.03
        data = pd.DataFrame(
            np.vstack([low, high]), columns=["A", "B", "C"]
        )
        sample_var = data.cov().values[0, 0]
        ewma_cov = estimate_covariance(
            data, method=CovarianceMethod.EWMA, halflife=10
        )
        # EWMA with short halflife should give higher variance than sample
        # because it weights the recent high-vol period more
        assert ewma_cov[0, 0] > sample_var


class TestAnnualization:
    def test_annualize_flag(self):
        df = _random_returns(100, 4)
        raw = estimate_covariance(df, method=CovarianceMethod.LEDOIT_WOLF)
        ann = estimate_covariance(
            df, method=CovarianceMethod.LEDOIT_WOLF, annualize=True
        )
        np.testing.assert_allclose(ann, raw * 252, rtol=1e-12)

    def test_custom_factor(self):
        df = _random_returns(100, 4)
        raw = estimate_covariance(df, method=CovarianceMethod.LEDOIT_WOLF)
        ann = estimate_covariance(
            df,
            method=CovarianceMethod.LEDOIT_WOLF,
            annualize=True,
            annualization_factor=12,
        )
        np.testing.assert_allclose(ann, raw * 12, rtol=1e-12)


class TestAutoSelect:
    def test_low_ratio_selects_oas(self):
        assert auto_select_method(30, 20) == CovarianceMethod.OAS

    def test_high_ratio_selects_lw(self):
        assert auto_select_method(200, 10) == CovarianceMethod.LEDOIT_WOLF

    def test_boundary(self):
        # Exactly 2x => not < 2x, so Ledoit-Wolf
        assert auto_select_method(40, 20) == CovarianceMethod.LEDOIT_WOLF


class TestEdgeCases:
    def test_empty_raises(self):
        df = pd.DataFrame(columns=["A", "B"])
        with pytest.raises(ValueError, match="at least 2"):
            estimate_covariance(df)

    def test_single_row_raises(self):
        df = pd.DataFrame({"A": [0.01], "B": [0.02]})
        with pytest.raises(ValueError, match="at least 2"):
            estimate_covariance(df)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_factor_weighter_backward_compat(self):
        """OptimalFactorWeighter() without cov args still works."""
        from src.factor_weighting import OptimalFactorWeighter

        rng = np.random.RandomState(42)
        loadings = pd.DataFrame(
            rng.randn(50, 5), columns=[f"F{i}" for i in range(5)]
        )
        returns = pd.DataFrame(
            rng.randn(100, 5) * 0.01, columns=[f"F{i}" for i in range(5)]
        )
        w = OptimalFactorWeighter(loadings, returns)
        weights = w.equal_weights()
        assert len(weights) == 5
        assert abs(sum(weights.values()) - 1.0) < 1e-10

    def test_risk_parity_with_shrinkage(self):
        """Risk parity with Ledoit-Wolf produces valid normalized weights."""
        from src.factor_weighting import OptimalFactorWeighter

        rng = np.random.RandomState(42)
        loadings = pd.DataFrame(
            rng.randn(50, 5), columns=[f"F{i}" for i in range(5)]
        )
        returns = pd.DataFrame(
            rng.randn(100, 5) * 0.01, columns=[f"F{i}" for i in range(5)]
        )
        w = OptimalFactorWeighter(loadings, returns)
        weights = w.risk_parity_weights(lookback=63)
        vals = np.array(list(weights.values()))
        assert abs(vals.sum() - 1.0) < 1e-6
        assert np.all(vals >= -1e-10)  # non-negative

    def test_sharpe_optimizer_threads_cov_method(self):
        """SharpeOptimizer passes cov_method to its internal weighter."""
        from src.factor_optimization import SharpeOptimizer
        from src.covariance import CovarianceMethod

        rng = np.random.RandomState(42)
        returns = pd.DataFrame(
            rng.randn(100, 4) * 0.01, columns=[f"F{i}" for i in range(4)]
        )
        opt = SharpeOptimizer(
            factor_returns=returns,
            cov_method=CovarianceMethod.OAS,
        )
        assert opt._weighter.cov_method == CovarianceMethod.OAS
