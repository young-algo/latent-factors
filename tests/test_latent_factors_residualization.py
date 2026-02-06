import numpy as np
import pandas as pd

from src.latent_factors import residualize_returns


def test_residualize_returns_removes_regime_dependent_market_beta():
    rng = np.random.default_rng(42)
    n_obs = 420
    dates = pd.date_range("2022-01-03", periods=n_obs, freq="B")

    market = pd.Series(rng.normal(0.0, 0.01, n_obs), index=dates, name="SPY")
    beta = np.concatenate([np.full(n_obs // 2, 1.5), np.full(n_obs - n_obs // 2, 0.8)])
    noise = rng.normal(0.0, 0.005, n_obs)
    stock = beta * market.to_numpy() + noise
    returns = pd.DataFrame({"TECH": stock}, index=dates)

    residualized = residualize_returns(
        returns=returns,
        market_returns=market,
        min_observations=30,
        method="rolling",
        rolling_window=126,
        verbose=False,
    )

    # Ignore warm-up region where no time-varying beta is estimated yet.
    first_regime = dates[30 : n_obs // 2]
    # Skip transition so rolling-window betas have time to adapt to the new regime.
    second_regime = dates[(n_obs // 2 + 126) :]

    raw_corr_1 = returns.loc[first_regime, "TECH"].corr(market.loc[first_regime])
    raw_corr_2 = returns.loc[second_regime, "TECH"].corr(market.loc[second_regime])
    resid_corr_1 = residualized.loc[first_regime, "TECH"].corr(market.loc[first_regime])
    resid_corr_2 = residualized.loc[second_regime, "TECH"].corr(market.loc[second_regime])

    assert abs(resid_corr_1) < abs(raw_corr_1)
    assert abs(resid_corr_2) < abs(raw_corr_2)
    assert abs(resid_corr_1) < 0.25
    assert abs(resid_corr_2) < 0.25


def test_residualize_returns_is_not_affected_by_future_market_shock():
    rng = np.random.default_rng(7)
    n_obs = 260
    dates = pd.date_range("2021-01-04", periods=n_obs, freq="B")

    market_base = pd.Series(rng.normal(0.0, 0.01, n_obs), index=dates, name="SPY")
    stock = 1.2 * market_base.to_numpy() + rng.normal(0.0, 0.004, n_obs)
    returns = pd.DataFrame({"AAPL": stock}, index=dates)

    market_shock = market_base.copy()
    shock_ix = 210
    market_shock.iloc[shock_ix] += 0.2

    residual_base = residualize_returns(
        returns=returns,
        market_returns=market_base,
        min_observations=30,
        method="rolling",
        rolling_window=126,
        verbose=False,
    )
    residual_shock = residualize_returns(
        returns=returns,
        market_returns=market_shock,
        min_observations=30,
        method="rolling",
        rolling_window=126,
        verbose=False,
    )

    pre_shock_idx = dates[:shock_ix]
    max_diff_pre_shock = (
        residual_base.loc[pre_shock_idx, "AAPL"] - residual_shock.loc[pre_shock_idx, "AAPL"]
    ).abs().max()

    assert max_diff_pre_shock < 1e-12


def test_sector_time_varying_stage_reduces_sector_exposure():
    rng = np.random.default_rng(123)
    n_obs = 340
    dates = pd.date_range("2020-01-02", periods=n_obs, freq="B")

    market = pd.Series(rng.normal(0.0, 0.009, n_obs), index=dates, name="SPY")
    xlk = pd.Series(rng.normal(0.0, 0.011, n_obs), index=dates, name="XLK")
    xlf = pd.Series(rng.normal(0.0, 0.010, n_obs), index=dates, name="XLF")
    sector_df = pd.DataFrame({"XLK": xlk, "XLF": xlf}, index=dates)

    beta_xlk = np.concatenate([np.full(n_obs // 2, 0.9), np.full(n_obs - n_obs // 2, 0.2)])
    beta_xlf = np.full(n_obs, 0.5)
    stock = (
        1.0 * market.to_numpy()
        + beta_xlk * xlk.to_numpy()
        + beta_xlf * xlf.to_numpy()
        + rng.normal(0.0, 0.004, n_obs)
    )
    returns = pd.DataFrame({"MSFT": stock}, index=dates)

    residualized = residualize_returns(
        returns=returns,
        market_returns=market,
        sector_returns=sector_df,
        min_observations=30,
        method="rolling",
        rolling_window=63,
        verbose=False,
    )

    eval_idx = dates[80:]
    raw_corr_xlk = returns.loc[eval_idx, "MSFT"].corr(xlk.loc[eval_idx])
    raw_corr_xlf = returns.loc[eval_idx, "MSFT"].corr(xlf.loc[eval_idx])
    resid_corr_xlk = residualized.loc[eval_idx, "MSFT"].corr(xlk.loc[eval_idx])
    resid_corr_xlf = residualized.loc[eval_idx, "MSFT"].corr(xlf.loc[eval_idx])

    assert abs(resid_corr_xlk) < abs(raw_corr_xlk)
    assert abs(resid_corr_xlf) < abs(raw_corr_xlf)
    assert abs(resid_corr_xlk) < 0.30
    assert abs(resid_corr_xlf) < 0.25


def test_residualize_returns_kalman_reduces_market_exposure():
    rng = np.random.default_rng(21)
    n_obs = 300
    dates = pd.date_range("2021-01-04", periods=n_obs, freq="B")

    market = pd.Series(rng.normal(0.0, 0.01, n_obs), index=dates, name="SPY")
    stock = 1.2 * market.to_numpy() + rng.normal(0.0, 0.004, n_obs)
    returns = pd.DataFrame({"NVDA": stock}, index=dates)

    residualized = residualize_returns(
        returns=returns,
        market_returns=market,
        min_observations=30,
        method="kalman",
        verbose=False,
    )

    eval_idx = dates[60:]
    raw_corr = returns.loc[eval_idx, "NVDA"].corr(market.loc[eval_idx])
    resid_corr = residualized.loc[eval_idx, "NVDA"].corr(market.loc[eval_idx])

    assert abs(resid_corr) < abs(raw_corr)
    assert abs(resid_corr) < 0.25


def test_residualize_returns_kalman_has_no_future_leakage():
    rng = np.random.default_rng(11)
    n_obs = 280
    dates = pd.date_range("2021-01-04", periods=n_obs, freq="B")

    market_base = pd.Series(rng.normal(0.0, 0.01, n_obs), index=dates, name="SPY")
    stock = 0.9 * market_base.to_numpy() + rng.normal(0.0, 0.005, n_obs)
    returns = pd.DataFrame({"AMD": stock}, index=dates)

    market_shock = market_base.copy()
    shock_ix = 220
    market_shock.iloc[shock_ix] += 0.25

    residual_base = residualize_returns(
        returns=returns,
        market_returns=market_base,
        min_observations=30,
        method="kalman",
        verbose=False,
    )
    residual_shock = residualize_returns(
        returns=returns,
        market_returns=market_shock,
        min_observations=30,
        method="kalman",
        verbose=False,
    )

    pre_shock_idx = dates[:shock_ix]
    max_diff_pre_shock = (
        residual_base.loc[pre_shock_idx, "AMD"] - residual_shock.loc[pre_shock_idx, "AMD"]
    ).abs().max()

    assert max_diff_pre_shock < 1e-12
