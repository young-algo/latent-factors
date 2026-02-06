from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.ui.view_models import (
    compute_factor_dna_table,
    compute_portfolio_vitals,
    resolve_as_of,
    validate_portfolio_inputs,
)


def test_validate_portfolio_inputs_overlap() -> None:
    idx = pd.date_range("2024-01-01", periods=100, freq="B")
    portfolio = pd.Series(np.random.default_rng(0).normal(0, 0.01, size=len(idx)), index=idx)
    benchmark = pd.Series(np.random.default_rng(1).normal(0, 0.01, size=len(idx)), index=idx)

    status = validate_portfolio_inputs(portfolio, benchmark, min_rows=63)
    assert status.portfolio_ok is True
    assert status.benchmark_ok is True
    assert status.overlap_ok is True
    assert status.overall_ok is True
    assert status.overlap_rows == 100


def test_compute_portfolio_vitals_beta_tracking_error() -> None:
    idx = pd.date_range("2024-01-01", periods=120, freq="B")
    rng = np.random.default_rng(42)
    bench = pd.Series(rng.normal(0, 0.01, size=len(idx)), index=idx)
    # portfolio is 0.5 * benchmark + idiosyncratic noise
    portfolio = 0.5 * bench + pd.Series(rng.normal(0, 0.005, size=len(idx)), index=idx)

    factor_returns = pd.DataFrame(
        {
            "F1": rng.normal(0, 0.01, size=len(idx)),
            "F2": rng.normal(0, 0.01, size=len(idx)),
        },
        index=idx,
    )

    vitals = compute_portfolio_vitals(
        factor_returns=factor_returns,
        portfolio_returns=portfolio,
        benchmark_returns=bench,
        lookback=63,
    )

    assert vitals.beta is not None
    assert 0.2 < vitals.beta < 0.8
    assert vitals.tracking_error is not None
    assert vitals.tracking_error >= 0.0


def test_compute_factor_dna_table_market_proxy_basis() -> None:
    idx = pd.date_range("2024-01-01", periods=90, freq="B")
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(
        {
            "F1": rng.normal(0, 0.01, size=len(idx)),
            "F2": rng.normal(0, 0.01, size=len(idx)),
        },
        index=idx,
    )
    loadings = pd.DataFrame(
        {
            "F1": [0.1, -0.2, 0.3],
            "F2": [-0.05, 0.02, 0.01],
        },
        index=["AAPL", "MSFT", "NVDA"],
    )

    as_of = resolve_as_of(returns, "2024-04-30")
    table, meta = compute_factor_dna_table(
        factor_returns=returns,
        factor_loadings=loadings,
        factor_names={"F1": "Factor 1", "F2": "Factor 2"},
        as_of=as_of,
        lookback=63,
        benchmark_returns=None,
    )

    assert meta.leakage_basis == "Market Proxy"
    assert not table.empty
    assert set(["Factor", "Name", "Purity", "Leakage Corr", "Crowding"]).issubset(set(table.columns))


def test_no_ui_placeholders() -> None:
    root = Path(__file__).resolve().parents[1]
    ui_root = root / "src" / "ui"
    assert ui_root.exists()

    banned_substrings = [
        "np.random",
        "Alpha Vantage: Connected",
        "OpenAI (LLM): Available",
    ]

    for path in ui_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for banned in banned_substrings:
            assert banned not in text

