from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.ui.components.cards import badge, status_kind_from_threshold
from src.ui.components.empty_states import missing_data
from src.ui.context import AppContext
from src.ui.view_models import (
    compute_portfolio_vitals,
    resolve_as_of,
    validate_portfolio_inputs,
)


def _parse_return_series(data: bytes, preferred_names: list[str], default_name: str) -> Optional[pd.Series]:
    try:
        frame = pd.read_csv(io.BytesIO(data), index_col=0, parse_dates=True)
    except Exception:
        return None

    if frame.empty:
        return None

    if frame.shape[1] == 1:
        series = frame.iloc[:, 0]
    else:
        normalized = {c.lower(): c for c in frame.columns}
        selected = None
        for name in preferred_names:
            if name.lower() in normalized:
                selected = normalized[name.lower()]
                break
        if selected is None:
            selected = frame.columns[0]
        series = frame[selected]

    series = pd.to_numeric(series, errors="coerce").dropna()
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index, errors="coerce")
    series = series.sort_index().dropna()
    series.name = default_name
    return series if not series.empty else None


def _risk_stats(series: pd.Series) -> dict[str, float]:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return {}

    vol = float(series.std(ddof=0) * np.sqrt(252))

    cum = (1.0 + series).cumprod()
    running_max = cum.expanding().max()
    dd = (cum - running_max) / running_max
    max_dd = float(dd.min())

    var_95 = float(np.nanpercentile(series.to_numpy(dtype=float), 5))
    cvar_95 = float(series[series <= var_95].mean()) if (series <= var_95).any() else float("nan")
    return {"vol": vol, "max_dd": max_dd, "var_95": var_95, "cvar_95": cvar_95}


def render(ctx: AppContext) -> None:
    st.title("Portfolio & Risk")
    st.caption("Portfolio inputs, risk metrics, attribution, and trade basket review.")

    factor_returns = ctx.data.factor_returns
    if factor_returns is None or factor_returns.empty:
        missing_data(
            title="Factor returns not available",
            message="This page uses factor returns for attribution and analytics.",
            steps=("Run discovery in **Research**.",),
        )
        return

    as_of = resolve_as_of(factor_returns, ctx.state.as_of)
    returns = factor_returns.loc[factor_returns.index <= as_of]

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    st.markdown("### Inputs")
    st.caption("Upload portfolio/benchmark return series or use local CSV artifacts.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Portfolio returns**")
        st.caption("Expected CSV: date index + one return column.")
        upl_port = st.file_uploader("Upload portfolio returns", type=["csv"], key="upl_portfolio")

    with col2:
        st.markdown("**Benchmark returns**")
        st.caption("Expected CSV: date index + one return column.")
        upl_bench = st.file_uploader("Upload benchmark returns", type=["csv"], key="upl_benchmark")

    use_uploaded = st.checkbox("Use uploaded series for this session", value=bool(upl_port or upl_bench))
    save_uploaded = st.checkbox("Save uploaded series to disk", value=False, help="Writes to portfolio_returns.csv and benchmark_returns.csv")

    portfolio = ctx.data.portfolio_returns
    benchmark = ctx.data.benchmark_returns

    if use_uploaded:
        if upl_port is not None:
            parsed = _parse_return_series(upl_port.getvalue(), preferred_names=["portfolio", "strategy", "return"], default_name="portfolio")
            if parsed is not None:
                portfolio = parsed
        if upl_bench is not None:
            parsed = _parse_return_series(upl_bench.getvalue(), preferred_names=["benchmark", "spy", "return"], default_name="benchmark")
            if parsed is not None:
                benchmark = parsed

    if save_uploaded and use_uploaded:
        root = ctx.paths.project_root
        if portfolio is not None:
            portfolio.rename("portfolio").to_frame().to_csv(root / "portfolio_returns.csv")
        if benchmark is not None:
            benchmark.rename("benchmark").to_frame().to_csv(root / "benchmark_returns.csv")
        st.success("Saved uploaded series to disk.")

    status = validate_portfolio_inputs(portfolio, benchmark, min_rows=63)
    if status.overall_ok:
        badge(f"Ready ({status.overlap_rows} overlapping rows)", "ok")
        st.caption(f"Overlap: {status.overlap_start.date()} → {status.overlap_end.date()}")
    else:
        badge("Incomplete inputs", "neutral")
        for msg in status.messages:
            st.caption(f"- {msg}")

    st.divider()

    # ------------------------------------------------------------------
    # Core vitals
    # ------------------------------------------------------------------
    st.markdown("### Core vitals")
    vitals = compute_portfolio_vitals(
        factor_returns=returns,
        portfolio_returns=portfolio,
        benchmark_returns=benchmark,
        lookback=min(ctx.state.lookback, len(returns)),
    )

    v1, v2, v3, v4, v5 = st.columns(5)
    with v1:
        st.metric("Active risk", f"{vitals.active_risk:.2%}" if vitals.active_risk is not None else "—")
    with v2:
        st.metric("Tracking error", f"{vitals.tracking_error:.2%}" if vitals.tracking_error is not None else "—")
    with v3:
        st.metric("Beta", f"{vitals.beta:.2f}" if vitals.beta is not None else "—")
        badge(
            "Market neutral" if vitals.beta is not None else "Need data",
            status_kind_from_threshold(vitals.beta, ok_max_abs=0.10, watch_max_abs=0.20) if vitals.beta is not None else "neutral",
        )
    with v4:
        st.metric("Info ratio", f"{vitals.info_ratio:.2f}" if vitals.info_ratio is not None else "—")
    with v5:
        st.metric("Unexplained risk", f"{vitals.unexplained_risk_ratio:.0%}" if vitals.unexplained_risk_ratio is not None else "—")

    # ------------------------------------------------------------------
    # Risk stats
    # ------------------------------------------------------------------
    if portfolio is not None and not portfolio.empty:
        st.divider()
        st.markdown("### Risk statistics")
        stats = _risk_stats(portfolio.tail(min(ctx.state.lookback, len(portfolio))))
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("Volatility (ann.)", f"{stats.get('vol', float('nan')):.2%}" if stats else "—")
        with r2:
            st.metric("Max drawdown", f"{stats.get('max_dd', float('nan')):.2%}" if stats else "—")
        with r3:
            st.metric("VaR 95% (1D)", f"{stats.get('var_95', float('nan')):.2%}" if stats else "—")
        with r4:
            st.metric("CVaR 95% (1D)", f"{stats.get('cvar_95', float('nan')):.2%}" if stats else "—")

    # ------------------------------------------------------------------
    # Trade basket
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("### Trade basket preview")
    basket = ctx.data.latest_trade_basket
    if basket is not None and not basket.empty:
        src = basket.attrs.get("source_path", "")
        st.caption(f"Source: {src}" if src else "Source: (unknown)")
        st.dataframe(basket, use_container_width=True, height=360)

        adv_cols = [c for c in basket.columns if c.lower() in {"adv", "avg_daily_volume", "average_daily_volume"}]
        if not adv_cols:
            st.caption("Liquidity checks unavailable (no ADV/volume columns found in basket).")
    else:
        st.caption("No trade basket CSV found in the project root.")

