from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ui.components.empty_states import missing_data
from src.ui.components.tables import dataframe
from src.ui.context import AppContext
from src.ui.data import load_fundamentals_from_cache
from src.ui.plotly_theme import apply_light_theme
from src.ui.view_models import (
    compute_factor_dna_table,
    resolve_as_of,
    sector_tilt_from_fundamentals,
)
from src.ui.components.charts import cumulative_return_chart, drawdown_chart


def _format_name(factor: str, names: dict[str, str]) -> str:
    return names.get(factor, factor)


def render(ctx: AppContext) -> None:
    st.title("Factors")
    st.caption("Factor performance, diagnostics, and drill-down analysis.")

    factor_returns = ctx.data.factor_returns
    factor_loadings = ctx.data.factor_loadings
    factor_names = ctx.data.factor_names or {}

    if factor_returns is None or factor_returns.empty or factor_loadings is None or factor_loadings.empty:
        missing_data(
            title="Factor artifacts not available",
            message="This page requires `factor_returns.csv` and `factor_loadings.csv` in the project root.",
            steps=("Run discovery in **Research**.",),
        )
        return

    as_of = resolve_as_of(factor_returns, ctx.state.as_of)
    lookback = min(ctx.state.lookback, len(factor_returns))

    dna, meta = compute_factor_dna_table(
        factor_returns=factor_returns,
        factor_loadings=factor_loadings,
        factor_names=factor_names,
        as_of=as_of,
        lookback=lookback,
        benchmark_returns=ctx.data.benchmark_returns,
    )

    st.markdown("### Factor DNA")
    st.caption(
        f"Leakage correlation basis: **{meta.leakage_basis}** "
        "(Benchmark if available, otherwise a market proxy computed from factor mean.)"
    )

    query = st.text_input("Search", value="", placeholder="Filter by factor id or name")
    filtered = dna
    if query.strip():
        q = query.strip().lower()
        mask = (
            filtered["Factor"].astype(str).str.lower().str.contains(q)
            | filtered["Name"].astype(str).str.lower().str.contains(q)
        )
        filtered = filtered.loc[mask]

    selection = dataframe(filtered, key="factors_dna", height=420, selection_mode="single-row")
    try:
        selected_rows = getattr(selection, "selection", None)
        if selected_rows and selected_rows.rows:
            row = int(selected_rows.rows[0])
            ctx.state.selected_factor = str(filtered.iloc[row]["Factor"])
    except Exception:
        pass

    st.divider()

    # ------------------------------------------------------------------
    # Detail panel
    # ------------------------------------------------------------------
    factor = ctx.state.selected_factor
    if not factor or factor not in factor_returns.columns:
        factor = str(factor_returns.columns[0])
        ctx.state.selected_factor = factor

    st.markdown(f"### {factor}: {_format_name(factor, factor_names)}")

    series = factor_returns.loc[factor_returns.index <= as_of, factor]

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(cumulative_return_chart(series, title="Cumulative return"), use_container_width=True)
    with c2:
        st.plotly_chart(drawdown_chart(series, title="Drawdown"), use_container_width=True)

    st.markdown("### Exposures")
    if factor in factor_loadings.columns:
        load = factor_loadings[factor].dropna()
        top_long = load.nlargest(15).rename("Loading").to_frame()
        top_short = load.nsmallest(15).rename("Loading").to_frame()
        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**Top longs**")
            st.dataframe(top_long, use_container_width=True)
        with e2:
            st.markdown("**Top shorts**")
            st.dataframe(top_short, use_container_width=True)
    else:
        st.info("No loadings available for selected factor.")

    st.markdown("### Sector tilt")
    if factor in factor_loadings.columns:
        tickers = factor_loadings[factor].abs().nlargest(80).index.tolist()
        fundamentals = load_fundamentals_from_cache(tickers, db_path=str(ctx.paths.db_path), fields=["Sector", "Industry", "Name"])
        sector = sector_tilt_from_fundamentals(factor_loadings[factor], fundamentals, field="Sector", top_n=60)
        if not sector.empty:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Bar(x=sector["weight"], y=sector["sector"], orientation="h", marker_color="#2563EB"))
            fig.update_layout(height=320, xaxis_tickformat=".0%")
            st.plotly_chart(apply_light_theme(fig), use_container_width=True)
        else:
            st.info("Sector tilt not available (no cached fundamentals for these tickers).")
            st.caption("Fetch fundamentals in **Research** to enable sector attribution.")

    st.markdown("### Style attribution (heuristic)")
    st.caption("Proxy-based view using correlations to other latent factors (not a formal FF model).")
    others = [c for c in factor_returns.columns if c != factor]
    if others:
        recent = factor_returns.loc[factor_returns.index <= as_of].tail(lookback)
        corrs = recent[others].corrwith(recent[factor]).dropna().sort_values(key=lambda s: s.abs(), ascending=False).head(8)
        if not corrs.empty:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Bar(x=corrs.values, y=[o for o in corrs.index], orientation="h", marker_color="#0EA5E9"))
            fig.update_layout(height=280, xaxis_tickformat=".2f", xaxis_title="Correlation")
            st.plotly_chart(apply_light_theme(fig), use_container_width=True)
        else:
            st.caption("No correlations available.")

