from __future__ import annotations

from dataclasses import asdict
from datetime import date
from pathlib import Path

import streamlit as st

from src.database import get_database_path
from src.ui import data as data_loader
from src.ui.context import AppContext, AppData, AppPaths
from src.ui.state import DashboardState, load_state, save_state, state_equals
from src.ui.view_models import compute_factor_display_names, resolve_as_of, validate_portfolio_inputs


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


PAGES = {
    "Home": "src.ui.pages.home",
    "Factors": "src.ui.pages.factors",
    "Signals": "src.ui.pages.signals",
    "Portfolio & Risk": "src.ui.pages.portfolio_risk",
    "Research": "src.ui.pages.research",
    "Settings": "src.ui.pages.settings",
}


def _init_session_state(state: DashboardState) -> None:
    st.session_state.setdefault("page", state.last_page)
    st.session_state.setdefault("universe", state.universe)
    st.session_state.setdefault("lookback", state.lookback)
    try:
        st.session_state.setdefault("as_of", date.fromisoformat(state.as_of))
    except Exception:
        st.session_state.setdefault("as_of", date.today())


def _infer_universe_options(state: DashboardState, analysis_cfg: dict | None) -> list[str]:
    options = {"VTHR", "SPY", "IWM", state.universe}
    if analysis_cfg:
        symbols = analysis_cfg.get("symbols") or analysis_cfg.get("symbol") or ""
        if isinstance(symbols, str):
            for s in symbols.split(","):
                if s.strip():
                    options.add(s.strip().upper())
    return sorted(options)


def _build_context(state: DashboardState) -> AppContext:
    root = _project_root()
    paths = AppPaths(
        project_root=root,
        state_path=root / "data" / "dashboard_state.json",
        db_path=get_database_path(),
    )

    factor_returns = data_loader.load_factor_returns(str(root / "factor_returns.csv"))
    factor_loadings = data_loader.load_factor_loadings(str(root / "factor_loadings.csv"))
    raw_factor_names = data_loader.load_factor_names(str(root))
    factor_names = compute_factor_display_names(factor_loadings, raw_factor_names)
    portfolio_returns, benchmark_returns = data_loader.load_portfolio_inputs(str(root))
    basket = data_loader.load_latest_trade_basket(str(root))
    qa_report = data_loader.load_factor_qa_report(str(root))
    analysis_cfg = data_loader.load_factor_analysis_config(str(root))

    data = AppData(
        factor_returns=factor_returns,
        factor_loadings=factor_loadings,
        factor_names=factor_names,
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        latest_trade_basket=basket,
        factor_qa_report=qa_report,
        factor_analysis_config=analysis_cfg,
    )

    # Clamp as-of to available data so pages don't error on stale state.
    if data.factor_returns is not None and not data.factor_returns.empty:
        resolved = resolve_as_of(data.factor_returns, state.as_of)
        state.as_of = resolved.date().isoformat()

    return AppContext(paths=paths, state=state, data=data)


def _render_sidebar(ctx: AppContext) -> None:
    st.sidebar.title("Equity Factors Dashboard")

    cfg = ctx.data.factor_analysis_config
    if cfg:
        symbols = cfg.get("symbols")
        method = cfg.get("method")
        k = cfg.get("k")
        if symbols:
            st.sidebar.caption(f"Active artifacts: {symbols} | {method} | k={k}")
        else:
            st.sidebar.caption("Active artifacts loaded from project root.")
    else:
        st.sidebar.caption("Active artifacts loaded from project root.")

    st.sidebar.divider()

    st.sidebar.markdown("**Global controls**")

    universe_options = _infer_universe_options(ctx.state, cfg)
    ctx.state.universe = st.sidebar.selectbox("Universe", options=universe_options, key="universe")

    if ctx.data.factor_returns is not None and not ctx.data.factor_returns.empty:
        lo = ctx.data.factor_returns.index.min().date()
        hi = ctx.data.factor_returns.index.max().date()
        st.session_state["as_of"] = min(max(st.session_state["as_of"], lo), hi)
        selected = st.sidebar.date_input("As-of date", key="as_of", min_value=lo, max_value=hi)
        ctx.state.as_of = selected.isoformat()
    else:
        selected = st.sidebar.date_input("As-of date", key="as_of")
        ctx.state.as_of = selected.isoformat()

    lookback_opts = [21, 63, 126, 252, 504]
    if ctx.state.lookback not in lookback_opts:
        ctx.state.lookback = 63
    ctx.state.lookback = st.sidebar.selectbox("Lookback window", options=lookback_opts, key="lookback")

    # Portfolio input status
    status = validate_portfolio_inputs(ctx.data.portfolio_returns, ctx.data.benchmark_returns, min_rows=63)
    st.sidebar.markdown("**Portfolio inputs**")
    if status.overall_ok:
        st.sidebar.success(f"Ready ({status.overlap_rows} rows)")
        st.sidebar.caption(f"{status.overlap_start.date()} → {status.overlap_end.date()}")
    else:
        st.sidebar.info("Not configured")
        for msg in status.messages[:3]:
            st.sidebar.caption(f"- {msg}")

    st.sidebar.divider()

    if st.sidebar.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown("**Navigation**")
    page_names = list(PAGES.keys())
    if ctx.state.last_page not in page_names:
        ctx.state.last_page = "Home"
    selected_page = st.sidebar.radio("Page", options=page_names, key="page")
    ctx.state.last_page = selected_page


def render_app() -> None:
    root = _project_root()
    state_path = root / "data" / "dashboard_state.json"
    loaded_state = load_state(state_path)
    original_state = DashboardState.from_mapping(asdict(loaded_state))
    _init_session_state(loaded_state)

    ctx = _build_context(loaded_state)
    _render_sidebar(ctx)

    # Route to selected page.
    module_path = PAGES.get(ctx.state.last_page, PAGES["Home"])
    module = __import__(module_path, fromlist=["render"])
    module.render(ctx)

    # Persist state if changed.
    if not state_equals(original_state, ctx.state):
        save_state(ctx.state, state_path)
