from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.config import config
from src.ui.context import AppContext
from src.ui.data import get_db_health
from src.ui.state import DashboardState, Watchlists


def _parse_list(text: str) -> list[str]:
    items = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        items.append(s)
    return items


def render(ctx: AppContext) -> None:
    st.title("Settings")
    st.caption("Configuration, health checks, and personal preferences.")

    st.markdown("### Configuration")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Alpha Vantage**")
        st.write("Status:", "Configured" if config.ALPHAVANTAGE_API_KEY else "Not configured")
    with c2:
        st.write("**OpenAI**")
        st.write("Status:", "Configured" if config.OPENAI_API_KEY else "Not configured")
        st.write("Model:", config.OPENAI_MODEL)

    st.divider()

    st.markdown("### Database health")
    health = get_db_health()
    if health.get("is_healthy"):
        st.success("Database healthy")
    else:
        st.warning("Database health check failed")
    st.json(health)

    st.divider()

    st.markdown("### Preferences")
    st.caption("These preferences are stored locally and restored on startup.")

    ctx.state.universe = st.text_input("Default universe", value=ctx.state.universe).strip().upper() or ctx.state.universe
    ctx.state.lookback = int(
        st.selectbox("Default lookback", options=[21, 63, 126, 252, 504], index=[21, 63, 126, 252, 504].index(ctx.state.lookback) if ctx.state.lookback in [21, 63, 126, 252, 504] else 1)
    )

    st.markdown("### Watchlists")
    w1, w2 = st.columns(2)
    with w1:
        factors_text = st.text_area(
            "Factor watchlist (one per line)",
            value="\n".join(ctx.state.watchlists.factors),
            height=160,
        )
    with w2:
        tickers_text = st.text_area(
            "Ticker watchlist (one per line)",
            value="\n".join(ctx.state.watchlists.tickers),
            height=160,
        )

    ctx.state.watchlists = Watchlists(
        factors=_parse_list(factors_text),
        tickers=_parse_list(tickers_text),
    )

    st.divider()

    st.markdown("### Reset")
    if st.button("Reset dashboard state", type="secondary"):
        try:
            ctx.paths.state_path.unlink(missing_ok=True)
        except Exception:
            pass
        for key in ("page", "universe", "lookback", "as_of"):
            st.session_state.pop(key, None)
        default = DashboardState()
        ctx.state.last_page = default.last_page
        ctx.state.universe = default.universe
        ctx.state.as_of = default.as_of
        ctx.state.lookback = default.lookback
        ctx.state.selected_factor = default.selected_factor
        ctx.state.watchlists = default.watchlists
        ctx.state.saved_views = default.saved_views
        st.success("State reset.")
        st.rerun()
