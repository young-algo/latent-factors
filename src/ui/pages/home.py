from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from src.decision_synthesizer import ActionCategory, DecisionSynthesizer
from src.regime_detection import RegimeDetector
from src.trading_signals import FactorMomentumAnalyzer
from src.ui.components.cards import badge, status_kind_from_threshold
from src.ui.components.empty_states import missing_data
from src.ui.components.tables import dataframe
from src.ui.context import AppContext
from src.ui.plotly_theme import apply_light_theme
from src.ui.view_models import (
    briefing_markdown,
    compute_portfolio_vitals,
    compute_top_movers,
    recommendations_to_json,
    resolve_as_of,
    validate_portfolio_inputs,
)


@st.cache_resource(ttl=3600)
def _fit_regime_detector(returns: pd.DataFrame) -> RegimeDetector:
    detector = RegimeDetector(returns)
    detector.fit_hmm(n_regimes=4)
    return detector


def render(ctx: AppContext) -> None:
    st.title("Equity Factors Dashboard")
    st.caption("Daily insights and research workflows for factor-based equity strategies.")

    factor_returns = ctx.data.factor_returns
    factor_loadings = ctx.data.factor_loadings
    factor_names = ctx.data.factor_names or {}

    # If names are exposure-derived, prompt to generate thematic LLM names.
    if factor_names:
        fallback = [
            f
            for f, name in factor_names.items()
            if isinstance(name, str) and f in name and "— L:" in name
        ]
        if len(fallback) >= max(3, int(0.5 * len(factor_names))):
            st.info(
                "Factor names are exposure-derived (tickers). "
                "Generate thematic names in **Research → LLM factor names**."
            )
            if st.button("Go to Research"):
                st.session_state["page"] = "Research"
                ctx.state.last_page = "Research"
                st.rerun()

    if factor_returns is None or factor_returns.empty:
        missing_data(
            title="Factor data not available",
            message="`factor_returns.csv` was not found or could not be loaded.",
            steps=(
                "Run discovery in **Research** (or `python -m src discover ...`).",
                "Ensure `factor_returns.csv` and `factor_loadings.csv` exist in the project root.",
            ),
        )
        return

    as_of = resolve_as_of(factor_returns, ctx.state.as_of)
    returns = factor_returns.loc[factor_returns.index <= as_of]

    # ---------------------------------------------------------------------
    # Top row: situational awareness
    # ---------------------------------------------------------------------
    portfolio_status = validate_portfolio_inputs(
        ctx.data.portfolio_returns, ctx.data.benchmark_returns, min_rows=63
    )
    vitals = compute_portfolio_vitals(
        factor_returns=returns,
        portfolio_returns=ctx.data.portfolio_returns,
        benchmark_returns=ctx.data.benchmark_returns,
        lookback=min(ctx.state.lookback, len(returns)),
    )

    regime_name = "Unknown"
    regime_conf = None
    regime_changed = False
    regime_probs: dict[str, float] = {}
    try:
        detector = _fit_regime_detector(returns)
        current = detector.detect_current_regime(as_of=as_of)
        regime_name = current.regime.value.replace("_", " ").title()
        regime_conf = float(current.probability)
        probs = detector.get_regime_probabilities(as_of=as_of)
        regime_probs = {k.value.replace("_", " ").title(): float(v) for k, v in probs.items()}

        if detector.regime_history is not None and len(detector.regime_history) >= 2:
            last = detector.regime_history["regime"].iloc[-1]
            prev = detector.regime_history["regime"].iloc[-2]
            regime_changed = bool(last != prev)
    except Exception:
        detector = None

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        conf_txt = f"{regime_conf:.0%}" if regime_conf is not None else "—"
        st.metric("Regime", regime_name, conf_txt)
        badge("Changed" if regime_changed else "Stable", "watch" if regime_changed else "ok")

    with c2:
        te = vitals.tracking_error
        st.metric("Tracking error", f"{te:.2%}" if te is not None else "—")
        kind = status_kind_from_threshold(te, ok_max_abs=0.06, watch_max_abs=0.08)
        badge("Target 4–6%" if te is not None else "Need data", kind if te is not None else "neutral")

    with c3:
        beta = vitals.beta
        st.metric("Beta", f"{beta:.2f}" if beta is not None else "—")
        kind = status_kind_from_threshold(beta, ok_max_abs=0.10, watch_max_abs=0.20)
        badge("Market neutral" if beta is not None else "Need data", kind if beta is not None else "neutral")

    with c4:
        ir = vitals.info_ratio
        st.metric("Info ratio", f"{ir:.2f}" if ir is not None else "—")
        badge("Strong" if (ir is not None and ir > 0.5) else "Developing" if (ir is not None and ir > 0) else "Need data", "ok" if (ir is not None and ir > 0.5) else "watch" if (ir is not None and ir > 0) else "neutral")

    with c5:
        unexplained = vitals.unexplained_risk_ratio
        st.metric("Unexplained risk", f"{unexplained:.0%}" if unexplained is not None else "—")
        badge("Clean attribution" if (unexplained is not None and unexplained < 0.30) else "Review model" if unexplained is not None else "Need data", "ok" if (unexplained is not None and unexplained < 0.30) else "watch" if unexplained is not None else "neutral")

    with c6:
        latest = returns.index.max()
        st.metric("Data as-of", latest.strftime("%Y-%m-%d"))
        if portfolio_status.overall_ok:
            badge(
                f"Portfolio overlap: {portfolio_status.overlap_rows} rows",
                "ok",
            )
        else:
            badge("Portfolio inputs incomplete", "neutral")

    st.divider()

    # ---------------------------------------------------------------------
    # Main panels
    # ---------------------------------------------------------------------
    left, right = st.columns([1.4, 1.0])

    with left:
        st.subheader("Today's actions")

        synthesizer = DecisionSynthesizer()
        loadings_for_signals = (
            factor_loadings if factor_loadings is not None else pd.DataFrame(index=[], columns=returns.columns)
        )
        state = synthesizer.collect_all_signals(
            factor_returns=returns,
            factor_loadings=loadings_for_signals,
            factor_names=factor_names,
        )
        state.date = datetime.combine(as_of.date(), datetime.min.time())
        recs = synthesizer.generate_recommendations(state)

        # Alignment summary (1–10) using the same scorer as the synthesizer.
        from src.decision_synthesizer import ConvictionScorer

        scorer = ConvictionScorer()
        bullish_regimes = {"Low-Vol Bull", "High-Vol Bull"}
        regime_bullish = state.regime.name in bullish_regimes
        momentum_bullish = [fm.return_7d > 0 for fm in state.factor_momentum]
        cross_bullish = state.cross_sectional_spread > 1.0
        alignment = scorer.calculate_alignment(regime_bullish, momentum_bullish, cross_bullish)
        st.metric("Signal alignment", f"{alignment}/10")

        if recs:
            grouped: dict[ActionCategory, list[Any]] = {}
            for rec in recs:
                grouped.setdefault(rec.category, []).append(rec)

            for category in (ActionCategory.OPPORTUNISTIC, ActionCategory.WEEKLY_REBALANCE, ActionCategory.WATCH):
                items = grouped.get(category, [])
                if not items:
                    continue
                st.markdown(f"#### {category.value.title().replace('_', ' ')}")
                for rec in items:
                    title = f"{rec.action} — {rec.conviction.value} ({rec.conviction_score}/10)"
                    with st.expander(title):
                        st.markdown("**Why**")
                        for reason in rec.reasons[:4]:
                            st.markdown(f"- {reason}")

                        if rec.conflicts:
                            st.markdown("**Conflicts**")
                            for conflict in rec.conflicts[:3]:
                                st.markdown(f"- {conflict}")

                        if rec.expressions:
                            st.markdown("**Suggested expressions**")
                            for expr in rec.expressions[:3]:
                                st.markdown(f"- {expr.description}: {expr.trade} ({expr.size_pct:.0%})")

                        st.markdown(f"**Exit**: {rec.exit_trigger}")
        else:
            st.info("No actionable recommendations at this time.")

        md = briefing_markdown(state, recs)
        js = recommendations_to_json(state, recs)
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "Download briefing (Markdown)",
                data=md,
                file_name=f"morning_briefing_{as_of.strftime('%Y%m%d')}.md",
                mime="text/markdown",
            )
        with dl2:
            st.download_button(
                "Download briefing (JSON)",
                data=js,
                file_name=f"morning_briefing_{as_of.strftime('%Y%m%d')}.json",
                mime="application/json",
            )

    with right:
        st.subheader("Alerts & movers")

        # Extremes
        try:
            analyzer = FactorMomentumAnalyzer(returns)
            alerts = analyzer.get_all_extreme_alerts(date=as_of, z_threshold=2.0, percentile_threshold=95.0)
        except Exception:
            alerts = []

        if alerts:
            rows = []
            for alert in sorted(alerts, key=lambda a: abs(float(a.z_score)), reverse=True)[:8]:
                rows.append(
                    {
                        "Factor": alert.factor_name,
                        "Name": factor_names.get(alert.factor_name, alert.factor_name),
                        "Z": float(alert.z_score),
                        "Percentile": float(alert.percentile) / 100.0,
                        "Direction": "High" if alert.direction == "extreme_high" else "Low",
                    }
                )
            st.markdown("**Extremes**")
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Percentile": st.column_config.NumberColumn(format="%.0%"),
                    "Z": st.column_config.NumberColumn(format="%.2f"),
                },
            )
        else:
            st.caption("No extreme alerts detected at current thresholds.")

        # Factor QA flags (optional)
        if ctx.data.factor_qa_report:
            rep = ctx.data.factor_qa_report
            high_corr = rep.get("high_corr_factors") or []
            high_r2 = rep.get("high_r2_factors") or []
            st.markdown("**Factor QA**")
            st.write(f"- High correlation factors: {len(high_corr)}")
            st.write(f"- High R² factors: {len(high_r2)}")

        # Top movers
        movers = compute_top_movers(returns, factor_names, as_of, n=10)
        if not movers.empty:
            st.markdown("**Top movers**")
            selection = dataframe(
                movers,
                key="home_movers",
                height=280,
                selection_mode="single-row",
            )
            try:
                selected_rows = getattr(selection, "selection", None)
                if selected_rows and selected_rows.rows:
                    row = int(selected_rows.rows[0])
                    ctx.state.selected_factor = str(movers.iloc[row]["Factor"])
            except Exception:
                pass
        else:
            st.caption("No mover data available.")

        # Regime probability chart
        if regime_probs:
            st.markdown("**Regime probabilities**")
            import plotly.graph_objects as go

            items = sorted(regime_probs.items(), key=lambda kv: kv[1])
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h", marker_color="#2563EB"))
            fig.update_layout(height=260, xaxis_tickformat=".0%")
            st.plotly_chart(apply_light_theme(fig), use_container_width=True)
