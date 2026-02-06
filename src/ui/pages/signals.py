from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import streamlit as st

from src.regime_detection import RegimeDetector
from src.signal_aggregator import MetaModelAggregator
from src.trading_signals import FactorMomentumAnalyzer
from src.ui.components.empty_states import missing_data
from src.ui.context import AppContext
from src.ui.plotly_theme import apply_light_theme
from src.ui.view_models import resolve_as_of


@st.cache_resource(ttl=3600)
def _fit_regime_detector(returns: pd.DataFrame) -> RegimeDetector:
    detector = RegimeDetector(returns)
    detector.fit_hmm(n_regimes=4)
    return detector


def render(ctx: AppContext) -> None:
    st.title("Signals")
    st.caption("Regime, momentum, extremes, and meta-model consensus.")

    factor_returns = ctx.data.factor_returns
    factor_names = ctx.data.factor_names or {}

    if factor_returns is None or factor_returns.empty:
        missing_data(
            title="Factor returns not available",
            message="This page requires `factor_returns.csv` in the project root.",
            steps=("Run discovery in **Research**.",),
        )
        return

    as_of = resolve_as_of(factor_returns, ctx.state.as_of)
    returns = factor_returns.loc[factor_returns.index <= as_of]

    tabs = st.tabs(["Regime", "Momentum", "Extremes", "Meta-model"])

    # ------------------------------------------------------------------
    # Regime
    # ------------------------------------------------------------------
    with tabs[0]:
        st.subheader("Regime")
        try:
            detector = _fit_regime_detector(returns)
            current = detector.detect_current_regime(as_of=as_of)
            probs = detector.get_regime_probabilities(as_of=as_of)
            prob_map = {k.value.replace("_", " ").title(): float(v) for k, v in probs.items()}

            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Regime", current.regime.value.replace("_", " ").title(), f"{current.probability:.0%}")
                st.caption(current.description)
                st.write(f"Volatility (recent): {current.volatility:.2%}")
                st.write(f"Trend (recent): {current.trend:.3%}")

            with c2:
                import plotly.graph_objects as go

                items = sorted(prob_map.items(), key=lambda kv: kv[1])
                fig = go.Figure()
                fig.add_trace(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h", marker_color="#2563EB"))
                fig.update_layout(height=320, xaxis_tickformat=".0%")
                st.plotly_chart(apply_light_theme(fig), use_container_width=True)
        except Exception as exc:
            st.warning(f"Regime detection unavailable: {exc}")

    # ------------------------------------------------------------------
    # Momentum
    # ------------------------------------------------------------------
    with tabs[1]:
        st.subheader("Momentum")
        analyzer = FactorMomentumAnalyzer(returns)

        watch_only = st.checkbox("Watchlist factors only", value=bool(ctx.state.watchlists.factors))
        if watch_only and ctx.state.watchlists.factors:
            candidates = [f for f in ctx.state.watchlists.factors if f in returns.columns]
        else:
            candidates = list(returns.columns)

        selected = st.multiselect(
            "Factors",
            options=candidates,
            default=candidates[: min(12, len(candidates))],
        )

        rows = []
        for factor in selected:
            try:
                sig = analyzer.get_momentum_signals(factor, date=as_of)
                alert = analyzer.check_extreme_levels(factor, date=as_of)
            except Exception:
                continue
            rows.append(
                {
                    "Factor": factor,
                    "Name": factor_names.get(factor, factor),
                    "RSI": float(sig["rsi"]),
                    "RSI signal": str(sig["rsi_signal"]),
                    "MACD": str(sig["macd_signal"]),
                    "ADX": float(sig["adx"]),
                    "Combined": str(sig["combined_signal"]),
                    "Extreme z": float(alert.z_score) if alert else None,
                }
            )

        frame = pd.DataFrame(rows)
        if not frame.empty:
            st.dataframe(
                frame,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "RSI": st.column_config.NumberColumn(format="%.1f"),
                    "ADX": st.column_config.NumberColumn(format="%.1f"),
                    "Extreme z": st.column_config.NumberColumn(format="%.2f"),
                },
            )
        else:
            st.info("No momentum data available for the selected factors.")

    # ------------------------------------------------------------------
    # Extremes
    # ------------------------------------------------------------------
    with tabs[2]:
        st.subheader("Extremes")
        analyzer = FactorMomentumAnalyzer(returns)

        z_thr = st.slider("Z-score threshold", min_value=1.0, max_value=4.0, value=2.0, step=0.25)
        pct_thr = st.slider("Percentile threshold", min_value=80.0, max_value=99.0, value=95.0, step=1.0)
        alerts = analyzer.get_all_extreme_alerts(date=as_of, z_threshold=z_thr, percentile_threshold=pct_thr)

        if alerts:
            rows = []
            for alert in sorted(alerts, key=lambda a: abs(float(a.z_score)), reverse=True):
                rows.append(
                    {
                        "Factor": alert.factor_name,
                        "Name": factor_names.get(alert.factor_name, alert.factor_name),
                        "Alert": alert.alert_type,
                        "Direction": "High" if alert.direction == "extreme_high" else "Low",
                        "Z": float(alert.z_score),
                        "Percentile": float(alert.percentile) / 100.0,
                        "Value": float(alert.current_value),
                        "Date": alert.timestamp.strftime("%Y-%m-%d"),
                    }
                )
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
            st.info("No extremes at current thresholds.")

    # ------------------------------------------------------------------
    # Meta-model
    # ------------------------------------------------------------------
    with tabs[3]:
        st.subheader("Meta-model consensus")
        st.caption("Walk-forward gradient boosting model combining multiple signal sources.")

        with st.form("meta_model_config"):
            min_training = st.select_slider(
                "Minimum training samples",
                options=[63, 126, 252, 504],
                value=252,
            )
            horizon = st.select_slider("Prediction horizon (days)", options=[1, 3, 5, 10, 21], value=5)

            adv = st.expander("Advanced", expanded=False)
            with adv:
                n_estimators = st.slider("n_estimators", 50, 500, 100, 50)
                max_depth = st.slider("max_depth", 2, 10, 3, 1)
                learning_rate = st.slider("learning_rate", 0.01, 0.30, 0.05, 0.01)
                subsample = st.slider("subsample", 0.5, 1.0, 0.8, 0.1)
                purge_gap = st.slider("purge_gap (days)", 0, 10, 5, 1)

            use_proxy_market = st.checkbox("Use factor mean as market proxy", value=True)
            submitted = st.form_submit_button("Train meta-model")

        if submitted:
            with st.spinner("Training meta-model (walk-forward)…"):
                params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    "subsample": subsample,
                }

                aggregator = MetaModelAggregator(
                    SimpleNamespace(),
                    model_params=params,
                    min_training_samples=int(min_training),
                    prediction_horizon=int(horizon),
                    use_voting_fallback=True,
                )

                market_proxy = returns.mean(axis=1) if use_proxy_market else returns.mean(axis=1)
                aggregator.set_market_returns(market_proxy)

                momentum = FactorMomentumAnalyzer(returns)
                aggregator.add_momentum_signals(momentum)
                try:
                    regime = _fit_regime_detector(returns)
                    aggregator.add_regime_signals(regime)
                except Exception:
                    pass

                aggregator.train_walk_forward(
                    min_window=int(min_training),
                    step_size=21,
                    purge_gap=int(purge_gap),
                    verbose=False,
                )

                st.session_state["meta_model"] = aggregator
                st.success("Meta-model training complete.")

        aggregator = st.session_state.get("meta_model")
        if aggregator is not None:
            st.divider()
            st.markdown("### Current prediction")
            if st.button("Generate meta-consensus"):
                with st.spinner("Generating prediction…"):
                    result = aggregator.generate_meta_consensus()
                prob_up = float(result.get("probability_up", 0.0))
                meta_score = float(result.get("meta_score", 0.0))
                signal = result.get("consensus_signal")

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("P(positive return)", f"{prob_up:.0%}")
                    st.metric("Meta score", f"{meta_score:.1f}")
                with c2:
                    if signal is not None:
                        st.write(f"**Direction:** {signal.consensus_direction.value}")
                        st.write(f"**Confidence:** {signal.confidence:.1f}%")
                        st.write(f"**Recommendation:** {signal.recommendation}")

            importance = aggregator.get_model_feature_importance()
            if importance is not None and not importance.empty:
                st.divider()
                st.markdown("### Feature importance")
                top = importance.head(15).copy()
                import plotly.graph_objects as go

                fig = go.Figure()
                fig.add_trace(go.Bar(x=top["importance"], y=[f"Feature {i}" for i in top["feature_idx"]], orientation="h", marker_color="#2563EB"))
                fig.update_layout(height=360)
                st.plotly_chart(apply_light_theme(fig), use_container_width=True)

