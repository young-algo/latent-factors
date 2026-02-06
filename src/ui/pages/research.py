from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from src.alphavantage_system import DataBackend
from src.config import config
from src.discover_and_label import run_discovery
from src.factor_labeler import batch_name_factors
from src.factor_optimization import SharpeOptimizer
from src.ui.components.empty_states import missing_data
from src.ui.context import AppContext
from src.ui.data import load_fundamentals_from_cache


def _write_log(label: str, text: str) -> None:
    st.markdown(f"**{label}**")
    st.text_area(label, value=text, height=220)


def render(ctx: AppContext) -> None:
    st.title("Research")
    st.caption("Run discovery, QA, optimization, backtests, and (optional) LLM enrichment.")

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    st.markdown("### Discovery")
    st.caption("Generates `factor_returns.csv` and `factor_loadings.csv` artifacts used by the dashboard.")

    with st.form("discovery_form"):
        symbols = st.text_input("Universe symbols (comma-separated)", value=ctx.state.universe)
        start = st.text_input("Start date (YYYY-MM-DD)", value=config.DEFAULT_START_DATE)
        method = st.selectbox("Method", options=["PCA", "ICA", "NMF", "AE"], index=0)
        k = st.slider("Number of factors (k)", min_value=3, max_value=25, value=10, step=1)
        rolling = st.number_input("Rolling window (days, 0 = static)", min_value=0, max_value=2520, value=0, step=21)
        use_llm_default = bool(config.OPENAI_API_KEY)
        use_llm = st.checkbox("LLM naming (requires OPENAI_API_KEY)", value=use_llm_default)
        submitted = st.form_submit_button("Run discovery")

    if submitted:
        if not config.ALPHAVANTAGE_API_KEY:
            st.error("ALPHAVANTAGE_API_KEY is required to run discovery.")
        else:
            if use_llm and not config.OPENAI_API_KEY:
                st.error("OPENAI_API_KEY is required when LLM naming is enabled.")
            else:
                with st.spinner("Running discovery… This can take several minutes."):
                    try:
                        run_discovery(
                            symbols=symbols,
                            start_date=start,
                            method=method,
                            k=int(k),
                            rolling=int(rolling),
                            name_out="factor_names.csv",
                            use_llm_naming=bool(use_llm),
                        )
                        st.success("Discovery complete. Refresh the dashboard to load the new artifacts.")
                        st.cache_data.clear()
                    except Exception as exc:
                        st.error(f"Discovery failed: {exc}")

    st.divider()

    # ------------------------------------------------------------------
    # Factor QA
    # ------------------------------------------------------------------
    st.markdown("### QA")
    st.caption("Runs residualization/leakage checks and writes `factor_qa_*` artifacts.")

    qa_col1, qa_col2 = st.columns([1, 1])
    with qa_col1:
        corr_thr = st.slider("Corr-to-SPY threshold", 0.1, 0.95, 0.70, 0.05)
    with qa_col2:
        r2_thr = st.slider("Regression R² threshold", 0.1, 0.95, 0.70, 0.05)

    if st.button("Run factor QA"):
        cmd = [
            sys.executable,
            str(Path(ctx.paths.project_root) / "scripts" / "factor_qa.py"),
            "--factor-returns",
            str(Path(ctx.paths.project_root) / "factor_returns.csv"),
            "--output-prefix",
            "factor_qa",
            "--corr-threshold",
            str(float(corr_thr)),
            "--r2-threshold",
            str(float(r2_thr)),
            "--db-path",
            str(ctx.paths.db_path),
        ]
        if (Path(ctx.paths.project_root) / "benchmark_returns.csv").exists():
            cmd.extend(["--benchmark-returns", str(Path(ctx.paths.project_root) / "benchmark_returns.csv")])

        with st.spinner("Running QA…"):
            try:
                proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
                st.success("QA complete. Outputs written with prefix `factor_qa_`.")
                if proc.stdout:
                    _write_log("QA output", proc.stdout)
                if proc.stderr:
                    _write_log("QA warnings/errors", proc.stderr)
                st.cache_data.clear()
            except subprocess.CalledProcessError as exc:
                st.error("QA failed.")
                if exc.stdout:
                    _write_log("QA output", exc.stdout)
                if exc.stderr:
                    _write_log("QA errors", exc.stderr)

    st.divider()

    # ------------------------------------------------------------------
    # Optimization / backtest (walk-forward)
    # ------------------------------------------------------------------
    st.markdown("### Optimization & walk-forward backtest")
    st.caption("Runs a walk-forward optimization on factor returns (research-only).")

    if ctx.data.factor_returns is None or ctx.data.factor_loadings is None:
        missing_data(
            title="Factor artifacts not available",
            message="Optimization requires `factor_returns.csv` and `factor_loadings.csv`.",
        )
        return

    returns = ctx.data.factor_returns
    loadings = ctx.data.factor_loadings

    with st.form("wf_opt_form"):
        train_window = st.select_slider("Train window (days)", options=[63, 126, 252, 504], value=126)
        test_window = st.select_slider("Test window (days)", options=[21, 42, 63], value=21)
        methods = st.multiselect(
            "Methods",
            options=["sharpe", "momentum", "risk_parity", "min_variance", "equal", "pca"],
            default=["sharpe", "momentum", "risk_parity"],
        )
        run_wf = st.form_submit_button("Run walk-forward")

    if run_wf:
        with st.spinner("Running walk-forward optimization…"):
            try:
                optimizer = SharpeOptimizer(returns, loadings)
                wf = optimizer.walk_forward_optimize(
                    train_window=int(train_window),
                    test_window=int(test_window),
                    methods=list(methods),
                    technique="differential",
                    verbose=False,
                )
                st.session_state["wf_results"] = wf
                st.success("Walk-forward complete.")
            except Exception as exc:
                st.error(f"Walk-forward failed: {exc}")

    wf = st.session_state.get("wf_results")
    if wf is not None and isinstance(wf, pd.DataFrame) and not wf.empty:
        st.markdown("#### Results")
        st.dataframe(wf, use_container_width=True, height=320)
        st.markdown("#### Summary")
        avg_train = float(wf["train_sharpe"].mean()) if "train_sharpe" in wf.columns else float("nan")
        avg_test = float(wf["test_sharpe"].mean()) if "test_sharpe" in wf.columns else float("nan")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Avg train Sharpe", f"{avg_train:.2f}")
        with c2:
            st.metric("Avg test Sharpe", f"{avg_test:.2f}")
        with c3:
            st.metric("Windows", f"{len(wf)}")

        st.download_button(
            "Download walk-forward CSV",
            data=wf.to_csv(index=False),
            file_name="walk_forward_results.csv",
            mime="text/csv",
        )

    st.divider()

    # ------------------------------------------------------------------
    # Fundamentals + LLM naming (optional)
    # ------------------------------------------------------------------
    st.markdown("### Enrichment (optional)")
    st.caption("Fetch fundamentals for attribution and run LLM factor naming (research only).")

    if not config.ALPHAVANTAGE_API_KEY:
        st.info("Set ALPHAVANTAGE_API_KEY to fetch fundamentals.")
        return

    if ctx.data.factor_loadings is None or ctx.data.factor_loadings.empty:
        st.info("Factor loadings are required for enrichment.")
        return

    factor = ctx.state.selected_factor or (ctx.data.factor_loadings.columns[0] if len(ctx.data.factor_loadings.columns) else None)
    if factor is None:
        return

    top_n = st.slider("Top exposures per side", min_value=5, max_value=25, value=10, step=1)
    tickers = (
        ctx.data.factor_loadings[factor].nlargest(top_n).index.tolist()
        + ctx.data.factor_loadings[factor].nsmallest(top_n).index.tolist()
    )

    if st.button("Fetch fundamentals for selected factor tickers"):
        with st.spinner("Fetching fundamentals (cached)…"):
            backend = DataBackend(api_key=config.ALPHAVANTAGE_API_KEY, db_path=str(ctx.paths.db_path))
            fnd = backend.get_fundamentals(tickers, fields=["Name", "Sector", "Industry", "MarketCapitalization", "PERatio", "Beta"])
            st.success(f"Fetched fundamentals for {len(fnd)} tickers.")
            st.dataframe(fnd.head(20), use_container_width=True)
            st.cache_data.clear()

    if config.OPENAI_API_KEY:
        st.markdown("#### LLM factor names (thematic)")
        st.caption("Generates thematic factor names from top long/short exposures and cached fundamentals.")

        with st.form("llm_naming_form"):
            name_scope = st.radio(
                "Scope",
                options=["All factors", "Selected factor"],
                horizontal=True,
            )
            fetch_missing_fundamentals = st.checkbox(
                "Fetch missing fundamentals (Alpha Vantage)",
                value=True,
                help="Recommended unless you already have fundamentals cached for the tickers used in exposures.",
            )
            force_refresh_fundamentals = st.checkbox(
                "Force refresh fundamentals",
                value=False,
                help="Bypasses cache and re-fetches fundamentals (slower; uses API quota).",
            )
            submitted_naming = st.form_submit_button("Generate LLM names")

        if submitted_naming:
            with st.spinner("Preparing fundamentals and naming factors…"):
                if ctx.data.factor_returns is None or ctx.data.factor_returns.empty:
                    st.error("Factor returns are required for naming.")
                    return

                if ctx.data.factor_loadings is None or ctx.data.factor_loadings.empty:
                    st.error("Factor loadings are required for naming.")
                    return

                factor_exposures = ctx.data.factor_loadings
                factor_ids = list(factor_exposures.columns)
                if name_scope == "Selected factor":
                    if factor not in factor_ids:
                        st.error("Selected factor not present in loadings.")
                        return
                    factor_ids = [factor]

                all_tickers: set[str] = set()
                for fac in factor_ids:
                    s = factor_exposures[fac].dropna()
                    all_tickers.update(s.nlargest(top_n).index)
                    all_tickers.update(s.nsmallest(top_n).index)

                tickers_list = sorted({str(t).upper() for t in all_tickers if t})
                if not tickers_list:
                    st.error("No tickers found in top exposures.")
                    return

                fields = ["Name", "Sector", "Industry", "MarketCapitalization", "PERatio", "Beta", "Description"]

                fundamentals = load_fundamentals_from_cache(
                    tickers_list,
                    db_path=str(ctx.paths.db_path),
                    fields=fields,
                )

                if fetch_missing_fundamentals and (fundamentals.empty or len(fundamentals) < len(tickers_list) * 0.5):
                    backend = DataBackend(api_key=config.ALPHAVANTAGE_API_KEY, db_path=str(ctx.paths.db_path))
                    fetched = backend.get_fundamentals(
                        tickers_list,
                        fields=fields,
                        force_refresh=bool(force_refresh_fundamentals),
                    )
                    # Prefer freshly fetched rows, but keep any cached rows too.
                    if not fetched.empty:
                        fundamentals = pd.concat([fundamentals, fetched]).groupby(level=0).last()

                if fundamentals.empty:
                    st.error("No fundamentals available. Fetch fundamentals first (or enable fetching).")
                    return

                exposures_subset = factor_exposures[factor_ids]
                names = batch_name_factors(
                    factor_exposures=exposures_subset,
                    fundamentals=fundamentals,
                    factor_returns=ctx.data.factor_returns,
                    top_n=int(top_n),
                    model=config.OPENAI_MODEL,
                )

                short_names = {k: v.short_name for k, v in names.items()}
                full_names = {k: v.to_dict() for k, v in names.items()}

                (Path(ctx.paths.project_root) / "factor_names.json").write_text(
                    json.dumps(short_names, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                full_path = Path(ctx.paths.project_root) / "data" / "factor_names_full.json"
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(
                    json.dumps(full_names, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                pd.DataFrame(
                    [{"factor": k, "name": v} for k, v in short_names.items()]
                ).to_csv(Path(ctx.paths.project_root) / "factor_names.csv", index=False)

                st.success("Saved factor names. Home/Factors pages will now show thematic names.")
                preview = (
                    pd.DataFrame.from_dict(full_names, orient="index")
                    .reset_index()
                    .rename(columns={"index": "factor"})
                    .loc[:, ["factor", "short_name", "theme", "confidence", "quality_score"]]
                )
                st.dataframe(preview, use_container_width=True, height=320)

                st.cache_data.clear()
                st.rerun()
    else:
        st.caption("Set OPENAI_API_KEY to enable LLM naming.")
