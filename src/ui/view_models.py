from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from math import sqrt
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.decision_synthesizer import (
    ActionCategory,
    Recommendation,
    SignalState,
)


def compute_factor_display_names(
    factor_loadings: Optional[pd.DataFrame],
    raw_names: dict[str, str],
    max_len: int = 64,
    top_n: int = 3,
) -> dict[str, str]:
    """
    Compute actionable factor display names.

    Rules:
    - Prefer a short, human-readable name from `raw_names` when it looks sane.
    - Otherwise fall back to an exposure-derived label (top longs/shorts).
    """

    def _looks_like_bad_name(name: str, factor_id: str) -> bool:
        cleaned = " ".join(str(name).split()).strip()
        if not cleaned:
            return True
        if cleaned == factor_id:
            return True
        # Artifact from earlier pipelines: "AAPL, MSFT, ... Negative: ..."
        if "negative:" in cleaned.lower():
            return True
        # Very long multi-ticker sentences tend to be unusable in a dashboard.
        if len(cleaned) > 120 and "," in cleaned:
            return True
        return False

    def _truncate(text: str) -> str:
        cleaned = " ".join(str(text).split()).strip()
        if len(cleaned) <= max_len:
            return cleaned
        return cleaned[: max_len - 1].rstrip() + "…"

    def _exposure_label(factor_id: str, load: pd.Series) -> str:
        series = pd.to_numeric(load, errors="coerce").dropna()
        if series.empty:
            return factor_id

        top_long = [str(t) for t in series.nlargest(top_n).index.tolist()]
        top_short = [str(t) for t in series.nsmallest(top_n).index.tolist()]

        long_str = "/".join(top_long) if top_long else "—"
        short_str = "/".join(top_short) if top_short else "—"
        label = f"{factor_id} — L: {long_str} | S: {short_str}"
        return _truncate(label)

    factor_ids: set[str] = set(raw_names.keys())
    if factor_loadings is not None and isinstance(factor_loadings, pd.DataFrame):
        factor_ids.update([str(c) for c in factor_loadings.columns])

    output: dict[str, str] = {}
    for factor_id in sorted(factor_ids):
        raw = raw_names.get(factor_id)
        if raw and not _looks_like_bad_name(raw, factor_id):
            output[factor_id] = _truncate(raw)
            continue

        if factor_loadings is not None and factor_id in factor_loadings.columns:
            output[factor_id] = _exposure_label(factor_id, factor_loadings[factor_id])
        else:
            output[factor_id] = factor_id

    return output


def truncate_as_of(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    ts = pd.Timestamp(as_of)
    clipped = df.loc[df.index <= ts]
    return clipped if not clipped.empty else df.iloc[:0]


def resolve_as_of(factor_returns: Optional[pd.DataFrame], as_of_str: str) -> pd.Timestamp:
    fallback = pd.Timestamp(date.today())
    try:
        requested = pd.Timestamp(as_of_str)
    except Exception:
        requested = fallback

    if factor_returns is None or factor_returns.empty:
        return requested

    lo = factor_returns.index.min()
    hi = factor_returns.index.max()
    if requested < lo:
        return lo
    if requested > hi:
        return hi
    return requested


@dataclass(frozen=True)
class PortfolioInputStatus:
    portfolio_ok: bool
    benchmark_ok: bool
    overlap_ok: bool
    overall_ok: bool
    overlap_rows: int = 0
    overlap_start: Optional[pd.Timestamp] = None
    overlap_end: Optional[pd.Timestamp] = None
    messages: tuple[str, ...] = ()


def validate_portfolio_inputs(
    portfolio_returns: Optional[pd.Series],
    benchmark_returns: Optional[pd.Series],
    min_rows: int = 63,
) -> PortfolioInputStatus:
    messages: list[str] = []

    portfolio_ok = isinstance(portfolio_returns, pd.Series) and not portfolio_returns.empty
    benchmark_ok = isinstance(benchmark_returns, pd.Series) and not benchmark_returns.empty

    overlap_ok = False
    overlap_rows = 0
    overlap_start = None
    overlap_end = None

    if not portfolio_ok:
        messages.append("Portfolio returns not found (expected portfolio_returns.csv).")
    if not benchmark_ok:
        messages.append("Benchmark returns not found (expected benchmark_returns.csv).")

    if portfolio_ok and benchmark_ok:
        merged = pd.concat(
            [
                portfolio_returns.rename("portfolio"),
                benchmark_returns.rename("benchmark"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        overlap_rows = int(len(merged))
        if not merged.empty:
            overlap_start = pd.Timestamp(merged.index.min())
            overlap_end = pd.Timestamp(merged.index.max())
        if len(merged) >= min_rows:
            overlap_ok = True
        else:
            messages.append(
                f"Portfolio/benchmark overlap is {len(merged)} rows; need >= {min_rows}."
            )

    overall_ok = portfolio_ok and benchmark_ok and overlap_ok
    return PortfolioInputStatus(
        portfolio_ok=portfolio_ok,
        benchmark_ok=benchmark_ok,
        overlap_ok=overlap_ok,
        overall_ok=overall_ok,
        overlap_rows=overlap_rows,
        overlap_start=overlap_start,
        overlap_end=overlap_end,
        messages=tuple(messages),
    )


@dataclass(frozen=True)
class PortfolioVitals:
    beta: Optional[float]
    tracking_error: Optional[float]
    info_ratio: Optional[float]
    active_risk: Optional[float]
    unexplained_risk_ratio: Optional[float]


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)
    except Exception:
        return None


def compute_portfolio_vitals(
    factor_returns: Optional[pd.DataFrame],
    portfolio_returns: Optional[pd.Series],
    benchmark_returns: Optional[pd.Series],
    lookback: int = 63,
) -> PortfolioVitals:
    if portfolio_returns is None or not isinstance(portfolio_returns, pd.Series):
        return PortfolioVitals(beta=None, tracking_error=None, info_ratio=None, active_risk=None, unexplained_risk_ratio=None)

    portfolio = pd.to_numeric(portfolio_returns, errors="coerce").dropna()
    if portfolio.empty:
        return PortfolioVitals(beta=None, tracking_error=None, info_ratio=None, active_risk=None, unexplained_risk_ratio=None)

    portfolio = portfolio.tail(lookback)

    beta = None
    tracking_error = None
    info_ratio = None
    active_risk = None

    if benchmark_returns is not None and isinstance(benchmark_returns, pd.Series) and not benchmark_returns.empty:
        bench = pd.to_numeric(benchmark_returns, errors="coerce").dropna()
        merged = pd.concat([portfolio.rename("portfolio"), bench.rename("benchmark")], axis=1, join="inner").dropna()
        merged = merged.tail(lookback)
        if not merged.empty:
            active = merged["portfolio"] - merged["benchmark"]
            denom = active.std(ddof=0)
            tracking_error = _safe_float(denom * sqrt(252))
            active_risk = tracking_error
            if denom and denom > 0:
                info_ratio = _safe_float((active.mean() / denom) * sqrt(252))

            market_var = merged["benchmark"].var(ddof=0)
            if market_var and market_var > 0:
                beta = _safe_float(merged["portfolio"].cov(merged["benchmark"]) / market_var)
    else:
        active_risk = _safe_float(portfolio.std(ddof=0) * sqrt(252))

    unexplained_ratio = None
    if factor_returns is not None and isinstance(factor_returns, pd.DataFrame) and not factor_returns.empty:
        factors = factor_returns.replace([np.inf, -np.inf], np.nan)
        factors = factors.loc[factors.index.intersection(portfolio.index)].dropna(how="all", axis=0)
        factors = factors.dropna(axis=1, how="all")
        merged = pd.concat([portfolio.rename("portfolio"), factors], axis=1).dropna()
        if len(merged) >= 10 and merged["portfolio"].var(ddof=0) > 0 and merged.shape[1] > 1:
            y = merged["portfolio"].to_numpy(dtype=float)
            X = merged.drop(columns=["portfolio"]).to_numpy(dtype=float)
            X = np.column_stack([np.ones(len(X)), X])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            residuals = y - X @ coef
            unexplained_ratio = _safe_float(np.var(residuals, ddof=0) / np.var(y, ddof=0))

    return PortfolioVitals(
        beta=beta,
        tracking_error=tracking_error,
        info_ratio=info_ratio,
        active_risk=active_risk,
        unexplained_risk_ratio=unexplained_ratio,
    )


@dataclass(frozen=True)
class FactorDNAMeta:
    leakage_basis: str  # "Benchmark" or "Market Proxy"


def compute_factor_dna_table(
    factor_returns: pd.DataFrame,
    factor_loadings: Optional[pd.DataFrame],
    factor_names: dict[str, str],
    as_of: pd.Timestamp,
    lookback: int = 63,
    benchmark_returns: Optional[pd.Series] = None,
) -> tuple[pd.DataFrame, FactorDNAMeta]:
    returns = truncate_as_of(factor_returns, as_of)
    recent = returns.tail(lookback)

    leakage_basis = "Market Proxy"
    benchmark_aligned = None
    if benchmark_returns is not None and isinstance(benchmark_returns, pd.Series) and not benchmark_returns.empty:
        bench = pd.to_numeric(benchmark_returns, errors="coerce").dropna()
        benchmark_aligned = bench.reindex(recent.index).dropna()
        if not benchmark_aligned.empty:
            leakage_basis = "Benchmark"

    if leakage_basis != "Benchmark":
        proxy = recent.mean(axis=1, skipna=True)
        benchmark_aligned = proxy

    rows: list[dict[str, Any]] = []
    for factor in returns.columns:
        series = pd.to_numeric(recent[factor], errors="coerce").dropna()
        if series.empty:
            continue

        one_d = float(series.iloc[-1])
        five_d = float((1.0 + series.tail(5)).prod() - 1.0) if len(series) >= 5 else float((1.0 + series).prod() - 1.0)
        one_m = float((1.0 + series.tail(21)).prod() - 1.0) if len(series) >= 21 else float((1.0 + series).prod() - 1.0)

        vol = float(series.std(ddof=0) * sqrt(252))
        sharpe = float((series.mean() * 252) / vol) if vol > 0 else 0.0

        corr = np.nan
        if benchmark_aligned is not None:
            aligned = pd.concat([series.rename("factor"), pd.Series(benchmark_aligned, index=recent.index).rename("bench")], axis=1).dropna()
            if len(aligned) >= 5:
                corr = float(aligned["factor"].corr(aligned["bench"]))

        purity = float(1.0 - abs(corr)) if not np.isnan(corr) else np.nan
        leakage_flag = bool(abs(corr) > 0.7) if not np.isnan(corr) else False

        crowding = np.nan
        top_ticker = None
        if factor_loadings is not None and isinstance(factor_loadings, pd.DataFrame) and factor in factor_loadings.columns:
            load = factor_loadings[factor].dropna()
            if not load.empty:
                top_ticker = str(load.abs().idxmax())
                abs_load = load.abs()
                top_n = max(1, int(len(abs_load) * 0.10))
                crowding = float(abs_load.nlargest(top_n).sum() / abs_load.sum()) if abs_load.sum() > 0 else np.nan

        rows.append(
            {
                "Factor": factor,
                "Name": factor_names.get(factor, factor),
                "1D": one_d,
                "5D": five_d,
                "1M": one_m,
                "Vol (ann.)": vol,
                "Sharpe (ann.)": sharpe,
                "Purity": purity,
                "Leakage Corr": corr,
                "Leakage Flag": leakage_flag,
                "Crowding": crowding,
                "Top Ticker": top_ticker,
            }
        )

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(["Leakage Flag", "Sharpe (ann.)"], ascending=[True, False]).reset_index(drop=True)

    return table, FactorDNAMeta(leakage_basis=leakage_basis)


def compute_top_movers(
    factor_returns: pd.DataFrame,
    factor_names: dict[str, str],
    as_of: pd.Timestamp,
    n: int = 8,
) -> pd.DataFrame:
    returns = truncate_as_of(factor_returns, as_of)
    if returns.empty:
        return pd.DataFrame()

    one_d = returns.iloc[-1].rename("1D")
    five_d = (1.0 + returns.tail(5)).prod() - 1.0
    five_d = five_d.rename("5D")
    frame = pd.concat([one_d, five_d], axis=1).dropna(how="all")
    frame["Name"] = [factor_names.get(idx, idx) for idx in frame.index]
    frame["Factor"] = frame.index
    frame = frame[["Factor", "Name", "1D", "5D"]]

    frame["abs_1d"] = frame["1D"].abs()
    frame = frame.sort_values("abs_1d", ascending=False).head(n).drop(columns=["abs_1d"])
    return frame.reset_index(drop=True)


def sector_tilt_from_fundamentals(
    loadings: pd.Series,
    fundamentals: pd.DataFrame,
    field: str = "Sector",
    top_n: int = 60,
) -> pd.DataFrame:
    if loadings is None or loadings.empty or fundamentals is None or fundamentals.empty:
        return pd.DataFrame()

    abs_load = loadings.abs().dropna().sort_values(ascending=False).head(top_n)
    if abs_load.empty:
        return pd.DataFrame()

    mapped = fundamentals.reindex(abs_load.index)
    sectors = mapped.get(field)
    if sectors is None:
        return pd.DataFrame()

    frame = pd.DataFrame({"weight": abs_load / abs_load.sum(), "sector": sectors})
    frame = frame.dropna(subset=["sector"])
    if frame.empty:
        return pd.DataFrame()

    grouped = frame.groupby("sector", as_index=False)["weight"].sum()
    grouped = grouped.sort_values("weight", ascending=False)
    grouped["weight"] = grouped["weight"].astype(float)
    return grouped


def recommendations_to_json(state: SignalState, recs: list[Recommendation]) -> str:
    def normalize(obj: Any) -> Any:
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if hasattr(obj, "value"):  # Enum
            return obj.value
        if hasattr(obj, "__dict__") and not isinstance(obj, dict):
            try:
                return {k: normalize(v) for k, v in asdict(obj).items()}
            except Exception:
                return str(obj)
        if isinstance(obj, dict):
            return {k: normalize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [normalize(v) for v in obj]
        return obj

    payload = {
        "state": normalize(state),
        "recommendations": [normalize(r) for r in recs],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def briefing_markdown(state: SignalState, recs: list[Recommendation]) -> str:
    lines: list[str] = []
    lines.append(f"# Morning Briefing — {state.date.strftime('%Y-%m-%d')}")
    lines.append("")

    lines.append("## Market regime")
    lines.append(f"- **Regime:** {state.regime.name}")
    lines.append(f"- **Confidence:** {state.regime.confidence:.0%}")
    lines.append(f"- **Trend:** {state.regime.trend}")
    lines.append("")

    lines.append("## Signal alignment")
    from src.decision_synthesizer import ConvictionScorer

    scorer = ConvictionScorer()
    bullish_regimes = {"Low-Vol Bull", "High-Vol Bull"}
    regime_bullish = state.regime.name in bullish_regimes
    momentum_bullish = [fm.return_7d > 0 for fm in state.factor_momentum]
    cross_bullish = state.cross_sectional_spread > 1.0
    alignment = scorer.calculate_alignment(regime_bullish, momentum_bullish, cross_bullish)
    lines.append(f"- **Alignment:** {alignment}/10")
    lines.append("")

    lines.append("## Recommendations")
    if not recs:
        lines.append("- No actionable recommendations.")
        lines.append("")
        return "\n".join(lines) + "\n"

    grouped: dict[ActionCategory, list[Recommendation]] = {}
    for rec in recs:
        grouped.setdefault(rec.category, []).append(rec)

    for category in (ActionCategory.OPPORTUNISTIC, ActionCategory.WEEKLY_REBALANCE, ActionCategory.WATCH):
        items = grouped.get(category, [])
        if not items:
            continue
        lines.append(f"### {category.value.title().replace('_', ' ')}")
        for rec in items:
            lines.append(f"- **{rec.action}** — {rec.conviction.value} ({rec.conviction_score}/10)")
            for reason in rec.reasons[:3]:
                lines.append(f"  - {reason}")
            if rec.conflicts:
                lines.append("  - Conflicts:")
                for conflict in rec.conflicts[:2]:
                    lines.append(f"    - {conflict}")
            if rec.expressions:
                lines.append("  - Suggested expressions:")
                for expr in rec.expressions[:2]:
                    lines.append(f"    - {expr.description}: {expr.trade} ({expr.size_pct:.0%})")
            lines.append(f"  - Exit: {rec.exit_trigger}")
        lines.append("")

    return "\n".join(lines) + "\n"
