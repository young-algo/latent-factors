#!/usr/bin/env python3
"""
Nightly factor QA checks.

Checks implemented:
1) Factor regression vs market/sector benchmarks (beta leakage check)
2) Factor correlation matrix diagnostics
3) Simple rolling stability snapshot on factor return behavior
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Ensure project root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.alphavantage_system import DataBackend
from src.config import config
from src.latent_factors import MARKET_ETF, SECTOR_ETFS


@dataclass
class FactorLeakageRow:
    factor: str
    observations: int
    corr_to_spy: float
    beta_to_spy: float
    regression_r2: float
    high_corr_flag: bool
    high_r2_flag: bool


def _load_factor_returns(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Factor returns file not found: {path}")

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df.sort_index().replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if df.empty:
        raise ValueError(f"No factor returns found in {path}")
    return df


def _load_or_fetch_benchmark_returns(
    factors: pd.DataFrame,
    benchmark_returns_path: Path | None,
    api_key: str,
    db_path: str,
) -> pd.DataFrame:
    if benchmark_returns_path is not None:
        bench = pd.read_csv(benchmark_returns_path, index_col=0, parse_dates=True)
        if not isinstance(bench.index, pd.DatetimeIndex):
            bench.index = pd.to_datetime(bench.index)
        return bench.sort_index()

    backend = DataBackend(api_key=api_key, db_path=db_path)
    benchmark_tickers = [MARKET_ETF] + SECTOR_ETFS
    prices = backend.get_prices(benchmark_tickers, end=factors.index.max().strftime("%Y-%m-%d"))
    if prices.empty:
        raise ValueError("No benchmark prices were fetched")

    returns = prices.pct_change(fill_method=None).dropna(how="all")
    available = [c for c in benchmark_tickers if c in returns.columns]
    if MARKET_ETF not in available:
        raise ValueError("SPY returns are required for leakage QA")
    return returns[available]


def _ols_r2(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(x_design, y, rcond=None)
    fitted = x_design @ coef
    residual = y - fitted

    y_var = np.var(y, ddof=1)
    if y_var <= 0:
        r2 = np.nan
    else:
        r2 = 1.0 - (np.var(residual, ddof=1) / y_var)

    return coef[1:], float(r2)


def _factor_stability_snapshot(factors: pd.DataFrame, window: int = 63) -> Dict[str, float]:
    if len(factors) < window * 2:
        return {f: np.nan for f in factors.columns}

    prev = factors.iloc[-2 * window : -window].reset_index(drop=True)
    curr = factors.iloc[-window:].reset_index(drop=True)

    output: Dict[str, float] = {}
    for col in factors.columns:
        output[col] = float(prev[col].corr(curr[col]))
    return output


def run_factor_qa(
    factor_returns: Path,
    output_prefix: Path,
    api_key: str,
    db_path: str,
    corr_threshold: float,
    r2_threshold: float,
    benchmark_returns_path: Path | None = None,
) -> Dict[str, object]:
    factors = _load_factor_returns(factor_returns)
    benchmarks = _load_or_fetch_benchmark_returns(
        factors=factors,
        benchmark_returns_path=benchmark_returns_path,
        api_key=api_key,
        db_path=db_path,
    )

    aligned = pd.concat([factors, benchmarks], axis=1, join="inner").dropna()
    if aligned.empty:
        raise ValueError("No overlapping observations between factors and benchmarks")

    factor_cols = list(factors.columns)
    benchmark_cols = [c for c in benchmarks.columns if c in aligned.columns]

    results: List[FactorLeakageRow] = []
    for factor in factor_cols:
        y = aligned[factor].to_numpy(dtype=float)
        x = aligned[benchmark_cols].to_numpy(dtype=float)

        beta_vec, r2 = _ols_r2(y, x)
        beta_spy = float(beta_vec[benchmark_cols.index(MARKET_ETF)])
        corr_spy = float(aligned[factor].corr(aligned[MARKET_ETF]))

        results.append(
            FactorLeakageRow(
                factor=factor,
                observations=int(len(aligned)),
                corr_to_spy=corr_spy,
                beta_to_spy=beta_spy,
                regression_r2=float(r2),
                high_corr_flag=abs(corr_spy) > corr_threshold,
                high_r2_flag=(r2 if not np.isnan(r2) else 0.0) > r2_threshold,
            )
        )

    leakage_df = pd.DataFrame(asdict(r) for r in results).sort_values(
        ["high_corr_flag", "high_r2_flag", "regression_r2"],
        ascending=[False, False, False],
    )

    factor_corr = aligned[factor_cols].corr()
    corr_values = factor_corr.to_numpy(dtype=float)
    np.fill_diagonal(corr_values, np.nan)

    stability = _factor_stability_snapshot(aligned[factor_cols], window=63)

    leakage_path = Path(f"{output_prefix}_residualization.csv")
    corr_path = Path(f"{output_prefix}_factor_corr.csv")
    report_path = Path(f"{output_prefix}_report.json")

    leakage_df.to_csv(leakage_path, index=False)
    factor_corr.to_csv(corr_path)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "factor_file": str(factor_returns),
        "benchmark_file": str(benchmark_returns_path) if benchmark_returns_path else "fetched_via_backend",
        "observations": int(len(aligned)),
        "thresholds": {"corr_to_spy": corr_threshold, "regression_r2": r2_threshold},
        "max_abs_factor_correlation": float(np.nanmax(np.abs(corr_values))),
        "high_corr_factors": leakage_df.loc[leakage_df["high_corr_flag"], "factor"].tolist(),
        "high_r2_factors": leakage_df.loc[leakage_df["high_r2_flag"], "factor"].tolist(),
        "stability_snapshot": stability,
        "outputs": {
            "residualization_csv": str(leakage_path),
            "factor_corr_csv": str(corr_path),
            "report_json": str(report_path),
        },
    }

    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run nightly factor QA checks")
    parser.add_argument(
        "--factor-returns",
        default="factor_returns.csv",
        help="Path to factor return time series CSV (default: factor_returns.csv)",
    )
    parser.add_argument(
        "--benchmark-returns",
        default=None,
        help="Optional benchmark returns CSV path. If omitted, benchmarks are fetched.",
    )
    parser.add_argument(
        "--output-prefix",
        default="factor_qa",
        help="Prefix for output artifacts (default: factor_qa)",
    )
    parser.add_argument(
        "--db-path",
        default="./av_cache.db",
        help="Cache DB path used when fetching benchmarks",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.20,
        help="Absolute corr-to-SPY threshold flag (default: 0.20)",
    )
    parser.add_argument(
        "--r2-threshold",
        type=float,
        default=0.20,
        help="Regression R^2 threshold flag (default: 0.20)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Alpha Vantage API key (defaults to ALPHAVANTAGE_API_KEY env/config)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    api_key = args.api_key or config.ALPHAVANTAGE_API_KEY

    if not api_key and not args.benchmark_returns:
        raise SystemExit(
            "ALPHAVANTAGE_API_KEY is required when --benchmark-returns is not provided."
        )

    report = run_factor_qa(
        factor_returns=Path(args.factor_returns),
        output_prefix=Path(args.output_prefix),
        api_key=api_key or "",
        db_path=args.db_path,
        corr_threshold=args.corr_threshold,
        r2_threshold=args.r2_threshold,
        benchmark_returns_path=Path(args.benchmark_returns) if args.benchmark_returns else None,
    )

    print("Factor QA complete")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
