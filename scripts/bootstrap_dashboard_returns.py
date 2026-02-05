#!/usr/bin/env python3
"""
Bootstrap explicit dashboard return inputs from factor return history.

This utility creates:
- portfolio_returns.csv
- benchmark_returns.csv

It is intended as a local bootstrap when true strategy/benchmark return
histories are not yet available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap dashboard return inputs")
    parser.add_argument(
        "--factor-returns",
        default="factor_returns.csv",
        help="Path to factor return CSV (default: factor_returns.csv)",
    )
    parser.add_argument(
        "--portfolio-out",
        default="portfolio_returns.csv",
        help="Output CSV for portfolio return series",
    )
    parser.add_argument(
        "--benchmark-out",
        default="benchmark_returns.csv",
        help="Output CSV for benchmark return series",
    )
    parser.add_argument(
        "--benchmark-column",
        default=None,
        help="Factor column to use as synthetic benchmark (default: first column)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    factor_path = Path(args.factor_returns)
    if not factor_path.exists():
        raise SystemExit(f"Factor return file not found: {factor_path}")

    factors = pd.read_csv(factor_path, index_col=0, parse_dates=True)
    factors = factors.sort_index().replace([float("inf"), float("-inf")], pd.NA).dropna(how="all")

    if factors.empty:
        raise SystemExit("No factor return data available after cleaning")

    benchmark_col = args.benchmark_column or factors.columns[0]
    if benchmark_col not in factors.columns:
        raise SystemExit(
            f"Benchmark column '{benchmark_col}' not found. Available: {list(factors.columns)}"
        )

    # Equal-weight factor blend as a bootstrap portfolio series.
    portfolio = factors.mean(axis=1, skipna=True).dropna().rename("portfolio")
    benchmark = factors[benchmark_col].dropna().rename("benchmark")

    portfolio.to_frame().to_csv(args.portfolio_out)
    benchmark.to_frame().to_csv(args.benchmark_out)

    print(f"Wrote {args.portfolio_out} ({len(portfolio)} rows)")
    print(
        f"Wrote {args.benchmark_out} ({len(benchmark)} rows) using column '{benchmark_col}'"
    )


if __name__ == "__main__":
    main()
