#!/usr/bin/env python3
"""
Prepare explicit dashboard return series from local research artifacts.

Outputs
-------
- portfolio_returns.csv  (signal backtest-derived strategy returns)
- benchmark_returns.csv  (broad-market proxy from local price cache)

This avoids placeholder equal-weight factor proxies and creates explicit
return series the dashboard can use for PM vitals.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pandas as pd

# Make project imports work when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cross_sectional import CrossSectionalAnalyzer
from src.signal_aggregator import SignalAggregator
from src.signal_backtest import SignalBacktester
from src.trading_signals import FactorMomentumAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare dashboard portfolio/benchmark return files")
    parser.add_argument("--factor-returns", default="factor_returns.csv", help="Factor return CSV path")
    parser.add_argument("--factor-loadings", default="factor_loadings.csv", help="Factor loading CSV path")
    parser.add_argument("--db-path", default="av_cache.db", help="SQLite cache DB path")
    parser.add_argument("--portfolio-out", default="portfolio_returns.csv", help="Output portfolio CSV")
    parser.add_argument("--benchmark-out", default="benchmark_returns.csv", help="Output benchmark CSV")
    parser.add_argument("--train-size", type=int, default=252, help="Backtest train window")
    parser.add_argument("--test-size", type=int, default=63, help="Backtest test window")
    parser.add_argument("--min-confidence", type=float, default=70.0, help="Backtest confidence threshold")
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0, help="Transaction cost in bps")
    parser.add_argument("--benchmark-universe", type=int, default=400, help="Number of tickers for benchmark proxy")
    parser.add_argument("--benchmark-min-coverage", type=float, default=0.95, help="Minimum date coverage for benchmark constituents")
    return parser.parse_args()


def _load_factor_inputs(factor_returns_path: Path, factor_loadings_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not factor_returns_path.exists():
        raise SystemExit(f"Factor returns not found: {factor_returns_path}")
    if not factor_loadings_path.exists():
        raise SystemExit(f"Factor loadings not found: {factor_loadings_path}")

    factor_returns = pd.read_csv(factor_returns_path, index_col=0, parse_dates=True)
    factor_returns = factor_returns.sort_index().dropna(how="all")

    factor_loadings = pd.read_csv(factor_loadings_path, index_col=0)

    if factor_returns.empty:
        raise SystemExit("Factor returns are empty")
    if factor_loadings.empty:
        raise SystemExit("Factor loadings are empty")

    return factor_returns, factor_loadings


def _derive_portfolio_returns(
    factor_returns: pd.DataFrame,
    factor_loadings: pd.DataFrame,
    train_size: int,
    test_size: int,
    min_confidence: float,
    transaction_cost_bps: float,
) -> pd.Series:
    momentum = FactorMomentumAnalyzer(factor_returns)
    cross = CrossSectionalAnalyzer(factor_loadings)

    aggregator = SignalAggregator(SimpleNamespace())
    aggregator.add_momentum_signals(momentum)
    aggregator.add_cross_sectional_signals(cross)

    tx_cost = transaction_cost_bps / 10000.0
    backtester = SignalBacktester(aggregator, factor_returns, transaction_costs=tx_cost)

    total = len(factor_returns)
    possible_walks = ((total - train_size - test_size) // test_size) + 1
    if possible_walks < 1:
        raise SystemExit(
            f"Insufficient history ({total}) for train={train_size}, test={test_size}"
        )

    backtester.run_backtest(
        train_size=train_size,
        test_size=test_size,
        n_walks=possible_walks,
        min_confidence=min_confidence,
    )

    slices: List[pd.Series] = []
    for walk in backtester.results:
        eq = walk.equity_curve.copy()
        if eq.empty:
            continue
        returns = eq.pct_change()
        returns.iloc[0] = eq.iloc[0] - 1.0
        slices.append(returns)

    if not slices:
        raise SystemExit("Backtest did not produce any portfolio returns")

    portfolio = pd.concat(slices).sort_index()
    portfolio = portfolio[~portfolio.index.duplicated(keep="last")].dropna()
    return portfolio.rename("portfolio")


def _derive_benchmark_from_cache(
    db_path: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    benchmark_universe: int,
    min_coverage: float,
) -> pd.Series:
    if not db_path.exists():
        raise SystemExit(f"Price cache not found: {db_path}")

    start = start_date.strftime("%Y-%m-%d")
    end = end_date.strftime("%Y-%m-%d")

    with sqlite3.connect(db_path) as con:
        expected_days = con.execute(
            "SELECT COUNT(DISTINCT date) FROM prices WHERE date >= ? AND date <= ?",
            (start, end),
        ).fetchone()[0]
        if not expected_days:
            raise SystemExit("No price history found in DB for requested date range")

        min_days = int(expected_days * min_coverage)
        rows = con.execute(
            """
            SELECT ticker, COUNT(*) AS n
            FROM prices
            WHERE date >= ? AND date <= ?
            GROUP BY ticker
            HAVING n >= ?
            ORDER BY n DESC
            LIMIT ?
            """,
            (start, end, min_days, benchmark_universe),
        ).fetchall()

        tickers = [r[0] for r in rows]
        if not tickers:
            raise SystemExit("No sufficiently covered tickers found for benchmark proxy")

        placeholders = ",".join(["?"] * len(tickers))
        sql = (
            f"SELECT ticker, date, adj_close FROM prices "
            f"WHERE date >= ? AND date <= ? AND ticker IN ({placeholders})"
        )
        params = [start, end, *tickers]
        px = pd.read_sql_query(sql, con, params=params)

    if px.empty:
        raise SystemExit("Benchmark proxy query returned no rows")

    px["date"] = pd.to_datetime(px["date"])
    panel = px.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    returns = panel.pct_change(fill_method=None).dropna(how="all")

    benchmark = returns.mean(axis=1, skipna=True).dropna().rename("benchmark")
    if benchmark.empty:
        raise SystemExit("Benchmark proxy series is empty after return conversion")
    return benchmark


def main() -> None:
    args = parse_args()

    factor_returns, factor_loadings = _load_factor_inputs(
        Path(args.factor_returns), Path(args.factor_loadings)
    )

    portfolio = _derive_portfolio_returns(
        factor_returns=factor_returns,
        factor_loadings=factor_loadings,
        train_size=args.train_size,
        test_size=args.test_size,
        min_confidence=args.min_confidence,
        transaction_cost_bps=args.transaction_cost_bps,
    )

    benchmark = _derive_benchmark_from_cache(
        db_path=Path(args.db_path),
        start_date=factor_returns.index.min(),
        end_date=factor_returns.index.max(),
        benchmark_universe=args.benchmark_universe,
        min_coverage=args.benchmark_min_coverage,
    )

    # Keep overlapping region so active return metrics are well-defined.
    aligned = pd.concat([portfolio, benchmark], axis=1, join="inner").dropna()
    if aligned.empty:
        raise SystemExit("No overlap between portfolio and benchmark return series")

    aligned["portfolio"].to_frame().to_csv(args.portfolio_out)
    aligned["benchmark"].to_frame().to_csv(args.benchmark_out)

    print(f"Wrote {args.portfolio_out}: {len(aligned)} rows")
    print(f"Wrote {args.benchmark_out}: {len(aligned)} rows")
    print(
        "Range:",
        aligned.index.min().strftime("%Y-%m-%d"),
        "->",
        aligned.index.max().strftime("%Y-%m-%d"),
    )


if __name__ == "__main__":
    main()
