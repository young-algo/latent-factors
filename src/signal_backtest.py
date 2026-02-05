"""
Signal Backtesting Framework
============================

Walk-forward backtesting for aggregated trading signals.

Key upgrades in this implementation:
- Signals are generated as-of date ``t`` and applied to realized returns at ``t+1``.
- PnL is computed from entity-level weights and returns (not universe-average proxies).
- Transaction costs are applied from daily turnover.
- Trade provenance is captured in per-walk trade logs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.WARNING)

if not _LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
    handler.setFormatter(formatter)
    _LOGGER.addHandler(handler)
    _LOGGER.propagate = False


class BacktestMetric(Enum):
    """Enumeration of backtest performance metrics."""

    TOTAL_RETURN = "total_return"
    ANNUALIZED_RETURN = "annualized_return"
    VOLATILITY = "volatility"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    CALMAR_RATIO = "calmar_ratio"
    HIT_RATE = "hit_rate"
    PROFIT_FACTOR = "profit_factor"
    WIN_LOSS_RATIO = "win_loss_ratio"


@dataclass
class BacktestResult:
    """Data class representing backtest results."""

    start_date: datetime
    end_date: datetime
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    hit_rate: float
    profit_factor: float
    win_loss_ratio: float
    num_trades: int
    avg_trade_return: float
    signal_attribution: Dict[str, Dict[str, float]]
    equity_curve: pd.Series
    drawdown_series: pd.Series
    monthly_returns: pd.Series
    trade_log: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class SignalPerformance:
    """Data class representing signal type performance."""

    signal_type: str
    num_signals: int
    hit_rate: float
    avg_return: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float


class SignalBacktester:
    """Walk-forward backtester for multi-source consensus signals."""

    def __init__(
        self,
        signal_aggregator: Any,
        returns_data: pd.DataFrame,
        transaction_costs: float = 0.001,
    ):
        self.aggregator = signal_aggregator
        self.returns = returns_data.copy()
        self.transaction_costs = transaction_costs
        self.results: List[BacktestResult] = []

        if not isinstance(self.returns.index, pd.DatetimeIndex):
            self.returns.index = pd.to_datetime(self.returns.index)
        self.returns = self.returns.sort_index()

    def run_backtest(
        self,
        train_size: int = 252,
        test_size: int = 63,
        n_walks: int = 10,
        min_confidence: float = 70.0,
    ) -> Dict[str, float]:
        """Run walk-forward backtest with train/test windows."""
        if self.returns.empty:
            raise ValueError("returns_data is empty")
        if train_size <= 1 or test_size <= 1:
            raise ValueError("train_size and test_size must both be > 1")

        total_periods = len(self.returns)
        windows: List[Tuple[int, int, int]] = []

        walk_start = 0
        while walk_start + train_size + test_size <= total_periods:
            train_start = walk_start
            test_start = walk_start + train_size
            test_end = test_start + test_size
            windows.append((train_start, test_start, test_end))
            walk_start += test_size

        if not windows:
            raise ValueError(
                "Insufficient data for requested window sizes "
                f"(have {total_periods}, need at least {train_size + test_size})"
            )

        # Prefer most recent windows when the user requests fewer walks.
        selected_windows = windows[-n_walks:]

        walk_results: List[BacktestResult] = []
        for walk_idx, (train_start, test_start, test_end) in enumerate(selected_windows, start=1):
            train_returns = self.returns.iloc[train_start:test_start]
            test_returns = self.returns.iloc[test_start:test_end]

            result = self._run_single_walk(train_returns, test_returns, min_confidence)
            if result is None:
                _LOGGER.warning("Walk %d produced no valid trades; skipping", walk_idx)
                continue
            walk_results.append(result)

        if not walk_results:
            raise ValueError("No valid backtest results generated")

        self.results = walk_results
        return self._aggregate_results(walk_results)

    def _run_single_walk(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        min_confidence: float,
    ) -> Optional[BacktestResult]:
        """Run one train/test walk-forward segment."""
        try:
            if len(train_data) < 2 or len(test_data) < 2:
                return None

            signals = self._generate_test_signals(test_data, min_confidence=min_confidence)
            strategy_returns, trade_log = self._calculate_strategy_returns(test_data, signals)

            if strategy_returns.empty:
                return None

            benchmark_returns = (
                test_data.mean(axis=1)
                .shift(-1)
                .reindex(strategy_returns.index)
                .fillna(0.0)
            )

            return self._calculate_metrics(
                strategy_returns=strategy_returns,
                benchmark_returns=benchmark_returns,
                signals=signals,
                trade_log=trade_log,
            )
        except Exception as exc:
            _LOGGER.error("Error in walk: %s", exc)
            return None

    def _direction_to_score(self, direction: str) -> float:
        """Map direction labels to numeric signal strength."""
        mapping = {
            "strong_buy": 2.0,
            "buy": 1.0,
            "long": 1.0,
            "neutral": 0.0,
            "sell": -1.0,
            "short": -1.0,
            "strong_sell": -2.0,
        }
        return mapping.get(str(direction).lower(), 0.0)

    def _generate_test_signals(
        self,
        test_data: pd.DataFrame,
        min_confidence: float,
    ) -> Dict[datetime, List[Dict[str, float]]]:
        """Generate daily tradable signals for the test period."""
        signals_by_date: Dict[datetime, List[Dict[str, float]]] = {}
        tradable_entities = set(test_data.columns)

        # No t+1 return exists for the final date in a test window.
        for date in test_data.index[:-1]:
            consensus = self.aggregator.aggregate_signals(date=pd.Timestamp(date))
            qualified: List[Dict[str, float]] = []

            for entity, signal in consensus.items():
                direction = signal.consensus_direction.value
                score = self._direction_to_score(direction)
                confidence = float(signal.confidence)

                if confidence < min_confidence:
                    continue
                if abs(score) < 1e-12:
                    continue

                # market_regime is a global overlay signal; all others must map to a return column.
                if entity != "market_regime" and entity not in tradable_entities:
                    continue

                qualified.append(
                    {
                        "entity": entity,
                        "direction": direction,
                        "confidence": confidence,
                        "score": float(signal.consensus_score),
                    }
                )

            if qualified:
                signals_by_date[date] = qualified

        return signals_by_date

    def _build_target_weights(
        self,
        date_signals: List[Dict[str, float]],
        universe_columns: List[str],
    ) -> pd.Series:
        """Map qualified date signals into normalized target portfolio weights."""
        entity_scores: Dict[str, float] = {}
        global_scores: List[float] = []

        for sig in date_signals:
            direction_score = self._direction_to_score(sig["direction"])
            if abs(direction_score) < 1e-12:
                continue

            confidence_scale = max(0.0, float(sig["confidence"])) / 100.0
            signed_strength = direction_score * confidence_scale
            entity = sig["entity"]

            if entity == "market_regime":
                global_scores.append(signed_strength)
            elif entity in universe_columns:
                entity_scores[entity] = entity_scores.get(entity, 0.0) + signed_strength

        if not entity_scores and global_scores:
            global_bias = float(np.mean(global_scores))
            if abs(global_bias) > 1e-12:
                weights = pd.Series(np.sign(global_bias), index=universe_columns, dtype=float)
            else:
                weights = pd.Series(dtype=float)
        else:
            weights = pd.Series(entity_scores, dtype=float)
            if global_scores and not weights.empty:
                # Apply a small global overlay tilt if regime signal is present.
                weights += np.sign(float(np.mean(global_scores))) * 0.25

        if weights.empty:
            return weights

        weights = weights[weights.abs() > 1e-12]
        if weights.empty:
            return weights

        gross = float(weights.abs().sum())
        if gross <= 0:
            return pd.Series(dtype=float)

        return weights / gross

    def _calculate_strategy_returns(
        self,
        test_data: pd.DataFrame,
        signals: Dict[datetime, List[Dict[str, float]]],
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """Compute daily strategy returns from target weights and next-day returns."""
        columns = list(test_data.columns)
        prev_weights = pd.Series(0.0, index=columns, dtype=float)

        return_rows: List[Dict[str, float]] = []
        trade_rows: List[Dict[str, float]] = []

        for idx, signal_date in enumerate(test_data.index[:-1]):
            next_date = test_data.index[idx + 1]
            date_signals = signals.get(signal_date, [])

            target_sparse = self._build_target_weights(date_signals, columns)
            target_weights = pd.Series(0.0, index=columns, dtype=float)
            if not target_sparse.empty:
                target_weights.update(target_sparse)

            turnover = float((target_weights - prev_weights).abs().sum())
            costs = turnover * self.transaction_costs

            next_returns = test_data.loc[next_date].fillna(0.0)
            gross_return = float((target_weights * next_returns).sum())
            net_return = gross_return - costs

            return_rows.append(
                {
                    "date": next_date,
                    "return": net_return,
                    "gross_return": gross_return,
                    "transaction_cost": costs,
                    "turnover": turnover,
                }
            )

            for sig in date_signals:
                entity = sig["entity"]
                if entity not in target_weights.index:
                    continue
                weight = float(target_weights.loc[entity])
                if abs(weight) < 1e-12:
                    continue

                realized = float(next_returns.loc[entity])
                contribution = weight * realized

                trade_rows.append(
                    {
                        "signal_date": signal_date,
                        "execution_date": next_date,
                        "entity": entity,
                        "direction": sig["direction"],
                        "confidence": float(sig["confidence"]),
                        "score": float(sig["score"]),
                        "weight": weight,
                        "asset_return": realized,
                        "contribution": contribution,
                    }
                )

            prev_weights = target_weights

        if not return_rows:
            return pd.Series(dtype=float), pd.DataFrame()

        returns_df = pd.DataFrame(return_rows).set_index("date")
        trade_df = pd.DataFrame(trade_rows)

        return returns_df["return"], trade_df

    def _calculate_metrics(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        signals: Dict[datetime, List[Dict[str, float]]],
        trade_log: pd.DataFrame,
    ) -> BacktestResult:
        """Compute walk-level performance metrics."""
        total_return = float((1.0 + strategy_returns).prod() - 1.0)

        n_periods = len(strategy_returns)
        periods_per_year = 252
        years = max(n_periods / periods_per_year, 1.0 / periods_per_year)

        annualized_return = float((1.0 + total_return) ** (1.0 / years) - 1.0)
        volatility = float(strategy_returns.std(ddof=0) * np.sqrt(periods_per_year))

        risk_free_rate = 0.0
        sharpe_ratio = (
            (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0.0
        )

        downside = strategy_returns[strategy_returns < 0]
        downside_dev = float(downside.std(ddof=0) * np.sqrt(periods_per_year))
        sortino_ratio = (
            (annualized_return - risk_free_rate) / downside_dev if downside_dev > 0 else 0.0
        )

        equity_curve = (1.0 + strategy_returns).cumprod()
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

        hit_rate = float((strategy_returns > 0).mean())
        gross_profits = float(strategy_returns[strategy_returns > 0].sum())
        gross_losses = float(abs(strategy_returns[strategy_returns < 0].sum()))
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float("inf")

        avg_win = float(strategy_returns[strategy_returns > 0].mean())
        avg_loss = float(abs(strategy_returns[strategy_returns < 0].mean()))
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")

        signal_attribution = self._calculate_signal_attribution(signals, trade_log)

        monthly_returns = strategy_returns.resample("ME").apply(
            lambda x: (1.0 + x).prod() - 1.0
        )

        if trade_log.empty:
            avg_trade_return = float(strategy_returns.mean())
            num_trades = 0
        else:
            avg_trade_return = float(trade_log["contribution"].mean())
            num_trades = int(len(trade_log))

        return BacktestResult(
            start_date=strategy_returns.index[0],
            end_date=strategy_returns.index[-1],
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            hit_rate=hit_rate,
            profit_factor=profit_factor,
            win_loss_ratio=win_loss_ratio,
            num_trades=num_trades,
            avg_trade_return=avg_trade_return,
            signal_attribution=signal_attribution,
            equity_curve=equity_curve,
            drawdown_series=drawdown,
            monthly_returns=monthly_returns,
            trade_log=trade_log,
        )

    def _calculate_signal_attribution(
        self,
        signals: Dict[datetime, List[Dict[str, float]]],
        trade_log: pd.DataFrame,
    ) -> Dict[str, Dict[str, float]]:
        """Compute attribution summary by traded entity."""
        if trade_log.empty:
            # Fallback attribution when no mapped trades were generated.
            attribution: Dict[str, Dict[str, float]] = {}
            for _, sigs in signals.items():
                for sig in sigs:
                    entity = str(sig["entity"])
                    if entity not in attribution:
                        attribution[entity] = {
                            "count": 0.0,
                            "avg_confidence": 0.0,
                            "avg_score": 0.0,
                        }
                    attribution[entity]["count"] += 1.0
                    attribution[entity]["avg_confidence"] += float(sig["confidence"])
                    attribution[entity]["avg_score"] += float(sig["score"])

            for entity, vals in attribution.items():
                count = max(vals["count"], 1.0)
                vals["avg_confidence"] /= count
                vals["avg_score"] /= count
            return attribution

        grouped = trade_log.groupby("entity")
        attribution = {}
        for entity, g in grouped:
            attribution[str(entity)] = {
                "count": float(len(g)),
                "avg_confidence": float(g["confidence"].mean()),
                "avg_score": float(g["score"].mean()),
                "total_contribution": float(g["contribution"].sum()),
                "hit_rate": float((g["contribution"] > 0).mean()),
            }
        return attribution

    def _aggregate_results(self, results: List[BacktestResult]) -> Dict[str, float]:
        """Aggregate walk-level results into a summary dictionary."""
        if not results:
            return {}

        def _mean(values: List[float]) -> float:
            arr = np.asarray(values, dtype=float)
            return float(np.nanmean(arr))

        return {
            "total_return": _mean([r.total_return for r in results]),
            "annualized_return": _mean([r.annualized_return for r in results]),
            "volatility": _mean([r.volatility for r in results]),
            "sharpe_ratio": _mean([r.sharpe_ratio for r in results]),
            "sortino_ratio": _mean([r.sortino_ratio for r in results]),
            "max_drawdown": _mean([r.max_drawdown for r in results]),
            "calmar_ratio": _mean([r.calmar_ratio for r in results]),
            "hit_rate": _mean([r.hit_rate for r in results]),
            "profit_factor": _mean([r.profit_factor for r in results]),
            "win_loss_ratio": _mean([r.win_loss_ratio for r in results]),
            "num_trades": float(sum(r.num_trades for r in results)),
            "avg_trade_return": _mean([r.avg_trade_return for r in results]),
            "num_walks": float(len(results)),
        }

    def calculate_signal_hit_rate(
        self,
        signals: Optional[Dict] = None,
        forward_returns: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """Calculate hit-rate by entity from trade logs or supplied signals."""
        if self.results:
            logs = [r.trade_log for r in self.results if not r.trade_log.empty]
            if logs:
                trades = pd.concat(logs, ignore_index=True)
                return (
                    trades.groupby("entity")["contribution"]
                    .apply(lambda x: float((x > 0).mean()))
                    .to_dict()
                )

        if signals is None:
            signals = self.aggregator.aggregate_signals()

        hit_rates: Dict[str, float] = {}
        for entity, consensus in signals.items():
            if forward_returns is None or entity not in getattr(forward_returns, "index", []):
                hit_rates[entity] = 0.0
                continue

            direction = self._direction_to_score(consensus.consensus_direction.value)
            realized = float(forward_returns.loc[entity])
            hit_rates[entity] = 1.0 if direction * realized > 0 else 0.0

        return hit_rates

    def analyze_drawdown_periods(
        self,
        equity_curve: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Analyze drawdown periods to identify stress windows."""
        if equity_curve is None and self.results:
            equity_curve = self.results[0].equity_curve

        if equity_curve is None or equity_curve.empty:
            return pd.DataFrame()

        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        is_drawdown = drawdown < -0.05

        periods: List[Dict[str, Any]] = []
        in_drawdown = False
        start_date = None

        for date, _ in drawdown.items():
            if is_drawdown.loc[date] and not in_drawdown:
                in_drawdown = True
                start_date = date
            elif not is_drawdown.loc[date] and in_drawdown:
                in_drawdown = False
                periods.append(
                    {
                        "start": start_date,
                        "end": date,
                        "depth": float(drawdown.loc[start_date:date].min()),
                        "duration": (date - start_date).days,
                    }
                )

        return pd.DataFrame(periods)

    def optimize_thresholds(
        self,
        metric: str = "sharpe_ratio",
        threshold_range: Tuple[float, float] = (50.0, 95.0),
        n_steps: int = 10,
    ) -> Dict[str, Any]:
        """Grid-search confidence thresholds for backtest performance."""
        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_steps)
        optimization_rows: List[Dict[str, float]] = []

        for threshold in thresholds:
            try:
                backtest_result = self.run_backtest(
                    train_size=252,
                    test_size=63,
                    n_walks=5,
                    min_confidence=float(threshold),
                )
                optimization_rows.append(
                    {
                        "threshold": float(threshold),
                        "sharpe_ratio": float(backtest_result.get("sharpe_ratio", 0.0)),
                        "total_return": float(backtest_result.get("total_return", 0.0)),
                        "hit_rate": float(backtest_result.get("hit_rate", 0.0)),
                        "max_drawdown": float(backtest_result.get("max_drawdown", 0.0)),
                    }
                )
            except Exception as exc:
                _LOGGER.warning("Error at threshold %.2f: %s", threshold, exc)

        if not optimization_rows:
            return {"error": "No valid results generated"}

        results_df = pd.DataFrame(optimization_rows)

        if metric == "total_return":
            best_idx = int(results_df["total_return"].idxmax())
        elif metric == "hit_rate":
            best_idx = int(results_df["hit_rate"].idxmax())
        else:
            best_idx = int(results_df["sharpe_ratio"].idxmax())

        return {
            "threshold": float(results_df.loc[best_idx, "threshold"]),
            "best_metric_value": float(results_df.loc[best_idx, metric]),
            "all_results": results_df.to_dict("records"),
            "metric_optimized": metric,
        }

    def get_performance_attribution(self) -> Dict[str, SignalPerformance]:
        """Return per-entity performance attribution across all walks."""
        if not self.results:
            return {}

        logs = [r.trade_log for r in self.results if not r.trade_log.empty]
        if not logs:
            return {}

        trades = pd.concat(logs, ignore_index=True)
        performance: Dict[str, SignalPerformance] = {}

        for entity, g in trades.groupby("entity"):
            pnl = g["contribution"]
            vol = float(pnl.std(ddof=0))
            sharpe = float((pnl.mean() / vol) * np.sqrt(252)) if vol > 0 else 0.0
            cumulative = (1.0 + pnl).cumprod()
            dd = (cumulative - cumulative.cummax()) / cumulative.cummax()

            performance[str(entity)] = SignalPerformance(
                signal_type=str(entity),
                num_signals=int(len(g)),
                hit_rate=float((pnl > 0).mean()),
                avg_return=float(pnl.mean()),
                total_return=float(pnl.sum()),
                sharpe_ratio=sharpe,
                max_drawdown=float(dd.min()) if not dd.empty else 0.0,
            )

        return performance

    def generate_backtest_report(self) -> str:
        """Generate a text backtest report."""
        if not self.results:
            return "No backtest results available. Run run_backtest() first."

        aggregated = self._aggregate_results(self.results)

        lines = [
            "=" * 70,
            "SIGNAL BACKTEST REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            "",
            "PERFORMANCE SUMMARY",
            "-" * 70,
            f"Total Return:          {aggregated['total_return']:>10.2%}",
            f"Annualized Return:     {aggregated['annualized_return']:>10.2%}",
            f"Volatility:            {aggregated['volatility']:>10.2%}",
            f"Sharpe Ratio:          {aggregated['sharpe_ratio']:>10.2f}",
            f"Sortino Ratio:         {aggregated['sortino_ratio']:>10.2f}",
            f"Maximum Drawdown:      {aggregated['max_drawdown']:>10.2%}",
            f"Calmar Ratio:          {aggregated['calmar_ratio']:>10.2f}",
            "",
            "SIGNAL QUALITY",
            "-" * 70,
            f"Hit Rate:              {aggregated['hit_rate']:>10.1%}",
            f"Profit Factor:         {aggregated['profit_factor']:>10.2f}",
            f"Win/Loss Ratio:        {aggregated['win_loss_ratio']:>10.2f}",
            f"Number of Trades:      {aggregated['num_trades']:>10.0f}",
            f"Avg Trade Return:      {aggregated['avg_trade_return']:>10.2%}",
            "",
            "WALK-FORWARD DETAILS",
            "-" * 70,
            f"Number of Walks:       {aggregated['num_walks']:>10.0f}",
        ]

        for i, result in enumerate(self.results, start=1):
            lines.append(f"\nWalk {i}: {result.start_date.date()} to {result.end_date.date()}")
            lines.append(
                f"  Return: {result.total_return:>8.2%} | "
                f"Sharpe: {result.sharpe_ratio:>6.2f} | "
                f"Trades: {result.num_trades:>4d}"
            )

        lines.extend(["", "=" * 70])
        return "\n".join(lines)

    def export_results_to_csv(self, filepath: str) -> None:
        """Export walk-level summary results to CSV."""
        if not self.results:
            _LOGGER.warning("No results to export")
            return

        rows = [
            {
                "start_date": result.start_date,
                "end_date": result.end_date,
                "total_return": result.total_return,
                "annualized_return": result.annualized_return,
                "volatility": result.volatility,
                "sharpe_ratio": result.sharpe_ratio,
                "sortino_ratio": result.sortino_ratio,
                "max_drawdown": result.max_drawdown,
                "hit_rate": result.hit_rate,
                "num_trades": result.num_trades,
            }
            for result in self.results
        ]

        pd.DataFrame(rows).to_csv(filepath, index=False)
        _LOGGER.info("Exported %d results to %s", len(rows), filepath)
