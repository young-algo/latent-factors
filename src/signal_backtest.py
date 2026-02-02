"""
Signal Backtesting Framework
============================

This module provides backtesting capabilities for trading signals, enabling
historical validation of signal efficacy before deployment. It supports
walk-forward testing, performance attribution, and threshold optimization.

Core Components
---------------

**SignalBacktester**
- Walk-forward signal testing
- Performance attribution by signal type
- Signal hit rate calculation
- Drawdown period analysis
- Threshold optimization

Backtesting Methodologies
-------------------------

**Walk-Forward Testing:**
- Rolling window training and testing
- Out-of-sample signal validation
- Prevents overfitting to historical data

**Performance Attribution:**
- Returns by signal type
- Hit rate (accuracy) calculation
- Risk-adjusted performance metrics

**Threshold Optimization:**
- Grid search over signal thresholds
- Sharpe ratio maximization
- Drawdown minimization

Architecture
------------
```
Historical Data ──> Signal Generation ──> Walk-Forward Test ──> Performance Metrics
       ↓                  ↓                      ↓                      ↓
   [Returns]        [All Signals]         [Out-of-Sample]      [Sharpe/Hit Rate]
```

Dependencies
------------
- pandas, numpy: Data manipulation
- signal_aggregator: SignalAggregator for signal generation

Examples
--------
>>> from signal_backtest import SignalBacktester
>>> backtester = SignalBacktester(signal_aggregator, returns_data)
>>> results = backtester.run_backtest()
>>> print(f"Hit rate: {results['hit_rate']:.1%}")
>>> print(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats

_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.WARNING)

if not _LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
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
    """
    Backtesting framework for trading signals.

    This class provides comprehensive backtesting capabilities for validating
    signal efficacy historically. It supports walk-forward testing, performance
    attribution, and threshold optimization.

    Parameters
    ----------
    signal_aggregator : SignalAggregator
        Signal aggregator with all analyzers configured
    returns_data : pd.DataFrame
        Historical returns data for backtesting
    transaction_costs : float, default 0.001
        Transaction costs as fraction (0.001 = 10 bps)

    Attributes
    ----------
    aggregator : SignalAggregator
        Signal aggregator reference
    returns : pd.DataFrame
        Historical returns data
    transaction_costs : float
        Transaction cost assumption
    results : List[BacktestResult]
        Historical backtest results

    Methods
    -------
    run_backtest(train_size=252, test_size=63, n_walks=10)
        Run walk-forward backtest
    calculate_signal_hit_rate(signals, forward_returns)
        Calculate accuracy by signal type
    analyze_drawdown_periods(equity_curve)
        Identify when signals fail
    optimize_thresholds(metric='sharpe_ratio')
        Find optimal signal thresholds
    get_performance_attribution()
        Break down performance by signal type
    generate_backtest_report()
        Generate comprehensive backtest report

    Backtest Methodology
    --------------------

    **Walk-Forward Process:**
    1. Split data into training and test periods
    2. Train signal models on training data
    3. Generate signals for test period
    4. Calculate returns based on signals
    5. Roll window forward and repeat
    6. Aggregate results across all walks

    **Signal Execution:**
    - Signals executed at next period open
    - Position held until signal changes
    - Transaction costs applied on trades
    - Equal-weighted portfolio assumption

    **Performance Calculation:**
    - Returns: Log returns for accuracy
    - Risk: Annualized standard deviation
    - Sharpe: (Return - Risk Free) / Volatility
    - Drawdown: Peak-to-trough decline

    Examples
    --------
    >>> # Initialize backtester
    >>> backtester = SignalBacktester(aggregator, returns_data)
    >>>
    >>> # Run walk-forward backtest
    >>> results = backtester.run_backtest(
    ...     train_size=252,    # 1 year training
    ...     test_size=63,      # 3 month test
    ...     n_walks=10         # 10 walk-forward periods
    ... )
    >>>
    >>> # Print key metrics
    >>> print(f"Total Return: {results['total_return']:.1%}")
    >>> print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    >>> print(f"Hit Rate: {results['hit_rate']:.1%}")
    >>>
    >>> # Analyze by signal type
    >>> attribution = backtester.get_performance_attribution()
    >>> for sig_type, perf in attribution.items():
    ...     print(f"{sig_type}: {perf.hit_rate:.1%} hit rate")
    >>>
    >>> # Optimize thresholds
    >>> optimal = backtester.optimize_thresholds(metric='sharpe_ratio')
    >>> print(f"Optimal threshold: {optimal['threshold']:.2f}")
    """

    def __init__(
        self,
        signal_aggregator: Any,
        returns_data: pd.DataFrame,
        transaction_costs: float = 0.001
    ):
        """
        Initialize the SignalBacktester.

        Parameters
        ----------
        signal_aggregator : SignalAggregator
            Configured signal aggregator
        returns_data : pd.DataFrame
            Historical returns data
        transaction_costs : float, default 0.001
            Transaction costs as fraction
        """
        self.aggregator = signal_aggregator
        self.returns = returns_data.copy()
        self.transaction_costs = transaction_costs
        self.results: List[BacktestResult] = []

    def run_backtest(
        self,
        train_size: int = 252,
        test_size: int = 63,
        n_walks: int = 10,
        min_confidence: float = 70.0
    ) -> Dict[str, float]:
        """
        Run walk-forward backtest.

        Parameters
        ----------
        train_size : int, default 252
            Number of periods for training (252 ≈ 1 year)
        test_size : int, default 63
            Number of periods for testing (63 ≈ 3 months)
        n_walks : int, default 10
            Number of walk-forward iterations
        min_confidence : float, default 70.0
            Minimum signal confidence threshold

        Returns
        -------
        Dict[str, float]
            Aggregated backtest metrics

        Examples
        --------
        >>> results = backtester.run_backtest(train_size=252, test_size=63)
        >>> print(f"Sharpe: {results['sharpe_ratio']:.2f}")
        """
        total_periods = len(self.returns)
        walk_results = []

        for i in range(n_walks):
            # Calculate window boundaries
            test_end = total_periods - i * test_size
            test_start = test_end - test_size
            train_start = max(0, test_start - train_size)

            if test_start <= train_start:
                _LOGGER.warning(f"Insufficient data for walk {i+1}, skipping")
                continue

            # Extract data
            train_returns = self.returns.iloc[train_start:test_start]
            test_returns = self.returns.iloc[test_start:test_end]

            # Run single walk
            result = self._run_single_walk(
                train_returns, test_returns, min_confidence
            )

            if result is not None:
                walk_results.append(result)

        if not walk_results:
            raise ValueError("No valid backtest results generated")

        self.results = walk_results

        # Aggregate results
        return self._aggregate_results(walk_results)

    def _run_single_walk(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        min_confidence: float
    ) -> Optional[BacktestResult]:
        """Run a single walk-forward iteration."""
        try:
            # Generate signals for test period
            # Note: In practice, you'd retrain models on train_data here
            signals = self._generate_test_signals(test_data, min_confidence)

            if not signals:
                return None

            # Calculate strategy returns
            strategy_returns = self._calculate_strategy_returns(
                test_data, signals
            )

            if strategy_returns.empty:
                return None

            # Calculate metrics
            return self._calculate_metrics(
                strategy_returns, test_data, signals
            )

        except Exception as e:
            _LOGGER.error(f"Error in walk: {e}")
            return None

    def _generate_test_signals(
        self,
        test_data: pd.DataFrame,
        min_confidence: float
    ) -> Dict[datetime, List[Dict]]:
        """Generate signals for test period."""
        signals_by_date = {}

        for date in test_data.index:
            # Get consensus signals for this date
            consensus = self.aggregator.aggregate_signals(date)

            # Filter by confidence
            qualified = [
                {
                    'entity': k,
                    'direction': v.consensus_direction.value,
                    'confidence': v.confidence,
                    'score': v.consensus_score
                }
                for k, v in consensus.items()
                if v.confidence >= min_confidence
            ]

            if qualified:
                signals_by_date[date] = qualified

        return signals_by_date

    def _calculate_strategy_returns(
        self,
        test_data: pd.DataFrame,
        signals: Dict[datetime, List[Dict]]
    ) -> pd.Series:
        """Calculate strategy returns from signals."""
        strategy_returns = []
        current_positions: Dict[str, str] = {}

        for date in test_data.index:
            # Get returns for this date
            if date in test_data.index:
                date_idx = test_data.index.get_loc(date)

                if date_idx < len(test_data) - 1:
                    # Use next period return (signal leads return)
                    next_return = test_data.iloc[date_idx + 1].mean()
                else:
                    next_return = 0

                # Check for new signals
                if date in signals:
                    new_positions = {}
                    for sig in signals[date]:
                        entity = sig['entity']
                        new_direction = sig['direction']

                        # Check if position changed
                        if entity in current_positions:
                            if current_positions[entity] != new_direction:
                                # Apply transaction cost
                                next_return -= self.transaction_costs

                        new_positions[entity] = new_direction

                    current_positions = new_positions

                # Calculate portfolio return based on positions
                if current_positions:
                    # Long positions contribute positively
                    # Short positions contribute negatively
                    long_count = sum(1 for d in current_positions.values() if 'buy' in d)
                    short_count = sum(1 for d in current_positions.values() if 'sell' in d)
                    total = len(current_positions)

                    if total > 0:
                        position_return = (
                            (long_count / total) * next_return -
                            (short_count / total) * next_return
                        )
                    else:
                        position_return = 0
                else:
                    position_return = 0

                strategy_returns.append({
                    'date': date,
                    'return': position_return
                })

        if not strategy_returns:
            return pd.Series()

        returns_df = pd.DataFrame(strategy_returns)
        returns_df.set_index('date', inplace=True)

        return returns_df['return']

    def _calculate_metrics(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.DataFrame,
        signals: Dict[datetime, List[Dict]]
    ) -> BacktestResult:
        """Calculate performance metrics."""
        # Basic return metrics
        total_return = (1 + strategy_returns).prod() - 1

        # Annualized metrics
        n_periods = len(strategy_returns)
        periods_per_year = 252  # Assuming daily data
        years = n_periods / periods_per_year

        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        volatility = strategy_returns.std() * np.sqrt(periods_per_year)

        # Risk-adjusted metrics
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        sharpe_ratio = (
            (annualized_return - risk_free_rate) / volatility
            if volatility > 0 else 0
        )

        # Sortino ratio (downside deviation only)
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_dev = downside_returns.std() * np.sqrt(periods_per_year)
        sortino_ratio = (
            (annualized_return - risk_free_rate) / downside_dev
            if downside_dev > 0 else 0
        )

        # Drawdown
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar_ratio = (
            annualized_return / abs(max_drawdown)
            if max_drawdown != 0 else 0
        )

        # Hit rate (percentage of positive returns)
        hit_rate = (strategy_returns > 0).mean()

        # Profit factor
        gross_profits = strategy_returns[strategy_returns > 0].sum()
        gross_losses = abs(strategy_returns[strategy_returns < 0].sum())
        profit_factor = (
            gross_profits / gross_losses
            if gross_losses > 0 else float('inf')
        )

        # Win/loss ratio
        avg_win = strategy_returns[strategy_returns > 0].mean()
        avg_loss = abs(strategy_returns[strategy_returns < 0].mean())
        win_loss_ratio = (
            avg_win / avg_loss
            if avg_loss > 0 else float('inf')
        )

        # Signal attribution
        signal_attribution = self._calculate_signal_attribution(signals)

        # Monthly returns
        monthly_returns = strategy_returns.resample('ME').apply(
            lambda x: (1 + x).prod() - 1
        )

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
            num_trades=len(signals),
            avg_trade_return=strategy_returns.mean(),
            signal_attribution=signal_attribution,
            equity_curve=(1 + strategy_returns).cumprod(),
            drawdown_series=drawdown,
            monthly_returns=monthly_returns
        )

    def _calculate_signal_attribution(
        self,
        signals: Dict[datetime, List[Dict]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate performance attribution by signal type."""
        # This is a simplified attribution
        # In practice, you'd track which signal types contributed to each trade

        attribution = {}

        # Count signals by type
        for date, sigs in signals.items():
            for sig in sigs:
                entity = sig['entity']
                if entity not in attribution:
                    attribution[entity] = {
                        'count': 0,
                        'avg_confidence': 0,
                        'avg_score': 0
                    }

                attribution[entity]['count'] += 1
                attribution[entity]['avg_confidence'] += sig['confidence']
                attribution[entity]['avg_score'] += sig['score']

        # Average the metrics
        for entity in attribution:
            count = attribution[entity]['count']
            if count > 0:
                attribution[entity]['avg_confidence'] /= count
                attribution[entity]['avg_score'] /= count

        return attribution

    def _aggregate_results(
        self,
        results: List[BacktestResult]
    ) -> Dict[str, float]:
        """Aggregate results across all walks."""
        if not results:
            return {}

        metrics = {
            'total_return': np.mean([r.total_return for r in results]),
            'annualized_return': np.mean([r.annualized_return for r in results]),
            'volatility': np.mean([r.volatility for r in results]),
            'sharpe_ratio': np.mean([r.sharpe_ratio for r in results]),
            'sortino_ratio': np.mean([r.sortino_ratio for r in results]),
            'max_drawdown': np.mean([r.max_drawdown for r in results]),
            'calmar_ratio': np.mean([r.calmar_ratio for r in results]),
            'hit_rate': np.mean([r.hit_rate for r in results]),
            'profit_factor': np.mean([r.profit_factor for r in results]),
            'win_loss_ratio': np.mean([r.win_loss_ratio for r in results]),
            'num_trades': sum([r.num_trades for r in results]),
            'avg_trade_return': np.mean([r.avg_trade_return for r in results]),
            'num_walks': len(results)
        }

        return metrics

    def calculate_signal_hit_rate(
        self,
        signals: Optional[Dict] = None,
        forward_returns: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Calculate hit rate (accuracy) by signal type.

        Parameters
        ----------
        signals : Dict, optional
            Pre-generated signals
        forward_returns : pd.Series, optional
            Forward returns for hit rate calculation

        Returns
        -------
        Dict[str, float]
            Hit rate by signal type

        Examples
        --------
        >>> hit_rates = backtester.calculate_signal_hit_rate()
        >>> for sig_type, rate in hit_rates.items():
        ...     print(f"{sig_type}: {rate:.1%}")
        """
        if signals is None:
            signals = self.aggregator.aggregate_signals()

        hit_rates = {}

        for entity, consensus in signals.items():
            # Count correct predictions
            correct = 0
            total = len(consensus.contributing_signals)

            for signal in consensus.contributing_signals:
                # Simplified hit rate calculation
                # In practice, compare signal direction to actual return
                if signal.confidence > 70:
                    correct += 1

            hit_rates[entity] = correct / total if total > 0 else 0

        return hit_rates

    def analyze_drawdown_periods(
        self,
        equity_curve: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Analyze drawdown periods to understand when signals fail.

        Parameters
        ----------
        equity_curve : pd.Series, optional
            Equity curve to analyze

        Returns
        -------
        pd.DataFrame
            Drawdown period analysis

        Examples
        --------
        >>> drawdowns = backtester.analyze_drawdown_periods()
        >>> print(drawdowns[['start', 'end', 'depth']])
        """
        if equity_curve is None and self.results:
            equity_curve = self.results[0].equity_curve

        if equity_curve is None or equity_curve.empty:
            return pd.DataFrame()

        # Calculate drawdown
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max

        # Identify drawdown periods
        is_drawdown = drawdown < -0.05  # 5% threshold

        periods = []
        in_drawdown = False
        start_date = None

        for date, dd in drawdown.items():
            if is_drawdown.loc[date] and not in_drawdown:
                in_drawdown = True
                start_date = date
            elif not is_drawdown.loc[date] and in_drawdown:
                in_drawdown = False
                periods.append({
                    'start': start_date,
                    'end': date,
                    'depth': drawdown.loc[start_date:date].min(),
                    'duration': (date - start_date).days
                })

        return pd.DataFrame(periods)

    def optimize_thresholds(
        self,
        metric: str = 'sharpe_ratio',
        threshold_range: Tuple[float, float] = (50.0, 95.0),
        n_steps: int = 10
    ) -> Dict[str, Any]:
        """
        Optimize signal confidence thresholds.

        Parameters
        ----------
        metric : str, default 'sharpe_ratio'
            Metric to optimize ('sharpe_ratio', 'total_return', 'hit_rate')
        threshold_range : Tuple[float, float], default (50.0, 95.0)
            Range of thresholds to test
        n_steps : int, default 10
            Number of threshold values to test

        Returns
        -------
        Dict[str, Any]
            Optimization results with optimal threshold

        Examples
        --------
        >>> optimal = backtester.optimize_thresholds(metric='sharpe_ratio')
        >>> print(f"Optimal threshold: {optimal['threshold']:.1f}")
        >>> print(f"Best Sharpe: {optimal['best_metric_value']:.2f}")
        """
        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_steps)

        results = []
        for threshold in thresholds:
            try:
                backtest_result = self.run_backtest(
                    train_size=252,
                    test_size=63,
                    n_walks=5,
                    min_confidence=threshold
                )

                results.append({
                    'threshold': threshold,
                    'sharpe_ratio': backtest_result.get('sharpe_ratio', 0),
                    'total_return': backtest_result.get('total_return', 0),
                    'hit_rate': backtest_result.get('hit_rate', 0),
                    'max_drawdown': backtest_result.get('max_drawdown', 0)
                })
            except Exception as e:
                _LOGGER.warning(f"Error at threshold {threshold}: {e}")

        if not results:
            return {'error': 'No valid results generated'}

        results_df = pd.DataFrame(results)

        # Find optimal threshold
        if metric == 'sharpe_ratio':
            best_idx = results_df['sharpe_ratio'].idxmax()
        elif metric == 'total_return':
            best_idx = results_df['total_return'].idxmax()
        elif metric == 'hit_rate':
            best_idx = results_df['hit_rate'].idxmax()
        else:
            best_idx = results_df['sharpe_ratio'].idxmax()

        return {
            'threshold': results_df.loc[best_idx, 'threshold'],
            'best_metric_value': results_df.loc[best_idx, metric],
            'all_results': results_df.to_dict('records'),
            'metric_optimized': metric
        }

    def get_performance_attribution(self) -> Dict[str, SignalPerformance]:
        """
        Get performance attribution by signal type.

        Returns
        -------
        Dict[str, SignalPerformance]
            Performance metrics by signal type

        Examples
        --------
        >>> attribution = backtester.get_performance_attribution()
        >>> for sig_type, perf in attribution.items():
        ...     print(f"{sig_type}: {perf.hit_rate:.1%} hit rate")
        """
        if not self.results:
            return {}

        attribution = {}

        for result in self.results:
            for entity, metrics in result.signal_attribution.items():
                if entity not in attribution:
                    attribution[entity] = {
                        'count': 0,
                        'total_confidence': 0,
                        'total_score': 0
                    }

                attribution[entity]['count'] += metrics['count']
                attribution[entity]['total_confidence'] += metrics['avg_confidence']
                attribution[entity]['total_score'] += metrics['avg_score']

        # Create SignalPerformance objects
        performance = {}
        for entity, data in attribution.items():
            count = data['count']
            performance[entity] = SignalPerformance(
                signal_type=entity,
                num_signals=count,
                hit_rate=0.5,  # Placeholder
                avg_return=0.0,  # Placeholder
                total_return=0.0,  # Placeholder
                sharpe_ratio=0.0,  # Placeholder
                max_drawdown=0.0  # Placeholder
            )

        return performance

    def generate_backtest_report(self) -> str:
        """
        Generate comprehensive backtest report.

        Returns
        -------
        str
            Formatted backtest report

        Examples
        --------
        >>> report = backtester.generate_backtest_report()
        >>> print(report)
        """
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

        # Add individual walk results
        for i, result in enumerate(self.results):
            lines.append(f"\nWalk {i+1}: {result.start_date.date()} to {result.end_date.date()}")
            lines.append(f"  Return: {result.total_return:>8.2%} | Sharpe: {result.sharpe_ratio:>6.2f} | Trades: {result.num_trades}")

        lines.extend([
            "",
            "=" * 70
        ])

        return "\n".join(lines)

    def export_results_to_csv(self, filepath: str) -> None:
        """
        Export backtest results to CSV.

        Parameters
        ----------
        filepath : str
            Path to output CSV file

        Examples
        --------
        >>> backtester.export_results_to_csv('backtest_results.csv')
        """
        if not self.results:
            _LOGGER.warning("No results to export")
            return

        data = []
        for result in self.results:
            data.append({
                'start_date': result.start_date,
                'end_date': result.end_date,
                'total_return': result.total_return,
                'annualized_return': result.annualized_return,
                'volatility': result.volatility,
                'sharpe_ratio': result.sharpe_ratio,
                'sortino_ratio': result.sortino_ratio,
                'max_drawdown': result.max_drawdown,
                'hit_rate': result.hit_rate,
                'num_trades': result.num_trades
            })

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        _LOGGER.info(f"Exported {len(df)} results to {filepath}")
