"""
Sharpe Ratio Optimization Strategy Demonstration
=================================================

This example demonstrates how to find optimal blended factor weights
that maximize Sharpe ratio over a lookback window.

Run with:
    uv run python examples/sharpe_optimization_demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from src.factor_optimization import SharpeOptimizer, OptimizationResult
from src.factor_weighting import OptimalFactorWeighter


def generate_sample_data(n_stocks=100, n_factors=5, n_periods=500):
    """Generate realistic factor data with regime changes."""
    np.random.seed(42)
    
    tickers = [f'STK{i:03d}' for i in range(1, n_stocks + 1)]
    factors = ['Value', 'Momentum', 'Quality', 'LowVol', 'Growth']
    dates = pd.date_range(end=datetime.now(), periods=n_periods, freq='D')
    
    # Generate returns with regime changes
    returns_data = []
    
    for i in range(n_periods):
        # Simulate different regimes
        if i < n_periods // 3:
            # Regime 1: Value and Quality work
            value_ret = np.random.randn() * 0.008 + 0.001
            momentum_ret = np.random.randn() * 0.015 - 0.0005
        elif i < 2 * n_periods // 3:
            # Regime 2: Momentum and Growth work
            value_ret = np.random.randn() * 0.008 - 0.0005
            momentum_ret = np.random.randn() * 0.015 + 0.001
        else:
            # Regime 3: LowVol defensive
            value_ret = np.random.randn() * 0.008
            momentum_ret = np.random.randn() * 0.015 - 0.001
        
        quality_ret = np.random.randn() * 0.007 + 0.0003
        lowvol_ret = np.random.randn() * 0.005 + 0.0002
        growth_ret = np.random.randn() * 0.018 + 0.0001
        
        returns_data.append({
            'Value': value_ret,
            'Momentum': momentum_ret,
            'Quality': quality_ret,
            'LowVol': lowvol_ret,
            'Growth': growth_ret
        })
    
    factor_returns = pd.DataFrame(returns_data, index=dates)
    
    # Generate loadings
    loadings = pd.DataFrame(
        np.random.randn(n_stocks, n_factors) * 0.5,
        index=tickers,
        columns=factors
    )
    
    return loadings, factor_returns


def print_optimization_result(result: OptimizationResult, title: str):
    """Pretty print optimization result."""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}")
    
    print(f"\nOptimal Method Allocation:")
    for method, weight in sorted(result.method_allocation.items(), key=lambda x: -x[1]):
        if weight > 0.01:
            bar = "█" * int(weight * 50)
            print(f"  {method:<20} {weight:>6.1%} {bar}")
    
    print(f"\nOptimal Factor Weights:")
    for factor, weight in sorted(result.optimal_weights.items(), key=lambda x: -x[1]):
        bar = "█" * int(weight * 50)
        print(f"  {factor:<20} {weight:>6.1%} {bar}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Sharpe Ratio:         {result.sharpe_ratio:>8.2f}")
    print(f"  Annualized Return:    {result.annualized_return:>8.2%}")
    print(f"  Annualized Volatility:{result.annualized_volatility:>8.2%}")
    print(f"{'='*70}")


def main():
    """Run the Sharpe optimization demonstration."""
    print("\n" + "="*80)
    print("SHARPE RATIO OPTIMIZATION STRATEGY DEMONSTRATION")
    print("="*80)
    
    # Generate sample data
    print("\n1. Generating sample data with regime changes...")
    loadings, factor_returns = generate_sample_data(n_stocks=100, n_factors=5, n_periods=500)
    print(f"   Factor returns shape: {factor_returns.shape}")
    print(f"   Date range: {factor_returns.index[0].date()} to {factor_returns.index[-1].date()}")
    
    # Initialize optimizer
    print("\n2. Initializing SharpeOptimizer...")
    optimizer = SharpeOptimizer(factor_returns, loadings, risk_free_rate=0.0)
    
    # ========================================================================
    # Example 1: Single-period optimization with different techniques
    # ========================================================================
    print("\n" + "="*80)
    print("EXAMPLE 1: SINGLE-PERIOD OPTIMIZATION (Lookback = 126 days)")
    print("="*80)
    
    methods = ['sharpe', 'momentum', 'risk_parity']
    lookback = 126
    
    # Gradient-based
    print("\n3a. Gradient-Based Optimization (faster, local optima)...")
    try:
        grad_result = optimizer.optimize_blend(
            lookback=lookback,
            methods=methods,
            technique='gradient'
        )
        print_optimization_result(grad_result, "Gradient Optimization Result")
    except Exception as e:
        print(f"   Gradient optimization failed: {e}")
    
    # Differential Evolution
    print("\n3b. Differential Evolution (recommended - global optimization)...")
    try:
        de_result = optimizer.optimize_blend(
            lookback=lookback,
            methods=methods,
            technique='differential'
        )
        print_optimization_result(de_result, "Differential Evolution Result")
    except Exception as e:
        print(f"   Differential evolution failed: {e}")
    
    # Bayesian Optimization with Optuna
    print("\n3c. Bayesian Optimization with Optuna (smart search)...")
    print("   (Install optuna with: uv add optuna)")
    try:
        bayes_result = optimizer.optimize_blend(
            lookback=lookback,
            methods=methods,
            technique='bayesian',
            n_trials=50
        )
        print_optimization_result(bayes_result, "Bayesian Optimization Result")
    except Exception as e:
        print(f"   Bayesian optimization failed: {e}")
        print("   (Falls back to differential evolution if optuna not installed)")
    
    # ========================================================================
    # Example 2: Compare different lookback horizons
    # ========================================================================
    print("\n" + "="*80)
    print("EXAMPLE 2: OPTIMAL LOOKBACK HORIZON ANALYSIS")
    print("="*80)
    
    print("\n4. Testing different lookback periods...")
    horizons = [42, 63, 126, 252]
    horizon_results = optimizer.compare_methods_over_horizons(
        lookback_horizons=horizons,
        methods=methods
    )
    
    print("\n" + "-"*70)
    print(f"{'Lookback':<12} {'Sharpe':>10} {'Return':>12} {'Volatility':>12} {'Best Method':<20}")
    print("-"*70)
    
    for _, row in horizon_results.iterrows():
        best_method = max(row['method_allocation'].items(), key=lambda x: x[1])[0]
        print(f"{int(row['lookback']):<12} "
              f"{row['sharpe_ratio']:>9.2f} "
              f"{row['annualized_return']:>11.2%} "
              f"{row['annualized_volatility']:>11.2%} "
              f"{best_method:<20}")
    
    best_horizon = horizon_results.loc[horizon_results['sharpe_ratio'].idxmax()]
    print("-"*70)
    print(f"\nOptimal lookback: {int(best_horizon['lookback'])} days "
          f"(Sharpe: {best_horizon['sharpe_ratio']:.2f})")
    
    # ========================================================================
    # Example 3: Walk-forward optimization
    # ========================================================================
    print("\n" + "="*80)
    print("EXAMPLE 3: WALK-FORWARD OPTIMIZATION")
    print("="*80)
    
    print("\n5. Running walk-forward optimization...")
    print("   Training window: 126 days")
    print("   Testing window: 21 days")
    print("   This simulates real-world rebalancing...")
    
    rolling_results = optimizer.walk_forward_optimize(
        train_window=126,
        test_window=21,
        methods=methods,
        technique='differential',
        verbose=False
    )
    
    print(f"\n   Completed {len(rolling_results)} optimization windows")
    
    # Show sample of rolling weights
    print("\n   Sample rolling allocations:")
    print(f"   {'Date':<15} {'Sharpe':>8} {'Mom':>8} {'RiskPar':>8} {'Train SR':>10} {'Test SR':>10}")
    print("   " + "-"*65)
    
    for i in range(0, len(rolling_results), max(1, len(rolling_results)//5)):
        row = rolling_results.iloc[i]
        ma = row['method_weights']
        print(f"   {row['date'].strftime('%Y-%m-%d'):<15} "
              f"{ma.get('sharpe', 0):>7.1%} "
              f"{ma.get('momentum', 0):>7.1%} "
              f"{ma.get('risk_parity', 0):>7.1%} "
              f"{row['train_sharpe']:>9.2f} "
              f"{row['test_sharpe']:>9.2f}")
    
    # Calculate average test Sharpe
    avg_test_sharpe = rolling_results['test_sharpe'].mean()
    print(f"\n   Average out-of-sample Sharpe: {avg_test_sharpe:.2f}")
    
    # ========================================================================
    # Example 4: Backtest with transaction costs
    # ========================================================================
    print("\n" + "="*80)
    print("EXAMPLE 4: OUT-OF-SAMPLE BACKTEST")
    print("="*80)
    
    print("\n6. Backtesting optimal weights with transaction costs...")
    
    # Get optimal weights from the best result
    if 'de_result' in dir():
        optimal_weights = de_result.optimal_weights
    else:
        # Re-run optimization
        result = optimizer.optimize_blend(
            lookback=126,
            methods=methods,
            technique='differential',
            verbose=False
        )
        optimal_weights = result.optimal_weights
    
    # Backtest without costs
    print("\n   Scenario A: No transaction costs")
    backtest_no_cost = optimizer.backtest_optimal_weights(
        optimal_weights=optimal_weights,
        test_periods=126,
        transaction_cost=0.0,
        rebalance_freq=21
    )
    
    print(f"     Total Return:       {backtest_no_cost['total_return']:>8.2%}")
    print(f"     Annualized Return:  {backtest_no_cost['annualized_return']:>8.2%}")
    print(f"     Annualized Vol:     {backtest_no_cost['annualized_volatility']:>8.2%}")
    print(f"     Sharpe Ratio:       {backtest_no_cost['sharpe_ratio']:>8.2f}")
    print(f"     Max Drawdown:       {backtest_no_cost['max_drawdown']:>8.2%}")
    
    # Backtest with costs
    print("\n   Scenario B: 10 bps transaction costs")
    backtest_with_cost = optimizer.backtest_optimal_weights(
        optimal_weights=optimal_weights,
        test_periods=126,
        transaction_cost=0.001,
        rebalance_freq=21
    )
    
    print(f"     Total Return:       {backtest_with_cost['total_return']:>8.2%}")
    print(f"     Annualized Return:  {backtest_with_cost['annualized_return']:>8.2%}")
    print(f"     Annualized Vol:     {backtest_with_cost['annualized_volatility']:>8.2%}")
    print(f"     Sharpe Ratio:       {backtest_with_cost['sharpe_ratio']:>8.2f}")
    print(f"     Max Drawdown:       {backtest_with_cost['max_drawdown']:>8.2%}")
    
    cost_drag = backtest_no_cost['total_return'] - backtest_with_cost['total_return']
    print(f"\n     Transaction cost drag: {cost_drag:.2%}")
    
    # ========================================================================
    # Example 5: Compare with equal weight baseline
    # ========================================================================
    print("\n" + "="*80)
    print("EXAMPLE 5: COMPARISON WITH EQUAL-WEIGHT BASELINE")
    print("="*80)
    
    print("\n7. Comparing optimized vs equal-weight portfolios...")
    
    # Calculate equal-weight Sharpe
    equal_weights = {f: 1.0/len(factor_returns.columns) for f in factor_returns.columns}
    recent_returns = factor_returns.tail(126)
    equal_portfolio = (recent_returns * pd.Series(equal_weights)).sum(axis=1)
    equal_sharpe = equal_portfolio.mean() / equal_portfolio.std() * np.sqrt(252)
    
    # Get optimized Sharpe
    if 'de_result' in dir():
        opt_sharpe = de_result.sharpe_ratio
    else:
        opt_sharpe = 1.0  # placeholder
    
    print(f"\n   Equal-Weight Portfolio:")
    print(f"     Sharpe Ratio: {equal_sharpe:.2f}")
    print(f"     Factor Weights: 20% each")
    
    print(f"\n   Optimized Portfolio:")
    print(f"     Sharpe Ratio: {opt_sharpe:.2f}")
    print(f"     Improvement: {(opt_sharpe/equal_sharpe - 1)*100:.1f}%")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)
    
    print("""
KEY FINDINGS:

1. OPTIMIZATION TECHNIQUES:
   - Gradient: Fastest, may find local optima
   - Differential Evolution: Global optimization, recommended for production
   - Bayesian (Optuna): Smart search with TPE sampler, best for many methods
     Install: uv add optuna

2. LOOKBACK HORIZON:
   - Shorter (21-63 days): Faster adaptation, more noise
   - Medium (126 days): Good balance for most strategies
   - Longer (252 days): More stable, slower to adapt
   - Optimal horizon depends on factor momentum

3. WALK-FORWARD OPTIMIZATION:
   - More realistic than single-period optimization
   - Shows out-of-sample performance
   - Helps avoid overfitting
   - Recommended for production strategies

4. TRANSACTION COSTS:
   - 10 bps cost can significantly impact returns
   - Consider lower rebalancing frequency with costs
   - Optimization should account for turnover

RECOMMENDED WORKFLOW FOR PRODUCTION:

1. DETERMINE LOOKBACK HORIZON:
   ```python
   horizons = [42, 63, 126, 252]
   results = optimizer.compare_methods_over_horizons(horizons)
   best_lookback = results.loc[results['sharpe_ratio'].idxmax(), 'lookback']
   ```

2. RUN WALK-FORWARD OPTIMIZATION:
   ```python
   rolling = optimizer.walk_forward_optimize(
       train_window=best_lookback,
       test_window=21,
       methods=['sharpe', 'momentum', 'risk_parity']
   )
   ```

3. BACKTEST WITH COSTS:
   ```python
   performance = optimizer.backtest_optimal_weights(
       optimal_weights,
       test_periods=126,
       transaction_cost=0.001,
       rebalance_freq=21
   )
   ```

4. IMPLEMENT WITH REBALANCING:
   - Monthly rebalancing is typical
   - Adjust for transaction costs
   - Monitor out-of-sample performance

IMPORTANT CAVEATS:

⚠️  Past performance does not guarantee future results
⚠️  Optimization can overfit to historical data
⚠️  Walk-forward testing is essential
⚠️  Transaction costs matter significantly
⚠️  Factor regimes change over time

    """)


if __name__ == '__main__':
    main()
