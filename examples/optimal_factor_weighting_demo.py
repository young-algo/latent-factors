"""
Demonstration of Optimal Factor Weighting Methods
================================================

This example shows how to use various optimal weighting methods for
creating cross-sectional long-short portfolios.

Run with:
    uv run python examples/optimal_factor_weighting_demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.factor_weighting import OptimalFactorWeighter, FactorCharacteristics
from src.cross_sectional import CrossSectionalAnalyzer


def generate_sample_data(n_stocks=100, n_factors=5, n_periods=252):
    """Generate realistic sample factor data."""
    np.random.seed(42)
    
    # Stock tickers
    tickers = [f'STK{i:03d}' for i in range(1, n_stocks + 1)]
    
    # Factor names
    factors = ['Value', 'Momentum', 'Quality', 'LowVol', 'Growth']
    
    # Generate factor returns with different characteristics
    dates = pd.date_range(end=datetime.now(), periods=n_periods, freq='D')
    
    # Value: High Sharpe, consistent
    value_rets = np.random.randn(n_periods) * 0.008 + 0.0005
    
    # Momentum: High return but volatile (momentum crashes)
    momentum_rets = np.random.randn(n_periods) * 0.015 + 0.0008
    # Add occasional momentum crash
    crash_periods = np.random.choice(n_periods, size=3, replace=False)
    for cp in crash_periods:
        if cp < n_periods - 5:
            momentum_rets[cp:cp+5] = -0.03
    
    # Quality: Medium Sharpe, defensive
    quality_rets = np.random.randn(n_periods) * 0.007 + 0.0004
    
    # LowVol: Low return, very low vol
    lowvol_rets = np.random.randn(n_periods) * 0.005 + 0.0002
    
    # Growth: High beta, volatile
    growth_rets = np.random.randn(n_periods) * 0.018 + 0.0006
    
    factor_returns = pd.DataFrame({
        'Value': value_rets,
        'Momentum': momentum_rets,
        'Quality': quality_rets,
        'LowVol': lowvol_rets,
        'Growth': growth_rets
    }, index=dates)
    
    # Generate factor loadings (exposures)
    loadings = pd.DataFrame(
        np.random.randn(n_stocks, n_factors),
        index=tickers,
        columns=factors
    )
    
    # Add some structure to loadings
    # Value stocks have positive Value loading
    value_stocks = np.random.choice(tickers, size=20, replace=False)
    loadings.loc[value_stocks, 'Value'] += 1.0
    
    # Growth stocks have positive Growth loading
    growth_stocks = np.random.choice(tickers, size=20, replace=False)
    loadings.loc[growth_stocks, 'Growth'] += 1.0
    
    return loadings, factor_returns


def generate_forward_returns(loadings, horizon=21):
    """Generate synthetic forward returns for IC calculation."""
    np.random.seed(43)
    
    # Simulate forward returns based on factor exposures
    fwd_rets = (
        loadings['Value'] * 0.02 +
        loadings['Momentum'] * 0.015 +
        loadings['Quality'] * 0.01 +
        loadings['LowVol'] * 0.005 -
        loadings['Growth'] * 0.01 +
        np.random.randn(len(loadings)) * 0.05
    )
    
    return fwd_rets


def print_weights_table(weights_dict, title="Factor Weights"):
    """Pretty print weights comparison."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    print(f"{'Method':<25} {'Value':>8} {'Mom':>8} {'Qual':>8} {'LowV':>8} {'Grow':>8}")
    print(f"{'-'*60}")
    
    for method, weights in weights_dict.items():
        print(f"{method:<25} " + 
              f"{weights.get('Value', 0):>7.2%} " +
              f"{weights.get('Momentum', 0):>7.2%} " +
              f"{weights.get('Quality', 0):>7.2%} " +
              f"{weights.get('LowVol', 0):>7.2%} " +
              f"{weights.get('Growth', 0):>7.2%}")
    print(f"{'='*60}")


def print_characteristics_table(chars_dict):
    """Pretty print factor characteristics."""
    print(f"\n{'='*80}")
    print(f"{'Factor Characteristics':^80}")
    print(f"{'='*80}")
    print(f"{'Factor':<12} {'Sharpe':>10} {'Vol':>10} {'Mean Ret':>12} {'Max DD':>10} {'Win Rate':>10}")
    print(f"{'-'*80}")
    
    for name, char in chars_dict.items():
        print(f"{name:<12} " +
              f"{char.sharpe_ratio:>9.2f} " +
              f"{char.volatility:>9.2%} " +
              f"{char.mean_return:>11.4%} " +
              f"{char.max_drawdown:>9.2%} " +
              f"{char.win_rate:>9.1%}")
    print(f"{'='*80}")


def main():
    """Run the optimal factor weighting demonstration."""
    print("\n" + "="*80)
    print("OPTIMAL FACTOR WEIGHTING DEMONSTRATION")
    print("="*80)
    
    # Generate sample data
    print("\n1. Generating sample factor data...")
    loadings, factor_returns = generate_sample_data(n_stocks=100, n_factors=5)
    print(f"   Loadings shape: {loadings.shape}")
    print(f"   Returns shape: {factor_returns.shape}")
    
    # Initialize weighter
    print("\n2. Initializing OptimalFactorWeighter...")
    weighter = OptimalFactorWeighter(loadings, factor_returns)
    
    # Calculate factor characteristics
    print("\n3. Calculating factor characteristics...")
    forward_returns = generate_forward_returns(loadings)
    characteristics = weighter.get_factor_characteristics(
        lookback=63,
        forward_returns=forward_returns
    )
    print_characteristics_table(characteristics)
    
    # Compare different weighting methods
    print("\n4. Comparing optimal weighting methods...")
    all_weights = {}
    
    # 1. Equal weights (baseline)
    print("   Calculating equal weights...")
    all_weights['Equal Weight'] = weighter.equal_weights()
    
    # 2. Sharpe weights
    print("   Calculating Sharpe weights...")
    all_weights['Sharpe Ratio'] = weighter.sharpe_weights(lookback=63)
    
    # 3. IC weights
    print("   Calculating IC weights...")
    all_weights['Information Coeff.'] = weighter.ic_weights(forward_returns)
    
    # 4. Momentum weights
    print("   Calculating momentum weights...")
    all_weights['Factor Momentum'] = weighter.momentum_weights(lookback=21)
    
    # 5. Risk parity
    print("   Calculating risk parity weights...")
    all_weights['Risk Parity'] = weighter.risk_parity_weights(lookback=63)
    
    # 6. Min variance
    print("   Calculating min variance weights...")
    all_weights['Minimum Variance'] = weighter.min_variance_weights(lookback=63)
    
    # 7. Max diversification
    print("   Calculating max diversification weights...")
    all_weights['Max Diversification'] = weighter.max_diversification_weights(lookback=63)
    
    # 8. PCA weights
    print("   Calculating PCA weights...")
    all_weights['PCA'] = weighter.pca_weights(variance_threshold=0.95)
    
    # Print comparison
    print_weights_table(all_weights, "Weight Comparison by Method")
    
    # Demonstrate blended approach
    print("\n5. Demonstrating blended weighting approach...")
    blended = weighter.blend_weights(
        {
            'sharpe': 0.3,
            'ic': 0.3,
            'momentum': 0.2,
            'risk_parity': 0.2
        },
        forward_returns=forward_returns
    )
    
    print("\n   Blended weights (Sharpe 30%, IC 30%, Momentum 20%, Risk Parity 20%):")
    for factor, weight in sorted(blended.items()):
        bar = "█" * int(weight * 50)
        print(f"   {factor:<12} {weight:>6.2%} {bar}")
    
    # Demonstrate cross-sectional portfolio construction
    print("\n6. Constructing cross-sectional portfolios...")
    
    # Create analyzer
    analyzer = CrossSectionalAnalyzer(loadings)
    
    # Compare equal-weighted vs optimal-weighted portfolios
    print("\n   A. Equal-weighted factor composite:")
    equal_scores = analyzer.calculate_factor_scores(method='weighted_sum')
    equal_rankings = analyzer.rank_universe(equal_scores)
    equal_longs = equal_rankings[equal_rankings['decile'] == 1].index.tolist()
    print(f"      Top decile (longs): {len(equal_longs)} stocks")
    print(f"      Example longs: {', '.join(equal_longs[:5])}")
    
    print("\n   B. Sharpe-weighted factor composite:")
    sharpe_scores = analyzer.calculate_factor_scores(
        weights=all_weights['Sharpe Ratio'],
        method='weighted_sum'
    )
    sharpe_rankings = analyzer.rank_universe(sharpe_scores)
    sharpe_longs = sharpe_rankings[sharpe_rankings['decile'] == 1].index.tolist()
    print(f"      Top decile (longs): {len(sharpe_longs)} stocks")
    print(f"      Example longs: {', '.join(sharpe_longs[:5])}")
    
    # Check overlap
    overlap = set(equal_longs) & set(sharpe_longs)
    print(f"\n   Overlap between methods: {len(overlap)} stocks ({len(overlap)/len(equal_longs):.0%})")
    
    # Momentum-adjusted approach
    print("\n7. Dynamic momentum-adjusted weights (changing over time)...")
    print("   Simulating 3-month rebalancing:")
    
    months = ['Jan', 'Feb', 'Mar']
    for i, month in enumerate(months):
        # Shift returns to simulate different periods
        if i == 0:
            period_returns = factor_returns
        else:
            period_returns = factor_returns.shift(-i * 21)
        
        period_weighter = OptimalFactorWeighter(loadings, period_returns)
        
        # Blend Sharpe and momentum (adapt to recent performance)
        period_weights = period_weighter.blend_weights(
            {'sharpe': 0.5, 'momentum': 0.5},
            lookback=21
        )
        
        print(f"\n   {month}: Momentum takes over from weak factors")
        for factor, weight in sorted(period_weights.items(), key=lambda x: -x[1])[:3]:
            print(f"      {factor}: {weight:.1%}")
    
    print("\n" + "="*80)
    print("CONCLUSIONS")
    print("="*80)
    print("""
1. SHARPE WEIGHTING: Favors Value and Quality (consistent risk-adjusted returns)
   
2. IC WEIGHTING: Favors factors with strongest predictive power for your universe
   
3. MOMENTUM WEIGHTING: Dynamic, adapts to recent factor performance
   
4. RISK PARITY: Balances contribution, penalizes Momentum's high volatility
   
5. MIN VARIANCE: Overweights LowVol, underweights Growth
   
6. BLENDED APPROACH: Combines benefits, reduces single-method risk

RECOMMENDED WORKFLOW:
- Use SHARPE for long-term strategic allocation
- Use IC for stock selection (if you have forward returns data)
- Use MOMENTUM for tactical tilts
- Blend methods for robustness (e.g., 40% Sharpe, 30% IC, 30% Momentum)
    """)


if __name__ == '__main__':
    main()
