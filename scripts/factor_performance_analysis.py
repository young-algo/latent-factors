#!/usr/bin/env python
"""
Factor Performance Analysis: Recent Performance Tracking
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add src directory to path
sys.path.append('src')

from research import FactorResearchSystem
from latent_factors import statistical_factors, autoencoder_factors, StatMethod

def load_config():
    """Load the most recent factor analysis configuration."""
    config_file = "factor_analysis_config.json"
    if not Path(config_file).exists():
        raise FileNotFoundError(f"No factor analysis found. Run discover_and_label.py first.")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def load_factor_names():
    """Load factor names from the most recent analysis."""
    config = load_config()
    factor_names_file = config.get('factor_names_file', 'factor_names.csv')
    
    if not Path(factor_names_file).exists():
        raise FileNotFoundError(f"Factor names file '{factor_names_file}' not found.")
    
    factor_names = pd.read_csv(factor_names_file, index_col=0, header=None)
    factor_names.columns = ['description']
    # Filter out nan index entries from CSV parsing
    factor_names = factor_names[factor_names.index.notna()]
    return factor_names

def regenerate_factor_returns():
    """Regenerate factor returns from the most recent analysis configuration."""
    config = load_config()
    
    API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
    if not API_KEY:
        raise ValueError("ALPHAVANTAGE_API_KEY environment variable required")
    
    # Extract symbols and convert to list
    symbols_str = config['symbols']
    symbols_list = [s.strip().upper() for s in symbols_str.split(',')]
    
    print(f"🔄 Regenerating factor returns for {config['method']} analysis...")
    print(f"   Universe: {symbols_str}")
    print(f"   Method: {config['method']}")
    print(f"   Factors: {config['k']}")
    
    # Initialize system
    frs = FactorResearchSystem(API_KEY, universe=symbols_list, start_date=config['start_date'])
    prices = frs.get_prices(frs._resolve_symbols(symbols_list))
    returns = prices.pct_change().dropna()
    
    # Apply factor discovery method
    if config['method'] == "AE":
        factor_ret, loadings = autoencoder_factors(returns, k=config['k'])
    else:
        factor_ret, loadings = statistical_factors(returns,
                                                 n_components=config['k'],
                                                 method=StatMethod[config['method']])
    
    print(f"✅ Generated factor returns: {factor_ret.shape}")
    return factor_ret

def analyze_factor_performance(period_days=14):
    """Analyze factor performance over a specified period."""
    print(f"📊 Factor Performance Analysis: Last {period_days} Days")
    print("=" * 60)
    
    # Load factor names
    try:
        factor_names = load_factor_names()
        print(f"✅ Loaded {len(factor_names)} factor names")
    except Exception as e:
        print(f"❌ Error loading factor names: {e}")
        return
    
    # Generate factor returns
    try:
        factor_returns = regenerate_factor_returns()
    except Exception as e:
        print(f"❌ Error generating factor returns: {e}")
        return
    
    # Calculate recent period performance
    end_date = factor_returns.index.max()
    start_date = end_date - timedelta(days=period_days)
    
    print(f"\n📅 Analysis Period:")
    print(f"   Start: {start_date.strftime('%Y-%m-%d')}")
    print(f"   End: {end_date.strftime('%Y-%m-%d')}")
    
    # Filter to recent period
    recent_returns = factor_returns[factor_returns.index >= start_date]
    
    if len(recent_returns) == 0:
        print(f"❌ No data available for the last {period_days} days")
        return
    
    print(f"✅ Found {len(recent_returns)} days of recent data")
    
    # Calculate performance metrics
    total_returns = recent_returns.sum() * 100  # Convert to basis points
    volatility = recent_returns.std() * np.sqrt(252) * 100  # Annualized volatility
    
    # Create performance summary
    performance_data = []
    for factor in factor_names.index:
        if factor in total_returns.index:
            name = factor_names.loc[factor, 'description']
            # Extract factor name from description (between quotes)
            if '"' in name:
                factor_name = name.split('"')[1]
            else:
                factor_name = name[:40]
            
            performance_data.append({
                'Factor': factor,
                'Name': factor_name,
                'Return_bps': total_returns[factor],
                'Volatility': volatility[factor]
            })
    
    performance_summary = pd.DataFrame(performance_data)
    performance_summary = performance_summary.sort_values('Return_bps', ascending=False)
    
    print(f"\n🏆 BEST PERFORMING FACTORS (Last {period_days} Days):")
    print("-" * 70)
    print(f"{'Rank':<4} {'Factor':<8} {'Return (bps)':<12} {'Name':<40}")
    print("-" * 70)
    
    for i, (_, row) in enumerate(performance_summary.head(5).iterrows(), 1):
        print(f"{i:<4} {row['Factor']:<8} {row['Return_bps']:>+8.0f}      {row['Name']:<40}")
    
    print(f"\n📉 WORST PERFORMING FACTORS (Last {period_days} Days):")
    print("-" * 70)
    print(f"{'Rank':<4} {'Factor':<8} {'Return (bps)':<12} {'Name':<40}")
    print("-" * 70)
    
    for i, (_, row) in enumerate(performance_summary.tail(5).iloc[::-1].iterrows(), 1):
        print(f"{i:<4} {row['Factor']:<8} {row['Return_bps']:>+8.0f}      {row['Name']:<40}")
    
    # Summary statistics
    print(f"\n📈 SUMMARY STATISTICS (Last {period_days} Days):")
    print(f"   Best Factor Return: {performance_summary['Return_bps'].max():+.0f} bps")
    print(f"   Worst Factor Return: {performance_summary['Return_bps'].min():+.0f} bps")
    print(f"   Average Return: {performance_summary['Return_bps'].mean():+.1f} bps")
    print(f"   Positive Factors: {(performance_summary['Return_bps'] > 0).sum()}")
    print(f"   Negative Factors: {(performance_summary['Return_bps'] < 0).sum()}")
    
    # Save results
    performance_summary.to_csv(f'factor_performance_{period_days}d.csv', index=False)
    print(f"\n💾 Results saved to 'factor_performance_{period_days}d.csv'")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze factor performance")
    parser.add_argument("-d", "--days", type=int, default=14, help="Days to analyze")
    args = parser.parse_args()
    
    try:
        analyze_factor_performance(period_days=args.days)
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()