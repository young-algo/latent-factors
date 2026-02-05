#!/usr/bin/env python
"""
Quick Factor Performance Analysis

This script demonstrates how to analyze factor performance using the
existing factor analysis results. It shows recent performance of factors
discovered in the most recent analysis.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def load_config_and_names():
    """Load configuration and factor names."""
    # Load config
    config_file = "factor_analysis_config.json"
    if not Path(config_file).exists():
        raise FileNotFoundError("No factor analysis found. Run discover_and_label.py first.")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Load factor names
    factor_names_file = config.get('factor_names_file', 'factor_names.csv')
    if not Path(factor_names_file).exists():
        raise FileNotFoundError(f"Factor names file '{factor_names_file}' not found.")
    
    factor_names = pd.read_csv(factor_names_file, index_col=0, header=None)
    factor_names.columns = ['description']
    factor_names = factor_names[factor_names.index.notna()]
    
    return config, factor_names

def analyze_recent_performance(num_factors, period_days=14):
    """
    Analyze actual factor performance from saved factor returns.
    Uses real factor returns from the most recent analysis.
    """
    print(f" Factor Performance Analysis: Last {period_days} Days")
    print("=" * 60)
    
    config, factor_names = load_config_and_names()
    
    print(f" Analysis based on {config['method']} factors from {config['symbols']}")
    print(f" Loaded {len(factor_names)} factor names")
    
    # Load actual factor returns
    factor_returns_file = 'factor_returns.csv'
    if not Path(factor_returns_file).exists():
        raise FileNotFoundError(f"Factor returns file '{factor_returns_file}' not found. Run discover_and_label.py first.")
    
    factor_returns_df = pd.read_csv(factor_returns_file, index_col=0, parse_dates=True)
    
    # Calculate recent performance
    end_date = factor_returns_df.index.max()
    start_date = end_date - timedelta(days=period_days)
    
    # Filter to recent period
    recent_returns = factor_returns_df[factor_returns_df.index >= start_date]
    
    if len(recent_returns) == 0:
        raise ValueError(f"No factor returns found for the last {period_days} days")
    
    # Calculate cumulative returns for the period (in basis points)
    factor_returns = {}
    for factor in factor_names.index:
        if factor in recent_returns.columns:
            period_return = (1 + recent_returns[factor]).prod() - 1
            factor_returns[factor] = period_return * 10000  # Convert to basis points
        else:
            print(f"Warning: Factor {factor} not found in returns data")
    
    # Create performance summary
    performance_data = []
    for factor in factor_names.index:
        name = factor_names.loc[factor, 'description']
        # Extract factor name from description
        if '"' in name:
            factor_name = name.split('"')[1]
        else:
            factor_name = name[:40]
        
        performance_data.append({
            'Factor': factor,
            'Name': factor_name,
            'Return_bps': factor_returns[factor]
        })
    
    performance_summary = pd.DataFrame(performance_data)
    performance_summary = performance_summary.sort_values('Return_bps', ascending=False)
    
    # Actual date range from data
    
    print(f"\n Analysis Period:")
    print(f"   Start: {start_date.strftime('%Y-%m-%d')}")
    print(f"   End: {end_date.strftime('%Y-%m-%d')}")
    print(f"   Duration: {period_days} days")
    
    print(f"\n BEST PERFORMING FACTORS (Last {period_days} Days):")
    print("-" * 75)
    print(f"{'Rank':<4} {'Factor':<8} {'Return (bps)':<12} {'Name':<45}")
    print("-" * 75)
    
    for i, (_, row) in enumerate(performance_summary.head(5).iterrows(), 1):
        print(f"{i:<4} {row['Factor']:<8} {row['Return_bps']:>+8.0f}      {row['Name']:<45}")
    
    print(f"\n WORST PERFORMING FACTORS (Last {period_days} Days):")
    print("-" * 75)
    print(f"{'Rank':<4} {'Factor':<8} {'Return (bps)':<12} {'Name':<45}")
    print("-" * 75)
    
    for i, (_, row) in enumerate(performance_summary.tail(5).iloc[::-1].iterrows(), 1):
        print(f"{i:<4} {row['Factor']:<8} {row['Return_bps']:>+8.0f}      {row['Name']:<45}")
    
    # Summary statistics
    print(f"\n SUMMARY STATISTICS (Last {period_days} Days):")
    print(f"   Best Factor Return: {performance_summary['Return_bps'].max():+.0f} bps ({performance_summary.iloc[0]['Factor']})")
    print(f"   Worst Factor Return: {performance_summary['Return_bps'].min():+.0f} bps ({performance_summary.iloc[-1]['Factor']})")
    print(f"   Average Return: {performance_summary['Return_bps'].mean():+.1f} bps")
    print(f"   Return Spread: {performance_summary['Return_bps'].max() - performance_summary['Return_bps'].min():.0f} bps")
    print(f"   Positive Factors: {(performance_summary['Return_bps'] > 0).sum()}/{len(performance_summary)}")
    print(f"   Negative Factors: {(performance_summary['Return_bps'] < 0).sum()}/{len(performance_summary)}")
    
    # Top factor categories
    print(f"\n FACTOR INSIGHTS:")
    best_factor = performance_summary.iloc[0]
    worst_factor = performance_summary.iloc[-1]
    print(f"    Best: {best_factor['Name']} ({best_factor['Return_bps']:+.0f} bps)")
    print(f"    Worst: {worst_factor['Name']} ({worst_factor['Return_bps']:+.0f} bps)")
    
    # Save results
    performance_summary.to_csv(f'factor_performance_{period_days}d.csv', index=False)
    print(f"\n Detailed results saved to 'factor_performance_{period_days}d.csv'")
    
    print(f"\n NOTE: This analysis uses actual factor returns from your recent analysis.")
    print(f"   Data based on {len(recent_returns)} days of factor return history.")
    print(f"   Factor returns calculated from {config.get('resolved_symbols', 'N/A')} stocks.")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick factor performance analysis")
    parser.add_argument("-d", "--days", type=int, default=14, help="Days to analyze (default: 14)")
    args = parser.parse_args()
    
    try:
        config, factor_names = load_config_and_names()
        analyze_recent_performance(len(factor_names), period_days=args.days)
    except Exception as e:
        print(f" Error: {e}")

if __name__ == "__main__":
    main()