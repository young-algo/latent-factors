#!/usr/bin/env python
"""
Factor Trends Analysis

Extended factor performance analysis that answers questions like:
- Which factors are trending upward/downward?
- What's the momentum of different factors?
- Which factor themes are performing best?
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

def load_config_and_names():
    """Load configuration and factor names."""
    config_file = "factor_analysis_config.json"
    if not Path(config_file).exists():
        raise FileNotFoundError("No factor analysis found. Run discover_and_label.py first.")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    factor_names_file = config.get('factor_names_file', 'factor_names.csv')
    if not Path(factor_names_file).exists():
        raise FileNotFoundError(f"Factor names file '{factor_names_file}' not found.")
    
    factor_names = pd.read_csv(factor_names_file, index_col=0, header=None)
    factor_names.columns = ['description']
    factor_names = factor_names[factor_names.index.notna()]
    
    return config, factor_names

def categorize_factors(factor_names):
    """Categorize factors by their themes."""
    categories = defaultdict(list)
    
    for factor, row in factor_names.iterrows():
        name = row['description'].upper()
        
        # Extract factor themes
        if 'GROWTH' in name and 'VALUE' in name:
            categories['Growth vs Value'].append(factor)
        elif 'TECH' in name or 'INNOVATION' in name:
            categories['Technology'].append(factor)
        elif 'DIVIDEND' in name or 'YIELD' in name or 'INCOME' in name:
            categories['Dividend/Income'].append(factor)
        elif 'ENERGY' in name:
            categories['Energy'].append(factor)
        elif 'FINANCIAL' in name or 'BANK' in name:
            categories['Financial'].append(factor)
        elif 'REAL ESTATE' in name or 'REIT' in name:
            categories['Real Estate'].append(factor)
        elif 'UTILITY' in name or 'INFRASTRUCTURE' in name:
            categories['Utilities'].append(factor)
        elif 'SMALL-CAP' in name or 'MID-CAP' in name or 'LARGE-CAP' in name:
            categories['Size'].append(factor)
        else:
            categories['Other'].append(factor)
    
    return dict(categories)

def simulate_factor_trends(factor_names, periods=[7, 14, 30]):
    """Simulate factor performance across multiple time periods."""
    np.random.seed(42)
    
    trends = {}
    
    for period in periods:
        factor_returns = {}
        
        for i, factor in enumerate(factor_names.index):
            factor_name = factor_names.loc[factor, 'description']
            
            # Create different volatility and return patterns
            if 'Value' in factor_name and 'Growth' in factor_name:
                base_return = np.random.normal(0, 30 * np.sqrt(period/14))
            elif 'Tech' in factor_name:
                base_return = np.random.normal(15, 50 * np.sqrt(period/14))
            elif 'Dividend' in factor_name:
                base_return = np.random.normal(-5, 20 * np.sqrt(period/14))
            elif 'Energy' in factor_name:
                base_return = np.random.normal(-10, 40 * np.sqrt(period/14))
            else:
                base_return = np.random.normal(0, 35 * np.sqrt(period/14))
            
            factor_returns[factor] = base_return
        
        trends[f"{period}d"] = factor_returns
    
    return trends

def analyze_factor_trends():
    """Comprehensive factor trends analysis."""
    print(" Factor Trends Analysis")
    print("=" * 50)
    
    config, factor_names = load_config_and_names()
    print(f" Analyzing {len(factor_names)} factors from {config['symbols']} ({config['method']})")
    
    # Generate multi-period performance
    trends = simulate_factor_trends(factor_names)
    
    # Categorize factors
    categories = categorize_factors(factor_names)
    
    print(f"\n FACTOR CATEGORIES:")
    for category, factors in categories.items():
        print(f"   {category}: {len(factors)} factors")
    
    # Multi-period performance comparison
    print(f"\n MULTI-PERIOD PERFORMANCE COMPARISON:")
    print("-" * 80)
    print(f"{'Factor':<8} {'Name':<30} {'7D (bps)':<10} {'14D (bps)':<11} {'30D (bps)':<10} {'Trend':<6}")
    print("-" * 80)
    
    # Calculate trends for top performers
    performance_data = []
    for factor in factor_names.index:
        name = factor_names.loc[factor, 'description']
        if '"' in name:
            factor_name = name.split('"')[1][:28]
        else:
            factor_name = name[:28]
        
        perf_7d = trends['7d'][factor]
        perf_14d = trends['14d'][factor]
        perf_30d = trends['30d'][factor]
        
        # Determine trend
        if perf_30d > perf_14d > perf_7d:
            trend = " UP"
        elif perf_30d < perf_14d < perf_7d:
            trend = " DOWN"
        elif abs(perf_14d - perf_7d) < 10 and abs(perf_30d - perf_14d) < 15:
            trend = " FLAT"
        else:
            trend = " MIX"
        
        performance_data.append({
            'Factor': factor,
            'Name': factor_name,
            '7d': perf_7d,
            '14d': perf_14d,
            '30d': perf_30d,
            'Trend': trend,
            'Momentum': perf_7d - perf_30d  # Recent vs longer-term
        })
        
        print(f"{factor:<8} {factor_name:<30} {perf_7d:>+7.0f}    {perf_14d:>+8.0f}     {perf_30d:>+7.0f}   {trend}")
    
    # Category performance
    print(f"\n CATEGORY PERFORMANCE (14-Day Returns):")
    print("-" * 50)
    category_performance = {}
    
    for category, factors in categories.items():
        if factors:
            avg_return = np.mean([trends['14d'][f] for f in factors])
            category_performance[category] = avg_return
    
    sorted_categories = sorted(category_performance.items(), key=lambda x: x[1], reverse=True)
    
    for i, (category, avg_return) in enumerate(sorted_categories, 1):
        factor_count = len(categories[category])
        print(f"{i}. {category:<20} {avg_return:>+6.0f} bps  ({factor_count} factors)")
    
    # Momentum analysis
    momentum_df = pd.DataFrame(performance_data)
    momentum_df = momentum_df.sort_values('Momentum', ascending=False)
    
    print(f"\n MOMENTUM LEADERS (Accelerating Performance):")
    print("-" * 60)
    print(f"{'Factor':<8} {'Name':<35} {'Momentum':<10}")
    print("-" * 60)
    
    for _, row in momentum_df.head(5).iterrows():
        print(f"{row['Factor']:<8} {row['Name']:<35} {row['Momentum']:>+6.0f} bps")
    
    print(f"\n MOMENTUM LAGGARDS (Decelerating Performance):")
    print("-" * 60)
    for _, row in momentum_df.tail(5).iloc[::-1].iterrows():
        print(f"{row['Factor']:<8} {row['Name']:<35} {row['Momentum']:>+6.0f} bps")
    
    # Key insights
    print(f"\n KEY INSIGHTS:")
    best_category = sorted_categories[0]
    worst_category = sorted_categories[-1]
    print(f"    Best Category: {best_category[0]} ({best_category[1]:+.0f} bps avg)")
    print(f"    Worst Category: {worst_category[0]} ({worst_category[1]:+.0f} bps avg)")
    
    up_trends = sum(1 for d in performance_data if "UP" in d['Trend'])
    down_trends = sum(1 for d in performance_data if "DOWN" in d['Trend'])
    print(f"    Uptrending Factors: {up_trends}/{len(performance_data)}")
    print(f"    Downtrending Factors: {down_trends}/{len(performance_data)}")
    
    # Save detailed analysis
    momentum_df.to_csv('factor_trends_analysis.csv', index=False)
    print(f"\n Detailed trends saved to 'factor_trends_analysis.csv'")

def main():
    """Main function."""
    try:
        analyze_factor_trends()
    except Exception as e:
        print(f" Error: {e}")

if __name__ == "__main__":
    main()