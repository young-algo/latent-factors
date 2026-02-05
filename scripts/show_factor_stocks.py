#!/usr/bin/env python
"""
Quick script to show representative stocks for best/worst performing factors.
"""

import pandas as pd
import sys
import argparse
from pathlib import Path

def show_factor_stocks(n=5, days=14):
    """Show top and bottom n factors with their representative stocks."""
    
    # Load performance data
    perf_file = f'factor_performance_{days}d.csv'
    try:
        perf = pd.read_csv(perf_file)
    except FileNotFoundError:
        print(f" Error: {perf_file} not found. Run quick_factor_performance.py -d {days} first")
        # List available files
        available = list(Path('.').glob('factor_performance_*d.csv'))
        if available:
            print("\n Available performance files:")
            for f in sorted(available):
                print(f"   - {f}")
        return
    
    # Load factor summary with stocks
    try:
        summary = pd.read_csv('factor_summary.csv')
    except FileNotFoundError:
        print(" Error: factor_summary.csv not found")
        return
    
    print(f" FACTOR PERFORMANCE WITH REPRESENTATIVE STOCKS ({days}-Day)")
    print("=" * 80)
    
    # Best performers
    print(f"\n TOP {n} PERFORMING FACTORS:")
    print("-" * 80)
    
    for i in range(min(n, len(perf))):
        factor = perf.iloc[i]['Factor']
        name = perf.iloc[i]['Name']
        returns = perf.iloc[i]['Return_bps']
        
        # Find matching factor in summary
        factor_info = summary[summary['Factor'] == factor]
        if not factor_info.empty:
            top_long = factor_info.iloc[0]['Top_Long']
            top_short = factor_info.iloc[0]['Top_Short']
            
            print(f"\n{i+1}. {factor}: {name}")
            print(f"   Return: {returns:+.1f} bps")
            print(f"   Long:  {top_long}")
            print(f"   Short: {top_short}")
    
    # Worst performers
    print(f"\n\n BOTTOM {n} PERFORMING FACTORS:")
    print("-" * 80)
    
    for i in range(min(n, len(perf))):
        idx = len(perf) - n + i
        if idx < 0:
            continue
            
        factor = perf.iloc[idx]['Factor']
        name = perf.iloc[idx]['Name']
        returns = perf.iloc[idx]['Return_bps']
        
        # Find matching factor in summary
        factor_info = summary[summary['Factor'] == factor]
        if not factor_info.empty:
            top_long = factor_info.iloc[0]['Top_Long']
            top_short = factor_info.iloc[0]['Top_Short']
            
            print(f"\n{i+1}. {factor}: {name}")
            print(f"   Return: {returns:+.1f} bps")
            print(f"   Long:  {top_long}")
            print(f"   Short: {top_short}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show representative stocks for best/worst performing factors")
    parser.add_argument("-n", "--number", type=int, default=5, 
                        help="Number of top/bottom factors to show (default: 5)")
    parser.add_argument("-d", "--days", type=int, default=14,
                        help="Time horizon in days (default: 14)")
    
    args = parser.parse_args()
    show_factor_stocks(n=args.number, days=args.days)