#!/usr/bin/env python3
"""
Factor Analysis Tool: View Representative Stocks by Factor
=========================================================

This script analyzes the results from discover_and_label.py to show
which stocks are most representative of each discovered factor.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add src directory to path
sys.path.append('./src')
from research import FactorResearchSystem

def analyze_factor_composition(factor_names_file="factor_names.csv", n_stocks=10):
    """
    Analyze factor composition and show representative stocks.
    
    Parameters
    ----------
    factor_names_file : str
        Path to CSV file with factor names from discover_and_label.py
    n_stocks : int
        Number of top stocks to show per factor
    """
    
    print("🔍 Analyzing Factor Composition")
    print("=" * 50)
    
    # Load configuration from discover_and_label.py run
    config_file = "factor_analysis_config.json"
    if not Path(config_file).exists():
        print(f"❌ Configuration file '{config_file}' not found!")
        print("   This file is created automatically by discover_and_label.py")
        print("   Run discover_and_label.py first, then this analysis script.")
        return
    
    try:
        import json
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"✅ Loaded configuration from previous run:")
        print(f"   Symbols: {config['symbols']}")
        print(f"   Method: {config['method']}")
        print(f"   Factors: {config['k']}")
        print(f"   Start Date: {config['start_date']}")
        print(f"   Resolved to: {config['resolved_symbols']} stocks")
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return
    
    # Check if factor names file exists
    factor_names_file = config.get('factor_names_file', factor_names_file)
    if not Path(factor_names_file).exists():
        print(f"❌ Factor names file '{factor_names_file}' not found!")
        return
    
    # Load factor names
    try:
        factor_names = pd.read_csv(factor_names_file, index_col=0, header=None)
        factor_names.columns = ['description']
        print(f"✅ Loaded {len(factor_names)} factor names")
    except Exception as e:
        print(f"❌ Error loading factor names: {e}")
        return
    
    # Re-run the factor discovery to get loadings
    print("\n🔄 Re-running factor discovery to get loadings...")
    
    API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
    if not API_KEY:
        print("❌ ALPHAVANTAGE_API_KEY environment variable required")
        return
    
    # Extract symbols and convert to list
    symbols_str = config['symbols']
    symbols_list = [s.strip().upper() for s in symbols_str.split(',')]
    
    # Initialize system with the same parameters used in discover_and_label
    method_map = {
        'PCA': 'pca',
        'ICA': 'ica', 
        'NMF': 'nmf',
        'AE': 'autoencoder'
    }
    factor_method = method_map.get(config['method'], 'pca')
    
    frs = FactorResearchSystem(API_KEY, 
                              universe=symbols_list, 
                              factor_method=factor_method,
                              start_date=config['start_date'])
    
    try:
        # Load the symbols that were used (using the same symbols from config)
        symbols = frs._resolve_symbols(symbols_list)  # Expands ETFs to constituents if needed
        prices = frs.get_prices(symbols)
        returns = prices.pct_change().dropna()
        
        # Re-run factor discovery with same parameters
        from latent_factors import statistical_factors, StatMethod
        
        # Map method name to StatMethod enum
        method_enum_map = {
            'PCA': StatMethod.PCA,
            'ICA': StatMethod.ICA,
            'NMF': StatMethod.NMF
        }
        
        if config['method'] == 'AE':
            from latent_factors import autoencoder_factors
            factor_returns, loadings = autoencoder_factors(returns, k=config['k'])
        else:
            method_enum = method_enum_map.get(config['method'], StatMethod.PCA)
            factor_returns, loadings = statistical_factors(returns, 
                                                          n_components=config['k'], 
                                                          method=method_enum)
        
        print(f"✅ Factor loadings shape: {loadings.shape}")
        
        # Analyze each factor
        print("\n📊 Factor Composition Analysis")
        print("=" * 60)
        
        for i, (factor_id, row) in enumerate(factor_names.iterrows()):
            factor_name = row['description']
            factor_col = loadings.columns[i]
            
            print(f"\n🎯 Factor {i+1}: {factor_id}")
            print(f"📝 Description: {factor_name}")
            print("-" * 50)
            
            # Get factor loadings for this factor
            factor_loadings = loadings[factor_col]
            
            # Top positive exposures (long positions)
            top_positive = factor_loadings.nlargest(n_stocks)
            print(f"📈 TOP {n_stocks} POSITIVE EXPOSURES (Long Positions):")
            for j, (stock, loading) in enumerate(top_positive.items(), 1):
                print(f"   {j:2d}. {stock:6s} ({loading:+.4f})")
            
            # Top negative exposures (short positions)  
            top_negative = factor_loadings.nsmallest(n_stocks)
            print(f"\n📉 TOP {n_stocks} NEGATIVE EXPOSURES (Short Positions):")
            for j, (stock, loading) in enumerate(top_negative.items(), 1):
                print(f"   {j:2d}. {stock:6s} ({loading:+.4f})")
            
            # Summary statistics
            print(f"\n📊 Factor Statistics:")
            print(f"   Range: {factor_loadings.min():.4f} to {factor_loadings.max():.4f}")
            print(f"   Std Dev: {factor_loadings.std():.4f}")
            print(f"   Stocks with |loading| > 0.01: {(factor_loadings.abs() > 0.01).sum()}")
            
        print(f"\n✅ Analysis complete! Analyzed {len(factor_names)} factors with {loadings.shape[0]} stocks.")
            
    except Exception as e:
        print(f"❌ Error during factor analysis: {e}")
        import traceback
        traceback.print_exc()

def quick_factor_summary(factor_names_file="factor_names.csv"):
    """Quick summary showing just the top 3 stocks per factor"""
    
    if not Path(factor_names_file).exists():
        print(f"❌ Run discover_and_label.py first to generate {factor_names_file}")
        return
        
    # Load factor names
    factor_names = pd.read_csv(factor_names_file, index_col=0, header=None)
    factor_names.columns = ['description']
    
    print("📋 Quick Factor Summary")
    print("=" * 40)
    
    for i, (factor_id, row) in enumerate(factor_names.iterrows()):
        factor_name = row['description'].split(':')[0] if ':' in row['description'] else row['description']
        print(f"{i+1:2d}. {factor_id}: {factor_name}")
    
    print(f"\n💡 Run with full analysis: python analyze_factors.py")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_factor_summary()
    else:
        # Full analysis
        n_stocks = 8  # Number of stocks to show per factor
        if len(sys.argv) > 1:
            try:
                n_stocks = int(sys.argv[1])
            except:
                print("Usage: python analyze_factors.py [n_stocks] or --quick")
                sys.exit(1)
        
        analyze_factor_composition(n_stocks=n_stocks)