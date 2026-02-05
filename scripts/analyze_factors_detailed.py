#!/usr/bin/env python3
"""
Detailed Factor Analysis: Stocks with Fundamental Context
========================================================

Enhanced factor analysis showing representative stocks with their
fundamental characteristics (sector, market cap, P/E ratio, etc.)
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.append('./src')
from research import FactorResearchSystem
from latent_factors import statistical_factors, StatMethod

def detailed_factor_analysis(factor_names_file="factor_names.csv", n_stocks=5):
    """
    Analyze factors with fundamental context for each stock.
    """
    
    print(" Detailed Factor Analysis with Fundamentals")
    print("=" * 55)
    
    # Load configuration from discover_and_label.py run
    config_file = "factor_analysis_config.json"
    if not Path(config_file).exists():
        print(f" Configuration file '{config_file}' not found!")
        print("   This file is created automatically by discover_and_label.py")
        print("   Run discover_and_label.py first, then this analysis script.")
        return
    
    try:
        import json
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f" Loaded configuration from previous run:")
        print(f"   Symbols: {config['symbols']}")
        print(f"   Method: {config['method']}")
        print(f"   Factors: {config['k']}")
    except Exception as e:
        print(f" Error loading configuration: {e}")
        return
    
    # Load factor names
    factor_names_file = config.get('factor_names_file', factor_names_file)
    if not Path(factor_names_file).exists():
        print(f" Factor names file '{factor_names_file}' not found!")
        return
    
    factor_names = pd.read_csv(factor_names_file, index_col=0, header=None)
    factor_names.columns = ['description']
    # Filter out nan index entries from CSV parsing
    factor_names = factor_names[factor_names.index.notna()]
    
    API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
    if not API_KEY:
        print(" ALPHAVANTAGE_API_KEY environment variable required")
        return
    
    # Extract symbols and convert to list
    symbols_str = config['symbols']
    symbols_list = [s.strip().upper() for s in symbols_str.split(',')]
    
    # Initialize system with the same parameters
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
        # Get data using the same symbols from config
        symbols = frs._resolve_symbols(symbols_list)  # Expands ETFs to constituents if needed
        prices = frs.get_prices(symbols)
        returns = prices.pct_change().dropna()
        
        # Get fundamentals for context
        print(" Loading fundamental data...")
        fundamental_fields = [
            "Sector", "MarketCapitalization", "PERatio", "DividendYield",
            "PriceToSalesRatioTTM", "PriceToBookRatio", "ForwardPE", "ProfitMargin", 
            "ReturnOnEquityTTM", "QuarterlyEarningsGrowthYOY", "Beta", 
            "OperatingMarginTTM", "PercentInstitutions"
        ]
        fundamentals = frs.get_fundamentals(returns.columns.tolist(), fields=fundamental_fields)
        
        # Re-run factor discovery with same parameters
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
        
        print(f" Loaded fundamentals for {len(fundamentals)} stocks")
        print(f" Factor loadings shape: {loadings.shape}")
        
        # Analyze each factor with fundamental context
        for i, (factor_id, row) in enumerate(factor_names.iterrows()):
            factor_name = row['description']
            factor_col = loadings.columns[i]
            
            print(f"\n FACTOR {i+1}: {factor_id}")
            print(f" {factor_name}")
            print("=" * 70)
            
            factor_loadings = loadings[factor_col]
            
            # Analyze positive exposures
            print(f"\n TOP {n_stocks} LONG POSITIONS:")
            print(f"{'Rank':<4} {'Stock':<6} {'Loading':<8} {'Sector':<20} {'Mkt Cap':<10} {'P/E':<6}")
            print("-" * 65)
            
            top_positive = factor_loadings.nlargest(n_stocks)
            for j, (stock, loading) in enumerate(top_positive.items(), 1):
                if stock in fundamentals.index:
                    fund = fundamentals.loc[stock]
                    sector = str(fund.get('Sector', 'N/A'))[:18]
                    market_cap = fund.get('MarketCapitalization', 0)
                    if pd.notna(market_cap) and market_cap > 0:
                        market_cap_str = f"${float(market_cap)/1e9:.1f}B"
                    else:
                        market_cap_str = "N/A"
                    pe_ratio = fund.get('PERatio', 'N/A')
                    if pd.notna(pe_ratio):
                        pe_str = f"{float(pe_ratio):.1f}"
                    else:
                        pe_str = "N/A"
                else:
                    sector, market_cap_str, pe_str = "N/A", "N/A", "N/A"
                
                print(f"{j:<4} {stock:<6} {loading:+.4f}  {sector:<20} {market_cap_str:<10} {pe_str:<6}")
            
            # Analyze negative exposures
            print(f"\n TOP {n_stocks} SHORT POSITIONS:")
            print(f"{'Rank':<4} {'Stock':<6} {'Loading':<8} {'Sector':<20} {'Mkt Cap':<10} {'P/E':<6}")
            print("-" * 65)
            
            top_negative = factor_loadings.nsmallest(n_stocks)
            for j, (stock, loading) in enumerate(top_negative.items(), 1):
                if stock in fundamentals.index:
                    fund = fundamentals.loc[stock]
                    sector = str(fund.get('Sector', 'N/A'))[:18]
                    market_cap = fund.get('MarketCapitalization', 0)
                    if pd.notna(market_cap) and market_cap > 0:
                        market_cap_str = f"${float(market_cap)/1e9:.1f}B"
                    else:
                        market_cap_str = "N/A"
                    pe_ratio = fund.get('PERatio', 'N/A')
                    if pd.notna(pe_ratio):
                        pe_str = f"{float(pe_ratio):.1f}"
                    else:
                        pe_str = "N/A"
                else:
                    sector, market_cap_str, pe_str = "N/A", "N/A", "N/A"
                
                print(f"{j:<4} {stock:<6} {loading:+.4f}  {sector:<20} {market_cap_str:<10} {pe_str:<6}")
            
            # Factor summary
            print(f"\n Factor Summary:")
            print(f"   Loading Range: {factor_loadings.min():.4f} to {factor_loadings.max():.4f}")
            print(f"   Standard Deviation: {factor_loadings.std():.4f}")
            significant_stocks = (factor_loadings.abs() > 0.01).sum()
            print(f"   Significant Stocks (|loading| > 0.01): {significant_stocks}")
            
        # Export detailed results
        print(f"\n Exporting detailed results...")
        
        # Combine loadings with fundamentals
        detailed_results = loadings.copy()
        if not fundamentals.empty:
            # Add fundamental data
            fund_cols = ['Sector', 'MarketCapitalization', 'PERatio', 'DividendYield',
                        'PriceToSalesRatioTTM', 'PriceToBookRatio', 'ForwardPE', 'ProfitMargin', 
                        'ReturnOnEquityTTM', 'QuarterlyEarningsGrowthYOY', 'Beta', 
                        'OperatingMarginTTM', 'PercentInstitutions']
            for col in fund_cols:
                if col in fundamentals.columns:
                    detailed_results[f'Fund_{col}'] = fundamentals[col]
        
        detailed_results.to_csv('detailed_factor_analysis.csv')
        print(f" Detailed results saved to 'detailed_factor_analysis.csv'")
        
        # Save factor summary
        factor_summary = []
        for i, factor_col in enumerate(loadings.columns):
            top_long = loadings[factor_col].nlargest(3).index.tolist()
            top_short = loadings[factor_col].nsmallest(3).index.tolist()
            
            factor_summary.append({
                'Factor': f'F{i+1}',
                'Description': factor_names.iloc[i, 0],
                'Top_Long': ', '.join(top_long),
                'Top_Short': ', '.join(top_short),
                'Loading_Range': f"{loadings[factor_col].min():.3f} to {loadings[factor_col].max():.3f}",
                'Std_Dev': f"{loadings[factor_col].std():.3f}"
            })
        
        summary_df = pd.DataFrame(factor_summary)
        summary_df.to_csv('factor_summary.csv', index=False)
        print(f" Factor summary saved to 'factor_summary.csv'")
        
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    n_stocks = 5
    if len(sys.argv) > 1:
        try:
            n_stocks = int(sys.argv[1])
        except:
            print("Usage: python analyze_factors_detailed.py [n_stocks]")
            sys.exit(1)
    
    detailed_factor_analysis(n_stocks=n_stocks)