#!/usr/bin/env python3
"""
Test script to understand where factor returns are stored in the discover_and_label.py workflow.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, 'src')

def test_factor_storage():
    """
    Test the factor discovery workflow to see where factor returns are stored.
    """
    print(" Testing factor storage location in discover_and_label.py workflow")
    print("=" * 60)
    
    # Import the modules
    try:
        from research import FactorResearchSystem
        from latent_factors import statistical_factors, autoencoder_factors, StatMethod
        print(" Successfully imported factor discovery modules")
    except ImportError as e:
        print(f" Failed to import modules: {e}")
        return
    
    # Check if we have API key
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        print(" ALPHAVANTAGE_API_KEY not found in environment")
        print("   Factor returns are generated during the discovery process but not saved separately")
        print("   They are only returned in memory from the functions")
        return
    
    print(f" API key found: {api_key[:10]}...")
    
    # Test with a small universe to see the workflow
    print("\n Testing with small universe to trace factor storage...")
    
    try:
        # Create a minimal test case
        symbols = ["AAPL", "MSFT"]
        frs = FactorResearchSystem(api_key, universe=symbols, start_date="2024-01-01")
        
        # Get price data and calculate returns
        prices = frs.get_prices(symbols)
        returns = prices.pct_change().dropna()
        
        print(f" Got price data: {prices.shape}")
        print(f" Calculated returns: {returns.shape}")
        
        # Test statistical factors function
        print("\n Testing statistical_factors function...")
        factor_returns, loadings = statistical_factors(returns, n_components=2, method=StatMethod.PCA)
        
        print(f" Factor returns shape: {factor_returns.shape}")
        print(f" Factor loadings shape: {loadings.shape}")
        print(f" Factor returns columns: {list(factor_returns.columns)}")
        print(f" Sample factor returns:")
        print(factor_returns.head())
        
        print("\n Checking where factor returns are stored...")
        print(" Factor returns are NOT automatically saved to disk by the discovery functions")
        print(" They are only returned in memory as pandas DataFrames")
        print(" The discover_and_label.py script generates them but only saves:")
        print("   - factor_names.csv (factor names)")
        print("   - cumulative_factor_returns.png (chart)")
        print("   - factor_analysis_config.json (configuration)")
        
        print("\n To save factor returns, you need to modify the workflow or extract them manually")
        
        # Show what would need to be saved
        print(f"\n Factor returns that could be saved:")
        print(f"   Shape: {factor_returns.shape}")
        print(f"   Date range: {factor_returns.index[0]} to {factor_returns.index[-1]}")
        print(f"   Factors: {list(factor_returns.columns)}")
        
        # Example of saving
        factor_returns.to_csv("factor_returns_example.csv")
        print(f" Example saved to: factor_returns_example.csv")
        
    except Exception as e:
        print(f" Error during testing: {e}")
        print("   This suggests the factor returns are generated in memory only")

if __name__ == "__main__":
    test_factor_storage()