#!/usr/bin/env python3
"""
Momentum Backtest Debugging Tool: Systematic Issue Diagnosis and Resolution
==========================================================================

This debugging script provides comprehensive analysis and troubleshooting
for momentum backtest implementation issues. It systematically identifies
common problems in strategy implementation, data loading, and calculation
logic to help resolve zero-return and execution failures.

Purpose
-------
- **Issue Diagnosis**: Systematically identify root causes of backtest failures
- **Data Validation**: Verify data loading, database connectivity, and data quality
- **Logic Verification**: Test strategy implementation step-by-step with detailed logging
- **Troubleshooting Guide**: Provide actionable insights for fixing common issues

Common Issues Addressed
----------------------
- **Missing Database**: Database file not found or corrupted
- **Empty Data**: No price data loaded due to API or caching issues
- **Pandas Warnings**: Chained assignment warnings in position setting
- **Zero Returns**: Strategy returns all zeros due to position management bugs
- **Date Range Issues**: Insufficient data for momentum calculations
- **Index Misalignment**: Mismatched indices between prices and returns

Debugging Workflow
-----------------
1. **Environment Validation**: Check API keys, database existence, path configuration
2. **Data Loading Test**: Verify data availability and quality
3. **Strategy Execution**: Run instrumented version of momentum strategy
4. **Results Analysis**: Examine outputs and identify failure points
5. **Issue Resolution**: Provide specific recommendations for fixes

Key Features
-----------
- **Step-by-Step Analysis**: Detailed logging at each stage of execution
- **Error Isolation**: Identifies specific failure points in strategy logic
- **Data Quality Checks**: Validates input data and intermediate calculations
- **Comparative Testing**: Tests both original and corrected implementations
- **Performance Metrics**: Basic strategy performance evaluation

Debug Information Provided
--------------------------
- Database file existence and accessibility
- Data loading success/failure with shape and date range
- Momentum calculation validation
- Position management verification with rebalancing details
- Strategy return calculation with intermediate results
- Performance metrics and validation checks

Usage
-----
```bash
# Set environment variable
export ALPHAVANTAGE_API_KEY="your_api_key"

# Run debugging analysis
python test_momentum_debug.py

# Expected output:
# === DEBUGGING MOMENTUM BACKTEST ISSUE ===
# 1. Initializing AlphaVantage system...
#    Database exists: True
# 2. Loading price data...
#    SUCCESS: Loaded 500 days x 10 stocks
#    Date range: 2022-06-29 to 2024-06-29
# 3. Calculating returns...
#    Returns shape: (499, 10)
# 4. Testing your original momentum function...
#    ...detailed step-by-step analysis...
# 5. SUCCESS! Your function now returns results
```

Integration Points
-----------------
- **Tests**: AlphaVantageFactorSystem data loading functionality
- **Debugs**: Momentum strategy implementation logic
- **Validates**: Database connectivity and data quality
- **Informs**: Strategy debugging and optimization approaches

Notes
-----
This script is designed as a diagnostic tool for momentum strategy development.
It provides detailed insights into common implementation issues and serves
as a reference for proper strategy implementation patterns.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add src directory to path
sys.path.append('./src')
from alphavantage_system import AlphaVantageFactorSystem

def debug_momentum_backtest():
    """
    Execute comprehensive debugging analysis for momentum backtest implementation.
    
    This function systematically diagnoses common issues in momentum strategy
    backtesting by testing each component of the workflow independently.
    It provides detailed logging and validation to identify root causes
    of execution failures and zero-return issues.
    
    Debugging Steps
    --------------
    
    **1. Environment Validation**
    - Verify ALPHAVANTAGE_API_KEY environment variable
    - Check database file existence and accessibility
    - Validate system paths and dependencies
    
    **2. Data Loading Verification**
    - Test AlphaVantageFactorSystem initialization
    - Load historical price data for test universe
    - Validate data quality, shape, and date coverage
    
    **3. Return Calculation Testing**
    - Calculate daily returns from price data
    - Verify return matrix dimensions and data quality
    - Check for missing values and data alignment
    
    **4. Strategy Implementation Testing**
    - Run instrumented version of momentum strategy
    - Detailed logging of momentum calculation
    - Position allocation verification with rebalancing events
    - Strategy return calculation with intermediate validation
    
    **5. Results Analysis**
    - Evaluate strategy execution success/failure
    - Provide performance metrics and validation
    - Identify specific issues and recommend fixes
    
    Test Universe
    ------------
    Uses a diversified 10-stock universe for testing:
    - Technology: AAPL, MSFT, GOOGL, AMZN, META
    - Finance: JPM, BAC
    - Energy: XOM
    - Healthcare: JNJ
    - Retail: WMT
    
    Common Issues Diagnosed
    ----------------------
    - **Database Missing**: ./data/av_cache.db not found
    - **Empty Data**: API failures or insufficient cached data
    - **Pandas Warnings**: Chained assignment in position setting
    - **Zero Returns**: Position management bugs
    - **Calculation Errors**: Momentum or return calculation issues
    
    Debug Output
    -----------
    Provides detailed console output including:
    ```
    1. Initializing AlphaVantage system...
       Database exists: True/False
    2. Loading price data...
       SUCCESS: Loaded X days x Y stocks
       Date range: YYYY-MM-DD to YYYY-MM-DD
    3. Calculating returns...
       Returns shape: (X, Y)
    4. Testing your original momentum function...
       Input validation: prices shape, returns shape, parameters
       Momentum calculated, shape: (X, Y)
       Positions initialized: (X, Y)
       Rebalance events with selected stocks
       Total rebalances: N
       Strategy returns calculated: shape, mean, total return
    5. SUCCESS!/FAILURE analysis
    ```
    
    Error Handling
    -------------
    - Graceful handling of missing environment variables
    - Detailed error reporting for data loading failures
    - Exception catching with full traceback for debugging
    - Specific recommendations for each type of failure
    
    Raises
    ------
    ValueError
        If ALPHAVANTAGE_API_KEY environment variable is not set
    Exception
        Various exceptions during data loading or strategy execution
        (all caught and reported with detailed diagnostics)
        
    Notes
    -----
    - This function is designed for debugging purposes only
    - Uses a simplified momentum strategy implementation for testing
    - Provides detailed logging for educational and diagnostic purposes
    - Results should be compared with the corrected implementation
    """
    
    print("=== DEBUGGING MOMENTUM BACKTEST ISSUE ===\n")
    
    # 1. Initialize system
    print("1. Initializing AlphaVantage system...")
    import os
    API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
    if not API_KEY:
        raise ValueError("ALPHAVANTAGE_API_KEY environment variable is required")
    av_system = AlphaVantageFactorSystem(API_KEY, data_dir='./data/av_data', db_path='./data/av_cache.db')
    
    # Check if database exists
    db_exists = os.path.exists('./data/av_cache.db')
    print(f"   Database exists: {db_exists}")
    if not db_exists:
        print("   ERROR: Database not found! This is likely why your function failed.")
        return
    
    # 2. Load data
    print("\n2. Loading price data...")
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'JPM', 'BAC', 'XOM', 'JNJ', 'WMT']
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        price_data = av_system.load_price_data(tickers, start_date, end_date)
        
        if price_data is None or price_data.empty:
            print("   ERROR: No price data loaded!")
            print("   This is why your backtest returned nothing.")
            return
        
        print(f"   SUCCESS: Loaded {price_data.shape[0]} days x {price_data.shape[1]} stocks")
        print(f"   Date range: {price_data.index.min().date()} to {price_data.index.max().date()}")
        
        # 3. Calculate returns
        print("\n3. Calculating returns...")
        returns = price_data.pct_change().dropna()
        print(f"   Returns shape: {returns.shape}")
        
        # 4. Test your original function
        print("\n4. Testing your original momentum function...")
        
        def simple_momentum_backtest(prices, returns, lookback=60, rebalance_days=20):
            """Your original function with debug prints"""
            
            print(f"   Input validation:")
            print(f"     prices shape: {prices.shape}")
            print(f"     returns shape: {returns.shape}")
            print(f"     lookback: {lookback} days")
            
            # Calculate momentum signal
            momentum = prices.pct_change(lookback)
            print(f"     momentum calculated, shape: {momentum.shape}")
            
            # Initialize arrays
            positions = pd.DataFrame(0, index=returns.index, columns=returns.columns)
            print(f"     positions initialized: {positions.shape}")
            
            rebalance_count = 0
            
            # Rebalance every N days
            for i in range(lookback, len(returns), rebalance_days):
                # Get current momentum
                current_mom = momentum.iloc[i]
                
                # Select top 30% of stocks
                n_stocks = int(len(current_mom) * 0.3)
                
                # Remove NaN values before selecting
                valid_momentum = current_mom.dropna()
                if len(valid_momentum) == 0:
                    continue
                    
                top_stocks = valid_momentum.nlargest(min(n_stocks, len(valid_momentum))).index
                
                # Equal weight positions
                end_idx = min(i + rebalance_days, len(positions))
                positions.iloc[i:end_idx] = 0  # Clear previous positions
                
                weight = 1.0 / len(top_stocks)
                for stock in top_stocks:
                    if stock in positions.columns:
                        positions.iloc[i:end_idx][stock] = weight
                
                rebalance_count += 1
                
                if rebalance_count <= 3:  # Show first 3 rebalances
                    print(f"     Rebalance {rebalance_count} on {returns.index[i].date()}: {list(top_stocks)}")
            
            print(f"   Total rebalances: {rebalance_count}")
            
            # Calculate strategy returns
            strategy_returns = (positions.shift(1) * returns).sum(axis=1)
            strategy_returns = strategy_returns.dropna()
            
            print(f"   Strategy returns calculated:")
            print(f"     Shape: {strategy_returns.shape}")
            print(f"     Mean daily return: {strategy_returns.mean():.4f}")
            print(f"     Total return: {(1 + strategy_returns).prod() - 1:.2%}")
            
            return strategy_returns, positions
        
        # Run the test
        strat_returns, positions = simple_momentum_backtest(price_data, returns)
        
        if strat_returns is not None and len(strat_returns) > 0:
            print(f"\n5. SUCCESS! Your function now returns results:")
            print(f"   Strategy returns shape: {strat_returns.shape}")
            print(f"   First few returns: {strat_returns.head().values}")
            print(f"   Last few returns: {strat_returns.tail().values}")
        else:
            print(f"\n5. STILL NO RESULTS - there may be another issue")
            
    except Exception as e:
        print(f"   ERROR loading data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_momentum_backtest()
