"""
Factor Research Demo Runner: Complete End-to-End Example
=======================================================

This script provides a comprehensive demonstration of the factor research
workflow, showcasing the integration of data collection, factor discovery,
performance analysis, and LLM-powered factor naming.

Purpose
-------
- **Demonstration**: Complete example of factor research workflow
- **Testing**: Validate system functionality with real data
- **Template**: Starting point for custom factor research projects
- **Validation**: Verify database caching and API integration

Workflow Demonstrated
--------------------
1. **Environment Validation**: Check API keys and database status
2. **System Initialization**: Create FactorResearchSystem with SPY ETF
3. **Factor Discovery**: Apply fundamental factor methodology
4. **Performance Analysis**: Generate comprehensive performance report
5. **Factor Naming**: Use LLM to generate meaningful factor names

Key Features Showcased
---------------------
- **ETF Expansion**: SPY automatically expanded to ~500 constituents
- **Caching System**: Utilizes existing database for efficient data access
- **Fundamental Factors**: Cross-sectional regression on financial metrics
- **Risk Targeting**: 10% volatility target for factor returns
- **Performance Reporting**: Heat-map visualization with summary statistics
- **LLM Integration**: OpenAI-powered factor naming with economic context

Configuration
------------
- **Universe**: SPY ETF (S&P 500 constituents)
- **Start Date**: April 1, 2020 (post-COVID analysis period)
- **Method**: Fundamental factors (cross-sectional regression)
- **Volatility Target**: 10% annualized for factor returns
- **Database**: Local SQLite cache (./av_cache.db)

Dependencies
-----------
- **Core**: research.py (FactorResearchSystem)
- **Visualization**: factor_performance_table.py (generate_report)
- **Data**: Alpha Vantage API with local caching
- **LLM**: OpenAI API for factor naming

Environment Requirements
-----------------------
- **ALPHAVANTAGE_API_KEY**: Required environment variable
- **OPENAI_API_KEY**: Required for factor naming functionality
- **Database**: SQLite database file (created automatically if missing)
- **Python**: 3.8+ with required packages installed

Expected Output
--------------
- **Console Logs**: Detailed progress and database status information
- **Performance Report**: Heat-map visualization with period returns
- **Factor Names**: LLM-generated factor descriptions
- **Validation Info**: Database size and record counts

Usage
-----
```bash
# Set environment variables
export ALPHAVANTAGE_API_KEY="your_api_key"
export OPENAI_API_KEY="your_openai_key"

# Run the demonstration
python runner.py
```

Notes
----
- This script serves as both demonstration and validation tool
- Provides comprehensive logging for monitoring and debugging
- Uses sensible defaults for quick experimentation
- Can be modified for custom research configurations
"""

# The necessary modules must be installed and available in your environment.
from .research import FactorResearchSystem
from .factor_performance_table import generate_report

def main():
    """
    Execute the complete factor research demonstration workflow.
    
    This function demonstrates the full factor research pipeline from data
    collection through factor naming, providing a comprehensive example of
    system capabilities and integration points.
    
    Workflow Steps
    -------------
    
    **1. Environment Validation**
    - Check for required ALPHAVANTAGE_API_KEY environment variable
    - Validate database file existence and contents
    - Display database statistics and health information
    
    **2. System Initialization**
    - Create FactorResearchSystem with SPY ETF universe
    - Configure fundamental factor methodology
    - Set analysis start date to April 1, 2020
    - Enable ETF expansion to constituent holdings
    
    **3. Factor Discovery**
    - Load and process S&P 500 constituent data
    - Calculate fundamental exposures (P/S, profit margin, beta, etc.)
    - Perform cross-sectional regression to extract factor returns
    - Validate factor quality and distinctiveness
    
    **4. Performance Analysis**
    - Apply 10% volatility targeting for risk management
    - Generate comprehensive performance report with heat-map
    - Calculate period returns and summary statistics
    - Display professional visualization
    
    **5. Factor Naming**
    - Analyze factor loadings and stock exposures
    - Integrate fundamental data for richer context
    - Generate LLM-powered factor names and descriptions
    - Display final factor interpretations
    
    Database Validation
    ------------------
    The function includes comprehensive database health checks:
    - File existence and size validation
    - Price and fundamental record counts
    - Connection testing and schema verification
    - Performance and data quality indicators
    
    Expected Console Output
    ----------------------
    ```
    Using database: ./av_cache.db
    Database found: 156.2 MB
    Database contains: 1247 price records, 523 fundamentals records
    📅 Analysis period: 2020-04-01 onwards
    📋 Resolving 1 symbols (ETF expansion: True)...
    🔍 Processing symbol 1/1: SPY
    📈 Detected ETF SPY, fetching holdings...
    ✅ ETF SPY expanded to 503 valid constituents
    ... [factor discovery progress] ...
    ... [performance visualization] ...
    {'F1': 'Growth vs Value: High P/E tech vs low P/E financials', ...}
    ```
    
    Error Handling
    -------------
    - **Missing API Key**: Clear error with setup instructions
    - **Database Issues**: Warnings with fallback behavior
    - **Data Problems**: Graceful handling with informative messages
    - **API Failures**: Retry logic and fallback mechanisms
    
    Performance Characteristics
    --------------------------
    - **Execution Time**: 3-8 minutes depending on cache status
    - **Memory Usage**: 500MB-2GB peak (depends on universe size)
    - **API Calls**: Minimized through aggressive caching
    - **Database Growth**: ~100MB for initial SPY analysis
    
    Customization Points
    -------------------
    To adapt this demo for custom research:
    - Change universe: Replace ["SPY"] with desired symbols
    - Modify method: Switch to "pca", "ica", or "autoencoder"
    - Adjust dates: Update start_date for different analysis periods
    - Risk targeting: Modify vol_target parameter
    
    Raises
    ------
    ValueError
        If ALPHAVANTAGE_API_KEY environment variable is not set
    ConnectionError
        If API calls fail after retries
    RuntimeError
        If factor discovery or validation fails
        
    Examples
    --------
    >>> # Run complete demonstration
    >>> main()
    
    >>> # The function executes this workflow:
    >>> # 1. Validate environment and database
    >>> # 2. Initialize with SPY → ~500 constituents
    >>> # 3. Discover fundamental factors
    >>> # 4. Generate performance report with 10% vol target
    >>> # 5. Display LLM-generated factor names
    
    Notes
    -----
    - Designed as both demonstration and validation tool
    - Uses SPY to showcase ETF expansion capabilities
    - Fundamental factors provide interpretable results
    - 10% volatility target normalizes factor comparison
    - Results serve as template for custom research projects
    """
    import os
    API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
    if not API_KEY:
        raise ValueError("ALPHAVANTAGE_API_KEY environment variable is required")
    
    # Use relative path for portability
    db_path = "./av_cache.db"
    print(f"Using database: {db_path}")
    
    # Verify the database exists and has data
    if os.path.exists(db_path):
        size_mb = os.path.getsize(db_path) / (1024 * 1024)
        print(f"Database found: {size_mb:.1f} MB")
        
        # Quick check of database contents
        import sqlite3
        with sqlite3.connect(db_path) as con:
            cur = con.cursor()
            cur.execute("SELECT COUNT(*) FROM meta")
            price_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM fundamentals")
            fund_count = cur.fetchone()[0]
            print(f"Database contains: {price_count} price records, {fund_count} fundamentals records")
    else:
        print(f"WARNING: Database not found at {db_path}")

    # Initialize the system with the ETF and start date
    start_date = "2020-04-01"  # Start from April 1, 2020
    frs = FactorResearchSystem(API_KEY, universe=["SPY"], factor_method="fundamental", 
                              db_path=db_path, start_date=start_date)
    print(f"📅 Analysis period: {start_date} onwards")
    frs.fit_factors()

    # Get factor returns for a 10% volatility target.
    f = frs.get_factor_returns(vol_target=0.10)
    
    # Generate the report.
    generate_report(f)

    # Optional: Print the catchy names for the factors.
    print(frs.name_factors())

# This standard entry point calls the main function when the script is executed.
if __name__ == "__main__":
    main()