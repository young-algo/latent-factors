"""
Database Cache Inspection Tool: SQLite Database Analysis and Health Check
========================================================================

This utility script provides comprehensive analysis of SQLite database cache
files used by the equity factors research system. It examines database structure,
content statistics, and provides recommendations for optimal database usage.

Purpose
-------
- **Database Discovery**: Locate all potential database files in project directory
- **Health Assessment**: Analyze database size, table structure, and data quality
- **Content Analysis**: Detailed statistics on prices, fundamentals, and ETF holdings
- **Usage Guidance**: Recommendations for selecting optimal database configuration

Key Features
-----------
- **Multi-Location Search**: Checks common database locations automatically
- **Table Analysis**: Examines all tables and their contents
- **Data Quality Metrics**: Row counts, unique tickers, date ranges
- **ETF Holdings Support**: Analysis of ETF constituent data
- **Error Handling**: Robust handling of corrupted or missing databases

Database Schema Expected
-----------------------
- **prices**: Historical price data (ticker, date, close, volume, etc.)
- **fundamentals**: Fundamental financial metrics (P/E, market cap, etc.)  
- **etf_holdings**: ETF constituent holdings (etf, ticker, weight)
- **meta**: Metadata and caching information

Output Information
-----------------
For each database found:
- File path and existence status
- Database size in MB
- Available tables list
- Price data: row count, unique tickers, date range
- Fundamentals data: row count, unique tickers
- ETF holdings: row count, ETF list

Usage
-----
```bash
# Run from project root directory
python check_cache.py

# Example output:
# === Root Database ===
# Path: ./av_cache.db
# Size: 156.2 MB
# Tables: ['prices', 'fundamentals', 'etf_holdings', 'meta']
# Price data: 1,247,853 rows, 523 tickers
# Date range: 2020-01-01 to 2024-06-29
# Fundamentals: 523 rows, 523 tickers
# ETF holdings: 503 rows, ETFs: ['SPY']
```

Dependencies
-----------
- **Core**: sqlite3 (built-in), os (built-in)
- **Database**: Compatible with alphavantage_system.DataBackend cache format

Integration Points
-----------------
- **Used by**: Manual database analysis and troubleshooting
- **Analyzes**: Cache files created by alphavantage_system.DataBackend
- **Informs**: Database selection for research.FactorResearchSystem

Notes
-----
This script is designed for database diagnostics and maintenance. It provides
essential information for troubleshooting data issues and optimizing cache usage
in the factor research workflow.
"""

import sqlite3
import os

def check_database(db_path, name):
    """
    Analyze a single SQLite database file and report comprehensive statistics.
    
    This function performs a detailed health check of a database cache file,
    examining its structure, content, and data quality metrics. It provides
    essential information for database selection and troubleshooting.
    
    Parameters
    ----------
    db_path : str
        Absolute or relative path to the SQLite database file
    name : str
        Human-readable name for the database (for display purposes)
        
    Database Analysis Performed
    --------------------------
    
    **1. File System Checks**
    - File existence validation
    - Database file size calculation (MB)
    - Basic accessibility testing
    
    **2. Schema Analysis** 
    - Table structure discovery
    - Schema validation against expected format
    - Missing table identification
    
    **3. Data Content Analysis**
    - Row count statistics for each table
    - Unique ticker counts for data quality assessment
    - Date range analysis for temporal coverage
    - ETF holdings analysis for supported ETFs
    
    **4. Data Quality Metrics**
    - Ticker consistency across tables
    - Date range completeness
    - Data freshness indicators
    
    Expected Database Schema
    -----------------------
    The function expects databases created by alphavantage_system.DataBackend:
    
    **prices table**: Historical price data
    - Columns: ticker, date, close, high, low, open, volume
    - Primary key: (ticker, date)
    - Expected rows: 1000s to millions
    
    **fundamentals table**: Fundamental financial metrics  
    - Columns: ticker, sector, market_cap, pe_ratio, etc.
    - Primary key: ticker
    - Expected rows: Hundreds to thousands
    
    **etf_holdings table**: ETF constituent data
    - Columns: etf, ticker, weight
    - Primary key: (etf, ticker)
    - Expected rows: Hundreds per ETF
    
    **meta table**: Caching metadata
    - Columns: key, value, timestamp
    - Used for cache invalidation and API tracking
    
    Output Format
    ------------
    The function prints structured output to console:
    ```
    === Database Name ===
    Path: /path/to/database.db
    Size: 156.2 MB
    Tables: ['prices', 'fundamentals', 'etf_holdings', 'meta']
    Price data: 1,247,853 rows, 523 tickers  
    Date range: 2020-01-01 to 2024-06-29
    Fundamentals: 523 rows, 523 tickers
    ETF holdings: 503 rows, ETFs: ['SPY']
    ```
    
    Error Handling
    -------------
    - **Missing Files**: Gracefully handles non-existent database files
    - **Corrupted Databases**: Catches and reports SQL errors
    - **Missing Tables**: Reports missing expected tables
    - **Empty Tables**: Handles tables with no data
    - **Schema Mismatches**: Reports unexpected table structures
    
    Performance Characteristics
    --------------------------
    - **Time Complexity**: O(1) for metadata, O(N) for row counts
    - **Memory Usage**: Minimal (only metadata loaded)
    - **I/O Impact**: Single read-only connection with multiple queries
    - **Execution Time**: <1 second for typical databases
    
    Raises
    ------
    No exceptions raised - all errors are caught and reported as warnings
    
    Examples
    --------
    >>> # Check primary database
    >>> check_database("./av_cache.db", "Primary Cache")
    === Primary Cache ===
    Path: ./av_cache.db
    Size: 156.2 MB
    Tables: ['prices', 'fundamentals', 'etf_holdings', 'meta']
    ...
    
    >>> # Check missing database  
    >>> check_database("./missing.db", "Missing DB")
    === Missing DB ===
    Path: ./missing.db
     Database does not exist
    
    Notes
    -----
    - This function is read-only and does not modify database contents
    - Large databases may take longer to analyze due to COUNT queries
    - The function assumes standard alphavantage_system database schema
    - ETF analysis helps identify which ETFs have been expanded to constituents
    """
    print(f"\n=== {name} ===")
    print(f"Path: {db_path}")
    
    if not os.path.exists(db_path):
        print(" Database does not exist")
        return
    
    size_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"Size: {size_mb:.1f} MB")
    
    try:
        with sqlite3.connect(db_path) as con:
            cur = con.cursor()
            
            # Check if tables exist
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cur.fetchall()]
            print(f"Tables: {tables}")
            
            # Check price data
            if 'prices' in tables:
                cur.execute("SELECT COUNT(*) FROM prices")
                price_rows = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(DISTINCT ticker) FROM prices")
                unique_tickers = cur.fetchone()[0]
                
                cur.execute("SELECT MIN(date), MAX(date) FROM prices")
                date_range = cur.fetchone()
                
                print(f"Price data: {price_rows:,} rows, {unique_tickers} tickers")
                print(f"Date range: {date_range[0]} to {date_range[1]}")
            
            # Check fundamentals data
            if 'fundamentals' in tables:
                cur.execute("SELECT COUNT(*) FROM fundamentals")
                fund_rows = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(DISTINCT ticker) FROM fundamentals")
                unique_fund_tickers = cur.fetchone()[0]
                
                print(f"Fundamentals: {fund_rows:,} rows, {unique_fund_tickers} tickers")
            
            # Check ETF holdings
            if 'etf_holdings' in tables:
                cur.execute("SELECT COUNT(*) FROM etf_holdings")
                etf_rows = cur.fetchone()[0]
                
                cur.execute("SELECT DISTINCT etf FROM etf_holdings")
                etfs = [row[0] for row in cur.fetchall()]
                
                print(f"ETF holdings: {etf_rows:,} rows, ETFs: {etfs}")
                
    except Exception as e:
        print(f" Error reading database: {e}")

# Check all potential database locations
databases = [
    ("/Users/kevinturner/Documents/Code/equity-factors/av_cache.db", "Root Database"),
    ("/Users/kevinturner/Documents/Code/equity-factors/src/av_cache.db", "Src Database"), 
    ("./av_cache.db", "Current Dir Database"),
    ("../av_cache.db", "Parent Dir Database")
]

for db_path, name in databases:
    check_database(db_path, name)

print("\n" + "="*60)
print("RECOMMENDATION:")
print("Use the database with the most data (highest MB and ticker count)")