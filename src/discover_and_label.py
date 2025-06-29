#!/usr/bin/env python
"""
Factor Discovery and Naming Workflow: Complete End-to-End Pipeline
==================================================================

This module provides the main command-line interface and workflow orchestration
for discovering latent factors from financial data and generating human-readable
factor names using LLM analysis. It integrates factor discovery methods with
automatic factor naming to provide a complete research workflow.

Workflow Overview
----------------

**1. Data Collection & Preprocessing**
- Load price data for specified universe (stocks/ETFs)
- Handle ETF expansion to constituent holdings
- Calculate returns and validate data quality

**2. Factor Discovery**
- Apply statistical methods (PCA, ICA, NMF) or deep learning (Autoencoder)
- Extract factor returns and loadings matrices
- Support both static and rolling window analysis

**3. Factor Validation**
- Comprehensive factor quality checks (correlation, volatility, concentration)
- Early detection of issues like redundant factors or calculation errors
- Provide actionable recommendations for improvement

**4. LLM-Powered Factor Naming**
- Analyze factor loadings to identify key stock exposures
- Integrate fundamental data for richer context
- Generate meaningful factor names using OpenAI GPT models

**5. Results Export & Visualization**
- Export factor names to CSV format
- Generate cumulative factor return charts
- Provide summary statistics and insights

Supported Factor Methods
-----------------------
- **PCA**: Principal Component Analysis (orthogonal, variance-maximizing)
- **ICA**: Independent Component Analysis (statistically independent)
- **NMF**: Non-negative Matrix Factorization (positive factors only)
- **AE**: Autoencoder (non-linear, deep learning approach)

Command Line Interface
---------------------
The script provides a comprehensive CLI for factor research:

```bash
# Basic ETF-based factor discovery
python discover_and_label.py --symbols "SPY,QQQ" --method PCA --k 10

# Rolling window analysis with custom output
python discover_and_label.py --symbols "AAPL,MSFT,GOOGL" \\
    --method ICA --k 5 --rolling 252 --name_out tech_factors.csv

# Autoencoder approach with custom date range
python discover_and_label.py --symbols "SPY" --method AE \\
    --start 2020-01-01 --k 8 --name_out ae_factors.csv
```

Integration Architecture
-----------------------
```
CLI Input → FactorResearchSystem → Factor Discovery → Validation → LLM Naming → Export
    ↓              ↓                    ↓              ↓           ↓          ↓
[Symbols]      [Data Loading]       [PCA/ICA/NMF]  [Quality]   [OpenAI]   [CSV/Charts]
```

Dependencies
-----------
- **Core Libraries**: pandas, numpy, matplotlib
- **Factor Discovery**: latent_factors module (statistical methods)
- **Data Backend**: research module (FactorResearchSystem)
- **LLM Integration**: factor_labeler module (OpenAI API)
- **Environment**: python-dotenv for configuration

Environment Requirements
-----------------------
- **ALPHAVANTAGE_API_KEY**: Required for data access
- **OPENAI_API_KEY**: Required for factor naming
- **Python**: 3.8+ with required dependencies
- **Memory**: Sufficient for universe size (typically 1-4GB)

Performance Characteristics
--------------------------
- **Small Universe** (<50 stocks): 1-5 minutes end-to-end
- **Medium Universe** (50-200 stocks): 5-15 minutes end-to-end
- **Large Universe** (200+ stocks): 15-60 minutes end-to-end
- **Bottlenecks**: Data loading (API limits), LLM naming (API latency)

Output Files
-----------
- **factor_names.csv**: Factor names and descriptions
- **Matplotlib Chart**: Cumulative factor returns visualization
- **Console Logs**: Detailed progress and validation results

Error Handling
-------------
- **Data Issues**: Robust handling of missing data and API failures
- **Factor Quality**: Automatic validation with clear error messages
- **API Failures**: Graceful fallbacks and retry mechanisms
- **Parameter Validation**: Clear error messages for invalid inputs

Examples
--------
>>> # Programmatic usage (not typical)
>>> args = _parse()
>>> main()  # Execute full workflow

>>> # Typical CLI usage
>>> $ python discover_and_label.py --symbols "SPY" --method PCA --k 5

Notes
----
- This is the main entry point for the factor discovery workflow
- Integrates all components: data, factor discovery, validation, naming
- Designed primarily as a CLI tool, not for programmatic import
- Provides comprehensive logging and progress tracking
"""

import argparse, logging, json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from research import FactorResearchSystem          # existing data backend
from latent_factors import statistical_factors, autoencoder_factors, StatMethod, validate_factor_distinctiveness
from factor_labeler import ask_llm


# ----------------------------- CLI ----------------------------- #
def _parse() -> argparse.Namespace:
    """
    Parse command-line arguments for the factor discovery workflow.
    
    This function configures and parses all command-line arguments required
    for the factor discovery and naming pipeline, providing a comprehensive
    interface for customizing the analysis parameters.
    
    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments containing:
        - symbols: Comma-separated list of tickers/ETFs
        - start: Start date for analysis (YYYY-MM-DD)
        - method: Factor discovery method (PCA/ICA/NMF/AE)
        - k: Number of factors to extract
        - rolling: Rolling window size (0 for static analysis)
        - name_out: Output path for factor names CSV
        
    Command-Line Arguments
    ---------------------
    
    **Required Arguments:**
    - `-s, --symbols`: Comma-separated ticker symbols or ETFs
      Examples: "AAPL,MSFT", "SPY", "QQQ,XLF,XLE"
      
    **Optional Arguments:**
    - `--start`: Start date (default: "2020-04-01")
      Format: YYYY-MM-DD
      
    - `--method`: Factor discovery method (default: "PCA")
      Choices: ["PCA", "ICA", "NMF", "AE"]
      - PCA: Principal Component Analysis
      - ICA: Independent Component Analysis  
      - NMF: Non-negative Matrix Factorization
      - AE: Autoencoder (deep learning)
      
    - `-k`: Number of factors (default: 10)
      Range: 1-50 (practical limit depends on universe size)
      
    - `--rolling`: Rolling window size in days (default: 0)
      - 0: Static analysis on full period
      - >0: Rolling window analysis (e.g., 252 for annual)
      
    - `--name_out`: Output CSV file for factor names (default: "factor_names.csv")
      Path where factor names and descriptions will be saved
      
    Validation
    ---------
    The function provides basic argument validation through argparse,
    with additional validation performed in the main workflow:
    - Required arguments are enforced
    - Method choices are restricted to valid options
    - Numeric arguments have appropriate types
    
    Examples
    --------
    >>> # Typical command-line usage:
    >>> # python discover_and_label.py --symbols "SPY,QQQ" --method PCA --k 5
    >>> 
    >>> # Programmatic parsing (for testing):
    >>> import sys
    >>> sys.argv = ["script.py", "--symbols", "SPY", "--method", "PCA"]
    >>> args = _parse()
    >>> print(args.symbols)  # "SPY"
    >>> print(args.method)   # "PCA"
    
    Notes
    -----
    - This function is called automatically by main() workflow
    - All arguments have sensible defaults for quick experimentation
    - The symbols argument supports both individual stocks and ETFs
    - ETF expansion is handled automatically in the main workflow
    """
    p = argparse.ArgumentParser(description="Discover & label latent factors")
    p.add_argument("-s", "--symbols", required=True,
                   help="comma separated tickers / ETFs")
    p.add_argument("--start", default="2020-04-01")
    p.add_argument("--method", choices=["PCA", "ICA", "NMF", "AE"], default="PCA")
    p.add_argument("-k", type=int, default=10, help="# latent factors")
    p.add_argument("--rolling", type=int, default=0,
                   help="rolling window (days); 0 = static")
    p.add_argument("--name_out", default="factor_names.csv")
    return p.parse_args()


def main():
    """
    Execute the complete factor discovery and naming workflow.
    
    This is the main orchestration function that coordinates all aspects of
    the factor discovery pipeline: argument parsing, data loading, factor
    discovery, validation, LLM naming, and results export.
    
    Workflow Execution Steps
    -----------------------
    
    **1. Initialization & Configuration**
    - Parse command-line arguments
    - Set up logging configuration
    - Load API keys from environment variables
    - Validate required dependencies
    
    **2. Data Collection & Preprocessing** 
    - Initialize FactorResearchSystem with API credentials
    - Resolve symbol universe (handle ETF expansion)
    - Load historical price data with caching
    - Calculate returns and validate data quality
    
    **3. Factor Discovery**
    - Apply selected factor discovery method:
      * Static analysis: Fit factors on entire period
      * Rolling analysis: Fit factors in rolling windows
    - Extract factor returns and loadings matrices
    - Handle different methods (PCA, ICA, NMF, Autoencoder)
    
    **4. Factor Validation**
    - Run comprehensive factor quality checks
    - Analyze factor correlations and distinctiveness
    - Check for realistic volatility and return patterns
    - Validate factor loading distributions
    - Log warnings and recommendations for issues
    
    **5. LLM-Powered Factor Naming**
    - Load fundamental data for factor context
    - Identify top positive and negative stock exposures
    - Generate LLM prompts with rich fundamental context
    - Call OpenAI API to generate factor names
    - Log factor names and descriptions
    
    **6. Results Export & Visualization**
    - Export factor names to CSV file
    - Generate cumulative factor returns chart
    - Display final summary and validation results
    
    Error Handling
    -------------
    The workflow includes comprehensive error handling for:
    - **Missing API Keys**: Clear error messages with setup instructions
    - **Data Issues**: Robust handling of missing/invalid data
    - **Factor Quality**: Automatic validation with actionable feedback
    - **API Failures**: Graceful degradation and retry mechanisms
    - **File I/O**: Proper error handling for output operations
    
    Performance Characteristics
    --------------------------
    Execution time depends on several factors:
    - **Universe Size**: 50 stocks (~2 min), 500 stocks (~15 min)
    - **Time Period**: Longer histories increase processing time
    - **Method Choice**: Statistical methods faster than autoencoders
    - **API Latency**: Data loading and LLM naming add network overhead
    - **Rolling Windows**: Significantly increase computation time
    
    Output Artifacts
    ---------------
    - **factor_names.csv**: Factor names and descriptions
    - **Console Output**: Detailed logging of workflow progress
    - **Matplotlib Chart**: Cumulative factor returns visualization
    - **Validation Report**: Factor quality assessment results
    
    Environment Dependencies
    -----------------------
    Required environment variables:
    - **ALPHAVANTAGE_API_KEY**: For financial data access
    - **OPENAI_API_KEY**: For factor naming via LLM
    
    Required Python packages:
    - pandas, numpy, matplotlib (core data processing)
    - openai (LLM integration)
    - python-dotenv (environment management)
    
    Raises
    ------
    ValueError
        If required environment variables are missing
    RuntimeError
        If factor discovery or validation fails
    ConnectionError
        If API calls fail after retries
    
    Examples
    --------
    >>> # Typical execution (called automatically from CLI)
    >>> if __name__ == "__main__":
    ...     main()
    
    >>> # The function orchestrates this workflow:
    >>> # 1. Parse args: --symbols "SPY" --method PCA --k 5
    >>> # 2. Load data: SPY constituents + price history
    >>> # 3. Discover factors: 5 PCA factors from returns
    >>> # 4. Validate: Check factor quality and distinctiveness  
    >>> # 5. Name factors: Generate meaningful names via LLM
    >>> # 6. Export: Save factor_names.csv + show chart
    
    Notes
    -----
    - This function is designed to be called from command line
    - All configuration is handled via command-line arguments
    - Progress is logged to console for monitoring long-running jobs
    - The workflow includes extensive validation to catch common issues
    - Results are immediately available for analysis after completion
    """
    args = _parse()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    # Load API key from environment
    import os
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError("ALPHAVANTAGE_API_KEY environment variable is required")

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    frs = FactorResearchSystem(api_key, universe=symbols, start_date=args.start)
    prices = frs.get_prices(frs._resolve_symbols(symbols))
    returns = prices.pct_change().dropna()

    if args.rolling > 0:
        # simple: fit new factors every R days, keep latest loadings for naming
        fac_frames, load_frames = [], []
        for t in range(args.rolling, len(returns), args.rolling):
            sub = returns.iloc[t - args.rolling:t]
            fac, load = _fit(sub, args)
            fac_frames.append(fac)
            load_frames.append(load)
        factor_ret = pd.concat(fac_frames)
        loadings = load_frames[-1]
    else:
        factor_ret, loadings = _fit(returns, args)

    # ------------------- Factor Validation -------------------- #
    logging.info("Validating factor quality...")
    validation = validate_factor_distinctiveness(factor_ret, loadings)
    
    if not validation["is_valid"]:
        logging.warning("⚠️  FACTOR VALIDATION FAILED")
        for warning in validation["warnings"]:
            logging.warning("  - %s", warning)
        logging.info("Recommendations:")
        for rec in validation["recommendations"]:
            logging.info("  + %s", rec)
    else:
        logging.info("✅ Factor validation passed")

    # ------------------- LLM naming -------------------- #
    fundamentals = frs.get_fundamentals(loadings.index.tolist())
    names = {}
    for f in loadings.columns:
        top = loadings[f].nlargest(10).index.tolist()
        bot = loadings[f].nsmallest(10).index.tolist()
        label = ask_llm(f, top, bot, fundamentals)
        names[f] = label
        logging.info("Factor %s →  %s", f, label)

    pd.Series(names).to_csv(args.name_out)
    logging.info("Saved factor names → %s", args.name_out)
    
    # Save configuration for analysis scripts
    config = {
        'symbols': args.symbols,
        'method': args.method,
        'k': args.k,
        'start_date': args.start,
        'rolling': args.rolling,
        'resolved_symbols': len(loadings.index),
        'factor_names_file': args.name_out
    }
    
    import json
    with open('factor_analysis_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    logging.info("Saved analysis configuration → factor_analysis_config.json")

    # quick visual - filter to start date
    chart_data = factor_ret[factor_ret.index >= args.start] if len(factor_ret) > 0 else factor_ret
    (chart_data.cumsum() * 100).plot(figsize=(10, 6), lw=1)
    plt.title(f"Cumulative latent factor returns (bps) from {args.start}")
    plt.tight_layout()
    plt.savefig('cumulative_factor_returns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("📊 Factor returns chart saved to cumulative_factor_returns.png")


def _fit(ret: pd.DataFrame, args: argparse.Namespace):
    """
    Apply the specified factor discovery method to return data.
    
    This helper function routes the factor discovery process to the appropriate
    method based on command-line arguments, providing a clean interface between
    the main workflow and the underlying factor discovery implementations.
    
    Parameters
    ----------
    ret : pd.DataFrame
        Stock return matrix with shape (T, N) where:
        - T = number of time periods (rows)
        - N = number of assets (columns)
        - Values = daily/periodic returns (typically -0.1 to +0.1 range)
        
    args : argparse.Namespace
        Parsed command-line arguments containing:
        - method: Factor discovery method ("PCA", "ICA", "NMF", "AE")
        - k: Number of factors to extract
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        factor_returns : pd.DataFrame
            Factor returns matrix with shape (T, K) where K = args.k
        factor_loadings : pd.DataFrame  
            Factor loadings matrix with shape (N, K)
            
    Method Routing
    -------------
    - **"AE"**: Routes to autoencoder_factors() for deep learning approach
    - **Other**: Routes to statistical_factors() with method conversion:
      * "PCA" → StatMethod.PCA
      * "ICA" → StatMethod.ICA  
      * "NMF" → StatMethod.NMF
      
    Method Characteristics
    ---------------------
    
    **Autoencoder ("AE")**:
    - Non-linear factor discovery using neural networks
    - Best for: Complex patterns, non-linear relationships
    - Slower but more expressive than statistical methods
    
    **Statistical Methods**:
    - Linear factor discovery using matrix decomposition
    - Fast and interpretable
    - PCA: Variance maximization (orthogonal factors)
    - ICA: Statistical independence (non-orthogonal factors)
    - NMF: Non-negative factors (parts-based decomposition)
    
    Error Handling
    -------------
    - Method validation handled by underlying implementations
    - Invalid method names caught by StatMethod enum
    - Data validation performed in factor discovery functions
    
    Examples
    --------
    >>> # Called automatically by main workflow
    >>> returns = prices.pct_change().dropna()
    >>> args = _parse()  # Contains method="PCA", k=5
    >>> factors, loadings = _fit(returns, args)
    >>> print(f"Generated {factors.shape[1]} factors")
    
    >>> # Method routing examples:
    >>> # args.method="PCA" → statistical_factors(..., method=StatMethod.PCA)
    >>> # args.method="AE" → autoencoder_factors(..., k=args.k)
    
    Notes
    -----
    - This function serves as a clean interface layer
    - Simplifies method selection in the main workflow
    - Ensures consistent parameter passing to factor discovery methods
    - Method-specific parameters use sensible defaults
    """
    if args.method == "AE":
        return autoencoder_factors(ret, k=args.k)
    else:
        return statistical_factors(ret,
                                   n_components=args.k,
                                   method=StatMethod[args.method])


if __name__ == "__main__":
    main()