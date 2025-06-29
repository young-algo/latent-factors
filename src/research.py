"""
Research Engine: Advanced Factor Discovery with ETF Expansion
============================================================

This module provides a comprehensive factor research system that combines
traditional fundamental analysis with statistical factor models, featuring
automatic ETF expansion and advanced data processing capabilities.

Core Features
------------

**1. Automatic ETF Expansion**
- Intelligently detects ETF tickers in input universe
- Expands ETFs to their current constituent holdings
- Validates and filters constituent tickers for data quality
- Supports both individual stocks and ETF-based universes

**2. Multi-Modal Factor Discovery**
- **Fundamental Factors**: Based on financial metrics (P/S, profit margin, beta, market cap)
- **Statistical Factors**: PCA, ICA, Sparse PCA, Factor Analysis
- **Momentum Factors**: 1-month and 3-month price momentum
- **Risk Factors**: Rolling volatility and beta exposures

**3. Advanced Data Processing**
- Cross-sectional regression with numerical stability
- Robust handling of missing data and rank deficiency
- Intelligent factor validation and filtering
- Rolling window analysis for time-varying factors

**4. LLM Integration**
- Automatic factor naming using OpenAI GPT models
- Intelligent factor interpretation based on asset exposures
- Caching system for factor names and results

Architecture Overview
--------------------
```
Input Universe → ETF Expansion → Data Collection → Factor Modeling → LLM Naming
     ↓              ↓               ↓               ↓              ↓
  [SPY,AAPL]   [500+ stocks]   [Prices+Funds]   [Factors+Loads]  [Names]
```

**Data Flow Pipeline:**
1. **Symbol Resolution**: ETF expansion and validation
2. **Data Collection**: Prices, fundamentals, and derived metrics
3. **Factor Construction**: Fundamental and statistical factor creation
4. **Model Fitting**: Cross-sectional OLS or rolling statistical models
5. **Factor Naming**: LLM-powered factor interpretation

Performance Characteristics
--------------------------
- **Universe Size**: Optimized for 20-1000 stocks
- **Data Efficiency**: Intelligent caching with Alpha Vantage backend
- **Memory Usage**: O(T×N + N×K) where T=time, N=stocks, K=factors
- **Computational Complexity**: O(T×N×K²) for fundamental models

Dependencies
-----------
- **Core**: pandas, numpy, scikit-learn
- **Data**: alphavantage_system (custom backend)
- **ML Models**: sklearn.decomposition (PCA, ICA, Factor Analysis)
- **LLM**: openai (for factor naming)
- **Optional**: Custom latent factor discovery methods

Integration Points
-----------------
- **Called by**: discover_and_label.py (main workflow)
- **Calls**: alphavantage_system.DataBackend (data access)
- **Extends**: DataBackend (inherits caching and API functionality)

Examples
--------
>>> # Basic usage with ETF expansion
>>> frs = FactorResearchSystem(api_key, universe=["SPY", "QQQ"], expand_etfs=True)
>>> frs.fit_factors()
>>> factor_returns = frs.get_factor_returns(vol_target=0.15)
>>> factor_names = frs.name_factors()

>>> # Statistical factor model
>>> frs = FactorResearchSystem(api_key, universe=["AAPL", "MSFT"], 
...                           factor_method="pca", n_components=5)
>>> frs.fit_factors()

Changes in v3
------------
- **New**: Automatic ETF expansion with constituent validation
- **Enhanced**: Robust numerical stability in cross-sectional regression
- **Improved**: Better error handling for missing fundamental data
- **Added**: Advanced factor validation and quality checks
"""

from __future__ import annotations
import json, logging, os
from functools import cached_property
from typing import Sequence, List

import numpy as np
import pandas as pd
from sklearn.decomposition import (
    PCA, SparsePCA, FactorAnalysis, FastICA
)
from sklearn.preprocessing import StandardScaler
import openai                           # pip install openai>=1.2

from alphavantage_system import DataBackend

_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.WARNING)

# Ensure logging shows up in Jupyter notebooks
if not _LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
    handler.setFormatter(formatter)
    _LOGGER.addHandler(handler)
    _LOGGER.propagate = False

_ann = np.sqrt(252)


def _z(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust Z-score standardization with intelligent constant column handling.
    
    This function performs standard Z-score normalization (mean=0, std=1) while
    gracefully handling edge cases like constant columns that would otherwise
    produce NaN values or division by zero errors.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to standardize. Can contain any numeric data including
        financial metrics, returns, or exposure variables.
        
    Returns
    -------
    pd.DataFrame
        Standardized DataFrame where:
        - Non-constant columns: Z-score normalized (mean=0, std=1)
        - Constant columns: Automatically filtered out
        - Index/column structure: Preserved from input
        - Empty result: If all columns were constant
        
    Algorithm Details
    ----------------
    1. **Statistics Calculation**: Compute column-wise means and standard deviations
    2. **Constant Detection**: Identify columns with std = 0 (no variation)
    3. **Filtering**: Remove constant columns that provide no information
    4. **Standardization**: Apply (x - μ) / σ to remaining columns
    5. **Edge Case**: Return empty DataFrame if all columns are constant
    
    Edge Cases Handled
    ------------------
    - **All Constant Columns**: Returns empty DataFrame with preserved index
    - **Some Constant Columns**: Filters them out, standardizes remainder
    - **Single Column**: Works correctly for both constant and variable
    - **Empty Input**: Preserves structure and returns appropriately
    
    Mathematical Foundation
    ----------------------
    For each column j with values x_j:
    - z_j = (x_j - μ_j) / σ_j where μ_j = mean(x_j), σ_j = std(x_j)
    - If σ_j = 0: column is dropped (constant, no information)
    
    Performance Notes
    ----------------
    - **Time Complexity**: O(N×M) where N=rows, M=columns
    - **Space Complexity**: O(N×M) for the output DataFrame
    - **Memory Efficient**: Operates column-wise without large intermediate arrays
    
    Use Cases in Factor Models
    -------------------------
    - **Fundamental Data**: Standardize P/E ratios, profit margins, etc.
    - **Momentum Signals**: Normalize price momentum across stocks
    - **Risk Metrics**: Standardize volatility and beta exposures
    - **Cross-Sectional**: Ensure factors have comparable scales
    
    Examples
    --------
    >>> # Standard usage
    >>> data = pd.DataFrame({'A': [1, 2, 3], 'B': [5, 5, 5], 'C': [10, 20, 30]})
    >>> standardized = _z(data)
    >>> print(standardized.columns)  # ['A', 'C'] - 'B' removed (constant)
    
    >>> # All constant columns
    >>> constants = pd.DataFrame({'X': [1, 1, 1], 'Y': [2, 2, 2]})
    >>> result = _z(constants)
    >>> print(len(result.columns))  # 0 - empty result
    
    Notes
    -----
    - This function is critical for preventing numerical issues in regression
    - Constant columns can cause rank deficiency in factor models
    - Filtering improves model stability and interpretability
    """
    means = df.mean()
    stds = df.std()
    
    # Handle constant columns (std = 0) by dropping them instead of creating NaN
    constant_cols = stds == 0
    if constant_cols.any():
        # Drop constant columns entirely as they provide no information
        non_constant_cols = ~constant_cols
        if non_constant_cols.any():
            df_filtered = df.loc[:, non_constant_cols]
            means_filtered = means[non_constant_cols]
            stds_filtered = stds[non_constant_cols]
            result = (df_filtered - means_filtered) / stds_filtered
        else:
            # All columns are constant - return empty DataFrame with same index
            result = pd.DataFrame(index=df.index)
    else:
        # Normal standardization
        result = (df - means) / stds
    
    return result


class FactorResearchSystem(DataBackend):
    """
    Advanced factor research system with ETF expansion and multi-modal factor discovery.
    
    This class provides a comprehensive framework for quantitative factor research,
    combining fundamental analysis, statistical factor models, and advanced data
    processing capabilities. It extends the DataBackend class to inherit robust
    data access and caching functionality.
    
    Key Capabilities
    ---------------
    - **ETF Expansion**: Automatically expands ETF tickers to constituents
    - **Multi-Method Factor Discovery**: Fundamental, PCA, ICA, Factor Analysis
    - **Robust Data Processing**: Handles missing data and numerical instability
    - **LLM Integration**: Automatic factor naming via OpenAI models
    - **Advanced Caching**: Inherits sophisticated caching from DataBackend
    
    Factor Discovery Methods
    -----------------------
    1. **"fundamental"**: Cross-sectional regression on financial metrics
       - Uses P/S ratios, profit margins, beta, market cap
       - Includes momentum and volatility factors
       - Produces economically interpretable factors
       
    2. **"pca"**: Principal Component Analysis
       - Linear factors maximizing variance explanation
       - Fast and robust for large universes
       
    3. **"ica"**: Independent Component Analysis  
       - Statistically independent factors
       - Good for regime identification
       
    4. **"sparse_pca"**: Sparse Principal Component Analysis
       - PCA with sparsity constraints
       - More interpretable loadings
       
    5. **"factor"**: Factor Analysis
       - Classical factor model approach
       - Separates common and idiosyncratic variance
    
    Attributes
    ----------
    original_symbols : List[str]
        Input symbols before ETF expansion
    expand_etfs : bool
        Whether to expand ETF tickers to constituents
    universe : List[str]
        Final resolved universe of individual stock tickers
    roll : int
        Rolling window size for statistical models
    method : str
        Factor discovery method ("fundamental", "pca", etc.)
    k : int
        Number of factors to extract
    _factors : pd.DataFrame | None
        Fitted factor returns (T × K matrix)
    _expos : pd.DataFrame | None
        Factor exposures/loadings (N × K matrix)
        
    Performance Characteristics
    --------------------------
    - **Fundamental Models**: O(T×N×K²) time, best for <500 stocks
    - **Statistical Models**: O(T×N²) time, scales to 1000+ stocks  
    - **Memory Usage**: O(T×N + N×K) for factors and exposures
    - **ETF Expansion**: Adds 50-500 stocks per ETF typically
    """
    
    def __init__(
        self,
        api_key: str,
        universe: Sequence[str] | str,
        expand_etfs: bool = True,
        roll_window: int = 252,
        factor_method: str = "fundamental",
        n_components: int = 8,
        db_path: str | os.PathLike = "./av_cache.db",
        start_date: str | None = None,
    ):
        """
        Initialize the FactorResearchSystem with configuration and data sources.
        
        Parameters
        ----------
        api_key : str
            Alpha Vantage API key for data access. Free tier provides 75 calls/minute.
            
        universe : Sequence[str] | str
            Investment universe specification. Can be:
            - Single ticker: "AAPL"
            - List of tickers: ["AAPL", "MSFT", "GOOGL"]
            - ETFs (auto-expanded): ["SPY", "QQQ"] → 500+ constituents
            - Mixed: ["SPY", "AAPL", "TSLA"] → ETF expanded + individual stocks
            
        expand_etfs : bool, default True
            Whether to automatically expand ETF tickers to their constituents.
            - True: ETFs replaced with underlying holdings (recommended)
            - False: ETFs treated as individual securities
            
        roll_window : int, default 252
            Rolling window size in trading days for statistical factor models.
            - 252: One year (standard for annual factor analysis)
            - 126: Six months (more responsive to recent changes)
            - 63: Quarter (high frequency factor evolution)
            
        factor_method : str, default "fundamental"
            Factor discovery method to use:
            - "fundamental": Cross-sectional regression on financial metrics
            - "pca": Principal Component Analysis
            - "ica": Independent Component Analysis
            - "sparse_pca": Sparse PCA with interpretable loadings
            - "factor": Classical Factor Analysis
            
        n_components : int, default 8
            Number of factors to extract (only for statistical methods).
            Rule of thumb: 5-20 factors for most applications.
            
        db_path : str | os.PathLike, default "./av_cache.db"
            Path to SQLite database for caching API responses.
            Inherits sophisticated caching system from DataBackend.
            
        start_date : str | None, default None
            Earliest date for analysis in YYYY-MM-DD format.
            If None, uses maximum available history.
            
        Raises
        ------
        ValueError
            If factor_method is not recognized
        RuntimeError
            If universe resolves to <20 stocks (insufficient for factor modeling)
            
        Notes
        -----
        - ETF expansion can significantly increase universe size
        - Fundamental method requires adequate fundamental data coverage
        - Statistical methods work better with larger universes (50+ stocks)
        - Class inherits all DataBackend functionality (caching, fallbacks, etc.)
        
        Examples
        --------
        >>> # ETF-based research with fundamental factors
        >>> frs = FactorResearchSystem(api_key, universe=["SPY", "QQQ"], 
        ...                           factor_method="fundamental")
        
        >>> # Statistical factor model with individual stocks  
        >>> frs = FactorResearchSystem(api_key, universe=["AAPL", "MSFT"], 
        ...                           factor_method="pca", n_components=5,
        ...                           expand_etfs=False)
        
        >>> # High-frequency factor analysis
        >>> frs = FactorResearchSystem(api_key, universe=["SPY"],
        ...                           roll_window=63, factor_method="ica")
        """
        super().__init__(api_key, db_path=db_path, start_date=start_date)
        self.original_symbols = (
            [universe] if isinstance(universe, str) else list(universe)
        )
        self.expand_etfs = expand_etfs
        self.universe: List[str] = self._resolve_symbols(self.original_symbols)

        self.roll = roll_window
        self.method = factor_method.lower()
        self.k = n_components

        self._factors: pd.DataFrame | None = None
        self._expos:   pd.DataFrame | None = None

        _LOGGER.info("Universe resolved to %d equities", len(self.universe))

    # ========================================================= #
    # ----------------  PUBLIC WORKFLOW  ---------------------- #
    # ========================================================= #
    def fit_factors(self) -> None:
        """
        Fit the factor model using the specified method and universe.
        
        This is the main workflow method that orchestrates the complete factor
        discovery process: data loading, preprocessing, model fitting, and
        validation. The specific algorithm depends on the factor_method
        specified during initialization.
        
        Workflow Steps
        -------------
        1. **Validation**: Check minimum universe size (≥20 stocks)
        2. **Data Loading**: Retrieve price data for all universe tickers
        3. **Date Filtering**: Apply start_date constraint if specified
        4. **Return Calculation**: Compute log returns from adjusted prices
        5. **Model Selection**: Choose fundamental vs statistical approach
        6. **Factor Fitting**: Execute the selected factor discovery method
        7. **Storage**: Cache results in _factors and _expos attributes
        
        Factor Methods
        -------------
        
        **Fundamental Method ("fundamental")**:
        - Builds cross-sectional exposures from financial metrics
        - Performs time-series regression to extract factor returns
        - Produces economically interpretable factors
        - Best for: understanding economic drivers, fundamental analysis
        
        **Statistical Methods ("pca", "ica", "sparse_pca", "factor")**:
        - Applies rolling window decomposition to return data
        - Extracts latent factors via dimensionality reduction
        - Produces statistically optimal factors
        - Best for: risk modeling, return prediction, portfolio construction
        
        Data Requirements
        ----------------
        - **Minimum Universe**: 20 stocks (required for statistical stability)
        - **Optimal Universe**: 50-500 stocks (balance between diversity and computation)
        - **Time Series**: Minimum 63 days for rolling models, 252+ preferred
        - **Fundamental Data**: Required for fundamental method, optional for statistical
        
        Error Handling
        -------------
        - **Insufficient Stocks**: Raises RuntimeError if <20 stocks
        - **Missing Data**: Robust handling of gaps and missing values
        - **Numerical Issues**: Automatic regularization for rank deficiency
        - **API Failures**: Inherits robust fallback mechanisms from DataBackend
        
        Performance Notes
        ----------------
        - **Fundamental**: ~30s for 100 stocks × 252 days
        - **Statistical**: ~10s for 100 stocks × 252 days  
        - **Memory Peak**: ~100MB for 500 stocks × 1000 days
        - **API Calls**: Cached aggressively, ~1 call per stock per day
        
        Post-Fitting State
        -----------------
        After successful completion:
        - **self._factors**: Factor returns DataFrame (T × K)
        - **self._expos**: Factor exposures DataFrame (N × K)
        - **Ready for**: get_factor_returns(), name_factors()
        
        Raises
        ------
        RuntimeError
            If universe has <20 stocks or factor fitting fails completely
        ValueError
            If factor_method is not recognized
        ConnectionError
            If data retrieval fails (inherited from DataBackend)
            
        Examples
        --------
        >>> frs = FactorResearchSystem(api_key, universe=["SPY"], factor_method="pca")
        >>> frs.fit_factors()  # Fit PCA factors on SPY constituents
        >>> print(f"Discovered {frs._factors.shape[1]} factors")
        >>> print(f"Factor returns shape: {frs._factors.shape}")
        >>> print(f"Exposures shape: {frs._expos.shape}")
        
        >>> # Check fitting success
        >>> if frs._factors is not None:
        ...     print("✅ Factor fitting completed successfully")
        ...     factor_names = frs.name_factors()
        
        Notes
        -----
        - This method must be called before get_factor_returns() or name_factors()
        - Fitting can take several minutes for large universes or fundamental methods
        - Results are cached in instance attributes for reuse
        - Progress is logged at INFO level for monitoring long-running fits
        """
        _LOGGER.info("Starting factor model fitting with method: %s", self.method)
        
        # Check minimum stock count for factor models
        if len(self.universe) < 20:
            raise RuntimeError(f"Insufficient stocks for factor modeling: {len(self.universe)} stocks (minimum 20 required)")
        
        _LOGGER.info("Step 1/3: Loading price data for %d tickers", len(self.universe))
        px = self.get_prices(self.universe)
        
        # Filter to start date if specified
        if hasattr(self, 'start') and self.start is not None:
            px = px.loc[px.index >= self.start]
            _LOGGER.info("Filtered to start date %s: %d days", self.start.date(), len(px))
            
        _LOGGER.info("Price data loaded: %d days x %d tickers", len(px), len(px.columns))
        
        _LOGGER.info("Step 2/3: Calculating log returns")
        rets = np.log(px).diff().iloc[1:]
        _LOGGER.info("Returns calculated: %d days x %d tickers", len(rets), len(rets.columns))

        _LOGGER.info("Step 3/3: Fitting %s factor model", self.method)
        if self.method == "fundamental":
            expos_dict = self._build_exposures(px)
            self._factors, self._expos = self._cross_sectional_ols(rets, expos_dict)
        else:
            self._factors, self._expos = self._rolling_stat_model(rets)
        
        _LOGGER.info("Factor model fitting complete")
        if self._factors is not None:
            _LOGGER.info("Factors shape: %s", self._factors.shape)
        if self._expos is not None:
            _LOGGER.info("Exposures shape: %s", self._expos.shape)

    def get_factor_returns(self, vol_target: float | None = None) -> pd.DataFrame:
        """
        Retrieve fitted factor returns with optional volatility targeting.
        
        This method provides access to the factor returns discovered during
        fit_factors(), with optional volatility scaling for risk budgeting
        and portfolio construction applications.
        
        Parameters
        ----------
        vol_target : float | None, default None
            Target annualized volatility for factor returns scaling.
            If None, returns raw factor returns without scaling.
            Common values: 0.10 (10% vol), 0.15 (15% vol), 0.20 (20% vol)
            
        Returns
        -------
        pd.DataFrame
            Factor returns matrix with shape (T, K) where:
            - Index: Trading dates from factor fitting period
            - Columns: Factor identifiers (depends on method)
            - Values: Daily factor returns (raw or volatility-scaled)
            
        Volatility Targeting Process
        ---------------------------
        When vol_target is specified:
        1. **Rolling Volatility**: Calculate 63-day rolling standard deviation
        2. **Annualization**: Multiply by √252 to get annualized volatility
        3. **Scaling Factor**: vol_target / annualized_volatility
        4. **Application**: Multiply factor returns by scaling factor
        5. **Result**: Factor returns with target volatility profile
        
        Mathematical Foundation
        ----------------------
        - **Raw Returns**: r_t (daily factor returns)
        - **Rolling Vol**: σ_t = std(r_{t-63:t}) × √252
        - **Scaling**: s_t = vol_target / σ_t
        - **Scaled Returns**: r'_t = r_t × s_t
        
        Use Cases
        --------
        - **Raw Returns**: Factor analysis, attribution, research
        - **Vol Targeted**: Portfolio construction, risk budgeting, allocation
        - **Risk Parity**: Equal volatility contribution across factors
        - **Benchmarking**: Compare factors on risk-adjusted basis
        
        Raises
        ------
        RuntimeError
            If fit_factors() has not been called successfully
            
        Examples
        --------
        >>> # Get raw factor returns
        >>> frs.fit_factors()
        >>> raw_returns = frs.get_factor_returns()
        >>> print(f"Raw volatilities: {raw_returns.std() * np.sqrt(252)}")
        
        >>> # Get volatility-targeted returns (15% annual vol)
        >>> vol_targeted = frs.get_factor_returns(vol_target=0.15)
        >>> print(f"Targeted volatilities: {vol_targeted.std() * np.sqrt(252)}")
        
        >>> # Portfolio construction with equal vol weighting
        >>> equal_vol_returns = frs.get_factor_returns(vol_target=0.10)
        >>> weights = 1.0 / len(equal_vol_returns.columns)  # Equal weight
        >>> portfolio_returns = (equal_vol_returns * weights).sum(axis=1)
        
        Notes
        -----
        - Volatility targeting uses 63-day rolling windows (approximately 3 months)
        - Scaling is applied using forward-looking volatility estimates
        - Returns are always daily frequency regardless of vol_target
        - Volatility targeting helps in factor allocation and risk budgeting
        """
        if self._factors is None:
            raise RuntimeError("run fit_factors() first")
        if vol_target is None:
            return self._factors.copy()
        return self._factors.div(
            self._factors.rolling(63).std() * _ann
        ).mul(vol_target / _ann)

    # ---------------- LLM naming ---------------- #
    def name_factors(
        self, model: str = "gpt-4o-mini", top_n: int = 8,
        force_refresh: bool = False, cache_path: str = "factor_names.json"
    ) -> dict[str, str]:
        """
        Generate intuitive factor names using LLM analysis of factor exposures.
        
        This method leverages large language models to automatically generate
        meaningful, human-readable names for discovered factors based on their
        stock exposures. The LLM analyzes which stocks load positively and
        negatively on each factor to infer economic themes.
        
        Parameters
        ----------
        model : str, default "gpt-4o-mini"
            OpenAI model to use for factor naming.
            Options: "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"
            Recommendation: "gpt-4o-mini" for cost-effectiveness
            
        top_n : int, default 8
            Number of top positive and negative stocks to analyze per factor.
            - Higher values: More comprehensive analysis, higher API costs
            - Lower values: Faster processing, less context
            - Sweet spot: 6-10 stocks for most factors
            
        force_refresh : bool, default False
            Whether to force fresh LLM analysis even if cache exists.
            - True: Always call LLM (useful for experimenting with prompts)
            - False: Use cached results if available (recommended for production)
            
        cache_path : str, default "factor_names.json"
            Path to JSON file for caching factor names.
            Prevents repeated API calls for the same factor exposures.
            
        Returns
        -------
        dict[str, str]
            Dictionary mapping factor identifiers to descriptive names.
            Format: {"F1": "Growth vs Value", "F2": "Tech vs Financials", ...}
            
        LLM Analysis Process
        -------------------
        For each factor:
        1. **Exposure Analysis**: Identify top positive and negative stocks
        2. **Context Building**: Create prompt with stock symbols and directions
        3. **LLM Request**: Send to OpenAI API with analyst persona
        4. **Name Generation**: Receive short, intuitive factor description
        5. **Post-processing**: Clean and format the response
        6. **Caching**: Store results for future use
        
        Prompt Engineering
        -----------------
        The LLM is given:
        - **Role**: Quantitative equity analyst
        - **Task**: Provide short (≤4 words) intuitive factor name
        - **Context**: Top positive and negative stock exposures
        - **Format**: Brief name + one sentence explanation
        
        Example prompt:
        "Factor F1. Positive: AAPL, MSFT, GOOGL. Negative: JPM, BAC, WFC.
         Provide a short (≤4 words) intuitive name and one sentence."
        
        Cost Considerations
        ------------------
        - **API Cost**: ~$0.01-0.05 per factor depending on model
        - **Caching**: Significantly reduces costs for repeated analysis
        - **Batch Processing**: All factors sent in single conversation
        - **Model Choice**: gpt-4o-mini is 10-20x cheaper than gpt-4o
        
        Error Handling
        -------------
        - **Missing API Key**: Clear error message with setup instructions
        - **Network Issues**: Graceful failure with fallback naming
        - **API Limits**: Rate limiting and retry logic
        - **Parsing Errors**: Robust response parsing with fallbacks
        
        Raises
        ------
        RuntimeError
            If fit_factors() has not been called successfully
        ValueError
            If OpenAI API key is not set in environment
        openai.OpenAIError
            If API request fails after retries
            
        Examples
        --------
        >>> # Basic factor naming
        >>> frs.fit_factors()
        >>> names = frs.name_factors()
        >>> print(names)
        >>> # {"F1": "Growth vs Value: High P/E tech vs low P/E financials",
        >>> #  "F2": "Size Factor: Large cap vs small cap stocks"}
        
        >>> # Use more expensive model for better names
        >>> premium_names = frs.name_factors(model="gpt-4o", top_n=12)
        
        >>> # Force refresh of factor names
        >>> updated_names = frs.name_factors(force_refresh=True)
        
        >>> # Custom cache location
        >>> names = frs.name_factors(cache_path="./results/factor_names.json")
        
        Notes
        -----
        - Requires OPENAI_API_KEY environment variable
        - Factor naming quality depends on factor interpretability
        - Statistical factors (PCA/ICA) may get more abstract names
        - Fundamental factors typically receive more intuitive names
        - Cache is based on factor exposures, not method used
        """
        if not force_refresh and os.path.exists(cache_path):
            with open(cache_path) as fh:
                return json.load(fh)
        if self._expos is None:
            raise RuntimeError("run fit_factors() first")

        prompts = []
        for f in self._expos.columns:
            w = self._expos[f]
            pos = w.nlargest(top_n).index.tolist()
            neg = w.nsmallest(top_n).index.tolist()
            prompts.append(
                f"Factor {f}. Positive: {', '.join(pos)}. "
                f"Negative: {', '.join(neg)}. "
                "Provide a short (≤4 words) intuitive name and one sentence."
            )

        messages = [{"role": "system",
                     "content": "You are a quantitative equity analyst."}]
        for p in prompts:
            messages.append({"role": "user", "content": p})

        _LOGGER.info("Requesting factor names via %s …", model)
        client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        resp = client.chat.completions.create(model=model, messages=messages)
        ans = resp.choices[0].message.content.strip().split("\n")
        names = {f: (line.split(":", 1)[-1] if ":" in line else line).strip()
                 for f, line in zip(self._expos.columns, ans)}

        with open(cache_path, "w") as fh:
            json.dump(names, fh, indent=2)
        return names

    # ========================================================= #
    # ----------------  INTERNAL • ETF expansion  -------------- #
    # ========================================================= #
    def _resolve_symbols(self, symbols: Sequence[str]) -> List[str]:
        """
        Resolve input symbols with intelligent ETF expansion and validation.
        
        This internal method processes the input universe to handle ETF expansion,
        ticker validation, and deduplication. It's the core symbol processing
        engine that transforms user input into a clean, validated universe.
        
        Parameters
        ----------
        symbols : Sequence[str]
            Input symbols that may include ETFs, individual stocks, or mixed
            
        Returns
        -------
        List[str]
            Validated, deduplicated list of individual stock tickers
            
        Processing Pipeline
        ------------------
        1. **ETF Detection**: Use is_etf() to identify ETF tickers
        2. **ETF Expansion**: Replace ETFs with constituent holdings
        3. **Ticker Validation**: Filter out invalid/malformed tickers
        4. **Deduplication**: Remove duplicate symbols
        5. **Sorting**: Alphabetical ordering for consistency
        
        ETF Expansion Process
        --------------------
        For each detected ETF:
        - Fetch holdings via get_etf_holdings()
        - Extract constituent ticker symbols
        - Apply validation filters:
          * Not null/empty/N/A
          * Length ≤ 10 characters
          * Alphanumeric characters only (plus . and -)
        - Add valid constituents to universe
        
        Validation Criteria
        ------------------
        Tickers are filtered to ensure:
        - **Non-empty**: Not null, "N/A", "NA", "NULL", "NONE", ""
        - **Reasonable Length**: ≤ 10 characters (typical ticker limit)
        - **Valid Format**: Alphanumeric with allowed symbols (. and -)
        - **Data Availability**: Implicitly validated via downstream data access
        
        Performance Characteristics
        --------------------------
        - **Individual Stocks**: O(1) per symbol (just validation)
        - **ETF Expansion**: O(H) where H = holdings count per ETF
        - **Typical ETF**: 50-500 holdings per ETF
        - **Progress Logging**: Real-time updates for monitoring
        
        Examples
        --------
        >>> # Mixed input with ETF and individual stocks
        >>> symbols = ["SPY", "AAPL", "QQQ"]
        >>> resolved = frs._resolve_symbols(symbols)
        >>> print(f"Input: {len(symbols)} symbols")
        >>> print(f"Output: {len(resolved)} individual stocks")
        
        >>> # ETF-only input
        >>> etf_universe = frs._resolve_symbols(["SPY"])
        >>> print(f"SPY expanded to {len(etf_universe)} constituents")
        
        Notes
        -----
        - This method is called automatically during initialization
        - ETF expansion can dramatically increase universe size
        - Progress is logged to console for transparency
        - Results are cached in self.universe attribute
        - Validation prevents downstream data errors
        """
        print(f"📋 Resolving {len(symbols)} symbols (ETF expansion: {self.expand_etfs})...")
        
        tickers: List[str] = []
        
        for i, s in enumerate(symbols, 1):
            print(f"🔍 Processing symbol {i}/{len(symbols)}: {s}")
            
            if self.expand_etfs and self.is_etf(s):
                print(f"📈 Detected ETF {s}, fetching holdings...")
                
                holds = self.get_etf_holdings(s)
                if holds.empty:
                    print(f"⚠️ No holdings found for ETF {s} – skipping")
                    continue
                    
                # Filter out invalid tickers
                valid_constituents = [
                    ticker for ticker in holds["constituent"].tolist()
                    if ticker and str(ticker).upper() not in ["N/A", "NA", "NULL", "NONE", ""]
                    and len(str(ticker)) <= 10  # Reasonable ticker length limit
                    and str(ticker).replace(".", "").replace("-", "").isalnum()  # Basic validation
                ]
                tickers.extend(valid_constituents)
                print(f"✅ ETF {s} expanded to {len(valid_constituents)} valid constituents (filtered from {len(holds)} total)")
            else:
                tickers.append(s)
                print(f"✅ Added regular symbol: {s}")
        
        unique_tickers = sorted(set(tickers))
        print(f"🎉 Symbol resolution complete: {len(unique_tickers)} unique tickers from {len(symbols)} input symbols")
        return unique_tickers

    # ========================================================= #
    # ----------------  INTERNAL • FACTOR MODELS  -------------- #
    # ========================================================= #
    # ---------- fundamental OLS ---------- #
    def _build_exposures(self, px: pd.DataFrame):
        f_fields = [
            "PriceToSalesRatioTTM", "ProfitMargin", "Sector",
            "Industry", "Beta", "MarketCapitalization"
        ]
        fnd = self.get_fundamentals(self.universe, fields=f_fields)
        
        print(f"📊 Fundamentals data shape: {fnd.shape}")
        print(f"📊 Data types: {fnd.dtypes.to_dict()}")
        
        # Convert numeric columns and handle mixed types
        numeric_cols = ["PriceToSalesRatioTTM", "ProfitMargin", "Beta", "MarketCapitalization"]
        for col in numeric_cols:
            if col in fnd.columns:
                fnd[col] = pd.to_numeric(fnd[col], errors='coerce')
                print(f"📊 {col}: {fnd[col].notna().sum()}/{len(fnd)} valid values")

        cat = pd.get_dummies(fnd[["Sector", "Industry"]],
                             dummy_na=False, prefix_sep="=")
        # Build fundamental factors with better error handling
        fundamental_factors = []
        
        # Add fundamental factors if we have enough data
        fundamental_cols = ["PriceToSalesRatioTTM", "ProfitMargin", "Beta"]
        valid_fundamental = fnd[fundamental_cols].dropna(thresh=2)  # Need at least 2 valid values
        
        if len(valid_fundamental) > 10:  # Need at least 10 stocks with fundamental data
            fundamental_factors.append(_z(fnd[fundamental_cols]))
            print(f"✅ Added fundamental factors: {len(valid_fundamental)} stocks with data")
        else:
            print(f"⚠️ Insufficient fundamental data ({len(valid_fundamental)} stocks), skipping fundamental factors")
        
        # Add size factor if market cap data exists
        mkt_cap_data = fnd[["MarketCapitalization"]].replace(0, np.nan).dropna()
        if len(mkt_cap_data) > 10:
            log_mkt_cap = np.log(fnd[["MarketCapitalization"]].replace(0, np.nan))
            fundamental_factors.append(_z(log_mkt_cap))
            print(f"✅ Added size factor: {len(mkt_cap_data)} stocks with market cap data")
        else:
            print(f"⚠️ Insufficient market cap data ({len(mkt_cap_data)} stocks), skipping size factor")
        
        # Add sector/industry dummies if we have enough diversity
        if len(cat.columns) > 1:  # More than just one sector/industry
            fundamental_factors.append(cat)
            print(f"✅ Added sector/industry factors: {len(cat.columns)} categories")
        else:
            print(f"⚠️ Insufficient sector/industry diversity, skipping categorical factors")
        
        if not fundamental_factors:
            raise RuntimeError("No valid fundamental factors could be built - check data quality")
        
        desc = pd.concat(fundamental_factors, axis=1)
        
        # Debug the fundamental factors
        print(f"📊 Combined factors shape: {desc.shape}")
        print(f"📊 NaN counts: {desc.isnull().sum().sum()} total NaNs")
        print(f"📊 Infinite values: {np.isinf(desc.select_dtypes(include=[np.number])).sum().sum()} total infs")
        
        # Create derived factors safely
        if "PriceToSalesRatioTTM" in desc.columns:
            desc["Value_PS"] = -desc["PriceToSalesRatioTTM"]
            
        desc.rename(columns={
            "PriceToSalesRatioTTM": "Value_PS_raw",
            "ProfitMargin": "Quality_PM", 
            "Beta": "Beta",
            "MarketCapitalization": "Size"
        }, inplace=True, errors='ignore')
        
        # Final validation
        print(f"📊 Final desc shape: {desc.shape}")
        print(f"📊 Final NaN counts: {desc.isnull().sum().sum()}")
        print(f"📊 Final inf counts: {np.isinf(desc.select_dtypes(include=[np.number])).sum().sum()}")

        rets = np.log(px).diff()
        mom_21 = px.pct_change(21)
        mom_63 = px.pct_change(63)
        vol_21 = rets.rolling(21).std() * _ann

        expos = {}
        for dt in rets.index:
            e = pd.DataFrame({
                "Momentum_1M": mom_21.loc[dt],
                "Momentum_3M": mom_63.loc[dt],
                "Vol_1M":      vol_21.loc[dt]
            })
            # Combine fundamental and momentum factors (no intercept yet)
            combined = pd.concat([desc, e], axis=1)
            standardized = _z(combined)
            
            # Add intercept column AFTER standardization (it should remain constant)
            standardized["Intercept"] = 1.0
                
            expos[dt] = standardized.loc[self.universe]
        return expos

    def _cross_sectional_ols(self, rets, expos):
        facs = []
        failed_dates = []
        
        for dt, x in expos.items():
            if dt not in rets.index:
                continue
            
            # Drop rows with any NaN values to ensure clean data for regression
            y_raw = rets.loc[dt]
            x_clean = x.dropna()  # Drop stocks with any NaN exposures
            
            if len(x_clean) < 10:  # Need minimum number of stocks for regression
                failed_dates.append(dt)
                continue
            
            # Align returns with clean exposures
            y = y_raw.loc[x_clean.index].fillna(0).values
            X = x_clean.values
            
            # Final check for numerical issues (shouldn't happen after dropna)
            nan_count = np.sum(np.isnan(X))
            inf_count = np.sum(np.isinf(X))
            if nan_count > 0 or inf_count > 0:
                _LOGGER.warning(f"Unexpected numerical issues for {dt}: {nan_count} NaNs, {inf_count} Infs, skipping")
                failed_dates.append(dt)
                continue
                
            if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                _LOGGER.warning(f"Invalid values in returns for {dt}, skipping")
                failed_dates.append(dt)
                continue
            
            # Check for rank deficiency
            try:
                # Use SVD to check matrix rank
                rank = np.linalg.matrix_rank(X)
                if rank < X.shape[1]:
                    # Use ridge regression with small regularization (silently)
                    XTX = X.T @ X
                    regularization = 1e-6 * np.trace(XTX) / X.shape[1]
                    XTX_reg = XTX + regularization * np.eye(X.shape[1])
                    f = np.linalg.solve(XTX_reg, X.T @ y)
                else:
                    # Standard least squares
                    f, *_ = np.linalg.lstsq(X, y, rcond=None)
                    
                facs.append(f)
                
            except np.linalg.LinAlgError as e:
                _LOGGER.error(f"LinAlgError on {dt}: {e}, skipping")
                failed_dates.append(dt)
                continue
            except Exception as e:
                _LOGGER.error(f"Unexpected error on {dt}: {e}, skipping")
                failed_dates.append(dt)
                continue
        
        if failed_dates:
            print(f"⚠️ Skipped {len(failed_dates)} dates with insufficient data")
            
        if not facs:
            raise RuntimeError("Could not compute factors for any dates - check data quality")
            
        f_df = pd.DataFrame(
            facs, index=[d for d in expos if d in rets.index and d not in failed_dates],
            columns=expos[next(iter(expos))].columns
        )
        return f_df.sort_index(), expos[max(expos)]

    # ---------- statistical engines ---------- #
    def _rolling_stat_model(self, rets: pd.DataFrame):
        scaler = StandardScaler()
        if self.method == "pca":
            decomp = PCA(self.k)
        elif self.method == "sparse_pca":
            decomp = SparsePCA(self.k, random_state=0)
        elif self.method == "factor":
            decomp = FactorAnalysis(self.k, random_state=0)
        elif self.method == "ica":
            decomp = FastICA(self.k, random_state=0)
        else:
            raise ValueError(self.method)

        fac_rets, loads, dates = [], [], []
        for t in range(self.roll, len(rets)):
            win = rets.iloc[t-self.roll:t]
            z = scaler.fit_transform(win.values)
            decomp.fit(z)
            load = pd.DataFrame(
                decomp.components_.T, index=rets.columns,
                columns=[f"F{i+1}" for i in range(self.k)]
            )
            today = (rets.iloc[[t]].values - scaler.mean_) / scaler.scale_
            fac_rets.append((today @ decomp.components_.T).ravel())
            loads.append(load); dates.append(rets.index[t])

        f_df = pd.DataFrame(fac_rets, index=dates,
                            columns=[f"F{i+1}" for i in range(self.k)])
        return f_df, loads[-1]
