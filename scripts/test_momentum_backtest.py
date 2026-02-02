"""
Legacy Factor Research System Test: Historical PCA-Based Factor Discovery
========================================================================

This module represents an earlier version of the factor research system,
providing a comprehensive implementation of PCA-based factor discovery
with ETF holdings expansion and momentum-based factor ranking.

Purpose
-------
- **Historical Reference**: Demonstrates earlier approach to factor research
- **PCA Implementation**: Pure statistical factor discovery using Principal Component Analysis
- **ETF Expansion**: Automatic expansion of ETF tickers to constituent holdings
- **Factor Analysis**: Comprehensive factor discovery, loading analysis, and ranking

Key Features
-----------
- **Data Preparation**: Robust handling of missing data and standardization
- **PCA Factor Discovery**: Principal component analysis for unsupervised factor extraction
- **Factor Attribution**: Assignment of stocks to factors based on loadings
- **Performance Ranking**: Time-series analysis of factor performance
- **Visualization**: Explained variance plots and factor analysis charts

Legacy Architecture
------------------
This implementation extends AlphaVantageFactorSystem with:
- Enhanced data cleaning and standardization
- PCA-based factor discovery methodology
- Factor loading analysis and stock attribution
- Time-series factor return calculation
- Performance ranking and evaluation

Comparison with Current System
-----------------------------
**Legacy (this file):**
- Pure PCA approach
- Basic data cleaning
- Simple factor attribution
- Limited factor validation

**Current (research.py):**
- Multi-modal factor discovery (fundamental + statistical)
- Advanced data processing with cross-sectional regression
- LLM-powered factor naming
- Comprehensive factor validation
- ETF auto-expansion with validation

Dependencies
-----------
- **Core**: pandas, numpy, scikit-learn
- **Visualization**: matplotlib, seaborn  
- **Data**: alphavantage_system.AlphaVantageFactorSystem
- **API**: requests for ETF holdings

Integration Notes
----------------
- **Legacy Status**: This implementation is superseded by research.py
- **Educational Value**: Demonstrates PCA-based factor discovery approach
- **Reference Implementation**: Shows evolution of factor research methodology
- **Testing Purpose**: Validates basic PCA functionality and data handling

Usage
-----
This script serves as a historical reference and testing tool for
PCA-based factor discovery. For current factor research, use the
modern research.py implementation with enhanced capabilities.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import requests

# Assuming alphavantage_system.py is in the same directory or accessible in the path
# It provides the AlphaVantageFactorSystem class for data handling.
from alphavantage_system import AlphaVantageFactorSystem

class FactorResearchSystem(AlphaVantageFactorSystem):
    """
    Legacy factor research system implementing PCA-based factor discovery.
    
    This class extends AlphaVantageFactorSystem to provide comprehensive
    factor discovery capabilities using Principal Component Analysis (PCA).
    It includes data preparation, factor extraction, loading analysis,
    and performance evaluation functionality.
    
    Core Functionality
    -----------------
    - **Data Preparation**: Robust cleaning, missing value handling, standardization
    - **Factor Discovery**: PCA-based unsupervised factor extraction
    - **Loading Analysis**: Factor-to-stock attribution based on loadings
    - **Return Calculation**: Time-series factor return computation
    - **Performance Ranking**: Factor ranking based on historical performance
    - **Visualization**: Explained variance and factor analysis plots
    
    Attributes
    ----------
    pca : sklearn.decomposition.PCA
        Fitted PCA model for factor discovery
    components : pd.DataFrame
        Factor loadings matrix (stocks × factors)
    factor_returns : pd.DataFrame
        Time series of factor returns
    factor_loadings : pd.DataFrame
        Scaled factor loadings for interpretation
    standardized_returns : pd.DataFrame
        Standardized stock returns used for PCA
    returns_df : pd.DataFrame
        Cleaned stock returns before standardization
        
    Methods
    -------
    prepare_returns_data(price_data, nan_threshold=0.1)
        Clean and standardize return data for factor analysis
    discover_factors(n_factors=10)
        Extract factors using PCA methodology
    calculate_factor_returns()
        Compute time series of factor returns
    assign_equities_to_factors(n_top_equities=5)
        Attribute stocks to factors based on loadings
    rank_factors_by_date(target_date, lookback_days=20)
        Rank factors by performance over specified period
    plot_explained_variance()
        Visualize cumulative explained variance
        
    Legacy Notes
    -----------
    This implementation represents an earlier approach to factor research
    focused on pure statistical methods. The current research.py provides
    enhanced capabilities including fundamental factors, LLM naming,
    and advanced validation.
    """

    def __init__(self, api_key, data_dir='./av_data', db_path='./av_cache.db'):
        """
        Initialize the legacy factor research system.
        
        Parameters
        ----------
        api_key : str
            Alpha Vantage API key for data access
        data_dir : str, default './av_data'
            Directory for cached data files
        db_path : str, default './av_cache.db'
            Path to SQLite database for caching
        """
        super().__init__(api_key, data_dir, db_path)
        self.pca = None
        self.components = None
        self.factor_returns = None
        self.factor_loadings = None
        self.standardized_returns = None
        self.returns_df = None
        self.logger.info("FactorResearchSystem initialized.")

    def get_etf_holdings(self, etf_symbol):
        """
        Fetches ETF holdings from Alpha Vantage.
        
        Args:
            etf_symbol (str): The symbol of the ETF (e.g., "IWM").

        Returns:
            list: A list of ticker symbols held by the ETF.
        """
        self.logger.info(f"Fetching holdings for {etf_symbol}...")
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'ETF_PROFILE',
            'symbol': etf_symbol,
            'apikey': self.api_key
        }
        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            
            if "holdings" in data and data["holdings"]:
                tickers = [holding['symbol'] for holding in data['holdings'] if 'symbol' in holding]
                self.logger.info(f"Successfully retrieved {len(tickers)} holdings for {etf_symbol}.")
                # Clean tickers to remove any that might cause issues (e.g., with '.')
                cleaned_tickers = [t.replace('.', '-') for t in tickers if isinstance(t, str) and t.strip()]
                return cleaned_tickers
            else:
                self.logger.error(f"Could not retrieve holdings from API response for {etf_symbol}: {data}")
                return []
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching ETF holdings for {etf_symbol}: {e}")
            return []


    def prepare_returns_data(self, price_data, nan_threshold=0.1):
        """
        Prepares the daily returns data for PCA. This involves calculating returns, 
        handling missing values more robustly, and standardizing the data.

        Args:
            price_data (pd.DataFrame): DataFrame of adjusted close prices, with dates as index and tickers as columns.
            nan_threshold (float): The maximum percentage of NaN values allowed for a stock column. 
                                   Stocks exceeding this threshold will be dropped. Default is 0.1 (10%).

        Returns:
            pd.DataFrame: A cleaned and standardized DataFrame of daily returns.
        """
        self.logger.info("Preparing returns data for PCA...")
        if price_data.empty:
            self.logger.error("Input price_data is empty. Cannot prepare returns.")
            return pd.DataFrame()

        # Calculate daily percentage returns
        returns = price_data.pct_change()
        self.logger.info(f"Initial returns shape: {returns.shape}")

        # Drop rows where all values are NaN (typically the first row)
        returns.dropna(how='all', inplace=True)

        # Drop columns (stocks) that have too many missing values
        n_rows = len(returns)
        min_valid_points = int(n_rows * (1 - nan_threshold))
        returns_filtered_cols = returns.dropna(axis=1, thresh=min_valid_points)
        self.logger.info(f"Shape after dropping columns with >{nan_threshold:.0%} NaNs: {returns_filtered_cols.shape}")
        
        # Forward-fill remaining missing values. This assumes the last known price holds.
        returns_filled = returns_filtered_cols.fillna(method='ffill')
        
        # Back-fill any NaNs that might remain at the beginning of the series
        returns_filled = returns_filled.fillna(method='bfill')

        # Drop any rows or columns that might still have NaNs
        returns_cleaned = returns_filled.dropna(how='any', axis=0)
        returns_cleaned = returns_cleaned.dropna(how='any', axis=1)
        self.logger.info(f"Final shape after all cleaning: {returns_cleaned.shape}")

        if returns_cleaned.empty:
            self.logger.error("Dataframe is empty after cleaning. Check data quality or increase nan_threshold.")
            return pd.DataFrame()

        self.returns_df = returns_cleaned.copy()

        # Standardize the returns data (mean=0, variance=1)
        scaler = StandardScaler()
        standardized_returns = pd.DataFrame(scaler.fit_transform(returns_cleaned),
                                            index=returns_cleaned.index,
                                            columns=returns_cleaned.columns)
        self.standardized_returns = standardized_returns
        self.logger.info("Returns data has been cleaned and standardized.")
        return standardized_returns

    def discover_factors(self, n_factors=10):
        """
        Perform PCA-based factor discovery on standardized return data.
        
        This method applies Principal Component Analysis to extract orthogonal
        factors that capture the main sources of variation in stock returns.
        It automatically adjusts the number of factors if the requested amount
        exceeds the maximum possible components.
        
        Parameters
        ----------
        n_factors : int, default 10
            Number of principal components (factors) to extract.
            Will be adjusted to min(n_samples, n_features) if too large.
            
        Attributes Set
        -------------
        pca : sklearn.decomposition.PCA
            Fitted PCA model with explained variance information
        components : pd.DataFrame
            Factor loadings matrix with shape (stocks, factors)
        factor_loadings : pd.DataFrame
            Scaled factor loadings for interpretation (loadings × √eigenvalue)
            
        Factor Interpretation
        --------------------
        - **Components**: Raw PCA eigenvectors (unit vectors)
        - **Factor Loadings**: Scaled components representing factor influence
        - **Explained Variance**: Proportion of total variance captured by each factor
        - **Cumulative Variance**: Running total of explained variance
        
        Notes
        -----
        - Requires standardized_returns to be available from prepare_returns_data()
        - Factor loadings are scaled by square root of explained variance for interpretation
        - First factor typically captures market-wide movements
        - Subsequent factors capture sector, style, or other systematic effects
        """
        if self.standardized_returns is None or self.standardized_returns.empty:
            self.logger.error("Standardized returns not available. Please run prepare_returns_data() first.")
            return

        n_samples, n_features = self.standardized_returns.shape
        max_components = min(n_samples, n_features)

        if n_factors > max_components:
            self.logger.warning(
                f"Requested n_factors ({n_factors}) is greater than the max possible components ({max_components}). "
                f"Adjusting n_factors to {max_components}."
            )
            n_factors = max_components

        if n_factors == 0:
            self.logger.error("Cannot discover factors with 0 components. Check data preparation.")
            return

        self.logger.info(f"Discovering {n_factors} factors using PCA...")
        self.pca = PCA(n_components=n_factors)
        self.pca.fit(self.standardized_returns)

        self.components = pd.DataFrame(self.pca.components_.T,
                                       columns=[f'Factor_{i+1}' for i in range(n_factors)],
                                       index=self.standardized_returns.columns)
        self.factor_loadings = self.components * np.sqrt(self.pca.explained_variance_)

        self.logger.info("Factor discovery complete.")
        self.logger.info(f"Explained variance by each factor: {[f'{var:.2%}' for var in self.pca.explained_variance_ratio_]}")
        self.logger.info(f"Total variance explained by {n_factors} factors: {np.sum(self.pca.explained_variance_ratio_):.2%}")


    def plot_explained_variance(self):
        """
        Plots the cumulative explained variance of the principal components.
        """
        if self.pca is None:
            self.logger.error("PCA has not been run yet. Run discover_factors() first.")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_), marker='o', linestyle='--')
        plt.title('Cumulative Explained Variance by PCA Factors')
        plt.xlabel('Number of Factors')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid(True)
        plt.show()


    def assign_equities_to_factors(self, n_top_equities=5):
        """
        Assigns equities to each discovered factor based on factor loadings.
        """
        if self.factor_loadings is None:
            self.logger.error("Factor loadings not available. Run discover_factors() first.")
            return {}

        self.logger.info(f"Assigning top {n_top_equities} equities to each factor...")
        assignments = {}
        for factor in self.factor_loadings.columns:
            abs_loadings = self.factor_loadings[factor].abs().sort_values(ascending=False)
            top_equities = abs_loadings.head(n_top_equities)
            assignments[factor] = top_equities.to_dict()
            print(f"\n--- {factor} ---")
            print(f"Top {n_top_equities} most influential equities (by absolute loading):")
            print(top_equities)
        
        return assignments

    def calculate_factor_returns(self):
        """
        Calculates the time series of returns for each discovered factor.
        """
        if self.components is None:
            self.logger.error("PCA components not available. Run discover_factors() first.")
            return

        self.logger.info("Calculating factor returns time series...")
        factor_returns = self.standardized_returns.dot(self.components)
        factor_returns.columns = [f'Factor_{i+1}_Returns' for i in range(self.components.shape[1])]
        self.factor_returns = factor_returns
        self.logger.info("Factor returns calculation complete.")
        return factor_returns

    def rank_factors_by_date(self, target_date, lookback_days=20):
        """
        Ranks the discovered factors based on their performance up to a given date.
        """
        if self.factor_returns is None:
            self.logger.error("Factor returns not available. Run calculate_factor_returns() first.")
            return None

        target_dt = pd.to_datetime(target_date)
        self.logger.info(f"Ranking factors for date {target_dt.date()} with a {lookback_days}-day lookback...")
        
        if target_dt not in self.factor_returns.index:
            try:
                target_dt = self.factor_returns.index[self.factor_returns.index <= target_dt][-1]
                self.logger.warning(f"Target date not found. Using closest previous date: {target_dt.date()}")
            except IndexError:
                self.logger.error("No data available on or before the target date.")
                return None

        end_idx = self.factor_returns.index.get_loc(target_dt)
        start_idx = max(0, end_idx - lookback_days + 1)
        window = self.factor_returns.iloc[start_idx:end_idx + 1]

        cumulative_returns = np.exp(np.log1p(window).sum()) - 1
        ranked_factors = cumulative_returns.sort_values(ascending=False)
        ranked_factors.name = f"CumulativeReturn_{lookback_days}d"
        
        self.logger.info("Factor ranking complete.")
        return ranked_factors

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Initialize the system with your API key
    import os
    API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
    if not API_KEY:
        raise ValueError("ALPHAVANTAGE_API_KEY environment variable is required")
    research_system = FactorResearchSystem(api_key=API_KEY)

    # 2. Define a universe of tickers from the IWM ETF
    etf_symbol = "IWM"
    universe_tickers = research_system.get_etf_holdings(etf_symbol)
    
    if not universe_tickers:
         print(f"Could not retrieve tickers for {etf_symbol}. Exiting.")
    else:
        # 3. Download data (will use cache if available and recent)
        research_system.download_universe_data(tickers=universe_tickers, skip_cached=True)

        # 4. Load price data from the cache
        # Using a more recent start date is better for IWM to keep more stocks.
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3*365) # 3-year lookback
        price_data = research_system.load_price_data(tickers=universe_tickers, start_date=start_date, end_date=end_date)
        
        if price_data.empty or price_data.shape[1] < 2:
            print("\nNot enough price data loaded to perform analysis. Check data source, API key, or date range. Exiting.")
        else:
            # 5. Prepare the data with the robust method
            standardized_returns = research_system.prepare_returns_data(price_data, nan_threshold=0.1)

            if standardized_returns is not None and not standardized_returns.empty:
                # 6. Discover the underlying factors
                research_system.discover_factors(n_factors=20)
                
                if research_system.pca:
                    research_system.plot_explained_variance()
                    print("\n--- Equity Assignments to Factors ---")
                    research_system.assign_equities_to_factors(n_top_equities=5)
                    research_system.calculate_factor_returns()
                    print("\n--- Factor Returns Head ---")
                    print(research_system.factor_returns.head())
                    
                    target_date_for_ranking = datetime.now() - timedelta(days=1)
                    print(f"\n--- Ranking factors for {target_date_for_ranking.strftime('%Y-%m-%d')} ---")
                    ranked_factors = research_system.rank_factors_by_date(target_date=target_date_for_ranking, lookback_days=20)
                    
                    if ranked_factors is not None:
                        print(ranked_factors)
            else:
                print("\nCould not prepare standardized returns. Factor analysis aborted.")
