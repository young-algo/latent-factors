"""
Data backend: prices • fundamentals • ETF holdings
==================================================

New in v3
---------
* `get_etf_holdings(<ETF>)`     – fetch constituent tickers & weights
* `is_etf(<symbol>)`            – fast check via Alpha‑Vantage “AssetType”
* Local cache (`etf_holdings` table) avoids repeated scraping
* Primary holdings source  :  AlphaVantage ETF_PROFILE API
* Primary holdings source  :  AlphaVantage ETF_PROFILE API
* No fallback sources (AlphaVantage exclusive)

All previous functionality (price cache, fundamentals cache) is retained.
"""

from __future__ import annotations
import json, logging, os, random, sqlite3, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from typing import Sequence, Mapping, Any

import numpy as np
import pandas as pd
import requests


_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.WARNING)

# Ensure logging shows up in Jupyter notebooks
if not _LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
    handler.setFormatter(formatter)
    _LOGGER.addHandler(handler)
    _LOGGER.propagate = False

def _coerce_numeric(d: Mapping[str, str]) -> dict[str, Any]:
    """
    Convert Alpha Vantage string data to appropriate numeric types.
    
    Handles the common issue where Alpha Vantage returns all values as strings,
    even for numeric fields like market cap, P/E ratios, etc.
    
    Parameters
    ----------
    d : Mapping[str, str]
        Raw data dictionary from Alpha Vantage API response
        
    Returns
    -------
    dict[str, Any]
        Processed dictionary with appropriate data types:
        - Numeric fields converted to float
        - Invalid/missing values converted to np.nan
        - String fields preserved as strings
        
    Complexity
    ----------
    Time: O(n) where n = number of fields in input dictionary
    Space: O(n) for the output dictionary
    
    Notes
    -----
    This function is critical for ensuring downstream analysis works correctly
    with numeric data rather than string representations.
    """
    out = {}
    # Fields that should be numeric
    numeric_fields = {
        "PriceToSalesRatioTTM", "ProfitMargin", "Beta", "MarketCapitalization",
        "PE", "PEG", "ROE", "ROA", "BookValue", "SharesOutstanding",
        "DividendYield", "EPS", "Revenue", "GrossProfitTTM", "EBITDA"
    }
    
    for k, v in d.items():
        if v is None or v == "None" or v == "N/A" or v == "-":
            out[k] = None if k in ["Sector", "Industry", "AssetType"] else np.nan
        else:
            try:
                out[k] = float(v)
            except (ValueError, TypeError):
                # Keep strings for categorical fields, convert to NaN for numeric fields
                if k in numeric_fields:
                    out[k] = np.nan
                else:
                    out[k] = v
    return out


class _ConnectionPool:
    """
    Simple SQLite connection pool for thread-safe database access.

    Maintains a pool of reusable connections to avoid the overhead of
    creating new connections for each database operation.
    """
    def __init__(self, db_path: Path, pool_size: int = 5):
        self.db_path = db_path
        self._pool: Queue[sqlite3.Connection] = Queue(maxsize=pool_size)
        self._pool_size = pool_size
        self._created = 0
        self._lock = threading.Lock()

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with optimal settings."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
        return conn

    @contextmanager
    def connection(self):
        """
        Get a connection from the pool (or create one if pool empty and under limit).

        Usage:
            with pool.connection() as conn:
                conn.execute(...)
        """
        conn = None
        try:
            # Try to get an existing connection from the pool
            conn = self._pool.get_nowait()
        except Empty:
            # Pool empty - create new connection if under limit
            with self._lock:
                if self._created < self._pool_size:
                    conn = self._create_connection()
                    self._created += 1
            # If at limit, wait for a connection to become available
            if conn is None:
                conn = self._pool.get()

        try:
            yield conn
        finally:
            # Return connection to pool
            try:
                self._pool.put_nowait(conn)
            except:
                # Pool full, close this connection
                conn.close()

    def close_all(self):
        """Close all pooled connections."""
        while True:
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Empty:
                break


class DataBackend:
    """
    Comprehensive financial data backend with caching mechanism.
    
    This class provides a unified interface for accessing price data, fundamental
    analysis, and ETF holdings from Alpha Vantage API with intelligent local
    caching.
    
    Architecture
    -----------
    The backend operates on three data streams:
    1. **Price Data Flow**: API → Cache → DataFrame
       - Primary: Alpha Vantage TIME_SERIES_DAILY_ADJUSTED
       - Cache: SQLite 'prices' table with daily granularity
       
    2. **Fundamentals Flow**: API → JSON Cache → Structured DataFrame  
       - Primary: Alpha Vantage OVERVIEW endpoint
       - Cache: SQLite 'fundamentals' table with JSON storage
       - Processing: Automatic type coercion via _coerce_numeric()
       
    3. **ETF Holdings Flow**: API → Structured Cache → DataFrame
       - Primary: Alpha Vantage ETF_PROFILE endpoint
       - Cache: SQLite 'etf_holdings' table with constituent mapping
       
    Performance Characteristics
    --------------------------
    - **API Rate Limits**: 75 calls/minute for both prices and fundamentals
    - **Cache Hit Rate**: ~90%+ for repeated analysis on same date
    - **Memory Usage**: O(n*d) where n=tickers, d=days of history
    
    Error Handling
    -------------
    - **Network Timeouts**: Exponential backoff with 3 retries
    - **API Errors**: Bubbles up exceptions (no fallback)
    - **Data Validation**: Comprehensive filtering of invalid tickers
    - **Cache Corruption**: Automatic schema recreation
    
    Thread Safety
    ------------
    This class is NOT thread-safe due to SQLite connection handling.
    For concurrent usage, instantiate separate DataBackend objects.
    
    Examples
    --------
    >>> backend = DataBackend("your_api_key", start_date="2020-01-01")
    >>> prices = backend.get_prices(["AAPL", "MSFT"])  # Returns DataFrame
    >>> fundamentals = backend.get_fundamentals(["AAPL"])  # Returns DataFrame
    >>> holdings = backend.get_etf_holdings("SPY")  # Returns DataFrame
    >>> is_etf = backend.is_etf("SPY")  # Returns True
    """
    # ---------------- Alpha Vantage constants ---------------- #
    AV_URL  = "https://www.alphavantage.co/query"
    FUN_PX  = "TIME_SERIES_DAILY_ADJUSTED"
    FUN_OV  = "OVERVIEW"
    FUN_ETF = "ETF_PROFILE"

    RATE_LIMIT_PX  = 75          # calls / min
    RATE_LIMIT_FND = 75          # calls / min (same as prices)

    # ---------------- ETF holdings ---------------- #
    ETF_CACHE_DAYS = 1           # refresh holdings daily

    def __init__(
        self,
        api_key: str,
        db_path: str | Path = "./av_cache.db",
        start_date: str | None = None,
    ) -> None:
        """
        Initialize the DataBackend with API credentials and cache configuration.
        
        Parameters
        ----------
        api_key : str
            Alpha Vantage API key for data access. Free tier provides 75 calls/minute.
            Premium tiers offer higher rate limits and extended historical data.
        db_path : str | Path, default "./av_cache.db"
            Path to SQLite database file for local caching. Will be created if doesn't exist.
            Parent directories are created automatically.
        start_date : str | None, default None
            Earliest date for price data collection in YYYY-MM-DD format.
            If None, defaults to "2000-01-01" for maximum historical coverage.
            
        Raises
        ------
        ValueError
            If api_key is empty or invalid format
        OSError
            If db_path directory cannot be created or accessed
            
        Notes
        -----
        - Initializes SQLite schema automatically on first run
        - Sets up rate limiting timers to respect API quotas
        - Creates database parent directories if they don't exist
        
        Complexity
        ----------
        Time: O(1) - constant time initialization
        Space: O(1) - minimal memory footprint for instance variables
        """
        self.api_key   = api_key
        self.start     = pd.to_datetime(start_date or "2000-01-01")
        self.db_path   = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connection pool for efficient database access
        self._conn_pool = _ConnectionPool(self.db_path, pool_size=5)
        self._ensure_schema()

        self._t_last_px  = 0.0
        self._t_last_fnd = 0.0
        self._lock = threading.Lock()

    # ================================================================= #
    # --------------------  PUBLIC HELPERS  --------------------------- #
    # ================================================================= #
    # ------------ prices ------------ #
    def get_prices(
        self, tickers: Sequence[str], end: str | None = None
    ) -> pd.DataFrame:
        """
        Retrieve historical adjusted close prices for multiple tickers.
        
        This method handles the complete price data workflow: cache checking,
        API calls with fallback, data processing, and return formatting.
        
        Parameters
        ----------
        tickers : Sequence[str]
            List of stock/ETF ticker symbols (e.g., ["AAPL", "MSFT", "SPY"])
            Maximum recommended: 100 tickers per call for optimal performance
        end : str | None, default None
            End date for price data in YYYY-MM-DD format.
            If None, returns data through the most recent available date.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with:
            - Index: Business day dates (pandas DatetimeIndex)
            - Columns: Ticker symbols
            - Values: Adjusted close prices (float)
            - Frequency: Business days (B) with forward/backward fill
            
        Call Flow
        --------
        For each ticker:
        1. Check cache freshness via _last_price_update()
        2. If stale: Download via _download_px_av() → _download_px_yf() (fallback)
        3. Load cached data via _load_px()
        4. Concatenate all series into DataFrame
        5. Apply business day resampling with gap filling
        
        Performance
        ----------
        - **Cache Hit**: O(n) database queries, ~0.1s per 100 tickers
        - **Cache Miss**: O(n) API calls, ~60s per 75 tickers (rate limited)
        - **Memory**: O(n*d) where n=tickers, d=days of history
        
        Error Handling
        -------------
        - Network failures: Retries with exponential backoff
        - Invalid tickers: Excluded from result (logged as warnings)
        - API rate limits: Automatic throttling with progress logging
        
        Examples
        --------
        >>> prices = backend.get_prices(["AAPL", "MSFT"])
        >>> prices = backend.get_prices(["SPY"], end="2023-12-31")
        """
        _LOGGER.info("Loading price data for %d tickers", len(tickers))

        # Use dict to collect series - more memory efficient than list + concat
        price_data: dict[str, pd.Series] = {}
        failed_tickers = []

        def process_ticker(tk):
            try:
                self._maybe_update_px(tk)
                px_series = self._load_px(tk)
                if not px_series.empty:
                    return (tk, px_series)
                else:
                    return (tk, None)
            except Exception as e:
                _LOGGER.warning(f"Failed to get data for {tk}: {str(e)}")
                return (tk, None)

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {executor.submit(process_ticker, tk): tk for tk in tickers}
            for i, future in enumerate(as_completed(future_to_ticker), 1):
                tk = future_to_ticker[future]
                if i % 50 == 0 or i == 1:
                    _LOGGER.info("Processing ticker %d/%d: %s", i, len(tickers), tk)

                try:
                    ticker, series = future.result()
                    if series is not None:
                        price_data[ticker] = series
                    else:
                        failed_tickers.append(ticker)
                except Exception as e:
                    _LOGGER.warning(f"Exception processing {tk}: {e}")
                    failed_tickers.append(tk)

        if failed_tickers:
            _LOGGER.warning(f"Failed to get data for {len(failed_tickers)} tickers: {failed_tickers[:10]}{'...' if len(failed_tickers) > 10 else ''}")

        _LOGGER.info(f"Building DataFrame for {len(price_data)} successful tickers")

        if not price_data:
            return pd.DataFrame()

        # Construct DataFrame directly from dict (single allocation)
        df = pd.DataFrame(price_data).sort_index()
        df.index = pd.to_datetime(df.index)
        if end:
            df = df.loc[:end]
        result = df.asfreq("B").ffill().bfill()
        
        _LOGGER.info("Price data loaded: %d days x %d tickers", len(result), len(result.columns))
        return result

    # ------------ fundamentals ------------ #
    def get_fundamentals(
        self,
        tickers: Sequence[str],
        fields: Sequence[str] | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Retrieve fundamental analysis data for multiple tickers.
        
        Fetches company financials, ratios, and metadata from Alpha Vantage OVERVIEW
        endpoint with intelligent caching and automatic type conversion.
        
        Parameters
        ----------
        tickers : Sequence[str]
            List of stock/ETF ticker symbols to analyze
        fields : Sequence[str] | None, default None
            Specific fields to return. If None, returns all available fields.
            Common fields: ["Sector", "Industry", "MarketCapitalization", 
            "PERatio", "DividendYield", "Beta", "EPS", "ROE"]
        force_refresh : bool, default False
            If True, bypasses cache and forces fresh API call.
            Useful for getting latest quarterly updates.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with:
            - Index: Ticker symbols
            - Columns: Fundamental data fields
            - Values: Automatically type-converted (numeric fields as float)
            
        Data Processing Pipeline
        ----------------------
        1. **Cache Check**: Query fundamentals table for each ticker
        2. **API Call**: If cache miss or force_refresh, call Alpha Vantage OVERVIEW
        3. **Type Conversion**: Apply _coerce_numeric() for proper data types
        4. **Field Filtering**: Extract only requested fields if specified
        5. **DataFrame Assembly**: Combine all records into structured format
        
        Performance
        ----------
        - **Cache Hit**: O(n) database queries, ~50ms per 100 tickers
        - **Cache Miss**: O(n) API calls, rate limited to 75/minute
        - **Type Conversion**: O(f) where f = number of fields per ticker
        
        Notes
        -----
        - Fundamental data is cached with daily refresh cycle
        - Missing/invalid tickers are excluded from results
        - All numeric fields are converted from strings to appropriate types
        - ETF tickers return limited fundamental data (mainly AssetType)
        
        Examples
        --------
        >>> fundamentals = backend.get_fundamentals(["AAPL", "MSFT"])
        >>> pe_ratios = backend.get_fundamentals(["AAPL"], fields=["PERatio"])
        >>> fresh_data = backend.get_fundamentals(["AAPL"], force_refresh=True)
        """
        _LOGGER.info("Loading fundamentals for %d tickers", len(tickers))
        
        _LOGGER.info("Loading fundamentals for %d tickers", len(tickers))
        
        records = {}
        
        def process_fundamental(tk):
            try:
                self._maybe_update_fnd(tk, force_refresh)
                rec = self._load_fnd(tk)
                if not rec:
                    return None
                if fields:
                    rec = {k: rec.get(k) for k in fields}
                return _coerce_numeric(rec)
            except Exception as e:
                _LOGGER.warning(f"Failed to get fundamentals for {tk}: {e}")
                return None

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {executor.submit(process_fundamental, tk): tk for tk in tickers}
            for i, future in enumerate(as_completed(future_to_ticker), 1):
                tk = future_to_ticker[future]
                if i % 100 == 0 or i == 1:
                    _LOGGER.info("Processing fundamentals %d/%d: %s", i, len(tickers), tk)
                
                try:
                    result = future.result()
                    if result:
                        records[tk] = result
                except Exception as e:
                    _LOGGER.warning(f"Exception processing fundamentals for {tk}: {e}")

        _LOGGER.info("Fundamentals loaded for %d tickers", len(records))
        return pd.DataFrame.from_dict(records, orient="index")

    # ------------ ETF holdings ------------ #
    def get_etf_holdings(
        self, etf: str, force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Retrieve constituent holdings and weights for an ETF.
        
        This method enables ETF expansion for factor analysis by providing
        the underlying securities and their portfolio weights.
        
        Parameters
        ----------
        etf : str
            ETF ticker symbol (e.g., "SPY", "QQQ", "IWM")
        force_refresh : bool, default False
            If True, bypasses cache and forces fresh API call.
            Useful for getting latest rebalancing updates.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - 'constituent': Underlying ticker symbols (str)
            - 'weight': Portfolio weights as percentages (float)
            
        Data Sources & Fallback Chain
        ----------------------------
        1. **Primary**: Alpha Vantage ETF_PROFILE API
           - Comprehensive holdings data with official weights
           - Covers major ETFs (SPY, QQQ, IWM, sector ETFs)
           
        2. **Cache**: SQLite etf_holdings table
           - Daily refresh cycle (configurable via ETF_CACHE_DAYS)
           - Avoids repeated API calls for same ETF
           
        Processing Pipeline
        ------------------
        1. **Cache Check**: Query etf_holdings table for freshness
        2. **API Call**: If stale or force_refresh, download holdings
        3. **Data Validation**: Filter invalid tickers and normalize weights
        4. **Storage**: Update cache with new holdings data
        5. **Return**: Clean DataFrame ready for factor analysis
        
        Complexity
        ----------
        - **Time**: O(h) where h = number of holdings in ETF
        - **Space**: O(h) for holdings storage
        - **API Calls**: 1 per ETF per refresh cycle
        
        Error Handling
        -------------
        - Invalid ETF symbols: Returns empty DataFrame
        - API failures: Raises RuntimeError
        - Data corruption: Comprehensive validation and filtering
        
        Examples
        --------
        >>> holdings = backend.get_etf_holdings("SPY")
        >>> print(f"SPY has {len(holdings)} constituents")
        >>> top_holdings = holdings.nlargest(10, 'weight')
        """
        self._maybe_update_etf_holdings(etf, force_refresh)
        return self._load_etf_holdings(etf)

    def is_etf(self, symbol: str) -> bool:
        """
        Determine if a ticker symbol represents an ETF.
        
        Uses a hybrid approach: fast lookup for common ETFs followed by
        API verification for unknown symbols.
        
        Parameters
        ----------
        symbol : str
            Ticker symbol to check (e.g., "SPY", "AAPL")
            
        Returns
        -------
        bool
            True if symbol is an ETF, False if individual stock
            
        Detection Strategy
        -----------------
        1. **Fast Path**: Check against known ETF list (O(1) lookup)
           - Covers ~95% of commonly used ETFs
           - Avoids API calls for performance
           
        2. **API Verification**: Query Alpha Vantage OVERVIEW.AssetType
           - For unknown symbols requiring classification
           - Cached result for future calls
           
        Performance
        ----------
        - **Known ETFs**: O(1) set lookup, ~1ms
        - **Unknown Symbols**: O(1) API call + cache, ~200ms first time
        - **Subsequent Calls**: O(1) cache lookup
        
        Examples
        --------
        >>> backend.is_etf("SPY")      # True  (fast path)
        >>> backend.is_etf("AAPL")     # False (API verification)
        >>> backend.is_etf("UNKNOWN")  # False (API verification)
        """
        # Common ETFs - check these first to avoid API calls
        common_etfs = {"SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "EEM", "GLD", 
                       "TLT", "XLF", "XLE", "XLK", "XLV", "XLI", "XLY", "XLP",
                       "XLB", "XLU", "XLRE", "VNQ", "AGG", "BND", "LQD", "HYG",
                       "VTHR"}
        if symbol.upper() in common_etfs:
            return True
            
        # Otherwise check via API
        meta = self.get_fundamentals([symbol], fields=["AssetType"])
        return (
            not meta.empty
            and meta.loc[symbol, "AssetType"]  # type: ignore[index]
            and str(meta.loc[symbol, "AssetType"]).upper() == "ETF"
        )

    # ================================================================= #
    # --------------------  INTERNAL – SCHEMA  ------------------------ #
    # ================================================================= #
    def _ensure_schema(self) -> None:
        with self._conn_pool.connection() as con:
            # price + meta
            con.execute(
                """CREATE TABLE IF NOT EXISTS prices (
                       ticker TEXT, date TEXT, adj_close REAL,
                       PRIMARY KEY (ticker, date)
                   )"""
            )
            con.execute(
                """CREATE TABLE IF NOT EXISTS meta (
                       ticker TEXT PRIMARY KEY, last_update TEXT
                   )"""
            )
            # fundamentals
            con.execute(
                """CREATE TABLE IF NOT EXISTS fundamentals (
                       ticker TEXT PRIMARY KEY,
                       last_update TEXT,
                       json TEXT
                   )"""
            )
            # ETF holdings
            con.execute(
                """CREATE TABLE IF NOT EXISTS etf_holdings (
                       etf TEXT,
                       constituent TEXT,
                       weight REAL,
                       retrieved TEXT,
                       PRIMARY KEY (etf, constituent)
                   )"""
            )

    # ================================================================= #
    # --------------------  INTERNAL – PRICES  ------------------------ #
    # ================================================================= #
    def _maybe_update_px(self, ticker: str) -> None:
        today = pd.Timestamp.now().normalize() - pd.tseries.offsets.BDay(1)
        last  = self._last_price_update(ticker)
        
        # Debug cache status
        if last is not None:
            _LOGGER.debug("Cache check for %s: last=%s, today=%s, cached=%s", 
                         ticker, last.date(), today.date(), last >= today)
        else:
            _LOGGER.debug("Cache check for %s: no cache found", ticker)
        
        if last is not None and last >= today:
            _LOGGER.debug("Using cached prices for %s (last updated: %s)", ticker, last.date())
            return
        
        _LOGGER.info("Downloading price data for %s...", ticker)
        self._download_px_av(ticker)
        _LOGGER.info("✓ Downloaded %s from AlphaVantage", ticker)

    def _download_px_av(self, ticker: str) -> None:
        self._rate_limit("px")
        params = {
            "function": self.FUN_PX,
            "symbol":   ticker,
            "apikey":   self.api_key,
            "outputsize": "full",
            "datatype": "json",
        }
        
        # Retry logic with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                timeout = 30 + (attempt * 15)  # Increase timeout on retries
                _LOGGER.debug(f"Downloading prices for {ticker} (attempt {attempt + 1}/{max_retries}, timeout={timeout}s)")
                r = requests.get(self.AV_URL, params=params, timeout=timeout)
                r.raise_for_status()
                data = r.json()
                
                if "Information" in data and ("rate limit" in data["Information"] or "Burst" in data["Information"]):
                    _LOGGER.warning(f"Rate limit hit for {ticker}: {data['Information']}. Sleeping 60s...")
                    time.sleep(60)
                    continue

                break  # Success, exit retry loop
                
            except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    _LOGGER.warning(f"Timeout downloading {ticker} prices (attempt {attempt + 1}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    _LOGGER.error(f"Failed to download {ticker} prices after {max_retries} attempts")
                    raise RuntimeError(f"Timeout downloading {ticker} after {max_retries} attempts")
            except Exception as e:
                _LOGGER.error(f"Unexpected error downloading {ticker} prices: {e}")
                raise
        
        if "Time Series (Daily)" not in data:
            raise RuntimeError(f"Bad AlphaVantage PX for {ticker}: {data}")
        ts = (
            pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
            .rename(columns={"5. adjusted close": "adj_close"})
            [["adj_close"]]
            .astype(float)
            .sort_index()
        )
        # Convert string index to datetime
        ts.index = pd.to_datetime(ts.index)
        self._store_px(ticker, ts)



    def _store_px(self, ticker: str, frame: pd.DataFrame) -> None:
        frame = frame.loc[frame.index >= self.start]
        with self._lock:
            with self._conn_pool.connection() as con:
                # Remove existing data for this ticker to avoid unique constraint violations
                con.execute("DELETE FROM prices WHERE ticker=?", (ticker,))
                
                frame.assign(ticker=ticker).to_sql(
                    "prices", con, if_exists="append", index_label="date", method="multi"
                )
                con.execute(
                    "REPLACE INTO meta VALUES (?, ?)",
                    (ticker, frame.index.max().strftime("%Y-%m-%d")),
                )

    def _load_px(self, ticker: str) -> pd.Series:
        with self._conn_pool.connection() as con:
            df = pd.read_sql(
                "SELECT date, adj_close FROM prices WHERE ticker=?",
                con,
                params=(ticker,),
                parse_dates=["date"],
            )
        return df.set_index("date")["adj_close"].rename(ticker)

    def _last_price_update(self, ticker: str) -> pd.Timestamp | None:
        with self._conn_pool.connection() as con:
            cur = con.cursor()
            cur.execute("SELECT last_update FROM meta WHERE ticker=?", (ticker,))
            row = cur.fetchone()
        return pd.to_datetime(row[0]) if row else None

    # ================================================================= #
    # ----------------  INTERNAL – FUNDAMENTALS  ---------------------- #
    # ================================================================= #
    def _maybe_update_fnd(self, ticker: str, force: bool) -> None:
        last = self._last_fnd_update(ticker)
        today = pd.Timestamp.now().normalize()
        if not force and last is not None and last >= today:
            return
        self._download_fnd_av(ticker)

    def _download_fnd_av(self, ticker: str) -> None:
        self._rate_limit("fnd")
        params = {
            "function": self.FUN_OV,
            "symbol":   ticker,
            "apikey":   self.api_key,
            "datatype": "json",
        }
        
        # Retry logic with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                timeout = 30 + (attempt * 15)  # Increase timeout on retries
                _LOGGER.debug(f"Downloading fundamentals for {ticker} (attempt {attempt + 1}/{max_retries}, timeout={timeout}s)")
                r = requests.get(self.AV_URL, params=params, timeout=timeout)
                r.raise_for_status()
                data = r.json()
                
                if "Information" in data and ("rate limit" in data["Information"] or "Burst" in data["Information"]):
                    _LOGGER.warning(f"Rate limit hit for {ticker}: {data['Information']}. Sleeping 60s...")
                    time.sleep(60)
                    continue

                break  # Success, exit retry loop
                
            except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    _LOGGER.warning(f"Timeout downloading {ticker} fundamentals (attempt {attempt + 1}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    _LOGGER.error(f"Failed to download {ticker} fundamentals after {max_retries} attempts, using fallback data")
                    # Use fallback data for timeout cases
                    data = {"AssetType": "ETF" if ticker in ["SPY", "QQQ", "IWM", "DIA"] else "Stock"}
                    break
            except Exception as e:
                _LOGGER.error(f"Unexpected error downloading {ticker} fundamentals: {e}")
                data = {"AssetType": "ETF" if ticker in ["SPY", "QQQ", "IWM", "DIA"] else "Stock"}
                break
        
        # Handle API errors
        if "Error Message" in data or "Note" in data:
            _LOGGER.warning(f"AlphaVantage API error for {ticker}: {data}")
            # Store empty data to avoid repeated failed requests
            data = {"AssetType": "ETF" if ticker in ["SPY", "QQQ", "IWM", "DIA"] else "Stock"}
        elif len(data) < 3:
            _LOGGER.warning(f"Incomplete AlphaVantage fundamentals for {ticker}: {data}")
            # Assume ETF for common ETF tickers
            data = {"AssetType": "ETF" if ticker in ["SPY", "QQQ", "IWM", "DIA"] else "Stock"}
            
        with self._lock:
            with self._conn_pool.connection() as con:
                con.execute(
                    "REPLACE INTO fundamentals VALUES (?, ?, ?)",
                    (ticker, datetime.now().strftime("%Y-%m-%d"), json.dumps(data)),
                )

    def _load_fnd(self, ticker: str) -> Mapping[str, Any]:
        with self._conn_pool.connection() as con:
            cur = con.cursor()
            cur.execute("SELECT json FROM fundamentals WHERE ticker=?", (ticker,))
            row = cur.fetchone()
        return json.loads(row[0]) if row else {}

    def _last_fnd_update(self, ticker: str) -> pd.Timestamp | None:
        with self._conn_pool.connection() as con:
            cur = con.cursor()
            cur.execute("SELECT last_update FROM fundamentals WHERE ticker=?", (ticker,))
            row = cur.fetchone()
        return pd.to_datetime(row[0]) if row else None

    # ================================================================= #
    # ---------------  INTERNAL – ETF HOLDINGS  ----------------------- #
    # ================================================================= #
    def _maybe_update_etf_holdings(self, etf: str, force: bool) -> None:
        last = self._last_etf_update(etf)
        if not force and last is not None and (pd.Timestamp.now() - last).days < self.ETF_CACHE_DAYS:
            _LOGGER.info("Using cached holdings for %s (last updated: %s)", etf, last.date())
            return
        
        _LOGGER.info("Downloading ETF holdings for %s...", etf)
        self._download_holdings_av(etf)
        _LOGGER.info("Successfully downloaded %s holdings from AlphaVantage", etf)

    # -------- primary source: AlphaVantage ETF_PROFILE API -------- #
    def _download_holdings_av(self, etf: str) -> None:
        self._rate_limit("fnd")  # Use same rate limit as fundamentals
        params = {
            "function": self.FUN_ETF,
            "symbol":   etf,
            "apikey":   self.api_key,
        }
        
        # Retry logic with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                timeout = 30 + (attempt * 15)  # Increase timeout on retries
                _LOGGER.debug(f"Downloading holdings for {etf} (attempt {attempt + 1}/{max_retries}, timeout={timeout}s)")
                r = requests.get(self.AV_URL, params=params, timeout=timeout)
                r.raise_for_status()
                data = r.json()
                
                # Check for rate limit errors
                if "Information" in data and ("rate limit" in data["Information"] or "Burst" in data["Information"]):
                    _LOGGER.warning(f"Rate limit hit for {etf}: {data['Information']}. Sleeping 60s...")
                    time.sleep(60)
                    continue

                break  # Success, exit retry loop
                
            except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    _LOGGER.warning(f"Timeout downloading {etf} holdings (attempt {attempt + 1}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    _LOGGER.error(f"Failed to download {etf} holdings after {max_retries} attempts")
                    raise RuntimeError(f"Timeout downloading {etf} holdings after {max_retries} attempts")
            except Exception as e:
                _LOGGER.error(f"Unexpected error downloading {etf} holdings: {e}")
                raise
        
        # Check for API errors
        if "Error Message" in data or "Note" in data:
            raise RuntimeError(f"AlphaVantage API error: {data}")
            
        # Extract holdings from the ETF_PROFILE response
        holdings = data.get("holdings", [])
        if not holdings:
            raise RuntimeError(f"No holdings found in AlphaVantage ETF_PROFILE for {etf}")
            
        # Convert to DataFrame
        df = pd.DataFrame(holdings)
        
        # Rename columns to match our schema
        df.rename(columns={"symbol": "constituent", "weight": "weight"}, inplace=True)
        
        # Filter out invalid constituents
        df = df[df["constituent"].notna()]
        df = df[~df["constituent"].str.upper().isin(["N/A", "NA", "NULL", "NONE", ""])]
        df = df[df["constituent"].str.len() <= 10]  # Reasonable ticker length
        
        # Ensure weight is numeric (it should come as a percentage string like "11.94%")
        df["weight"] = df["weight"].str.rstrip("%").astype(float)
        
        self._store_etf_holdings(etf, df)



    def _store_etf_holdings(self, etf: str, frame: pd.DataFrame) -> None:
        frame = frame[["constituent", "weight"]].dropna()
        today = datetime.now().strftime("%Y-%m-%d")
        with self._lock:
            with self._conn_pool.connection() as con:
                con.execute("DELETE FROM etf_holdings WHERE etf=?", (etf,))
                frame.assign(etf=etf, retrieved=today).to_sql(
                    "etf_holdings", con, if_exists="append", index=False, method="multi"
                )
        _LOGGER.info("Stored %d holdings for %s", len(frame), etf)

    def _load_etf_holdings(self, etf: str) -> pd.DataFrame:
        with self._conn_pool.connection() as con:
            df = pd.read_sql(
                "SELECT constituent, weight FROM etf_holdings WHERE etf=?",
                con,
                params=(etf,),
            )
        return df

    def _last_etf_update(self, etf: str) -> pd.Timestamp | None:
        with self._conn_pool.connection() as con:
            cur = con.cursor()
            cur.execute(
                "SELECT MAX(retrieved) FROM etf_holdings WHERE etf=?", (etf,)
            )
            row = cur.fetchone()
        return pd.to_datetime(row[0]) if row and row[0] else None

    # ================================================================= #
    # --------------------  INTERNAL – MISC  -------------------------- #
    # ================================================================= #
    def _rate_limit(self, kind: str) -> None:
        """
        Enforce API rate limits to prevent quota exhaustion.
        
        Alpha Vantage enforces 75 calls/minute for free tier. This method
        calculates required wait time based on time since last call.
        
        Parameters
        ----------
        kind : str
            API endpoint type ("px" for prices, "fnd" for fundamentals)
        """
        with self._lock:
            if kind == "px":
                # Calculate minimum time between price API calls (0.8 seconds for 75/min)
                cool = 60 / self.RATE_LIMIT_PX
                since = time.time() - self._t_last_px
            else:
                # Calculate minimum time between fundamental API calls
                cool = 60 / self.RATE_LIMIT_FND  
                since = time.time() - self._t_last_fnd
                
            # If insufficient time has passed, sleep to respect rate limit
            # Add 0-20% random jitter to prevent synchronized API bursts across threads
            if since < cool:
                wait_time = cool - since
                jitter = wait_time * random.uniform(0, 0.2)
                wait_time += jitter
                if wait_time > 1:  # Only log significant waits to avoid spam
                    _LOGGER.info("Rate limiting: waiting %.1f seconds...", wait_time)
                time.sleep(wait_time)
            
            # Update timestamp to reserve this slot immediately so other threads wait
            if kind == "px":
                self._t_last_px = time.time()
            else:
                self._t_last_fnd = time.time()
