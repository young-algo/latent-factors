"""
Point-in-Time (PIT) Universe QA Verification Tests
==================================================

This module contains the mandatory QA tests for the PIT Universe Generation system.
These tests verify that the system correctly eliminates survivorship bias by
including delisted stocks in historical universes.

CRITICAL TEST CASES (from Engineering Spec):
---------------------------------------------
1. Test Date: 2008-09-15 (Lehman Brothers bankruptcy)
   - Run: build_point_in_time_universe('2008-09-01')
   - Verify: LEH (Lehman Brothers) must be present in historical_universes table

2. Test Date: 2023-01-01
   - Verify: SIVB (Silicon Valley Bank) must be present in historical_universes table

If these tickers are missing, the system is still biased and NOT ready for live capital.

Usage
-----
Run these tests before deploying to production:

    pytest tests/test_pit_universe.py -v

Or run the standalone verification:

    python -m pytest tests/test_pit_universe::test_lehman_brothers_in_2008_universe -v
    python -m pytest tests/test_pit_universe::test_silicon_valley_bank_in_2023_universe -v

Environment Variables
---------------------
- ALPHAVANTAGE_API_KEY: Required for fetching PIT data
- PIT_TEST_BUILD_UNIVERSE: Set to 'true' to build universes during tests
  (otherwise tests assume universes are pre-built)
"""

import os
import sys
import pytest
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config
from src.database import get_db_connection, ensure_schema


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def api_key():
    """Get Alpha Vantage API key from environment."""
    key = config.ALPHAVANTAGE_API_KEY
    if not key:
        pytest.skip("ALPHAVANTAGE_API_KEY not configured")
    return key


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    """Create a temporary database for testing."""
    tmp_path = tmp_path_factory.mktemp("pit_test")
    db_file = tmp_path / "test_pit.db"
    ensure_schema(db_file)
    return str(db_file)


@pytest.fixture(scope="module")
def backend(api_key, db_path):
    """Create a DataBackend instance for testing."""
    from src.alphavantage_system import DataBackend
    return DataBackend(api_key, db_path=db_path, start_date="2000-01-01")


# =============================================================================
# CRITICAL QA TESTS (Must Pass Before Live Capital)
# =============================================================================

class TestPITUniverseQA:
    """
    Critical QA tests for PIT Universe Construction.
    
    These tests verify that the system correctly includes delisted stocks,
    eliminating survivorship bias from backtests.
    """
    
    def test_lehman_brothers_in_2008_universe(self, backend, db_path):
        """
        QA Test #1: Verify LEH (Lehman Brothers) is in 2008-09-15 universe.
        
        Context: Lehman Brothers filed for bankruptcy on September 15, 2008.
        A biased system (using current constituents) would NOT include LEH
        because it no longer exists today.
        
        Expected Result: LEH ticker must be present in the universe.
        """
        test_date = "2008-09-15"
        expected_ticker = "LEH"
        
        # Build or retrieve the PIT universe
        universe_df = backend.build_point_in_time_universe(
            date_str=test_date,
            top_n=1000,  # Large enough to include LEH
            exchanges=["NYSE"]  # LEH traded on NYSE
        )
        
        # Verify universe was created
        assert not universe_df.empty, f"Failed to build universe for {test_date}"
        
        # Get ticker list
        tickers = universe_df['ticker'].tolist()
        
        # CRITICAL ASSERTION: LEH must be present
        assert expected_ticker in tickers, (
            f"CRITICAL QA FAILURE: {expected_ticker} (Lehman Brothers) "
            f"not found in {test_date} universe. "
            f"This indicates survivorship bias - the system is not PIT-correct. "
            f"Universe size: {len(tickers)}"
        )
        
        # Additional verification: Check database storage
        with get_db_connection(db_path) as con:
            result = con.execute(
                "SELECT COUNT(*) FROM historical_universes WHERE date = ? AND ticker = ?",
                (test_date, expected_ticker)
            ).fetchone()
            
            assert result[0] == 1, (
                f"{expected_ticker} not found in historical_universes table for {test_date}"
            )
    
    def test_silicon_valley_bank_in_2023_universe(self, backend, db_path):
        """
        QA Test #2: Verify SIVB (Silicon Valley Bank) is in 2023-01-01 universe.
        
        Context: Silicon Valley Bank failed in March 2023. On January 1, 2023,
        SIVB was a thriving, investable company. A biased system would exclude it.
        
        Expected Result: SIVB ticker must be present in the universe.
        """
        test_date = "2023-01-01"
        expected_ticker = "SIVB"
        
        # Build or retrieve the PIT universe
        universe_df = backend.build_point_in_time_universe(
            date_str=test_date,
            top_n=1000,
            exchanges=["NASDAQ"]  # SIVB traded on NASDAQ
        )
        
        # Verify universe was created
        assert not universe_df.empty, f"Failed to build universe for {test_date}"
        
        # Get ticker list
        tickers = universe_df['ticker'].tolist()
        
        # CRITICAL ASSERTION: SIVB must be present
        assert expected_ticker in tickers, (
            f"CRITICAL QA FAILURE: {expected_ticker} (Silicon Valley Bank) "
            f"not found in {test_date} universe. "
            f"This indicates survivorship bias - the system is not PIT-correct. "
            f"Universe size: {len(tickers)}"
        )
        
        # Additional verification: Check database storage
        with get_db_connection(db_path) as con:
            result = con.execute(
                "SELECT COUNT(*) FROM historical_universes WHERE date = ? AND ticker = ?",
                (test_date, expected_ticker)
            ).fetchone()
            
            assert result[0] == 1, (
                f"{expected_ticker} not found in historical_universes table for {test_date}"
            )


# =============================================================================
# ADDITIONAL VERIFICATION TESTS
# =============================================================================

class TestPITUniverseFeatures:
    """Additional tests for PIT universe functionality."""
    
    def test_universe_stored_in_database(self, backend, db_path):
        """Verify that universes are correctly stored in the database."""
        test_date = "2022-06-01"
        
        # Build universe
        universe_df = backend.build_point_in_time_universe(
            date_str=test_date,
            top_n=100,
            skip_delisted=False
        )
        
        # Verify storage
        with get_db_connection(db_path) as con:
            count = con.execute(
                "SELECT COUNT(*) FROM historical_universes WHERE date = ?",
                (test_date,)
            ).fetchone()[0]
            
            assert count > 0, f"No entries found in historical_universes for {test_date}"
            assert count <= 100, f"Expected <= 100 entries, got {count}"
    
    def test_get_historical_universe_retrieval(self, backend, db_path):
        """Test retrieving a stored universe via get_historical_universe."""
        test_date = "2022-06-01"
        
        # First build the universe
        backend.build_point_in_time_universe(date_str=test_date, top_n=50)
        
        # Then retrieve it
        tickers = backend.get_historical_universe(date_str=test_date, top_n=50)
        
        assert len(tickers) > 0, "No tickers retrieved"
        assert len(tickers) <= 50, f"Expected <= 50 tickers, got {len(tickers)}"
    
    def test_database_schema_has_pit_table(self, db_path):
        """Verify the historical_universes table exists in the schema."""
        with get_db_connection(db_path) as con:
            tables = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = [t[0] for t in tables]
            
            assert 'historical_universes' in table_names, (
                "historical_universes table not found in database schema"
            )
    
    def test_database_schema_has_universe_index(self, db_path):
        """Verify the index on historical_universes table exists."""
        with get_db_connection(db_path) as con:
            indexes = con.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='historical_universes'"
            ).fetchall()
            index_names = [i[0] for i in indexes]
            
            assert 'idx_universe_date' in index_names, (
                "idx_universe_date index not found on historical_universes table"
            )


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestPITIntegrationWithResearch:
    """Tests for PIT integration with FactorResearchSystem."""
    
    def test_get_backtest_universe_method_exists(self, backend, api_key, db_path):
        """Verify FactorResearchSystem has the get_backtest_universe method."""
        from src.research import FactorResearchSystem
        
        frs = FactorResearchSystem(
            api_key=api_key,
            universe=["SPY"],
            db_path=db_path
        )
        
        # Check method exists
        assert hasattr(frs, 'get_backtest_universe'), (
            "FactorResearchSystem missing get_backtest_universe method"
        )
        
        # Check method is callable
        assert callable(getattr(frs, 'get_backtest_universe')), (
            "get_backtest_universe is not callable"
        )
    
    def test_verify_pit_universe_method_exists(self, backend, api_key, db_path):
        """Verify FactorResearchSystem has the verify_pit_universe method."""
        from src.research import FactorResearchSystem
        
        frs = FactorResearchSystem(
            api_key=api_key,
            universe=["SPY"],
            db_path=db_path
        )
        
        # Check method exists
        assert hasattr(frs, 'verify_pit_universe'), (
            "FactorResearchSystem missing verify_pit_universe method"
        )
        
        # Check method is callable
        assert callable(getattr(frs, 'verify_pit_universe')), (
            "verify_pit_universe is not callable"
        )


# =============================================================================
# STANDALONE VERIFICATION SCRIPT
# =============================================================================

def run_critical_qa_checks():
    """
    Run the critical QA checks as a standalone script.
    
    This can be run directly without pytest:
        python tests/test_pit_universe.py
    """
    print("=" * 70)
    print("POINT-IN-TIME UNIVERSE - CRITICAL QA VERIFICATION")
    print("=" * 70)
    print()
    
    # Check for API key
    api_key = config.ALPHAVANTAGE_API_KEY
    if not api_key:
        print("ERROR: ALPHAVANTAGE_API_KEY not configured")
        print("Set it in your .env file or environment variables")
        sys.exit(1)
    
    print(f"Database: {config.get_cache_path('av_cache.db')}")
    print()
    
    # Initialize backend
    from src.alphavantage_system import DataBackend
    backend = DataBackend(api_key)
    
    all_passed = True
    
    # Test 1: Lehman Brothers
    print("-" * 70)
    print("QA TEST #1: Lehman Brothers (LEH) on 2008-09-15")
    print("-" * 70)
    print("Context: Lehman Brothers filed for bankruptcy on this date.")
    print("Expected: LEH must be present in the universe.")
    print()
    
    try:
        universe_df = backend.build_point_in_time_universe(
            date_str="2008-09-15",
            top_n=1000,
            exchanges=["NYSE"]
        )
        tickers = universe_df['ticker'].tolist()
        
        if 'LEH' in tickers:
            print(" PASS: LEH found in universe")
            print(f"  Universe size: {len(tickers)} stocks")
            print(f"  LEH rank by dollar volume: {tickers.index('LEH') + 1}")
        else:
            print(" FAIL: LEH NOT FOUND in universe")
            print(f"  Universe size: {len(tickers)} stocks")
            print("  This indicates survivorship bias!")
            all_passed = False
    except Exception as e:
        print(f" ERROR: {e}")
        all_passed = False
    
    print()
    
    # Test 2: Silicon Valley Bank
    print("-" * 70)
    print("QA TEST #2: Silicon Valley Bank (SIVB) on 2023-01-01")
    print("-" * 70)
    print("Context: SIVB was a thriving bank before its March 2023 failure.")
    print("Expected: SIVB must be present in the universe.")
    print()
    
    try:
        universe_df = backend.build_point_in_time_universe(
            date_str="2023-01-01",
            top_n=1000,
            exchanges=["NASDAQ"]
        )
        tickers = universe_df['ticker'].tolist()
        
        if 'SIVB' in tickers:
            print(" PASS: SIVB found in universe")
            print(f"  Universe size: {len(tickers)} stocks")
            print(f"  SIVB rank by dollar volume: {tickers.index('SIVB') + 1}")
        else:
            print(" FAIL: SIVB NOT FOUND in universe")
            print(f"  Universe size: {len(tickers)} stocks")
            print("  This indicates survivorship bias!")
            all_passed = False
    except Exception as e:
        print(f" ERROR: {e}")
        all_passed = False
    
    print()
    print("=" * 70)
    if all_passed:
        print("ALL CRITICAL QA TESTS PASSED ")
        print("System is ready for backtesting without survivorship bias.")
    else:
        print("CRITICAL QA TESTS FAILED ")
        print("DO NOT deploy to live capital until these tests pass!")
        print()
        print("The system is still suffering from survivorship bias.")
        print("Ensure LISTING_STATUS endpoint is returning delisted stocks.")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_critical_qa_checks())
