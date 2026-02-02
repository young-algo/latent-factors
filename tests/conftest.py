"""
Pytest configuration and fixtures for Equity Factors Research System tests.

This module provides shared fixtures and configuration for all tests.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Ensure src is importable
import sys
from pathlib import Path

# Add project root to path if running tests directly
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_returns_data():
    """Create sample returns data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
    
    returns = pd.DataFrame({
        'F1_Trend': np.cumsum(np.random.randn(252) * 0.01),
        'F2_MeanRev': np.sin(np.linspace(0, 4*np.pi, 252)) * 0.02 + np.random.randn(252) * 0.005,
        'F3_Volatile': np.random.randn(252) * 0.03
    }, index=dates)
    
    return returns


@pytest.fixture
def sample_factor_loadings():
    """Create sample factor loadings for testing."""
    np.random.seed(42)
    
    tickers = [f'STK{i:03d}' for i in range(1, 51)]
    
    loadings = pd.DataFrame({
        'Value': np.random.randn(50),
        'Momentum': np.random.randn(50),
        'Quality': np.random.randn(50),
        'Size': np.random.randn(50),
        'Volatility': np.random.randn(50)
    }, index=tickers)
    
    return loadings


@pytest.fixture
def mock_config(monkeypatch):
    """Mock configuration for tests that don't need real API keys."""
    from src.config import Configuration
    
    mock_config = Configuration()
    mock_config.ALPHAVANTAGE_API_KEY = "test_key"
    mock_config.OPENAI_API_KEY = "test_openai_key"
    
    monkeypatch.setattr('src.config.config', mock_config)
    return mock_config
