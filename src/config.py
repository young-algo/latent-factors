"""
Centralized Configuration Management for Equity Factors Research System
======================================================================

This module provides centralized configuration management, ensuring environment
variables are loaded once and validated consistently across the codebase.

Usage
-----
>>> from src.config import config
>>> api_key = config.ALPHAVANTAGE_API_KEY
>>> config.validate()  # Raises if required keys are missing

Environment Variables
---------------------
- ALPHAVANTAGE_API_KEY: Required for market data access
- OPENAI_API_KEY: Optional, for LLM-powered factor naming
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Load environment variables from .env file before any other imports
# Find .env file relative to this file's location
def _find_and_load_env():
    """Find and load .env file from project root."""
    # Start from this file's directory and go up
    current = Path(__file__).parent
    for _ in range(5):  # Search up to 5 levels up
        env_file = current / '.env'
        if env_file.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_file)
                return env_file
            except ImportError:
                pass  # dotenv not installed
        current = current.parent
    
    # Try loading from current working directory as fallback
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    return None


_env_file = _find_and_load_env()


class Configuration:
    """
    Centralized configuration for the Equity Factors Research System.
    
    All environment variables and configuration settings are accessed through
    this class to ensure consistency and proper validation.
    
    Attributes
    ----------
    ALPHAVANTAGE_API_KEY : str | None
        Alpha Vantage API key for market data
    OPENAI_API_KEY : str | None
        OpenAI API key for LLM factor naming
    CACHE_DIR : Path
        Directory for caching data
    LOG_LEVEL : str
        Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    
    def __init__(self):
        self.ALPHAVANTAGE_API_KEY: Optional[str] = os.getenv('ALPHAVANTAGE_API_KEY')
        self.OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')
        self.CACHE_DIR: Path = Path(os.getenv('EQUITY_FACTORS_CACHE', 'av_cache.db')).parent
        self.LOG_LEVEL: str = os.getenv('EQUITY_FACTORS_LOG_LEVEL', 'INFO')
        
        # Default date ranges
        self.DEFAULT_START_DATE: str = os.getenv('EQUITY_FACTORS_START_DATE', '2020-04-01')
        
        # API rate limits
        self.ALPHAVANTAGE_RATE_LIMIT: int = int(os.getenv('ALPHAVANTAGE_RATE_LIMIT', '75'))
        self.ALPHAVANTAGE_RATE_WINDOW: int = int(os.getenv('ALPHAVANTAGE_RATE_WINDOW', '60'))
        
        # Factor discovery defaults
        self.DEFAULT_FACTOR_METHOD: str = os.getenv('DEFAULT_FACTOR_METHOD', 'PCA')
        self.DEFAULT_N_COMPONENTS: int = int(os.getenv('DEFAULT_N_COMPONENTS', '10'))
        
        # Point-in-Time (PIT) Universe Configuration
        # These settings control the "Time Machine" universe construction
        # that eliminates survivorship bias in backtesting
        self.PIT_UNIVERSE_TOP_N: int = int(os.getenv('PIT_UNIVERSE_TOP_N', '500'))
        self.PIT_UNIVERSE_EXCHANGES: list[str] = os.getenv(
            'PIT_UNIVERSE_EXCHANGES', 'NYSE,NASDAQ'
        ).split(',')
        self.PIT_VOLUME_WINDOW_DAYS: int = int(os.getenv('PIT_VOLUME_WINDOW_DAYS', '20'))
        self.PIT_DEFAULT_TO_PIT: bool = os.getenv(
            'PIT_DEFAULT_TO_PIT', 'true'
        ).lower() in ('true', '1', 'yes')
        
        # OpenAI settings
        self.OPENAI_MODEL: str = os.getenv('OPENAI_MODEL', 'gpt-5.2-mini')
        self.OPENAI_MAX_TOKENS: int = int(os.getenv('OPENAI_MAX_TOKENS', '500'))
    
    def validate(self, require_alpha_vantage: bool = True) -> None:
        """
        Validate that required configuration is present.
        
        Parameters
        ----------
        require_alpha_vantage : bool, default True
            Whether to require Alpha Vantage API key
            
        Raises
        ------
        ValueError
            If required configuration is missing
            
        Examples
        --------
        >>> from src.config import config
        >>> config.validate()  # Raises ValueError if ALPHAVANTAGE_API_KEY missing
        >>> config.validate(require_alpha_vantage=False)  # Only check optional keys
        """
        errors = []
        
        if require_alpha_vantage and not self.ALPHAVANTAGE_API_KEY:
            errors.append(
                "ALPHAVANTAGE_API_KEY not found. Set it as an environment variable "
                "or in a .env file in the project root."
            )
        
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    def validate_openai(self) -> bool:
        """
        Check if OpenAI API key is configured.
        
        Returns
        -------
        bool
            True if OpenAI API key is present, False otherwise
            
        Examples
        --------
        >>> from src.config import config
        >>> if config.validate_openai():
        ...     # Proceed with LLM factor naming
        ... else:
        ...     print("OpenAI not configured, skipping factor naming")
        """
        return self.OPENAI_API_KEY is not None and len(self.OPENAI_API_KEY) > 0
    
    def get_cache_path(self, filename: str) -> Path:
        """
        Get full path for a cache file.
        
        Parameters
        ----------
        filename : str
            Name of the cache file
            
        Returns
        -------
        Path
            Full path to cache file
        """
        return self.CACHE_DIR / filename
    
    def __repr__(self) -> str:
        return (
            f"Configuration("
            f"ALPHAVANTAGE_API_KEY={'***' if self.ALPHAVANTAGE_API_KEY else None}, "
            f"OPENAI_API_KEY={'***' if self.OPENAI_API_KEY else None}, "
            f"CACHE_DIR={self.CACHE_DIR}, "
            f"LOG_LEVEL={self.LOG_LEVEL}"
            f")"
        )


# Global configuration instance
config = Configuration()


# Convenience function for quick validation
def validate_config(require_alpha_vantage: bool = True) -> None:
    """
    Quick validation of configuration.
    
    This is a convenience function that calls config.validate().
    
    Parameters
    ----------
    require_alpha_vantage : bool, default True
        Whether to require Alpha Vantage API key
        
    Raises
    ------
    ValueError
        If required configuration is missing
        
    Examples
    --------
    >>> from src.config import validate_config
    >>> validate_config()  # Raises if ALPHAVANTAGE_API_KEY missing
    """
    config.validate(require_alpha_vantage=require_alpha_vantage)
