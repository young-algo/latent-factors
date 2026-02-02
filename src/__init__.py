"""
Equity Factors Research System
==============================

A comprehensive quantitative finance research platform for factor discovery, 
analysis, and backtesting with LLM-powered factor naming.

Quick Start
-----------
>>> from src import FactorResearchSystem, config
>>> config.validate()
>>> frs = FactorResearchSystem(
...     api_key=config.ALPHAVANTAGE_API_KEY,
...     universe=["SPY"],
...     factor_method="fundamental"
... )
>>> frs.fit_factors()
>>> factor_returns = frs.get_factor_returns()

Modules
-------
- research : Core factor research system with ETF expansion
- trading_signals : Technical indicators for factor returns
- cross_sectional : Cross-sectional stock ranking and selection
- regime_detection : HMM-based market regime identification
- signal_aggregator : Combines multiple signal types
- signal_backtest : Walk-forward backtesting framework
- latent_factors : Statistical factor discovery (PCA, ICA, NMF, AE)
- factor_labeler : LLM-powered factor naming
- config : Centralized configuration management

Command Line
------------
Use the unified CLI for all operations:

    # Discover and name factors
    uv run python -m src discover --symbols SPY --method PCA -k 10
    
    # Generate trading signals
    uv run python -m src signals generate --universe SPY
    
    # Launch dashboard
    uv run python -m src dashboard
    
    # Run backtest
    uv run python -m src backtest --universe SPY

For detailed help:
    uv run python -m src --help
"""

__version__ = "0.1.0"
__author__ = "Equity Factors Research Team"

# Core research system
from .research import FactorResearchSystem

# Signal generation modules
from .trading_signals import (
    FactorMomentumAnalyzer,
    MomentumRegime,
    SignalStrength,
    TradingSignal,
    ExtremeAlert,
)
from .cross_sectional import (
    CrossSectionalAnalyzer,
    SignalDirection,
    StyleFactor,
    StockSignal,
    StyleRotation,
)
from .regime_detection import (
    RegimeDetector,
    SimpleRegimeDetector,
    MarketRegime,
    RegimeState,
    RegimeAllocation,
)
from .signal_aggregator import (
    SignalAggregator,
    SignalType,
    SignalDirection as AggregatorSignalDirection,
    Signal,
    ConsensusSignal,
    TradingOpportunity,
)
from .signal_backtest import (
    SignalBacktester,
    BacktestMetric,
    BacktestResult,
    SignalPerformance,
)

# Factor discovery
from .latent_factors import (
    statistical_factors,
    autoencoder_factors,
    StatMethod,
    validate_factor_distinctiveness,
)

# Factor weighting
from .factor_weighting import (
    OptimalFactorWeighter,
    FactorCharacteristics,
    WeightingMethod,
)

# Factor naming
from .factor_labeler import (
    ask_llm,
    batch_name_factors,
    validate_api_key,
)
from .factor_naming import (
    FactorName,
    validate_name,
    score_quality,
    detect_tags,
    generate_name_prompt,
    parse_name_response,
    generate_quality_report,
    BANNED_WORDS,
)

# Configuration
from .config import config, validate_config, Configuration

__all__ = [
    # Version
    "__version__",
    
    # Core
    "FactorResearchSystem",
    
    # Trading Signals
    "FactorMomentumAnalyzer",
    "MomentumRegime",
    "SignalStrength",
    "TradingSignal",
    "ExtremeAlert",
    
    # Cross-Sectional
    "CrossSectionalAnalyzer",
    "SignalDirection",
    "StyleFactor",
    "StockSignal",
    "StyleRotation",
    
    # Regime Detection
    "RegimeDetector",
    "SimpleRegimeDetector",
    "MarketRegime",
    "RegimeState",
    "RegimeAllocation",
    
    # Signal Aggregation
    "SignalAggregator",
    "SignalType",
    "AggregatorSignalDirection",
    "Signal",
    "ConsensusSignal",
    "TradingOpportunity",
    
    # Backtesting
    "SignalBacktester",
    "BacktestMetric",
    "BacktestResult",
    "SignalPerformance",
    
    # Factor Discovery
    "statistical_factors",
    "autoencoder_factors",
    "StatMethod",
    "validate_factor_distinctiveness",
    
    # Factor Weighting
    "OptimalFactorWeighter",
    "FactorCharacteristics",
    "WeightingMethod",
    
    # Factor Naming
    "ask_llm",
    "batch_name_factors",
    "validate_api_key",
    "FactorName",
    "validate_name",
    "score_quality",
    "detect_tags",
    "generate_name_prompt",
    "parse_name_response",
    "generate_quality_report",
    "BANNED_WORDS",
    
    # Configuration
    "config",
    "validate_config",
    "Configuration",
]


def get_version() -> str:
    """Return the package version."""
    return __version__


def print_version():
    """Print the package version."""
    print(f"Equity Factors Research System v{__version__}")
