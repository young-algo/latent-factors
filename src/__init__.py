"""
Equity Factors Research Library: Comprehensive Factor Discovery and Analysis
===========================================================================

This package provides a complete toolkit for quantitative factor research,
combining statistical factor discovery, fundamental analysis, and LLM-powered
factor interpretation. It's designed for portfolio managers, quantitative
analysts, and researchers working with equity factor models.

Core Components
--------------

**Data Backend (alphavantage_system.py)**
- Alpha Vantage API integration with intelligent caching
- Financial data collection: prices, fundamentals, ETF holdings
- Robust error handling and rate limiting

**Factor Discovery (latent_factors.py)**  
- Statistical methods: PCA, ICA, NMF, Autoencoder
- Factor validation and quality assessment
- Rolling window and static analysis support

**Research Engine (research.py)**
- Comprehensive factor research system
- Automatic ETF expansion to constituents
- Multi-modal factor discovery (fundamental + statistical)

**Factor Naming (factor_labeler.py)**
- LLM-powered factor interpretation using OpenAI
- Rich fundamental context integration
- Robust error handling and retry logic

**Workflow Orchestration (discover_and_label.py)**
- Command-line interface for complete workflow
- End-to-end pipeline from data to factor names
- Comprehensive validation and error handling

**Performance Analysis (factor_performance_table.py)**
- Multi-period performance reporting
- Heat-map visualizations and summary statistics
- Professional presentation and CSV export

**Demo and Testing (runner.py)**
- Complete demonstration workflow
- System validation and benchmarking
- Template for custom research projects

Quick Start
----------
```python
# Complete factor research workflow
from src.research import FactorResearchSystem
from src.factor_performance_table import generate_report

# Initialize with ETF universe (auto-expands to constituents)
frs = FactorResearchSystem(api_key, universe=["SPY"], 
                          factor_method="fundamental")
frs.fit_factors()

# Generate performance report with risk targeting
factors = frs.get_factor_returns(vol_target=0.10)
performance = generate_report(factors)

# Get LLM-generated factor names
factor_names = frs.name_factors()
```

Installation Requirements
------------------------
- **Python**: 3.8+ with pandas, numpy, scikit-learn
- **API Keys**: ALPHAVANTAGE_API_KEY, OPENAI_API_KEY
- **Dependencies**: openai, matplotlib, seaborn
- **Optional**: tensorflow (for autoencoder methods)

Environment Setup
----------------
```bash
# Set environment variables
export ALPHAVANTAGE_API_KEY="your_api_key"
export OPENAI_API_KEY="your_openai_key"

# Install dependencies
pip install pandas numpy scikit-learn openai matplotlib seaborn
```

Package Architecture
-------------------
```
src/
   alphavantage_system.py    # Data backend and caching
   latent_factors.py         # Statistical factor discovery  
   research.py               # Research engine with ETF expansion
   factor_labeler.py         # LLM-powered factor naming
   discover_and_label.py     # Main workflow orchestration
   factor_performance_table.py  # Performance analysis
   runner.py                 # Demo and validation
   __init__.py               # Package initialization
```

Performance Characteristics
--------------------------
- **Small Universe** (50 stocks): ~2-5 minutes end-to-end
- **Large Universe** (500+ stocks): ~10-30 minutes end-to-end  
- **Memory Usage**: 1-4GB depending on universe size
- **API Efficiency**: Intelligent caching minimizes repeated calls

Key Features
-----------
- **ETF Auto-Expansion**: SPY � 500+ constituents automatically
- **Multi-Method Support**: Fundamental, PCA, ICA, NMF, Autoencoder
- **Robust Data Handling**: Missing data, API failures, validation
- **Professional Output**: Heat-maps, CSV exports, factor names
- **Comprehensive Logging**: Progress tracking and debugging

Notes
-----
This library was developed for quantitative factor research with emphasis
on practical usability, robust error handling, and professional presentation
of results. It integrates modern LLM capabilities with traditional
quantitative methods.
"""

# Package metadata
__version__ = "1.0.0"
__author__ = "Equity Factors Research"
__description__ = "Comprehensive factor discovery and analysis toolkit"

# Core imports for easy access
from .research import FactorResearchSystem
from .factor_performance_table import generate_report
from .latent_factors import statistical_factors, autoencoder_factors, validate_factor_distinctiveness
from .factor_labeler import ask_llm, batch_name_factors
from .alphavantage_system import DataBackend

# Define public API
__all__ = [
    "FactorResearchSystem",
    "generate_report", 
    "statistical_factors",
    "autoencoder_factors",
    "validate_factor_distinctiveness",
    "ask_llm",
    "batch_name_factors",
    "DataBackend"
]