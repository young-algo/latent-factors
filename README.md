# Equity Factors Research System

A comprehensive quantitative finance research platform for factor discovery, analysis, and backtesting. This system combines market data acquisition, factor modeling, and performance evaluation with innovative LLM-powered factor naming.

## Features

- **Multi-source Data Backend**: Alpha Vantage API with yfinance fallback
- **Factor Discovery**: Statistical methods (PCA, ICA, NMF) and deep learning (autoencoders)
- **ETF Universe Expansion**: Automatically expand ETFs to their constituent holdings
- **LLM Factor Naming**: Automatic factor labeling using OpenAI's GPT models
- **Performance Analytics**: Comprehensive backtesting and performance reporting
- **Intelligent Caching**: SQLite-based caching to minimize API costs

## Quick Start

### Prerequisites

- Python 3.11+
- Alpha Vantage API key (free tier available)
- OpenAI API key (for factor naming)
- uv package manager (recommended) or pip

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd equity-factors
```

2. Install dependencies (using uv):
```bash
uv sync
```

Or using pip:
```bash
pip install -e .
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys:
# ALPHAVANTAGE_API_KEY=your-alpha-vantage-key
# OPENAI_API_KEY=your-openai-key
```

### Basic Usage

1. **Discover and Label Factors** (Main Workflow):
```bash
# Discover factors in an ETF using PCA
uv run python src/discover_and_label.py -s "SPY" --method PCA -k 10

# Use different factor methods
uv run python src/discover_and_label.py -s "XLF" --method NMF -k 8
uv run python src/discover_and_label.py -s "QQQ" --method ICA -k 15
uv run python src/discover_and_label.py -s "ARKK" --method AE -k 5

# Multiple ETFs/tickers
uv run python src/discover_and_label.py -s "SPY,QQQ,IWM" --method PCA -k 20

# With custom date range
uv run python src/discover_and_label.py -s "XLK" --start 2020-01-01 --method PCA -k 10
```

The workflow will:
- Expand ETFs to constituent stocks automatically
- Fetch historical price data (with caching)
- Discover latent factors using the specified method
- Send factor loadings to OpenAI for intelligent naming
- Save factor names to `factor_names.csv`
- Display cumulative factor returns plot

2. **Run Factor Research** (Programmatic):
```python
from src.research import FactorResearchSystem

# Initialize with ETF universe (automatically expanded to constituents)
frs = FactorResearchSystem(
    api_key=os.getenv("ALPHAVANTAGE_API_KEY"),
    universe=["SPY"],  # S&P 500 ETF
    factor_method="fundamental"
)

# Fit factor model
frs.fit_factors()

# Get factor returns with volatility targeting
factor_returns = frs.get_factor_returns(vol_target=0.10)

# Generate performance report
from src.factor_performance_table import generate_report
generate_report(factor_returns)
```

3. **Momentum Backtesting**:
```bash
uv run python fixed_momentum_backtest.py
```

4. **Full Research Pipeline**:
```bash
uv run python src/runner.py
```

## Project Structure

```
equity-factors/
   src/
      alphavantage_system.py     # Data backend with caching and ETF expansion
      research.py                # Main research system and factor models
      latent_factors.py          # Factor discovery (PCA, ICA, NMF, Autoencoder)
      factor_labeler.py          # LLM-powered factor naming with OpenAI
      discover_and_label.py      # Main CLI workflow for factor discovery
      factor_performance_table.py # Performance reporting and analytics
      runner.py                  # Legacy entry point
   notebooks/                    # Jupyter analysis notebooks
   pyproject.toml               # Project dependencies
   .env.example                 # Environment variable template
   README.md
```

## Configuration

### Environment Variables

- `ALPHAVANTAGE_API_KEY`: Your Alpha Vantage API key
- `OPENAI_API_KEY`: Your OpenAI API key (for factor naming)

### Factor Methods

For `discover_and_label.py`:
- `PCA`: Principal Component Analysis - orthogonal linear factors
- `ICA`: Independent Component Analysis - statistically independent factors
- `NMF`: Non-negative Matrix Factorization - parts-based decomposition (note: slower for large ETFs)
- `AE`: Deep Autoencoder - non-linear latent factors (requires PyTorch)

For `research.py`:
- `"fundamental"`: Cross-sectional factor model using fundamental data
- `"pca"`: Principal Component Analysis on returns

### Data Sources

- **Primary**: Alpha Vantage (price data, fundamentals, ETF holdings)
- **Fallback**: yfinance (when Alpha Vantage fails)
- **Cache**: Local SQLite database (`av_cache.db`)

## API Rate Limits

The system implements intelligent rate limiting:
- Price data: 75 calls/minute
- Fundamentals: 75 calls/minute
- Automatic throttling and progress tracking

## Workflow Details

### How Factor Discovery Works

1. **ETF Expansion**: When you specify an ETF (e.g., SPY, QQQ, IWM), the system:
   - Fetches ETF constituents from Alpha Vantage
   - Downloads historical price data for all component stocks
   - Caches data locally to avoid repeated API calls

2. **Factor Discovery**: The system applies your chosen method:
   - Calculates daily returns from price data
   - Applies dimensionality reduction to find latent factors
   - Returns factor loadings (stock exposures) and factor returns

3. **Intelligent Naming**: For each discovered factor:
   - Identifies top positive and negative loading stocks
   - Retrieves fundamental data (sector, market cap)
   - Sends context to OpenAI for descriptive naming
   - Example: "Finance Beta Balance: High-beta growth vs low-beta value stocks"

## Examples

### Discovering Tech Sector Factors
```bash
# Analyze technology sector with 10 factors
uv run python src/discover_and_label.py -s "XLK" --method PCA -k 10

# Output:
# - Fetches ~70 tech stocks from XLK ETF
# - Discovers 10 latent factors
# - Names like "Cloud Computing Leaders", "Hardware Cyclicals", etc.
```

### Multi-ETF Analysis
```bash
# Discover factors across market segments
uv run python src/discover_and_label.py -s "SPY,QQQ,IWM" --method ICA -k 20

# Finds factors that span large-cap (SPY), tech (QQQ), and small-cap (IWM)
```

### Custom Date Ranges
```bash
# Analyze factors during specific periods
uv run python src/discover_and_label.py -s "XLF" --start 2020-01-01 --method PCA -k 8

# Useful for regime-specific analysis (e.g., COVID period, rate hikes)
```

## Performance Notes

- **NMF Performance**: For large ETFs like IWM (2000+ stocks), NMF can be slow. Consider using PCA or ICA for faster results
- **Caching**: First run will be slower due to API calls. Subsequent runs use cached data
- **API Limits**: Free Alpha Vantage tier allows 75 requests/minute. Premium tier recommended for production use

## Performance Monitoring

The system provides comprehensive performance analytics:
- Multi-period returns (1D, 1W, 1M, 3M, 6M, 1Y)
- Risk metrics (Sharpe ratio, maximum drawdown, hit rate)
- Visual heatmaps and charts
- CSV export functionality

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

[Add your license here]

## Support

For issues and questions:
- Check existing issues in the repository
- Create a new issue with detailed description
- Include error messages and environment details
