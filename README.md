# Equity Factors Research System

A comprehensive quantitative finance research platform for factor discovery, optimization, and backtesting. This system combines market data acquisition, factor modeling, and performance evaluation with LLM-powered factor naming. Designed for active traders who want to understand what is driving returns in their portfolios and translate factor insights into tradeable stock baskets.

## Overview

This system helps active traders answer critical questions:
- What hidden factors are driving my universe? (Factor Discovery)
- How should I optimally weight factors for maximum risk-adjusted return? (Factor Optimization)
- Which specific stocks should I buy/sell to implement my factor view? (Basket Generation)
- Is now a good time to buy/sell specific factors? (Trading Signals)
- Which stocks should I long/short based on factor exposures? (Cross-Sectional Analysis)
- What market regime are we in and how should I position? (Regime Detection)
- Would my strategy have worked historically? (Backtesting)

## Key Features

| Feature | What It Does | Why Traders Care |
|---------|--------------|------------------|
| Factor Discovery | Extracts latent factors using PCA, ICA, NMF, or Autoencoders from residualized returns | Discover true alpha factors, not just market beta and sector exposures |
| Factor Optimization | Bayesian optimization of factor weight blends | Maximize Sharpe ratio with data-driven weightings |
| Basket Generation | Convert factor weights to specific stock positions | Go from abstract factors to tradeable long/short baskets |
| LLM Factor Naming | Auto-generates intuitive names using GPT models | Transform cryptic "F3" into "Small-Cap Growth Momentum" |
| Trading Signals | RSI, MACD, ADX, extreme value alerts, z-scores | Time factor entries/exits with confidence |
| Cross-Sectional Analysis | Rank stocks by composite factor scores | Generate quantifiable long/short candidates |
| Regime Detection | HMM-based market state identification | Adjust factor exposure based on bull/bear/volatile conditions |
| Signal Backtesting | Walk-forward testing with performance attribution | Validate strategies before risking capital |
| PIT Universe Construction | Reconstructs historical market state including delisted stocks | Eliminate survivorship bias from backtests |
| Interactive Dashboard | Streamlit UI for real-time monitoring | Visual factor monitoring without coding |
| **Alpha Command Center** | Institutional-grade PM dashboard | Risk attribution, explainability, execution |

## Alpha Command Center: Factor Operations Terminal

An institutional-grade dashboard designed for Portfolio Managers with $1B+ AUM. Shifts design philosophy from **"Research Tool"** (exploring data) to **"Command Center"** (monitoring risk, understanding drivers, and executing mandates).

### Launch the Dashboard

```bash
# Using the launch script (recommended)
./scripts/launch_alpha_command_center.sh

# Or manually
streamlit run src/dashboard_alpha_command_center.py
```

### Core Philosophy

1. **Alpha vs. Beta Separation:** Every screen clearly distinguishes between "Market Drift" and "True Latent Alpha"
2. **Explainability First:** Never show "Factor 3." Show "Factor 3 (Tech-Momentum)" with breakdown of *why* it exists
3. **Actionable Intelligence:** Charts lead to decisions (Rebalance, Hedge, or De-risk)

### Dashboard Sections

#### Section 1: "Morning Coffee" Header
Immediate situational awareness for the PM:
- **Market Regime Gauge:** HMM-based gauge showing current state ("Low Vol Bull" vs "Crisis")
- **Active Risk (Tracking Error):** vs Benchmark
- **Estimated Beta:** Must be near 0.0 for Market Neutral fund
- **Unexplained PnL:** The "Ghost" alpha or risk not explained by factors
- **Information Ratio:** Risk-adjusted alpha metric

#### Section 2: The "Factor Lab"
Explainable AI (XAI) for factor understanding:
- **Factor DNA Table:** Master table with Alpha Purity scores (flags "Beta in Disguise")
  - Auto-generated factor names based on loading analysis
  - Theme classification (Concentrated Exposure, Statistical Arbitrage, etc.)
  - Crowding scores based on concentration metrics
- **Factor X-Ray:** Drill-down with semantic description, style attribution, sector heatmap
  - **Investment Rationale:** Actionable description of what the factor captures
  - **Top Exposures:** Top 5 long/short positions with actual loading values
  - **Conviction Score:** High/Moderate/Low based on dispersion metrics
  - **🤖 LLM Enhancement:** Button to call OpenAI API for intuitive naming
- **Style Attribution:** Regression vs Fama-French factors (Value, Momentum, Quality, etc.)
- **Liquidity Warnings:** Pre-trade execution alerts

#### Section 3: Portfolio Constructor
Execution and "What-If" analysis:
- **Optimizer Sandbox:** Stress-test with turnover constraints, beta limits
- **Efficient Frontier:** Visualize Sharpe vs Turnover trade-off
- **Trade Basket Preview:** Top Buys/Sells with liquidity warnings
- **% ADV Alerts:** Flags trades exceeding 2% of Average Daily Volume

#### Section 4: Risk & Drawdown
Risk monitoring and attribution:
- **Factor Risk Metrics:** VaR, CVaR, Max Drawdown per factor
- **Drawdown Chart:** Visual tracking of underwater periods
- **Skewness/Kurtosis:** Tail risk indicators

## Factor Discovery with Residualization

The factor discovery system automatically residualizes stock returns against market (SPY) and sector ETFs before extracting latent factors. This is critical for alpha generation:

**The Problem:** Running PCA/ICA on raw returns mathematically guarantees that the first principal component will be the Market Factor (Beta), and components 2-5 will be Sector factors. Without residualization, you are rediscovering SPY and sector ETFs, not true alpha.

**The Solution:** Before factor extraction, the system:
1. Regresses stock returns against SPY: `R_i = alpha_i + beta_i * R_mkt + epsilon_i`
2. Regresses residuals against sector ETFs: `epsilon_i = alpha'_i + sum(gamma_ij * R_sector_j) + eta_i`
3. Runs factor discovery on `eta_i` (pure idiosyncratic returns)

**Sector ETFs Used:** XLK (Technology), XLF (Financials), XLE (Energy), XLI (Industrials), XLP (Consumer Staples), XLY (Consumer Discretionary), XLB (Materials), XLU (Utilities), XLRE (Real Estate), XLC (Communication Services)

**Factor Orthogonality:** Non-orthogonal methods (ICA, NMF, Autoencoder) are automatically orthogonalized post-discovery using symmetric SVD-based orthogonalization. This prevents optimizers from double-counting correlated signals.

## Point-in-Time (PIT) Universe Construction

The PIT Universe Generation system eliminates **survivorship bias** and **look-ahead bias** from backtesting by reconstructing historical market states using Alpha Vantage LISTING_STATUS data.

### The Problem

Traditional backtests using `get_etf_holdings()` project current ETF constituents into the past. This creates false performance metrics because:
- **Survivorship Bias**: Delisted stocks (Lehman Brothers, Silicon Valley Bank) are excluded
- **Look-ahead Bias**: Companies that didn't exist yet are included in historical universes

### The Solution

The PIT system reconstructs the actual market state on any historical date:

1. **Fetches both active AND delisted stocks** from LISTING_STATUS endpoint
2. **Filters by liquidity** using dollar volume calculations
3. **Stores historical state** in the `historical_universes` table
4. **Retrieves true PIT universe** during backtesting

### Quick Example

```python
from src.research import FactorResearchSystem

frs = FactorResearchSystem(api_key, universe=["SPY"])

# Get the TRUE investable universe on Sept 15, 2008 (Lehman bankruptcy)
tickers = frs.get_backtest_universe('2008-09-15', top_n=500)

# Verify PIT is working (LEH = Lehman Brothers)
if 'LEH' in tickers:
    print("✓ PIT correctly includes delisted stocks")

# Verify with QA helper
result = frs.verify_pit_universe('2008-09-15', ['LEH'])
assert result['pass'], "Survivorship bias detected!"
```

### PIT vs Legacy Comparison

```python
# ❌ WRONG: Current constituents projected to past (biased)
biased_universe = backend.get_etf_holdings("SPY")['constituent'].tolist()
# Returns: ['AAPL', 'MSFT', 'NVDA', ...]  # LEH missing!

# ✅ CORRECT: True historical state (unbiased)
unbiased_universe = frs.get_backtest_universe('2008-09-15')
# Returns: ['AAPL', 'MSFT', 'LEH', ...]  # LEH included!
```

### Building PIT Universes

```python
from src.alphavantage_system import DataBackend

backend = DataBackend(api_key)

# Build universe for a specific date (API intensive - run monthly)
universe_df = backend.build_point_in_time_universe(
    date_str='2008-09-15',
    top_n=500,
    exchanges=['NYSE', 'NASDAQ']
)

# Check results
print(f"Universe size: {len(universe_df)}")
print(f"Delisted stocks: {(universe_df['status'] == 'Delisted').sum()}")
```

### Configuration

Environment variables for PIT behavior:

```bash
# Default universe size
PIT_UNIVERSE_TOP_N=500

# Exchanges to include (reduces API calls)
PIT_UNIVERSE_EXCHANGES=NYSE,NASDAQ

# Dollar volume calculation window
PIT_VOLUME_WINDOW_DAYS=20

# Use PIT by default for backtesting
PIT_DEFAULT_TO_PIT=true
```

### QA Verification (Critical Tests)

Before deploying to live capital, run these tests:

```bash
# Run all PIT tests
pytest tests/test_pit_universe.py -v

# Critical QA checks
pytest tests/test_pit_universe.py::TestPITUniverseQA -v
```

**Test 1: Lehman Brothers (2008)**
- Date: `2008-09-15`
- Expected: `LEH` ticker present
- Failure indicates survivorship bias

**Test 2: Silicon Valley Bank (2023)**
- Date: `2023-01-01`
- Expected: `SIVB` ticker present
- Failure indicates survivorship bias

```bash
# Standalone verification script
python tests/test_pit_universe.py
```

### Usage in Backtesting

```python
from src.research import FactorResearchSystem
import pandas as pd

frs = FactorResearchSystem(api_key, universe=["SPY"])

# Walk-forward backtest with PIT universes
backtest_dates = pd.date_range('2008-01-01', '2023-12-31', freq='MS')

for date in backtest_dates:
    # Get true historical universe for this date
    universe = frs.get_backtest_universe(date.strftime('%Y-%m-%d'))
    
    # Run factor model on this date's universe only
    frs.universe = universe
    frs.fit_factors()
    
    # Generate signals...
```

### Database Schema

The `historical_universes` table stores PIT state:

| Column | Type | Description |
|--------|------|-------------|
| date | TEXT | Reference date (YYYY-MM-DD) |
| ticker | TEXT | Stock symbol |
| asset_type | TEXT | 'Stock' or 'ETF' |
| status | TEXT | 'Active' or 'Delisted' |
| dollar_volume | REAL | 20-day avg dollar volume |

```sql
-- Query historical universe
SELECT ticker, status, dollar_volume
FROM historical_universes
WHERE date = '2008-09-15'
ORDER BY dollar_volume DESC
LIMIT 500;
```

## Quick Start

### Prerequisites

- Python 3.11+
- Alpha Vantage API key (free tier available)
- OpenAI API key (for factor naming - optional but recommended)
- uv package manager (recommended) or pip

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd equity-factors

# Install dependencies (using uv)
uv sync

# Or using pip
pip install -e .

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys:
# ALPHAVANTAGE_API_KEY=your-alpha-vantage-key
# OPENAI_API_KEY=your-openai-key
```

## Usage Guide for Active Traders

### Unified CLI Quick Reference

All functionality is accessible through the unified CLI:

```bash
# Main commands
uv run python -m src discover      # Factor discovery and naming
uv run python -m src optimize      # Factor weight optimization
uv run python -m src basket        # Generate tradeable basket
uv run python -m src signals       # Trading signals (extremes, momentum, cross-section)
uv run python -m src regime        # Market regime detection
uv run python -m src backtest      # Signal backtesting
uv run python -m src dashboard     # Launch Streamlit dashboard
uv run python -m src report        # Generate reports
uv run python -m src clean         # Clean cache files

# Get help for any command
uv run python -m src <command> --help
```

### Factor Discovery and Naming

**What it does:** Automatically discovers latent factors in any ETF or stock universe and gives them meaningful names.

**Why use it:** Discovers true alpha factors from residualized returns, not just market beta and sector exposures.

```bash
# Discover factors in SPY using PCA
uv run python -m src discover --symbols SPY --method PCA -k 10

# Analyze with ICA for regime-sensitive factors (auto-orthogonalized)
uv run python -m src discover --symbols SPY --method ICA -k 8

# Multi-ETF analysis for broader factor discovery
uv run python -m src discover --symbols "SPY,QQQ,IWM" --method PCA -k 12
```

**Output:**
- `factor_names.csv` - Human-readable factor names
- `factor_returns.csv` - Daily factor return time series
- `factor_loadings.csv` - Stock exposures to each factor
- `cumulative_factor_returns.png` - Visual performance chart

### Factor Weight Optimization

**What it does:** Finds the optimal blend of factors that maximizes Sharpe ratio using advanced optimization techniques.

**Why use it:** Instead of equal-weighting factors, let the data tell you which factors deserve higher/lower weights. The system uses Nested Walk-Forward Optimization to prevent overfitting, ensuring that the strategies discovered in-sample actually generalize out-of-sample.

**Key Features:**
- **Anti-Overfitting Logic:** Splits lookback windows into Estimation and Validation sets. Weights are trained on one and optimized on the other, eliminating the double-dipping bias where the model simply picks past winners.
- **High Performance:** Weight generation for all methods (Risk Parity, Min-Var, etc.) is pre-calculated once, making the meta-optimization 10-50x faster.
- **Realistic Backtesting:** Simulates daily weight drift and buy-and-hold returns between rebalancing periods, providing a more accurate view of real-world performance than simple constant-mix models.

```bash
# Basic optimization with PCA factors
uv run python -m src optimize --universe SPY \
  --factor-method pca --n-components 10 \
  --lookback 126 --technique bayesian \
  --export-weights optimal_weights.csv \
  --output optimization_results.json

# Walk-forward optimization (more robust)
uv run python -m src optimize --universe VTHR \
  --factor-method ica --n-components 12 \
  --walk-forward --train-window 90 --test-window 30 \
  --technique bayesian \
  --methods sharpe momentum risk_parity \
  --export-weights vthr_weights.csv \
  --output vthr_results.json
```

**Available Factor Methods:**
- `pca` - Principal Component Analysis (default, stable, orthogonal by construction)
- `ica` - Independent Component Analysis (regime-sensitive, auto-orthogonalized)
- `sparse_pca` - Sparse PCA (interpretable loadings)
- `factor` - Factor Analysis (classical approach)
- `fundamental` - Fama-French style (P/E, market cap, etc.)

**Optimization Techniques:**
- `bayesian` - Bayesian optimization with Optuna (recommended)
- `differential` - Differential evolution (global search)
- `gradient` - Gradient ascent (fast, may find local optima)

**Weighting Methods for Blending:**
- `sharpe` - Historical risk-adjusted returns (default)
- `momentum` - Recent performance persistence (default)
- `risk_parity` - Equal risk contribution (default)
- `min_variance` - Minimum variance weighting
- `max_diversification` - Maximum diversification ratio
- `equal` - Equal weighting
- `pca` - Weight by eigenvalue (variance explained)

**Default blend:** `['sharpe', 'momentum', 'risk_parity']`

**Output:**
- `*_weights.csv` - Optimal factor weights
- `*_results.json` - Full optimization results with performance metrics
- `*_factor_names.csv` - Human-readable factor names (if --name-factors)

### Generate Tradeable Basket

**What it does:** Converts optimal factor weights into specific long/short stock positions.

**Why use it:** Go from abstract factor allocations to concrete tradeable positions.

```bash
# Generate basket from optimization results
uv run python -m src basket \
  --results vthr_results.json \
  --universe VTHR \
  --factor-method pca \
  --long-pct 0.05 \
  --short-pct 0.05 \
  --capital 100000 \
  --net-exposure 1.0 \
  -o trade_basket.csv
```

**How it works:**
1. Takes optimal factor weights from optimization
2. Calculates composite factor exposure for each stock
3. Ranks stocks by composite score
4. Selects top X% for longs, bottom X% for shorts
5. Calculates position sizes based on target weights

**Output (`trade_basket.csv`):**
| ticker | composite_score | target_weight | side | position_dollars |
|--------|----------------|---------------|------|-----------------|
| AAPL | 0.0523 | 0.025 | LONG | $2,500 |
| MSFT | 0.0489 | 0.023 | LONG | $2,300 |
| ... | ... | ... | ... | ... |
| XYZ | -0.0412 | -0.021 | SHORT | -$2,100 |

**Parameters:**
- `--long-pct 0.05` - Top 5% of stocks go long
- `--short-pct 0.05` - Bottom 5% of stocks go short
- `--capital 100000` - Total capital to allocate
- `--net-exposure 1.0` - Net exposure (1.0 = 100% long, 0.0 = market neutral, -1.0 = 100% short)

### Trading Signals

**What it does:** Generates actionable buy/sell signals based on factor momentum, extreme values, and cross-sectional rankings.

```bash
# Check for extreme value alerts (mean reversion opportunities)
uv run python -m src signals extremes --universe SPY --threshold 2.0 --trade

# Analyze factor momentum (RSI, MACD, ADX)
uv run python -m src signals momentum --universe SPY --factor F1

# Cross-sectional stock rankings (long/short candidates)
uv run python -m src signals cross-section --universe SPY --top-pct 0.1 --bottom-pct 0.1
```

**Signal Types:**

| Signal | Best For | When to Use |
|--------|----------|-------------|
| RSI | Timing mean reversion | RSI > 70 = overbought; RSI < 30 = oversold |
| MACD | Trend confirmation | Bullish crossover = enter long |
| ADX | Trend strength | ADX > 25 = strong trend; ADX < 20 = weak trend |
| Z-Score | Statistical extremes | |z| > 2 = rare event, likely to revert |

### Market Regime Detection

**What it does:** Uses Hidden Markov Models to identify market regimes and recommend factor allocations.

```bash
# Detect current regime
uv run python -m src regime detect --universe SPY --regimes 3 --predict 5
```

**Regime Types:**

| Regime | Characteristics | Optimal Strategy |
|--------|-----------------|------------------|
| Low Vol Bull | Rising prices, low volatility | Overweight momentum, growth |
| High Vol Bear | Declining, choppy | Maximum defensive, minimum volatility |
| Crisis | Extreme volatility | Risk-off, cash/quality focus |

### Signal Backtesting

**What it does:** Validates signal efficacy using walk-forward testing with Point-in-Time (PIT) universes to eliminate survivorship bias.

```bash
# Run backtest
uv run python -m src backtest --universe SPY \
  --train-size 252 --test-size 63 --walks 10 \
  --optimize --report
```

**PIT Universe in Backtesting:**

```python
from src.research import FactorResearchSystem

frs = FactorResearchSystem(api_key, universe=["SPY"])

# Get PIT universe for a specific backtest date
tickers = frs.get_backtest_universe('2008-09-15', top_n=500)

# Use in walk-forward loop
for date in backtest_dates:
    # True historical universe (includes delisted stocks)
    universe = frs.get_backtest_universe(date.strftime('%Y-%m-%d'))
    
    # Run analysis on this date's actual investable universe
    # ...
```

### Interactive Dashboard

```bash
# Launch the Streamlit dashboard
uv run python -m src dashboard
```

## Complete Trading Workflow Example

### Discover, Optimize, Trade

```bash
# Step 1: Discover factors in your universe
uv run python -m src discover --symbols VTHR --method PCA -k 12

# Step 2: Optimize factor weights for maximum Sharpe
uv run python -m src optimize --universe VTHR \
  --factor-method pca --n-components 12 \
  --walk-forward --train-window 90 --test-window 30 \
  --technique bayesian --name-factors \
  --export-weights optimal_weights.csv \
  --output optimization_results.json

# Step 3: Generate tradeable basket
uv run python -m src basket \
  --results optimization_results.json \
  --universe VTHR \
  --factor-method pca \
  --long-pct 0.05 --short-pct 0.05 \
  --capital 100000 \
  -o my_trade_basket.csv

# Step 4: (Manual) Execute trades from my_trade_basket.csv
```

### Daily Pre-Market Routine

```bash
# Check overnight regime status
uv run python -m src regime detect --universe SPY --predict 5

# Review extreme value alerts
uv run python -m src signals extremes --universe SPY --trade

# Generate cross-sectional rankings
uv run python -m src signals cross-section --universe SPY --top-pct 0.05
```

## Project Structure

```
equity-factors/
├── src/
│   ├── __main__.py                 # Unified CLI entry point
│   ├── database.py                 # Robust SQLite management
│   ├── alphavantage_system.py      # Data backend with caching
│   ├── research.py                 # Main research system
│   ├── factor_optimization.py      # Factor weight optimization
│   ├── factor_weighting.py         # Factor weighting methods
│   ├── latent_factors.py           # Factor discovery (PCA, ICA, NMF, AE)
│   ├── factor_labeler.py           # LLM-powered factor naming
│   ├── trading_signals.py          # Technical indicators
│   ├── cross_sectional.py          # Cross-sectional ranking
│   ├── regime_detection.py         # HMM-based regime identification
│   ├── signal_aggregator.py        # Signal combination
│   ├── signal_backtest.py          # Walk-forward backtesting
│   ├── dashboard.py                # Streamlit dashboard
│   └── reporting.py                # Report generation
├── tests/                          # Unit tests
│   └── test_pit_universe.py        # PIT QA verification tests
├── notebooks/                      # Jupyter notebooks
├── pyproject.toml                  # Dependencies
└── README.md                       # This file
```

## Configuration

### Environment Variables

```bash
# Required for data fetching
ALPHAVANTAGE_API_KEY=your_alpha_vantage_key

# Required for factor naming (optional but recommended)
OPENAI_API_KEY=your_openai_key

# Optional: Database busy timeout (milliseconds)
SQLITE_BUSY_TIMEOUT_MS=30000
```

### API Rate Limits

The system implements intelligent rate limiting:
- Price data: 75 calls/minute
- Fundamentals: 75 calls/minute
- Automatic throttling and progress tracking

**Cost Optimization:**
- SQLite caching minimizes repeated API calls
- First run downloads all data; subsequent runs use cache
- Free Alpha Vantage tier: 25 calls/day (premium recommended)

## Database Architecture

The system uses a robust SQLite architecture with:

- **DELETE journal mode** instead of WAL (eliminates persistent locks)
- **30-second busy timeout** on all connections
- **Per-operation connections** with guaranteed cleanup
- **Automatic WAL migration** from legacy databases
- **Lock detection and recovery** utilities

**Troubleshooting:**
```bash
# Check database health
uv run python -c "from src.database import check_database_health; print(check_database_health())"

# Optimize database
uv run python -c "from src.database import optimize_database; optimize_database()"

# Reset if corrupted
uv run python -m src clean --all
```

## Performance Notes

- **NMF Performance:** For large ETFs (2000+ stocks), NMF can be slow. Use PCA or ICA.
- **Autoencoder:** Requires PyTorch; benefits from GPU acceleration.
- **Caching:** First run downloads data; subsequent runs use SQLite cache.
- **Memory:** Large universes (1000+ stocks) require approximately 2GB RAM.
- **ICA Convergence:** May not converge with very large universes (>2000 stocks).

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

## Risk Disclaimer

**This system is for research and educational purposes only.** Past performance of signals and factors does not guarantee future results. Always conduct your own due diligence before making investment decisions. The authors are not responsible for any trading losses incurred from using this software.
