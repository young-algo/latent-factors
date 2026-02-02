# Equity Factors Research System

A comprehensive quantitative finance research platform for factor discovery, optimization, and backtesting. This system combines market data acquisition, factor modeling, and performance evaluation with innovative LLM-powered factor naming. Designed for active traders who want to understand what's *actually* driving returns in their portfolios and translate factor insights into tradeable stock baskets.

## What's New

### 🎯 Factor Weight Optimization
Automatically find optimal factor blends that maximize Sharpe ratio using Bayesian optimization, with support for walk-forward validation to avoid overfitting.

### 📊 Tradeable Basket Generation  
Convert factor weights into specific long/short stock positions for immediate execution.

### 🤖 LLM-Powered Factor Naming
Automatically generate human-readable factor names during optimization to understand what each discovered factor represents.

### 🗄️ Robust Database Architecture
Complete re-architecture of SQLite handling eliminates "database is locked" errors with per-operation connections and automatic WAL migration.

### 🔧 Unified CLI
Single entry point for all functionality with consistent command structure.

## Overview

This system helps active traders answer critical questions:
- **What hidden factors are driving my universe?** (Factor Discovery)
- **How should I optimally weight factors for maximum risk-adjusted return?** (Factor Optimization)
- **Which specific stocks should I buy/sell to implement my factor view?** (Basket Generation)
- **Is now a good time to buy/sell specific factors?** (Trading Signals)
- **Which stocks should I long/short based on factor exposures?** (Cross-Sectional Analysis)
- **What market regime are we in and how should I position?** (Regime Detection)
- **Would my strategy have worked historically?** (Backtesting)

## Key Features

| Feature | What It Does | Why Traders Care |
|---------|--------------|------------------|
| **Factor Discovery** | Extracts latent factors using PCA, ICA, NMF, or Autoencoders | Understand the true drivers of returns beyond obvious sector ETFs |
| **Factor Optimization** | Bayesian optimization of factor weight blends | Maximize Sharpe ratio with data-driven weightings |
| **Basket Generation** | Convert factor weights to specific stock positions | Go from abstract factors to tradeable long/short baskets |
| **LLM Factor Naming** | Auto-generates intuitive names using GPT models | Transform cryptic "F3" into "Small-Cap Growth Momentum" |
| **Trading Signals** | RSI, MACD, ADX, extreme value alerts, z-scores | Time factor entries/exits with confidence |
| **Cross-Sectional Analysis** | Rank stocks by composite factor scores | Generate quantifiable long/short candidates |
| **Regime Detection** | HMM-based market state identification | Adjust factor exposure based on bull/bear/volatile conditions |
| **Signal Backtesting** | Walk-forward testing with performance attribution | Validate strategies before risking capital |
| **Interactive Dashboard** | Streamlit UI for real-time monitoring | Visual factor monitoring without coding |

---

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

---

## Usage Guide for Active Traders

### 0. Unified CLI Quick Reference

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

---

### 1. Factor Discovery & Naming

**What it does:** Automatically discovers latent factors in any ETF or stock universe and gives them meaningful names.

**Why use it:** Traditional ETFs (XLK, XLF, etc.) tell you what sector a stock is in, but not what's *actually* driving returns.

```bash
# Discover factors in SPY using PCA
uv run python -m src discover --symbols SPY --method PCA -k 10

# Analyze with ICA for regime-sensitive factors
uv run python -m src discover --symbols SPY --method ICA -k 8

# Multi-ETF analysis for broader factor discovery
uv run python -m src discover --symbols "SPY,QQQ,IWM" --method PCA -k 12
```

**Output:**
- `factor_names.csv` - Human-readable factor names
- `factor_returns.csv` - Daily factor return time series
- `factor_loadings.csv` - Stock exposures to each factor
- `cumulative_factor_returns.png` - Visual performance chart

---

### 2. Factor Weight Optimization ⭐ NEW

**What it does:** Finds the optimal blend of factors that maximizes Sharpe ratio using Bayesian optimization.

**Why use it:** Instead of equal-weighting factors, let the data tell you which factors deserve higher/lower weights.

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

# With factor naming (requires OPENAI_API_KEY)
uv run python -m src optimize --universe SPY \
  --factor-method pca --n-components 10 \
  --name-factors --factor-names-output factor_names.csv \
  --export-weights weights.csv --output results.json
```

**Available Factor Methods:**
- `pca` - Principal Component Analysis (default, stable)
- `ica` - Independent Component Analysis (good for regime detection)
- `sparse_pca` - Sparse PCA (interpretable loadings)
- `factor` - Factor Analysis (classical approach)
- `fundamental` - Fama-French style (P/E, market cap, etc.)

**Optimization Techniques:**
- `bayesian` - Bayesian optimization with Optuna (recommended)
- `differential` - Differential evolution (global search)
- `gradient` - Gradient ascent (fast, may find local optima)

**Weighting Methods for Blending:**
- `sharpe` - Historical risk-adjusted returns
- `momentum` - Recent performance persistence
- `risk_parity` - Equal risk contribution
- `min_variance` - Minimum variance weighting
- `max_diversification` - Maximum diversification ratio
- `equal` - Equal weighting

**Output:**
- `*_weights.csv` - Optimal factor weights
- `*_results.json` - Full optimization results with performance metrics
- `*_factor_names.csv` - Human-readable factor names (if --name-factors)

---

### 3. Generate Tradeable Basket ⭐ NEW

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

---

### 4. Trading Signals

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
| **RSI** | Timing mean reversion | RSI > 70 = overbought; RSI < 30 = oversold |
| **MACD** | Trend confirmation | Bullish crossover = enter long |
| **ADX** | Trend strength | ADX > 25 = strong trend; ADX < 20 = weak trend |
| **Z-Score** | Statistical extremes | \|z\| > 2 = rare event, likely to revert |

---

### 5. Market Regime Detection

**What it does:** Uses Hidden Markov Models to identify market regimes and recommend factor allocations.

```bash
# Detect current regime
uv run python -m src regime detect --universe SPY --regimes 3 --predict 5
```

**Regime Types:**

| Regime | Characteristics | Optimal Strategy |
|--------|-----------------|------------------|
| **Low Vol Bull** | Rising prices, low volatility | Overweight momentum, growth |
| **High Vol Bear** | Declining, choppy | Maximum defensive, minimum volatility |
| **Crisis** | Extreme volatility | Risk-off, cash/quality focus |

---

### 6. Signal Backtesting

**What it does:** Validates signal efficacy using walk-forward testing.

```bash
# Run backtest
uv run python -m src backtest --universe SPY \
  --train-size 252 --test-size 63 --walks 10 \
  --optimize --report
```

---

### 7. Interactive Dashboard

```bash
# Launch the Streamlit dashboard
uv run python -m src dashboard
```

---

## Complete Trading Workflow Example

### Discover → Optimize → Trade

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

---

## Project Structure

```
equity-factors/
├── src/
│   ├── __main__.py                 # Unified CLI entry point ⭐ NEW
│   ├── database.py                 # Robust SQLite management ⭐ NEW
│   ├── alphavantage_system.py      # Data backend with caching
│   ├── research.py                 # Main research system
│   ├── factor_optimization.py      # Factor weight optimization ⭐ NEW
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
├── notebooks/                      # Jupyter notebooks
├── pyproject.toml                  # Dependencies
└── README.md                       # This file
```

---

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

---

## Database Architecture ⭐ NEW

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

---

## Performance Notes

- **NMF Performance:** For large ETFs (2000+ stocks), NMF can be slow. Use PCA or ICA.
- **Autoencoder:** Requires PyTorch; benefits from GPU acceleration.
- **Caching:** First run downloads data; subsequent runs use SQLite cache.
- **Memory:** Large universes (1000+ stocks) require ~2GB RAM.
- **ICA Convergence:** May not converge with very large universes (>2000 stocks).

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## License

[Add your license here]

---

## Support

For issues and questions:
- Check existing issues in the repository
- Create a new issue with detailed description
- Include error messages and environment details

---

## Risk Disclaimer

**This system is for research and educational purposes only.** Past performance of signals and factors does not guarantee future results. Always conduct your own due diligence before making investment decisions. The authors are not responsible for any trading losses incurred from using this software.
