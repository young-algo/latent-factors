# Equity Factors Research System

A comprehensive quantitative finance research platform for factor discovery, analysis, and backtesting. This system combines market data acquisition, factor modeling, and performance evaluation with innovative LLM-powered factor naming. Designed for active traders who want to understand what's *actually* driving returns in their portfolios.

## Overview

This system helps active traders answer critical questions:
- **What hidden factors are driving my universe?** (Factor Discovery)
- **Is now a good time to buy/sell specific factors?** (Trading Signals)
- **Which stocks should I long/short based on factor exposures?** (Cross-Sectional Analysis)
- **What market regime are we in and how should I position?** (Regime Detection)
- **Would my strategy have worked historically?** (Backtesting)

## Key Features

| Feature | What It Does | Why Traders Care |
|---------|--------------|------------------|
| **Factor Discovery** | Extracts latent factors using PCA, ICA, NMF, or Autoencoders | Understand the true drivers of returns in your universe beyond obvious sector ETFs |
| **LLM Factor Naming** | Auto-generates intuitive names using GPT models | Transform cryptic "F3" into "Small-Cap Growth Momentum" for actionable insights |
| **Trading Signals** | RSI, MACD, ADX, extreme value alerts, z-scores | Know when factors are overbought/oversold for entry/exit timing |
| **Cross-Sectional Analysis** | Rank stocks by composite factor scores | Generate quantifiable long/short candidates from your universe |
| **Regime Detection** | HMM-based market state identification | Adjust factor exposure based on bull/bear/volatile conditions |
| **Signal Backtesting** | Walk-forward testing with performance attribution | Validate your factor timing strategy before risking capital |
| **Interactive Dashboard** | Streamlit UI for real-time monitoring | Visual factor monitoring without writing code |

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

### 1. Factor Discovery & Naming

**What it does:** Automatically discovers latent factors in any ETF or stock universe and gives them meaningful names.

**Why use it:** Traditional ETFs (XLK, XLF, etc.) tell you what sector a stock is in, but not what's *actually* driving returns. This reveals hidden patterns like "High Beta Disruptors" or "Defensive Dividend Growers."

```bash
# Discover factors in SPY using PCA
uv run python src/discover_and_label.py -s "SPY" --method PCA -k 10

# Analyze tech sector with 10 factors
uv run python src/discover_and_label.py -s "XLK" --method PCA -k 10

# Multi-ETF analysis for broader factor discovery
uv run python src/discover_and_label.py -s "SPY,QQQ,IWM" --method ICA -k 20

# Use deep learning for non-linear factors (requires PyTorch)
uv run python src/discover_and_label.py -s "ARKK" --method AE -k 5
```

**Output:**
- `factor_names.csv` - Human-readable factor names
- `factor_returns.csv` - Daily factor return time series
- `factor_loadings.csv` - Stock exposures to each factor
- `cumulative_factor_returns.png` - Visual performance chart

**Trading Application:**
- Identify which factors are currently outperforming
- Understand factor correlations to avoid concentration risk
- Use factor returns as features in your own models

---

### 2. Trading Signals CLI

**What it does:** Generates actionable buy/sell signals based on factor momentum, extreme values, and cross-sectional rankings.

**Why use it:** Instead of guessing when to enter/exit, get data-driven signals with confidence scores.

```bash
# Generate comprehensive trading signals
uv run python src/signals_cli.py generate --universe SPY --method PCA --components 10

# Check for extreme value alerts (mean reversion opportunities)
uv run python src/signals_cli.py extremes --universe SPY --threshold 2.0 --trade

# Analyze factor momentum (RSI, MACD, ADX)
uv run python src/signals_cli.py momentum --universe SPY --factor F1

# Cross-sectional stock rankings (long/short candidates)
uv run python src/signals_cli.py cross-section --universe SPY --top-pct 0.1 --bottom-pct 0.1

# Detect market regime with allocation recommendations
uv run python src/signals_cli.py regime --universe SPY --regimes 3 --predict 5

# Backtest signals with walk-forward analysis
uv run python src/signals_cli.py backtest --universe SPY --walks 10 --optimize

# Generate HTML report
uv run python src/signals_cli.py report --universe SPY --output report.html --open

# Interactive factor naming
uv run python src/signals_cli.py name-factors --universe SPY --interactive
```

**Signal Types Explained:**

| Signal Type | Best For | When to Use |
|-------------|----------|-------------|
| **Momentum (RSI)** | Timing mean reversion | RSI > 70 = overbought (consider short); RSI < 30 = oversold (consider long) |
| **MACD Crossover** | Trend confirmation | Bullish crossover = enter long; Bearish = exit/short |
| **ADX** | Trend strength | ADX > 25 = strong trend (follow); ADX < 20 = weak trend (mean revert) |
| **Z-Score** | Statistical extremes | |z| > 2 = rare event, likely to revert |
| **Percentile Rank** | Historical context | 95th+ percentile = extreme high; 5th- percentile = extreme low |

---

### 3. Interactive Dashboard

**What it does:** Real-time visual monitoring of factors, signals, and regimes through a web interface.

**Why use it:** Get a trading desk view without writing any code. Monitor multiple factors simultaneously.

```bash
# Launch the Streamlit dashboard
uv run python src/signals_cli.py dashboard
```

**Dashboard Sections:**
- **Market Overview** - Factor count, date range, average returns, active alerts
- **Signal Status Table** - Color-coded buy/sell signals with RSI/MACD/ADX
- **Extreme Value Alerts** - Red/green alerts for statistical extremes
- **Cumulative Returns** - Factor performance over time
- **Factor Deep Dive** - Individual factor analysis with top exposures
- **Stock Watchlist** - Factor exposure heatmap for selected stocks
- **Regime Analysis** - Current market state and predictions

**Trading Application:**
- Morning pre-trade briefing: Check overnight signal changes
- Intraday: Monitor factor momentum for timing
- End of day: Review regime status for next-day positioning

---

### 4. Cross-Sectional Analysis

**What it does:** Ranks all stocks in your universe by composite factor scores to identify long/short candidates.

**Why use it:** Instead of picking stocks based on gut feel, rank them quantitatively by factor exposures.

```python
from src.cross_sectional import CrossSectionalAnalyzer
import pandas as pd

# Load factor loadings
loadings = pd.read_csv("factor_loadings.csv", index_col=0)

# Initialize analyzer
analyzer = CrossSectionalAnalyzer(loadings)

# Calculate composite scores with custom weights
weights = {'F1': 0.4, 'F2': 0.3, 'F3': 0.3}  # Overweight specific factors
scores = analyzer.calculate_factor_scores(weights=weights)

# Generate rankings
rankings = analyzer.rank_universe(scores)

# Get long/short signals
signals = analyzer.generate_long_short_signals(top_pct=0.1, bottom_pct=0.1)

# Filter for actionable signals
longs = [s for s in signals if s.direction.value == 'long']
shorts = [s for s in signals if s.direction.value == 'short']
```

**Trading Application:**
- Build quant-style long/short portfolios
- Identify factor tilts for existing positions
- Screen universe for factor-based entry candidates

---

### 5. Market Regime Detection

**What it does:** Uses Hidden Markov Models to identify market regimes (low vol bull, high vol bear, crisis, etc.) and recommends factor allocations.

**Why use it:** Factors perform differently in different regimes. Momentum works in trending markets; mean reversion works in choppy markets.

```python
from src.regime_detection import RegimeDetector
import pandas as pd

# Load factor returns
returns = pd.read_csv("factor_returns.csv", index_col=0, parse_dates=True)

# Initialize and fit regime detector
detector = RegimeDetector(returns)
detector.fit_hmm(n_regimes=3)

# Detect current regime
current = detector.detect_current_regime()
print(f"Current: {current.regime.value} (confidence: {current.probability:.1%})")
print(f"Description: {current.description}")

# Get allocation recommendation
allocation = detector.generate_regime_signals()
print(f"Risk-on score: {allocation.risk_on_score:.2f}")
print(f"Recommended action: {allocation.recommended_action}")

# Predict future regimes
predictions = detector.predict_regime(duration=5)
```

**Regime Types:**

| Regime | Characteristics | Optimal Strategy |
|--------|-----------------|------------------|
| **Low Vol Bull** | Rising prices, low volatility | Overweight momentum, growth factors |
| **High Vol Bull** | Rising but choppy | Reduce size, maintain momentum with quality tilt |
| **Low Vol Bear** | Declining prices, low volatility | Defensive positioning, value and quality factors |
| **High Vol Bear** | Declining, choppy | Maximum defensive, minimum volatility |
| **Crisis** | Extreme volatility, correlations → 1 | Risk-off, cash/quality focus |
| **Transition** | Mixed signals | Maintain diversification, await clarity |

**Trading Application:**
- Adjust factor exposure before regime shifts
- Avoid momentum crashes during volatile bear markets
- Position for style rotation (value vs growth, large vs small)

---

### 6. Signal Backtesting

**What it does:** Validates signal efficacy using walk-forward testing with performance attribution.

**Why use it:** Before deploying a factor strategy, verify it would have worked historically. Avoid curve-fitting.

```python
from src.signal_backtest import SignalBacktester
from src.signal_aggregator import SignalAggregator
from src.trading_signals import FactorMomentumAnalyzer
from src.cross_sectional import CrossSectionalAnalyzer
from src.regime_detection import RegimeDetector
import pandas as pd

# Load data
returns = pd.read_csv("factor_returns.csv", index_col=0, parse_dates=True)
loadings = pd.read_csv("factor_loadings.csv", index_col=0)

# Build signal aggregator
aggregator = SignalAggregator(None)
momentum = FactorMomentumAnalyzer(returns)
cross = CrossSectionalAnalyzer(loadings)
regime = RegimeDetector(returns)
regime.fit_hmm(n_regimes=3)

aggregator.add_momentum_signals(momentum)
aggregator.add_cross_sectional_signals(cross)
aggregator.add_regime_signals(regime)

# Run backtest
backtester = SignalBacktester(aggregator, returns)
results = backtester.run_backtest(
    train_size=252,    # 1 year training
    test_size=63,      # 3 month test
    n_walks=10,        # 10 walk-forward periods
    min_confidence=70  # Only trade high-confidence signals
)

print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
print(f"Hit Rate: {results['hit_rate']:.1%}")

# Optimize signal thresholds
optimal = backtester.optimize_thresholds(metric='sharpe_ratio')
print(f"Optimal confidence threshold: {optimal['threshold']:.1f}")
```

**Key Metrics:**
- **Hit Rate** - Percentage of winning signals
- **Profit Factor** - Gross profits / gross losses (>1.5 is good)
- **Win/Loss Ratio** - Average winner / average loser
- **Calmar Ratio** - Annualized return / max drawdown

---

### 7. Factor Momentum Analysis

**What it does:** Applies technical analysis to factor returns instead of individual stocks.

**Why use it:** Factors exhibit momentum and mean reversion just like stocks. Timing factor entries can significantly improve returns.

```python
from src.trading_signals import FactorMomentumAnalyzer
import pandas as pd

returns = pd.read_csv("factor_returns.csv", index_col=0, parse_dates=True)
analyzer = FactorMomentumAnalyzer(returns)

# Calculate indicators for a factor
rsi = analyzer.calculate_rsi('F1')
macd, signal, hist = analyzer.calculate_macd('F1')
adx = analyzer.calculate_adx('F1')
zscore = analyzer.calculate_zscore('F1')

# Detect momentum regime
regime = analyzer.detect_momentum_regime('F1')
print(f"Current regime: {regime.value}")

# Get all signals
signals = analyzer.get_all_signals()

# Check for extremes
alerts = analyzer.get_all_extreme_alerts(z_threshold=2.0)
for alert in alerts:
    print(f"{alert.factor_name}: {alert.direction} (z={alert.z_score:.2f})")
```

**Trading Application:**
- Time factor ETF entries (e.g., MTUM for momentum, VLUE for value)
- Avoid chasing factors that are statistically extended
- Identify factor rotation opportunities

---

## Project Structure

```
equity-factors/
├── src/
│   ├── alphavantage_system.py      # Data backend with caching and ETF expansion
│   ├── research.py                 # Main research system and factor models
│   ├── latent_factors.py           # Factor discovery (PCA, ICA, NMF, Autoencoder)
│   ├── factor_labeler.py           # LLM-powered factor naming with OpenAI
│   ├── factor_naming.py            # Factor naming data structures and validation
│   ├── discover_and_label.py       # Main CLI workflow for factor discovery
│   ├── trading_signals.py          # Technical indicators for factor returns
│   ├── cross_sectional.py          # Cross-sectional ranking and stock selection
│   ├── regime_detection.py         # HMM-based market regime identification
│   ├── signal_aggregator.py        # Combines multiple signal types
│   ├── signal_backtest.py          # Walk-forward backtesting framework
│   ├── signals_cli.py              # Trading signals command-line interface
│   ├── dashboard.py                # Streamlit interactive dashboard
│   ├── reporting.py                # HTML/Markdown report generation
│   ├── factor_performance_table.py # Performance analytics
│   ├── cli.py                      # Main CLI entry point
│   └── runner.py                   # Legacy entry point
├── tests/                          # Unit tests
├── notebooks/                      # Jupyter analysis notebooks
├── data/                           # Data storage
├── av_cache.db                     # SQLite cache for API data
├── factor_returns.csv              # Generated factor returns
├── factor_loadings.csv             # Generated factor loadings
├── factor_names.csv                # Generated factor names
├── pyproject.toml                  # Project dependencies
├── .env.example                    # Environment variable template
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
```

### Factor Methods

**For `discover_and_label.py`:**
- `PCA` - Fast, orthogonal factors; best for most use cases
- `ICA` - Statistically independent; good for regime analysis
- `NMF` - Non-negative; slower but interpretable for portfolio construction
- `AE` - Deep autoencoder; captures non-linear relationships (requires PyTorch)

**For `research.py`:**
- `fundamental` - Cross-sectional factor model using fundamental data
- `pca` - Principal Component Analysis on returns

---

## API Rate Limits

The system implements intelligent rate limiting:
- Price data: 75 calls/minute
- Fundamentals: 75 calls/minute
- Automatic throttling and progress tracking

**Cost Optimization:**
- SQLite caching minimizes repeated API calls
- First run will be slower; subsequent runs use cached data
- Free Alpha Vantage tier allows 25 calls/day (premium recommended for production)

---

## Trading Workflow Examples

### Daily Pre-Market Routine

```bash
# 1. Check overnight regime status
uv run python src/signals_cli.py regime --universe SPY

# 2. Review extreme value alerts for mean reversion opportunities
uv run python src/signals_cli.py extremes --universe SPY --trade

# 3. Generate cross-sectional rankings for stock selection
uv run python src/signals_cli.py cross-section --universe SPY --top-pct 0.05

# 4. Launch dashboard for visual monitoring
uv run python src/signals_cli.py dashboard
```

### Weekly Strategy Review

```bash
# 1. Run comprehensive backtest
uv run python src/signals_cli.py backtest --universe SPY --walks 20 --optimize

# 2. Generate full HTML report
uv run python src/signals_cli.py report --universe SPY --output weekly_report.html

# 3. Re-discover factors if market structure has changed
uv run python src/discover_and_label.py -s "SPY" --method PCA -k 10
```

### Factor Rotation Strategy

```python
# Detect regime and adjust factor exposure
from src.regime_detection import RegimeDetector
from src.trading_signals import FactorMomentumAnalyzer
import pandas as pd

returns = pd.read_csv("factor_returns.csv", index_col=0, parse_dates=True)

# Check current regime
detector = RegimeDetector(returns)
detector.fit_hmm(n_regimes=3)
regime = detector.detect_current_regime()

# Get optimal factor weights for current regime
weights = detector.get_regime_optimal_factors(regime.regime)

# Check factor momentum for timing
analyzer = FactorMomentumAnalyzer(returns)
for factor in returns.columns:
    signals = analyzer.get_momentum_signals(factor)
    # Only trade factors with strong momentum in regime-appropriate direction
    if signals['combined_signal'] in ['strong_buy', 'buy']:
        print(f"Consider long {factor} (weight: {weights.get(factor, 0):.2%})")
```

---

## Performance Notes

- **NMF Performance:** For large ETFs like IWM (2000+ stocks), NMF can be slow. Use PCA or ICA for faster results.
- **Autoencoder:** Requires PyTorch and benefits from GPU acceleration for large universes.
- **Caching:** First run downloads all data; subsequent runs are near-instant using SQLite cache.
- **Memory:** Large universes (1000+ stocks) require ~2GB RAM for factor computation.

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
