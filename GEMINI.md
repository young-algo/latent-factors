# Equity Factors Research System

## Project Overview
**Equity Factors** is a quantitative finance research platform designed for factor discovery, optimization, and backtesting. It enables users to discover latent alpha factors in stock universes, optimize factor weights using Bayesian techniques, and generate tradeable baskets. It features an institutional-grade dashboard ("Alpha Command Center") and an LLM-powered "Morning Briefing" decision synthesizer.

**Key Technologies:** Python 3.11+, Pandas, Scikit-learn, PyTorch, Streamlit, Optuna, HMMLearn, Alpha Vantage API, OpenAI API.

## Environment & Setup

### Prerequisites
*   **Python:** >= 3.11
*   **Package Manager:** `uv` (recommended) or `pip`
*   **API Keys:** Alpha Vantage (Market Data), OpenAI (Factor Naming/Analysis)

### Configuration
The project relies on environment variables defined in a `.env` file.
```bash
cp .env.example .env
# Edit .env to add ALPHAVANTAGE_API_KEY and OPENAI_API_KEY
```

### Installation
```bash
# Using uv (Recommended)
uv sync

# Using pip
pip install -e .
```

## Architecture & Directory Structure

*   `src/`: Main source code.
    *   `__main__.py`: CLI entry point.
    *   `alphavantage_system.py`: Data backend and caching (SQLite).
    *   `latent_factors.py`: Factor discovery engines (PCA, ICA, NMF).
    *   `factor_optimization.py`: Bayesian optimization for factor weights.
    *   `decision_synthesizer.py`: Logic for Morning Briefing.
    *   `dashboard_alpha_command_center.py` & `dashboard.py`: Streamlit dashboards.
    *   `database.py`: SQLite connection management.
*   `tests/`: Unit and integration tests (Pytest).
*   `scripts/`: Utility scripts and launch helpers.
*   `notebooks/`: Research and prototyping notebooks.
*   `data/`: Local data storage (if applicable, though SQLite is primary).

## Key Workflows & Commands

All commands are accessible via the `src` module. Prefix with `uv run` if using `uv`, or just `python` if the venv is active.

### 1. Factor Discovery
Discover latent factors in a universe (e.g., SPY, VTHR).
```bash
python -m src discover --symbols SPY --method PCA -k 10
python -m src discover --symbols VTHR --method ICA -k 8
```

### 2. Optimization
Optimize factor weights to maximize metrics like Sharpe Ratio.
```bash
python -m src optimize --universe SPY --factor-method pca --technique bayesian
```

### 3. Basket Generation
Convert optimized weights into a tradeable stock basket.
```bash
python -m src basket --results optimization_results.json --universe SPY --capital 100000
```

### 4. Trading Signals & Analysis
Generate technical signals or cross-sectional rankings.
```bash
python -m src signals extremes --universe SPY --trade
python -m src signals cross-section --universe SPY --top-pct 0.1
```

### 5. Morning Briefing
Generate a synthesized decision report with conviction scores.
```bash
python -m src briefing --universe SPY --method pca
```

### 6. Dashboards
Launch the interactive web interfaces.
```bash
# Alpha Command Center (PM Dashboard)
./scripts/launch_alpha_command_center.sh
# OR
python -m src dashboard
```

## Testing & Quality Assurance

The project uses `pytest` for testing.
```bash
# Run all tests
pytest

# Run specific tests for PIT (Point-in-Time) universe verification
pytest tests/test_pit_universe.py
```

## Data Management
*   **Database:** Uses SQLite for caching API responses to minimize costs and latency.
*   **Rate Limiting:** Built-in throttling for Alpha Vantage (75 calls/min).
*   **Maintenance:** Use `python -m src clean` to manage cache if needed.
