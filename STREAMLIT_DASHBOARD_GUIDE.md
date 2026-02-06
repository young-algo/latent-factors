# Streamlit Dashboard Guide

## Running the dashboard

From the project root:

```bash
# Recommended
./scripts/launch_dashboard.sh

# Or manually
streamlit run src/dashboard.py
```

## Navigation

- **Home**: morning briefing + alerts + movers (default landing page)
- **Factors**: factor DNA table + factor drill-down (X-ray)
- **Signals**: regime, momentum, extremes, meta-model
- **Portfolio & Risk**: portfolio inputs, beta/TE/IR, attribution, basket preview
- **Research**: discovery, QA, optimization/backtests, optional LLM enrichment
- **Settings**: health checks, preferences, watchlists

## Required artifacts

Most pages require these files in the project root:
- `factor_returns.csv`
- `factor_loadings.csv`

Optional (enables true PM vitals):
- `portfolio_returns.csv`
- `benchmark_returns.csv`

## Phase 2 features (where to find them)

- **Regime conditional optimization (RS-MVO):** **Research**
- **Meta-model signal aggregation:** **Signals → Meta-model**

