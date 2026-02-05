# SharpeOptimizer Usage Guide

## Overview

This guide shows you practical ways to use `SharpeOptimizer` for finding optimal factor weights. Choose the approach that best fits your workflow.

---

## Option 1: Command Line (Easiest)

### Quick Start

```bash
# Single-period optimization (default: 126-day lookback)
uv run python -m src optimize --universe SPY

# Custom lookback and methods
uv run python -m src optimize \
    --universe SPY QQQ \
    --lookback 63 \
    --methods sharpe momentum risk_parity \
    --technique differential

# Walk-forward optimization
uv run python -m src optimize \
    --universe SPY \
    --walk-forward \
    --train-window 126 \
    --test-window 21

# Export weights for trading
uv run python -m src optimize \
    --universe SPY \
    --export-weights optimal_weights.csv \
    --output optimization_results.json
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--universe` | Stock/ETF universe | SPY |
| `--lookback` | Lookback window (days) | 126 |
| `--methods` | Methods to blend | sharpe momentum risk_parity |
| `--technique` | Optimization algorithm | differential |
| `--walk-forward` | Enable walk-forward mode | False |
| `--train-window` | Training window (walk-forward) | 126 |
| `--test-window` | Test window (walk-forward) | 21 |
| `--output` | Save results to JSON | None |
| `--export-weights` | Export weights to CSV | None |

---

## Option 2: Python Script (For Automation)

### Basic Script

Create `run_optimization.py`:

```python
#!/usr/bin/env python3
"""Run factor weight optimization."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.factor_optimization import SharpeOptimizer
from src.research import FactorResearchSystem
from src.config import config

# Configuration
UNIVERSE = ["SPY"]
LOOKBACK = 126
METHODS = ['sharpe', 'momentum', 'risk_parity']
TECHNIQUE = 'differential'

# Load/generate factors
print("Loading factors...")
frs = FactorResearchSystem(
    config.ALPHAVANTAGE_API_KEY,
    universe=UNIVERSE,
    factor_method='fundamental',
    n_components=8
)
frs.fit_factors()

factor_returns = frs.get_factor_returns()
factor_loadings = frs._expos

# Optimize
print(f"Optimizing with {TECHNIQUE}...")
optimizer = SharpeOptimizer(factor_returns, factor_loadings)

result = optimizer.optimize_blend(
    lookback=LOOKBACK,
    methods=METHODS,
    technique=TECHNIQUE
)

# Print results
print(f"\nOptimal Sharpe: {result.sharpe_ratio:.2f}")
print(f"Optimal Weights: {result.optimal_weights}")

# Save results
import json
with open('optimal_weights.json', 'w') as f:
    json.dump(result.optimal_weights, f, indent=2)
```

Run it:
```bash
uv run python run_optimization.py
```

---

## Option 3: Jupyter Notebook (For Analysis)

### Interactive Exploration

Create `optimization_analysis.ipynb`:

```python
# Cell 1: Setup
import pandas as pd
import matplotlib.pyplot as plt
from src.factor_optimization import SharpeOptimizer
from src.research import FactorResearchSystem

# Cell 2: Load Data
frs = FactorResearchSystem(
    api_key="your_key",
    universe=["SPY"],
    factor_method="fundamental"
)
frs.fit_factors()

returns = frs.get_factor_returns()
loadings = frs._expos

# Cell 3: Single Optimization
optimizer = SharpeOptimizer(returns, loadings)
result = optimizer.optimize_blend(
    lookback=126,
    methods=['sharpe', 'momentum', 'risk_parity'],
    technique='differential'
)

print(f"Sharpe: {result.sharpe_ratio:.2f}")

# Cell 4: Visualize Weights
weights = pd.Series(result.optimal_weights)
weights.plot(kind='barh')
plt.title('Optimal Factor Weights')
plt.show()

# Cell 5: Walk-Forward Analysis
rolling = optimizer.walk_forward_optimize(
    train_window=126,
    test_window=21
)

# Plot rolling Sharpe
rolling.set_index('date')['test_sharpe'].plot()
plt.title('Out-of-Sample Sharpe Ratio Over Time')
plt.show()
```

Run notebook:
```bash
uv run jupyter notebook optimization_analysis.ipynb
```

---

## Option 4: Ad-Hoc Python (Quick Tests)

### Interactive Python Session

```bash
# Start interactive Python
uv run python
```

```python
>>> from src.factor_optimization import SharpeOptimizer
>>> import pandas as pd

# Load your data
>>> returns = pd.read_csv('factor_returns.csv', index_col=0, parse_dates=True)
>>> loadings = pd.read_csv('factor_loadings.csv', index_col=0)

# Quick optimization
>>> optimizer = SharpeOptimizer(returns, loadings)
>>> result = optimizer.optimize_blend(lookback=63)

>>> print(f"Sharpe: {result.sharpe_ratio:.2f}")
>>> print(result.optimal_weights)
```

---

## Common Workflows

### Daily Rebalancing Workflow

```bash
#!/bin/bash
# daily_rebalance.sh

cd /path/to/equity-factors

# Morning optimization
echo "Running factor optimization..."
uv run python -m src optimize \
    --universe SPY \
    --lookback 126 \
    --export-weights weights/today.csv

# Generate signals
echo "Generating signals..."
uv run python -m src signals cross-section \
    --universe SPY

# Send to trading system (example)
# python send_to_broker.py weights/today.csv
```

### Research Workflow (Notebook)

```python
# 1. Load data
# 2. Compare optimization techniques
# 3. Analyze rolling weights
# 4. Backtest with transaction costs
# 5. Export best configuration
```

### Production Workflow (Script)

```python
# 1. Load cached factors
# 2. Run walk-forward optimization
# 3. Generate cross-sectional signals
# 4. Export to trading system
# 5. Log results
```

---

## Choosing Your Approach

| Use Case | Recommended Approach | Why |
|----------|---------------------|-----|
| Quick test | CLI | Fastest, no setup |
| Daily automation | Bash script + cron | Set and forget |
| Research/analysis | Jupyter notebook | Visualizations, exploration |
| Production trading | Python script | Error handling, logging |
| Ad-hoc analysis | Interactive Python | Flexibility |

---

## Output Examples

### CLI Output

```
======================================================================
 FACTOR WEIGHT OPTIMIZATION
======================================================================

 Loading factor data for universe: SPY
 Loading cached factors from factor_cache_SPY_fundamental.pkl
   Factor returns shape: (252, 5)
   Factor loadings shape: (100, 5)

  Running single-period optimization...
   Lookback: 126 days
   Methods: sharpe, momentum, risk_parity
   Technique: differential

Running differential evolution with 3 methods...

Optimal blend found:
  sharpe: 67.8%
  momentum: 25.7%
  risk_parity: 6.5%

Sharpe Ratio: 2.45
Annualized Return: 18.32%
Annualized Volatility: 7.48%

======================================================================
OPTIMIZATION RESULTS
======================================================================

 Performance Metrics:
   Sharpe Ratio:          2.45
   Annualized Return:     18.32%
   Annualized Volatility: 7.48%

 Optimal Method Blend:
   sharpe                 67.8% 
   momentum               25.7% 
   risk_parity             6.5% 

 Optimal Factor Weights:
   F1_Value               45.2% 
   F2_Momentum            28.1% 
   F3_Quality             15.3% 
   F4_LowVol               8.9% 
   F5_Growth               2.5% 

 Weights exported to: optimal_weights.csv
```

### Exported CSV Format

```csv
factor,weight
F1_Value,0.452
F2_Momentum,0.281
F3_Quality,0.153
F4_LowVol,0.089
F5_Growth,0.025
```

---

## Tips

1. **Start with CLI** for quick tests
2. **Use walk-forward** for more robust results
3. **Export weights** for use in trading systems
4. **Cache factors** to speed up repeated runs
5. **Try different techniques** to compare results
6. **Monitor out-of-sample Sharpe** to detect overfitting
