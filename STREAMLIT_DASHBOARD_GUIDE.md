# Streamlit Dashboard Guide - Phase 2 Integration

## Overview

The Alpha Command Center dashboard has been enhanced with Phase 2 capabilities:

1. ** Regime RS-MVO Tab** - Conditional optimization based on market regimes
2. ** Meta-Model Tab** - XGBoost-based signal aggregation with walk-forward training

## Running the Dashboard

```bash
# Standard dashboard
cd /Users/kevinturner/Documents/Code/equity-factors
streamlit run src/dashboard.py

# Alpha Command Center (with Phase 2 features)
streamlit run src/dashboard_alpha_command_center.py
```

## Phase 2 Features

### 1.  Regime Conditional Optimization (RS-MVO)

**What it does:**
- Filters historical data by market regime (e.g., Low Vol Bull, High Vol Bear)
- Computes optimal factor weights CONDITIONAL on the current regime
- Uses Ledoit-Wolf shrinkage for covariance stability
- Compares data-driven weights vs heuristic (rules-based) weights

**How to use:**
1. Navigate to the " Regime RS-MVO" tab
2. Configure parameters:
   - **Lookback Window**: Historical period to analyze (default: 2520 days ~ 10 years)
   - **Min Observations**: Minimum regime-specific samples needed (default: 50)
   - **Target Regime**: Select which regime to optimize for
3. Click " Run RS-MVO Optimization"
4. Review the weight comparison charts

**Key Outputs:**
- RS-MVO weights (data-driven)
- Heuristic weights (rules-based fallback)
- Weight differences showing which factors get more/less weight

### 2.  Meta-Model Signal Aggregation

**What it does:**
- Trains XGBoost classifier on historical signal features
- Predicts probability of positive forward returns
- Uses walk-forward cross-validation to prevent overfitting
- Extracts features from momentum, regime, and cross-sectional signals

**How to use:**
1. Navigate to the " Meta-Model" tab
2. Configure training parameters:
   - **Min Training Samples**: Default 252 (~1 year)
   - **Prediction Horizon**: Forward return horizon (default: 5 days)
   - **Advanced**: Model hyperparameters (n_estimators, max_depth, etc.)
3. Select market returns source:
   - Use factor mean as proxy, OR
   - Upload market returns CSV
4. Click " Train Meta-Model (Walk-Forward)"
5. Once trained, click "Generate Meta-Consensus Signal" for predictions

**Key Outputs:**
- Feature importance chart (which signals matter most)
- Probability of positive return gauge
- Meta-score (-100 to +100)
- Trading recommendation based on model prediction

## Feature Extraction

The meta-model automatically extracts features from:

**Momentum Signals:**
- RSI (normalized to -1 to 1)
- MACD signal encoding (-1 to 1)
- ADX trend strength (0 to 1)
- Combined signal strength

**Regime Signals:**
- Regime probability
- Volatility level
- Trend strength
- Regime type probabilities

**Cross-Sectional Signals:**
- Average decile scores
- Long/short ratio
- Confidence metrics

## System Requirements

### For RS-MVO:
- Fitted RegimeDetector with HMM model
- Factor returns data
- Sufficient historical data (recommend 1000+ days)

### For Meta-Model:
- **XGBoost** (optional but recommended):
  ```bash
  pip install xgboost
  # or
  uv add xgboost
  ```
- Market returns for label generation
- Minimum 252 days of training data

## Troubleshooting

### "Regime detector not initialized"
- Run factor discovery first to generate returns data
- Ensure HMM is fitted (happens automatically on startup)

### "XGBoost not installed"
- The system will use voting fallback
- Install XGBoost for full gradient boosting capabilities

### "Insufficient training samples"
- Increase lookback window in RS-MVO
- Reduce min_training_samples for meta-model
- Ensure sufficient historical data is available

### "Market returns required"
- Enable "Use Factor Mean as Market Proxy" checkbox, OR
- Upload a CSV with market returns (e.g., SPY)

## Dashboard Navigation

| Tab | Purpose |
|-----|---------|
|  Factor Lab | Factor discovery, naming, and X-ray analysis |
|  Portfolio Constructor | Optimization sandbox and trade basket generation |
|  Regime RS-MVO | **Phase 2** - Conditional regime-based optimization |
|  Meta-Model | **Phase 2** - ML-based signal aggregation |
|  Risk & Drawdown | Risk metrics and drawdown monitoring |

## Version History

- **v2.0**: Original Alpha Command Center
- **v2.1**: Phase 2 Integration (RS-MVO + Meta-Model)

## API Endpoints (for programmatic access)

The underlying functionality is also available via CLI:

```bash
# Regime detection
python -m src regime --fit-hmm

# Optimization
python -m src optimize --lookback 252

# Meta-model training (custom script)
python scripts/train_meta_model.py
```

## Support

For issues or questions:
1. Check the main README.md
2. Review PHASE2_IMPLEMENTATION_SUMMARY.md
3. Run tests: `python -m pytest tests/test_phase2_enhancements.py -v`
