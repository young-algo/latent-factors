# Optimal Factor Weighting Guide

## Overview

This guide explains optimal methods for determining factor weights in cross-sectional portfolios, moving beyond ad hoc weighting (e.g., "50% F2, 30% F1, 20% F9") to principled approaches based on factor characteristics.

## The Problem with Ad Hoc Weighting

**Ad hoc weighting has several drawbacks:**
1. **No risk consideration** - High-volatility factors get same weight as stable ones
2. **No performance consideration** - Underperforming factors keep their weights
3. **No predictive power consideration** - Factors with poor IC get same weight
4. **Static** - Doesn't adapt to changing market conditions
5. **Arbitrary** - Weights chosen by intuition rather than data

## Optimal Weighting Methods

### 1. Sharpe Ratio Weighting  **RECOMMENDED BASELINE**

**What it does:** Weights factors by their risk-adjusted returns.

**Formula:**
```
Sharpe_i = (mean_return_i - risk_free) / std_i
w_i = max(0, Sharpe_i - min_threshold) / Σmax(0, Sharpe_j - min_threshold)
```

**Best for:**
- Long-term strategic allocation
- Balancing return and risk
- Avoiding factors with poor risk-adjusted performance

**Pros:**
-  Simple and interpretable
-  Penalizes volatile underperformers
-  Can require positive Sharpe (min_threshold)

**Cons:**
-  Ignores correlation between factors
-  Assumes past Sharpe predicts future

**Usage:**
```python
weights = weighter.sharpe_weights(lookback=63, min_sharpe=0.5)
```

---

### 2. Information Coefficient (IC) Weighting  **BEST FOR SELECTION**

**What it does:** Weights factors by their predictive power (correlation with forward returns).

**Formula:**
```
IC_i = Corr(exposure_i, forward_returns)
w_i = |IC_i| / Σ|IC_j|
```

**Best for:**
- Stock selection within a universe
- Identifying truly predictive factors
- Factor timing

**Pros:**
-  Directly measures predictive power
-  Robust to outliers (Spearman rank)
-  Data-driven

**Cons:**
-  Requires forward returns (can't use for pure historical analysis)
-  IC can be unstable over time

**Usage:**
```python
# forward_returns = next_month_returns for each stock
weights = weighter.ic_weights(forward_returns, method='spearman')
```

---

### 3. Factor Momentum Weighting  **BEST FOR TACTICAL**

**What it does:** Overweights factors that have been performing well recently.

**Formula:**
```
Momentum_i = Π(1 + r_i,t) - 1  [cumulative return]
w_i = max(0, momentum_i) / Σmax(0, momentum_j)
```

**Best for:**
- Tactical factor tilts
- Capturing factor trends
- Adapting to regime changes

**Pros:**
-  Dynamic and adaptive
-  Can capture factor trends
-  Multiple calculation methods (absolute, relative, vol-adjusted)

**Cons:**
-  Can chase performance
-  Requires frequent rebalancing
-  May increase turnover

**Usage:**
```python
weights = weighter.momentum_weights(lookback=21, method='vol_adjusted')
```

---

### 4. Risk Parity Weighting  **BEST FOR RISK MANAGEMENT**

**What it does:** Each factor contributes equally to portfolio risk.

**Formula:**
```
Solve: w_i × (Σw)_i = constant for all i
Where Σ is the factor covariance matrix
```

**Best for:**
- Risk-controlled portfolios
- Preventing concentration in volatile factors
- Balanced risk contribution

**Pros:**
-  Equal risk contribution
-  Penalizes high-volatility factors
-  Accounts for correlations

**Cons:**
-  Requires covariance estimation
-  Complex optimization
-  May overweight low-return factors

**Usage:**
```python
weights = weighter.risk_parity_weights(lookback=63)
```

---

### 5. Minimum Variance Weighting

**What it does:** Minimizes portfolio variance subject to constraints.

**Formula:**
```
min: w' Σ w
s.t.: Σw_i = 1, w_i ≥ 0
```

**Best for:**
- Conservative portfolios
- Volatility reduction
- Defensive positioning

**Pros:**
-  Lowest possible portfolio volatility
-  Analytically optimal
-  Works well with risk parity

**Cons:**
-  Ignores returns completely
-  May concentrate in few low-vol factors
-  Poor when low-vol underperforms

**Usage:**
```python
weights = weighter.min_variance_weights(lookback=63)
```

---

### 6. Maximum Diversification Weighting

**What it does:** Maximizes the diversification ratio.

**Formula:**
```
DR = (w'σ) / √(w'Σw)
maximize DR
```

**Best for:**
- Maximizing diversification benefits
- Portfolios with correlated factors
- Reducing concentration risk

**Pros:**
-  Maximizes diversification
-  Accounts for correlations
-  Better than equal weight for correlated factors

**Cons:**
-  Complex optimization
-  May not align with return objectives
-  Requires reliable covariance estimates

**Usage:**
```python
weights = weighter.max_diversification_weights(lookback=63)
```

---

### 7. PCA Weighting

**What it does:** Weights by contribution to variance via Principal Component Analysis.

**Formula:**
```
Run PCA on factor loadings
Weight_i = sum of squared loadings on significant PCs
```

**Best for:**
- Understanding factor structure
- Reducing dimensionality
- Statistical factor importance

**Pros:**
-  Data-driven importance
-  Handles collinearity
-  Dimensionality reduction

**Cons:**
-  May not align with returns
-  PCA factors not interpretable
-  Sensitive to scaling

**Usage:**
```python
weights = weighter.pca_weights(variance_threshold=0.95)
```

---

### 8. Blended Weighting  **RECOMMENDED FOR PRODUCTION**

**What it does:** Combines multiple methods using a weighted average.

**Formula:**
```
w_blend = Σ(method_weight_i × w_method_i)
```

**Best for:**
- Robust weighting that combines strengths
- Reducing single-method risk
- Production portfolios

**Pros:**
-  Combines benefits of multiple methods
-  More stable than single method
-  Customizable to objectives

**Cons:**
-  More complex
-  Requires tuning blend weights
-  May dilute strong signals

**Usage:**
```python
weights = weighter.blend_weights(
    {
        'sharpe': 0.4,
        'ic': 0.3,
        'momentum': 0.2,
        'risk_parity': 0.1
    },
    forward_returns=fwd_rets
)
```

---

## Method Comparison

| Method | Considers Returns | Considers Risk | Considers Correlation | Dynamic | Complexity |
|--------|-------------------|----------------|------------------------|---------|------------|
| Equal |  |  |  |  | Low |
| Sharpe |  |  |  |  | Low |
| IC |  |  |  |  | Low |
| Momentum |  |  |  |  | Low |
| Risk Parity |  |  |  |  | High |
| Min Variance |  |  |  |  | Medium |
| Max Diversification |  |  |  |  | High |
| PCA |  |  |  |  | Medium |
| **Blended** |  |  |  |  | Medium |

---

## Recommended Workflows

### For Long-Term Strategic Allocation
```python
# 60% Sharpe (risk-adjusted performance)
# 40% Risk Parity (balanced risk)
weights = weighter.blend_weights(
    {'sharpe': 0.6, 'risk_parity': 0.4},
    lookback=252
)
```

### For Tactical Factor Timing
```python
# 50% Momentum (trend following)
# 30% Sharpe (quality check)
# 20% IC (predictive power)
weights = weighter.blend_weights(
    {'momentum': 0.5, 'sharpe': 0.3, 'ic': 0.2},
    lookback=63,
    forward_returns=fwd_rets
)
```

### For Conservative/Defensive Portfolios
```python
# 50% Min Variance (low vol)
# 30% Risk Parity (balanced risk)
# 20% Sharpe (some return focus)
weights = weighter.blend_weights(
    {'min_variance': 0.5, 'risk_parity': 0.3, 'sharpe': 0.2},
    lookback=126
)
```

### For Aggressive Growth Portfolios
```python
# 50% IC (predictive power)
# 30% Momentum (trend)
# 20% Sharpe (risk check)
weights = weighter.blend_weights(
    {'ic': 0.5, 'momentum': 0.3, 'sharpe': 0.2},
    forward_returns=fwd_rets
)
```

---

## Practical Tips

### 1. Lookback Period Selection
- **Short (21 days):** Fast adaptation, more noise
- **Medium (63 days):** Balanced, ~3 months
- **Long (252 days):** Stable, ~1 year

### 2. Rebalancing Frequency
- **Daily:** High turnover, captures rapid changes
- **Weekly:** Balanced
- **Monthly:** Lower turnover, captures trends
- **Quarterly:** Strategic, low turnover

### 3. Handling Negative Weights
Most methods set negative weights to zero (no shorting factors). If you want factor neutrality:
```python
# Allow negative weights (factor long/short)
weights = weighter.sharpe_weights(min_sharpe=-10)  # No threshold
```

### 4. Dealing with Missing Forward Returns
If you don't have forward returns for IC weighting:
```python
# Use only methods that don't require forward returns
weights = weighter.blend_weights(
    {'sharpe': 0.5, 'momentum': 0.3, 'risk_parity': 0.2}
)
```

### 5. Monitoring and Debugging
```python
# Get factor characteristics
chars = weighter.get_factor_characteristics(lookback=63)

for name, char in chars.items():
    print(f"{name}: Sharpe={char.sharpe_ratio:.2f}, "
          f"IC={char.information_coefficient:.3f}")
```

---

## Example Output

```
============================================================
              Weight Comparison by Method
============================================================
Method                      Value     Mom     Qual    LowV    Grow
------------------------------------------------------------
Equal Weight                20.00%  20.00%  20.00%  20.00%  20.00%
Sharpe Ratio                28.57%   5.00%  32.14%  28.57%   5.71%
Information Coeff.          25.00%  35.00%  20.00%  10.00%  10.00%
Factor Momentum              5.00%  45.00%  15.00%   5.00%  30.00%
Risk Parity                 22.00%  15.00%  25.00%  30.00%   8.00%
Minimum Variance            15.00%   8.00%  20.00%  45.00%  12.00%
Max Diversification         20.00%  18.00%  22.00%  28.00%  12.00%
PCA                         23.00%  19.00%  21.00%  24.00%  13.00%
============================================================
```

---

## Summary

| Goal | Recommended Method | Blend Suggestion |
|------|-------------------|------------------|
| Maximize risk-adjusted returns | Sharpe | 60% Sharpe + 40% Momentum |
| Best stock selection | IC | 50% IC + 30% Sharpe + 20% Momentum |
| Minimize volatility | Min Variance | 50% Min Var + 30% Risk Parity + 20% Sharpe |
| Balanced risk | Risk Parity | 50% Risk Parity + 30% Sharpe + 20% Momentum |
| Tactical tilts | Momentum | 50% Momentum + 30% Sharpe + 20% IC |
| Maximum diversification | Max Diversification | 40% Max Div + 30% Risk Parity + 30% Sharpe |
| Production robustness | Blended | 40% Sharpe + 30% IC + 20% Momentum + 10% Risk Parity |

Remember: **No single method is always best.** The optimal approach depends on your:
- Investment objectives (return vs risk)
- Time horizon (tactical vs strategic)
- Data availability (forward returns for IC)
- Rebalancing constraints (turnover costs)
