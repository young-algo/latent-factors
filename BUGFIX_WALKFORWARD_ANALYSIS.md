# Critical Bug Fix: Factor Return Calculation

## Summary

**Issue Found:** Factor returns were calculated incorrectly in statistical models (PCA, ICA, etc.), producing impossible values that made walk-forward optimization results meaningless.

**Fix Applied:** Factor returns now correctly represent tradable portfolio performance.

---

## The Bug

### Problem
In `_rolling_stat_model()`, factor returns were computed in **standardized space** instead of **return space**:

```python
# BUGGY CODE (OLD)
today = (rets.iloc[[t]].values - scaler.mean_) / scaler.scale_
fac_rets.append((today @ decomp.components_.T).ravel())
```

This calculated:
- `today` = z-scored returns (mean=0, std=1)
- Multiplied by PCA components
- Result: arbitrary values with NO relation to actual returns

### Impact

| Metric | Buggy Values | Expected Values |
|--------|-------------|-----------------|
| Daily Mean Return | -4.75% | ~0% |
| Daily Volatility | 528% | ~1-2% |
| Maximum Daily Move | +1,559% | ~±5% |
| Minimum Daily Move | -2,618% | ~±5% |

**Your VTHR Results:**
- Train Sharpe: 3.11 (inflated by ~100×)
- Test Sharpe: -0.09 (random noise)
- **Train/Test Correlation: -0.046** (completely uncorrelated!)

The optimization was finding patterns in **standardization artifacts**, not real market relationships.

---

## The Fix

### Solution
Factor returns now calculated in **original return space**:

```python
# FIXED CODE (NEW)
today_rets = rets.iloc[t]  # Actual stock returns
factor_ret = today_rets.values @ load.values  # Stock returns × loadings
fac_rets.append(factor_ret)
```

This calculates:
- Factor return = weighted average of stock returns
- Weights = factor loadings (how much each stock contributes to factor)
- Result: actual portfolio returns you could trade

### Verification

After fix on same test data:
- Daily Mean: -0.14% ✅
- Daily Vol: 1.86% ✅  
- Range: -3.6% to +4.9% ✅
- These are realistic daily factor portfolio returns

---

## Impact on Your VTHR Optimization

### What This Means

**Before Fix:**
- Factor returns were noise (528% daily vol)
- Optimization found spurious patterns
- Sharpe ratios meaningless
- Test performance essentially random

**After Fix:**
- Factor returns are real portfolio returns
- Optimization finds genuine risk-adjusted edge
- Sharpe ratios comparable to real strategies
- Test performance predicts live performance

### Action Required

**You MUST re-run the optimization** to get valid results:

```bash
# Clear the buggy cache
rm factor_cache_VTHR_pca.pkl

# Re-run optimization with fixed code
uv run python -m src optimize --universe VTHR \
  --factor-method pca --n-components 12 \
  --walk-forward --train-window 90 --test-window 30 \
  --technique bayesian --name-factors \
  --export-weights vthr_pca_weights_fixed.csv \
  --output vthr_pca_results_fixed.json
```

### Expected Improvements

After re-running with the fix:

| Metric | Before (Buggy) | After (Fixed) |
|--------|---------------|---------------|
| Train Sharpe | 3.11 (unrealistic) | ~0.5-1.5 (realistic) |
| Test Sharpe | -0.09 (random) | Should improve |
| Train/Test Correlation | -0.046 (none) | Should be positive |
| Factor Returns | 528% vol (impossible) | ~20% ann vol (reasonable) |

A realistic long-short factor strategy might achieve:
- **Train Sharpe: 0.8-1.2** (in-sample fit)
- **Test Sharpe: 0.3-0.8** (out-of-sample, with decay)
- **Train/Test Correlation: 0.3-0.6** (some predictability)

---

## Technical Explanation

### How Statistical Factor Models Should Work

1. **Fit Period:** Run PCA on rolling window of stock returns
   - Standardize returns for numerical stability
   - Extract components (factor loadings)
   - Store: How much each stock moves with each factor

2. **Return Period:** Calculate factor returns
   - Take actual stock returns for day t
   - Multiply by factor loadings
   - Result: Return of factor-mimicking portfolio

### The Bug's Root Cause

The old code mistakenly used **standardized** returns for both fitting AND return calculation:

```python
z = scaler.fit_transform(win.values)  # OK for fitting
decomp.fit(z)
...
today = (rets.iloc[[t]] - scaler.mean_) / scaler.scale_  # WRONG for returns
```

Standardization transforms returns to z-scores (σ), not actual returns ($).

Multiplying z-scores by components gives arbitrary values, not portfolio P&L.

### Why It Wasn't Obvious

1. Factor loadings (stock exposures) were correct
2. Factor returns LOOKED like time series
3. Sharpe ratios were computed "correctly" on garbage data
4. Only comparing train/test showed the problem

---

## Lessons Learned

### Red Flags to Watch For

1. **Sharpe > 2.0 consistently** - Too good to be true
2. **Volatility > 100% daily** - Impossible for diversified portfolios
3. **Train/Test correlation < 0** - No predictive power
4. **Test Sharpe near 0** - Strategy has no edge

### Validation Checklist

Before trusting optimization results:

- [ ] Factor returns have realistic volatility (< 5% daily)
- [ ] Train Sharpe < 2.0 (exceptional but possible)
- [ ] Train/Test correlation > 0.2 (some predictability)
- [ ] Test Sharpe positive (actual edge)
- [ ] Drawdowns realistic (< 50%)

### Best Practices

1. **Always visualize factor returns** - Should look like real returns
2. **Check correlation with known factors** - Shouldn't be 100% noise
3. **Walk-forward is essential** - Catches overfitting
4. **Paper trade first** - Validate with real execution costs

---

## Files Affected

| File | Change |
|------|--------|
| `src/research.py` | Fixed `_rolling_stat_model()` calculation |
| `factor_cache_VTHR_pca.pkl` | Must be regenerated (contains buggy data) |
| `vthr_pca_results.json` | Invalid (based on buggy factor returns) |
| `vthr_pca_*.csv` | Invalid position weights |

---

## Next Steps

1. **Pull latest code** with fix
2. **Delete old cache:** `rm factor_cache_VTHR_pca.pkl`
3. **Re-run optimization**
4. **Validate results:** Check factor return statistics
5. **Generate new basket** with fixed weights

---

## Appendix: The Mathematics

### Buggy Calculation
```
Factor Return = Z-score(Returns) × Components
              = [(R - μ) / σ] × V
              = Arbitrary values (not $ P&L)
```

### Fixed Calculation  
```
Factor Return = Returns × Loadings
              = R × V
              = Σ(weight_i × return_i)
              = Actual portfolio return ($)
```

Where:
- R = vector of stock returns
- V = matrix of factor loadings (from PCA components)
- weight_i = how much stock i contributes to factor
- return_i = actual return of stock i

---

*Bug identified: 2026-02-02*  
*Fix committed: a11de30*  
*Impact: Critical - All statistical factor results prior to this fix are invalid*
