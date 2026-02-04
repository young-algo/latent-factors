"""
Factor Weight Optimization Strategy
===================================

This module implements walk-forward optimization for finding optimal blended
factor weights that maximize Sharpe ratio over a given lookback window.

The strategy can be used for:
1. Historical analysis (what would have worked best)
2. Walk-forward optimization (rolling window optimization)
3. Ensemble weighting (combining multiple lookback horizons)

Available Optimization Methods
------------------------------
1. **Gradient Ascent** - Efficient optimization using gradients
2. **Differential Evolution** - Global optimization using evolutionary algorithm
3. **Bayesian Optimization (Optuna)** - Smart search using TPE sampler
4. **Risk-Adjusted Optimization** - Maximize Sharpe with risk constraints

Usage
-----
>>> from factor_optimization import SharpeOptimizer
>>> optimizer = SharpeOptimizer(factor_returns)

>>> # Simple optimization
>>> optimal_weights = optimizer.optimize_blend(
...     lookback=63,
...     methods=['sharpe', 'momentum', 'risk_parity']
... )

>>> # Walk-forward optimization
>>> rolling_weights = optimizer.walk_forward_optimize(
...     train_window=126,
...     test_window=21,
...     methods=['sharpe', 'ic', 'momentum']
... )

>>> # Out-of-sample performance
>>> performance = optimizer.backtest_optimal_weights(
...     optimal_weights,
...     test_periods=63
... )
"""

from __future__ import annotations
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Callable, Literal
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

try:
    from .factor_weighting import OptimalFactorWeighter, WeightingMethod
except ImportError:
    from src.factor_weighting import OptimalFactorWeighter, WeightingMethod

_LOGGER = logging.getLogger(__name__)

# Optuna is optional - only needed for Bayesian optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not installed. Bayesian optimization will use differential evolution fallback. "
                  "Install with: uv add optuna")


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    optimal_weights: Dict[str, float]
    sharpe_ratio: float
    annualized_return: float
    annualized_volatility: float
    method_allocation: Dict[str, float]
    backtest_results: Optional[pd.DataFrame] = None
    optimization_metrics: Optional[Dict] = None


class SharpeOptimizer:
    """
    Optimizes factor weight blends to maximize Sharpe ratio.
    
    This class implements multiple optimization strategies for finding the best
    combination of factor weighting methods that would have maximized risk-adjusted
    returns over a historical lookback period.
    
    Parameters
    ----------
    factor_returns : pd.DataFrame
        Factor returns matrix (T×K) where:
        - T = number of time periods
        - K = number of factors
        - Values = daily/periodic factor returns
    factor_loadings : pd.DataFrame, optional
        Factor loadings matrix (N×K) for methods requiring exposures
    risk_free_rate : float, default 0.0
        Risk-free rate for Sharpe calculation
    
    Attributes
    ----------
    returns : pd.DataFrame
        Factor returns
    loadings : pd.DataFrame | None
        Factor loadings
    n_factors : int
        Number of factors
    factor_names : List[str]
        Factor names
    
    Methods
    -------
    optimize_blend(lookback, methods, technique='differential')
        Find optimal blend of weighting methods
    walk_forward_optimize(train_window, test_window, methods)
        Rolling window optimization
    gradient_optimize(lookback, methods, initial_guess=None)
        Gradient-based optimization
    bayesian_optimize(lookback, methods, n_trials=100)
        Bayesian optimization with Optuna TPE sampler
    differential_evolution_optimize(lookback, methods)
        Differential evolution global optimization
    backtest_optimal_weights(optimal_weights, test_periods)
        Test optimized weights out-of-sample
    
    Mathematical Framework
    ----------------------
    Given factor returns R (T×K) and blend weights w_method (M×1),
    where M is the number of weighting methods:
    
    1. Calculate method-specific factor weights:
       w_factor^m = method_m(R, lookback) for each method m
    
    2. Blend to get composite factor weights:
       w_factor = Σ_m (w_method^m × w_factor^m)
    
    3. Calculate portfolio returns (equal-weighted factors):
       r_portfolio = mean(R @ w_factor) across factors
    
    4. Calculate Sharpe ratio:
       Sharpe = (mean(r_portfolio) - r_f) / std(r_portfolio) × √252
    
    5. Optimize:
       max_w_method Sharpe(w_method)
       s.t. Σ_m w_method^m = 1, w_method^m ≥ 0
    
    Examples
    --------
    >>> # Initialize
    >>> optimizer = SharpeOptimizer(factor_returns, factor_loadings)
    
    >>> # Differential evolution (recommended)
    >>> result = optimizer.optimize_blend(
    ...     lookback=126,
    ...     methods=['sharpe', 'momentum', 'risk_parity'],
    ...     technique='differential'
    ... )
    >>> print(f"Optimal blend: {result.method_allocation}")
    >>> print(f"Sharpe ratio: {result.sharpe_ratio:.2f}")
    
    >>> # Bayesian optimization with Optuna (if installed)
    >>> result = optimizer.optimize_blend(
    ...     lookback=126,
    ...     methods=['sharpe', 'momentum', 'risk_parity', 'ic'],
    ...     technique='bayesian',
    ...     n_trials=100
    ... )
    
    >>> # Walk-forward optimization
    >>> rolling = optimizer.walk_forward_optimize(
    ...     train_window=252,
    ...     test_window=63,
    ...     methods=['sharpe', 'ic', 'momentum']
    ... )
    >>> 
    >>> # Plot rolling weights
    >>> rolling['method_weights'].plot()
    """
    
    def __init__(
        self,
        factor_returns: pd.DataFrame,
        factor_loadings: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.0,
        reset_index: bool = True
    ):
        """
        Initialize the SharpeOptimizer.
        
        Parameters
        ----------
        factor_returns : pd.DataFrame
            Factor returns matrix (T×K). Can have non-contiguous index
            (e.g., regime-filtered data).
        factor_loadings : pd.DataFrame, optional
            Factor loadings matrix (N×K)
        risk_free_rate : float, default 0.0
            Risk-free rate for Sharpe calculation
        reset_index : bool, default True
            If True, resets the index of factor_returns to handle 
            non-contiguous indices (e.g., from regime filtering).
            This is important for RS-MVO (Regime-Switching Mean-Variance Optimization)
            where returns are filtered by regime state.
        """
        # Handle non-contiguous indices (e.g., from regime filtering)
        # We preserve the original index for reference but reset for calculations
        self.returns = factor_returns.copy()
        if reset_index:
            # Store original index for reference
            self._original_index = self.returns.index.copy()
            # Reset index for calculations (positions 0, 1, 2, ...)
            self.returns = self.returns.reset_index(drop=True)
        else:
            self._original_index = None
            
        self.loadings = factor_loadings.copy() if factor_loadings is not None else None
        self.risk_free_rate = risk_free_rate
        self.n_factors = factor_returns.shape[1]
        self.factor_names = list(factor_returns.columns)
        
        # Initialize weighter for method-specific weights
        # Note: weighter also needs regime-filtered returns with reset index
        self._weighter = OptimalFactorWeighter(
            factor_loadings if factor_loadings is not None else pd.DataFrame(),
            self.returns  # Use reset-index returns
        )
    
    def optimize_blend(
        self,
        lookback: int = 63,
        methods: List[str] = None,
        technique: Literal['gradient', 'differential', 'bayesian'] = 'differential',
        validation_split: float = 0.2,
        **kwargs
    ) -> OptimizationResult:
        """
        Find optimal blend of weighting methods to maximize Sharpe ratio.
        
        Uses a nested walk-forward approach to prevent overfitting:
        1. Splits lookback into Estimation (first part) and Validation (last part)
        2. Calculates method weights using ONLY Estimation data
        3. Optimizes blending weights to maximize Sharpe on Validation data
        
        Parameters
        ----------
        lookback : int, default 63
            Total lookback window (Estimation + Validation)
        methods : List[str], optional
            List of methods to blend
        technique : str, default 'differential'
            Optimization technique
        validation_split : float, default 0.2
            Fraction of lookback to use for validation (out-of-sample optimization)
        **kwargs
            Additional parameters for optimization
            
        Returns
        -------
        OptimizationResult
            Contains optimal weights and metrics
        """
        if methods is None:
            methods = ['sharpe', 'momentum', 'risk_parity']
        
        # Validate inputs
        available_methods = self._get_available_methods()
        methods = [m for m in methods if m in available_methods]
        
        if len(methods) < 2:
            raise ValueError("Need at least 2 methods for blending")
        
        # 1. Prepare Data Split
        # ---------------------
        if len(self.returns) < lookback:
            _LOGGER.warning(f"Data length ({len(self.returns)}) < lookback ({lookback}). Using full data.")
            lookback = len(self.returns)
            
        full_window_data = self.returns.tail(lookback)
        
        # Determine split point
        val_size = int(lookback * validation_split)
        if val_size < 5:  # Ensure minimum validation size
            val_size = 5
        est_size = lookback - val_size
        
        if est_size < 20: # Warning for small estimation
             _LOGGER.warning(f"Small estimation window: {est_size} periods")

        # Split data
        estimation_data = full_window_data.iloc[:est_size]
        validation_data = full_window_data.iloc[est_size:]
        
        # 2. Pre-calculate Method Weights (Efficiency Fix)
        # -----------------------------------------------
        # Create a temporary weighter trained ONLY on estimation data
        # This prevents "Double Dipping" / Hindsight Bias
        # Handle case where loadings is None (pass empty DataFrame)
        if self.loadings is not None:
            loadings_for_weighter = self.loadings
        else:
            loadings_for_weighter = pd.DataFrame(
                0.0, 
                index=range(1), 
                columns=estimation_data.columns
            )
        est_weighter = OptimalFactorWeighter(
            loadings_for_weighter,
            estimation_data
        )
        
        precalculated_weights = []
        for method in methods:
            try:
                # Use the helper to call specific methods on the est_weighter
                # We pass the full est_size as lookback since the weighter contains only that data
                w = self._calculate_single_method_weights(est_weighter, method, est_size)
                precalculated_weights.append(pd.Series(w).fillna(0))
            except Exception as e:
                _LOGGER.warning(f"Failed to calculate {method} weights: {e}")
                # Fallback to equal weights or zero
                precalculated_weights.append(pd.Series(est_weighter.equal_weights()))

        # 3. Optimize Blend on Validation Data
        # ------------------------------------
        if technique == 'gradient':
            return self._gradient_optimize(validation_data, methods, precalculated_weights, **kwargs)
        elif technique == 'differential':
            return self._differential_evolution_optimize(validation_data, methods, precalculated_weights, **kwargs)
        elif technique == 'bayesian':
            return self._bayesian_optimize(validation_data, methods, precalculated_weights, **kwargs)
        else:
            raise ValueError(f"Unknown technique: {technique}")

    def _calculate_single_method_weights(self, weighter: OptimalFactorWeighter, method: str, lookback: int) -> Dict[str, float]:
        """Helper to call weighter methods with correct parameters."""
        if method == 'equal':
            return weighter.equal_weights()
        elif method == 'sharpe':
            return weighter.sharpe_weights(lookback=lookback)
        elif method == 'momentum':
            # Momentum usually needs shorter lookback, e.g., 21 or 63
            # If lookback is very long, cap it at 126
            mom_lookback = min(lookback, 252) 
            return weighter.momentum_weights(lookback=mom_lookback)
        elif method == 'risk_parity':
            return weighter.risk_parity_weights(lookback=lookback)
        elif method == 'min_variance':
            return weighter.min_variance_weights(lookback=lookback)
        elif method == 'max_diversification':
            return weighter.max_diversification_weights(lookback=lookback)
        elif method == 'pca':
            return weighter.pca_weights()
        else:
            return weighter.equal_weights()
    
    def _get_available_methods(self) -> List[str]:
        """Get list of methods that can be used without additional data."""
        methods = ['equal', 'sharpe', 'momentum', 'risk_parity', 
                   'min_variance', 'max_diversification', 'pca']
        return methods
    
    def _calculate_blend_sharpe(
        self,
        method_weights: np.ndarray,
        validation_returns: pd.DataFrame,
        component_factor_weights: List[pd.Series]
    ) -> float:
        """
        Calculate Sharpe ratio for a blend on validation data.
        
        Parameters
        ----------
        method_weights : np.ndarray
            Weights for each method
        validation_returns : pd.DataFrame
            Returns for the validation period
        component_factor_weights : List[pd.Series]
            Pre-calculated factor weights for each method
        """
        try:
            # Normalize method weights
            if method_weights.sum() == 0:
                return -999.0
            method_weights = method_weights / method_weights.sum()
            
            # Blend factor weights
            # Vectorized blend: sum(w_m * weights_m)
            blended_factor_weights = pd.Series(0.0, index=self.factor_names)
            
            for i, w_series in enumerate(component_factor_weights):
                blended_factor_weights += method_weights[i] * w_series.reindex(self.factor_names).fillna(0)
            
            # Normalize factor weights
            if blended_factor_weights.sum() > 0:
                blended_factor_weights = blended_factor_weights / blended_factor_weights.sum()
            else:
                return -999.0
            
            # Calculate portfolio returns on VALIDATION set
            portfolio_returns = (validation_returns * blended_factor_weights).sum(axis=1)
            
            # Calculate Sharpe
            mean_ret = portfolio_returns.mean()
            std_ret = portfolio_returns.std()
            
            if std_ret <= 1e-10:
                return -999.0
            
            sharpe = (mean_ret - self.risk_free_rate) / std_ret * np.sqrt(252)
            return sharpe
            
        except Exception as e:
            return -999.0
    
    def _gradient_optimize(
        self,
        validation_returns: pd.DataFrame,
        methods: List[str],
        precalculated_weights: List[pd.Series],
        initial_guess: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> OptimizationResult:
        """Gradient-based optimization using pre-calculated weights."""
        if verbose:
            print(f"Running gradient-based optimization with {len(methods)} methods...")
        
        n_methods = len(methods)
        
        # Initial guess
        if initial_guess is None:
            x0 = np.ones(n_methods) / n_methods
        else:
            x0 = initial_guess
        
        # Objective (negative Sharpe)
        def objective(x):
            return -self._calculate_blend_sharpe(x, validation_returns, precalculated_weights)
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
        bounds = [(0, 1) for _ in range(n_methods)]
        
        result = minimize(
            objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        return self._build_result(result, methods, precalculated_weights, validation_returns, -result.fun, verbose)

    def _differential_evolution_optimize(
        self,
        validation_returns: pd.DataFrame,
        methods: List[str],
        precalculated_weights: List[pd.Series],
        population_size: int = 15,
        max_iterations: int = 100,
        verbose: bool = True
    ) -> OptimizationResult:
        """Differential evolution with pre-calculated weights."""
        if verbose:
            print(f"Running differential evolution with {len(methods)} methods...")
        
        n_methods = len(methods)
        
        def objective(x):
            x = x / x.sum() if x.sum() > 0 else np.ones_like(x) / len(x)
            return -self._calculate_blend_sharpe(x, validation_returns, precalculated_weights)
        
        bounds = [(0, 1) for _ in range(n_methods)]
        
        def penalized_objective(x):
            sum_penalty = 1000 * (np.sum(x) - 1) ** 2
            return objective(x) + sum_penalty
        
        result = differential_evolution(
            penalized_objective, bounds, maxiter=max_iterations, popsize=population_size, seed=42, polish=True
        )
        
        best_sharpe = -objective(result.x)
        return self._build_result(result, methods, precalculated_weights, validation_returns, best_sharpe, verbose)

    def _bayesian_optimize(
        self,
        validation_returns: pd.DataFrame,
        methods: List[str],
        precalculated_weights: List[pd.Series],
        n_trials: int = 100,
        timeout: Optional[int] = None,
        verbose: bool = True
    ) -> OptimizationResult:
        """Bayesian optimization with pre-calculated weights."""
        if not OPTUNA_AVAILABLE:
            if verbose: print("Optuna not available, falling back to differential evolution...")
            return self._differential_evolution_optimize(validation_returns, methods, precalculated_weights, verbose=verbose)
        
        if verbose: print(f"Running Bayesian optimization with Optuna ({n_trials} trials)...")
        
        n_methods = len(methods)
        
        def objective(trial):
            raw_weights = [trial.suggest_float(f'w_{i}', 0.0, 1.0) for i in range(n_methods)]
            total = sum(raw_weights)
            if total < 1e-10: return -999.0
            method_weights = np.array(raw_weights) / total
            return self._calculate_blend_sharpe(method_weights, validation_returns, precalculated_weights)
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        optuna.logging.set_verbosity(optuna.logging.WARNING if not verbose else optuna.logging.INFO)
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=verbose)
        
        best_trial = study.best_trial
        best_sharpe = best_trial.value
        
        # Reconstruct weights
        raw_weights = [best_trial.params[f'w_{i}'] for i in range(n_methods)]
        optimal_method_weights = np.array(raw_weights) / sum(raw_weights)
        
        # Build fake result object to reuse _build_result logic
        class Result: pass
        result = Result()
        result.x = optimal_method_weights
        result.success = True
        result.nit = n_trials
        result.message = "Optuna Optimization"
        
        return self._build_result(result, methods, precalculated_weights, validation_returns, best_sharpe, verbose)

    def _build_result(
        self, 
        result, 
        methods: List[str], 
        precalculated_weights: List[pd.Series], 
        validation_returns: pd.DataFrame, 
        best_sharpe: float,
        verbose: bool
    ) -> OptimizationResult:
        """Helper to construct OptimizationResult."""
        n_methods = len(methods)
        optimal_method_weights = result.x / result.x.sum()
        
        method_allocation = {methods[i]: optimal_method_weights[i] for i in range(n_methods)}
        
        # Calculate final factor weights using the precalculated components
        optimal_factor_weights = pd.Series(0.0, index=self.factor_names)
        for i, w_series in enumerate(precalculated_weights):
            optimal_factor_weights += optimal_method_weights[i] * w_series.reindex(self.factor_names).fillna(0)
        
        # Enforce non-negative weights (long-only constraint)
        optimal_factor_weights = optimal_factor_weights.clip(lower=0)
            
        if optimal_factor_weights.sum() > 0:
            optimal_factor_weights = optimal_factor_weights / optimal_factor_weights.sum()
        else:
            # Fallback to equal weights if all weights are zero
            optimal_factor_weights = pd.Series(1.0 / len(self.factor_names), index=self.factor_names)
        
        optimal_factor_weights_dict = optimal_factor_weights.to_dict()
        
        # Calc stats on validation set (or could use full set, but validation is the 'test' here)
        portfolio_returns = self._get_portfolio_returns(optimal_factor_weights_dict, validation_returns)
        ann_return = portfolio_returns.mean() * 252
        ann_vol = portfolio_returns.std() * np.sqrt(252)
        
        if verbose:
            print(f"\nOptimal blend found:")
            for method, weight in method_allocation.items():
                if weight > 0.01:
                    print(f"  {method}: {weight:.1%}")
            print(f"\nSharpe Ratio: {best_sharpe:.2f}")
            print(f"Annualized Return: {ann_return:.2%}")
            print(f"Annualized Volatility: {ann_vol:.2%}")

        return OptimizationResult(
            optimal_weights=optimal_factor_weights_dict,
            sharpe_ratio=best_sharpe,
            annualized_return=ann_return,
            annualized_volatility=ann_vol,
            method_allocation=method_allocation,
            optimization_metrics={'success': getattr(result, 'success', True), 'n_iterations': getattr(result, 'nit', 0)}
        )

    def _get_portfolio_returns(
        self,
        factor_weights: Dict[str, float],
        returns: pd.DataFrame
    ) -> pd.Series:
        """Calculate portfolio returns given factor weights."""
        weights_series = pd.Series(factor_weights)
        portfolio_returns = (returns * weights_series).sum(axis=1)
        return portfolio_returns
    
    def walk_forward_optimize(
        self,
        train_window: int = 126,
        test_window: int = 21,
        methods: List[str] = None,
        technique: str = 'differential',
        step_size: Optional[int] = None,
        validation_split: float = 0.2, # Added
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Walk-forward optimization with internal validation split.
        
        Parameters
        ----------
        train_window : int
            Total window for training (Estimation + Validation)
        test_window : int
            Out-of-sample test window
        """
        if step_size is None:
            step_size = test_window
        
        if methods is None:
            methods = ['sharpe', 'momentum', 'risk_parity']
        
        n_periods = len(self.returns)
        results = []
        
        start_idx = train_window
        while start_idx + test_window <= n_periods:
            train_start = start_idx - train_window
            train_end = start_idx
            test_end = min(start_idx + test_window, n_periods)
            
            # Get training data (to be split internally by optimize_blend)
            train_returns = self.returns.iloc[train_start:train_end]
            
            window_optimizer = SharpeOptimizer(
                train_returns,
                self.loadings,
                self.risk_free_rate
            )
            
            if verbose:
                print(f"\nOptimizing window {len(results)+1}: "
                      f"{self.returns.index[train_start].date()} to "
                      f"{self.returns.index[train_end-1].date()}")
            
            try:
                # optimize_blend will handle the split internally now
                result = window_optimizer.optimize_blend(
                    lookback=train_window,
                    methods=methods,
                    technique=technique,
                    validation_split=validation_split,
                    verbose=verbose
                )
                
                # Test out-of-sample (True Hold-out)
                test_returns = self.returns.iloc[train_end:test_end]
                test_portfolio = self._get_portfolio_returns(
                    result.optimal_weights,
                    test_returns
                )
                test_sharpe = (test_portfolio.mean() / test_portfolio.std() * np.sqrt(252) 
                              if test_portfolio.std() > 0 else 0)
                
                results.append({
                    'date': self.returns.index[train_end],
                    'method_weights': result.method_allocation,
                    'factor_weights': result.optimal_weights,
                    'train_sharpe': result.sharpe_ratio,
                    'test_sharpe': test_sharpe,
                    'train_return': result.annualized_return,
                    'train_vol': result.annualized_volatility
                })
                
            except Exception as e:
                _LOGGER.error(f"Optimization failed for window {len(results)+1}: {e}")
            
            start_idx += step_size
        
        return pd.DataFrame(results)
    
    def backtest_optimal_weights(
        self,
        optimal_weights: Dict[str, float],
        test_periods: int = 63,
        transaction_cost: float = 0.0,
        rebalance_freq: int = 21
    ) -> Dict:
        """
        Backtest with realistic weight drift.
        
        Simulates buy-and-hold between rebalancing periods.
        """
        test_returns = self.returns.tail(test_periods)
        
        # Initialize
        current_capital = 1.0
        portfolio_values = [current_capital]
        
        # Initial allocation
        # Position values = Capital * Weight
        current_positions = {f: current_capital * optimal_weights.get(f, 0) 
                           for f in self.factor_names}
        
        for i in range(len(test_returns)):
            # 1. Rebalance Check (Start of day / End of previous day)
            if i > 0 and i % rebalance_freq == 0:
                # Target positions
                target_positions = {f: current_capital * optimal_weights.get(f, 0)
                                  for f in self.factor_names}
                
                # Calculate turnover
                turnover_value = sum(abs(target_positions[f] - current_positions.get(f, 0))
                                   for f in self.factor_names)
                
                # Pay costs
                cost = turnover_value * transaction_cost
                current_capital -= cost
                
                # Reset positions to target (adjusted for cost)
                current_positions = {f: current_capital * optimal_weights.get(f, 0)
                                   for f in self.factor_names}
            
            # 2. Market Movement (During day)
            day_ret_series = test_returns.iloc[i]
            day_pnl = 0.0
            
            new_positions = {}
            for f, pos_value in current_positions.items():
                ret = day_ret_series.get(f, 0.0)
                new_val = pos_value * (1 + ret)
                new_positions[f] = new_val
                day_pnl += (new_val - pos_value)
            
            current_positions = new_positions
            current_capital += day_pnl
            portfolio_values.append(current_capital)
        
        # Calculate metrics
        portfolio_values_series = pd.Series(portfolio_values)
        portfolio_returns = portfolio_values_series.pct_change().dropna()
        
        total_return = portfolio_values[-1] - 1
        ann_return = portfolio_returns.mean() * 252
        ann_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = (ann_return / ann_vol) if ann_vol > 0 else 0
        
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values_series - running_max) / running_max
        max_dd = drawdown.min()
        
        return {
            'total_return': total_return,
            'annualized_return': ann_return,
            'annualized_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'portfolio_values': portfolio_values,
            'portfolio_returns': portfolio_returns
        }
    
    def compare_methods_over_horizons(
        self,
        lookback_horizons: List[int] = None,
        methods: List[str] = None
    ) -> pd.DataFrame:
        """
        Compare optimal blends across different lookback horizons.
        
        Helps determine optimal lookback window.
        
        Parameters
        ----------
        lookback_horizons : List[int], optional
            Different lookback periods to test
        methods : List[str], optional
            Methods to blend
            
        Returns
        -------
        pd.DataFrame
            Performance by lookback horizon
        """
        if lookback_horizons is None:
            lookback_horizons = [21, 42, 63, 126, 252]
        
        if methods is None:
            methods = ['sharpe', 'momentum', 'risk_parity']
        
        results = []
        
        for lookback in lookback_horizons:
            if lookback > len(self.returns):
                continue
            
            try:
                result = self.optimize_blend(
                    lookback=lookback,
                    methods=methods,
                    technique='differential',
                    verbose=False
                )
                
                results.append({
                    'lookback': lookback,
                    'sharpe_ratio': result.sharpe_ratio,
                    'annualized_return': result.annualized_return,
                    'annualized_volatility': result.annualized_volatility,
                    'method_allocation': result.method_allocation
                })
                
            except Exception as e:
                _LOGGER.warning(f"Optimization failed for lookback {lookback}: {e}")
        
        return pd.DataFrame(results)
