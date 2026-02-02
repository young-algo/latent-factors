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
1. **Grid Search** - Exhaustive search over weight combinations
2. **Gradient Ascent** - Efficient optimization using gradients
3. **Bayesian Optimization** - Smart search using Gaussian Processes
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
from typing import Dict, List, Optional, Tuple, Callable, Literal
from dataclasses import dataclass
from itertools import product
from functools import partial

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

from .factor_weighting import OptimalFactorWeighter, WeightingMethod

_LOGGER = logging.getLogger(__name__)


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
    optimize_blend(lookback, methods, technique='grid')
        Find optimal blend of weighting methods
    walk_forward_optimize(train_window, test_window, methods)
        Rolling window optimization
    grid_search_optimize(lookback, methods, grid_points=5)
        Exhaustive grid search over weight combinations
    gradient_optimize(lookback, methods, initial_guess=None)
        Gradient-based optimization
    bayesian_optimize(lookback, methods, n_iterations=50)
        Bayesian optimization with Gaussian Processes
    max_sharpe_with_constraints(lookback, methods, max_volatility=None)
        Maximize Sharpe with optional volatility constraint
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
    
    >>> # Simple grid search optimization
    >>> result = optimizer.optimize_blend(
    ...     lookback=126,
    ...     methods=['sharpe', 'momentum', 'risk_parity'],
    ...     technique='grid'
    ... )
    >>> print(f"Optimal blend: {result.method_allocation}")
    >>> print(f"Sharpe ratio: {result.sharpe_ratio:.2f}")
    
    >>> # Walk-forward optimization
    >>> rolling = optimizer.walk_forward_optimize(
    ...     train_window=252,
    ...     test_window=63,
    ...     methods=['sharpe', 'ic', 'momentum']
    ... )
    >>> 
    >>> # Plot rolling weights
    >>> rolling['method_weights'].plot()
    
    >>> # Backtest with transaction costs
    >>> perf = optimizer.backtest_optimal_weights(
    ...     result.optimal_weights,
    ...     test_periods=126,
    ...     transaction_cost=0.001
    ... )
    """
    
    def __init__(
        self,
        factor_returns: pd.DataFrame,
        factor_loadings: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.0
    ):
        """
        Initialize the SharpeOptimizer.
        
        Parameters
        ----------
        factor_returns : pd.DataFrame
            Factor returns matrix (T×K)
        factor_loadings : pd.DataFrame, optional
            Factor loadings matrix (N×K)
        risk_free_rate : float, default 0.0
            Risk-free rate for Sharpe calculation
        """
        self.returns = factor_returns.copy()
        self.loadings = factor_loadings.copy() if factor_loadings is not None else None
        self.risk_free_rate = risk_free_rate
        self.n_factors = factor_returns.shape[1]
        self.factor_names = list(factor_returns.columns)
        
        # Initialize weighter for method-specific weights
        self._weighter = OptimalFactorWeighter(
            factor_loadings if factor_loadings is not None else pd.DataFrame(),
            factor_returns
        )
    
    def optimize_blend(
        self,
        lookback: int = 63,
        methods: List[str] = None,
        technique: Literal['grid', 'gradient', 'bayesian', 'differential'] = 'grid',
        **kwargs
    ) -> OptimizationResult:
        """
        Find optimal blend of weighting methods to maximize Sharpe ratio.
        
        This is the main entry point for optimization. It dispatches to
        different optimization techniques based on the 'technique' parameter.
        
        Parameters
        ----------
        lookback : int, default 63
            Lookback window for optimization (days)
        methods : List[str], optional
            List of methods to blend. Options:
            - 'equal', 'sharpe', 'momentum', 'risk_parity'
            - 'min_variance', 'max_diversification', 'pca', 'ic'
            If None, uses ['sharpe', 'momentum', 'risk_parity']
        technique : str, default 'grid'
            Optimization technique:
            - 'grid': Exhaustive grid search (exact but slower)
            - 'gradient': Gradient-based (faster, may find local optima)
            - 'bayesian': Bayesian optimization (best for complex spaces)
            - 'differential': Differential evolution (good balance)
        **kwargs
            Additional parameters for specific techniques
            
        Returns
        -------
        OptimizationResult
            Contains optimal weights, Sharpe ratio, and performance metrics
            
        Examples
        --------
        >>> optimizer = SharpeOptimizer(returns, loadings)
        
        >>> # Grid search (most thorough)
        >>> result = optimizer.optimize_blend(lookback=126, technique='grid')
        
        >>> # Fast gradient-based
        >>> result = optimizer.optimize_blend(lookback=126, technique='gradient')
        
        >>> # Bayesian (best for many methods)
        >>> result = optimizer.optimize_blend(
        ...     lookback=126,
        ...     methods=['sharpe', 'momentum', 'risk_parity', 'ic'],
        ...     technique='bayesian',
        ...     n_iterations=100
        ... )
        """
        if methods is None:
            methods = ['sharpe', 'momentum', 'risk_parity']
        
        # Filter methods requiring forward returns if not provided
        available_methods = self._get_available_methods()
        methods = [m for m in methods if m in available_methods]
        
        if len(methods) < 2:
            raise ValueError("Need at least 2 methods for blending")
        
        if technique == 'grid':
            return self._grid_search_optimize(lookback, methods, **kwargs)
        elif technique == 'gradient':
            return self._gradient_optimize(lookback, methods, **kwargs)
        elif technique == 'bayesian':
            return self._bayesian_optimize(lookback, methods, **kwargs)
        elif technique == 'differential':
            return self._differential_evolution_optimize(lookback, methods, **kwargs)
        else:
            raise ValueError(f"Unknown technique: {technique}")
    
    def _get_available_methods(self) -> List[str]:
        """Get list of methods that can be used without additional data."""
        methods = ['equal', 'sharpe', 'momentum', 'risk_parity', 
                   'min_variance', 'max_diversification', 'pca']
        return methods
    
    def _calculate_blend_sharpe(
        self,
        method_weights: np.ndarray,
        lookback_returns: pd.DataFrame,
        method_functions: List[Callable]
    ) -> float:
        """
        Calculate Sharpe ratio for a given blend of methods.
        
        Parameters
        ----------
        method_weights : np.ndarray
            Weights for each method (must sum to 1)
        lookback_returns : pd.DataFrame
            Returns for the lookback period
        method_functions : List[Callable]
            Functions to generate factor weights for each method
            
        Returns
        -------
        float
            Sharpe ratio (negative for minimization)
        """
        try:
            # Normalize method weights
            method_weights = method_weights / method_weights.sum()
            
            # Get factor weights from each method
            factor_weight_list = []
            for i, method_fn in enumerate(method_functions):
                weights = method_fn()
                factor_weight_list.append(pd.Series(weights))
            
            # Blend factor weights
            blended_factor_weights = pd.Series(0.0, index=self.factor_names)
            for i, fw in enumerate(factor_weight_list):
                blended_factor_weights += method_weights[i] * fw.reindex(self.factor_names).fillna(0)
            
            # Normalize factor weights
            if blended_factor_weights.sum() > 0:
                blended_factor_weights = blended_factor_weights / blended_factor_weights.sum()
            else:
                return -999.0  # Invalid weights
            
            # Calculate portfolio returns (weighted average of factor returns)
            portfolio_returns = (lookback_returns * blended_factor_weights).sum(axis=1)
            
            # Calculate Sharpe
            mean_ret = portfolio_returns.mean()
            std_ret = portfolio_returns.std()
            
            if std_ret <= 1e-10:
                return -999.0  # No variation
            
            sharpe = (mean_ret - self.risk_free_rate) / std_ret * np.sqrt(252)
            
            return sharpe
            
        except Exception as e:
            _LOGGER.warning(f"Error calculating Sharpe: {e}")
            return -999.0
    
    def _grid_search_optimize(
        self,
        lookback: int,
        methods: List[str],
        grid_points: int = 5,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Exhaustive grid search over weight combinations.
        
        Most thorough but computationally expensive for many methods.
        With M methods and G grid points: G^(M-1) evaluations.
        """
        if verbose:
            print(f"Running grid search with {len(methods)} methods, {grid_points} points each...")
        
        # Get lookback returns
        lookback_returns = self.returns.tail(lookback)
        
        # Generate method weight functions
        method_functions = self._get_method_functions(methods, lookback)
        
        # Generate grid
        # For M methods, we need M-1 dimensions (last is 1 - sum(others))
        n_methods = len(methods)
        grid_values = np.linspace(0, 1, grid_points)
        
        best_sharpe = -np.inf
        best_weights = None
        results = []
        
        # Generate all combinations that sum to <= 1
        # Use simplex sampling
        from itertools import combinations_with_replacement
        
        def generate_simplex_grid(n_dims, n_points):
            """Generate points on simplex (sum to 1)."""
            if n_dims == 2:
                # 2D simplex: line from (0,1) to (1,0)
                w1 = np.linspace(0, 1, n_points)
                return [[w, 1-w] for w in w1]
            elif n_dims == 3:
                # 3D simplex: triangle
                points = []
                for i in range(n_points):
                    for j in range(n_points - i):
                        w1 = i / (n_points - 1)
                        w2 = j / (n_points - 1)
                        w3 = 1 - w1 - w2
                        if w3 >= 0:
                            points.append([w1, w2, w3])
                return points
            elif n_dims == 4:
                # 4D simplex: tetrahedron
                points = []
                step = 1 / (n_points - 1)
                for i in range(n_points):
                    for j in range(n_points - i):
                        for k in range(n_points - i - j):
                            w1 = i * step
                            w2 = j * step
                            w3 = k * step
                            w4 = 1 - w1 - w2 - w3
                            if w4 >= -1e-10:
                                points.append([w1, w2, w3, max(0, w4)])
                return points
            else:
                # For higher dimensions, use random sampling
                np.random.seed(42)
                points = []
                for _ in range(n_points ** 2):
                    w = np.random.dirichlet(np.ones(n_dims))
                    points.append(w.tolist())
                return points
        
        weight_combinations = generate_simplex_grid(n_methods, grid_points)
        
        if verbose:
            print(f"Testing {len(weight_combinations)} weight combinations...")
        
        for w_combo in weight_combinations:
            method_weights = np.array(w_combo)
            sharpe = self._calculate_blend_sharpe(
                method_weights, lookback_returns, method_functions
            )
            
            results.append({
                'weights': method_weights,
                'sharpe': sharpe
            })
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = method_weights.copy()
        
        if best_weights is None:
            raise RuntimeError("Optimization failed to find valid weights")
        
        # Create result
        method_allocation = {methods[i]: best_weights[i] for i in range(n_methods)}
        
        # Get optimal factor weights
        optimal_factor_weights = self._weighter.blend_weights(
            {methods[i]: best_weights[i] for i in range(n_methods)}
        )
        
        # Calculate additional metrics
        portfolio_returns = self._get_portfolio_returns(optimal_factor_weights, lookback_returns)
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
            optimal_weights=optimal_factor_weights,
            sharpe_ratio=best_sharpe,
            annualized_return=ann_return,
            annualized_volatility=ann_vol,
            method_allocation=method_allocation,
            optimization_metrics={
                'grid_points': grid_points,
                'combinations_tested': len(weight_combinations),
                'all_results': results
            }
        )
    
    def _gradient_optimize(
        self,
        lookback: int,
        methods: List[str],
        initial_guess: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Gradient-based optimization using SLSQP.
        
        Faster than grid search but may converge to local optima.
        """
        if verbose:
            print(f"Running gradient-based optimization with {len(methods)} methods...")
        
        lookback_returns = self.returns.tail(lookback)
        method_functions = self._get_method_functions(methods, lookback)
        
        n_methods = len(methods)
        
        # Initial guess
        if initial_guess is None:
            x0 = np.ones(n_methods) / n_methods
        else:
            x0 = initial_guess
        
        # Objective (negative Sharpe for minimization)
        def objective(x):
            return -self._calculate_blend_sharpe(x, lookback_returns, method_functions)
        
        # Constraints: sum to 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
        
        # Bounds: 0 to 1
        bounds = [(0, 1) for _ in range(n_methods)]
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            _LOGGER.warning(f"Optimization warning: {result.message}")
        
        optimal_method_weights = result.x / result.x.sum()  # Normalize
        
        # Create result
        best_sharpe = -result.fun
        method_allocation = {methods[i]: optimal_method_weights[i] 
                           for i in range(n_methods)}
        
        optimal_factor_weights = self._weighter.blend_weights(
            {methods[i]: optimal_method_weights[i] for i in range(n_methods)}
        )
        
        portfolio_returns = self._get_portfolio_returns(optimal_factor_weights, lookback_returns)
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
            optimal_weights=optimal_factor_weights,
            sharpe_ratio=best_sharpe,
            annualized_return=ann_return,
            annualized_volatility=ann_vol,
            method_allocation=method_allocation,
            optimization_metrics={
                'success': result.success,
                'message': result.message,
                'n_iterations': result.nit
            }
        )
    
    def _differential_evolution_optimize(
        self,
        lookback: int,
        methods: List[str],
        population_size: int = 15,
        max_iterations: int = 100,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Differential evolution optimization.
        
        Good balance between thoroughness and speed. Global optimization.
        """
        if verbose:
            print(f"Running differential evolution with {len(methods)} methods...")
        
        lookback_returns = self.returns.tail(lookback)
        method_functions = self._get_method_functions(methods, lookback)
        
        n_methods = len(methods)
        
        # Objective
        def objective(x):
            # Normalize to sum to 1
            x = x / x.sum()
            return -self._calculate_blend_sharpe(x, lookback_returns, method_functions)
        
        # Bounds
        bounds = [(0, 1) for _ in range(n_methods)]
        
        # Constraints via penalty
        def penalized_objective(x):
            # Penalize if sum != 1
            sum_penalty = 1000 * (np.sum(x) - 1) ** 2
            return objective(x) + sum_penalty
        
        # Optimize
        result = differential_evolution(
            penalized_objective,
            bounds,
            maxiter=max_iterations,
            popsize=population_size,
            seed=42,
            polish=True
        )
        
        optimal_method_weights = result.x / result.x.sum()
        best_sharpe = -objective(optimal_method_weights)
        
        method_allocation = {methods[i]: optimal_method_weights[i] 
                           for i in range(n_methods)}
        
        optimal_factor_weights = self._weighter.blend_weights(
            {methods[i]: optimal_method_weights[i] for i in range(n_methods)}
        )
        
        portfolio_returns = self._get_portfolio_returns(optimal_factor_weights, lookback_returns)
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
            optimal_weights=optimal_factor_weights,
            sharpe_ratio=best_sharpe,
            annualized_return=ann_return,
            annualized_volatility=ann_vol,
            method_allocation=method_allocation,
            optimization_metrics={
                'success': result.success,
                'n_iterations': result.nit
            }
        )
    
    def _bayesian_optimize(
        self,
        lookback: int,
        methods: List[str],
        n_iterations: int = 50,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Bayesian optimization using scikit-optimize.
        
        Best for complex search spaces. Requires scikit-optimize.
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Real
        except ImportError:
            _LOGGER.warning("scikit-optimize not installed, falling back to differential evolution")
            return self._differential_evolution_optimize(lookback, methods, verbose=verbose)
        
        if verbose:
            print(f"Running Bayesian optimization with {len(methods)} methods...")
        
        lookback_returns = self.returns.tail(lookback)
        method_functions = self._get_method_functions(methods, lookback)
        
        n_methods = len(methods)
        
        # Objective
        def objective(x):
            x = np.array(x)
            x = x / x.sum()  # Normalize
            return -self._calculate_blend_sharpe(x, lookback_returns, method_functions)
        
        # Search space
        space = [Real(0, 1, name=f'w_{i}') for i in range(n_methods)]
        
        # Optimize
        result = gp_minimize(
            objective,
            space,
            n_calls=n_iterations,
            n_random_starts=10,
            random_state=42,
            verbose=verbose
        )
        
        optimal_method_weights = np.array(result.x) / np.sum(result.x)
        best_sharpe = -result.fun
        
        method_allocation = {methods[i]: optimal_method_weights[i] 
                           for i in range(n_methods)}
        
        optimal_factor_weights = self._weighter.blend_weights(
            {methods[i]: optimal_method_weights[i] for i in range(n_methods)}
        )
        
        portfolio_returns = self._get_portfolio_returns(optimal_factor_weights, lookback_returns)
        ann_return = portfolio_returns.mean() * 252
        ann_vol = portfolio_returns.std() * np.sqrt(252)
        
        return OptimizationResult(
            optimal_weights=optimal_factor_weights,
            sharpe_ratio=best_sharpe,
            annualized_return=ann_return,
            annualized_volatility=ann_vol,
            method_allocation=method_allocation
        )
    
    def _get_method_functions(
        self,
        methods: List[str],
        lookback: int
    ) -> List[Callable]:
        """Get functions that generate factor weights for each method."""
        functions = []
        
        for method in methods:
            if method == 'equal':
                fn = lambda: self._weighter.equal_weights()
            elif method == 'sharpe':
                fn = partial(self._weighter.sharpe_weights, lookback=lookback)
            elif method == 'momentum':
                fn = partial(self._weighter.momentum_weights, lookback=lookback)
            elif method == 'risk_parity':
                fn = partial(self._weighter.risk_parity_weights, lookback=lookback)
            elif method == 'min_variance':
                fn = partial(self._weighter.min_variance_weights, lookback=lookback)
            elif method == 'max_diversification':
                fn = partial(self._weighter.max_diversification_weights, lookback=lookback)
            elif method == 'pca':
                fn = lambda: self._weighter.pca_weights()
            else:
                raise ValueError(f"Unknown method: {method}")
            
            functions.append(fn)
        
        return functions
    
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
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Walk-forward optimization to avoid overfitting.
        
        Optimizes on train_window, holds weights for test_window,
        then re-optimizes. More realistic than single-period optimization.
        
        Parameters
        ----------
        train_window : int, default 126
            Window for optimization (6 months)
        test_window : int, default 21
            Window to hold weights (1 month)
        methods : List[str], optional
            Methods to blend
        technique : str, default 'differential'
            Optimization technique
        step_size : int, optional
            Step size for rolling (default = test_window)
        verbose : bool, default True
            Print progress
            
        Returns
        -------
        pd.DataFrame
            Rolling optimal weights and performance metrics
        """
        if step_size is None:
            step_size = test_window
        
        if methods is None:
            methods = ['sharpe', 'momentum', 'risk_parity']
        
        n_periods = len(self.returns)
        results = []
        
        # Generate walk-forward windows
        start_idx = train_window
        while start_idx + test_window <= n_periods:
            train_start = start_idx - train_window
            train_end = start_idx
            test_end = min(start_idx + test_window, n_periods)
            
            # Get training data
            train_returns = self.returns.iloc[train_start:train_end]
            
            # Create optimizer for this window
            window_optimizer = SharpeOptimizer(
                train_returns,
                self.loadings,
                self.risk_free_rate
            )
            
            # Optimize
            if verbose:
                print(f"\nOptimizing window {len(results)+1}: "
                      f"{self.returns.index[train_start].date()} to "
                      f"{self.returns.index[train_end-1].date()}")
            
            try:
                result = window_optimizer.optimize_blend(
                    lookback=train_window,
                    methods=methods,
                    technique=technique,
                    verbose=verbose
                )
                
                # Test out-of-sample
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
        Backtest optimized weights out-of-sample.
        
        Parameters
        ----------
        optimal_weights : Dict[str, float]
            Optimal factor weights
        test_periods : int, default 63
            Number of periods to test
        transaction_cost : float, default 0.0
            Transaction cost as fraction (0.001 = 10 bps)
        rebalance_freq : int, default 21
            Rebalancing frequency in periods
            
        Returns
        -------
        Dict
            Backtest performance metrics
        """
        # Get test data
        test_returns = self.returns.tail(test_periods)
        
        # Simulate portfolio
        portfolio_values = [1.0]
        current_weights = optimal_weights.copy()
        
        for i in range(len(test_returns)):
            # Check if rebalance needed
            if i > 0 and i % rebalance_freq == 0:
                # Apply transaction costs
                if transaction_cost > 0:
                    turnover = sum(abs(current_weights.get(f, 0) - optimal_weights.get(f, 0)) 
                                 for f in self.factor_names)
                    portfolio_values[-1] *= (1 - turnover * transaction_cost)
                current_weights = optimal_weights.copy()
            
            # Calculate daily return
            daily_ret = sum(test_returns.iloc[i][f] * current_weights.get(f, 0) 
                          for f in self.factor_names)
            portfolio_values.append(portfolio_values[-1] * (1 + daily_ret))
        
        # Calculate metrics
        portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
        
        total_return = portfolio_values[-1] - 1
        ann_return = portfolio_returns.mean() * 252
        ann_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = (ann_return / ann_vol) if ann_vol > 0 else 0
        
        # Max drawdown
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - running_max) / running_max
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
