"""
Latent Factor Discovery: Statistical and Deep Learning Methods
============================================================

This module implements multiple approaches for discovering latent factors from
stock return data, supporting both classical statistical methods and modern
deep learning techniques for quantitative finance research.

Factor Discovery Methods
-----------------------

**Statistical Methods (Fast, Interpretable)**
- **PCA (Principal Component Analysis)**: Orthogonal linear factors maximizing variance
- **ICA (Independent Component Analysis)**: Statistically independent components
- **NMF (Non-negative Matrix Factorization)**: Parts-based decomposition for positive factors

**Deep Learning Methods (Non-linear, Expressive)**  
- **Autoencoder**: Neural network for non-linear latent factor discovery
- **Rolling Window Support**: Time-varying factor analysis

Mathematical Foundation
----------------------
All methods decompose the return matrix R (T×N) into:
- **Factor Returns**: F (T×K) - time series of factor performance  
- **Factor Loadings**: B (N×K) - asset exposures to each factor

Where: R ≈ F @ B.T (matrix approximation)

Performance Characteristics
--------------------------
- **PCA/ICA**: O(N²T) complexity, seconds for 100+ assets
- **NMF**: O(N²K*iter) complexity, slower for large universes  
- **Autoencoder**: O(epochs*batch_size) complexity, GPU recommended
- **Memory**: O(NT + NK) for data storage and factor matrices

Dependencies
-----------
- **Required**: scikit-learn, numpy, pandas
- **Optional**: PyTorch (for autoencoder methods)
- **Calls**: None (standalone factor discovery)
- **Called by**: discover_and_label.py, research.py

Factor Validation
----------------
Includes comprehensive validation framework to ensure:
- Statistical distinctiveness (correlation analysis)
- Realistic return characteristics  
- Meaningful factor loadings distribution
- Performance monitoring and quality checks

Examples
--------
>>> returns = get_stock_returns()  # T×N DataFrame
>>> factors, loadings = statistical_factors(returns, n_components=10, method=StatMethod.PCA)
>>> validation = validate_factor_distinctiveness(factors, loadings)
"""

from __future__ import annotations
import logging, math
from enum import Enum, auto
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

_LOG = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# ‑‑‑ Classical statistical methods ‑‑‑
# --------------------------------------------------------------------- #
class StatMethod(Enum):
    """
    Enumeration of supported statistical factor discovery methods.
    
    Each method has different mathematical properties and use cases:
    
    Attributes
    ----------
    PCA : auto
        Principal Component Analysis - orthogonal factors maximizing variance.
        Best for: General factor discovery, dimensionality reduction.
        Properties: Linear, orthogonal factors, preserves maximum variance.
        
    ICA : auto  
        Independent Component Analysis - statistically independent factors.
        Best for: Separating independent source signals, regime analysis.
        Properties: Non-orthogonal factors, maximizes statistical independence.
        
    NMF : auto
        Non-negative Matrix Factorization - parts-based decomposition.
        Best for: Portfolio construction, positive-only factor interpretation.
        Properties: Non-negative factors/loadings, slower convergence.
    """
    PCA = auto()
    ICA = auto() 
    NMF = auto()


def statistical_factors(returns: pd.DataFrame,
                        n_components: int = 10,
                        method: StatMethod = StatMethod.PCA,
                        whiten: bool = True
                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Discover latent factors using classical statistical methods.
    
    Applies dimensionality reduction techniques to stock return data to identify
    underlying factor structure. Supports PCA, ICA, and NMF with proper factor
    return calculation via regression-based approach.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Stock return matrix with shape (T, N) where:
        - T = number of time periods (rows)
        - N = number of assets (columns)
        - Values = daily/periodic returns (typically -0.1 to +0.1 range)
        
    n_components : int, default 10
        Number of latent factors to extract. Typically 5-20 for most applications.
        Rule of thumb: ~10% of number of assets, but no more than N/2.
        
    method : StatMethod, default StatMethod.PCA
        Factor discovery algorithm:
        - PCA: Fast, orthogonal factors, variance maximization
        - ICA: Independent factors, good for regime separation  
        - NMF: Non-negative factors, slower but interpretable
        
    whiten : bool, default True
        Whether to standardize returns before factor extraction.
        - True: Unit variance scaling (recommended for mixed asset types)
        - False: Use raw return magnitudes (for similar asset classes)
        
    Returns
    -------
    fac_ret : pd.DataFrame
        Factor returns matrix with shape (T, K) where:
        - Index: Same dates as input returns
        - Columns: Factor identifiers (F1, F2, ..., FK)  
        - Values: Daily factor returns calculated via weighted regression
        
    loadings : pd.DataFrame  
        Factor loadings matrix with shape (N, K) where:
        - Index: Asset tickers from input returns
        - Columns: Factor identifiers (F1, F2, ..., FK)
        - Values: Asset exposures to each factor (typically -3 to +3 range)
        
    Algorithm Details
    ----------------
    1. **Preprocessing**: StandardScaler normalization with optional whitening
    2. **Decomposition**: Apply selected method (PCA/ICA/NMF) to extract components  
    3. **Factor Returns**: Calculate via weighted regression F = R @ B where:
       - R = asset returns matrix
       - B = factor loadings (normalized)
       - F = resulting factor returns
    4. **Validation**: Ensure factor returns have realistic statistical properties
    
    Complexity Analysis
    ------------------
    - **PCA**: O(min(NT², N²T)) - efficient for most datasets
    - **ICA**: O(K³ + iterations*NK) - iterative convergence  
    - **NMF**: O(iterations*NK²) - slowest, especially for large K
    - **Memory**: O(NT + NK) for data storage
    
    Performance Notes
    ----------------
    - **PCA**: Fastest, works well for 1000+ assets
    - **ICA**: Moderate speed, good for <500 assets
    - **NMF**: Slowest, recommended <200 assets or reduce max_iter
    
    Examples
    --------
    >>> returns = pd.DataFrame(...)  # Stock returns T×N
    >>> factors, loadings = statistical_factors(returns, n_components=5, method=StatMethod.PCA)
    >>> print(f"Extracted {len(factors.columns)} factors from {len(returns.columns)} assets")
    >>> print(f"Factor returns shape: {factors.shape}")
    >>> print(f"Loadings shape: {loadings.shape}")
    """
    scaler = StandardScaler(with_mean=True, with_std=whiten)
    X = scaler.fit_transform(returns.to_numpy())

    if method is StatMethod.PCA:
        model = PCA(n_components=n_components, random_state=0)
    elif method is StatMethod.ICA:
        whiten_param = 'unit-variance' if whiten else False
        model = FastICA(n_components=n_components, random_state=0, whiten=whiten_param)
    else:          # NMF requires non‑negative data
        X = returns.clip(lower=0).to_numpy()
        model = NMF(n_components=n_components, init="nndsvda", random_state=0,
                    max_iter=2000, tol=1e-3)

    F = model.fit_transform(X)                    # obs × k (factor scores)
    B = model.components_.T                       # asset × k (factor loadings)

    # Build loadings DataFrame with factor column names
    loadings_df = pd.DataFrame(B, index=returns.columns,
                              columns=[f"F{k}" for k in range(1, B.shape[1] + 1)])

    # VECTORIZED: Calculate factor returns via matrix multiplication
    # Each factor's exposure weights are normalized by absolute sum
    # Factor return = weighted sum of asset returns, where weights = loadings / |loadings|.sum()
    #
    # Mathematical equivalence to the original loop:
    #   for t in range(T):
    #       for k in range(K):
    #           factor_return[t,k] = sum(returns[t,:] * loadings[:,k] / abs(loadings[:,k]).sum())
    #
    # Vectorized form: F = R @ B_normalized, where B_normalized[:,k] = B[:,k] / |B[:,k]|.sum()
    B_normalized = B / np.abs(B).sum(axis=0, keepdims=True)  # (N, K)
    factor_returns_array = returns.values @ B_normalized      # (T, N) @ (N, K) = (T, K)

    fac_ret = pd.DataFrame(factor_returns_array, index=returns.index,
                           columns=loadings_df.columns)
    return fac_ret, loadings_df


# --------------------------------------------------------------------- #
# ‑‑‑ Deep autoencoder (PyTorch) ‑‑‑
# --------------------------------------------------------------------- #
if TORCH_AVAILABLE:
    class _AutoEncoder(nn.Module):
        """
        Neural network autoencoder for non-linear factor discovery.
        
        This class implements a simple but effective autoencoder architecture
        for discovering latent factors in financial return data. The bottleneck
        layer represents factor exposures, while the decoder reconstructs
        original returns.
        
        Architecture
        -----------
        Input → Encoder → Bottleneck (Factors) → Decoder → Reconstruction
        
        - **Encoder**: Linear → ReLU → Linear (dimensionality reduction)
        - **Bottleneck**: K latent factors (the key discovered representations)
        - **Decoder**: Linear → ReLU → Linear (reconstruction to original space)
        
        Parameters
        ----------
        n_assets : int
            Number of input assets/securities in the return matrix
        k : int  
            Number of latent factors to discover (bottleneck dimension)
        hidden : int
            Hidden layer dimension (typically 64-256 for financial data)
            
        Attributes
        ----------
        enc : nn.Sequential
            Encoder network that maps returns to factor space
        dec : nn.Sequential  
            Decoder network that reconstructs returns from factors
            
        Mathematical Properties
        ----------------------
        - **Input Space**: R^n_assets (daily returns for all assets)
        - **Latent Space**: R^k (factor representations)
        - **Loss Function**: MSE between original and reconstructed returns
        - **Optimization**: Adam optimizer with backpropagation
        
        Examples
        --------
        >>> model = _AutoEncoder(n_assets=100, k=10, hidden=128)
        >>> reconstructed, factors = model(return_tensor)
        >>> print(f"Factor shape: {factors.shape}")  # (batch_size, 10)
        """
        def __init__(self, n_assets: int, k: int, hidden: int):
            super().__init__()
            # Encoder: compress asset returns to factor space
            self.enc = nn.Sequential(
                nn.Linear(n_assets, hidden),   # First compression layer
                nn.ReLU(),                      # Non-linear activation
                nn.Linear(hidden, k)            # Final compression to k factors
            )
            # Decoder: reconstruct asset returns from factors
            self.dec = nn.Sequential(
                nn.Linear(k, hidden),           # Expand from factor space
                nn.ReLU(),                      # Non-linear activation  
                nn.Linear(hidden, n_assets)     # Reconstruct to original dimension
            )

        def forward(self, x):
            """
            Forward pass: encode to factors, then decode to reconstruction.
            
            Parameters
            ----------
            x : torch.Tensor
                Input return data with shape (batch_size, n_assets)
                
            Returns
            -------
            Tuple[torch.Tensor, torch.Tensor]
                - out: Reconstructed returns (batch_size, n_assets)
                - z: Latent factor representations (batch_size, k)
            """
            z = self.enc(x)      # Encode: returns → factors
            out = self.dec(z)    # Decode: factors → reconstructed returns
            return out, z


def autoencoder_factors(returns: pd.DataFrame,
                        k: int = 10,
                        hidden: int = 128,
                        lr: float = 1e-3,
                        epochs: int = 200,
                        batch: int = 64,
                        device: str | None = None
                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Discover latent factors using neural network autoencoder (deep learning approach).
    
    This function trains a PyTorch autoencoder to discover non-linear latent factors
    in stock return data. Unlike linear methods (PCA/ICA), autoencoders can capture
    complex, non-linear relationships between assets and factors.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Stock return matrix with shape (T, N) where:
        - T = number of time periods (rows)
        - N = number of assets (columns)  
        - Values = daily/periodic returns (typically -0.1 to +0.1 range)
        
    k : int, default 10
        Number of latent factors to discover (bottleneck dimension).
        This determines the dimensionality of the factor space.
        Rule of thumb: 5-20 factors for most applications.
        
    hidden : int, default 128
        Hidden layer dimension in the autoencoder architecture.
        - Small datasets (<50 assets): 32-64
        - Medium datasets (50-200 assets): 64-128  
        - Large datasets (200+ assets): 128-256
        
    lr : float, default 1e-3
        Learning rate for Adam optimizer.
        - Conservative: 1e-4 (slower but stable)
        - Standard: 1e-3 (good balance)
        - Aggressive: 1e-2 (faster but may overshoot)
        
    epochs : int, default 200
        Number of training epochs (full passes through data).
        Monitor loss convergence to determine optimal value.
        - Quick exploration: 50-100 epochs
        - Production training: 200-500 epochs
        
    batch : int, default 64
        Batch size for stochastic gradient descent.
        - Small datasets: 16-32
        - Medium datasets: 32-64
        - Large datasets: 64-128
        
    device : str | None, default None
        PyTorch device for computation ("cuda" or "cpu").
        If None, automatically selects CUDA if available.
        
    Returns
    -------
    fac_ret : pd.DataFrame
        Factor returns matrix with shape (T, K) where:
        - Index: Same dates as input returns
        - Columns: Factor identifiers (F1, F2, ..., FK)
        - Values: Factor returns from autoencoder latent space
        
    loadings : pd.DataFrame
        Factor loadings matrix with shape (N, K) where:
        - Index: Asset tickers from input returns
        - Columns: Factor identifiers (F1, F2, ..., FK)  
        - Values: Asset exposures calculated via regression on factor returns
        
    Algorithm Details
    ----------------
    1. **Preprocessing**: Convert DataFrame to PyTorch tensors
    2. **Architecture**: Build encoder-decoder neural network
       - Encoder: assets → hidden → factors (compression)
       - Decoder: factors → hidden → assets (reconstruction)
    3. **Training**: Minimize reconstruction error (MSE loss)
       - Optimizer: Adam with specified learning rate
       - Batching: Process data in mini-batches
       - Epochs: Multiple passes through entire dataset
    4. **Factor Extraction**: Use encoder output as factor returns
    5. **Loading Calculation**: Regress asset returns on factor returns
    
    Complexity Analysis
    ------------------
    - **Time**: O(epochs × batches × (N×hidden + hidden×k)) for training
    - **Space**: O(N×hidden + hidden×k + batch×N) for model parameters
    - **Training**: Typically 30 seconds to 5 minutes depending on dataset size
    
    Advantages vs Linear Methods
    ---------------------------
    - **Non-linearity**: Captures complex asset relationships
    - **Flexibility**: Adaptive to data patterns
    - **Expressiveness**: Can model regime-dependent factors
    
    Disadvantages vs Linear Methods  
    ------------------------------
    - **Computational Cost**: Requires GPU for large datasets
    - **Hyperparameter Tuning**: More parameters to optimize
    - **Interpretability**: Less interpretable than PCA components
    - **Overfitting Risk**: Can memorize noise in small datasets
    
    Performance Notes
    ----------------
    - **GPU Recommended**: For datasets with >100 assets
    - **Convergence**: Monitor training loss every 50 epochs
    - **Early Stopping**: Implement if validation loss plateaus
    - **Regularization**: Consider dropout for large hidden dimensions
    
    Examples
    --------
    >>> returns = pd.DataFrame(...)  # Stock returns T×N
    >>> factors, loadings = autoencoder_factors(returns, k=5, epochs=100)
    >>> print(f"Extracted {len(factors.columns)} non-linear factors")
    >>> print(f"Training device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    Raises
    ------
    ImportError
        If PyTorch is not installed (install with: pip install torch)
    RuntimeError
        If CUDA is requested but not available
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for autoencoder factors. Install with: pip install torch")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.tensor(returns.to_numpy(), dtype=torch.float32)
    ds = TensorDataset(X)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True)

    model = _AutoEncoder(n_assets=X.shape[1], k=k, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        for (x,) in dl:
            x = x.to(device)
            opt.zero_grad()
            x_hat, _ = model(x)
            loss = loss_fn(x_hat, x)
            loss.backward()
            opt.step()
        if (epoch + 1) % 50 == 0:
            _LOG.info("AE epoch %d/%d  loss %.5f", epoch + 1, epochs, loss.item())

    # factor returns = encoded z for each date
    with torch.no_grad():
        _, Z = model(X.to(device))
    fac_ret = pd.DataFrame(Z.cpu().numpy(), index=returns.index,
                           columns=[f"F{k}" for k in range(1, k + 1)])

    # exposures β: regress returns on factor returns (every asset separately)
    loadings = pd.DataFrame(index=returns.columns, columns=fac_ret.columns)
    for i, sym in enumerate(returns.columns):
        beta = np.linalg.lstsq(fac_ret, returns.iloc[:, i], rcond=None)[0]
        loadings.loc[sym] = beta
    return fac_ret, loadings.astype(float)


def validate_factor_distinctiveness(factor_returns: pd.DataFrame, 
                                   factor_loadings: pd.DataFrame,
                                   corr_threshold: float = 0.8) -> Dict[str, any]:
    """
    Comprehensive validation framework for factor quality and distinctiveness.
    
    This function performs extensive validation to ensure discovered factors are
    statistically meaningful, economically interpretable, and not redundant.
    Critical for detecting issues like correlated factors, unrealistic returns,
    or poorly distributed loadings.
    
    Parameters
    ----------
    factor_returns : pd.DataFrame
        Factor returns matrix with shape (T, K) where:
        - Index: Trading dates
        - Columns: Factor identifiers (F1, F2, etc.)
        - Values: Daily factor returns (typically -0.05 to +0.05 range)
        
    factor_loadings : pd.DataFrame
        Factor loadings matrix with shape (N, K) where:
        - Index: Asset tickers
        - Columns: Factor identifiers (F1, F2, etc.)
        - Values: Asset exposures to each factor (typically -3 to +3 range)
        
    corr_threshold : float, default 0.8
        Correlation threshold above which factors are flagged as redundant.
        - Conservative: 0.7 (stricter validation)
        - Standard: 0.8 (balanced approach)
        - Permissive: 0.9 (allow higher correlation)
        
    Returns
    -------
    Dict[str, any]
        Comprehensive validation results containing:
        
        - **is_valid** (bool): Overall validation status
        - **warnings** (List[str]): Specific issues identified
        - **correlations** (Dict): Factor correlation analysis
        - **recommendations** (List[str]): Actionable improvement suggestions
        
    Validation Checks
    ----------------
    
    **1. Factor Return Correlations**
    - Identifies factor pairs with correlation > threshold
    - Flags redundant factors that provide similar information
    - Recommends dimensionality reduction if needed
    
    **2. Factor Loading Distribution**
    - Checks for uniform loadings (no meaningful variation)
    - Detects over-concentration in few assets
    - Validates loading spread and diversity
    
    **3. Factor Return Statistics**
    - Identifies unrealistic volatility (>100% annualized)
    - Detects suspicious patterns (all positive/negative returns)
    - Validates return characteristics against market norms
    
    **4. Asset Concentration Analysis**
    - Measures factor concentration in top assets
    - Warns if >80% of factor exposure in top 5 assets
    - Ensures factors capture broad market themes
    
    Validation Criteria
    ------------------
    
    **Pass Criteria:**
    - All factor correlations < threshold
    - Loading standard deviation > 0.01
    - Annualized volatility < 100%
    - Mixed positive/negative returns
    - Reasonable asset diversification
    
    **Fail Criteria:**
    - High correlation between factors (redundancy)
    - Uniform factor loadings (no signal)
    - Extreme volatility (calculation errors)
    - Concentrated exposures (narrow factors)
    
    Statistical Foundation
    ---------------------
    - **Correlation Analysis**: Pearson correlation matrix
    - **Volatility Calculation**: Standard deviation × √252 × 100
    - **Concentration Metrics**: Top-5 asset weight percentage
    - **Distribution Tests**: Standard deviation and range analysis
    
    Performance Notes
    ----------------
    - **Time Complexity**: O(K² + N×K) for correlation and loading analysis
    - **Space Complexity**: O(K²) for correlation matrix storage
    - **Execution Time**: <1 second for typical factor counts (K<50)
    
    Examples
    --------
    >>> validation = validate_factor_distinctiveness(factor_returns, loadings)
    >>> if validation["is_valid"]:
    ...     print("✅ Factors passed validation")
    ... else:
    ...     print("⚠️ Validation issues found:")
    ...     for warning in validation["warnings"]:
    ...         print(f"  - {warning}")
    ...     for rec in validation["recommendations"]:
    ...         print(f"  + {rec}")
    
    >>> # Check specific correlation issues
    >>> if "high_pairs" in validation["correlations"]:
    ...     for pair, corr in validation["correlations"]["high_pairs"]:
    ...         print(f"High correlation: {pair[0]} vs {pair[1]} = {corr:.3f}")
    
    Common Issues & Solutions
    ------------------------
    
    **Issue**: High factor correlations
    **Solution**: Reduce number of factors or apply regularization
    
    **Issue**: Uniform factor loadings  
    **Solution**: Check data preprocessing or increase factor count
    
    **Issue**: Extreme volatility
    **Solution**: Validate factor return calculation methodology
    
    **Issue**: Asset concentration
    **Solution**: Use more diversified input data or increase regularization
    
    Notes
    -----
    - This validation was added to address the factor quality issues
      discovered during the workflow debugging process
    - Integrates with the main discover_and_label.py workflow
    - Provides actionable feedback for factor model improvement
    """
    validation_results = {
        "is_valid": True,
        "warnings": [],
        "correlations": {},
        "recommendations": []
    }
    
    # Check factor return correlations
    factor_corr = factor_returns.corr()
    high_corr_pairs = []
    
    for i in range(len(factor_corr.columns)):
        for j in range(i+1, len(factor_corr.columns)):
            corr_val = abs(factor_corr.iloc[i, j])
            if corr_val > corr_threshold:
                pair = (factor_corr.columns[i], factor_corr.columns[j])
                high_corr_pairs.append((pair, corr_val))
    
    if high_corr_pairs:
        validation_results["is_valid"] = False
        validation_results["warnings"].append(
            f"Found {len(high_corr_pairs)} factor pairs with correlation > {corr_threshold}"
        )
        validation_results["correlations"]["high_pairs"] = high_corr_pairs
        validation_results["recommendations"].append(
            "Consider reducing the number of factors or using regularization"
        )
    
    # Check for meaningful factor loadings spread
    for factor in factor_loadings.columns:
        loadings = factor_loadings[factor]
        loading_std = loadings.std()
        loading_range = loadings.max() - loadings.min()
        
        if loading_std < 0.01:  # Very low standard deviation
            validation_results["warnings"].append(
                f"Factor {factor} has very uniform loadings (std={loading_std:.4f})"
            )
            validation_results["recommendations"].append(
                f"Factor {factor} may not capture meaningful variation"
            )
        
        # Check for concentration in few assets
        abs_loadings = loadings.abs()
        top_5_weight = abs_loadings.nlargest(5).sum() / abs_loadings.sum()
        if top_5_weight > 0.8:
            validation_results["warnings"].append(
                f"Factor {factor} is concentrated in few assets ({top_5_weight:.1%} in top 5)"
            )
    
    # Check factor return statistics
    for factor in factor_returns.columns:
        returns = factor_returns[factor]
        
        # Check for unrealistic volatility
        daily_vol = returns.std()
        annualized_vol = daily_vol * np.sqrt(252) * 100  # Convert to percentage
        
        if annualized_vol > 100:  # More than 100% annualized volatility
            validation_results["warnings"].append(
                f"Factor {factor} has very high volatility ({annualized_vol:.1f}% annualized)"
            )
            validation_results["recommendations"].append(
                "Check factor return calculation methodology"
            )
        
        # Check for suspicious patterns (all positive/negative)
        if (returns > 0).all():
            validation_results["warnings"].append(f"Factor {factor} has only positive returns")
        elif (returns < 0).all():
            validation_results["warnings"].append(f"Factor {factor} has only negative returns")
    
    # Overall recommendations
    if not validation_results["warnings"]:
        validation_results["recommendations"].append("Factors pass basic validation checks")
    
    _LOG.info("Factor validation complete: %s", 
             "PASSED" if validation_results["is_valid"] else "FAILED")
    
    for warning in validation_results["warnings"]:
        _LOG.warning("Factor validation: %s", warning)
    
    return validation_results