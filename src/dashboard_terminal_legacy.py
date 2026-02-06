"""
Alpha Command Center: Factor Operations Terminal
=================================================
An institutional-grade dashboard for Portfolio Managers with $1B+ AUM.

Core Philosophy:
1. Alpha vs. Beta Separation: Every screen distinguishes "Market Drift" vs "True Latent Alpha"
2. Explainability First: Never show "Factor 3." Show "Factor 3 (Tech-Momentum)" 
3. Actionable Intelligence: Charts lead to decisions (Rebalance, Hedge, or De-risk)

Architecture:
- Section 1: "Morning Coffee" Header (Situational Awareness)
- Section 2: Factor Lab (Discovery & Understanding with XAI)
- Section 3: Portfolio Constructor (Execution & "What-If" Analysis)
- Section 4: Risk & Drawdown (Monitoring & Attribution)

Author: Factor Ops System
Version: 2.0 - Institutional Grade
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import system modules - handle both module and script execution
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.latent_factors import statistical_factors, StatMethod, residualize_returns
    from src.regime_detection import RegimeDetector, MarketRegime
    from src.factor_optimization import SharpeOptimizer, OptimizationResult
    from src.factor_weighting import OptimalFactorWeighter
    from src.covariance import estimate_covariance, CovarianceMethod
    from src.trading_signals import FactorMomentumAnalyzer
    from src.cross_sectional import CrossSectionalAnalyzer
    from src.database import check_database_health, get_db_connection
    from src.config import config
    from src.factor_labeler import batch_name_factors, FactorName
    # Phase 2 imports
    from src.signal_aggregator import MetaModelAggregator, FeatureExtractor
    XGBOOST_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import system modules: {e}")
    st.info("Make sure you're running from the project root directory")
    raise

# Check for optional XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# =============================================================================
# PAGE CONFIGURATION - Terminal Style
# =============================================================================
st.set_page_config(
    page_title="AlphaCmd | Factor Operations Terminal",
    layout="wide",
    page_icon="",
    initial_sidebar_state="expanded"
)

# Custom CSS for institutional terminal look
st.markdown("""
<style>
    /* Terminal-style dark theme */
    .stApp {
        background-color: #0a0a0a;
    }
    
    /* Headers */
    h1 {
        color: #00d4ff !important;
        font-family: 'Courier New', monospace !important;
        font-weight: bold !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    h2 {
        color: #00ff88 !important;
        font-family: 'Courier New', monospace !important;
        border-left: 4px solid #00ff88;
        padding-left: 12px;
        margin-top: 30px !important;
    }
    
    h3 {
        color: #ffa500 !important;
        font-family: 'Courier New', monospace !important;
    }
    
    /* Metric cards */
    .metric-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Alert styles */
    .alert-beta {
        background-color: #3d1f1f;
        border-left: 4px solid #ff4444;
        color: #ffaaaa;
        padding: 12px;
        border-radius: 4px;
        margin: 8px 0;
    }
    
    .alert-alpha {
        background-color: #1f3d1f;
        border-left: 4px solid #44ff44;
        color: #aaffaa;
        padding: 12px;
        border-radius: 4px;
        margin: 8px 0;
    }
    
    /* Dataframes */
    .dataframe {
        font-family: 'Courier New', monospace !important;
        font-size: 12px !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a2e !important;
        border-radius: 4px 4px 0 0 !important;
        padding: 12px 24px !important;
        font-family: 'Courier New', monospace !important;
        font-weight: bold !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2a2a4a !important;
        border-bottom: 2px solid #00d4ff !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #1a1a2e !important;
        border-radius: 4px !important;
        font-family: 'Courier New', monospace !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #0f0f1a !important;
    }
    
    /* Custom metric labels */
    .metric-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #8888aa;
    }
    
    /* Status indicators */
    .status-ok { color: #00ff88; }
    .status-warn { color: #ffa500; }
    .status-danger { color: #ff4444; }
    
    /* Section dividers */
    hr {
        border-color: #2a2a4a !important;
        margin: 30px 0 !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING & INITIALIZATION
# =============================================================================

@st.cache_data(ttl=300)
def load_system_data():
    """Load all factor system data with caching."""
    data = {
        'returns': None,
        'loadings': None,
        'names': {},
        'fundamentals': None,
        'portfolio_returns': None,
        'benchmark_returns': None
    }

    def _load_series_from_csv(path: Path, preferred_names: List[str]) -> Optional[pd.Series]:
        """Load a return series from CSV, preferring explicit column names."""
        try:
            frame = pd.read_csv(path, index_col=0, parse_dates=True)
            if frame.empty:
                return None

            if frame.shape[1] == 1:
                series = frame.iloc[:, 0]
            else:
                normalized = {c.lower(): c for c in frame.columns}
                selected_col = None
                for name in preferred_names:
                    if name.lower() in normalized:
                        selected_col = normalized[name.lower()]
                        break
                if selected_col is None:
                    selected_col = frame.columns[0]
                series = frame[selected_col]

            series = series.sort_index()
            series.index = pd.to_datetime(series.index)
            return pd.to_numeric(series, errors='coerce').dropna()
        except Exception:
            return None
    
    # Load factor returns
    if Path("factor_returns.csv").exists():
        try:
            data['returns'] = pd.read_csv("factor_returns.csv", index_col=0, parse_dates=True)
        except Exception as e:
            st.error(f"Error loading factor returns: {e}")
    
    # Load factor loadings
    if Path("factor_loadings.csv").exists():
        try:
            data['loadings'] = pd.read_csv("factor_loadings.csv", index_col=0)
        except Exception as e:
            st.error(f"Error loading factor loadings: {e}")
    
    # Load factor names (from multiple sources)
    if Path("factor_names.csv").exists():
        try:
            names_df = pd.read_csv("factor_names.csv", header=None, index_col=0)
            data['names'] = names_df[1].to_dict()
        except Exception:
            pass
    elif Path("factor_names.json").exists():
        try:
            with open("factor_names.json", "r") as f:
                data['names'] = json.load(f)
        except Exception:
            pass

    # Explicit portfolio return history (required for true PM vitals)
    portfolio_files = ["portfolio_returns.csv", "strategy_returns.csv"]
    for f in portfolio_files:
        path = Path(f)
        if not path.exists():
            continue
        series = _load_series_from_csv(path, preferred_names=["portfolio", "strategy", "return"])
        if series is not None and not series.empty:
            data["portfolio_returns"] = series
            break

    # Explicit benchmark return history
    benchmark_files = ["benchmark_returns.csv", "spy_returns.csv"]
    for f in benchmark_files:
        path = Path(f)
        if not path.exists():
            continue
        series = _load_series_from_csv(path, preferred_names=["benchmark", "spy", "return"])
        if series is not None and not series.empty:
            data["benchmark_returns"] = series
            break
    
    # Load basket data for liquidity analysis
    basket_files = [
        "vthr_pca_trade_basket_07_net.csv",
        "basket_fixed.csv", 
        "vthr_pca_long_positions.csv"
    ]
    for f in basket_files:
        if Path(f).exists():
            try:
                data['latest_basket'] = pd.read_csv(f)
                break
            except Exception:
                continue
    
    return data


def initialize_analyzers(data: dict):
    """Initialize all analytical components."""
    analyzers = {}
    
    returns = data.get('returns')
    loadings = data.get('loadings')
    
    if returns is not None:
        # Regime Detector
        try:
            analyzers['regime'] = RegimeDetector(returns)
            analyzers['regime'].fit_hmm(n_regimes=4)
        except Exception as e:
            st.sidebar.warning(f"Regime detection init: {e}")
            analyzers['regime'] = None
        
        # Momentum Analyzer
        try:
            analyzers['momentum'] = FactorMomentumAnalyzer(returns)
        except Exception as e:
            analyzers['momentum'] = None
        
        # Factor Weighter
        if loadings is not None:
            try:
                analyzers['weighter'] = OptimalFactorWeighter(loadings, returns)
            except Exception:
                analyzers['weighter'] = None
    
    if loadings is not None:
        # Cross-Sectional Analyzer
        try:
            analyzers['cross'] = CrossSectionalAnalyzer(loadings)
        except Exception:
            analyzers['cross'] = None
    
    return analyzers


# =============================================================================
# SECTION 1: "MORNING COFFEE" HEADER
# =============================================================================

def create_regime_gauge_chart(regime_detector: RegimeDetector) -> go.Figure:
    """Create a professional regime gauge chart."""
    if regime_detector is None or regime_detector.hmm_model is None:
        fig = go.Figure()
        fig.add_annotation(
            text="Regime detection unavailable",
            showarrow=False,
            font=dict(size=14, color="#666")
        )
        return fig
    
    current = regime_detector.detect_current_regime()
    probs = regime_detector.get_regime_probabilities()
    
    # Color mapping for regimes
    regime_colors = {
        MarketRegime.LOW_VOL_BULL: "#00ff88",
        MarketRegime.HIGH_VOL_BULL: "#88ff00",
        MarketRegime.LOW_VOL_BEAR: "#ffaa00",
        MarketRegime.HIGH_VOL_BEAR: "#ff6600",
        MarketRegime.TRANSITION: "#ffff00",
        MarketRegime.CRISIS: "#ff0000",
        MarketRegime.UNKNOWN: "#888888"
    }
    
    color = regime_colors.get(current.regime, "#888888")
    
    # Create gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current.probability * 100,
        number={'suffix': "%", 'font': {'size': 24, 'color': color}},
        title={
            'text': current.regime.value.replace('_', ' ').title(),
            'font': {'size': 16, 'color': color}
        },
        delta={'reference': 50, 'increasing': {'color': "#00ff88"}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': color},
            'bar': {'color': color},
            'bgcolor': "#1a1a2e",
            'borderwidth': 2,
            'bordercolor': "#2a2a4a",
            'steps': [
                {'range': [0, 33], 'color': "#3d1f1f"},
                {'range': [33, 66], 'color': "#3d3d1f"},
                {'range': [66, 100], 'color': "#1f3d1f"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': current.probability * 100
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'family': "Courier New"}
    )
    
    return fig


def calculate_portfolio_vitals(
    returns: pd.DataFrame,
    loadings: pd.DataFrame,
    analyzers: dict,
    lookback: int = 63,
    portfolio_returns: Optional[pd.Series] = None,
    benchmark_returns: Optional[pd.Series] = None
) -> Dict[str, any]:
    """Calculate portfolio vitals from explicit portfolio and benchmark history."""
    vitals = {
        'est_beta': np.nan,
        'tracking_error': np.nan,
        'unexplained_pnl': np.nan,
        'info_ratio': np.nan,
        'gross_leverage': 1.0,
        'net_exposure': 0.0,
        'active_risk': np.nan
    }

    if portfolio_returns is None or portfolio_returns.empty:
        return vitals

    portfolio_returns = pd.to_numeric(portfolio_returns, errors='coerce').dropna()
    if portfolio_returns.empty:
        return vitals

    portfolio_returns = portfolio_returns.tail(lookback)

    if benchmark_returns is not None:
        benchmark_returns = pd.to_numeric(benchmark_returns, errors='coerce').dropna()
        merged = pd.concat(
            [portfolio_returns.rename('portfolio'), benchmark_returns.rename('benchmark')],
            axis=1,
            join='inner'
        ).dropna()
        merged = merged.tail(lookback)

        if not merged.empty:
            active = merged['portfolio'] - merged['benchmark']
            tracking_error = active.std(ddof=0) * np.sqrt(252)
            vitals['tracking_error'] = float(tracking_error)
            vitals['active_risk'] = float(tracking_error)

            denom = active.std(ddof=0)
            if denom > 0:
                vitals['info_ratio'] = float((active.mean() / denom) * np.sqrt(252))

            market_var = merged['benchmark'].var(ddof=0)
            if market_var > 0:
                beta = merged['portfolio'].cov(merged['benchmark']) / market_var
                vitals['est_beta'] = float(beta)
    else:
        # Without benchmark data we can still show absolute portfolio volatility.
        portfolio_vol = portfolio_returns.std(ddof=0) * np.sqrt(252)
        vitals['active_risk'] = float(portfolio_vol)

    # Residual risk proxy via factor attribution: portfolio return ~ factor returns.
    if returns is not None and not returns.empty and len(returns.columns) > 0:
        factor_frame = returns.loc[portfolio_returns.index].copy().replace([np.inf, -np.inf], np.nan)
        factor_frame = factor_frame.dropna(axis=1, how='all')
        if not factor_frame.empty:
            merged = pd.concat([portfolio_returns.rename('portfolio'), factor_frame], axis=1).dropna()
            if len(merged) >= 10 and merged['portfolio'].var(ddof=0) > 0:
                y = merged['portfolio'].values
                X = merged.drop(columns=['portfolio']).values
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                # Add intercept and estimate residual variance ratio.
                X = np.column_stack([np.ones(len(X)), X])
                coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                residuals = y - X @ coef
                unexplained = residuals.var(ddof=0) / y.var(ddof=0)
                vitals['unexplained_pnl'] = float(np.clip(unexplained, 0.0, 1.0))
    
    return vitals


def render_morning_coffee_header(data: dict, analyzers: dict):
    """Render the 'Morning Coffee' header section."""
    st.markdown("<h1> ALPHACMD | FACTOR OPERATIONS TERMINAL</h1>", unsafe_allow_html=True)
    
    returns = data.get('returns')
    loadings = data.get('loadings')
    regime_detector = analyzers.get('regime')
    
    # Calculate vitals
    vitals = calculate_portfolio_vitals(
        returns,
        loadings,
        analyzers,
        portfolio_returns=data.get('portfolio_returns'),
        benchmark_returns=data.get('benchmark_returns')
    )
    
    # Create header columns
    col1, col2, col3, col4, col5 = st.columns([1.2, 1, 1, 1, 1])
    
    with col1:
        # Market Regime Gauge
        st.markdown("<h3 style='color:#00d4ff; font-size:14px; margin-bottom:5px;'> MARKET REGIME</h3>", 
                   unsafe_allow_html=True)
        fig = create_regime_gauge_chart(regime_detector)
        st.plotly_chart(fig, use_container_width=True, key="regime_gauge")
    
    with col2:
        st.markdown("<h3 style='color:#00d4ff; font-size:14px; margin-bottom:5px;'> ACTIVE RISK</h3>", 
                   unsafe_allow_html=True)
        risk_available = pd.notna(vitals['active_risk'])
        risk_value = f"{vitals['active_risk']:.2%}" if risk_available else "N/A"
        risk_delta = "Target: 4-6%" if pd.notna(vitals['tracking_error']) else "Need Portfolio/Benchmark"
        st.metric(
            label="Tracking Error",
            value=risk_value,
            delta=risk_delta,
            delta_color="off"
        )
        st.caption("Based on explicit portfolio + benchmark history")
        
        # Risk status indicator
        if risk_available:
            risk_status = "🟢" if 0.04 <= vitals['active_risk'] <= 0.06 else "🟡" if vitals['active_risk'] < 0.08 else ""
            risk_text = "In Range" if 0.04 <= vitals['active_risk'] <= 0.06 else "Review Required"
        else:
            risk_status = ""
            risk_text = "Need Data"
        st.markdown(f"**Status:** {risk_status} {risk_text}")
    
    with col3:
        st.markdown("<h3 style='color:#00d4ff; font-size:14px; margin-bottom:5px;'> ESTIMATED BETA</h3>", 
                   unsafe_allow_html=True)
        beta_available = pd.notna(vitals['est_beta'])
        beta_color = "normal" if beta_available and abs(vitals['est_beta']) < 0.1 else "inverse"
        beta_value = f"{vitals['est_beta']:.3f}" if beta_available else "N/A"
        beta_delta = (
            "Market Neutral"
            if beta_available and abs(vitals['est_beta']) < 0.1
            else "Need Benchmark" if not beta_available
            else "Review Hedge"
        )
        st.metric(
            label="Market Beta",
            value=beta_value,
            delta=beta_delta,
            delta_color=beta_color
        )
        st.caption("Target: ±0.1 for Market Neutral")
    
    with col4:
        st.markdown("<h3 style='color:#00d4ff; font-size:14px; margin-bottom:5px;'> UNEXPLAINED PnL</h3>", 
                   unsafe_allow_html=True)
        unexplained_available = pd.notna(vitals['unexplained_pnl'])
        ghost_color = "normal" if unexplained_available and vitals['unexplained_pnl'] < 0.3 else "inverse"
        unexplained_value = f"{vitals['unexplained_pnl']:.1%}" if unexplained_available else "N/A"
        unexplained_delta = (
            "Need Portfolio + Factor Returns"
            if not unexplained_available
            else "Clean Attribution" if vitals['unexplained_pnl'] < 0.3
            else "Check Factors"
        )
        st.metric(
            label="Ghost Alpha/Risk",
            value=unexplained_value,
            delta=unexplained_delta,
            delta_color=ghost_color
        )
        st.caption("Not explained by factors")
    
    with col5:
        st.markdown("<h3 style='color:#00d4ff; font-size:14px; margin-bottom:5px;'> INFO RATIO</h3>", 
                   unsafe_allow_html=True)
        ir_available = pd.notna(vitals['info_ratio'])
        ir_value = f"{vitals['info_ratio']:.2f}" if ir_available else "N/A"
        ir_delta = (
            "Need Benchmark"
            if not ir_available
            else "Strong" if vitals['info_ratio'] > 0.5
            else "Developing" if vitals['info_ratio'] > 0
            else "Review"
        )
        st.metric(
            label="Information Ratio",
            value=ir_value,
            delta=ir_delta,
            delta_color="normal" if ir_available and vitals['info_ratio'] > 0.5 else "off"
        )
        st.caption("Risk-adjusted alpha")
    
    st.markdown("---")


# =============================================================================
# SECTION 2: FACTOR LAB (Discovery & Understanding)
# =============================================================================

def calculate_factor_purity(returns: pd.DataFrame, factor_col: str) -> float:
    """Calculate factor purity as 1 - correlation to market proxy."""
    if len(returns.columns) < 2:
        return 0.0
    
    # Use mean of all other factors as market proxy
    other_factors = [c for c in returns.columns if c != factor_col]
    if not other_factors:
        return 0.0
    
    market_proxy = returns[other_factors].mean(axis=1)
    correlation = returns[factor_col].corr(market_proxy)
    
    # Purity = 1 - |correlation| (higher = more alpha, less beta)
    purity = 1 - abs(correlation)
    return purity


def calculate_style_attribution(factor_returns: pd.Series, all_returns: pd.DataFrame) -> Dict[str, float]:
    """Regress factor against style proxies and return attribution."""
    # Simple style attribution using factor correlations
    styles = {}
    
    if len(all_returns.columns) < 4:
        return {'Idiosyncratic': 1.0}
    
    # Use first few factors as style proxies
    # F1 ~ Market, F2 ~ Value/Growth, F3 ~ Momentum, etc.
    style_names = ['Market', 'Value', 'Momentum', 'Size', 'Quality']
    
    for i, style in enumerate(style_names[:min(5, len(all_returns.columns))]):
        if all_returns.columns[i] != factor_returns.name:
            corr = factor_returns.corr(all_returns.iloc[:, i])
            styles[style] = abs(corr)
    
    # Normalize
    total = sum(styles.values())
    if total > 0:
        styles = {k: v/total for k, v in styles.items()}
    
    # Calculate idiosyncratic
    styles['Idiosyncratic'] = max(0, 1 - sum(styles.values()))
    
    return styles


def calculate_sector_exposure_from_fundamentals(
    loadings: pd.Series,
    fundamentals: pd.DataFrame
) -> Dict[str, float]:
    """Calculate actual sector exposure using fundamental data.
    
    Weights each stock's sector by its absolute factor loading.
    """
    if fundamentals.empty or 'Sector' not in fundamentals.columns:
        return {}
    
    sector_exposure = {}
    total_weight = 0
    
    for ticker in loadings.index:
        if ticker in fundamentals.index:
            sector = fundamentals.loc[ticker].get('Sector')
            if sector and not pd.isna(sector):
                # Weight by absolute loading
                weight = abs(loadings[ticker])
                sector_exposure[sector] = sector_exposure.get(sector, 0) + weight
                total_weight += weight
    
    # Normalize to percentages
    if total_weight > 0:
        sector_exposure = {k: v/total_weight for k, v in sector_exposure.items()}
    
    # Sort by exposure
    return dict(sorted(sector_exposure.items(), key=lambda x: x[1], reverse=True))


def get_sector_exposure(loadings: pd.DataFrame, factor_col: str) -> Dict[str, float]:
    """Get sector exposure for a factor (requires sector data)."""
    # Placeholder - would need sector mapping in production
    return {}


def analyze_factor_characteristics(loadings: pd.Series) -> Dict[str, any]:
    """Deep analysis of factor characteristics for naming.
    
    Returns detailed statistics about the factor's structure.
    """
    if loadings is None or loadings.empty:
        return {}
    
    top_pos = loadings.nlargest(20)
    top_neg = loadings.nsmallest(20)
    
    # Count actual longs and shorts
    pos_count = (loadings > 0).sum()
    neg_count = (loadings < 0).sum()
    
    # Calculate percentiles
    pos_95 = loadings[loadings > 0].quantile(0.95) if pos_count > 0 else 0
    neg_5 = loadings[loadings < 0].quantile(0.05) if neg_count > 0 else 0
    
    # Top tickers
    top_5_pos = list(top_pos.head(5).index)
    top_5_neg = list(top_neg.head(5).index)
    
    # Calculate skew and concentration
    loadings_abs = loadings.abs()
    hhi = ((loadings_abs / loadings_abs.sum()) ** 2).sum()  # Herfindahl index
    
    return {
        'pos_count': pos_count,
        'neg_count': neg_count,
        'pos_95': pos_95,
        'neg_5': neg_5,
        'top_5_pos': top_5_pos,
        'top_5_neg': top_5_neg,
        'concentration': hhi,
        'skew': loadings.skew(),
        'dispersion': pos_95 - neg_5
    }


def generate_factor_description(factor_id: str, loadings: pd.Series) -> Dict[str, str]:
    """Generate meaningful factor description based on loadings analysis.
    
    Creates descriptive, unique names based on the actual factor structure.
    Names are derived from the top holdings to ensure distinctiveness.
    """
    if loadings is None or loadings.empty:
        return {
            'short_name': f"{factor_id}: Unnamed",
            'theme': 'Unknown',
            'long_desc': 'No data available',
            'short_desc': 'No data available',
            'rationale': 'Insufficient loading data'
        }
    
    stats = analyze_factor_characteristics(loadings)
    
    top_pos = loadings.nlargest(10)
    top_neg = loadings.nsmallest(10)
    
    # Unpack stats
    pos_count = stats['pos_count']
    neg_count = stats['neg_count']
    top_5_pos = stats['top_5_pos']
    top_5_neg = stats['top_5_neg']
    concentration = stats['concentration']
    dispersion = stats['dispersion']
    skew = stats['skew']
    
    # Extract ticker symbols for naming
    top_pos_ticker = top_5_pos[0] if top_5_pos else "None"
    top_neg_ticker = top_5_neg[0] if top_5_neg else "None"
    second_pos = top_5_pos[1] if len(top_5_pos) > 1 else None
    second_neg = top_5_neg[1] if len(top_5_neg) > 1 else None
    
    # Determine naming strategy based on factor characteristics
    top_pos_loading = top_pos.iloc[0] if not top_pos.empty else 0
    top_neg_loading = abs(top_neg.iloc[0]) if not top_neg.empty else 0
    second_pos_loading = top_pos.iloc[1] if len(top_pos) > 1 else 0
    
    # Is it dominated by a single stock?
    is_single_stock_dominated = top_pos_loading > 1.5 * second_pos_loading if second_pos_loading > 0 else False
    
    # Is it a clear pair trade (top long vs top short)?
    is_pair_trade = top_neg_loading > 0.03 and not is_single_stock_dominated
    
    # Is it a sector/theme basket?
    is_basket = concentration < 0.08 and len([x for x in top_pos.head(5) if x > 0.02]) >= 3
    
    # Generate unique name based on structure
    if is_single_stock_dominated:
        # Single stock factor
        short_name = f"{factor_id}: {top_pos_ticker[:10]} Tilt"
        theme = "Single Stock Exposure"
        
    elif is_pair_trade and top_neg_ticker != "None":
        # Pair trade - use both tickers
        # Make the name unique by including loadings
        short_name = f"{factor_id}: {top_pos_ticker[:5]}>{top_neg_ticker[:5]}"
        theme = "Pair Trade"
        
    elif is_basket:
        # Multi-stock basket - name after top 2-3 stocks
        if second_pos:
            short_name = f"{factor_id}: {top_pos_ticker[:4]}+{second_pos[:4]} Basket"
        else:
            short_name = f"{factor_id}: {top_pos_ticker[:8]} Basket"
        theme = "Thematic Basket"
        
    elif pos_count > neg_count * 3:
        # Heavily long biased
        short_name = f"{factor_id}: Long {top_pos_ticker[:8]}"
        theme = "Directional Long"
        
    elif neg_count > pos_count * 3:
        # Heavily short biased
        short_name = f"{factor_id}: Short {top_neg_ticker[:8]}"
        theme = "Directional Short"
        
    elif skew > 1.0:
        # Right skewed - more extreme positive values
        short_name = f"{factor_id}: {top_pos_ticker[:6]} Momentum"
        theme = "Momentum Factor"
        
    elif skew < -1.0:
        # Left skewed - more extreme negative values
        short_name = f"{factor_id}: {top_neg_ticker[:6]} Pressure"
        theme = "Pressure Factor"
        
    else:
        # Generic but descriptive
        if top_neg_ticker != "None" and top_pos_ticker != "None":
            short_name = f"{factor_id}: {top_pos_ticker[:4]} vs {top_neg_ticker[:4]}"
            theme = "Relative Value"
        else:
            short_name = f"{factor_id}: {top_pos_ticker[:10]} Focus"
            theme = "Concentrated"
    
    # Generate detailed rationale
    pos_tickers = ', '.join(top_5_pos[:5])
    neg_tickers = ', '.join(top_5_neg[:5])
    
    rationale = (
        f"{theme}: {pos_count} longs vs {neg_count} shorts. "
        f"Long basket: {pos_tickers}. "
        f"Short basket: {neg_tickers}. "
        f"HHI: {concentration:.3f}, Skew: {skew:.2f}."
    )
    
    return {
        'short_name': short_name,
        'theme': theme,
        'long_desc': f"Long: {pos_tickers}",
        'short_desc': f"Short: {neg_tickers}",
        'rationale': rationale,
        'top_pos': top_5_pos,
        'top_neg': top_5_neg,
        'dispersion': dispersion,
        'stats': stats
    }

# Add these functions after the generate_factor_description function in dashboard_alpha_command_center.py

@st.cache_data(ttl=3600)
def fetch_fundamentals_for_tickers(tickers: List[str]) -> pd.DataFrame:
    """Fetch fundamental data from Alpha Vantage for given tickers."""
    try:
        from src.alphavantage_system import DataBackend
        from src.config import config
        
        api_key = config.ALPHAVANTAGE_API_KEY
        if not api_key:
            st.warning("ALPHAVANTAGE_API_KEY not configured")
            return pd.DataFrame()
        
        backend = DataBackend(api_key)
        
        fields = [
            'Name', 'Sector', 'Industry', 'Description',
            'MarketCapitalization', 'PERatio', 'PriceToSalesRatioTTM',
            'PriceToBookRatio', 'DividendYield', 'Beta',
            'EPS', 'ROE', 'RevenueTTM', 'GrossProfitTTM',
            'OperatingMarginTTM', 'ProfitMargin', 'AnalystTargetPrice'
        ]
        
        with st.spinner(f"Fetching fundamentals for {len(tickers)} tickers..."):
            fundamentals = backend.get_fundamentals(tickers, fields=fields)
        
        return fundamentals
        
    except Exception as e:
        st.error(f"Failed to fetch fundamentals: {e}")
        return pd.DataFrame()


def create_enriched_llm_prompt(
    factor_id: str,
    loadings: pd.Series,
    fundamentals: pd.DataFrame,
    factor_returns: pd.Series = None,
    all_returns: pd.DataFrame = None
) -> str:
    """Create an enriched LLM prompt with fundamental data, style attribution, and sector tilt."""
    top_pos = loadings.nlargest(10)
    top_neg = loadings.nsmallest(10)
    
    # Calculate style attribution if returns data available
    style_attribution = {}
    if factor_returns is not None and all_returns is not None:
        style_attribution = calculate_style_attribution(factor_returns, all_returns)
    
    # Calculate sector exposure from fundamentals
    sector_exposure = calculate_sector_exposure_from_fundamentals(loadings, fundamentals)
    
    def format_stock_profile(ticker: str, loading: float) -> str:
        profile = f"  - {ticker} (loading: {loading:.4f})"
        
        if ticker in fundamentals.index:
            f = fundamentals.loc[ticker]
            
            name = f.get('Name', 'N/A')
            if name != 'N/A':
                profile += f"\n      Company: {name}"
            
            sector = f.get('Sector', 'N/A')
            industry = f.get('Industry', 'N/A')
            if sector != 'N/A':
                profile += f"\n      Sector: {sector} | Industry: {industry}"
            
            desc = f.get('Description', '')
            if desc and isinstance(desc, str):
                first_sentence = desc.split('.')[0][:150]
                profile += f"\n      Business: {first_sentence}"
            
            mcap = f.get('MarketCapitalization', None)
            if mcap and not pd.isna(mcap):
                if mcap > 1e12:
                    mcap_str = f"${mcap/1e12:.1f}T"
                elif mcap > 1e9:
                    mcap_str = f"${mcap/1e9:.1f}B"
                else:
                    mcap_str = f"${mcap/1e6:.1f}M"
                profile += f"\n      Market Cap: {mcap_str}"
            
            pe = f.get('PERatio', None)
            ps = f.get('PriceToSalesRatioTTM', None)
            pb = f.get('PriceToBookRatio', None)
            
            valuations = []
            if pe and not pd.isna(pe) and pe > 0:
                valuations.append(f"P/E: {pe:.1f}")
            if ps and not pd.isna(ps) and ps > 0:
                valuations.append(f"P/S: {ps:.1f}")
            if pb and not pd.isna(pb) and pb > 0:
                valuations.append(f"P/B: {pb:.1f}")
            
            if valuations:
                profile += f"\n      Valuation: {', '.join(valuations)}"
            
            roe = f.get('ROE', None)
            margin = f.get('ProfitMargin', None)
            
            profitability = []
            if roe and not pd.isna(roe):
                profitability.append(f"ROE: {roe*100:.1f}%")
            if margin and not pd.isna(margin):
                profitability.append(f"Margin: {margin*100:.1f}%")
            
            if profitability:
                profile += f"\n      Profitability: {', '.join(profitability)}"
            
            div_yield = f.get('DividendYield', None)
            if div_yield and not pd.isna(div_yield) and div_yield > 0:
                profile += f"\n      Dividend Yield: {div_yield*100:.1f}%"
        
        return profile
    
    high_profiles = [format_stock_profile(t, l) for t, l in top_pos.items()]
    low_profiles = [format_stock_profile(t, l) for t, l in top_neg.items()]
    
    # Aggregate analysis
    pos_mcaps = []
    neg_mcaps = []
    pos_sectors = []
    neg_sectors = []
    
    for ticker in top_pos.index:
        if ticker in fundamentals.index:
            mcap = fundamentals.loc[ticker].get('MarketCapitalization')
            if mcap and not pd.isna(mcap):
                pos_mcaps.append(mcap)
            sector = fundamentals.loc[ticker].get('Sector')
            if sector:
                pos_sectors.append(sector)
    
    for ticker in top_neg.index:
        if ticker in fundamentals.index:
            mcap = fundamentals.loc[ticker].get('MarketCapitalization')
            if mcap and not pd.isna(mcap):
                neg_mcaps.append(mcap)
            sector = fundamentals.loc[ticker].get('Sector')
            if sector:
                neg_sectors.append(sector)
    
    aggregate = []
    if pos_mcaps and neg_mcaps:
        pos_avg = np.mean(pos_mcaps)
        neg_avg = np.mean(neg_mcaps)
        aggregate.append(f"Long avg mcap: ${pos_avg/1e9:.1f}B, Short avg mcap: ${neg_avg/1e9:.1f}B")
        if pos_avg > neg_avg * 2:
            aggregate.append("Long LARGE-CAP, short small-cap")
        elif neg_avg > pos_avg * 2:
            aggregate.append("Long small-cap, short LARGE-CAP")
    
    from collections import Counter
    if pos_sectors:
        counts = Counter(pos_sectors).most_common(3)
        aggregate.append(f"Long sectors: {', '.join([f'{s}({c})' for s,c in counts])}")
    if neg_sectors:
        counts = Counter(neg_sectors).most_common(3)
        aggregate.append(f"Short sectors: {', '.join([f'{s}({c})' for s,c in counts])}")
    
    # Format style attribution
    style_lines = []
    if style_attribution:
        sorted_styles = sorted(style_attribution.items(), key=lambda x: x[1], reverse=True)
        for style, weight in sorted_styles:
            style_lines.append(f"  - {style}: {weight:.1%}")
    
    # Format sector exposure (top 5)
    sector_lines = []
    if sector_exposure:
        for sector, weight in list(sector_exposure.items())[:5]:
            sector_lines.append(f"  - {sector}: {weight:.1%}")
    
    # Factor-level statistics
    stats = analyze_factor_characteristics(loadings)
    factor_stats = []
    if stats:
        factor_stats.append(f"Total long positions: {stats.get('pos_count', 0)}")
        factor_stats.append(f"Total short positions: {stats.get('neg_count', 0)}")
        factor_stats.append(f"Concentration (HHI): {stats.get('concentration', 0):.3f}")
        factor_stats.append(f"Return skewness: {stats.get('skew', 0):.2f}")
        factor_stats.append(f"Loading dispersion: {stats.get('dispersion', 0):.3f}")
    
    prompt = f"""Analyze this financial factor based on detailed company profiles, style attribution, and sector exposure.

FACTOR: {factor_id}

FACTOR-LEVEL STATISTICS:
{chr(10).join(factor_stats) if factor_stats else "N/A"}

STYLE ATTRIBUTION (from factor return analysis):
{chr(10).join(style_lines) if style_lines else "N/A"}

SECTOR EXPOSURE (weighted by factor loadings):
{chr(10).join(sector_lines) if sector_lines else "N/A"}

AGGREGATE ANALYSIS:
{chr(10).join(aggregate) if aggregate else "N/A"}

TOP 10 LONG POSITIONS:
{chr(10).join(high_profiles) if high_profiles else "N/A"}

TOP 10 SHORT POSITIONS:
{chr(10).join(low_profiles) if low_profiles else "N/A"}

INSTRUCTIONS:
Based on all the data above (style attribution, sector exposure, individual company profiles), determine what economic theme this factor captures.

Key patterns to identify:
- STYLE factors: Is this Value vs Growth? Momentum vs Mean Reversion? Quality vs Junk?
- SECTOR rotation: Tech vs Energy? Healthcare vs Financials? Defensive vs Cyclical?
- SIZE factors: Large Cap vs Small Cap? Mega Cap vs Micro Cap?
- BUSINESS MODEL: SaaS vs Traditional? Asset-light vs Asset-heavy? High-margin vs Low-margin?
- PROFITABILITY: High ROE vs Low ROE? Stable earnings vs Volatile?

The style attribution shows how the factor correlates with known risk factors.
The sector exposure shows where the factor has net exposure.

Provide a specific, actionable name that captures the economic rationale.

Respond with JSON:
{{
  "short_name": "2-4 word specific name (e.g., 'Tech Growth vs Energy Value')",
  "description": "1 sentence economic rationale",
  "theme": "High-level theme like 'Sector Rotation' or 'Style Factor'",
  "high_exposure_desc": "What unifies the long positions (sector, style, etc.)",
  "low_exposure_desc": "What unifies the short positions (sector, style, etc.)",
  "confidence": "high|medium|low"
}}"""
    
    return prompt

@st.cache_data(ttl=3600)
def get_factor_names_with_llm(
    factor_ids: List[str],
    loadings: pd.DataFrame,
    use_llm: bool = False
) -> Dict[str, Dict[str, str]]:
    """Get factor names - either from local analysis or LLM.
    
    Args:
        factor_ids: List of factor identifiers
        loadings: Factor loadings DataFrame
        use_llm: Whether to call OpenAI API for naming
        
    Returns:
        Dictionary mapping factor_id to name info
    """
    results = {}
    
    for factor_id in factor_ids:
        if factor_id in loadings.columns:
            desc = generate_factor_description(factor_id, loadings[factor_id])
            results[factor_id] = desc
        else:
            results[factor_id] = {
                'short_name': f"{factor_id}: Unnamed",
                'theme': 'Unknown',
                'long_desc': 'N/A',
                'short_desc': 'N/A',
                'rationale': 'No loading data'
            }
    
    # If LLM is requested and available, enhance the names
    if use_llm:
        try:
            from src.factor_labeler import batch_name_factors
            # This would call the LLM - requires API key
            # For now, we use the local analysis which is quite good
            pass
        except Exception:
            pass
    
    return results


def create_factor_dna_table(
    returns: pd.DataFrame,
    loadings: pd.DataFrame,
    names: Dict[str, str],
    lookback: int = 63
) -> pd.DataFrame:
    """Create the enriched Factor DNA table."""
    if returns is None or returns.empty:
        return pd.DataFrame()
    
    # Get enhanced factor descriptions
    factor_descriptions = get_factor_names_with_llm(
        list(returns.columns), 
        loadings, 
        use_llm=False
    )
    
    recent = returns.tail(lookback)
    dna_data = []
    
    for factor in returns.columns:
        factor_rets = recent[factor]
        desc_info = factor_descriptions.get(factor, {})
        
        # Performance metrics
        total_ret = (1 + factor_rets).prod() - 1
        ann_vol = factor_rets.std() * np.sqrt(252)
        sharpe = (factor_rets.mean() * 252) / ann_vol if ann_vol > 0 else 0
        
        # Purity (alpha vs beta)
        purity = calculate_factor_purity(returns, factor)
        
        # Factor name (prefer generated name, fallback to stored name)
        factor_name = desc_info.get('short_name', names.get(factor, f"Factor {factor}"))
        theme = desc_info.get('theme', 'Unknown')
        
        # Top loading stock
        if factor in loadings.columns:
            top_stock = loadings[factor].abs().idxmax()
            top_loading = loadings[factor].loc[top_stock]
        else:
            top_stock = "N/A"
            top_loading = 0
        
        # Crowding score based on factor concentration
        if factor in loadings.columns:
            loadings_series = loadings[factor]
            # High concentration = potentially crowded
            top_10_pct = loadings_series.abs().nlargest(max(1, len(loadings_series)//10))
            crowding = top_10_pct.sum() / loadings_series.abs().sum()
        else:
            crowding = 0.0
        
        dna_data.append({
            'ID': factor,
            'Name': factor_name[:50] + "..." if len(factor_name) > 50 else factor_name,
            'Theme': theme,
            '1M Return': total_ret,
            'Sharpe': sharpe,
            'Alpha Purity': purity,
            'Correlation to SPY': 1 - purity,  # Proxy
            'Top Loading': top_stock,
            'Crowding Score': crowding
        })
    
    return pd.DataFrame(dna_data)


def render_factor_lab(data: dict, analyzers: dict):
    """Render the Factor Lab section with X-Ray view."""
    st.markdown("<h2> FACTOR LAB: Discovery & Explainability</h2>", unsafe_allow_html=True)
    
    returns = data.get('returns')
    loadings = data.get('loadings')
    names = data.get('names', {})
    
    if returns is None:
        st.error(" Factor returns data not available. Run discovery first.")
        return
    
    # Discovery Controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        st.markdown("** Discovery Settings**")
        method = st.selectbox(
            "Extraction Method",
            ["PCA (Linear)", "ICA (Independent)", "Autoencoder (Non-Linear)"],
            help="Method used to extract latent factors"
        )
        
        n_factors = st.slider("Active Factors", 5, min(20, len(returns.columns)), 10)
    
    with col2:
        st.markdown("** Beta Removal**")
        remove_beta = st.checkbox(
            "Remove Market Beta First",
            value=True,
            help="Regress out SPY before finding factors (RECOMMENDED)"
        )
        
        residualize = st.checkbox(
            "Residualize vs Sectors",
            value=True,
            help="Remove sector exposures"
        )
        
        st.info(" Beta removal prevents 'Beta in Disguise' factors")
    
    with col3:
        st.markdown("** Factor Summary Stats**")
        
        # Quick stats
        if returns is not None:
            total_factors = len(returns.columns)
            avg_sharpe = (returns.mean() * 252 / (returns.std() * np.sqrt(252))).mean()
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Factors", total_factors)
            with c2:
                st.metric("Avg Sharpe", f"{avg_sharpe:.2f}")
            with c3:
                high_beta_count = sum(1 for f in returns.columns 
                                     if calculate_factor_purity(returns, f) < 0.5)
                st.metric(" Beta Factors", high_beta_count, 
                         delta="Review" if high_beta_count > 0 else "OK",
                         delta_color="inverse" if high_beta_count > 0 else "normal")
    
    st.markdown("---")
    
    # LLM Naming Section
    st.markdown("###  Factor Naming with LLM")
    
    name_col1, name_col2, name_col3 = st.columns([2, 1, 1])
    
    with name_col1:
        st.markdown("**Auto-generate intuitive factor names using OpenAI GPT**")
        st.caption("The LLM receives top 10 long/short positions for each factor and generates descriptive names")
    
    with name_col2:
        # Check if API key is available
        import os
        api_key_available = bool(os.getenv("OPENAI_API_KEY"))
        
        if not api_key_available:
            st.warning(" OPENAI_API_KEY not set")
        else:
            st.success(" API Key ready")
    
    with name_col3:
        if st.button(" Name All Factors with LLM", type="primary", 
                    disabled=not api_key_available,
                    help="Fetch fundamental data and batch process all factors with LLM naming"):
            if not api_key_available:
                st.error("Set OPENAI_API_KEY in .env file")
            else:
                with st.spinner("Step 1/2: Fetching fundamental data for all tickers..."):
                    try:
                        from openai import OpenAI
                        import json
                        
                        # Step 1: Collect all unique tickers across all factors
                        all_tickers = set()
                        for factor in returns.columns:
                            if factor in loadings.columns:
                                all_tickers.update(loadings[factor].nlargest(10).index)
                                all_tickers.update(loadings[factor].nsmallest(10).index)
                        
                        all_tickers = list(all_tickers)
                        st.info(f"Found {len(all_tickers)} unique tickers across all factors")
                        
                        # Step 2: Fetch fundamentals for all tickers (cached)
                        fundamentals_all = fetch_fundamentals_for_tickers(all_tickers)
                        
                        if not fundamentals_all.empty:
                            st.success(f" Fetched fundamentals for {len(fundamentals_all)} tickers")
                            with st.expander("View sample fundamental data"):
                                sample_cols = ['Name', 'Sector', 'Industry', 'MarketCapitalization']
                                available_cols = [c for c in sample_cols if c in fundamentals_all.columns]
                                st.dataframe(fundamentals_all[available_cols].head(10))
                        else:
                            st.warning("Could not fetch fundamentals, proceeding with ticker symbols only")
                        
                        # Step 3: Name each factor with enriched data
                        llm_names = {}
                        llm_results = {}
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        from src.factor_labeler import batch_name_factors
                        
                        # Step 3: Name each factor with enriched data using batch processor
                        st.info(f"Starting batch naming for {len(loadings.columns)} factors using {config.OPENAI_MODEL}...")
                        
                        # Filter loadings to only those present in returns (active factors)
                        active_factors = [f for f in returns.columns if f in loadings.columns]
                        active_loadings = loadings[active_factors]
                        
                        # Call batch processing with full enrichment
                        llm_names_map = batch_name_factors(
                            factor_exposures=active_loadings,
                            fundamentals=fundamentals_all,
                            factor_returns=returns[active_factors],
                            top_n=10,
                            model=config.OPENAI_MODEL
                        )
                        
                        # Process results into display format
                        for factor, fname in llm_names_map.items():
                             llm_names[factor] = fname.short_name
                             llm_results[factor] = {
                                 'short_name': fname.short_name,
                                 'description': fname.description,
                                 'theme': fname.theme,
                                 'confidence': fname.confidence
                             }
                        
                        progress_bar.progress(1.0)
                        
                        status_text.empty()
                        
                        # Store in session state
                        st.session_state['llm_factor_names'] = llm_names
                        st.session_state['llm_factor_details'] = llm_results
                        
                        st.success(f" Named {len(llm_names)} factors with enriched data!")
                        
                        # Show results table
                        results_df = pd.DataFrame([
                            {'Factor': k, 'Name': v, 'Theme': llm_results.get(k, {}).get('theme', 'N/A'),
                             'Confidence': llm_results.get(k, {}).get('confidence', 'N/A')}
                            for k, v in llm_names.items()
                        ])
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Option to save
                        if st.button(" Save All Names to File"):
                            import json
                            with open('factor_names_llm.json', 'w') as f:
                                json.dump(llm_names, f, indent=2)
                            with open('factor_names_llm_detailed.json', 'w') as f:
                                json.dump(llm_results, f, indent=2)
                            st.success("Saved to factor_names_llm.json and factor_names_llm_detailed.json")
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"LLM naming failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())
    
    # Show LLM prompt preview
    with st.expander(" Preview Enriched LLM Prompt"):
        st.markdown("""
        **The LLM now receives rich fundamental data for each stock:**
        - Company Name (e.g., "Apple Inc")
        - Sector & Industry (e.g., "Technology | Consumer Electronics")
        - Business Description (first sentence)
        - Market Capitalization (e.g., "$2.8T")
        - Valuation metrics (P/E, P/S, P/B ratios)
        - Profitability metrics (ROE, Profit Margin)
        - Dividend Yield
        
        This enables the LLM to identify themes like:
        - "Tech Growth vs Energy Value"
        - "Large Cap Quality vs Small Cap Speculative"
        - "High Dividend Utilities vs Growth Tech"
        """)
        
        if len(returns.columns) > 0:
            sample_factor = returns.columns[0]
            if sample_factor in loadings.columns:
                # Show a sample of what data would be sent
                sample_tickers = list(loadings[sample_factor].nlargest(3).index)
                st.markdown(f"**Sample tickers for {sample_factor}:** {', '.join(sample_tickers)}")
                
                # Try to fetch and show sample fundamentals
                with st.spinner("Fetching sample fundamentals..."):
                    sample_fundamentals = fetch_fundamentals_for_tickers(sample_tickers)
                    if not sample_fundamentals.empty:
                        st.markdown("**Sample fundamental data that would be sent to LLM:**")
                        display_cols = ['Name', 'Sector', 'Industry', 'MarketCapitalization', 'PERatio']
                        available_cols = [c for c in display_cols if c in sample_fundamentals.columns]
                        st.dataframe(sample_fundamentals[available_cols])
                        
                        # Generate and show enriched prompt with returns data
                        enriched_prompt = create_enriched_llm_prompt(
                            factor_id=sample_factor,
                            loadings=loadings[sample_factor],
                            fundamentals=sample_fundamentals,
                            factor_returns=returns[sample_factor] if sample_factor in returns.columns else None,
                            all_returns=returns
                        )
                        st.markdown("**Full enriched prompt (includes style attribution + sector exposure):**")
                        st.code(enriched_prompt, language="text")
                    else:
                        st.warning("Could not fetch sample fundamentals (API key may not be configured)")
    
    st.markdown("---")
    
    # Factor DNA Table
    st.markdown("###  Factor DNA Table")
    st.caption("Master table of active latent factors with Alpha/Beta classification")
    
    # Merge LLM names if available
    if 'llm_factor_names' in st.session_state:
        llm_names = st.session_state['llm_factor_names']
        names = {**names, **llm_names}
    
    dna_df = create_factor_dna_table(returns, loadings, names)
    
    if not dna_df.empty:
        # Style the dataframe
        def highlight_beta_risk(val):
            """Highlight high correlation to SPY (Beta in Disguise)."""
            if isinstance(val, float) and val > 0.7:
                return 'background-color: #3d1f1f; color: #ff6666; font-weight: bold'
            return ''
        
        def color_purity(val):
            """Color code purity scores."""
            if isinstance(val, float):
                if val > 0.7:
                    return 'color: #00ff88'
                elif val > 0.4:
                    return 'color: #ffa500'
                else:
                    return 'color: #ff4444'
            return ''
        
        styled_dna = dna_df.style\
            .applymap(highlight_beta_risk, subset=['Correlation to SPY'])\
            .applymap(color_purity, subset=['Alpha Purity'])\
            .format({
                '1M Return': '{:.2%}',
                'Sharpe': '{:.2f}',
                'Alpha Purity': '{:.1%}',
                'Correlation to SPY': '{:.1%}',
                'Crowding Score': '{:.1%}'
            })
        
        st.dataframe(styled_dna, use_container_width=True, height=350)
        
        # Beta warning
        high_beta_factors = dna_df[dna_df['Correlation to SPY'] > 0.7]
        if not high_beta_factors.empty:
            st.markdown(f"""
            <div class="alert-beta">
                 <b>BETA ALERT:</b> {len(high_beta_factors)} factor(s) show high correlation to SPY (>70%). 
                These may be "Beta in Disguise" rather than true alpha factors.
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Factor X-Ray (Drill-Down View)
    st.markdown("###  Factor X-Ray (Drill-Down Analysis)")
    
    # Factor selection for X-Ray
    selected_factor = st.selectbox(
        "Select Factor for Deep Analysis",
        returns.columns,
        format_func=lambda x: f"{x}: {names.get(x, x)[:50]}"
    )
    
    if selected_factor:
        # Get enhanced description for selected factor
        selected_desc = generate_factor_description(
            selected_factor, 
            loadings[selected_factor] if selected_factor in loadings.columns else None
        )
        
        x1, x2, x3 = st.columns([1.5, 1, 1])
        
        with x1:
            st.markdown("** Factor Analysis & Rationale**")
            
            # Create rich description with actual analysis
            top_pos = loadings[selected_factor].nlargest(10) if selected_factor in loadings.columns else pd.Series()
            top_neg = loadings[selected_factor].nsmallest(10) if selected_factor in loadings.columns else pd.Series()
            
            # Calculate metrics
            pos_count = len(top_pos[top_pos > 0])
            neg_count = len(top_neg[top_neg < 0])
            pos_avg = top_pos.head(10).mean() if not top_pos.empty else 0
            neg_avg = top_neg.head(10).mean() if not top_neg.empty else 0
            dispersion = pos_avg - neg_avg
            
            # Determine actionable classification
            if dispersion > 0.03:
                conviction = "High Conviction"
                action_color = "#00ff88"
            elif dispersion > 0.015:
                conviction = "Moderate Conviction"
                action_color = "#ffa500"
            else:
                conviction = "Low Conviction"
                action_color = "#ff6666"
            
            description = f"""
            <div style="background:#1a1a2e; padding:15px; border-radius:8px; border-left:4px solid {action_color};">
            <b style="color:#00d4ff; font-size:16px;">{selected_desc.get('short_name', selected_factor)}</b>
            <span style="background:{action_color}; color:black; padding:2px 8px; border-radius:4px; font-size:11px; margin-left:10px;">
                {conviction}
            </span>
            <br><br>
            
            <b style="color:#8888aa;">THEME:</b> {selected_desc.get('theme', 'Unknown')}<br><br>
            
            <b style="color:#00ff88;"> TOP LONG EXPOSURES:</b><br>
            {' → '.join([f"{ticker} ({loading:.3f})" for ticker, loading in top_pos.head(5).items()])}<br><br>
            
            <b style="color:#ff6666;"> TOP SHORT EXPOSURES:</b><br>
            {' → '.join([f"{ticker} ({loading:.3f})" for ticker, loading in top_neg.head(5).items()])}<br><br>
            
            <div style="background:#0f0f1a; padding:10px; border-radius:4px; margin-top:10px;">
                <b style="color:#00d4ff;"> INVESTMENT RATIONALE:</b><br>
                <i>{selected_desc.get('rationale', 'No rationale available')}</i>
            </div>
            
            <div style="margin-top:10px; font-size:12px; color:#8888aa;">
                <b>METRICS:</b> 
                {pos_count} Longs / {neg_count} Shorts | 
                Avg Dispersion: {dispersion:.3f} | 
                Long Avg: {pos_avg:.3f} | 
                Short Avg: {neg_avg:.3f}
            </div>
            </div>
            """
            
            st.markdown(description, unsafe_allow_html=True)
            
            # Add LLM naming button
            st.markdown("<br>", unsafe_allow_html=True)
            
            llm_col1, llm_col2 = st.columns([2, 1])
            with llm_col1:
                if st.button(" Enhance with LLM Naming", key=f"llm_name_{selected_factor}", 
                           help="Fetch fundamental data and call OpenAI API for enriched naming"):
                    with st.spinner("Fetching fundamental data and calling LLM..."):
                        try:
                            from src.factor_labeler import ask_llm
                            from openai import OpenAI
                            import json
                            import os
                            
                            # Step 1: Fetch fundamentals for all tickers
                            all_tickers = list(top_pos.head(10).index) + list(top_neg.head(10).index)
                            fundamentals = fetch_fundamentals_for_tickers(all_tickers)
                            
                            if not fundamentals.empty:
                                st.success(f" Fetched fundamentals for {len(fundamentals)} tickers")
                                with st.expander("View Fundamental Data"):
                                    st.dataframe(fundamentals[['Name', 'Sector', 'Industry', 'MarketCapitalization', 'PERatio']].head(10))
                            
                            # Step 2: Create enriched prompt with returns data
                            enriched_prompt = create_enriched_llm_prompt(
                                factor_id=selected_factor,
                                loadings=loadings[selected_factor],
                                fundamentals=fundamentals,
                                factor_returns=returns[selected_factor] if selected_factor in returns.columns else None,
                                all_returns=returns
                            )
                            
                            # Show the enriched prompt
                            with st.expander(" View Enriched LLM Prompt (includes style attribution + sector exposure)"):
                                st.code(enriched_prompt, language="text")
                            
                            # Step 3: Call LLM with centralized logic and enrichment data
                            result_obj = ask_llm(
                                factor_id=selected_factor,
                                top_pos=top_pos.head(10).index.tolist(),
                                top_neg=top_neg.head(10).index.tolist(),
                                fundamentals=fundamentals,
                                # Pass returns data for style attribution
                                factor_returns=returns[selected_factor] if selected_factor in returns.columns else None,
                                all_returns=returns,
                                # Pass loadings for enriched stock profiles
                                loadings=loadings[selected_factor] if selected_factor in loadings.columns else None,
                                model=config.OPENAI_MODEL
                            )
                            
                            # Display results
                            st.success(f"**{result_obj.short_name}**")
                            st.info(result_obj.description)
                            st.caption(f"Theme: **{result_obj.theme}** | Confidence: **{result_obj.confidence}**")
                            
                            if result_obj.rationale:
                                st.markdown(f"**Rationale:** {result_obj.rationale}")
                            
                            # Store in session state
                            if 'llm_factor_names' not in st.session_state:
                                st.session_state['llm_factor_names'] = {}
                            st.session_state['llm_factor_names'][selected_factor] = result_obj.short_name
                            
                            # Force a rerun to update the UI immediately
                            st.rerun()
                                
                        except Exception as e:
                            st.error(f"LLM naming failed: {e}")
                            import traceback
                            st.code(traceback.format_exc())
            
            with llm_col2:
                if st.button(" Save Name", key=f"save_name_{selected_factor}"):
                    st.info("Would save to factor_names.json")
        
        with x2:
            st.markdown("** Style Attribution**")
            
            # Calculate style attribution
            styles = calculate_style_attribution(returns[selected_factor], returns)
            
            # Create horizontal bar chart
            fig_style = go.Figure()
            
            colors = ['#00ff88', '#00d4ff', '#ffa500', '#ff6b6b', '#9966ff']
            for i, (style, weight) in enumerate(styles.items()):
                fig_style.add_trace(go.Bar(
                    y=[style],
                    x=[weight],
                    orientation='h',
                    marker_color=colors[i % len(colors)],
                    text=f"{weight:.1%}",
                    textposition='inside',
                    name=style
                ))
            
            fig_style.update_layout(
                height=250,
                showlegend=False,
                xaxis_title="Attribution",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white", family="Courier New"),
                xaxis=dict(tickformat=".0%", range=[0, 1]),
                margin=dict(l=80, r=20, t=20, b=30)
            )
            
            st.plotly_chart(fig_style, use_container_width=True, key="style_attr")
            
            # Attribution insight
            max_style = max(styles, key=styles.get)
            st.caption(f"Primary driver: **{max_style}** ({styles[max_style]:.1%})")
        
        with x3:
            st.markdown("** Sector Tilt**")
            
            # Create sample sector data (would come from actual mapping)
            sectors = {
                'Technology': np.random.uniform(0.1, 0.4),
                'Financials': np.random.uniform(0.05, 0.25),
                'Healthcare': np.random.uniform(0.05, 0.2),
                'Energy': np.random.uniform(0, 0.15),
                'Consumer': np.random.uniform(0.05, 0.2)
            }
            
            # Normalize
            total = sum(sectors.values())
            sectors = {k: v/total for k, v in sectors.items()}
            
            # Treemap
            fig_tree = go.Figure(go.Treemap(
                labels=list(sectors.keys()),
                parents=[""] * len(sectors),
                values=list(sectors.values()),
                textinfo="label+percent parent",
                marker=dict(
                    colors=['#00ff88', '#00d4ff', '#ffa500', '#ff6b6b', '#9966ff'],
                    line=dict(color='#0a0a0a', width=2)
                ),
                textfont=dict(family="Courier New", size=12)
            ))
            
            fig_tree.update_layout(
                height=250,
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=0, b=0)
            )
            
            st.plotly_chart(fig_tree, use_container_width=True, key="sector_tilt")
            
            # Check for concentration
            max_sector = max(sectors, key=sectors.get)
            if sectors[max_sector] > 0.5:
                st.warning(f" Heavy {max_sector} tilt ({sectors[max_sector]:.1%})")
            else:
                st.success(f" Diversified sector exposure")


# =============================================================================
# SECTION 3: PORTFOLIO CONSTRUCTOR (Execution)
# =============================================================================

def calculate_efficient_frontier(
    returns: pd.DataFrame,
    n_points: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate efficient frontier points."""
    if returns is None or returns.empty:
        return np.array([]), np.array([]), np.array([])
    
    mean_rets = returns.mean() * 252
    cov_matrix = estimate_covariance(
        returns, method=CovarianceMethod.LEDOIT_WOLF, annualize=True,
    )
    n_assets = len(mean_rets)
    
    # Generate random portfolios
    n_portfolios = 1000
    weights = np.random.dirichlet(np.ones(n_assets), n_portfolios)
    
    port_rets = weights @ mean_rets
    port_vols = np.sqrt(np.sum(weights @ cov_matrix * weights, axis=1))
    port_sharpes = port_rets / port_vols
    
    # Get efficient frontier (max return for each volatility level)
    vols = np.linspace(port_vols.min(), port_vols.max(), n_points)
    efficient_rets = []
    efficient_weights = []
    
    for vol in vols:
        mask = port_vols <= vol + 0.01
        if mask.any():
            idx = port_rets[mask].argmax()
            efficient_rets.append(port_rets[mask][idx])
            efficient_weights.append(weights[mask][idx])
        else:
            efficient_rets.append(0)
            efficient_weights.append(np.ones(n_assets) / n_assets)
    
    return np.array(efficient_rets), vols, np.array(efficient_weights)


def estimate_liquidity_impact(position_value: float, avg_daily_volume: float) -> Dict[str, any]:
    """Estimate market impact and slippage for a trade."""
    if avg_daily_volume <= 0:
        return {'adv_pct': 0, 'slippage_bps': 0, 'warning': 'No volume data'}
    
    adv_pct = position_value / avg_daily_volume
    
    # Simple slippage model: 10 bps per 1% of ADV
    slippage_bps = min(adv_pct * 1000, 200)  # Cap at 200 bps
    
    warning = None
    if adv_pct > 0.05:  # >5% of ADV
        warning = "HIGH"
    elif adv_pct > 0.02:  # >2% of ADV
        warning = "MEDIUM"
    
    return {
        'adv_pct': adv_pct,
        'slippage_bps': slippage_bps,
        'warning': warning
    }


def render_portfolio_constructor(data: dict, analyzers: dict):
    """Render the Portfolio Constructor section."""
    st.markdown("<h2> PORTFOLIO CONSTRUCTOR: Execution & 'What-If' Analysis</h2>", 
               unsafe_allow_html=True)
    
    returns = data.get('returns')
    loadings = data.get('loadings')
    
    if returns is None or loadings is None:
        st.error(" Data not available for portfolio construction")
        return
    
    # Create tabs for different views
    tab_sandbox, tab_trades, tab_backtest = st.tabs([
        " Optimizer Sandbox",
        " Trade Basket Preview",
        " Backtest Analysis"
    ])
    
    with tab_sandbox:
        st.markdown("### Optimization Sandbox")
        st.caption("Stress-test the optimization before trading")
        
        o1, o2, o3 = st.columns([1, 1, 2])
        
        with o1:
            st.markdown("**Constraints**")
            
            max_turnover = st.slider(
                "Max Turnover / Month",
                min_value=0.05,
                max_value=1.0,
                value=0.20,
                step=0.05,
                format="%.0f%%"
            )
            
            max_beta = st.slider(
                "Beta Constraint",
                min_value=0.0,
                max_value=0.5,
                value=0.10,
                step=0.05
            )
            
            max_position = st.slider(
                "Max Single Position",
                min_value=0.01,
                max_value=0.20,
                value=0.05,
                step=0.01,
                format="%.0f%%"
            )
            
            gross_leverage = st.slider(
                "Gross Leverage",
                min_value=1.0,
                max_value=3.0,
                value=2.0,
                step=0.1
            )
        
        with o2:
            st.markdown("**Objective**")
            
            objective = st.radio(
                "Optimization Target",
                ["Max Sharpe", "Max Sortino", "Risk Parity", "Min Variance"]
            )
            
            methods = st.multiselect(
                "Weighting Methods",
                ['sharpe', 'momentum', 'risk_parity', 'min_variance', 'equal', 'pca'],
                default=['sharpe', 'momentum', 'risk_parity']
            )
            
            run_opt = st.button(" Run Optimizer", type="primary", use_container_width=True)
        
        with o3:
            st.markdown("**Efficient Frontier**")
            
            # Calculate and plot efficient frontier
            frontier_rets, frontier_vols, frontier_weights = calculate_efficient_frontier(returns)
            
            if len(frontier_rets) > 0:
                fig_frontier = go.Figure()
                
                # Efficient frontier line
                fig_frontier.add_trace(go.Scatter(
                    x=frontier_vols * 100,
                    y=frontier_rets * 100,
                    mode='lines+markers',
                    name='Efficient Frontier',
                    line=dict(color='#00d4ff', width=3),
                    marker=dict(size=6, color='#00d4ff')
                ))
                
                # Current portfolio point (equal weight)
                curr_weights = np.ones(len(returns.columns)) / len(returns.columns)
                curr_ret = (returns.mean() * 252 @ curr_weights) * 100
                cov_annual = estimate_covariance(
                    returns, method=CovarianceMethod.LEDOIT_WOLF, annualize=True,
                )
                curr_vol = np.sqrt(curr_weights @ cov_annual @ curr_weights) * 100
                
                fig_frontier.add_trace(go.Scatter(
                    x=[curr_vol],
                    y=[curr_ret],
                    mode='markers',
                    name='Current (Equal Weight)',
                    marker=dict(size=15, color='#ffa500', symbol='star')
                ))
                
                # Turnover cost lines
                for cost in [0.001, 0.005, 0.01]:  # 10bp, 50bp, 100bp
                    cost_impact = cost * max_turnover * 12 * 100  # Annual cost
                    fig_frontier.add_trace(go.Scatter(
                        x=frontier_vols * 100,
                        y=(frontier_rets - cost_impact) * 100,
                        mode='lines',
                        name=f'After {cost*100:.0f}bp Turnover',
                        line=dict(dash='dash', width=1),
                        opacity=0.5
                    ))
                
                fig_frontier.update_layout(
                    height=350,
                    xaxis_title="Volatility (%)",
                    yaxis_title="Expected Return (%)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white", family="Courier New"),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.4),
                    margin=dict(l=50, r=20, t=30, b=80)
                )
                
                st.plotly_chart(fig_frontier, use_container_width=True, key="efficient_frontier")
            
            if run_opt:
                with st.spinner("Running optimization..."):
                    try:
                        optimizer = SharpeOptimizer(returns, loadings)
                        result = optimizer.optimize_blend(
                            lookback=126,
                            methods=methods,
                            technique='differential'
                        )
                        
                        st.session_state['opt_result'] = result
                        st.success(f" Optimization complete! Sharpe: {result.sharpe_ratio:.2f}")
                    except Exception as e:
                        st.error(f"Optimization failed: {e}")
    
    with tab_trades:
        st.markdown("### Trade Basket Preview")
        st.caption("Review trades before execution with liquidity warnings")
        
        # Generate sample trade basket based on optimization or current loadings
        if 'opt_result' in st.session_state:
            factor_weights = st.session_state['opt_result'].optimal_weights
        else:
            # Use equal weights
            factor_weights = {f: 1/len(returns.columns) for f in returns.columns}
        
        # Calculate composite scores
        composite_score = pd.Series(0.0, index=loadings.index)
        for factor, weight in factor_weights.items():
            if factor in loadings.columns:
                composite_score += loadings[factor] * weight
        
        composite_score = composite_score.sort_values(ascending=False)
        
        # Select longs and shorts
        n_long = min(20, len(composite_score) // 10)
        n_short = n_long
        
        longs = composite_score.head(n_long)
        shorts = composite_score.tail(n_short)
        
        # Capital assumption
        capital = st.number_input(
            "Portfolio Capital ($)",
            min_value=1_000_000,
            max_value=10_000_000_000,
            value=1_000_000_000,
            step=100_000_000,
            format="%d"
        )
        
        # Calculate positions
        long_weights = longs / longs.sum()
        short_weights = shorts / shorts.sum() * -1
        
        long_positions = pd.DataFrame({
            'Ticker': longs.index,
            'Action': 'BUY',
            'Score': longs.values,
            'Weight': long_weights.values,
            'Value ($)': long_weights.values * capital * gross_leverage / 2
        })
        
        short_positions = pd.DataFrame({
            'Ticker': shorts.index,
            'Action': 'SELL',
            'Score': shorts.values,
            'Weight': np.abs(short_weights.values),
            'Value ($)': np.abs(short_weights.values) * capital * gross_leverage / 2
        })
        
        # Estimate liquidity
        # In production, would fetch real ADV data
        np.random.seed(42)  # For reproducible demo
        
        def estimate_adv(ticker: str) -> float:
            """Estimate ADV (placeholder - use real data in production)."""
            # Generate realistic ADV based on ticker (larger cap = higher ADV)
            base = np.random.lognormal(8, 1.5) * 1e6  # $1M to $100M typical
            return base
        
        long_positions['Est ADV ($)'] = long_positions['Ticker'].apply(estimate_adv)
        long_positions['% ADV'] = long_positions['Value ($)'] / long_positions['Est ADV ($)']
        
        short_positions['Est ADV ($)'] = short_positions['Ticker'].apply(estimate_adv)
        short_positions['% ADV'] = short_positions['Value ($)'] / short_positions['Est ADV ($)']
        
        # Display trades
        t1, t2 = st.columns(2)
        
        def highlight_liquidity(val):
            if isinstance(val, float):
                if val > 0.05:
                    return 'background-color: #3d1f1f; color: #ff6666; font-weight: bold'
                elif val > 0.02:
                    return 'color: #ffa500'
            return ''
        
        with t1:
            st.markdown("** Top Long Positions**")
            
            # Take head before styling
            long_display = long_positions.head(10).copy()
            styled_long = long_display.style\
                .applymap(highlight_liquidity, subset=['% ADV'])\
                .format({
                    'Score': '{:.4f}',
                    'Weight': '{:.2%}',
                    'Value ($)': '${:,.0f}',
                    'Est ADV ($)': '${:,.0f}',
                    '% ADV': '{:.1%}'
                })
            
            st.dataframe(styled_long, use_container_width=True, height=350)
        
        with t2:
            st.markdown("** Top Short Positions**")
            
            # Take head before styling
            short_display = short_positions.head(10).copy()
            styled_short = short_display.style\
                .applymap(highlight_liquidity, subset=['% ADV'])\
                .format({
                    'Score': '{:.4f}',
                    'Weight': '{:.2%}',
                    'Value ($)': '${:,.0f}',
                    'Est ADV ($)': '${:,.0f}',
                    '% ADV': '{:.1%}'
                })
            
            st.dataframe(styled_short, use_container_width=True, height=350)
        
        # Liquidity warnings
        high_impact_trades = pd.concat([
            long_positions[long_positions['% ADV'] > 0.02],
            short_positions[short_positions['% ADV'] > 0.02]
        ])
        
        if not high_impact_trades.empty:
            st.markdown("###  Liquidity Alerts")
            
            for _, trade in high_impact_trades.head(5).iterrows():
                impact = estimate_liquidity_impact(trade['Value ($)'], trade['Est ADV ($)'])
                
                st.markdown(f"""
                <div class="alert-beta">
                    <b>{trade['Ticker']}</b> ({trade['Action']})<br>
                    Position: ${trade['Value ($)']:,.0f} | 
                    ADV: {trade['% ADV']:.1%} | 
                    Est. Slippage: {impact['slippage_bps']:.0f} bps<br>
                    <i>Warning: Trade exceeds 2% of daily volume. Consider splitting into multiple tranches.</i>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success(" All trades within liquidity thresholds (<2% ADV)")
        
        # Export option
        all_trades = pd.concat([long_positions, short_positions])
        csv = all_trades.to_csv(index=False)
        
        col_dl1, col_dl2 = st.columns([1, 4])
        with col_dl1:
            st.download_button(
                label=" Download Trade File",
                data=csv,
                file_name=f"trade_basket_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with tab_backtest:
        st.markdown("### Backtest Analysis")
        st.info("Run walk-forward optimization to see historical performance")
        
        if st.button(" Run Walk-Forward Backtest", type="primary"):
            with st.spinner("Running walk-forward analysis (this may take a few minutes)..."):
                try:
                    optimizer = SharpeOptimizer(returns, loadings)
                    
                    wf_results = optimizer.walk_forward_optimize(
                        train_window=126,
                        test_window=21,
                        methods=['sharpe', 'momentum', 'risk_parity'],
                        technique='differential',
                        verbose=False
                    )
                    
                    st.session_state['wf_results'] = wf_results
                    
                    # Display summary
                    avg_train = wf_results['train_sharpe'].mean()
                    avg_test = wf_results['test_sharpe'].mean()
                    
                    b1, b2, b3, b4 = st.columns(4)
                    with b1:
                        st.metric("Avg Train Sharpe", f"{avg_train:.2f}")
                    with b2:
                        st.metric("Avg Test Sharpe", f"{avg_test:.2f}")
                    with b3:
                        st.metric("Periods", len(wf_results))
                    with b4:
                        consistency = (wf_results['test_sharpe'] > 0).mean()
                        st.metric("Win Rate", f"{consistency:.1%}")
                    
                    # Plot rolling performance
                    fig_wf = go.Figure()
                    
                    fig_wf.add_trace(go.Scatter(
                        x=wf_results['date'],
                        y=wf_results['train_sharpe'],
                        name='Train Sharpe',
                        line=dict(color='#00ff88')
                    ))
                    
                    fig_wf.add_trace(go.Scatter(
                        x=wf_results['date'],
                        y=wf_results['test_sharpe'],
                        name='Test Sharpe',
                        line=dict(color='#00d4ff')
                    ))
                    
                    fig_wf.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
                    
                    fig_wf.update_layout(
                        height=400,
                        title="Walk-Forward Sharpe Ratios Over Time",
                        xaxis_title="Date",
                        yaxis_title="Sharpe Ratio",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white", family="Courier New")
                    )
                    
                    st.plotly_chart(fig_wf, use_container_width=True, key="wf_chart")
                    
                    # Overfitting warning
                    if avg_train > 1.0 and avg_test < 0.5:
                        st.error("""
                         **OVERFITTING DETECTED**: High train Sharpe but low test Sharpe suggests 
                        the model is fitting to noise. Consider:
                        - Reducing number of factors
                        - Increasing regularization
                        - Using simpler weighting methods
                        """)
                
                except Exception as e:
                    st.error(f"Backtest failed: {e}")


# =============================================================================
# SECTION 4: PHASE 2 - REGIME CONDITIONAL OPTIMIZATION (RS-MVO)
# =============================================================================

def render_regime_conditional_optimization(data: dict, analyzers: dict):
    """Render the Phase 2 Regime-Switching Mean-Variance Optimization panel."""
    st.markdown("<h2> REGIME CONDITIONAL OPTIMIZATION (RS-MVO)</h2>", 
               unsafe_allow_html=True)
    st.caption("Data-driven factor weights conditional on market regime | Phase 2 Enhancement")
    
    returns = data.get('returns')
    regime_detector = analyzers.get('regime')
    
    if returns is None:
        st.error(" Factor returns data not available")
        return
    
    if regime_detector is None or regime_detector.hmm_model is None:
        st.error(" Regime detector not initialized. Fit HMM first.")
        return
    
    # Current regime display
    current_regime = regime_detector.detect_current_regime()
    regime_probs = regime_detector.get_regime_probabilities()
    
    st.markdown("###  Current Market Regime")
    
    col1, col2, col3 = st.columns([1, 1, 1.5])
    
    with col1:
        # Regime gauge
        fig_gauge = create_regime_gauge_chart(regime_detector)
        st.plotly_chart(fig_gauge, use_container_width=True, key="rsmvo_regime_gauge")
    
    with col2:
        st.markdown("**Regime Characteristics**")
        st.metric("Volatility", f"{current_regime.volatility:.2%}")
        st.metric("Trend", f"{current_regime.trend:.4f}")
        st.write(f"**Description:** {current_regime.description}")
    
    with col3:
        # Regime probability bars
        prob_df = pd.DataFrame({
            'Regime': [r.value.replace('_', ' ').title() for r in regime_probs.keys()],
            'Probability': list(regime_probs.values())
        }).sort_values('Probability', ascending=True)
        
        fig_probs = go.Figure()
        colors = ['#00ff88' if p > 0.5 else '#ffa500' if p > 0.2 else '#ff6666' 
                  for p in prob_df['Probability']]
        
        fig_probs.add_trace(go.Bar(
            x=prob_df['Probability'],
            y=prob_df['Regime'],
            orientation='h',
            marker_color=colors,
            text=[f"{p:.1%}" for p in prob_df['Probability']],
            textposition='inside'
        ))
        
        fig_probs.update_layout(
            height=250,
            xaxis_title="Probability",
            title="Regime Probability Distribution",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Courier New")
        )
        st.plotly_chart(fig_probs, use_container_width=True, key="rsmvo_regime_probs")
    
    st.markdown("---")
    
    # RS-MVO Configuration
    st.markdown("###  RS-MVO Configuration")
    
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        lookback_window = st.slider(
            "Lookback Window (days)",
            min_value=252,
            max_value=5040,
            value=2520,
            step=252,
            help="Historical period to consider for regime filtering (~10 years = 2520 days)"
        )
    
    with config_col2:
        min_observations = st.slider(
            "Min Regime Observations",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Minimum number of regime-specific observations required"
        )
    
    with config_col3:
        target_regime = st.selectbox(
            "Target Regime",
            options=list(MarketRegime),
            format_func=lambda r: r.value.replace('_', ' ').title(),
            index=list(MarketRegime).index(current_regime.regime)
        )
        fallback_to_global = st.checkbox("Fallback to Global", value=True,
                                         help="Use global optimization if insufficient regime data")
    
    st.markdown("---")
    
    # Run Conditional Optimization
    if st.button(" Run RS-MVO Optimization", type="primary"):
        with st.spinner(f"Running Regime-Switching MVO for {target_regime.value}..."):
            try:
                # Get conditional weights
                weights = regime_detector.get_conditional_optimal_weights(
                    regime=target_regime,
                    lookback_window=lookback_window,
                    min_observations=min_observations,
                    fallback_to_global=fallback_to_global
                )
                
                # Get heuristic weights for comparison
                heuristic_weights = regime_detector._get_heuristic_factor_weights(target_regime)
                
                # Store in session state
                st.session_state['rsmvo_weights'] = weights
                st.session_state['rsmvo_heuristic'] = heuristic_weights
                st.session_state['rsmvo_regime'] = target_regime
                
                st.success(f" RS-MVO complete for {target_regime.value}!")
                
            except Exception as e:
                st.error(f"RS-MVO failed: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Display results if available
    if 'rsmvo_weights' in st.session_state:
        weights = st.session_state['rsmvo_weights']
        heuristic = st.session_state['rsmvo_heuristic']
        target = st.session_state['rsmvo_regime']
        
        st.markdown("###  Optimization Results")
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.markdown("** RS-MVO Weights (Data-Driven)**")
            
            # Create comparison dataframe
            comp_df = pd.DataFrame({
                'Factor': list(weights.keys()),
                'RS-MVO Weight': list(weights.values()),
                'Heuristic Weight': [heuristic.get(f, 0) for f in weights.keys()]
            })
            comp_df['Difference'] = comp_df['RS-MVO Weight'] - comp_df['Heuristic Weight']
            
            # Sort by weight
            comp_df = comp_df.sort_values('RS-MVO Weight', ascending=True)
            
            # Bar chart
            fig_weights = go.Figure()
            
            fig_weights.add_trace(go.Bar(
                y=comp_df['Factor'],
                x=comp_df['RS-MVO Weight'],
                orientation='h',
                name='RS-MVO (Data-Driven)',
                marker_color='#00d4ff',
                text=[f"{w:.1%}" for w in comp_df['RS-MVO Weight']],
                textposition='inside'
            ))
            
            fig_weights.add_trace(go.Bar(
                y=comp_df['Factor'],
                x=comp_df['Heuristic Weight'],
                orientation='h',
                name='Heuristic (Rules-Based)',
                marker_color='#ff6666',
                opacity=0.7,
                text=[f"{w:.1%}" for w in comp_df['Heuristic Weight']],
                textposition='inside'
            ))
            
            fig_weights.update_layout(
                height=350,
                barmode='group',
                title="Factor Weights Comparison",
                xaxis_title="Weight",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white", family="Courier New"),
                legend=dict(orientation="h", yanchor="bottom", y=-0.3)
            )
            
            st.plotly_chart(fig_weights, use_container_width=True, key="rsmvo_weights_comp")
        
        with res_col2:
            st.markdown("** Weight Differences (RS-MVO vs Heuristic)**")
            
            # Difference chart
            diff_df = comp_df.sort_values('Difference', ascending=True)
            colors = ['#00ff88' if d > 0 else '#ff6666' for d in diff_df['Difference']]
            
            fig_diff = go.Figure()
            fig_diff.add_trace(go.Bar(
                y=diff_df['Factor'],
                x=diff_df['Difference'],
                orientation='h',
                marker_color=colors,
                text=[f"{d:+.1%}" for d in diff_df['Difference']],
                textposition='outside'
            ))
            
            fig_diff.add_vline(x=0, line_dash="dash", line_color="white")
            
            fig_diff.update_layout(
                height=350,
                title="Weight Differences (Positive = More Weight in RS-MVO)",
                xaxis_title="Weight Difference",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white", family="Courier New")
            )
            
            st.plotly_chart(fig_diff, use_container_width=True, key="rsmvo_diff")
        
        # Key insights
        st.markdown("###  Key Insights")
        
        # Find largest deviations
        top_increases = comp_df.nlargest(3, 'Difference')
        top_decreases = comp_df.nsmallest(3, 'Difference')
        
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            st.markdown("** Factors RS-MVO Overweights vs Heuristic:**")
            for _, row in top_increases.iterrows():
                if row['Difference'] > 0.01:
                    st.markdown(f"- **{row['Factor']}**: +{row['Difference']:.1%} "
                              f"({row['Heuristic Weight']:.1%} → {row['RS-MVO Weight']:.1%})")
        
        with insight_col2:
            st.markdown("** Factors RS-MVO Underweights vs Heuristic:**")
            for _, row in top_decreases.iterrows():
                if row['Difference'] < -0.01:
                    st.markdown(f"- **{row['Factor']}**: {row['Difference']:.1%} "
                              f"({row['Heuristic Weight']:.1%} → {row['RS-MVO Weight']:.1%})")
        
        # Export option
        csv = comp_df.to_csv(index=False)
        st.download_button(
            label=" Download RS-MVO Weights",
            data=csv,
            file_name=f"rsmvo_weights_{target.value}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )


# =============================================================================
# SECTION 5: PHASE 2 - META-MODEL SIGNAL AGGREGATION
# =============================================================================

def render_meta_model_aggregation(data: dict, analyzers: dict):
    """Render the Phase 2 XGBoost Meta-Model Signal Aggregation panel."""
    st.markdown("<h2> META-MODEL SIGNAL AGGREGATION</h2>", 
               unsafe_allow_html=True)
    st.caption("Gradient Boosting consensus with non-linear signal interactions | Phase 2 Enhancement")
    
    returns = data.get('returns')
    loadings = data.get('loadings')
    regime_detector = analyzers.get('regime')
    momentum_analyzer = analyzers.get('momentum')
    
    if returns is None:
        st.error(" Factor returns data not available")
        return
    
    # Check XGBoost availability
    if not XGBOOST_AVAILABLE:
        st.warning(" XGBoost not installed. Meta-model will use voting fallback.")
        st.info("Install with: `pip install xgboost` for full functionality")
    
    st.markdown("###  Meta-Model Configuration")
    
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        min_training_samples = st.slider(
            "Min Training Samples",
            min_value=63,
            max_value=504,
            value=252,
            step=63,
            help="Minimum historical samples required for model training (~1 year = 252)"
        )
    
    with config_col2:
        prediction_horizon = st.slider(
            "Prediction Horizon (days)",
            min_value=1,
            max_value=21,
            value=5,
            step=1,
            help="Forward return horizon for label generation"
        )
    
    with config_col3:
        model_type = st.selectbox(
            "Model Type",
            options=["XGBoost (Classifier)", "Voting Fallback"],
            index=0 if XGBOOST_AVAILABLE else 1,
            disabled=not XGBOOST_AVAILABLE
        )
        use_voting_fallback = st.checkbox("Allow Voting Fallback", value=True,
                                          help="Use voting if model untrained")
    
    # Advanced options
    with st.expander(" Advanced Model Parameters"):
        adv_col1, adv_col2, adv_col3 = st.columns(3)
        
        with adv_col1:
            n_estimators = st.slider("N Estimators", 50, 500, 100, 50)
            max_depth = st.slider("Max Depth", 2, 10, 3, 1)
        
        with adv_col2:
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.05, 0.01)
            subsample = st.slider("Subsample", 0.5, 1.0, 0.8, 0.1)
        
        with adv_col3:
            purge_gap = st.slider("Purge Gap (days)", 0, 10, 5, 1,
                                  help="Gap between train/test to avoid label overlap")
    
    st.markdown("---")
    
    # Feature extraction preview
    st.markdown("###  Feature Extraction Preview")
    st.caption("Features extracted from signal sources for meta-model input")
    
    # Create a sample feature extraction
    try:
        extractor = FeatureExtractor()
        
        # Sample momentum features
        sample_momentum = {
            'momentum': {'rsi': 65, 'macd_signal': 'buy', 'adx': 25, 'combined_signal': 'buy'},
            'value': {'rsi': 45, 'macd_signal': 'neutral', 'adx': 15, 'combined_signal': 'neutral'}
        }
        mom_features = extractor.extract_momentum_features(sample_momentum)
        
        # Sample regime features
        if regime_detector and regime_detector.hmm_model:
            current = regime_detector.detect_current_regime()
            regime_probs = regime_detector.get_regime_probabilities()
            regime_features = extractor.extract_regime_features(current, regime_probs)
        else:
            regime_features = {'regime_prob': 0.5, 'regime_vol': 0.2}
        
        feat_col1, feat_col2 = st.columns(2)
        
        with feat_col1:
            st.markdown("**Momentum Features (Sample)**")
            mom_df = pd.DataFrame(list(mom_features.items()), columns=['Feature', 'Value'])
            st.dataframe(mom_df.head(10), use_container_width=True, height=200)
        
        with feat_col2:
            st.markdown("**Regime Features (Sample)**")
            reg_df = pd.DataFrame(list(regime_features.items()), columns=['Feature', 'Value'])
            st.dataframe(reg_df, use_container_width=True, height=200)
        
    except Exception as e:
        st.warning(f"Could not generate feature preview: {e}")
    
    st.markdown("---")
    
    # Model Training
    st.markdown("###  Model Training")
    
    train_col1, train_col2 = st.columns([1, 2])
    
    with train_col1:
        # Need market returns for training
        st.markdown("**Market Returns (for Labels)**")
        use_proxy_market = st.checkbox("Use Factor Mean as Market Proxy", value=True)
        
        if use_proxy_market:
            market_proxy = returns.mean(axis=1)
            st.success(f" Using factor mean as market proxy ({len(market_proxy)} days)")
        else:
            uploaded_market = st.file_uploader("Upload Market Returns CSV", type=['csv'])
            if uploaded_market:
                market_proxy = pd.read_csv(uploaded_market, index_col=0, parse_dates=True).squeeze()
            else:
                market_proxy = None
                st.warning(" No market returns provided")
    
    with train_col2:
        if st.button(" Train Meta-Model (Walk-Forward)", type="primary", use_container_width=True):
            if market_proxy is None:
                st.error(" Market returns required for training")
            else:
                with st.spinner("Training XGBoost meta-model with walk-forward validation..."):
                    try:
                        # Initialize meta-model aggregator
                        class MockFRS:
                            pass
                        
                        model_params = {
                            'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'learning_rate': learning_rate,
                            'subsample': subsample
                        }
                        
                        aggregator = MetaModelAggregator(
                            MockFRS(),
                            model_params=model_params,
                            min_training_samples=min_training_samples,
                            prediction_horizon=prediction_horizon,
                            use_voting_fallback=use_voting_fallback
                        )
                        
                        aggregator.set_market_returns(market_proxy)
                        
                        # Add signal sources if available
                        if momentum_analyzer:
                            aggregator.add_momentum_signals(momentum_analyzer)
                        if regime_detector:
                            aggregator.add_regime_signals(regime_detector)
                        
                        # Train
                        aggregator.train_walk_forward(
                            min_window=min_training_samples,
                            step_size=21,
                            purge_gap=purge_gap,
                            verbose=True
                        )
                        
                        # Store in session state
                        st.session_state['meta_model'] = aggregator
                        st.session_state['model_trained'] = True
                        
                        st.success(" Meta-model training complete!")
                        
                        # Show feature importance if available
                        importance_df = aggregator.get_model_feature_importance()
                        if importance_df is not None:
                            st.session_state['feature_importance'] = importance_df
                        
                    except Exception as e:
                        st.error(f"Training failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())
    
    # Display training results
    if 'model_trained' in st.session_state and st.session_state['model_trained']:
        st.markdown("---")
        st.markdown("###  Model Performance & Predictions")
        
        aggregator = st.session_state['meta_model']
        
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        
        with perf_col1:
            st.metric("Model Status", " Trained" if aggregator._model_trained else "Fallback")
        
        with perf_col2:
            st.metric("Training Samples", len(aggregator._training_X) if hasattr(aggregator, '_training_X') else "N/A")
        
        with perf_col3:
            st.metric("Features", aggregator._training_X.shape[1] if hasattr(aggregator, '_training_X') else "N/A")
        
        # Feature importance
        if 'feature_importance' in st.session_state:
            st.markdown("** Feature Importance (Top 10)**")
            importance = st.session_state['feature_importance'].head(10)
            
            fig_imp = go.Figure()
            fig_imp.add_trace(go.Bar(
                x=importance['importance'],
                y=[f"Feature {i}" for i in importance['feature_idx']],
                orientation='h',
                marker_color='#00d4ff'
            ))
            
            fig_imp.update_layout(
                height=300,
                title="XGBoost Feature Importance",
                xaxis_title="Importance",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white", family="Courier New")
            )
            
            st.plotly_chart(fig_imp, use_container_width=True, key="feature_importance")
        
        # Generate current prediction
        st.markdown("###  Current Consensus Prediction")
        
        if st.button("Generate Meta-Consensus Signal"):
            try:
                result = aggregator.generate_meta_consensus()
                
                prob_up = result['probability_up']
                meta_score = result['meta_score']
                consensus_signal = result['consensus_signal']
                
                pred_col1, pred_col2, pred_col3 = st.columns(3)
                
                with pred_col1:
                    # Probability gauge
                    fig_prob = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob_up * 100,
                        number={'suffix': "%", 'font': {'size': 24}},
                        title={'text': "P(Positive Return)", 'font': {'size': 14}},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#00ff88" if prob_up > 0.5 else "#ff6666"},
                            'steps': [
                                {'range': [0, 50], 'color': "#3d1f1f"},
                                {'range': [50, 100], 'color': "#1f3d1f"}
                            ],
                            'threshold': {
                                'line': {'color': "white", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    
                    fig_prob.update_layout(
                        height=250,
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white")
                    )
                    
                    st.plotly_chart(fig_prob, use_container_width=True, key="prob_gauge")
                
                with pred_col2:
                    st.metric("Meta Score", f"{meta_score:.1f}", 
                             delta="Bullish" if meta_score > 0 else "Bearish")
                    st.metric("Signal Strength", f"{abs(meta_score):.1f}/100")
                    st.write(f"**Direction:** {consensus_signal.consensus_direction.value}")
                    st.write(f"**Confidence:** {consensus_signal.confidence:.1f}%")
                
                with pred_col3:
                    st.markdown("** Recommendation**")
                    st.info(consensus_signal.recommendation)
                    st.markdown(f"**Risk Level:** {consensus_signal.risk_level}")
                
                # Signal interpretation
                if prob_up > 0.7:
                    st.success("🟢 **STRONG BULLISH SIGNAL**: Model predicts high probability of positive returns. "
                              "Consider increasing long exposure.")
                elif prob_up > 0.55:
                    st.info("🟡 **MODERATE BULLISH**: Slight positive bias detected.")
                elif prob_up < 0.3:
                    st.error(" **STRONG BEARISH SIGNAL**: Model predicts high probability of negative returns. "
                            "Consider defensive positioning or hedging.")
                elif prob_up < 0.45:
                    st.warning("🟠 **MODERATE BEARISH**: Slight negative bias detected.")
                else:
                    st.write(" **NEUTRAL**: No clear directional signal. Maintain current positioning.")
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")


# =============================================================================
# SECTION 6: RISK & DRAWDOWN
# =============================================================================

def render_risk_drawdown(data: dict, analyzers: dict):
    """Render the Risk & Drawdown section."""
    st.markdown("<h2> RISK & DRAWDOWN: Monitoring & Attribution</h2>", 
               unsafe_allow_html=True)
    
    returns = data.get('returns')
    
    if returns is None:
        st.error(" Returns data not available")
        return
    
    # Risk metrics
    st.markdown("###  Factor Risk Metrics")
    
    lookback = st.selectbox(
        "Risk Calculation Window",
        [21, 63, 126, 252],
        format_func=lambda x: f"{x} days (~{x//21} months)"
    )
    
    recent = returns.tail(lookback)
    
    # Calculate risk metrics for each factor
    risk_data = []
    for factor in returns.columns:
        factor_rets = recent[factor]
        
        # Basic stats
        vol = factor_rets.std() * np.sqrt(252)
        
        # Drawdown
        cum_rets = (1 + factor_rets).cumprod()
        running_max = cum_rets.expanding().max()
        drawdown = (cum_rets - running_max) / running_max
        max_dd = drawdown.min()
        
        # VaR (95%)
        var_95 = np.percentile(factor_rets, 5)
        
        # CVaR (Expected Shortfall)
        cvar_95 = factor_rets[factor_rets <= var_95].mean()
        
        # Skewness and Kurtosis
        skew = factor_rets.skew()
        kurt = factor_rets.kurtosis()
        
        risk_data.append({
            'Factor': factor,
            'Volatility': vol,
            'Max Drawdown': max_dd,
            'VaR (95%)': var_95,
            'CVaR (95%)': cvar_95,
            'Skewness': skew,
            'Kurtosis': kurt
        })
    
    risk_df = pd.DataFrame(risk_data)
    
    # Display risk table
    def color_risk(val):
        if isinstance(val, float):
            if 'Drawdown' in str(val) or 'VaR' in str(val) or 'CVaR' in str(val):
                if val < -0.1:
                    return 'background-color: #3d1f1f; color: #ff6666'
                elif val < -0.05:
                    return 'color: #ffa500'
            elif 'Volatility' in str(val):
                if val > 0.5:
                    return 'color: #ff6666'
                elif val > 0.3:
                    return 'color: #ffa500'
        return ''
    
    styled_risk = risk_df.style\
        .format({
            'Volatility': '{:.1%}',
            'Max Drawdown': '{:.1%}',
            'VaR (95%)': '{:.2%}',
            'CVaR (95%)': '{:.2%}',
            'Skewness': '{:.2f}',
            'Kurtosis': '{:.2f}'
        })\
        .background_gradient(subset=['Max Drawdown'], cmap='Reds_r')\
        .background_gradient(subset=['Volatility'], cmap='Reds')
    
    st.dataframe(styled_risk, use_container_width=True, height=350)
    
    # Drawdown chart
    st.markdown("###  Factor Drawdown Chart")
    
    selected_dd_factors = st.multiselect(
        "Select Factors to Plot",
        returns.columns,
        default=returns.columns[:min(3, len(returns.columns))].tolist()
    )
    
    if selected_dd_factors:
        fig_dd = go.Figure()
        
        colors = ['#00ff88', '#00d4ff', '#ffa500', '#ff6b6b', '#9966ff']
        
        for i, factor in enumerate(selected_dd_factors):
            factor_rets = returns[factor]
            cum_rets = (1 + factor_rets).cumprod()
            running_max = cum_rets.expanding().max()
            drawdown = (cum_rets - running_max) / running_max * 100
            
            fig_dd.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown,
                name=factor,
                line=dict(color=colors[i % len(colors)], width=2),
                fill='tonexty' if i == 0 else None
            ))
        
        fig_dd.update_layout(
            height=400,
            title="Factor Drawdowns Over Time",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Courier New"),
            yaxis=dict(tickformat=".0%"),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_dd, use_container_width=True, key="drawdown_chart")


# =============================================================================
# SIDEBAR: GLOBAL CONTROLS
# =============================================================================

def _validate_pm_vitals_inputs(data: dict, min_rows: int = 63) -> Dict[str, any]:
    """Validate explicit portfolio/benchmark return inputs for PM vitals."""
    portfolio = data.get("portfolio_returns")
    benchmark = data.get("benchmark_returns")

    result: Dict[str, any] = {
        "portfolio_ok": False,
        "benchmark_ok": False,
        "overlap_ok": False,
        "overall_ok": False,
        "messages": [],
    }

    if isinstance(portfolio, pd.Series) and not portfolio.empty:
        result["portfolio_ok"] = True
    else:
        result["messages"].append("Missing portfolio_returns.csv")

    if isinstance(benchmark, pd.Series) and not benchmark.empty:
        result["benchmark_ok"] = True
    else:
        result["messages"].append("Missing benchmark_returns.csv")

    if result["portfolio_ok"] and result["benchmark_ok"]:
        overlap = pd.concat(
            [portfolio.rename("portfolio"), benchmark.rename("benchmark")],
            axis=1,
            join="inner",
        ).dropna()
        result["overlap_rows"] = len(overlap)
        result["overlap_ok"] = len(overlap) >= min_rows
        result["overall_ok"] = result["overlap_ok"]
        if not result["overlap_ok"]:
            result["messages"].append(
                f"Portfolio/benchmark overlap too short ({len(overlap)} rows)"
            )
        else:
            result["overlap_start"] = overlap.index.min()
            result["overlap_end"] = overlap.index.max()
    else:
        result["overlap_rows"] = 0

    return result


def render_sidebar(data: dict, analyzers: dict):
    """Render the sidebar with global controls."""
    st.sidebar.markdown("""
    <h1 style='color:#00d4ff; font-size:18px; text-align:center; margin-bottom:20px;'>
     ALPHA COMMAND
    </h1>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Universe Control
    st.sidebar.markdown("###  Universe Control")
    
    universe_date = st.sidebar.date_input(
        "Analysis Date",
        value=datetime.now(),
        max_value=datetime.now()
    )
    
    st.sidebar.info(f"PIT Universe: {universe_date.strftime('%Y-%m-%d')}")
    
    # Data status
    st.sidebar.markdown("###  Data Status")
    
    if data.get('returns') is not None:
        returns = data['returns']
        st.sidebar.success(f" Factor Returns: {len(returns.columns)} factors")
        st.sidebar.caption(f"  Range: {returns.index.min().date()} to {returns.index.max().date()}")
    else:
        st.sidebar.error(" Factor Returns: Not found")
    
    if data.get('loadings') is not None:
        loadings = data['loadings']
        st.sidebar.success(f" Factor Loadings: {len(loadings)} stocks")
    else:
        st.sidebar.error(" Factor Loadings: Not found")

    # Startup check: explicit PM vitals inputs
    st.sidebar.markdown("###  PM Vitals Inputs")
    pm_inputs = _validate_pm_vitals_inputs(data)
    if pm_inputs["overall_ok"]:
        st.sidebar.success(
            f" Portfolio/Benchmark: Ready ({pm_inputs['overlap_rows']} rows)"
        )
        st.sidebar.caption(
            f" Overlap: {pm_inputs['overlap_start'].date()} to {pm_inputs['overlap_end'].date()}"
        )
    else:
        st.sidebar.warning(" Portfolio/Benchmark: Incomplete")
        for msg in pm_inputs["messages"]:
            st.sidebar.caption(f"• {msg}")
    
    st.sidebar.markdown("---")
    
    # Quick Actions
    st.sidebar.markdown("###  Quick Actions")
    
    if st.sidebar.button(" Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    if st.sidebar.button(" Export Report", use_container_width=True):
        st.sidebar.info("Report generation would happen here")
    
    st.sidebar.markdown("---")
    
    # System Status
    st.sidebar.markdown("###  System Status")
    
    with st.sidebar.expander("Database Health"):
        try:
            health = check_database_health()
            st.write(f"**Status:** {' Healthy' if health['is_healthy'] else ' Issues'}")
            st.write(f"**Size:** {health['size_mb']:.1f} MB")
            st.write(f"**Locked:** {' Yes' if health['is_locked'] else ' No'}")
        except Exception as e:
            st.write(f" Check failed: {e}")
    
    with st.sidebar.expander("API Status"):
        st.write("Alpha Vantage: Connected")
        st.write("OpenAI (LLM): Available")
    
    # Phase 2 Status
    st.sidebar.markdown("###  Phase 2 Features")
    
    with st.sidebar.expander("System Capabilities"):
        st.write(f" RS-MVO: {'Available' if analyzers.get('regime') else 'Needs Regime Detector'}")
        st.write(f" Meta-Model: {'XGBoost Ready' if XGBOOST_AVAILABLE else 'Voting Fallback'}")
        st.write(" Conditional Optimization: Active")
        st.write(" Walk-Forward Training: Active")


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    
    # Load data
    data = load_system_data()
    
    # Initialize analyzers
    analyzers = initialize_analyzers(data)
    
    # Render sidebar
    render_sidebar(data, analyzers)
    
    # Check if data is available
    if data['returns'] is None:
        st.error("""
        ##  Data Not Available
        
        Factor returns data not found. Please run:
        ```bash
        python -m src discover --universe VTHR
        ```
        
        Or ensure `factor_returns.csv` exists in the working directory.
        """)
        return

    # Render main header. Block PM vitals if explicit inputs are not valid.
    pm_inputs = _validate_pm_vitals_inputs(data)
    if pm_inputs["overall_ok"]:
        render_morning_coffee_header(data, analyzers)
    else:
        st.markdown("<h1> ALPHACMD | FACTOR OPERATIONS TERMINAL</h1>", unsafe_allow_html=True)
        reason = "; ".join(pm_inputs["messages"]) if pm_inputs["messages"] else "missing required return inputs"
        st.warning(
            "PM vitals are disabled: "
            f"{reason}. Provide `portfolio_returns.csv` and `benchmark_returns.csv` "
            "with overlapping history (>= 63 rows)."
        )
        st.markdown("---")
    
    # Main tabs (including Phase 2 enhancements)
    tab_discover, tab_construct, tab_phase2_regime, tab_phase2_meta, tab_risk = st.tabs([
        " Factor Lab",
        " Portfolio Constructor",
        " Regime RS-MVO",  # Phase 2
        " Meta-Model",     # Phase 2
        " Risk & Drawdown"
    ])
    
    with tab_discover:
        render_factor_lab(data, analyzers)
    
    with tab_construct:
        render_portfolio_constructor(data, analyzers)
    
    with tab_phase2_regime:
        render_regime_conditional_optimization(data, analyzers)
    
    with tab_phase2_meta:
        render_meta_model_aggregation(data, analyzers)
    
    with tab_risk:
        render_risk_drawdown(data, analyzers)
    
    # Footer
    st.markdown("---")
    st.caption(f"""
    Alpha Command Center v2.1 | Factor Operations Terminal | 
    Phase 2: RS-MVO + XGBoost Meta-Model | 
    Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
    Data as of: {data['returns'].index.max().strftime('%Y-%m-%d') if data['returns'] is not None else 'N/A'}
    """)


if __name__ == "__main__":
    main()
