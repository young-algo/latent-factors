import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import pickle

# Import signal modules
try:
    # Relative imports when used as a package (e.g., python -m src.dashboard)
    from .trading_signals import FactorMomentumAnalyzer
    from .cross_sectional import CrossSectionalAnalyzer
    from .regime_detection import RegimeDetector
    from .factor_optimization import SharpeOptimizer
    from .factor_weighting import OptimalFactorWeighter
    from .database import check_database_health, get_database_path, get_db_connection
    from .config import config
except ImportError:
    # Absolute imports when PYTHONPATH is set (e.g., via CLI)
    from src.trading_signals import FactorMomentumAnalyzer
    from src.cross_sectional import CrossSectionalAnalyzer
    from src.regime_detection import RegimeDetector
    from src.factor_optimization import SharpeOptimizer
    from src.factor_weighting import OptimalFactorWeighter
    from src.database import check_database_health, get_database_path, get_db_connection
    from src.config import config

st.set_page_config(page_title="Equity Factors Dashboard", layout="wide")

@st.cache_data
def load_data():
    """Load factor data from CSV/JSON files."""
    try:
        returns = pd.read_csv("factor_returns.csv", index_col=0, parse_dates=True)
        loadings = pd.read_csv("factor_loadings.csv", index_col=0)
        
        # Try loading names from CSV first, then JSON
        names = {}
        if Path("factor_names.csv").exists():
            try:
                names_df = pd.read_csv("factor_names.csv", header=None, index_col=0)
                names = names_df[1].to_dict()
            except Exception:
                pass
        elif Path("factor_names.json").exists():
            with open("factor_names.json", "r") as f:
                names = json.load(f)
        
        return returns, loadings, names
    except FileNotFoundError:
        return None, None, None


def create_signal_dashboard(momentum_analyzer: FactorMomentumAnalyzer):
    """Create signal status dashboard."""
    st.header(" Trading Signals")

    # Signal summary
    signal_summary = momentum_analyzer.get_signal_summary()

    if not signal_summary.empty:
        # Display signal status table
        st.subheader("Signal Status Table")

        # Color coding
        def color_signal(val):
            if val == 'strong_buy':
                return 'background-color: darkgreen; color: white'
            elif val == 'buy':
                return 'background-color: lightgreen'
            elif val == 'strong_sell':
                return 'background-color: darkred; color: white'
            elif val == 'sell':
                return 'background-color: lightcoral'
            return ''

        styled_summary = signal_summary.style.map(
            color_signal, subset=['combined_signal']
        )
        st.dataframe(styled_summary, width='stretch')

        # Extreme alerts panel
        st.subheader(" Extreme Value Alerts")
        extreme_alerts = momentum_analyzer.get_all_extreme_alerts()

        if extreme_alerts:
            alert_data = []
            for alert in extreme_alerts:
                alert_data.append({
                    'Factor': alert.factor_name,
                    'Z-Score': f"{alert.z_score:.2f}",
                    'Percentile': f"{alert.percentile:.1f}",
                    'Direction': alert.direction,
                    'Alert Type': alert.alert_type
                })

            alert_df = pd.DataFrame(alert_data)

            # Color code alerts
            def color_alert(val):
                if 'extreme_high' in str(val):
                    return 'background-color: red; color: white'
                elif 'extreme_low' in str(val):
                    return 'background-color: green; color: white'
                return ''

            styled_alerts = alert_df.style.map(color_alert, subset=['Direction'])
            st.dataframe(styled_alerts, width='stretch')
        else:
            st.info("No extreme value alerts at this time.")


def create_momentum_charts(momentum_analyzer: FactorMomentumAnalyzer, factor_name: str):
    """Create momentum charts with RSI/MACD overlays."""
    st.subheader(f" Momentum Analysis: {factor_name}")

    col1, col2 = st.columns(2)

    with col1:
        # RSI Chart
        rsi = momentum_analyzer.calculate_rsi(factor_name)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(rsi.index, rsi.values, label='RSI', color='blue')
        ax.axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
        ax.axhline(y=30, color='g', linestyle='--', label='Oversold (30)')
        ax.fill_between(rsi.index, 30, 70, alpha=0.1, color='gray')
        ax.set_ylim(0, 100)
        ax.set_title(f'RSI - {factor_name}')
        ax.set_ylabel('RSI')
        ax.legend()
        st.pyplot(fig)

        # Current RSI value
        current_rsi = rsi.iloc[-1]
        st.metric("Current RSI", f"{current_rsi:.1f}")

    with col2:
        # MACD Chart
        macd_line, signal_line, histogram = momentum_analyzer.calculate_macd(factor_name)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(macd_line.index, macd_line.values, label='MACD', color='blue')
        ax.plot(signal_line.index, signal_line.values, label='Signal', color='red')
        ax.bar(histogram.index, histogram.values, label='Histogram', color='gray', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_title(f'MACD - {factor_name}')
        ax.set_ylabel('MACD')
        ax.legend()
        st.pyplot(fig)

        # MACD signal
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        macd_signal = "Bullish" if current_macd > current_signal else "Bearish"
        st.metric("MACD Signal", macd_signal)


def create_regime_gauge(regime_detector: RegimeDetector):
    """Create regime probability gauge chart."""
    st.header(" Market Regime")

    try:
        current_regime = regime_detector.detect_current_regime()
        regime_probs = regime_detector.get_regime_probabilities()

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Current Regime", current_regime.regime.value.replace('_', ' ').title())
            st.metric("Confidence", f"{current_regime.probability:.1%}")
            st.metric("Volatility", f"{current_regime.volatility:.2%}")
            st.write(f"**Description:** {current_regime.description}")

        with col2:
            # Regime probability bar chart
            if regime_probs:
                prob_df = pd.DataFrame({
                    'Regime': [r.value.replace('_', ' ').title() for r in regime_probs.keys()],
                    'Probability': list(regime_probs.values())
                })
                prob_df = prob_df.sort_values('Probability', ascending=True)

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(prob_df['Regime'], prob_df['Probability'])
                ax.set_xlabel('Probability')
                ax.set_title('Regime Probabilities')
                ax.set_xlim(0, 1)
                st.pyplot(fig)

    except Exception as e:
        st.warning(f"Regime detection not available: {e}")


def create_cross_sectional_heatmap(cross_analyzer: CrossSectionalAnalyzer):
    """Create cross-sectional ranking heatmap."""
    st.header(" Cross-Sectional Rankings")

    # Calculate scores and rankings
    scores = cross_analyzer.calculate_factor_scores()
    rankings = cross_analyzer.rank_universe(scores)

    # Display top and bottom deciles
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Decile (Long Candidates)")
        top_decile = rankings[rankings['decile'] == 1].head(10)
        st.dataframe(top_decile[['score', 'rank', 'percentile']], width='stretch')

    with col2:
        st.subheader("Bottom Decile (Short Candidates)")
        bottom_decile = rankings[rankings['decile'] == 10].tail(10)
        st.dataframe(bottom_decile[['score', 'rank', 'percentile']], width='stretch')

    # Full ranking heatmap
    st.subheader("Full Universe Rankings")

    # Create decile heatmap
    decile_counts = rankings.groupby('decile').size()
    fig, ax = plt.subplots(figsize=(12, 2))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, 10))
    bars = ax.bar(decile_counts.index, decile_counts.values, color=colors)
    ax.set_xlabel('Decile')
    ax.set_ylabel('Count')
    ax.set_title('Stock Distribution by Decile')
    st.pyplot(fig)


def create_optimization_panel(returns: pd.DataFrame, loadings: pd.DataFrame, factor_names: dict):
    """Create factor weight optimization panel."""
    st.header(" Factor Weight Optimization")
    
    with st.expander("Configure Optimization", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            lookback = st.slider("Lookback (days)", 21, 252, 126, key="opt_lookback")
        with col2:
            available_methods = ['equal', 'sharpe', 'momentum', 'risk_parity', 
                                'min_variance', 'max_diversification', 'pca']
            methods = st.multiselect(
                "Methods to Blend",
                available_methods,
                default=['sharpe', 'momentum', 'risk_parity']
            )
        with col3:
            technique = st.selectbox(
                "Optimization Technique",
                ['differential', 'gradient', 'bayesian'],
                help="Differential: global optimization (recommended). Bayesian: uses Optuna if available."
            )
        
        run_walk_forward = st.checkbox("Run Walk-Forward Optimization", value=False)
        
        if run_walk_forward:
            col1, col2 = st.columns(2)
            with col1:
                train_window = st.slider("Train Window (days)", 21, 504, 126, key="wf_train")
            with col2:
                test_window = st.slider("Test Window (days)", 5, 126, 21, key="wf_test")
    
    if st.button(" Run Optimization", type="primary"):
        if len(methods) < 2:
            st.error("Please select at least 2 methods to blend.")
            return
        
        try:
            with st.spinner("Initializing optimizer..."):
                optimizer = SharpeOptimizer(returns, loadings)
            
            if run_walk_forward:
                with st.spinner(f"Running walk-forward optimization... This may take a while."):
                    wf_results = optimizer.walk_forward_optimize(
                        train_window=train_window,
                        test_window=test_window,
                        methods=methods,
                        technique=technique,
                        verbose=False
                    )
                
                st.success(f"Walk-forward optimization complete! ({len(wf_results)} periods)")
                
                # Display walk-forward results
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_train_sharpe = wf_results['train_sharpe'].mean()
                    st.metric("Avg Train Sharpe", f"{avg_train_sharpe:.2f}")
                with col2:
                    avg_test_sharpe = wf_results['test_sharpe'].mean()
                    st.metric("Avg Test Sharpe", f"{avg_test_sharpe:.2f}")
                with col3:
                    st.metric("Periods", len(wf_results))
                
                # Plot rolling method weights
                st.subheader("Rolling Method Weights Over Time")
                method_weights_df = pd.DataFrame(wf_results['method_weights'].tolist())
                method_weights_df.index = wf_results['date']
                st.area_chart(method_weights_df)
                
                # Store results in session state for basket generation
                st.session_state['wf_results'] = wf_results
                st.session_state['factor_loadings'] = loadings
                
            else:
                with st.spinner(f"Optimizing blend using {technique}..."):
                    result = optimizer.optimize_blend(
                        lookback=lookback,
                        methods=methods,
                        technique=technique,
                        verbose=False
                    )
                
                st.success("Optimization complete!")
                
                # Display results in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Method Allocation")
                    method_df = pd.DataFrame({
                        'Method': list(result.method_allocation.keys()),
                        'Weight': list(result.method_allocation.values())
                    })
                    method_df = method_df[method_df['Weight'] > 0.01]  # Filter small weights
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    colors = plt.cm.Set3(np.linspace(0, 1, len(method_df)))
                    wedges, texts, autotexts = ax.pie(
                        method_df['Weight'], 
                        labels=method_df['Method'],
                        autopct='%1.1f%%',
                        colors=colors,
                        startangle=90
                    )
                    ax.set_title('Optimal Method Blend')
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("Factor Weights")
                    factor_df = pd.DataFrame({
                        'Factor': list(result.optimal_weights.keys()),
                        'Weight': list(result.optimal_weights.values())
                    })
                    factor_df = factor_df.sort_values('Weight', ascending=True)
                    
                    # Add factor names if available
                    if factor_names:
                        factor_df['Name'] = factor_df['Factor'].map(factor_names)
                        factor_df['Label'] = factor_df['Factor'] + ': ' + factor_df['Name']
                    else:
                        factor_df['Label'] = factor_df['Factor']
                    
                    fig, ax = plt.subplots(figsize=(10, max(4, len(factor_df) * 0.4)))
                    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.7, len(factor_df)))
                    bars = ax.barh(factor_df['Label'], factor_df['Weight'], color=colors)
                    ax.set_xlabel('Weight')
                    ax.set_title('Optimal Factor Weights')
                    ax.axvline(x=factor_df['Weight'].mean(), color='red', linestyle='--', alpha=0.5, label='Mean')
                    ax.legend()
                    st.pyplot(fig)
                
                # Performance metrics
                st.subheader("Performance Metrics")
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                with mcol1:
                    st.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
                with mcol2:
                    st.metric("Annualized Return", f"{result.annualized_return:.2%}")
                with mcol3:
                    st.metric("Annualized Volatility", f"{result.annualized_volatility:.2%}")
                with mcol4:
                    if result.annualized_volatility > 0:
                        return_risk = result.annualized_return / result.annualized_volatility
                        st.metric("Return/Risk", f"{return_risk:.2f}")
                
                # Store results in session state for basket generation
                st.session_state['opt_result'] = result
                st.session_state['factor_loadings'] = loadings
                
        except Exception as e:
            st.error(f"Optimization failed: {e}")
            import traceback
            st.code(traceback.format_exc())


def create_basket_generator(loadings: pd.DataFrame, factor_names: dict):
    """Create tradeable basket generator panel."""
    st.header(" Tradeable Basket Generator")
    
    # Check if optimization results exist
    has_single_opt = 'opt_result' in st.session_state
    has_wf_opt = 'wf_results' in st.session_state
    
    if not (has_single_opt or has_wf_opt):
        st.info(" Run Factor Weight Optimization first to generate a tradeable basket.")
        return
    
    # Source selection
    if has_single_opt and has_wf_opt:
        source = st.radio(
            "Select Optimization Source",
            ["Single-Period Optimization", "Walk-Forward Optimization (Averaged)"]
        )
        use_wf = source == "Walk-Forward Optimization (Averaged)"
    elif has_wf_opt:
        use_wf = True
        st.info("Using Walk-Forward optimization results (averaged across all periods).")
    else:
        use_wf = False
        st.info("Using Single-Period optimization results.")
    
    with st.expander("Basket Configuration", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            long_pct = st.slider("Long Percentile", 0.01, 0.5, 0.1, 
                                help="Top N% of stocks to go long", key="bg_long")
        with col2:
            short_pct = st.slider("Short Percentile", 0.01, 0.5, 0.1,
                                 help="Bottom N% of stocks to short", key="bg_short")
        with col3:
            capital = st.number_input("Capital ($)", min_value=1000, value=100000, step=1000, key="bg_capital")
        with col4:
            net_exposure = st.slider("Net Exposure", 0.0, 2.0, 1.0,
                                    help="1.0 = 100% net long, 0.0 = market neutral", key="bg_exposure")
    
    if st.button(" Generate Basket", type="primary"):
        try:
            # Get factor weights from optimization results
            if use_wf:
                wf_results = st.session_state['wf_results']
                # Average factor weights across all periods
                all_weights = [period['factor_weights'] for period in wf_results.to_dict('records')]
                weights_df = pd.DataFrame(all_weights)
                factor_weights = weights_df.mean().to_dict()
                st.info(f"Averaged factor weights across {len(wf_results)} walk-forward periods.")
            else:
                opt_result = st.session_state['opt_result']
                factor_weights = opt_result.optimal_weights
            
            # Calculate composite stock scores
            composite_score = pd.Series(0.0, index=loadings.index)
            for factor, weight in factor_weights.items():
                if factor in loadings.columns:
                    composite_score += loadings[factor] * weight
            
            composite_score = composite_score.sort_values(ascending=False)
            n_stocks = len(composite_score)
            
            # Select longs and shorts
            n_long = max(1, int(n_stocks * long_pct))
            n_short = max(1, int(n_stocks * short_pct))
            
            longs = composite_score.head(n_long)
            shorts = composite_score.tail(n_short)
            
            # Calculate position weights
            longs_weighted = longs / longs.sum() if longs.sum() > 0 else longs
            shorts_weighted = shorts / shorts.sum() * -1 if shorts.sum() != 0 else shorts
            
            # Create positions dataframe
            positions = pd.DataFrame({
                'ticker': list(longs.index) + list(shorts.index),
                'composite_score': list(longs.values) + list(shorts.values),
                'target_weight': list(longs_weighted.values * net_exposure) + 
                                list(shorts_weighted.values * net_exposure * -1),
                'side': ['LONG'] * n_long + ['SHORT'] * n_short
            })
            
            # Calculate dollar positions
            positions['position_dollars'] = positions['target_weight'] * capital
            
            # Display results
            st.subheader("Basket Positions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"** TOP {n_long} LONG POSITIONS:**")
                long_df = positions[positions['side'] == 'LONG'].copy()
                if factor_names:
                    long_df['name'] = long_df.index.map(lambda x: factor_names.get(x, ''))
                st.dataframe(
                    long_df[['ticker', 'composite_score', 'target_weight', 'position_dollars']].style.format({
                        'composite_score': '{:.4f}',
                        'target_weight': '{:.2%}',
                        'position_dollars': '${:,.0f}'
                    }),
                    width='stretch'
                )
            
            with col2:
                st.markdown(f"** TOP {n_short} SHORT POSITIONS:**")
                short_df = positions[positions['side'] == 'SHORT'].copy()
                st.dataframe(
                    short_df[['ticker', 'composite_score', 'target_weight', 'position_dollars']].style.format({
                        'composite_score': '{:.4f}',
                        'target_weight': '{:.2%}',
                        'position_dollars': '${:,.0f}'
                    }),
                    width='stretch'
                )
            
            # Portfolio summary
            st.subheader("Portfolio Summary")
            gross_exposure = positions['target_weight'].abs().sum()
            net_exposure = positions['target_weight'].sum()
            
            s_col1, s_col2, s_col3, s_col4 = st.columns(4)
            with s_col1:
                st.metric("Gross Exposure", f"{gross_exposure:.1%}")
            with s_col2:
                st.metric("Net Exposure", f"{net_exposure:.1%}")
            with s_col3:
                st.metric("Number of Positions", len(positions))
            with s_col4:
                long_short_ratio = (n_long / (n_long + n_short)) * 100
                st.metric("Long Bias", f"{long_short_ratio:.0f}%")
            
            # Weight distribution chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Long weights
            long_weights = positions[positions['side'] == 'LONG']['target_weight'].sort_values(ascending=True)
            ax1.barh(range(len(long_weights)), long_weights.values, color='green', alpha=0.7)
            ax1.set_yticks(range(len(long_weights)))
            ax1.set_yticklabels(long_weights.index, fontsize=8)
            ax1.set_xlabel('Weight')
            ax1.set_title('Long Position Weights')
            ax1.axvline(x=0, color='black', linewidth=0.5)
            
            # Short weights (absolute values)
            short_weights = positions[positions['side'] == 'SHORT']['target_weight'].abs().sort_values(ascending=True)
            ax2.barh(range(len(short_weights)), short_weights.values, color='red', alpha=0.7)
            ax2.set_yticks(range(len(short_weights)))
            ax2.set_yticklabels(short_weights.index, fontsize=8)
            ax2.set_xlabel('Weight (Absolute)')
            ax2.set_title('Short Position Weights')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Export option
            csv = positions.to_csv(index=False)
            st.download_button(
                label=" Download Basket as CSV",
                data=csv,
                file_name=f"trade_basket_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Basket generation failed: {e}")
            import traceback
            st.code(traceback.format_exc())


def create_factor_characteristics_panel(returns: pd.DataFrame, loadings: pd.DataFrame, factor_names: dict):
    """Create factor characteristics panel."""
    st.header(" Factor Characteristics")
    
    lookback = st.slider("Lookback Period (days)", 21, 252, 63, key="char_lookback")
    
    try:
        weighter = OptimalFactorWeighter(loadings, returns)
        characteristics = weighter.get_factor_characteristics(lookback=lookback)
        
        # Convert to DataFrame
        char_data = []
        for factor, char in characteristics.items():
            char_data.append({
                'Factor': factor,
                'Name': factor_names.get(factor, ''),
                'Sharpe Ratio': char.sharpe_ratio,
                'Mean Return': char.mean_return,
                'Volatility': char.volatility,
                'Max Drawdown': char.max_drawdown,
                'Win Rate': char.win_rate
            })
        
        char_df = pd.DataFrame(char_data)
        
        # Display with color coding
        st.dataframe(
            char_df.style.format({
                'Sharpe Ratio': '{:.2f}',
                'Mean Return': '{:.4f}',
                'Volatility': '{:.4f}',
                'Max Drawdown': '{:.2%}',
                'Win Rate': '{:.1%}'
            }).background_gradient(subset=['Sharpe Ratio'], cmap='RdYlGn', center=0)
              .background_gradient(subset=['Win Rate'], cmap='RdYlGn', vmin=0, vmax=1)
              .background_gradient(subset=['Max Drawdown'], cmap='RdYlGn_r', vmin=-0.5, vmax=0),
            width='stretch'
        )
        
        # Individual weighting method comparison
        st.subheader("Weighting Method Comparison")
        
        if st.button("Calculate All Weighting Methods"):
            with st.spinner("Calculating weights for all methods..."):
                methods_to_calc = {
                    'Equal': lambda: weighter.equal_weights(),
                    'Sharpe': lambda: weighter.sharpe_weights(lookback=lookback),
                    'Momentum': lambda: weighter.momentum_weights(lookback=min(lookback, 126)),
                    'Risk Parity': lambda: weighter.risk_parity_weights(lookback=lookback),
                    'Min Variance': lambda: weighter.min_variance_weights(lookback=lookback),
                    'Max Diversification': lambda: weighter.max_diversification_weights(lookback=lookback),
                    'PCA': lambda: weighter.pca_weights()
                }
                
                all_weights = {}
                for method_name, method_func in methods_to_calc.items():
                    try:
                        all_weights[method_name] = method_func()
                    except Exception as e:
                        st.warning(f"Could not calculate {method_name}: {e}")
                
                # Create comparison dataframe
                comp_df = pd.DataFrame(all_weights).T.fillna(0)
                
                # Heatmap
                fig, ax = plt.subplots(figsize=(12, max(4, len(comp_df) * 0.5)))
                sns.heatmap(comp_df, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=ax)
                ax.set_title('Factor Weights by Method')
                ax.set_xlabel('Factor')
                ax.set_ylabel('Weighting Method')
                plt.tight_layout()
                st.pyplot(fig)
                
    except Exception as e:
        st.error(f"Could not calculate characteristics: {e}")


def create_database_health_panel():
    """Create database health monitor panel."""
    st.header(" Database Health Monitor")
    
    try:
        health = check_database_health()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            exists = " Yes" if health['exists'] else " No"
            st.metric("Database Exists", exists)
        with col2:
            st.metric("Size (MB)", f"{health['size_mb']:.1f}")
        with col3:
            locked = " Locked" if health['is_locked'] else " Unlocked"
            st.metric("Lock Status", locked)
        with col4:
            healthy = " Healthy" if health['is_healthy'] else " Issues"
            st.metric("Health", healthy)
        
        if health['tables']:
            st.subheader("Table Record Counts")
            counts_df = pd.DataFrame([
                {'Table': k, 'Records': v} 
                for k, v in health['record_counts'].items()
            ])
            st.dataframe(counts_df, width='stretch')
        
        if health['errors']:
            st.error("Errors detected:")
            for error in health['errors']:
                st.write(f"- {error}")
        elif health['exists']:
            st.success("Database is healthy!")
        
    except Exception as e:
        st.warning(f"Could not check database health: {e}")


def create_pit_universe_panel():
    """Create Point-in-Time (PIT) Universe Construction panel."""
    st.header(" Point-in-Time (PIT) Universe Construction")
    
    st.markdown("""
    **Eliminate survivorship bias from backtesting by reconstructing true historical market state.**
    
    Unlike `get_etf_holdings()` which projects current constituents into the past, 
    PIT uses Alpha Vantage LISTING_STATUS to include delisted stocks like Lehman Brothers (LEH) 
    and Silicon Valley Bank (SIVB).
    """)
    
    # Check if PIT table exists
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='historical_universes'")
            pit_exists = cursor.fetchone() is not None
            
            if pit_exists:
                cursor.execute("SELECT COUNT(DISTINCT date) as num_dates, COUNT(*) as total_records FROM historical_universes")
                pit_stats = cursor.fetchone()
                num_dates, total_records = pit_stats
            else:
                num_dates, total_records = 0, 0
    except Exception as e:
        st.error(f"Could not check PIT table: {e}")
        pit_exists = False
        num_dates, total_records = 0, 0
    
    # PIT Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("PIT Table Exists", " Yes" if pit_exists else " No")
    with col2:
        st.metric("Stored Universe Dates", num_dates)
    with col3:
        st.metric("Total Records", f"{total_records:,}")
    
    # Tabs for different PIT functions
    pit_tab1, pit_tab2, pit_tab3 = st.tabs(["Build Universe", "View Universe", "QA Verification"])
    
    # Tab 1: Build Universe
    with pit_tab1:
        st.subheader("Build PIT Universe")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            pit_date = st.date_input(
                "Target Date",
                value=pd.Timestamp("2023-01-01"),
                min_value=pd.Timestamp("2000-01-01"),
                max_value=pd.Timestamp.now(),
                help="The historical date to reconstruct"
            )
        with col2:
            pit_top_n = st.number_input(
                "Top N Stocks",
                min_value=100,
                max_value=5000,
                value=500,
                step=100,
                help="Number of stocks by dollar volume"
            )
        with col3:
            pit_exchanges = st.multiselect(
                "Exchanges",
                options=["NYSE", "NASDAQ", "AMEX"],
                default=["NYSE", "NASDAQ"],
                help="Filter by exchange (reduces API calls)"
            )
        
        skip_delisted = st.checkbox(
            "Skip Delisted (Faster, Less Accurate)",
            value=False,
            help="Only fetch active listings (not recommended for backtesting)"
        )
        
        if st.button(" Build PIT Universe", type="primary"):
            api_key = config.ALPHAVANTAGE_API_KEY
            if not api_key:
                st.error(" ALPHAVANTAGE_API_KEY not configured. Set it in .env file.")
            else:
                try:
                    with st.spinner("Initializing DataBackend..."):
                        try:
                            from .alphavantage_system import DataBackend
                        except ImportError:
                            from src.alphavantage_system import DataBackend
                        
                        backend = DataBackend(api_key)
                    
                    with st.spinner(f"Building PIT universe for {pit_date}... This may take several minutes."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("Fetching listing status from Alpha Vantage...")
                        progress_bar.progress(10)
                        
                        universe_df = backend.build_point_in_time_universe(
                            date_str=pit_date.strftime('%Y-%m-%d'),
                            top_n=pit_top_n,
                            exchanges=pit_exchanges if pit_exchanges else None,
                            skip_delisted=skip_delisted
                        )
                        
                        progress_bar.progress(100)
                        status_text.text("Complete!")
                        
                        if not universe_df.empty:
                            st.success(f" Built PIT universe: {len(universe_df)} stocks")
                            
                            # Show summary stats
                            active_count = (universe_df['status'] == 'Active').sum()
                            delisted_count = (universe_df['status'] == 'Delisted').sum()
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Stocks", len(universe_df))
                            with col2:
                                st.metric("Active", active_count)
                            with col3:
                                st.metric("Delisted", delisted_count)
                            
                            # Show top stocks
                            st.subheader("Top 20 Stocks by Dollar Volume")
                            st.dataframe(
                                universe_df.head(20)[['ticker', 'status', 'dollar_volume', 'exchange']].style.format({
                                    'dollar_volume': '{:,.0f}'
                                }),
                                width='stretch'
                            )
                            
                            # Show delisted stocks if any
                            if delisted_count > 0:
                                st.subheader("Delisted Stocks in Universe")
                                delisted_df = universe_df[universe_df['status'] == 'Delisted']
                                st.dataframe(
                                    delisted_df[['ticker', 'dollar_volume', 'exchange']].style.format({
                                        'dollar_volume': '{:,.0f}'
                                    }),
                                    width='stretch'
                                )
                            
                            # Export option
                            csv = universe_df.to_csv(index=False)
                            st.download_button(
                                label=" Download Universe as CSV",
                                data=csv,
                                file_name=f"pit_universe_{pit_date.strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.error(" Failed to build universe. Check API key and try again.")
                            
                except Exception as e:
                    st.error(f"Error building PIT universe: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Tab 2: View Universe
    with pit_tab2:
        st.subheader("View Stored PIT Universe")
        
        try:
            # Get available dates
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT DISTINCT date FROM historical_universes ORDER BY date DESC LIMIT 100"
                )
                available_dates = [row[0] for row in cursor.fetchall()]
            
            if not available_dates:
                st.info("No PIT universes found. Build one in the 'Build Universe' tab.")
            else:
                view_date = st.selectbox(
                    "Select Universe Date",
                    options=available_dates,
                    format_func=lambda x: f"{x} ({len([d for d in available_dates if d == x])} records)"
                )
                
                view_top_n = st.slider("Show Top N", 10, 1000, 100)
                
                if st.button(" View Universe"):
                    with get_db_connection() as conn:
                        query = """
                            SELECT ticker, asset_type, status, dollar_volume
                            FROM historical_universes
                            WHERE date = ?
                            ORDER BY dollar_volume DESC
                            LIMIT ?
                        """
                        view_df = pd.read_sql(query, conn, params=(view_date, view_top_n))
                    
                    if not view_df.empty:
                        st.success(f"Showing top {len(view_df)} stocks from {view_date}")
                        
                        # Stats
                        active_count = (view_df['status'] == 'Active').sum()
                        delisted_count = (view_df['status'] == 'Delisted').sum()
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total", len(view_df))
                        with col2:
                            st.metric("Active", active_count, f"{active_count/len(view_df)*100:.1f}%")
                        with col3:
                            st.metric("Delisted", delisted_count, f"{delisted_count/len(view_df)*100:.1f}%")
                        
                        # Display with color coding
                        def color_status(val):
                            if val == 'Delisted':
                                return 'background-color: #ffcccc; color: #990000'
                            elif val == 'Active':
                                return 'background-color: #ccffcc; color: #006600'
                            return ''
                        
                        styled_df = view_df.style.applymap(
                            color_status, subset=['status']
                        ).format({
                            'dollar_volume': '{:,.0f}'
                        })
                        
                        st.dataframe(styled_df, width='stretch')
                        
                        # Show known delisted tickers
                        known_delisted = {'LEH', 'SIVB', 'FRC', 'WAMU', 'WB', 'WM', 'BEAR', 'BS'}
                        found_delisted = known_delisted.intersection(set(view_df['ticker'].tolist()))
                        
                        if found_delisted:
                            st.success(f" PIT Universe includes known delisted tickers: {', '.join(found_delisted)}")
                        elif delisted_count > 0:
                            st.info(f"ℹ Delisted stocks present but no famous bankruptcies found in top {view_top_n}")
                    else:
                        st.warning(f"No data found for {view_date}")
                        
        except Exception as e:
            st.error(f"Error viewing universe: {e}")
    
    # Tab 3: QA Verification
    with pit_tab3:
        st.subheader("QA Verification Tests")
        
        st.markdown("""
        **Critical QA Tests** (must pass before live capital deployment):
        
        1. **Lehman Brothers (LEH)** - Should be in 2008-09-15 universe
        2. **Silicon Valley Bank (SIVB)** - Should be in 2023-01-01 universe
        
        If these tests fail, the system suffers from **survivorship bias**.
        """)
        
        qa_col1, qa_col2 = st.columns(2)
        
        with qa_col1:
            st.markdown("**Test 1: Lehman Brothers**")
            st.caption("Date: 2008-09-15 (Bankruptcy)")
            
            if st.button(" Run LEH Test"):
                api_key = config.ALPHAVANTAGE_API_KEY
                if not api_key:
                    st.error("ALPHAVANTAGE_API_KEY not configured")
                else:
                    with st.spinner("Testing LEH inclusion..."):
                        try:
                            try:
                                from .research import FactorResearchSystem
                            except ImportError:
                                from src.research import FactorResearchSystem
                            
                            frs = FactorResearchSystem(api_key, universe=["SPY"])
                            result = frs.verify_pit_universe('2008-09-15', ['LEH'])
                            
                            if result['pass']:
                                st.success(f" PASS: LEH found in universe ({result['universe_size']} total stocks)")
                            else:
                                st.error(f" FAIL: LEH missing! Found: {result['found']}, Missing: {result['missing']}")
                                st.warning(" Survivorship bias detected - do not deploy to live capital!")
                        except Exception as e:
                            st.error(f"Test error: {e}")
        
        with qa_col2:
            st.markdown("**Test 2: Silicon Valley Bank**")
            st.caption("Date: 2023-01-01 (Pre-failure)")
            
            if st.button(" Run SIVB Test"):
                api_key = config.ALPHAVANTAGE_API_KEY
                if not api_key:
                    st.error("ALPHAVANTAGE_API_KEY not configured")
                else:
                    with st.spinner("Testing SIVB inclusion..."):
                        try:
                            try:
                                from .research import FactorResearchSystem
                            except ImportError:
                                from src.research import FactorResearchSystem
                            
                            frs = FactorResearchSystem(api_key, universe=["SPY"])
                            result = frs.verify_pit_universe('2023-01-01', ['SIVB'])
                            
                            if result['pass']:
                                st.success(f" PASS: SIVB found in universe ({result['universe_size']} total stocks)")
                            else:
                                st.error(f" FAIL: SIVB missing! Found: {result['found']}, Missing: {result['missing']}")
                                st.warning(" Survivorship bias detected - do not deploy to live capital!")
                        except Exception as e:
                            st.error(f"Test error: {e}")
        
        # Quick check section
        st.divider()
        st.subheader("Quick Ticker Check")
        
        quick_date = st.date_input(
            "Check Date",
            value=pd.Timestamp("2008-09-15"),
            key="quick_check_date"
        )
        quick_tickers = st.text_input(
            "Tickers to Check (comma-separated)",
            value="LEH, SIVB, FRC",
            help="Enter ticker symbols to verify presence in the universe"
        )
        
        if st.button(" Check Tickers"):
            try:
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT ticker, status, dollar_volume FROM historical_universes WHERE date = ?",
                        (quick_date.strftime('%Y-%m-%d'),)
                    )
                    universe_tickers = {row[0]: {'status': row[1], 'dv': row[2]} for row in cursor.fetchall()}
                
                if not universe_tickers:
                    st.warning(f"No universe found for {quick_date}. Build it first.")
                else:
                    check_list = [t.strip().upper() for t in quick_tickers.split(',')]
                    
                    results = []
                    for ticker in check_list:
                        if ticker in universe_tickers:
                            info = universe_tickers[ticker]
                            results.append({
                                'Ticker': ticker,
                                'Present': ' Yes',
                                'Status': info['status'],
                                'Dollar Volume': f"{info['dv']:,.0f}"
                            })
                        else:
                            results.append({
                                'Ticker': ticker,
                                'Present': ' No',
                                'Status': 'N/A',
                                'Dollar Volume': 'N/A'
                            })
                    
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, width='stretch')
                    
                    # Summary
                    present_count = sum(1 for r in results if r['Present'] == ' Yes')
                    if present_count == len(results):
                        st.success(f"All {len(results)} tickers found!")
                    else:
                        st.info(f"{present_count}/{len(results)} tickers found")
                        
            except Exception as e:
                st.error(f"Check error: {e}")


def main():
    st.title(" Equity Factors Research Dashboard")

    returns, loadings, names = load_data()

    if returns is None:
        st.error(" Data not found. Please run 'python -m src discover' first.")
        return

    # Initialize analyzers
    try:
        momentum_analyzer = FactorMomentumAnalyzer(returns)
    except Exception as e:
        st.error(f"Error initializing momentum analyzer: {e}")
        momentum_analyzer = None

    try:
        cross_analyzer = CrossSectionalAnalyzer(loadings)
    except Exception as e:
        st.error(f"Error initializing cross-sectional analyzer: {e}")
        cross_analyzer = None

    try:
        regime_detector = RegimeDetector(returns)
        regime_detector.fit_hmm(n_regimes=3)
    except Exception as e:
        st.warning(f"Regime detector initialization failed: {e}")
        regime_detector = None

    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # --- Overview Section ---
    st.header("Market Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Factors", len(returns.columns))
    with col2:
        st.metric("Date Range", f"{returns.index.min().date()} to {returns.index.max().date()}")
    with col3:
        latest_ret = returns.iloc[-1].mean()
        st.metric("Avg Daily Factor Return", f"{latest_ret:.4f}")
    with col4:
        if momentum_analyzer:
            extreme_count = len(momentum_analyzer.get_all_extreme_alerts())
            st.metric("Active Alerts", extreme_count)

    # --- Signal Dashboard Section ---
    if momentum_analyzer:
        create_signal_dashboard(momentum_analyzer)

    # Cumulative Returns Plot
    st.subheader("Cumulative Factor Returns")
    
    # Filter by date
    start_date = st.date_input("Start Date", returns.index.min())
    end_date = st.date_input("End Date", returns.index.max())
    
    filtered_returns = returns.loc[str(start_date):str(end_date)]
    cumulative_returns = (1 + filtered_returns).cumprod() - 1
    
    # Create display names mapping but keep original column names for charts
    display_names = {}
    if names:
        display_names = {c: f"{c}: {names.get(c, '')}" for c in cumulative_returns.columns}
        # Use original column names for the chart (avoids encoding issues)
        # But show friendly names in UI
    else:
        display_names = {c: c for c in cumulative_returns.columns}

    st.line_chart(cumulative_returns)

    # --- Factor Details Section ---
    st.header("Factor Deep Dive")

    # Create selectbox with display names but map back to original
    selected_display = st.selectbox("Select Factor", list(display_names.values()))
    # Extract original factor code (e.g., "F1" from "F1: Factor Name")
    selected_factor = [k for k, v in display_names.items() if v == selected_display][0]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Performance: {selected_display}")
        st.line_chart(cumulative_returns[selected_factor])

        # Stats
        fac_ret = filtered_returns[selected_factor]
        ann_vol = fac_ret.std() * np.sqrt(252)
        sharpe = (fac_ret.mean() * 252) / (ann_vol if ann_vol != 0 else 1)

        st.write(f"**Annualized Volatility:** {ann_vol:.2%}")
        st.write(f"**Sharpe Ratio:** {sharpe:.2f}")

        # Momentum charts if analyzer available
        if momentum_analyzer:
            create_momentum_charts(momentum_analyzer, selected_factor)

    with col2:
        st.subheader("Top Exposures")
        if selected_factor in loadings.columns:
            factor_loadings = loadings[selected_factor].sort_values(ascending=False)

            st.write("**Top Positive**")
            st.dataframe(factor_loadings.head(10))

            st.write("**Top Negative**")
            st.dataframe(factor_loadings.tail(10).sort_values())
        else:
            st.warning("Loadings data not available for this factor.")

        # Regime gauge if available
        if regime_detector:
            create_regime_gauge(regime_detector)

    # --- Watchlist Section ---
    st.header("Stock Watchlist")
    tickers = st.multiselect("Select Stocks", loadings.index.tolist(), default=loadings.index[:5].tolist())

    if tickers:
        st.subheader("Factor Exposures")
        watchlist_loadings = loadings.loc[tickers]

        # Heatmap
        fig, ax = plt.subplots(figsize=(10, len(tickers) * 0.5 + 2))
        sns.heatmap(watchlist_loadings, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax)
        st.pyplot(fig)

    # --- Cross-Sectional Rankings Section ---
    if cross_analyzer:
        create_cross_sectional_heatmap(cross_analyzer)

    # --- Factor Weight Optimization Section ---
    st.divider()
    create_optimization_panel(returns, loadings, names)
    
    # --- Tradeable Basket Generator Section ---
    st.divider()
    create_basket_generator(loadings, names)
    
    # --- Factor Characteristics Section ---
    st.divider()
    create_factor_characteristics_panel(returns, loadings, names)

    # --- Regime Section ---
    st.divider()
    if regime_detector:
        st.header(" Regime Analysis")

        try:
            regime_summary = regime_detector.get_regime_summary()
            st.subheader("Regime Statistics")
            st.dataframe(regime_summary, width='stretch')

            # Regime transition matrix
            st.subheader("Regime Transition Matrix")
            try:
                transitions = regime_detector.analyze_regime_transitions()
                if not transitions.empty:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(transitions, annot=True, fmt=".2f", cmap="Blues", ax=ax)
                    st.pyplot(fig)
            except Exception as e:
                st.warning(f"Transition analysis not available: {e}")

            # Regime predictions
            st.subheader("Regime Predictions (Next 5 Days)")
            try:
                predictions = regime_detector.predict_regime(duration=5)
                pred_data = []
                for i, pred in enumerate(predictions):
                    pred_data.append({
                        'Day': i + 1,
                        'Predicted Regime': pred.regime.value.replace('_', ' ').title(),
                        'Probability': f"{pred.probability:.1%}",
                        'Expected Trend': f"{pred.trend:.4f}"
                    })
                st.dataframe(pd.DataFrame(pred_data), width='stretch')
            except Exception as e:
                st.warning(f"Regime prediction not available: {e}")

        except Exception as e:
            st.warning(f"Regime analysis not available: {e}")
    
    # --- PIT Universe Section ---
    st.divider()
    create_pit_universe_panel()
    
    # --- Database Health Section (Sidebar) ---
    st.sidebar.divider()
    with st.sidebar.expander(" Database Health"):
        try:
            health = check_database_health()
            st.write(f"**Exists:** {'' if health['exists'] else ''}")
            st.write(f"**Size:** {health['size_mb']:.1f} MB")
            st.write(f"**Status:** {' Healthy' if health['is_healthy'] else ' Issues'}")
            if health['errors']:
                st.write("**Errors:**")
                for err in health['errors']:
                    st.write(f"- {err}")
        except Exception as e:
            st.write(f" Could not check: {e}")
    
    # PIT Quick Stats in Sidebar
    with st.sidebar.expander(" PIT Universe Status"):
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='historical_universes'"
                )
                if cursor.fetchone():
                    cursor.execute(
                        "SELECT COUNT(DISTINCT date), COUNT(*) FROM historical_universes"
                    )
                    num_dates, total = cursor.fetchone()
                    st.write(f"**Stored Dates:** {num_dates}")
                    st.write(f"**Total Records:** {total:,}")
                    
                    cursor.execute(
                        "SELECT MAX(date) FROM historical_universes"
                    )
                    latest = cursor.fetchone()[0]
                    if latest:
                        st.write(f"**Latest:** {latest}")
                else:
                    st.write(" PIT table not initialized")
        except Exception as e:
            st.write(f" Could not check: {e}")

if __name__ == "__main__":
    main()
