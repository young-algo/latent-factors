import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Import signal modules
from .trading_signals import FactorMomentumAnalyzer
from .cross_sectional import CrossSectionalAnalyzer
from .regime_detection import RegimeDetector

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
            names_df = pd.read_csv("factor_names.csv", header=None, index_col=0)
            names = names_df[1].to_dict()
        elif Path("factor_names.json").exists():
            with open("factor_names.json", "r") as f:
                names = json.load(f)
        
        return returns, loadings, names
    except FileNotFoundError:
        return None, None, None

def create_signal_dashboard(momentum_analyzer: FactorMomentumAnalyzer):
    """Create signal status dashboard."""
    st.header("📡 Trading Signals")

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

        styled_summary = signal_summary.style.applymap(
            color_signal, subset=['combined_signal']
        )
        st.dataframe(styled_summary, use_container_width=True)

        # Extreme alerts panel
        st.subheader("🚨 Extreme Value Alerts")
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

            styled_alerts = alert_df.style.applymap(color_alert, subset=['Direction'])
            st.dataframe(styled_alerts, use_container_width=True)
        else:
            st.info("No extreme value alerts at this time.")


def create_momentum_charts(momentum_analyzer: FactorMomentumAnalyzer, factor_name: str):
    """Create momentum charts with RSI/MACD overlays."""
    st.subheader(f"📈 Momentum Analysis: {factor_name}")

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
    st.header("🎯 Market Regime")

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
    st.header("🔥 Cross-Sectional Rankings")

    # Calculate scores and rankings
    scores = cross_analyzer.calculate_factor_scores()
    rankings = cross_analyzer.rank_universe(scores)

    # Display top and bottom deciles
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Decile (Long Candidates)")
        top_decile = rankings[rankings['decile'] == 1].head(10)
        st.dataframe(top_decile[['score', 'rank', 'percentile']], use_container_width=True)

    with col2:
        st.subheader("Bottom Decile (Short Candidates)")
        bottom_decile = rankings[rankings['decile'] == 10].tail(10)
        st.dataframe(bottom_decile[['score', 'rank', 'percentile']], use_container_width=True)

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


def main():
    st.title("📊 Equity Factors Research Dashboard")

    returns, loadings, names = load_data()

    if returns is None:
        st.error("❌ Data not found. Please run 'python src/cli.py discover' first.")
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

    # --- Regime Section ---
    if regime_detector:
        st.header("📊 Regime Analysis")

        try:
            regime_summary = regime_detector.get_regime_summary()
            st.subheader("Regime Statistics")
            st.dataframe(regime_summary, use_container_width=True)

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
                st.dataframe(pd.DataFrame(pred_data), use_container_width=True)
            except Exception as e:
                st.warning(f"Regime prediction not available: {e}")

        except Exception as e:
            st.warning(f"Regime analysis not available: {e}")

if __name__ == "__main__":
    main()
