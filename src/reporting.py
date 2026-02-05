import pandas as pd
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Mapping, Any, Optional, List, Dict

# Import signal modules
from .trading_signals import FactorMomentumAnalyzer
from .cross_sectional import CrossSectionalAnalyzer
from .regime_detection import RegimeDetector
from .signal_aggregator import SignalAggregator

def generate_daily_report(output_path: str = "daily_report.html"):
    """
    Generate a daily HTML report of factor performance.
    """
    try:
        returns = pd.read_csv("factor_returns.csv", index_col=0, parse_dates=True)
        loadings = pd.read_csv("factor_loadings.csv", index_col=0)
        
        names = {}
        if Path("factor_names.csv").exists():
            names_df = pd.read_csv("factor_names.csv", header=None, index_col=0)
            names = names_df[1].to_dict()
        elif Path("factor_names.json").exists():
            with open("factor_names.json", "r") as f:
                names = json.load(f)
    except FileNotFoundError:
        print(" Data not found. Please run discovery first.")
        return

    if returns.empty:
        print(" No return data available.")
        return

    last_date = returns.index[-1]
    daily_rets = returns.iloc[-1]
    
    # Sort by absolute return to find top movers
    top_movers = daily_rets.abs().sort_values(ascending=False).head(5)
    
    html_content = f"""
    <html>
    <head>
        <title>Daily Factor Report - {last_date.date()}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .positive {{ color: green; }}
            .negative {{ color: red; }}
        </style>
    </head>
    <body>
        <h1>Daily Factor Report</h1>
        <p><strong>Date:</strong> {last_date.date()}</p>
        
        <h2>Top Movers</h2>
        <table>
            <tr>
                <th>Factor</th>
                <th>Name</th>
                <th>Return</th>
            </tr>
    """
    
    for factor, abs_ret in top_movers.items():
        ret = daily_rets[factor]
        name = names.get(factor, "Unknown")
        color_class = "positive" if ret >= 0 else "negative"
        html_content += f"""
            <tr>
                <td>{factor}</td>
                <td>{name}</td>
                <td class="{color_class}">{ret:.4%}</td>
            </tr>
        """
        
    html_content += """
        </table>
        
        <h2>All Factors</h2>
        <table>
            <tr>
                <th>Factor</th>
                <th>Name</th>
                <th>Daily Return</th>
                <th>Cumulative (YTD)</th>
            </tr>
    """
    
    # Calculate YTD returns (approximate)
    ytd_start = datetime(last_date.year, 1, 1)
    ytd_returns = returns[returns.index >= ytd_start]
    if not ytd_returns.empty:
        cum_ytd = (1 + ytd_returns).cumprod().iloc[-1] - 1
    else:
        cum_ytd = pd.Series(0, index=returns.columns)

    for factor in returns.columns:
        ret = daily_rets[factor]
        ytd = cum_ytd.get(factor, 0)
        name = names.get(factor, "Unknown")
        color_class = "positive" if ret >= 0 else "negative"
        ytd_color = "positive" if ytd >= 0 else "negative"
        
        html_content += f"""
            <tr>
                <td>{factor}</td>
                <td>{name}</td>
                <td class="{color_class}">{ret:.4%}</td>
                <td class="{ytd_color}">{ytd:.4%}</td>
            </tr>
        """

    html_content += """
        </table>
    </body>
    </html>
    """
    
    with open(output_path, "w") as f:
        f.write(html_content)
    
def generate_detailed_markdown_report(output_path: str = "detailed_report.md"):
    """
    Generate a detailed Markdown report with loadings, fundamentals, and returns.
    """
    try:
        # Load analysis config to get start date and database path
        db_path = "av_cache.db"
        start_date = "2020-04-01"
        if Path("factor_analysis_config.json").exists():
            with open("factor_analysis_config.json", "r") as f:
                config = json.load(f)
                start_date = config.get("start_date", start_date)
        
        loadings = pd.read_csv("factor_loadings.csv", index_col=0)
        
        names = {}
        if Path("factor_names.csv").exists():
            names_df = pd.read_csv("factor_names.csv", header=None, index_col=0)
            names = names_df[1].to_dict()
            
        conn = sqlite3.connect(db_path)
    except Exception as e:
        print(f" Error loading data: {e}")
        return

    def get_fundamentals(ticker):
        cur = conn.cursor()
        cur.execute("SELECT json FROM fundamentals WHERE ticker=?", (ticker,))
        row = cur.fetchone()
        if row:
            try:
                data = json.loads(row[0])
                return {
                    "Name": data.get("Name", "N/A"),
                    "Sector": data.get("Sector", "N/A"),
                    "Industry": data.get("Industry", "N/A")
                }
            except: pass
        return {"Name": "N/A", "Sector": "N/A", "Industry": "N/A"}

    def get_return(ticker):
        try:
            df = pd.read_sql("SELECT date, adj_close FROM prices WHERE ticker=?", conn, params=(ticker,), parse_dates=["date"])
            df = df.sort_values("date")
            df = df[df["date"] >= pd.to_datetime(start_date)]
            if len(df) < 2: return 0.0
            return (df.iloc[-1]["adj_close"] / df.iloc[0]["adj_close"]) - 1
        except: return 0.0

    def to_markdown_table(df):
        if df.empty: return ""
        headers = " | ".join(df.columns)
        sep = " | ".join(["---"] * len(df.columns))
        rows = [" | ".join([str(x) for x in row.values]) for _, row in df.iterrows()]
        return f"| {headers} |\n| {sep} |\n| " + " |\n| ".join(rows) + " |"

    report = [f"# Detailed Factor Analysis Report\n*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"]
    
    for factor in loadings.columns:
        f_name = names.get(factor, "Unknown Factor")
        report.append(f"\n## {factor}: {f_name}")
        
        for kind, top_fn in [("Highest", loadings[factor].nlargest), ("Lowest", loadings[factor].nsmallest)]:
            report.append(f"\n### Top 10 {kind} Loadings")
            rows = []
            for ticker, loading in top_fn(10).items():
                fnd = get_fundamentals(ticker)
                rows.append({
                    "Ticker": ticker,
                    "Loading": f"{loading:.4f}",
                    "Name": fnd["Name"],
                    "Sector": fnd["Sector"],
                    "Industry": fnd["Industry"],
                    "Return": f"{get_return(ticker):.2%}"
                })
            report.append(to_markdown_table(pd.DataFrame(rows)))
        report.append("\n---")

    with open(output_path, "w") as f:
        f.write("\n".join(report))
    
    conn.close()
    print(f" Detailed report generated: {output_path}")

def generate_signal_report(
    factor_returns: pd.DataFrame,
    factor_loadings: pd.DataFrame,
    output_path: str = "signal_report.html"
):
    """
    Generate a comprehensive HTML report of trading signals.

    Parameters
    ----------
    factor_returns : pd.DataFrame
        Factor returns data
    factor_loadings : pd.DataFrame
        Factor loadings data
    output_path : str
        Path to save the HTML report
    """
    try:
        # Initialize analyzers
        momentum_analyzer = FactorMomentumAnalyzer(factor_returns)
        cross_analyzer = CrossSectionalAnalyzer(factor_loadings)
        regime_detector = RegimeDetector(factor_returns)
        regime_detector.fit_hmm(n_regimes=3)

        # Get signal data
        signal_summary = momentum_analyzer.get_signal_summary()
        extreme_alerts = momentum_analyzer.get_all_extreme_alerts()
        current_regime = regime_detector.detect_current_regime()
        regime_summary = regime_detector.get_regime_summary()

        # Get cross-sectional signals
        cs_signals = cross_analyzer.generate_long_short_signals(top_pct=0.1, bottom_pct=0.1)
        longs = [s for s in cs_signals if s.direction.value == 'long']
        shorts = [s for s in cs_signals if s.direction.value == 'short']

        # Generate HTML
        html_content = f"""
    <html>
    <head>
        <title>Trading Signal Report - {datetime.now().date()}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
            h2 {{ color: #555; margin-top: 30px; border-bottom: 2px solid #ddd; padding-bottom: 8px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 15px; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .positive {{ color: green; font-weight: bold; }}
            .negative {{ color: red; font-weight: bold; }}
            .alert {{ background-color: #fff3cd; border: 1px solid #ffc107; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .alert-high {{ background-color: #f8d7da; border-color: #dc3545; }}
            .metric-box {{ display: inline-block; padding: 15px 25px; margin: 10px; background-color: #e9ecef; border-radius: 8px; text-align: center; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
            .metric-label {{ font-size: 12px; color: #666; }}
            .signal-buy {{ background-color: #d4edda; color: #155724; padding: 4px 8px; border-radius: 4px; }}
            .signal-sell {{ background-color: #f8d7da; color: #721c24; padding: 4px 8px; border-radius: 4px; }}
            .signal-neutral {{ background-color: #e2e3e5; color: #383d41; padding: 4px 8px; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1> Trading Signal Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <h2> Market Regime</h2>
            <div class="metric-box">
                <div class="metric-value">{current_regime.regime.value.replace('_', ' ').title()}</div>
                <div class="metric-label">Current Regime</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{current_regime.probability:.1%}</div>
                <div class="metric-label">Confidence</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{current_regime.volatility:.2%}</div>
                <div class="metric-label">Volatility</div>
            </div>
            <p><strong>Description:</strong> {current_regime.description}</p>

            <h2> Extreme Value Alerts ({len(extreme_alerts)} active)</h2>
    """

        if extreme_alerts:
            html_content += """
            <table>
                <tr>
                    <th>Factor</th>
                    <th>Z-Score</th>
                    <th>Percentile</th>
                    <th>Direction</th>
                    <th>Alert Type</th>
                </tr>
    """
            for alert in extreme_alerts:
                direction_class = 'negative' if alert.direction == 'extreme_high' else 'positive'
                html_content += f"""
                <tr>
                    <td>{alert.factor_name}</td>
                    <td>{alert.z_score:.2f}</td>
                    <td>{alert.percentile:.1f}</td>
                    <td class="{direction_class}">{alert.direction.replace('_', ' ').title()}</td>
                    <td>{alert.alert_type}</td>
                </tr>
    """
            html_content += "</table>"
        else:
            html_content += '<p class="alert">No extreme value alerts at this time.</p>'

        html_content += f"""
            <h2> Factor Signals</h2>
            <table>
                <tr>
                    <th>Factor</th>
                    <th>RSI</th>
                    <th>RSI Signal</th>
                    <th>MACD Signal</th>
                    <th>ADX</th>
                    <th>Regime</th>
                    <th>Combined Signal</th>
                </tr>
    """

        for _, row in signal_summary.iterrows():
            signal_class = 'signal-neutral'
            if 'buy' in row['combined_signal']:
                signal_class = 'signal-buy'
            elif 'sell' in row['combined_signal']:
                signal_class = 'signal-sell'

            html_content += f"""
                <tr>
                    <td>{row['factor']}</td>
                    <td>{row['rsi']:.1f}</td>
                    <td>{row['rsi_signal']}</td>
                    <td>{row['macd_signal']}</td>
                    <td>{row['adx']:.1f}</td>
                    <td>{row['regime']}</td>
                    <td><span class="{signal_class}">{row['combined_signal'].replace('_', ' ').title()}</span></td>
                </tr>
    """

        html_content += "</table>"

        html_content += f"""
            <h2> Cross-Sectional Rankings</h2>
            <h3>Top 10 Long Candidates</h3>
            <table>
                <tr>
                    <th>Ticker</th>
                    <th>Composite Score</th>
                    <th>Decile</th>
                    <th>Rank</th>
                    <th>Confidence</th>
                </tr>
    """

        for sig in longs[:10]:
            html_content += f"""
                <tr>
                    <td><strong>{sig.ticker}</strong></td>
                    <td>{sig.composite_score:.4f}</td>
                    <td>{sig.decile}</td>
                    <td>{sig.rank}</td>
                    <td>{sig.confidence:.1%}</td>
                </tr>
    """

        html_content += """
            </table>

            <h3>Top 10 Short Candidates</h3>
            <table>
                <tr>
                    <th>Ticker</th>
                    <th>Composite Score</th>
                    <th>Decile</th>
                    <th>Rank</th>
                    <th>Confidence</th>
                </tr>
    """

        for sig in shorts[:10]:
            html_content += f"""
                <tr>
                    <td><strong>{sig.ticker}</strong></td>
                    <td>{sig.composite_score:.4f}</td>
                    <td>{sig.decile}</td>
                    <td>{sig.rank}</td>
                    <td>{sig.confidence:.1%}</td>
                </tr>
    """

        html_content += """
            </table>

            <h2> Regime Statistics</h2>
            <table>
                <tr>
                    <th>Regime</th>
                    <th>Count</th>
                    <th>% Time</th>
                    <th>Avg Return</th>
                    <th>Avg Volatility</th>
                    <th>Sharpe Ratio</th>
                </tr>
    """

        for _, row in regime_summary.iterrows():
            html_content += f"""
                <tr>
                    <td>{row['regime']}</td>
                    <td>{row['count']}</td>
                    <td>{row['pct_time']:.1%}</td>
                    <td class="{'positive' if row['avg_return'] > 0 else 'negative'}">{row['avg_return']:.4f}</td>
                    <td>{row['avg_volatility']:.4f}</td>
                    <td>{row['sharpe']:.2f}</td>
                </tr>
    """

        html_content += """
            </table>
        </div>
    </body>
    </html>
    """

        with open(output_path, "w") as f:
            f.write(html_content)

        print(f" Signal report generated: {output_path}")

    except Exception as e:
        print(f" Error generating signal report: {e}")


def export_signals_to_csv(
    factor_returns: pd.DataFrame,
    factor_loadings: pd.DataFrame,
    output_path: str = "signals_export.csv"
):
    """
    Export all signals to CSV for external use.

    Parameters
    ----------
    factor_returns : pd.DataFrame
        Factor returns data
    factor_loadings : pd.DataFrame
        Factor loadings data
    output_path : str
        Path to save the CSV file
    """
    try:
        # Initialize analyzers
        momentum_analyzer = FactorMomentumAnalyzer(factor_returns)
        cross_analyzer = CrossSectionalAnalyzer(factor_loadings)

        # Collect all signals
        all_signals = []

        # Factor momentum signals
        for factor in factor_returns.columns:
            signals = momentum_analyzer.get_momentum_signals(factor)
            all_signals.append({
                'entity': factor,
                'entity_type': 'factor',
                'signal_type': 'momentum',
                'direction': signals['combined_signal'],
                'confidence': 70 if 'strong' in signals['combined_signal'] else 50,
                'rsi': signals.get('rsi'),
                'adx': signals.get('adx'),
                'regime': signals.get('regime')
            })

        # Cross-sectional signals
        cs_signals = cross_analyzer.generate_long_short_signals()
        for sig in cs_signals:
            all_signals.append({
                'entity': sig.ticker,
                'entity_type': 'stock',
                'signal_type': 'cross_sectional',
                'direction': sig.direction.value,
                'confidence': sig.confidence * 100,
                'composite_score': sig.composite_score,
                'decile': sig.decile,
                'rank': sig.rank
            })

        # Extreme alerts
        extreme_alerts = momentum_analyzer.get_all_extreme_alerts()
        for alert in extreme_alerts:
            all_signals.append({
                'entity': alert.factor_name,
                'entity_type': 'factor',
                'signal_type': 'extreme_value',
                'direction': 'short' if alert.direction == 'extreme_high' else 'long',
                'confidence': min(95, 50 + abs(alert.z_score) * 15),
                'z_score': alert.z_score,
                'percentile': alert.percentile
            })

        # Export to CSV
        df = pd.DataFrame(all_signals)
        df.to_csv(output_path, index=False)
        print(f" Signals exported to: {output_path}")

    except Exception as e:
        print(f" Error exporting signals: {e}")


def create_signal_alert_email(
    factor_returns: pd.DataFrame,
    factor_loadings: pd.DataFrame,
    recipient: str = "trader@example.com"
) -> str:
    """
    Create a formatted email alert with trading signals.

    Parameters
    ----------
    factor_returns : pd.DataFrame
        Factor returns data
    factor_loadings : pd.DataFrame
        Factor loadings data
    recipient : str
        Email recipient

    Returns
    -------
    str
        Formatted email content
    """
    try:
        # Initialize analyzers
        momentum_analyzer = FactorMomentumAnalyzer(factor_returns)
        cross_analyzer = CrossSectionalAnalyzer(factor_loadings)
        regime_detector = RegimeDetector(factor_returns)
        regime_detector.fit_hmm(n_regimes=3)

        # Get signals
        extreme_alerts = momentum_analyzer.get_all_extreme_alerts()
        current_regime = regime_detector.detect_current_regime()
        cs_signals = cross_analyzer.generate_long_short_signals(top_pct=0.1, bottom_pct=0.1)

        longs = [s for s in cs_signals if s.direction.value == 'long'][:5]
        shorts = [s for s in cs_signals if s.direction.value == 'short'][:5]

        # Build email content
        email = f"""Subject: Trading Signal Alert - {datetime.now().date()}

Trading Signal Alert
====================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MARKET REGIME
-------------
Current Regime: {current_regime.regime.value.replace('_', ' ').title()}
Confidence: {current_regime.probability:.1%}
Description: {current_regime.description}

EXTREME VALUE ALERTS ({len(extreme_alerts)} active)
----------------------
"""

        if extreme_alerts:
            for alert in extreme_alerts:
                email += f"• {alert.factor_name}: Z-score = {alert.z_score:.2f} ({alert.direction})\n"
        else:
            email += "No extreme value alerts at this time.\n"

        email += f"""

TOP LONG CANDIDATES
-------------------
"""
        for sig in longs:
            email += f"• {sig.ticker}: Score = {sig.composite_score:.4f}, Decile = {sig.decile}\n"

        email += f"""

TOP SHORT CANDIDATES
--------------------
"""
        for sig in shorts:
            email += f"• {sig.ticker}: Score = {sig.composite_score:.4f}, Decile = {sig.decile}\n"

        email += """

---
This is an automated alert from the Equity Factors Trading System.
"""

        return email

    except Exception as e:
        return f"Error generating email alert: {e}"


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "detailed":
        generate_detailed_markdown_report()
    elif len(sys.argv) > 1 and sys.argv[1] == "signals":
        # Generate signal report if data exists
        try:
            returns = pd.read_csv("factor_returns.csv", index_col=0, parse_dates=True)
            loadings = pd.read_csv("factor_loadings.csv", index_col=0)
            generate_signal_report(returns, loadings)
        except FileNotFoundError:
            print(" Data not found. Please run discovery first.")
    else:
        generate_daily_report()
