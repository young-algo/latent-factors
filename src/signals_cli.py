#!/usr/bin/env python3
"""
Trading Signals CLI - Command Line Interface for Signal Generation

⚠️  DEPRECATION WARNING: This CLI is deprecated. Please use the unified CLI instead:
    
    uv run python -m src <command>
    
    Or after pip install:
    equity-factors <command>

Migration Guide:
    OLD: uv run python src/signals_cli.py generate --universe SPY
    NEW: uv run python -m src signals generate --universe SPY
    
    OLD: uv run python src/signals_cli.py extremes --threshold 2.0
    NEW: uv run python -m src signals extremes --threshold 2.0
    
    OLD: uv run python src/signals_cli.py regime --regimes 3
    NEW: uv run python -m src regime detect --regimes 3

The unified CLI provides all the same functionality with better organization.
"""

import argparse
import os
import sys
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from .research import FactorResearchSystem
from .trading_signals import FactorMomentumAnalyzer
from .cross_sectional import CrossSectionalAnalyzer
from .regime_detection import RegimeDetector
from .signal_aggregator import SignalAggregator
from .signal_backtest import SignalBacktester
from .reporting import generate_signal_report, export_signals_to_csv
from .factor_labeler import batch_name_factors, validate_api_key
from .factor_naming import FactorName, generate_quality_report, validate_name


def get_api_key():
    """Get Alpha Vantage API key from environment."""
    api_key = os.environ.get('ALPHAVANTAGE_API_KEY')
    if not api_key:
        print("❌ Error: ALPHAVANTAGE_API_KEY environment variable not set")
        print("   Set it with: export ALPHAVANTAGE_API_KEY='your_key'")
        sys.exit(1)
    return api_key


def load_or_generate_factors(universe, method='fundamental', n_components=8, expand_etfs=True):
    """Load cached factors or generate new ones."""
    cache_file = f"factor_cache_{'_'.join(universe)}_{method}.pkl"

    if Path(cache_file).exists():
        print(f"📂 Loading cached factors from {cache_file}")
        import pickle
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"🔍 Generating factors for universe: {', '.join(universe)}")
    frs = FactorResearchSystem(
        get_api_key(),
        universe=universe,
        factor_method=method,
        n_components=n_components,
        expand_etfs=expand_etfs
    )
    frs.fit_factors()

    # Cache results
    import pickle
    with open(cache_file, 'wb') as f:
        pickle.dump((frs, frs.get_factor_returns(), frs._expos), f)

    return frs, frs.get_factor_returns(), frs._expos


def cmd_generate(args):
    """Generate comprehensive trading signals."""
    print("=" * 70)
    print("📊 GENERATING TRADING SIGNALS")
    print("=" * 70)

    frs, returns, loadings = load_or_generate_factors(
        args.universe,
        method=args.method,
        n_components=args.components,
        expand_etfs=args.expand_etfs
    )

    signals = frs.get_trading_signals()

    print("\n" + signals['summary'])

    if args.output:
        frs.export_signals(args.output)
        print(f"\n💾 Signals exported to: {args.output}")

    return signals


def cmd_extremes(args):
    """Show extreme value alerts."""
    print("=" * 70)
    print("🚨 EXTREME VALUE ALERTS")
    print("=" * 70)

    frs, returns, loadings = load_or_generate_factors(args.universe)
    analyzer = FactorMomentumAnalyzer(returns)

    alerts = analyzer.get_all_extreme_alerts(
        z_threshold=args.threshold,
        percentile_threshold=args.percentile
    )

    if not alerts:
        print("\n✅ No extreme value alerts at this time.")
        return

    print(f"\nFound {len(alerts)} extreme alerts:\n")
    print(f"{'Factor':<15} {'Z-Score':>10} {'Percentile':>12} {'Direction':<15} {'Type':<20}")
    print("-" * 70)

    for alert in alerts:
        direction = "🔴 EXTREME HIGH" if alert.direction == 'extreme_high' else "🟢 EXTREME LOW"
        print(f"{alert.factor_name:<15} {alert.z_score:>10.2f} {alert.percentile:>11.1f}% {direction:<15} {alert.alert_type:<20}")

    if args.trade:
        print("\n💡 Trading Implications:")
        for alert in alerts:
            if alert.direction == 'extreme_high':
                print(f"   → Consider SHORT on {alert.factor_name} (mean reversion)")
            else:
                print(f"   → Consider LONG on {alert.factor_name} (mean reversion)")


def cmd_momentum(args):
    """Analyze factor momentum."""
    print("=" * 70)
    print("📈 FACTOR MOMENTUM ANALYSIS")
    print("=" * 70)

    frs, returns, loadings = load_or_generate_factors(args.universe)
    analyzer = FactorMomentumAnalyzer(returns)

    if args.factor:
        factors = [args.factor]
    else:
        factors = returns.columns.tolist()

    print(f"\nAnalyzing {len(factors)} factor(s):\n")
    print(f"{'Factor':<15} {'RSI':>8} {'RSI Signal':<15} {'MACD':<15} {'ADX':>8} {'Regime':<20}")
    print("-" * 90)

    for factor in factors:
        signals = analyzer.get_momentum_signals(factor)

        rsi_emoji = "🔴" if signals['rsi'] > 70 else "🟢" if signals['rsi'] < 30 else "⚪"
        macd_emoji = "🟢" if 'bullish' in signals['macd_signal'] else "🔴" if 'bearish' in signals['macd_signal'] else "⚪"

        print(f"{factor:<15} {rsi_emoji}{signals['rsi']:>6.1f} {signals['rsi_signal']:<15} "
              f"{macd_emoji} {signals['macd_signal']:<12} {signals['adx']:>7.1f} {signals['regime']:<20}")

    if args.plot:
        print("\n📊 Generating momentum charts...")
        for factor in factors[:3]:  # Plot first 3
            rsi = analyzer.calculate_rsi(factor)
            print(f"   {factor}: Current RSI = {rsi.iloc[-1]:.2f}")


def cmd_cross_section(args):
    """Generate cross-sectional rankings."""
    print("=" * 70)
    print("🔥 CROSS-SECTIONAL RANKINGS")
    print("=" * 70)

    frs, returns, loadings = load_or_generate_factors(args.universe)
    analyzer = CrossSectionalAnalyzer(loadings)

    scores = analyzer.calculate_factor_scores()
    rankings = analyzer.rank_universe(scores)
    signals = analyzer.generate_long_short_signals(
        top_pct=args.top_pct,
        bottom_pct=args.bottom_pct
    )

    longs = [s for s in signals if s.direction.value == 'long']
    shorts = [s for s in signals if s.direction.value == 'short']

    print(f"\n📈 TOP {int(args.top_pct*100)}% LONG CANDIDATES:")
    print("-" * 70)
    print(f"{'Ticker':<10} {'Score':>10} {'Decile':>8} {'Rank':>8} {'Conf':>8}")
    print("-" * 70)
    for sig in sorted(longs, key=lambda x: x.rank)[:args.limit]:
        print(f"{sig.ticker:<10} {sig.composite_score:>10.4f} {sig.decile:>8} {sig.rank:>8} {sig.confidence:>7.1%}")

    print(f"\n📉 TOP {int(args.bottom_pct*100)}% SHORT CANDIDATES:")
    print("-" * 70)
    print(f"{'Ticker':<10} {'Score':>10} {'Decile':>8} {'Rank':>8} {'Conf':>8}")
    print("-" * 70)
    for sig in sorted(shorts, key=lambda x: x.rank, reverse=True)[:args.limit]:
        print(f"{sig.ticker:<10} {sig.composite_score:>10.4f} {sig.decile:>8} {sig.rank:>8} {sig.confidence:>7.1%}")

    if args.output:
        rankings.to_csv(args.output)
        print(f"\n💾 Rankings exported to: {args.output}")


def cmd_regime(args):
    """Detect market regime."""
    print("=" * 70)
    print("🎯 MARKET REGIME DETECTION")
    print("=" * 70)

    frs, returns, loadings = load_or_generate_factors(args.universe)

    print(f"\n🔍 Fitting HMM with {args.regimes} regimes...")
    detector = RegimeDetector(returns)
    detector.fit_hmm(n_regimes=args.regimes)

    current = detector.detect_current_regime()
    allocation = detector.generate_regime_signals()

    print(f"\n📊 CURRENT REGIME:")
    print(f"   Regime: {current.regime.value.replace('_', ' ').title()}")
    print(f"   Confidence: {current.probability:.1%}")
    print(f"   Volatility: {current.volatility:.2%}")
    print(f"   Trend: {current.trend:.4f}")
    print(f"   Description: {current.description}")

    print(f"\n💼 ALLOCATION RECOMMENDATION:")
    print(f"   Risk-on Score: {allocation.risk_on_score:.2f} (0=defensive, 1=aggressive)")
    print(f"   Defensive Tilt: {'Yes' if allocation.defensive_tilt else 'No'}")
    print(f"   Action: {allocation.recommended_action}")

    print(f"\n📈 OPTIMAL FACTOR WEIGHTS:")
    for factor, weight in sorted(allocation.factor_weights.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(weight * 50)
        print(f"   {factor:<15} {weight:>6.2%} {bar}")

    if args.predict:
        print(f"\n🔮 REGIME PREDICTIONS (Next {args.predict} days):")
        predictions = detector.predict_regime(duration=args.predict)
        for i, pred in enumerate(predictions, 1):
            print(f"   Day {i}: {pred.regime.value.replace('_', ' ').title()} (prob: {pred.probability:.1%})")

    if args.summary:
        print("\n📋 REGIME STATISTICS:")
        summary = detector.get_regime_summary()
        print(summary.to_string(index=False))


def cmd_backtest(args):
    """Backtest signal performance."""
    print("=" * 70)
    print("📊 SIGNAL BACKTEST")
    print("=" * 70)

    frs, returns, loadings = load_or_generate_factors(args.universe)

    # Create signal aggregator
    aggregator = SignalAggregator(frs)
    momentum = FactorMomentumAnalyzer(returns)
    cross = CrossSectionalAnalyzer(loadings)
    detector = RegimeDetector(returns)
    detector.fit_hmm(n_regimes=3)

    aggregator.add_momentum_signals(momentum)
    aggregator.add_cross_sectional_signals(cross)
    aggregator.add_regime_signals(detector)

    # Get returns data for backtesting
    returns_data = returns  # Use factor returns as proxy

    print(f"\n🔍 Running walk-forward backtest...")
    print(f"   Train size: {args.train_size} days")
    print(f"   Test size: {args.test_size} days")
    print(f"   Walks: {args.walks}")

    backtester = SignalBacktester(aggregator, returns_data)

    try:
        results = backtester.run_backtest(
            train_size=args.train_size,
            test_size=args.test_size,
            n_walks=args.walks,
            min_confidence=args.confidence
        )

        print(f"\n📈 BACKTEST RESULTS:")
        print(f"   Total Return: {results['total_return']:.2%}")
        print(f"   Annualized Return: {results['annualized_return']:.2%}")
        print(f"   Volatility: {results['volatility']:.2%}")
        print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"   Sortino Ratio: {results['sortino_ratio']:.2f}")
        print(f"   Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"   Hit Rate: {results['hit_rate']:.1%}")
        print(f"   Profit Factor: {results['profit_factor']:.2f}")
        print(f"   Number of Trades: {results['num_trades']}")

        if args.optimize:
            print(f"\n🔧 OPTIMIZING THRESHOLDS...")
            optimal = backtester.optimize_thresholds(
                metric=args.metric,
                n_steps=args.steps
            )
            print(f"   Optimal threshold: {optimal['threshold']:.1f}")
            print(f"   Best {optimal['metric_optimized']}: {optimal['best_metric_value']:.2f}")

        if args.report:
            report = backtester.generate_backtest_report()
            print("\n" + report)

    except Exception as e:
        print(f"❌ Backtest failed: {e}")


def cmd_dashboard(args):
    """Launch Streamlit dashboard."""
    import subprocess

    print("🚀 Launching Streamlit dashboard...")
    print("   URL: http://localhost:8501")
    print("   Press Ctrl+C to stop")
    print("-" * 70)

    dashboard_path = Path(__file__).parent / "dashboard.py"
    subprocess.run(["streamlit", "run", str(dashboard_path)])


def cmd_report(args):
    """Generate HTML report."""
    print("=" * 70)
    print("📄 GENERATING SIGNAL REPORT")
    print("=" * 70)

    frs, returns, loadings = load_or_generate_factors(args.universe)

    output_path = args.output or f"signal_report_{datetime.now().strftime('%Y-%m-%d')}.html"

    generate_signal_report(returns, loadings, output_path)
    print(f"\n💾 Report saved to: {output_path}")

    if args.open:
        import webbrowser
        webbrowser.open(f"file://{Path(output_path).absolute()}")


def cmd_export(args):
    """Export signals to CSV."""
    print("=" * 70)
    print("💾 EXPORTING SIGNALS")
    print("=" * 70)

    frs, returns, loadings = load_or_generate_factors(args.universe)

    output_path = args.output or f"signals_{datetime.now().strftime('%Y-%m-%d')}.csv"

    frs.export_signals(output_path)
    print(f"\n✅ Signals exported to: {output_path}")

    # Also print summary
    signals = frs.get_trading_signals()
    print(f"\n📊 Export Summary:")
    print(f"   Factors: {len(signals['momentum_signals'])}")
    print(f"   Extreme Alerts: {len(signals['extreme_alerts'])}")
    print(f"   Cross-sectional Signals: {len(signals['cross_sectional_signals'])}")


def cmd_name_factors(args):
    """Interactive factor naming command."""
    print("=" * 70)
    print("🏷️  FACTOR NAMING")
    print("=" * 70)

    # Check API key
    if not validate_api_key():
        print("\n❌ Error: OpenAI API key not available or invalid")
        print("   Set OPENAI_API_KEY environment variable")
        sys.exit(1)

    # Load or generate factors
    print(f"\n🔍 Loading factor data for universe: {', '.join(args.universe)}")
    frs, returns, loadings = load_or_generate_factors(
        args.universe,
        method=args.method,
        n_components=args.components,
        expand_etfs=args.expand_etfs
    )

    # Get fundamental data for context
    fundamentals = frs.get_fundamentals() if hasattr(frs, 'get_fundamentals') else pd.DataFrame()

    # Load existing names if available
    existing_names = {}
    cache_file = Path("factor_names_cache.json")
    if cache_file.exists() and not args.regenerate:
        import json
        with open(cache_file, 'r') as f:
            data = json.load(f)
            for code, name_data in data.get('factors', {}).items():
                existing_names[code] = FactorName.from_dict(name_data)
        print(f"\n📂 Loaded {len(existing_names)} cached names")

    # Determine which factors to name
    if args.factor:
        factors_to_name = [args.factor]
    else:
        factors_to_name = [c for c in loadings.columns if c not in existing_names or args.regenerate]

    print(f"\n🎯 Naming {len(factors_to_name)} factors...")

    # Generate names
    from .factor_labeler import batch_name_factors
    new_names = batch_name_factors(
        loadings[factors_to_name],
        fundamentals,
        top_n=args.top_n,
        model=args.model
    )

    # Merge with existing
    all_names = {**existing_names, **new_names}

    # Interactive review mode
    if args.interactive or args.review:
        print("\n" + "=" * 70)
        print("📋 REVIEW GENERATED NAMES")
        print("=" * 70)

        for code, name in sorted(all_names.items()):
            if not args.factor and code not in new_names:
                continue  # Skip already-named factors unless reviewing all

            print(f"\n{code}: \"{name.short_name}\"")
            print(f"   Description: {name.description}")
            print(f"   Quality: {name.quality_score:.0f}/100 | Confidence: {name.confidence}")

            # Show top stocks for context
            if code in loadings.columns:
                top_stocks = loadings[code].nlargest(5)
                print(f"   Top stocks: {', '.join(top_stocks.index)}")

            issues = validate_name(name)
            if issues:
                print(f"   ⚠ Issues: {', '.join(issues)}")

            if args.interactive:
                action = input("\n   [a]pprove, [e]dit, [s]kip, [q]uit? ").lower().strip()

                if action == 'q':
                    break
                elif action == 'e':
                    new_name = input("   New short name: ").strip()
                    new_desc = input("   New description: ").strip()
                    if new_name:
                        name.short_name = new_name
                        name.edited = True
                        name.edited_at = datetime.now().isoformat()
                    if new_desc:
                        name.description = new_desc
                        name.edited = True
                        name.edited_at = datetime.now().isoformat()
                    action = 'a'  # Auto-approve after edit

                if action == 'a':
                    name.approved = True
                    print("   ✓ Approved")

    # Generate quality report
    print("\n" + generate_quality_report(all_names))

    # Save to cache
    cache_data = {
        'cache_version': '2.0',
        'generated_at': datetime.now().isoformat(),
        'factors': {code: name.to_dict() for code, name in all_names.items()}
    }

    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)
    print(f"\n💾 Saved {len(all_names)} names to {cache_file}")

    # Export to CSV for easy viewing
    csv_file = args.output or "factor_names.csv"
    df_data = []
    for code, name in sorted(all_names.items()):
        df_data.append({
            'factor_code': code,
            'short_name': name.short_name,
            'description': name.description,
            'theme': name.theme,
            'confidence': name.confidence,
            'quality_score': name.quality_score,
            'approved': name.approved,
            'edited': name.edited
        })

    pd.DataFrame(df_data).to_csv(csv_file, index=False)
    print(f"💾 Exported to {csv_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Trading Signals CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s generate --universe SPY QQQ
    %(prog)s extremes --threshold 2.5
    %(prog)s cross-section --top-pct 0.05 --limit 20
    %(prog)s regime --predict 5
    %(prog)s backtest --walks 10 --optimize
    %(prog)s dashboard
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate comprehensive signals')
    gen_parser.add_argument('--universe', nargs='+', default=['SPY'], help='Stock/ETF universe')
    gen_parser.add_argument('--method', default='fundamental', choices=['fundamental', 'pca', 'ica'], help='Factor method')
    gen_parser.add_argument('--components', type=int, default=8, help='Number of factors')
    gen_parser.add_argument('--no-expand', dest='expand_etfs', action='store_false', help='Disable ETF expansion')
    gen_parser.add_argument('-o', '--output', help='Output CSV file')
    gen_parser.set_defaults(expand_etfs=True)

    # Extremes command
    extremes_parser = subparsers.add_parser('extremes', help='Show extreme value alerts')
    extremes_parser.add_argument('--universe', nargs='+', default=['SPY'])
    extremes_parser.add_argument('-t', '--threshold', type=float, default=2.0, help='Z-score threshold')
    extremes_parser.add_argument('-p', '--percentile', type=float, default=95.0, help='Percentile threshold')
    extremes_parser.add_argument('--trade', action='store_true', help='Show trading implications')

    # Momentum command
    momentum_parser = subparsers.add_parser('momentum', help='Analyze factor momentum')
    momentum_parser.add_argument('--universe', nargs='+', default=['SPY'])
    momentum_parser.add_argument('-f', '--factor', help='Specific factor to analyze')
    momentum_parser.add_argument('--plot', action='store_true', help='Generate charts')

    # Cross-section command
    cs_parser = subparsers.add_parser('cross-section', help='Cross-sectional rankings')
    cs_parser.add_argument('--universe', nargs='+', default=['SPY'])
    cs_parser.add_argument('--top-pct', type=float, default=0.1, help='Top percentile')
    cs_parser.add_argument('--bottom-pct', type=float, default=0.1, help='Bottom percentile')
    cs_parser.add_argument('-l', '--limit', type=int, default=10, help='Number of stocks to show')
    cs_parser.add_argument('-o', '--output', help='Output CSV file')

    # Regime command
    regime_parser = subparsers.add_parser('regime', help='Detect market regime')
    regime_parser.add_argument('--universe', nargs='+', default=['SPY'])
    regime_parser.add_argument('-r', '--regimes', type=int, default=3, help='Number of regimes')
    regime_parser.add_argument('-p', '--predict', type=int, help='Predict N days ahead')
    regime_parser.add_argument('--summary', action='store_true', help='Show regime statistics')

    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Backtest signals')
    backtest_parser.add_argument('--universe', nargs='+', default=['SPY'])
    backtest_parser.add_argument('--train-size', type=int, default=252, help='Training window')
    backtest_parser.add_argument('--test-size', type=int, default=63, help='Test window')
    backtest_parser.add_argument('--walks', type=int, default=5, help='Number of walks')
    backtest_parser.add_argument('-c', '--confidence', type=float, default=70.0, help='Min confidence')
    backtest_parser.add_argument('--optimize', action='store_true', help='Optimize thresholds')
    backtest_parser.add_argument('--metric', default='sharpe_ratio', help='Metric to optimize')
    backtest_parser.add_argument('--steps', type=int, default=10, help='Optimization steps')
    backtest_parser.add_argument('--report', action='store_true', help='Show full report')

    # Dashboard command
    subparsers.add_parser('dashboard', help='Launch Streamlit dashboard')

    # Report command
    report_parser = subparsers.add_parser('report', help='Generate HTML report')
    report_parser.add_argument('--universe', nargs='+', default=['SPY'])
    report_parser.add_argument('-o', '--output', help='Output HTML file')
    report_parser.add_argument('--open', action='store_true', help='Open in browser')

    # Export command
    export_parser = subparsers.add_parser('export', help='Export signals to CSV')
    export_parser.add_argument('--universe', nargs='+', default=['SPY'])
    export_parser.add_argument('-o', '--output', help='Output CSV file')

    # Name-factors command
    name_parser = subparsers.add_parser('name-factors', help='Generate factor names with LLM')
    name_parser.add_argument('--universe', nargs='+', default=['SPY'], help='Stock/ETF universe')
    name_parser.add_argument('--method', default='fundamental', choices=['fundamental', 'pca', 'ica'], help='Factor method')
    name_parser.add_argument('--components', type=int, default=8, help='Number of factors')
    name_parser.add_argument('--no-expand', dest='expand_etfs', action='store_false', help='Disable ETF expansion')
    name_parser.add_argument('-f', '--factor', help='Name specific factor only')
    name_parser.add_argument('--model', default='gpt-5-mini', help='OpenAI model to use')
    name_parser.add_argument('--top-n', type=int, default=10, help='Number of stocks to show')
    name_parser.add_argument('--interactive', '-i', action='store_true', help='Interactive approval mode')
    name_parser.add_argument('--review', '-r', action='store_true', help='Review all names')
    name_parser.add_argument('--regenerate', action='store_true', help='Regenerate all names')
    name_parser.add_argument('-o', '--output', help='Output CSV file')
    name_parser.set_defaults(expand_etfs=True)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Dispatch to command handler
    commands = {
        'generate': cmd_generate,
        'extremes': cmd_extremes,
        'momentum': cmd_momentum,
        'cross-section': cmd_cross_section,
        'regime': cmd_regime,
        'backtest': cmd_backtest,
        'dashboard': cmd_dashboard,
        'report': cmd_report,
        'export': cmd_export,
        'name-factors': cmd_name_factors,
    }

    command_func = commands.get(args.command)
    if command_func:
        command_func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
