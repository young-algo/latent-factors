#!/usr/bin/env python3
"""
Unified CLI for Equity Factors Research System
==============================================

This module provides a single entry point for all project functionality.
It consolidates the previously separate CLIs into one unified interface.

Usage
-----
    # Factor discovery
    uv run python -m src discover --symbols SPY --method PCA -k 10
    
    # Trading signals
    uv run python -m src signals generate --universe SPY
    uv run python -m src signals extremes --universe SPY
    uv run python -m src signals momentum --universe SPY
    uv run python -m src signals cross-section --universe SPY
    
    # Regime detection
    uv run python -m src regime detect --universe SPY
    
    # Backtesting
    uv run python -m src backtest --universe SPY --walks 10
    
    # Dashboard
    uv run python -m src dashboard
    
    # Reports
    uv run python -m src report --type html
    
    # Maintenance
    uv run python -m src clean --all

For help on any command:
    uv run python -m src <command> --help
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path
from datetime import datetime

import pandas as pd

# Import centralized config first to ensure environment is set up
from src.config import config, validate_config

# Import module functions
from src.discover_and_label import run_discovery
from src.reporting import generate_daily_report, generate_detailed_markdown_report


def get_api_key():
    """Get and validate Alpha Vantage API key."""
    try:
        validate_config(require_alpha_vantage=True)
        return config.ALPHAVANTAGE_API_KEY
    except ValueError as e:
        print(f" Configuration Error: {e}")
        sys.exit(1)


# =============================================================================
# Factor Discovery Commands
# =============================================================================

def cmd_discover(args):
    """Run the factor discovery workflow."""
    print(f" Starting factor discovery for {args.symbols}...")
    try:
        run_discovery(
            symbols=args.symbols,
            start_date=args.start,
            method=args.method,
            k=args.k,
            rolling=args.rolling,
            name_out=args.name_out
        )
        print("\n Factor discovery complete!")
    except Exception as e:
        print(f"\n Error during discovery: {e}")
        sys.exit(1)


# =============================================================================
# Trading Signals Commands
# =============================================================================

def load_or_generate_factors(universe, method='fundamental', n_components=8, expand_etfs=True):
    """Load cached factors or generate new ones."""
    cache_file = f"factor_cache_{'_'.join(universe)}_{method}.pkl"
    
    if Path(cache_file).exists() and not os.getenv('EQUITY_FACTORS_REFRESH_CACHE'):
        print(f" Loading cached factors from {cache_file}")
        import pickle
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print(f" Generating factors for universe: {', '.join(universe)}")
    from src.research import FactorResearchSystem
    
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


def cmd_signals_generate(args):
    """Generate comprehensive trading signals."""
    print("=" * 70)
    print(" GENERATING TRADING SIGNALS")
    print("=" * 70)
    
    frs, returns, loadings = load_or_generate_factors(
        args.universe,
        method=args.method,
        n_components=args.components,
        expand_etfs=args.expand_etfs
    )
    
    signals = frs.get_trading_signals()
    print("\n" + signals.get('summary', 'No summary available'))
    
    if args.output:
        frs.export_signals(args.output)
        print(f"\n Signals exported to: {args.output}")


def cmd_signals_extremes(args):
    """Show extreme value alerts."""
    print("=" * 70)
    print(" EXTREME VALUE ALERTS")
    print("=" * 70)
    
    from src.trading_signals import FactorMomentumAnalyzer
    
    frs, returns, loadings = load_or_generate_factors(args.universe)
    analyzer = FactorMomentumAnalyzer(returns)
    
    alerts = analyzer.get_all_extreme_alerts(
        z_threshold=args.threshold,
        percentile_threshold=args.percentile
    )
    
    if not alerts:
        print("\n No extreme value alerts at this time.")
        return
    
    print(f"\nFound {len(alerts)} extreme alerts:\n")
    print(f"{'Factor':<15} {'Z-Score':>10} {'Percentile':>12} {'Direction':<15} {'Type':<20}")
    print("-" * 70)
    
    for alert in alerts:
        direction = " EXTREME HIGH" if alert.direction == 'extreme_high' else "🟢 EXTREME LOW"
        print(f"{alert.factor_name:<15} {alert.z_score:>10.2f} {alert.percentile:>11.1f}% {direction:<15} {alert.alert_type:<20}")
    
    if args.trade:
        print("\n Trading Implications:")
        for alert in alerts:
            if alert.direction == 'extreme_high':
                print(f"   → Consider SHORT on {alert.factor_name} (mean reversion)")
            else:
                print(f"   → Consider LONG on {alert.factor_name} (mean reversion)")


def cmd_signals_momentum(args):
    """Analyze factor momentum."""
    print("=" * 70)
    print(" FACTOR MOMENTUM ANALYSIS")
    print("=" * 70)
    
    from src.trading_signals import FactorMomentumAnalyzer
    
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
        
        rsi_emoji = "" if signals['rsi'] > 70 else "🟢" if signals['rsi'] < 30 else ""
        macd_emoji = "🟢" if 'bullish' in signals['macd_signal'] else "" if 'bearish' in signals['macd_signal'] else ""
        
        print(f"{factor:<15} {rsi_emoji}{signals['rsi']:>6.1f} {signals['rsi_signal']:<15} "
              f"{macd_emoji} {signals['macd_signal']:<12} {signals['adx']:>7.1f} {signals['regime']:<20}")


def cmd_signals_cross_section(args):
    """Generate cross-sectional rankings."""
    print("=" * 70)
    print(" CROSS-SECTIONAL RANKINGS")
    print("=" * 70)
    
    from src.cross_sectional import CrossSectionalAnalyzer
    
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
    
    print(f"\n TOP {int(args.top_pct*100)}% LONG CANDIDATES:")
    print("-" * 70)
    print(f"{'Ticker':<10} {'Score':>10} {'Decile':>8} {'Rank':>8} {'Conf':>8}")
    print("-" * 70)
    for sig in sorted(longs, key=lambda x: x.rank)[:args.limit]:
        print(f"{sig.ticker:<10} {sig.composite_score:>10.4f} {sig.decile:>8} {sig.rank:>8} {sig.confidence:>7.1%}")
    
    print(f"\n TOP {int(args.bottom_pct*100)}% SHORT CANDIDATES:")
    print("-" * 70)
    print(f"{'Ticker':<10} {'Score':>10} {'Decile':>8} {'Rank':>8} {'Conf':>8}")
    print("-" * 70)
    for sig in sorted(shorts, key=lambda x: x.rank, reverse=True)[:args.limit]:
        print(f"{sig.ticker:<10} {sig.composite_score:>10.4f} {sig.decile:>8} {sig.rank:>8} {sig.confidence:>7.1%}")
    
    if args.output:
        rankings.to_csv(args.output)
        print(f"\n Rankings exported to: {args.output}")


# =============================================================================
# Regime Detection Commands
# =============================================================================

def cmd_regime_detect(args):
    """Detect market regime."""
    print("=" * 70)
    print(" MARKET REGIME DETECTION")
    print("=" * 70)

    from src.regime_detection import RegimeDetector
    from src.factor_labeler import batch_name_factors, validate_api_key
    from src.config import config
    import json

    frs, returns, loadings = load_or_generate_factors(args.universe)

    # Load or generate factor names
    names_cache_file = f"factor_names_{'_'.join(args.universe)}_fundamental.json"
    factor_names = {}

    if Path(names_cache_file).exists():
        with open(names_cache_file, 'r') as f:
            factor_names = json.load(f)

    factors_needing_names = [f for f in loadings.columns if f not in factor_names]
    if factors_needing_names and validate_api_key() and frs is not None:
        print(f"\n  Generating names for {len(factors_needing_names)} factors...")
        tickers = list(loadings.index)
        fundamentals = frs.get_fundamentals(tickers) if hasattr(frs, 'get_fundamentals') else pd.DataFrame()
        new_names = batch_name_factors(
            factor_exposures=loadings[factors_needing_names],
            fundamentals=fundamentals,
            factor_returns=returns[factors_needing_names] if not returns.empty else None,
            top_n=10,
            model=config.OPENAI_MODEL
        )
        for factor_id, factor_name_obj in new_names.items():
            factor_names[factor_id] = factor_name_obj.short_name
        with open(names_cache_file, 'w') as f:
            json.dump(factor_names, f, indent=2)

    print(f"\n Fitting HMM with {args.regimes} regimes...")
    detector = RegimeDetector(returns)
    detector.fit_hmm(n_regimes=args.regimes)

    current = detector.detect_current_regime()
    allocation = detector.generate_regime_signals()

    print(f"\n CURRENT REGIME:")
    print(f"   Regime: {current.regime.value.replace('_', ' ').title()}")
    print(f"   Confidence: {current.probability:.1%}")
    print(f"   Volatility: {current.volatility:.2%}")
    print(f"   Trend: {current.trend:.4f}")
    print(f"   Description: {current.description}")

    print(f"\n ALLOCATION RECOMMENDATION:")
    print(f"   Risk-on Score: {allocation.risk_on_score:.2f} (0=defensive, 1=aggressive)")
    print(f"   Defensive Tilt: {'Yes' if allocation.defensive_tilt else 'No'}")
    print(f"   Action: {allocation.recommended_action}")

    print(f"\n OPTIMAL FACTOR WEIGHTS:")
    for factor, weight in sorted(allocation.factor_weights.items(), key=lambda x: x[1], reverse=True):
        bar = "" * int(weight * 50)
        display_name = factor_names.get(factor, factor)
        print(f"   {display_name:<30} {weight:>6.2%} {bar}")
    
    if args.predict:
        print(f"\n REGIME PREDICTIONS (Next {args.predict} days):")
        predictions = detector.predict_regime(duration=args.predict)
        for i, pred in enumerate(predictions, 1):
            print(f"   Day {i}: {pred.regime.value.replace('_', ' ').title()} (prob: {pred.probability:.1%})")
    
    if args.summary:
        print("\n REGIME STATISTICS:")
        summary = detector.get_regime_summary()
        print(summary.to_string(index=False))


# =============================================================================
# Backtest Commands
# =============================================================================

def cmd_backtest(args):
    """Backtest signal performance."""
    print("=" * 70)
    print(" SIGNAL BACKTEST")
    print("=" * 70)
    
    from src.trading_signals import FactorMomentumAnalyzer
    from src.cross_sectional import CrossSectionalAnalyzer
    from src.regime_detection import RegimeDetector
    from src.signal_aggregator import SignalAggregator
    from src.signal_backtest import SignalBacktester
    
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
    returns_data = returns
    
    print(f"\n Running walk-forward backtest...")
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
        
        print(f"\n BACKTEST RESULTS:")
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
            print(f"\n OPTIMIZING THRESHOLDS...")
            optimal = backtester.optimize_thresholds(
                metric=args.metric,
                n_steps=args.steps
            )
            print(f"   Optimal threshold: {optimal['threshold']:.1f}")
            print(f"   Best {optimal['metric_optimized']}: {optimal['best_metric_value']:.2f}")
        
        if args.report:
            report = backtester.generate_backtest_report()
            print("\n" + report)
            
        if args.output:
            backtester.export_results_to_csv(args.output)
            print(f"\n Results exported to: {args.output}")
    
    except Exception as e:
        print(f" Backtest failed: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# Dashboard Commands
# =============================================================================

def cmd_dashboard(args):
    """Launch Streamlit dashboard."""
    dashboard_path = Path(__file__).parent / "dashboard.py"
    if not dashboard_path.exists():
        print("  Dashboard file not found.")
        sys.exit(1)
    
    print(" Launching Streamlit dashboard...")
    print("   URL: http://localhost:8501")
    print("   Press Ctrl+C to stop")
    print("-" * 70)
    
    try:
        # Set PYTHONPATH to include the project root so imports work correctly
        env = os.environ.copy()
        project_root = str(Path(__file__).parent.parent)
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = project_root + os.pathsep + env['PYTHONPATH']
        else:
            env['PYTHONPATH'] = project_root
        
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
            check=True,
            env=env
        )
    except FileNotFoundError:
        print(" Streamlit not found. Install with: uv add streamlit")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n Dashboard stopped.")


# =============================================================================
# Report Commands
# =============================================================================

def cmd_report(args):
    """Generate reports."""
    output_path = args.out or f"report_{datetime.now().strftime('%Y-%m-%d')}"
    
    if args.type == "html":
        output_path = output_path if output_path.endswith('.html') else f"{output_path}.html"
        print(f" Generating HTML report: {output_path}")
        generate_daily_report(output_path)
    else:
        output_path = output_path if output_path.endswith('.md') else f"{output_path}.md"
        print(f" Generating Markdown report: {output_path}")
        generate_detailed_markdown_report(output_path)
    
    print(f"\n Report saved to: {output_path}")
    
    if args.open:
        import webbrowser
        webbrowser.open(f"file://{Path(output_path).absolute()}")


# =============================================================================
# Briefing Commands
# =============================================================================

def cmd_briefing(args):
    """Generate morning briefing with actionable recommendations."""
    from src.decision_synthesizer import DecisionSynthesizer
    from src.research import FactorResearchSystem
    from src.factor_labeler import batch_name_factors, validate_api_key
    from src.config import config
    import pickle
    from pathlib import Path
    import json

    print("" * 70)
    print(" GENERATING MORNING BRIEFING")
    print("" * 70)

    # Load or generate factors
    cache_file = f"factor_cache_{'_'.join(args.universe)}_{args.method}.pkl"
    names_cache_file = f"factor_names_{'_'.join(args.universe)}_{args.method}.json"

    # We need FRS for fundamentals even if factors are cached
    api_key = get_api_key()
    frs = FactorResearchSystem(
        api_key,
        universe=args.universe,
        factor_method=args.method,
        n_components=args.components,
        expand_etfs=True
    )

    if Path(cache_file).exists():
        print(f"\n Loading cached factors from {cache_file}")
        with open(cache_file, 'rb') as f:
            factor_returns, factor_loadings = pickle.load(f)
    else:
        print(f"\n Generating factors for {', '.join(args.universe)}...")
        frs.fit_factors()
        factor_returns = frs.get_factor_returns()
        factor_loadings = frs._expos

        # Cache for next time
        with open(cache_file, 'wb') as f:
            pickle.dump((factor_returns, factor_loadings), f)

    # Generate LLM factor names (always - factors should always have meaningful names)
    factor_names = {}

    # Try to load cached names first
    if Path(names_cache_file).exists():
        print(f"\n Loading cached factor names from {names_cache_file}")
        with open(names_cache_file, 'r') as f:
            factor_names = json.load(f)

    # Generate names for any factors that don't have names yet
    factors_needing_names = [f for f in factor_loadings.columns if f not in factor_names]

    if factors_needing_names and validate_api_key():
        print(f"\n  Generating names for {len(factors_needing_names)} factors...")

        # Get fundamentals for naming context (pass tickers from loadings index)
        tickers = list(factor_loadings.index)
        fundamentals = frs.get_fundamentals(tickers) if hasattr(frs, 'get_fundamentals') else pd.DataFrame()

        # Generate names with full enrichment
        new_names = batch_name_factors(
            factor_exposures=factor_loadings[factors_needing_names],
            fundamentals=fundamentals,
            factor_returns=factor_returns[factors_needing_names] if not factor_returns.empty else None,
            top_n=10,
            model=config.OPENAI_MODEL
        )

        # Extract short names and merge
        for factor_id, factor_name_obj in new_names.items():
            factor_names[factor_id] = factor_name_obj.short_name

        # Cache names for next time
        with open(names_cache_file, 'w') as f:
            json.dump(factor_names, f, indent=2)
        print(f"    Cached names to {names_cache_file}")
    elif factors_needing_names:
        print("\n  OpenAI API key not available - using factor IDs as names")
        for f in factors_needing_names:
            factor_names[f] = f

    # Generate briefing
    print("\n Analyzing signals...")
    synthesizer = DecisionSynthesizer()

    state = synthesizer.collect_all_signals(
        factor_returns=factor_returns,
        factor_loadings=factor_loadings,
        factor_names=factor_names
    )

    recommendations = synthesizer.generate_recommendations(state)
    briefing = synthesizer.render_briefing(state, recommendations)

    # Output
    print("\n")
    print(briefing)

    # Save if requested
    if args.output:
        output_path = args.output
        if args.format == "markdown":
            output_path = output_path if output_path.endswith('.md') else f"{output_path}.md"
        else:
            output_path = output_path if output_path.endswith('.txt') else f"{output_path}.txt"

        with open(output_path, 'w') as f:
            f.write(briefing)
        print(f"\n Briefing saved to: {output_path}")


# =============================================================================
# Maintenance Commands
# =============================================================================

def cmd_clean(args):
    """Clean cache and temporary files."""
    if args.all:
        db_path = Path("av_cache.db")
        if db_path.exists():
            db_path.unlink()
            print("  Deleted av_cache.db")
        else:
            print("  Cache file not found.")
        
        # Also clean factor cache files
        import glob
        cache_files = glob.glob("factor_cache_*.pkl")
        for f in cache_files:
            Path(f).unlink()
            print(f"  Deleted {f}")
        
        if not cache_files:
            print("No factor cache files found.")
    else:
        print("Use --all to confirm deletion of cache files.")
        print("This will delete:")
        print("  - av_cache.db (Alpha Vantage API cache)")
        print("  - factor_cache_*.pkl (Factor computation cache)")


def cmd_optimize(args):
    """Optimize factor weights for maximum Sharpe ratio."""
    print("=" * 70)
    print(" FACTOR WEIGHT OPTIMIZATION")
    print("=" * 70)

    from src.factor_optimization import SharpeOptimizer
    from src.research import FactorResearchSystem
    from src.factor_labeler import batch_name_factors, validate_api_key
    from src.config import config
    import json
    import pickle

    # Load or generate factors
    print(f"\n Loading factor data for universe: {', '.join(args.universe)}")

    cache_file = f"factor_cache_{'_'.join(args.universe)}_{args.factor_method}.pkl"
    names_cache_file = f"factor_names_{'_'.join(args.universe)}_{args.factor_method}.json"

    # Always create FRS for fundamentals (needed for naming)
    frs = FactorResearchSystem(
        get_api_key(),
        universe=args.universe,
        factor_method=args.factor_method,
        n_components=args.n_components,
        expand_etfs=True
    )

    if Path(cache_file).exists():
        print(f" Loading cached factors from {cache_file}")
        with open(cache_file, 'rb') as f:
            factor_returns, factor_loadings = pickle.load(f)
    else:
        print(f" Generating factors using '{args.factor_method}' method (this may take a while)...")
        frs.fit_factors()
        factor_returns = frs.get_factor_returns()
        factor_loadings = frs._expos

        # Cache factor data
        with open(cache_file, 'wb') as f:
            pickle.dump((factor_returns, factor_loadings), f)

    print(f"   Factor returns shape: {factor_returns.shape}")
    print(f"   Factor loadings shape: {factor_loadings.shape}")

    # Initialize optimizer
    optimizer = SharpeOptimizer(factor_returns, factor_loadings)

    # Always generate factor names (factors should always have meaningful names)
    factor_names = {}

    # Try to load cached names first
    if Path(names_cache_file).exists():
        print(f"\n Loading cached factor names from {names_cache_file}")
        with open(names_cache_file, 'r') as f:
            factor_names = json.load(f)

    # Generate names for any factors that don't have names yet
    factors_needing_names = [f for f in factor_loadings.columns if f not in factor_names]

    if factors_needing_names and validate_api_key():
        print(f"\n  Generating names for {len(factors_needing_names)} factors...")

        # Get fundamentals for naming context (pass tickers from loadings index)
        tickers = list(factor_loadings.index)
        fundamentals = frs.get_fundamentals(tickers) if hasattr(frs, 'get_fundamentals') else pd.DataFrame()

        # Generate names with full enrichment
        new_names = batch_name_factors(
            factor_exposures=factor_loadings[factors_needing_names],
            fundamentals=fundamentals,
            factor_returns=factor_returns[factors_needing_names] if not factor_returns.empty else None,
            top_n=10,
            model=config.OPENAI_MODEL
        )

        # Extract short names and merge
        for factor_id, factor_name_obj in new_names.items():
            factor_names[factor_id] = factor_name_obj.short_name

        # Cache names
        with open(names_cache_file, 'w') as f:
            json.dump(factor_names, f, indent=2)
        print(f"    Cached names to {names_cache_file}")

        # Also save to CSV if requested
        if args.name_factors:
            names_df = pd.DataFrame({
                'factor': list(factor_names.keys()),
                'name': list(factor_names.values())
            })
            names_df.to_csv(args.factor_names_output, index=False)
            print(f"    Factor names saved to: {args.factor_names_output}")
    elif factors_needing_names:
        print("\n  OpenAI API key not available - using factor IDs as names")
        for f in factors_needing_names:
            factor_names[f] = f

    # Display factor names
    if factor_names:
        print("\n Factor Names:")
        for factor, name in sorted(factor_names.items()):
            print(f"   {factor}: {name}")
    
    if args.walk_forward:
        print(f"\n Running walk-forward optimization...")
        print(f"   Training window: {args.train_window} days")
        print(f"   Test window: {args.test_window} days")
        print(f"   Methods: {', '.join(args.methods)}")
        print(f"   Technique: {args.technique}")
        
        results = optimizer.walk_forward_optimize(
            train_window=args.train_window,
            test_window=args.test_window,
            methods=args.methods,
            technique=args.technique,
            verbose=True
        )
        
        print(f"\n Walk-Forward Results Summary:")
        print(f"   Number of periods: {len(results)}")
        print(f"   Average train Sharpe: {results['train_sharpe'].mean():.2f}")
        print(f"   Average test Sharpe: {results['test_sharpe'].mean():.2f}")
        
        # Save results
        if args.output:
            # Add factor names to each period's results if available
            if factor_names:
                results_dict = results.to_dict('records')
                for period in results_dict:
                    period['factor_names'] = factor_names
                import json
                with open(args.output, 'w') as f:
                    json.dump(results_dict, f, indent=2, default=str)
            else:
                results.to_json(args.output, orient='records', date_format='iso')
            print(f"\n Results saved to: {args.output}")
    
    else:
        print(f"\n  Running single-period optimization...")
        print(f"   Lookback: {args.lookback} days")
        print(f"   Methods: {', '.join(args.methods)}")
        print(f"   Technique: {args.technique}")
        
        result = optimizer.optimize_blend(
            lookback=args.lookback,
            methods=args.methods,
            technique=args.technique
        )
        
        print("\n" + "=" * 70)
        print("OPTIMIZATION RESULTS")
        print("=" * 70)
        
        print(f"\n Performance Metrics:")
        print(f"   Sharpe Ratio:         {result.sharpe_ratio:.2f}")
        print(f"   Annualized Return:    {result.annualized_return:.2%}")
        print(f"   Annualized Volatility:{result.annualized_volatility:.2%}")
        
        print(f"\n Optimal Method Blend:")
        for method, weight in sorted(result.method_allocation.items(), key=lambda x: -x[1]):
            if weight > 0.01:
                bar = "" * int(weight * 50)
                print(f"   {method:<20} {weight:>6.1%} {bar}")
        
        print(f"\n Optimal Factor Weights:")
        for factor, weight in sorted(result.optimal_weights.items(), key=lambda x: -x[1]):
            bar = "" * int(weight * 50)
            display_name = factor_names.get(factor, factor)
            print(f"   {display_name:<30} {weight:>6.1%} {bar}")
        
        # Export weights
        if args.export_weights:
            weights_df = pd.DataFrame({
                'factor': list(result.optimal_weights.keys()),
                'weight': list(result.optimal_weights.values())
            })
            # Add factor names if available
            if factor_names:
                weights_df['factor_name'] = weights_df['factor'].map(factor_names)
            weights_df.to_csv(args.export_weights, index=False)
            print(f"\n Weights exported to: {args.export_weights}")
        
        # Save full results
        if args.output:
            output_data = {
                'optimal_weights': result.optimal_weights,
                'method_allocation': result.method_allocation,
                'sharpe_ratio': result.sharpe_ratio,
                'annualized_return': result.annualized_return,
                'annualized_volatility': result.annualized_volatility,
                'lookback': args.lookback,
                'methods': args.methods,
                'technique': args.technique
            }
            # Add factor names if available
            if factor_names:
                output_data['factor_names'] = factor_names
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f" Full results saved to: {args.output}")


def cmd_basket(args):
    """Generate tradeable stock basket from optimization results."""
    import json
    import pickle

    print("=" * 70)
    print(" GENERATING TRADEABLE BASKET")
    print("=" * 70)

    # Load optimization results
    print(f"\n Loading optimization results from: {args.results}")
    with open(args.results, 'r') as f:
        opt_results = json.load(f)

    # Load factor loadings from cache
    cache_file = f"factor_cache_{'_'.join(args.universe)}_{args.factor_method}.pkl"
    if not Path(cache_file).exists():
        print(f" Cache file not found: {cache_file}")
        print("   Run optimization first to generate factor loadings.")
        sys.exit(1)

    print(f" Loading factor loadings from: {cache_file}")
    with open(cache_file, 'rb') as f:
        _, factor_loadings = pickle.load(f)

    # Load factor names from cache or results
    names_cache_file = f"factor_names_{'_'.join(args.universe)}_{args.factor_method}.json"
    factor_names = {}

    # Try to get names from optimization results first
    if isinstance(opt_results, dict) and 'factor_names' in opt_results:
        factor_names = opt_results['factor_names']
    # Otherwise try the cache file
    elif Path(names_cache_file).exists():
        with open(names_cache_file, 'r') as f:
            factor_names = json.load(f)

    # Get factor weights
    if isinstance(opt_results, list):
        # Walk-forward results - average factor weights across all periods
        print(" Averaging factor weights across all walk-forward periods...")
        import pandas as pd
        all_weights = [period['factor_weights'] for period in opt_results]
        weights_df = pd.DataFrame(all_weights)
        factor_weights = weights_df.mean().to_dict()
        print(f"   Averaged {len(opt_results)} periods")
    elif 'optimal_weights' in opt_results:
        factor_weights = opt_results['optimal_weights']
    else:
        print(" No factor weights found in results file.")
        print("   Expected 'optimal_weights' key or a list of walk-forward results.")
        sys.exit(1)

    print(f"\n Optimal Factor Weights:")
    for f, w in sorted(factor_weights.items(), key=lambda x: -x[1]):
        display_name = factor_names.get(f, f)
        print(f"   {display_name:<30}: {w:.2%}")
    
    # Calculate composite stock scores
    print("\n Calculating target stock exposures...")
    composite_score = pd.Series(0.0, index=factor_loadings.index)
    
    for factor, weight in factor_weights.items():
        if factor in factor_loadings.columns:
            composite_score += factor_loadings[factor] * weight
    
    composite_score = composite_score.sort_values(ascending=False)
    n_stocks = len(composite_score)
    
    # Select longs and shorts
    n_long = int(n_stocks * args.long_pct)
    n_short = int(n_stocks * args.short_pct)
    
    longs = composite_score.head(n_long)
    shorts = composite_score.tail(n_short)
    
    print(f"\n Basket Composition:")
    print(f"   Universe: {n_stocks} stocks")
    print(f"   Longs: {n_long} stocks (top {args.long_pct:.0%})")
    print(f"   Shorts: {n_short} stocks (bottom {args.short_pct:.0%})")
    
    # Calculate position weights
    longs_weighted = longs / longs.sum() if longs.sum() > 0 else longs
    shorts_weighted = shorts / shorts.sum() * -1 if shorts.sum() != 0 else shorts
    
    # Create positions dataframe
    positions = pd.DataFrame({
        'ticker': list(longs.index) + list(shorts.index),
        'composite_score': list(longs.values) + list(shorts.values),
        'target_weight': list(longs_weighted.values * args.net_exposure) + 
                        list(shorts_weighted.values * args.net_exposure * -1),
        'side': ['LONG'] * n_long + ['SHORT'] * n_short
    })
    
    # Calculate dollar positions
    positions['position_dollars'] = positions['target_weight'] * args.capital
    
    # Display results
    print(f"\n TOP {n_long} LONG POSITIONS:")
    print("-" * 70)
    print(f"{'Ticker':<10} {'Score':>10} {'Weight':>10} {'Position ($)':>15}")
    print("-" * 70)
    for _, row in positions[positions['side'] == 'LONG'].head(10).iterrows():
        print(f"{row['ticker']:<10} {row['composite_score']:>10.4f} {row['target_weight']:>9.2%} ${row['position_dollars']:>14,.0f}")
    
    print(f"\n TOP {n_short} SHORT POSITIONS:")
    print("-" * 70)
    print(f"{'Ticker':<10} {'Score':>10} {'Weight':>10} {'Position ($)':>15}")
    print("-" * 70)
    for _, row in positions[positions['side'] == 'SHORT'].tail(10).iterrows():
        print(f"{row['ticker']:<10} {row['composite_score']:>10.4f} {row['target_weight']:>9.2%} ${row['position_dollars']:>14,.0f}")
    
    # Export
    if args.output:
        positions.to_csv(args.output, index=False)
        print(f"\n Basket exported to: {args.output}")
    
    # Summary stats
    gross_exposure = positions['target_weight'].abs().sum()
    net_exposure = positions['target_weight'].sum()
    
    print(f"\n Portfolio Summary:")
    print(f"   Gross Exposure: {gross_exposure:.1%}")
    print(f"   Net Exposure: {net_exposure:.1%}")
    print(f"   Number of Positions: {len(positions)}")
    
    return positions


def cmd_version(args):
    """Print version information."""
    from src import __version__, get_version
    print(f"Equity Factors Research System v{get_version()}")
    print(f"Config loaded from: {config.CACHE_DIR}")
    print(f"Log level: {config.LOG_LEVEL}")


# =============================================================================
# Main Argument Parser
# =============================================================================

def create_parser():
    """Create the main argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog='python -m src',
        description='Equity Factors Research System - Unified CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Discover factors in SPY
  python -m src discover --symbols SPY --method PCA -k 10
  
  # Generate trading signals
  python -m src signals generate --universe SPY --output signals.csv
  
  # Check for extreme values
  python -m src signals extremes --universe SPY --trade
  
  # Detect market regime
  python -m src regime detect --universe SPY --predict 5
  
  # Run backtest
  python -m src backtest --universe SPY --walks 10 --optimize
  
  # Launch dashboard
  python -m src dashboard
  
  # Generate report
  python -m src report --type html --open

For more help on a specific command:
  python -m src <command> --help
"""
    )
    
    parser.add_argument(
        '--version', 
        action='store_true', 
        help='Show version information'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # -------------------------------------------------------------------------
    # Discover command
    # -------------------------------------------------------------------------
    discover_parser = subparsers.add_parser(
        'discover', 
        help='Discover and label factors',
        description='Run factor discovery workflow with LLM-powered naming'
    )
    discover_parser.add_argument(
        '-s', '--symbols', 
        required=True, 
        help='Comma-separated tickers/ETFs (e.g., "SPY,QQQ,AAPL")'
    )
    discover_parser.add_argument(
        '--start', 
        default=config.DEFAULT_START_DATE, 
        help=f'Start date (YYYY-MM-DD), default: {config.DEFAULT_START_DATE}'
    )
    discover_parser.add_argument(
        '--method', 
        choices=['PCA', 'ICA', 'NMF', 'AE'], 
        default=config.DEFAULT_FACTOR_METHOD,
        help=f'Factor discovery method (default: {config.DEFAULT_FACTOR_METHOD})'
    )
    discover_parser.add_argument(
        '-k', 
        type=int, 
        default=config.DEFAULT_N_COMPONENTS, 
        help=f'Number of latent factors (default: {config.DEFAULT_N_COMPONENTS})'
    )
    discover_parser.add_argument(
        '--rolling', 
        type=int, 
        default=0, 
        help='Rolling window in days (0 = static)'
    )
    discover_parser.add_argument(
        '--name_out', 
        default='factor_names.csv', 
        help='Output CSV file for factor names'
    )
    
    # -------------------------------------------------------------------------
    # Signals command (with subcommands)
    # -------------------------------------------------------------------------
    signals_parser = subparsers.add_parser(
        'signals', 
        help='Trading signal commands',
        description='Generate and analyze trading signals'
    )
    signals_subparsers = signals_parser.add_subparsers(
        dest='signals_command', 
        help='Signal commands'
    )
    
    # signals generate
    signals_gen = signals_subparsers.add_parser(
        'generate', 
        help='Generate comprehensive signals'
    )
    signals_gen.add_argument(
        '--universe', 
        nargs='+', 
        default=['SPY'], 
        help='Stock/ETF universe'
    )
    signals_gen.add_argument(
        '--method', 
        default='fundamental', 
        choices=['fundamental', 'pca', 'ica'], 
        help='Factor method'
    )
    signals_gen.add_argument(
        '--components', 
        type=int, 
        default=8, 
        help='Number of factors'
    )
    signals_gen.add_argument(
        '--no-expand', 
        dest='expand_etfs', 
        action='store_false', 
        help='Disable ETF expansion'
    )
    signals_gen.add_argument(
        '-o', '--output', 
        help='Output CSV file'
    )
    signals_gen.set_defaults(expand_etfs=True)
    
    # signals extremes
    signals_ext = signals_subparsers.add_parser(
        'extremes', 
        help='Show extreme value alerts'
    )
    signals_ext.add_argument(
        '--universe', 
        nargs='+', 
        default=['SPY']
    )
    signals_ext.add_argument(
        '-t', '--threshold', 
        type=float, 
        default=2.0, 
        help='Z-score threshold'
    )
    signals_ext.add_argument(
        '-p', '--percentile', 
        type=float, 
        default=95.0, 
        help='Percentile threshold'
    )
    signals_ext.add_argument(
        '--trade', 
        action='store_true', 
        help='Show trading implications'
    )
    
    # signals momentum
    signals_mom = signals_subparsers.add_parser(
        'momentum', 
        help='Analyze factor momentum'
    )
    signals_mom.add_argument(
        '--universe', 
        nargs='+', 
        default=['SPY']
    )
    signals_mom.add_argument(
        '-f', '--factor', 
        help='Specific factor to analyze'
    )
    
    # signals cross-section
    signals_cs = signals_subparsers.add_parser(
        'cross-section', 
        help='Cross-sectional rankings'
    )
    signals_cs.add_argument(
        '--universe', 
        nargs='+', 
        default=['SPY']
    )
    signals_cs.add_argument(
        '--top-pct', 
        type=float, 
        default=0.1, 
        help='Top percentile for longs'
    )
    signals_cs.add_argument(
        '--bottom-pct', 
        type=float, 
        default=0.1, 
        help='Bottom percentile for shorts'
    )
    signals_cs.add_argument(
        '-l', '--limit', 
        type=int, 
        default=10, 
        help='Number of stocks to show'
    )
    signals_cs.add_argument(
        '-o', '--output', 
        help='Output CSV file'
    )
    
    # -------------------------------------------------------------------------
    # Regime command
    # -------------------------------------------------------------------------
    regime_parser = subparsers.add_parser(
        'regime', 
        help='Market regime detection',
        description='Detect market regimes and generate allocation recommendations'
    )
    regime_subparsers = regime_parser.add_subparsers(
        dest='regime_command', 
        help='Regime commands'
    )
    
    regime_detect = regime_subparsers.add_parser(
        'detect', 
        help='Detect current market regime'
    )
    regime_detect.add_argument(
        '--universe', 
        nargs='+', 
        default=['SPY']
    )
    regime_detect.add_argument(
        '-r', '--regimes', 
        type=int, 
        default=3, 
        help='Number of regimes'
    )
    regime_detect.add_argument(
        '-p', '--predict', 
        type=int, 
        help='Predict N days ahead'
    )
    regime_detect.add_argument(
        '--summary', 
        action='store_true', 
        help='Show regime statistics'
    )
    
    # -------------------------------------------------------------------------
    # Backtest command
    # -------------------------------------------------------------------------
    backtest_parser = subparsers.add_parser(
        'backtest', 
        help='Backtest signals',
        description='Walk-forward backtesting with performance attribution'
    )
    backtest_parser.add_argument(
        '--universe', 
        nargs='+', 
        default=['SPY']
    )
    backtest_parser.add_argument(
        '--train-size', 
        type=int, 
        default=252, 
        help='Training window in days'
    )
    backtest_parser.add_argument(
        '--test-size', 
        type=int, 
        default=63, 
        help='Test window in days'
    )
    backtest_parser.add_argument(
        '--walks', 
        type=int, 
        default=5, 
        help='Number of walk-forward periods'
    )
    backtest_parser.add_argument(
        '-c', '--confidence', 
        type=float, 
        default=70.0, 
        help='Minimum signal confidence'
    )
    backtest_parser.add_argument(
        '--optimize', 
        action='store_true', 
        help='Optimize thresholds'
    )
    backtest_parser.add_argument(
        '--metric', 
        default='sharpe_ratio', 
        help='Metric to optimize'
    )
    backtest_parser.add_argument(
        '--steps', 
        type=int, 
        default=10, 
        help='Optimization steps'
    )
    backtest_parser.add_argument(
        '--report', 
        action='store_true', 
        help='Show full report'
    )
    backtest_parser.add_argument(
        '-o', '--output', 
        help='Export results to CSV'
    )
    
    # -------------------------------------------------------------------------
    # Dashboard command
    # -------------------------------------------------------------------------
    subparsers.add_parser(
        'dashboard', 
        help='Launch Streamlit dashboard'
    )
    
    # -------------------------------------------------------------------------
    # Report command
    # -------------------------------------------------------------------------
    report_parser = subparsers.add_parser(
        'report', 
        help='Generate reports'
    )
    report_parser.add_argument(
        '--type', 
        choices=['html', 'markdown'], 
        default='markdown', 
        help='Report type'
    )
    report_parser.add_argument(
        '--out', 
        help='Output filename'
    )
    report_parser.add_argument(
        '--open',
        action='store_true',
        help='Open in browser (HTML only)'
    )

    # -------------------------------------------------------------------------
    # Briefing command
    # -------------------------------------------------------------------------
    briefing_parser = subparsers.add_parser(
        'briefing',
        help='Generate morning briefing with actionable recommendations',
        description='Synthesize all signals into a morning briefing with trade recommendations'
    )
    briefing_parser.add_argument(
        '--universe',
        nargs='+',
        default=['SPY'],
        help='Stock/ETF universe (default: SPY)'
    )
    briefing_parser.add_argument(
        '--method',
        default='pca',
        choices=['fundamental', 'pca', 'ica'],
        help='Factor method (default: pca)'
    )
    briefing_parser.add_argument(
        '--components',
        type=int,
        default=8,
        help='Number of factors (default: 8)'
    )
    briefing_parser.add_argument(
        '--output', '-o',
        help='Save briefing to file'
    )
    briefing_parser.add_argument(
        '--format',
        choices=['text', 'markdown'],
        default='text',
        help='Output format (default: text)'
    )

    # -------------------------------------------------------------------------
    # Clean command
    # -------------------------------------------------------------------------
    clean_parser = subparsers.add_parser(
        'clean', 
        help='Clean cache and temporary files'
    )
    clean_parser.add_argument(
        '--all', 
        action='store_true', 
        help='Delete all cache files'
    )
    
    # -------------------------------------------------------------------------
    # Optimize command (SharpeOptimizer)
    # -------------------------------------------------------------------------
    optimize_parser = subparsers.add_parser(
        'optimize',
        help='Optimize factor weights for maximum Sharpe ratio',
        description='Find optimal blended factor weights using various optimization techniques'
    )
    optimize_parser.add_argument(
        '--universe',
        nargs='+',
        default=['SPY'],
        help='Stock/ETF universe for factor data'
    )
    optimize_parser.add_argument(
        '--lookback',
        type=int,
        default=126,
        help='Lookback window in days for optimization (default: 126)'
    )
    optimize_parser.add_argument(
        '--methods',
        nargs='+',
        default=['sharpe', 'momentum', 'risk_parity'],
        help='Methods to blend (default: sharpe momentum risk_parity)'
    )
    optimize_parser.add_argument(
        '--technique',
        choices=['gradient', 'differential', 'bayesian'],
        default='differential',
        help='Optimization technique (default: differential)'
    )
    optimize_parser.add_argument(
        '--walk-forward',
        action='store_true',
        help='Use walk-forward optimization instead of single-period'
    )
    optimize_parser.add_argument(
        '--train-window',
        type=int,
        default=126,
        help='Training window for walk-forward (default: 126)'
    )
    optimize_parser.add_argument(
        '--test-window',
        type=int,
        default=21,
        help='Test window for walk-forward (default: 21)'
    )
    optimize_parser.add_argument(
        '--output',
        help='Save results to JSON file'
    )
    optimize_parser.add_argument(
        '--export-weights',
        help='Export optimal weights to CSV for use in trading'
    )
    optimize_parser.add_argument(
        '--name-factors',
        action='store_true',
        help='Enable LLM-powered factor naming (requires OPENAI_API_KEY)'
    )
    optimize_parser.add_argument(
        '--factor-names-output',
        default='factor_names.csv',
        help='Output file for factor names when --name-factors is enabled (default: factor_names.csv)'
    )
    optimize_parser.add_argument(
        '--factor-method',
        choices=['fundamental', 'pca', 'ica', 'sparse_pca', 'factor'],
        default='pca',
        help='Factor discovery method: fundamental (Fama-French), pca, ica, sparse_pca, or factor (default: pca)'
    )
    optimize_parser.add_argument(
        '--n-components',
        type=int,
        default=10,
        help='Number of factors to extract (default: 10)'
    )
    
    # -------------------------------------------------------------------------
    # Basket command (Generate tradeable basket from optimization)
    # -------------------------------------------------------------------------
    basket_parser = subparsers.add_parser(
        'basket',
        help='Generate tradeable stock basket from optimization results',
        description='Convert factor weights into specific long/short stock positions'
    )
    basket_parser.add_argument(
        '--results',
        required=True,
        help='Path to optimization results JSON file'
    )
    basket_parser.add_argument(
        '--universe',
        nargs='+',
        required=True,
        help='Stock/ETF universe used in optimization'
    )
    basket_parser.add_argument(
        '--factor-method',
        choices=['fundamental', 'pca', 'ica', 'sparse_pca', 'factor'],
        default='pca',
        help='Factor method used in optimization (default: pca)'
    )
    basket_parser.add_argument(
        '--long-pct',
        type=float,
        default=0.1,
        help='Percentile for long positions (default: 0.1 = top 10%%)'
    )
    basket_parser.add_argument(
        '--short-pct',
        type=float,
        default=0.1,
        help='Percentile for short positions (default: 0.1 = bottom 10%%)'
    )
    basket_parser.add_argument(
        '--capital',
        type=float,
        default=100000,
        help='Capital to allocate (default: 100000)'
    )
    basket_parser.add_argument(
        '--net-exposure',
        type=float,
        default=1.0,
        help='Net exposure target (default: 1.0 = 100%% long)'
    )
    basket_parser.add_argument(
        '-o', '--output',
        help='Export basket to CSV file'
    )
    
    return parser


def main():
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle version flag
    if args.version:
        cmd_version(args)
        return
    
    # Handle no command
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Dispatch to command handlers
    command_map = {
        'discover': cmd_discover,
        'dashboard': cmd_dashboard,
        'report': cmd_report,
        'briefing': cmd_briefing,
        'clean': cmd_clean,
        'regime': lambda args: cmd_regime_detect(args) if args.regime_command == 'detect' else None,
        'backtest': cmd_backtest,
        'optimize': cmd_optimize,
        'basket': cmd_basket,
    }
    
    # Handle signals subcommands
    if args.command == 'signals':
        if not args.signals_command:
            parser.parse_args(['signals', '--help'])
            sys.exit(1)
        
        signals_map = {
            'generate': cmd_signals_generate,
            'extremes': cmd_signals_extremes,
            'momentum': cmd_signals_momentum,
            'cross-section': cmd_signals_cross_section,
        }
        handler = signals_map.get(args.signals_command)
        if handler:
            handler(args)
        else:
            print(f"Unknown signals command: {args.signals_command}")
            sys.exit(1)
        return
    
    # Handle regime subcommands
    if args.command == 'regime':
        if not args.regime_command:
            parser.parse_args(['regime', '--help'])
            sys.exit(1)
        
        if args.regime_command == 'detect':
            cmd_regime_detect(args)
        return
    
    # Handle other commands
    handler = command_map.get(args.command)
    if handler:
        handler(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == '__main__':
    main()
