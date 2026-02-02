#!/usr/bin/env python
"""
Legacy CLI for Equity Factors Research
======================================

⚠️  DEPRECATION WARNING: This CLI is deprecated. Please use the unified CLI instead:

    uv run python -m src <command>
    
    Or after pip install:
    equity-factors <command>

Migration Guide:
    OLD: uv run python src/cli.py discover --symbols SPY
    NEW: uv run python -m src discover --symbols SPY
    
    OLD: uv run python src/cli.py dashboard
    NEW: uv run python -m src dashboard

The unified CLI in __main__.py provides all the same functionality with 
better organization and more commands.
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path

# Add src to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from src.discover_and_label import run_discovery
from src.reporting import generate_daily_report, generate_detailed_markdown_report

def cmd_discover(args):
    """Run the factor discovery workflow."""
    print(f"🚀 Starting factor discovery for {args.symbols}...")
    run_discovery(
        symbols=args.symbols,
        start_date=args.start,
        method=args.method,
        k=args.k,
        rolling=args.rolling,
        name_out=args.name_out
    )

def cmd_dashboard(args):
    """Launch the Streamlit dashboard."""
    dashboard_path = Path(__file__).parent / "dashboard.py"
    if not dashboard_path.exists():
        print("⚠️  Dashboard file not found. Please create src/dashboard.py first.")
        return
    
    print("📊 Launching dashboard...")
    subprocess.run(["streamlit", "run", str(dashboard_path)], check=True)

def cmd_report(args):
    """Generate performance reports."""
    if args.type == "html":
        generate_daily_report(args.out)
    else:
        generate_detailed_markdown_report(args.out)

def cmd_clean(args):
    """Clean cache files."""
    if args.all:
        db_path = Path("av_cache.db")
        if db_path.exists():
            db_path.unlink()
            print("🗑️  Deleted av_cache.db")
        else:
            print("Cache file not found.")
    else:
        print("Use --all to confirm deletion of the entire cache database.")

def main():
    parser = argparse.ArgumentParser(description="Equity Factors Research CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Discover command
    discover_parser = subparsers.add_parser("discover", help="Discover and label factors")
    discover_parser.add_argument("-s", "--symbols", required=True, help="Comma separated tickers / ETFs")
    discover_parser.add_argument("--start", default="2020-04-01", help="Start date (YYYY-MM-DD)")
    discover_parser.add_argument("--method", choices=["PCA", "ICA", "NMF", "AE"], default="PCA", help="Factor discovery method")
    discover_parser.add_argument("-k", type=int, default=10, help="# latent factors")
    discover_parser.add_argument("--rolling", type=int, default=0, help="Rolling window (days); 0 = static")
    discover_parser.add_argument("--name_out", default="factor_names.csv", help="Output CSV for factor names")

    # Dashboard command
    subparsers.add_parser("dashboard", help="Launch the interactive dashboard")

    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean cache and temporary files")
    clean_parser.add_argument("--all", action="store_true", help="Delete the entire cache database")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate performance reports")
    report_parser.add_argument("--type", choices=["html", "markdown"], default="markdown", help="Report type")
    report_parser.add_argument("--out", default="detailed_report.md", help="Output filename")

    args = parser.parse_args()

    if args.command == "discover":
        cmd_discover(args)
    elif args.command == "dashboard":
        cmd_dashboard(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "clean":
        cmd_clean(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
