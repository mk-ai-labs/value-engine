#!/usr/bin/env python3
"""CLI Runner - V3 Value Stock Scanner.

Scans the universe for top value picks using 5 guardrails:
  1. Forward PE < 30
  2. FCF Yield > -5%
  3. 3-month momentum > 0%
  4. RSI < 50 (entry zone)
  5. Sector diversification (max 2 per sector)

Scoring: composite of forwardPE (inverse), FCF Yield, momentum, RSI.

Usage:
    # Default scan (full universe, current regime)
    python run_scanner.py

    # Force a specific regime
    python run_scanner.py --regime GREEN

    # Custom universe
    python run_scanner.py --tickers AAPL MSFT GOOGL AMZN NVDA META JPM BAC

    # Top 5 picks, output as JSON
    python run_scanner.py --top 5 --json

    # Save results to CSV
    python run_scanner.py --output results/scan_today.csv
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure engine package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from engine import config
from engine.scanner import ValueScanner
from engine.regime import RegimeClassifier
from engine.utils import fmt_pct


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="V3 Value Engine - Stock Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_scanner.py                            # Full universe, auto-detect regime
  python run_scanner.py --regime GREEN             # Force GREEN regime rules
  python run_scanner.py --regime RED               # RED = no new buys (safety)
  python run_scanner.py --top 5                    # Return top 5 picks
  python run_scanner.py --tickers AAPL MSFT GOOGL  # Scan specific tickers
  python run_scanner.py --output picks.csv         # Save to CSV
  python run_scanner.py --json                     # Output as JSON
""",
    )

    # Scan parameters
    scan_group = parser.add_argument_group("Scan Options")
    scan_group.add_argument(
        "--tickers", nargs="+", default=None,
        help=f"Tickers to scan. Default: full {len(config.UNIVERSE)}-stock universe.",
    )
    scan_group.add_argument(
        "--regime", type=str, default=None, choices=["GREEN", "YELLOW", "RED"],
        help="Force VIX regime. Default: auto-detect from live VIX.",
    )
    scan_group.add_argument(
        "--top", type=int, default=3,
        help="Max number of picks to return. Default: 3",
    )
    scan_group.add_argument(
        "--exclude-sectors", nargs="+", default=None,
        help="Sectors to exclude (already held). E.g., 'Technology' 'Financial Services'",
    )

    # Output
    out_group = parser.add_argument_group("Output")
    out_group.add_argument(
        "--output", type=str, default=None,
        help="Save picks to CSV file path.",
    )
    out_group.add_argument(
        "--json", action="store_true",
        help="Output picks as JSON to stdout.",
    )
    out_group.add_argument(
        "--quiet", action="store_true",
        help="Minimal output (just the picks table).",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    start_time = time.time()

    if not args.quiet:
        print("=" * 60)
        print("V3 VALUE ENGINE - STOCK SCANNER")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

    # ---------------------------------------------------------------
    # Get current regime
    # ---------------------------------------------------------------
    if args.regime:
        regime = args.regime
        if not args.quiet:
            print(f"\nRegime: {regime} (forced via --regime)")
    else:
        if not args.quiet:
            print("\nDetecting VIX regime...")
        regime_info = RegimeClassifier.get_current_regime()
        regime = regime_info.get("regime", "YELLOW")
        vix_val = regime_info.get("vix", "N/A")
        if not args.quiet:
            print(f"  VIX: {vix_val}")
            print(f"  Regime: {regime}")
            desc = regime_info.get("description", "")
            if desc:
                print(f"  {desc}")

    # Check RED regime
    if regime == "RED":
        print("\n** RED REGIME: No new value picks recommended. Market stress elevated. **")
        if not args.json:
            return

    # ---------------------------------------------------------------
    # Initialize scanner
    # ---------------------------------------------------------------
    universe = args.tickers or config.UNIVERSE
    scanner = ValueScanner(universe=universe)

    if not args.quiet:
        print(f"\nScanning {len(universe)} tickers...")

    # ---------------------------------------------------------------
    # Run scan
    # ---------------------------------------------------------------
    existing_sectors = set(args.exclude_sectors) if args.exclude_sectors else None

    picks = scanner.scan(
        regime=regime,
        existing_sectors=existing_sectors,
        top_n=args.top,
    )

    # ---------------------------------------------------------------
    # Display results
    # ---------------------------------------------------------------
    elapsed = time.time() - start_time

    if args.json:
        output = {
            "scan_time": datetime.now().isoformat(),
            "regime": regime,
            "universe_size": len(universe),
            "picks_count": len(picks),
            "picks": picks,
        }
        print(json.dumps(output, indent=2))
        return

    if not picks:
        print("\nNo picks passed all guardrails.")
        print("Try widening the universe or adjusting guardrail thresholds in config.py.")
    else:
        # Print formatted report
        report = scanner.format_picks_report(picks)
        print("\n" + report)

        # Also print as a clean table
        print("\n" + "-" * 60)
        print(f"{'Ticker':<8} {'Score':>6} {'Sector':<22} {'FwdPE':>7} {'FCF%':>7} {'Mom3m':>7} {'RSI':>5} {'Price':>9}")
        print("-" * 60)
        for pick in picks:
            print(
                f"{pick['ticker']:<8} "
                f"{pick['score']:>6.3f} "
                f"{pick['sector']:<22} "
                f"{pick['forwardPE']:>7.1f} "
                f"{pick['fcfYield']:>6.1%} "
                f"{pick['momentum_3m']:>6.1%} "
                f"{pick['rsi']:>5.1f} "
                f"${pick['price']:>8.2f}"
            )
        print("-" * 60)

    # ---------------------------------------------------------------
    # Save to CSV (optional)
    # ---------------------------------------------------------------
    if args.output and picks:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(picks)
        df.to_csv(args.output, index=False)
        print(f"\nSaved {len(picks)} picks to {args.output}")

    if not args.quiet:
        print(f"\nCompleted in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
