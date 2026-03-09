#!/usr/bin/env python3
"""CLI Runner - V3 Value Engine Backtest.

Runs the four-strategy comparison backtest:
  1. Buy & Hold (baseline)
  2. V1 Active (trim/cut/redeploy)
  3. V2 Value+Active (V1 + naive value picks)
  4. V3 Guarded Value (regime-aware, guardrails, trailing stops)

Data can be fetched live via yfinance or loaded from local CSV files.

Usage:
    # Live fetch (default) - downloads 1yr of data from yfinance
    python run_backtest.py

    # From local CSV files
    python run_backtest.py --prices data/universe_prices_1yr.csv \
                           --fundamentals data/universe_fundamentals.csv

    # Custom output directory and initial capital
    python run_backtest.py --output ./results --cash 10000

    # Run only specific strategies
    python run_backtest.py --strategies v3 buyhold
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Ensure engine package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import yfinance as yf

from engine import config
from engine.backtest import BacktestEngine
from engine.regime import RegimeClassifier
from engine.utils import fmt_currency, fmt_pct


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def fetch_live_data(
    tickers: list[str],
    period: str = "1y",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Fetch prices, fundamentals, and VIX from yfinance."""
    print(f"Fetching price data for {len(tickers)} tickers ({period})...")
    all_tickers = list(set(tickers + ["^VIX"]))
    raw = yf.download(all_tickers, period=period, auto_adjust=True, progress=False)

    # Handle multi-level columns from yf.download
    if isinstance(raw.columns, pd.MultiIndex):
        prices_df = raw["Close"].copy()
    else:
        prices_df = raw[["Close"]].copy()
        prices_df.columns = [all_tickers[0]]

    # Extract VIX and drop from prices
    vix_series = prices_df["^VIX"].dropna() if "^VIX" in prices_df.columns else pd.Series(dtype=float)
    prices_df = prices_df.drop(columns=["^VIX"], errors="ignore")

    # Forward-fill prices for tickers that have gaps
    prices_df = prices_df.ffill()

    print(f"  Prices: {prices_df.shape[0]} days x {prices_df.shape[1]} tickers")
    print(f"  VIX: {len(vix_series)} days, latest={vix_series.iloc[-1]:.1f}" if len(vix_series) > 0 else "  VIX: no data")

    # Fetch fundamentals
    print("Fetching fundamentals...")
    fund_records = []
    for ticker_str in tickers:
        try:
            tk = yf.Ticker(ticker_str)
            info = tk.info
            market_cap = info.get("marketCap", 0) or 0
            fcf = info.get("freeCashflow", 0) or 0
            fcf_yield = fcf / market_cap if market_cap > 0 else 0.0
            fund_records.append({
                "ticker": ticker_str,
                "forwardPE": info.get("forwardPe", info.get("forwardEps", np.nan)),
                "earningsGrowth": info.get("earningsGrowth", np.nan),
                "freeCashflow": fcf,
                "marketCap": market_cap,
                "sector": info.get("sector", "Unknown"),
                "fcfYield": fcf_yield,
            })
        except Exception:
            fund_records.append({
                "ticker": ticker_str,
                "forwardPE": np.nan,
                "earningsGrowth": np.nan,
                "freeCashflow": 0,
                "marketCap": 0,
                "sector": "Unknown",
                "fcfYield": 0.0,
            })

    fundamentals_df = pd.DataFrame(fund_records)
    print(f"  Fundamentals: {len(fundamentals_df)} tickers")

    return prices_df, fundamentals_df, vix_series


def load_csv_data(
    prices_path: str,
    fundamentals_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load pre-saved CSV data files."""
    print(f"Loading prices from {prices_path}...")
    prices_df = pd.read_csv(prices_path, index_col=0, parse_dates=True)

    print(f"Loading fundamentals from {fundamentals_path}...")
    fundamentals_df = pd.read_csv(fundamentals_path)

    # Extract VIX from prices if present
    if "^VIX" in prices_df.columns:
        vix_series = prices_df["^VIX"].dropna()
        prices_df = prices_df.drop(columns=["^VIX"])
    elif "VIX" in prices_df.columns:
        vix_series = prices_df["VIX"].dropna()
        prices_df = prices_df.drop(columns=["VIX"])
    else:
        print("  Warning: No VIX column found in prices. Fetching separately...")
        vix_raw = yf.download("^VIX", period="1y", auto_adjust=True, progress=False)
        vix_series = vix_raw["Close"].squeeze().dropna()

    prices_df = prices_df.ffill()
    print(f"  Prices: {prices_df.shape[0]} days x {prices_df.shape[1]} tickers")
    print(f"  VIX: {len(vix_series)} days")
    print(f"  Fundamentals: {len(fundamentals_df)} tickers")

    return prices_df, fundamentals_df, vix_series


# ---------------------------------------------------------------------------
# Output / export helpers
# ---------------------------------------------------------------------------

def export_results(
    engine: BacktestEngine,
    results: dict,
    output_dir: str,
) -> list[str]:
    """Export results to CSV files and generate charts."""
    os.makedirs(output_dir, exist_ok=True)
    data_dir = os.path.join(output_dir, "data")
    chart_dir = os.path.join(output_dir, "charts")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(chart_dir, exist_ok=True)
    saved = []

    # Daily values
    daily_path = os.path.join(data_dir, "backtest_daily.csv")
    results["daily"].to_csv(daily_path)
    saved.append(daily_path)
    print(f"  Saved: {daily_path}")

    # Metrics
    metrics_df = pd.DataFrame(results["metrics"]).T
    metrics_path = os.path.join(data_dir, "backtest_metrics.csv")
    metrics_df.to_csv(metrics_path)
    saved.append(metrics_path)
    print(f"  Saved: {metrics_path}")

    # Trade logs per strategy
    for strategy_name, trades in results.get("trades", {}).items():
        if trades:
            trades_df = engine.trades_to_dataframe(trades)
            trades_path = os.path.join(data_dir, f"trades_{strategy_name.lower()}.csv")
            trades_df.to_csv(trades_path, index=False)
            saved.append(trades_path)
            print(f"  Saved: {trades_path}")

    # Monthly summary with VIX regime
    daily = results["daily"]
    monthly = daily.resample("ME").last()
    if hasattr(engine, "vix") and len(engine.vix) > 0:
        monthly_vix = engine.vix.resample("ME").mean()
        monthly["VIX_Avg"] = monthly_vix
        monthly["Regime"] = monthly["VIX_Avg"].apply(
            lambda v: RegimeClassifier.classify(v) if pd.notna(v) else "UNKNOWN"
        )
    monthly_path = os.path.join(data_dir, "backtest_monthly.csv")
    monthly.to_csv(monthly_path)
    saved.append(monthly_path)
    print(f"  Saved: {monthly_path}")

    # Charts
    try:
        vix_monthly = monthly[["VIX_Avg", "Regime"]] if "Regime" in monthly.columns else None
        chart_files = engine.generate_charts(results["daily"], vix_monthly=vix_monthly, save_dir=chart_dir)
        saved.extend(chart_files)
        for f in chart_files:
            print(f"  Saved: {f}")
    except Exception as e:
        print(f"  Warning: Chart generation failed: {e}")

    return saved


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="V3 Value Engine - Backtest Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_backtest.py                          # Live fetch, default settings
  python run_backtest.py --cash 10000              # Start with $10,000 cash
  python run_backtest.py --prices data/prices.csv --fundamentals data/fund.csv
  python run_backtest.py --output ./my_results     # Custom output directory
  python run_backtest.py --strategies v3 buyhold   # Only run specific strategies
  python run_backtest.py --period 2y               # 2 years of data
""",
    )

    # Data source
    data_group = parser.add_argument_group("Data Source")
    data_group.add_argument(
        "--prices", type=str, default=None,
        help="Path to prices CSV (date-indexed, tickers as columns). If omitted, fetches live.",
    )
    data_group.add_argument(
        "--fundamentals", type=str, default=None,
        help="Path to fundamentals CSV (columns: ticker, forwardPE, earningsGrowth, ...). If omitted, fetches live.",
    )
    data_group.add_argument(
        "--period", type=str, default="1y",
        help="yfinance period for live fetch (e.g., 1y, 2y, 6mo). Default: 1y",
    )

    # Portfolio
    port_group = parser.add_argument_group("Portfolio")
    port_group.add_argument(
        "--cash", type=float, default=None,
        help=f"Starting cash balance. Default: ${config.INITIAL_CASH:,.0f}",
    )
    port_group.add_argument(
        "--holdings", type=str, default=None,
        help='Initial holdings as JSON, e.g. \'{"AAPL": 5, "MSFT": 3}\'',
    )

    # Strategy selection
    strat_group = parser.add_argument_group("Strategies")
    strat_group.add_argument(
        "--strategies", nargs="+", default=["all"],
        choices=["all", "buyhold", "v1", "v2", "v3"],
        help="Which strategies to run. Default: all",
    )

    # Output
    out_group = parser.add_argument_group("Output")
    out_group.add_argument(
        "--output", type=str, default="./output",
        help="Output directory for CSVs and charts. Default: ./output",
    )
    out_group.add_argument(
        "--no-charts", action="store_true",
        help="Skip chart generation (faster).",
    )
    out_group.add_argument(
        "--json", action="store_true",
        help="Print metrics summary as JSON to stdout.",
    )
    out_group.add_argument(
        "--quiet", action="store_true",
        help="Minimal console output.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    start_time = time.time()
    print("=" * 60)
    print("V3 VALUE ENGINE - BACKTEST RUNNER")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------
    if args.prices and args.fundamentals:
        prices_df, fundamentals_df, vix_series = load_csv_data(
            args.prices, args.fundamentals
        )
    else:
        if args.prices or args.fundamentals:
            print("Warning: Both --prices and --fundamentals must be provided for CSV mode.")
            print("         Falling back to live fetch.")
        prices_df, fundamentals_df, vix_series = fetch_live_data(
            config.UNIVERSE, period=args.period
        )

    # ---------------------------------------------------------------
    # Parse holdings
    # ---------------------------------------------------------------
    holdings = None
    if args.holdings:
        try:
            holdings = json.loads(args.holdings)
            print(f"Custom holdings: {holdings}")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON for --holdings: {args.holdings}")
            sys.exit(1)

    # ---------------------------------------------------------------
    # Initialize engine
    # ---------------------------------------------------------------
    print("\nInitializing BacktestEngine...")
    engine = BacktestEngine(
        prices_df=prices_df,
        fundamentals_df=fundamentals_df,
        vix_series=vix_series,
        initial_holdings=holdings,
        initial_cash=args.cash,
    )
    print(f"  Tickers: {len(engine.prices.columns)}")
    print(f"  Date range: {engine.dates[0].strftime('%Y-%m-%d')} to {engine.dates[-1].strftime('%Y-%m-%d')}")
    print(f"  Trading days: {len(engine.dates)}")
    print(f"  Cash: {fmt_currency(engine.cash)}")

    # ---------------------------------------------------------------
    # Run strategies
    # ---------------------------------------------------------------
    print("\n" + "-" * 60)
    print("RUNNING STRATEGIES")
    print("-" * 60)

    if "all" in args.strategies:
        results = engine.run_all()
    else:
        # Run individual strategies
        daily_frames = []
        all_trades = {}
        all_metrics = {}

        if "buyhold" in args.strategies:
            print("Running Buy & Hold...")
            bh_df = engine.run_buy_and_hold()
            daily_frames.append(bh_df)
            m = engine.compute_metrics(bh_df["BuyHold"])
            m["Trades"] = "-"
            all_metrics["BuyHold"] = m

        if "v1" in args.strategies:
            print("Running V1 Active...")
            v1_df, v1_trades = engine.run_v1_active()
            daily_frames.append(v1_df)
            m = engine.compute_metrics(v1_df["V1_Active"])
            m["Trades"] = len(v1_trades)
            all_metrics["V1_Active"] = m
            all_trades["V1_Active"] = v1_trades

        if "v2" in args.strategies:
            print("Running V2 Value+Active...")
            v2_df, v2_trades = engine.run_v2_value_active()
            daily_frames.append(v2_df)
            m = engine.compute_metrics(v2_df["V2_Value"])
            m["Trades"] = len(v2_trades)
            all_metrics["V2_Value"] = m
            all_trades["V2_Value"] = v2_trades

        if "v3" in args.strategies:
            print("Running V3 Guarded Value...")
            v3_df, v3_trades = engine.run_v3_guarded_value()
            daily_frames.append(v3_df)
            m = engine.compute_metrics(v3_df["V3_Guarded"])
            m["Trades"] = len(v3_trades)
            all_metrics["V3_Guarded"] = m
            all_trades["V3_Guarded"] = v3_trades

        # Combine
        if daily_frames:
            daily = daily_frames[0]
            for df in daily_frames[1:]:
                daily = daily.join(df)
        else:
            print("Error: No strategies selected.")
            sys.exit(1)

        results = {
            "daily": daily,
            "metrics": all_metrics,
            "trades": all_trades,
        }

    # ---------------------------------------------------------------
    # Print metrics summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(engine.format_metrics_table(results["metrics"]))

    # Trade counts
    for strategy_name, trades in results.get("trades", {}).items():
        print(f"  {strategy_name}: {len(trades)} trades")

    # ---------------------------------------------------------------
    # Export
    # ---------------------------------------------------------------
    print("\n" + "-" * 60)
    print("EXPORTING RESULTS")
    print("-" * 60)
    saved = export_results(engine, results, args.output)

    # ---------------------------------------------------------------
    # JSON output (optional)
    # ---------------------------------------------------------------
    if args.json:
        json_metrics = {}
        for strat, m in results["metrics"].items():
            json_metrics[strat] = {
                k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                for k, v in m.items()
            }
        print("\n--- JSON METRICS ---")
        print(json.dumps(json_metrics, indent=2))

    # ---------------------------------------------------------------
    # Done
    # ---------------------------------------------------------------
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"COMPLETED in {elapsed:.1f}s")
    print(f"Output: {os.path.abspath(args.output)}")
    print(f"Files saved: {len(saved)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
