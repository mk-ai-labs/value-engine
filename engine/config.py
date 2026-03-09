"""V3 Value Engine - Configuration Constants.

All tunable parameters for the value investing engine in one place.
Override via environment variables or by importing and patching.
"""

import os

# ---------------------------------------------------------------------------
# VIX Regime Thresholds
# ---------------------------------------------------------------------------
VIX_GREEN = 18       # VIX < 18  -> GREEN (risk-on)
VIX_YELLOW = 25      # 18 <= VIX < 25 -> YELLOW (caution)
# VIX >= 25 -> RED (risk-off)

# ---------------------------------------------------------------------------
# Portfolio Defaults
# ---------------------------------------------------------------------------
INITIAL_CASH = 5000.0
INITIAL_HOLDINGS = {
    "AAPL": 5,
    "GOOGL": 6,
    "NVDA": 10,
    "AMD": 6,
    "AMZN": 4,
    "MSFT": 3,
    "RVPH": 4,
}
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Trading Rules - Active Strategy (V1/V2/V3)
# ---------------------------------------------------------------------------
TRIM_GAIN_THRESHOLD = 0.30      # Trim winners above 30% gain in YELLOW/RED
TRIM_SELL_FRACTION = 0.50       # Sell 50% of position when trimming
CUT_LOSS_THRESHOLD = -0.25      # Cut losers below -25% loss
REDEPLOY_RSI_THRESHOLD = 35     # RSI threshold for quality redeployment

# ---------------------------------------------------------------------------
# V3 Guarded Value - Scanner Guardrails
# ---------------------------------------------------------------------------
MAX_FORWARD_PE = 30.0           # Guardrail 1: Forward PE < 30
MIN_FCF_YIELD = -0.05           # Guardrail 2: FCF Yield > -5% (allows growth)
MIN_MOM_3M = 0.0                # Guardrail 3: 3-month momentum > 0
MAX_RSI_ENTRY = 50              # Guardrail 4: RSI < 50 for entries
MAX_VALUE_PICKS_PER_MONTH = 3   # Max new value picks per month

# ---------------------------------------------------------------------------
# V3 Guarded Value - Trailing Stop
# ---------------------------------------------------------------------------
TRAILING_STOP_PCT = 0.10        # 10% trailing stop on value picks
TRAILING_STOP_DAYS = 5          # Activate trailing stop after 5 days

# ---------------------------------------------------------------------------
# V3 Position Sizing
# ---------------------------------------------------------------------------
VALUE_POSITION_SIZE_PCT = 0.03  # 3% of portfolio per value pick

# ---------------------------------------------------------------------------
# Quarterly Rescore Noise (for backtest simulation)
# ---------------------------------------------------------------------------
QUARTERLY_RESCORE_NOISE = 0.05  # +/- 5% noise when rescoring fundamentals

# ---------------------------------------------------------------------------
# Stock Universes
# ---------------------------------------------------------------------------
UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    # Semiconductors
    "AMD", "AVGO", "INTC", "QCOM", "ON",
    # Software / Cloud
    "ORCL", "CRM", "ADBE", "NFLX",
    # Financials
    "JPM", "BAC", "GS", "V", "MA",
    # Healthcare
    "JNJ", "UNH", "PFE", "BMY", "LLY",
    # Industrials
    "GE", "CAT", "LMT", "BA", "RTX",
    # Consumer
    "NKE", "PEP", "KO", "WMT", "COST",
    # Energy
    "XOM", "CVX", "COP",
    # Materials
    "NEM", "FCX",
    # Growth / Speculative
    "RIVN", "KULR", "PAYO", "RVPH",
]

QUALITY_REDEPLOY_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NFLX",
    "JNJ", "UNH", "PEP", "KO", "WMT",
    "JPM", "V", "MA",
    "PAYO", "KULR",  # Small-cap oversold candidates
]

# S&P 100 tickers (subset) for market breadth analysis
SP100_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM", "V", "UNH", "JNJ", "XOM", "PG", "MA", "HD", "CVX", "ABBV",
    "MRK", "LLY", "KO", "PEP", "COST", "AVGO", "BAC", "WMT", "MCD",
    "CSCO", "TMO", "ABT", "CRM", "ACN", "ADBE", "AMD", "ORCL", "NFLX",
    "NKE", "INTC", "IBM", "GE", "CAT", "BA", "RTX", "GS", "LOW",
    "QCOM", "TXN", "SBUX", "GIS", "COP", "FCX", "NEM", "LMT",
    "BMY", "PFE", "GILD", "MDT", "SYK", "ISRG", "AMGN",
    "CME", "BLK", "SPGI", "ICE", "AXP", "MMM", "HON", "DE",
    "UPS", "FDX", "DIS", "CMCSA", "TMUS", "VZ", "T",
    "NEE", "DUK", "SO", "D", "AEP",
    "PLD", "AMT", "SPG", "CCI", "EQIX",
]

# ---------------------------------------------------------------------------
# Sector ETFs for Market Analyzer
# ---------------------------------------------------------------------------
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

# ---------------------------------------------------------------------------
# Output Directories
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.environ.get("VALUE_ENGINE_OUTPUT", "output")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")
