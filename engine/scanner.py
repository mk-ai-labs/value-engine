"""V3 Value Engine - Value Stock Scanner with 5 Guardrails.

Scans the universe for value opportunities using a composite scoring model
and applies 5 guardrails to filter picks:

1. Regime-aware: Only buy in GREEN regime
2. Forward PE < 30 (relaxed for broader coverage)
3. FCF Yield > -5% (allows growth stocks)
4. 3-month momentum > 0 AND RSI < 50 (tighter momentum filter)
5. Max 1 new pick per sector per month (sector cap)

The trailing stop (guardrail #5 in execution) is enforced by the backtest
engine daily, not by the scanner itself.
"""

from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
import yfinance as yf

from engine import config
from engine.regime import RegimeClassifier
from engine.utils import compute_rsi, compute_momentum


class ValueScanner:
    """Live value stock scanner with composite scoring and guardrails.

    The scanner fetches fundamentals and price data from yfinance,
    computes a composite value score, and applies V3 guardrails
    to produce filtered picks.

    Attributes:
        universe: List of tickers to scan.
    """

    def __init__(self, universe: Optional[List[str]] = None) -> None:
        """Initialize scanner with a stock universe.

        Args:
            universe: List of tickers to scan. Defaults to config.UNIVERSE.
        """
        self.universe = universe or config.UNIVERSE

    def fetch_fundamentals(self) -> pd.DataFrame:
        """Fetch fundamental data for all tickers in the universe.

        Uses yfinance to retrieve forward PE, earnings growth,
        free cash flow, market cap, and sector for each ticker.

        Returns:
            DataFrame with columns: ticker, forwardPE, earningsGrowth,
            freeCashflow, marketCap, sector, fcfYield.
        """
        records: List[Dict[str, Any]] = []

        for ticker_str in self.universe:
            try:
                tk = yf.Ticker(ticker_str)
                info = tk.info
                market_cap = info.get('marketCap', 0) or 0
                fcf = info.get('freeCashflow', 0) or 0
                fcf_yield = fcf / market_cap if market_cap > 0 else 0.0

                records.append({
                    'ticker': ticker_str,
                    'forwardPE': info.get('forwardPe', info.get('forwardEps', np.nan)),
                    'earningsGrowth': info.get('earningsGrowth', np.nan),
                    'freeCashflow': fcf,
                    'marketCap': market_cap,
                    'sector': info.get('sector', 'Unknown'),
                    'fcfYield': fcf_yield,
                })
            except Exception:
                records.append({
                    'ticker': ticker_str,
                    'forwardPE': np.nan,
                    'earningsGrowth': np.nan,
                    'freeCashflow': 0,
                    'marketCap': 0,
                    'sector': 'Unknown',
                    'fcfYield': 0.0,
                })

        df = pd.DataFrame(records)
        return df

    def fetch_prices(self, period: str = '6mo') -> pd.DataFrame:
        """Fetch daily closing prices for the universe.

        Args:
            period: yfinance period string (e.g., '6mo', '1y').

        Returns:
            DataFrame with Date index and ticker columns.
        """
        try:
            data = yf.download(
                self.universe,
                period=period,
                progress=False,
                group_by='ticker',
                threads=True,
            )
            # Extract Close prices
            if isinstance(data.columns, pd.MultiIndex):
                closes = pd.DataFrame()
                for ticker in self.universe:
                    try:
                        closes[ticker] = data[ticker]['Close']
                    except (KeyError, TypeError):
                        pass
                return closes
            else:
                return data[['Close']].rename(columns={'Close': self.universe[0]})
        except Exception as e:
            print(f"Warning: Could not fetch prices: {e}")
            return pd.DataFrame()

    def compute_scores(
        self,
        fundamentals_df: pd.DataFrame,
        prices_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute composite value scores for all tickers.

        Composite score = normalized rank of (1/PE_rank + FCF_rank + momentum_rank)
        where lower PE = better, higher FCF = better, higher momentum = better.

        Args:
            fundamentals_df: DataFrame from fetch_fundamentals().
            prices_df: DataFrame from fetch_prices().

        Returns:
            DataFrame with all fundamentals plus score, momentum, and RSI columns.
        """
        df = fundamentals_df.copy()

        # Compute momentum and RSI for each ticker
        mom_values = {}
        rsi_values = {}
        for ticker in df['ticker'].values:
            if ticker in prices_df.columns:
                prices = prices_df[ticker].dropna()
                mom_values[ticker] = compute_momentum(prices, period=63)
                rsi_series = compute_rsi(prices, period=14)
                rsi_values[ticker] = float(rsi_series.iloc[-1]) if not rsi_series.empty else np.nan
            else:
                mom_values[ticker] = np.nan
                rsi_values[ticker] = np.nan

        df['momentum_3m'] = df['ticker'].map(mom_values)
        df['rsi'] = df['ticker'].map(rsi_values)

        # Drop tickers with missing data
        scored = df.dropna(subset=['forwardPE', 'fcfYield', 'momentum_3m', 'rsi']).copy()

        if scored.empty:
            df['score'] = np.nan
            return df

        # Rank components (higher rank = better)
        # Lower PE is better -> rank ascending (rank 1 = lowest PE = best)
        scored['pe_rank'] = scored['forwardPE'].rank(ascending=True, method='min')
        # Higher FCF yield is better -> rank descending
        scored['fcf_rank'] = scored['fcfYield'].rank(ascending=False, method='min')
        # Higher momentum is better -> rank descending
        scored['mom_rank'] = scored['momentum_3m'].rank(ascending=False, method='min')

        # Normalize ranks to 0-1 scale
        n = len(scored)
        scored['pe_rank_norm'] = 1.0 - (scored['pe_rank'] - 1) / max(n - 1, 1)
        scored['fcf_rank_norm'] = 1.0 - (scored['fcf_rank'] - 1) / max(n - 1, 1)
        scored['mom_rank_norm'] = 1.0 - (scored['mom_rank'] - 1) / max(n - 1, 1)

        # Composite score (equal weight)
        scored['score'] = (
            scored['pe_rank_norm'] + scored['fcf_rank_norm'] + scored['mom_rank_norm']
        ) / 3.0

        # Merge scores back
        df = df.merge(
            scored[['ticker', 'score', 'pe_rank_norm', 'fcf_rank_norm', 'mom_rank_norm']],
            on='ticker',
            how='left',
        )
        return df

    def apply_guardrails(
        self,
        scored_df: pd.DataFrame,
        regime: str,
        existing_sectors: Optional[Set[str]] = None,
    ) -> pd.DataFrame:
        """Apply V3 guardrails to filter value picks.

        Guardrails:
        1. Regime must be GREEN (no buying in YELLOW/RED)
        2. Forward PE < MAX_FORWARD_PE
        3. FCF yield > MIN_FCF_YIELD
        4. 3-month momentum > 0 AND RSI < 50
        5. Max 1 per sector (excluding sectors already picked this month)

        Args:
            scored_df: DataFrame from compute_scores().
            regime: Current VIX regime string.
            existing_sectors: Set of sectors already picked this month.

        Returns:
            Filtered DataFrame of eligible picks, sorted by score descending.
        """
        if existing_sectors is None:
            existing_sectors = set()

        df = scored_df.copy()

        # Guardrail 1: Regime must be GREEN
        if regime != 'GREEN':
            return df.iloc[0:0]  # Empty DataFrame with same columns

        # Guardrail 2: Forward PE filter
        df = df[df['forwardPE'] < config.MAX_FORWARD_PE]
        df = df[df['forwardPE'] > 0]  # Exclude negative PE (unprofitable)

        # Guardrail 3: FCF yield filter
        df = df[df['fcfYield'] > config.MIN_FCF_YIELD]

        # Guardrail 4: Tighter momentum filter
        df = df[df['momentum_3m'] > config.MIN_MOM_3M]
        df = df[df['rsi'] < config.MAX_RSI_ENTRY]

        # Guardrail 5: Sector cap
        df = df[~df['sector'].isin(existing_sectors)]

        # Sort by score descending
        df = df.sort_values('score', ascending=False)

        # Apply sector cap: max 1 per sector
        seen_sectors: Set[str] = set()
        filtered_indices = []
        for idx, row in df.iterrows():
            sector = row['sector']
            if sector not in seen_sectors:
                seen_sectors.add(sector)
                filtered_indices.append(idx)
                if len(filtered_indices) >= config.MAX_VALUE_PICKS_PER_MONTH:
                    break

        return df.loc[filtered_indices] if filtered_indices else df.iloc[0:0]

    def scan(
        self,
        regime: Optional[str] = None,
        existing_sectors: Optional[Set[str]] = None,
        top_n: int = 3,
        fundamentals_df: Optional[pd.DataFrame] = None,
        prices_df: Optional[pd.DataFrame] = None,
    ) -> List[Dict[str, Any]]:
        """Full scanning pipeline: fetch data, score, filter, return picks.

        Args:
            regime: VIX regime override. If None, fetches current regime.
            existing_sectors: Sectors already picked this month.
            top_n: Maximum number of picks to return.
            fundamentals_df: Pre-fetched fundamentals (optional, for backtest).
            prices_df: Pre-fetched prices (optional, for backtest).

        Returns:
            List of pick dictionaries with all metadata.
        """
        # Get regime if not provided
        if regime is None:
            regime_info = RegimeClassifier.get_current_regime()
            regime = regime_info.get('regime', 'UNKNOWN')
            if regime == 'UNKNOWN':
                print("Warning: Could not determine regime, defaulting to YELLOW (defensive)")
                regime = 'YELLOW'

        # Fetch data if not provided
        if fundamentals_df is None:
            print("Fetching fundamentals...")
            fundamentals_df = self.fetch_fundamentals()

        if prices_df is None:
            print("Fetching prices...")
            prices_df = self.fetch_prices()

        # Score
        scored_df = self.compute_scores(fundamentals_df, prices_df)

        # Apply guardrails
        filtered = self.apply_guardrails(scored_df, regime, existing_sectors)

        # Limit to top_n
        filtered = filtered.head(top_n)

        # Convert to list of dicts
        picks = []
        for _, row in filtered.iterrows():
            ticker = row['ticker']
            price = float(prices_df[ticker].dropna().iloc[-1]) if ticker in prices_df.columns else np.nan
            picks.append({
                'ticker': ticker,
                'score': round(float(row.get('score', 0)), 3),
                'sector': row.get('sector', 'Unknown'),
                'forwardPE': round(float(row.get('forwardPE', np.nan)), 2),
                'fcfYield': round(float(row.get('fcfYield', 0)), 3),
                'momentum_3m': round(float(row.get('momentum_3m', 0)), 3),
                'rsi': round(float(row.get('rsi', 50)), 1),
                'price': round(price, 2),
            })

        return picks

    def format_picks_report(self, picks: List[Dict[str, Any]]) -> str:
        """Format picks as a clean text report for Telegram/CLI.

        Args:
            picks: List of pick dictionaries from scan().

        Returns:
            Formatted multi-line string report.
        """
        if not picks:
            return "No value picks found (regime may not be GREEN or no stocks pass guardrails)."

        lines = [
            "=== V3 VALUE SCANNER PICKS ===",
            "",
        ]

        for i, pick in enumerate(picks, 1):
            lines.extend([
                f"#{i}  {pick['ticker']}  (Score: {pick['score']:.3f})",
                f"    Sector:     {pick['sector']}",
                f"    Fwd PE:     {pick['forwardPE']:.1f}",
                f"    FCF Yield:  {pick['fcfYield']:.1%}",
                f"    3m Mom:     {pick['momentum_3m']:.1%}",
                f"    RSI:        {pick['rsi']:.1f}",
                f"    Price:      ${pick['price']:.2f}",
                "",
            ])

        lines.append(f"Total picks: {len(picks)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ValueScanner(universe={len(self.universe)} tickers)"
