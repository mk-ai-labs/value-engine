"""V3 Value Engine - Backtest Engine.

Implements four portfolio strategies and runs historical backtests:

1. Buy & Hold: Hold initial positions, no trades
2. V1 Active: Trim winners in YELLOW/RED, cut losers, redeploy into oversold quality
3. V2 Value+Active: V1 rules + simple value scanner (no guardrails)
4. V3 Guarded Value: V1 rules + value scanner WITH 5 guardrails

The engine processes daily price data, applies regime-dependent trading rules,
and tracks portfolio values, trade logs, and performance metrics.
"""

import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yfinance as yf

from engine import config
from engine.regime import RegimeClassifier
from engine.utils import (
    compute_calmar,
    compute_drawdown,
    compute_max_drawdown,
    compute_momentum,
    compute_rsi,
    compute_sharpe,
    compute_sortino,
    compute_win_rate,
)


@dataclass
class Position:
    """Represents a portfolio position.

    Attributes:
        ticker: Stock ticker symbol.
        shares: Number of shares held.
        avg_cost: Average cost per share.
        entry_date: Date position was initiated or last added to.
        is_value_pick: Whether this was a V3 value scanner pick.
        high_watermark: Highest price since entry (for trailing stop).
        sector: Stock sector classification.
    """
    ticker: str
    shares: float
    avg_cost: float
    entry_date: pd.Timestamp = None
    is_value_pick: bool = False
    high_watermark: float = 0.0
    sector: str = 'Unknown'


@dataclass
class Trade:
    """Records a single trade execution.

    Attributes:
        date: Trade execution date.
        action: Trade type (e.g., 'TRIM_YELLOW', 'BUY_VALUE', 'CUT_LOSER').
        ticker: Stock ticker.
        shares: Number of shares traded.
        price: Execution price per share.
        value: Total trade value (shares * price).
        reason: Human-readable reason for the trade.
        regime: Market regime at time of trade.
    """
    date: pd.Timestamp
    action: str
    ticker: str
    shares: float
    price: float
    value: float
    reason: str
    regime: str


class BacktestEngine:
    """Multi-strategy backtest engine for the V3 Value Engine.

    Runs four strategies over historical price data and produces
    daily portfolio values, trade logs, and performance metrics.

    Args:
        prices_df: DataFrame with Date index, columns = tickers, values = daily closes.
        fundamentals_df: DataFrame with ticker, forwardPE, fcfYield, sector, etc.
        vix_series: Series of daily VIX closing values.
        initial_holdings: Dict of {ticker: shares} for starting portfolio.
        initial_cash: Starting cash balance.
    """

    def __init__(
        self,
        prices_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
        vix_series: pd.Series,
        initial_holdings: Optional[Dict[str, int]] = None,
        initial_cash: Optional[float] = None,
    ) -> None:
        self.prices = prices_df.copy()
        self.fundamentals = fundamentals_df.copy()
        self.vix = vix_series.copy()
        self.holdings = initial_holdings or config.INITIAL_HOLDINGS.copy()
        self.cash = initial_cash if initial_cash is not None else config.INITIAL_CASH
        self.dates = self.prices.index.tolist()

        # Seed for reproducibility
        self.rng = np.random.RandomState(config.RANDOM_SEED)

    # ------------------------------------------------------------------
    # Portfolio valuation helpers
    # ------------------------------------------------------------------

    def _portfolio_value(
        self,
        positions: Dict[str, Position],
        cash: float,
        date: pd.Timestamp,
    ) -> float:
        """Compute total portfolio value at a given date.

        Args:
            positions: Current positions dict.
            cash: Current cash balance.
            date: Valuation date.

        Returns:
            Total portfolio value (positions + cash).
        """
        total = cash
        for ticker, pos in positions.items():
            if ticker in self.prices.columns:
                price = self.prices.loc[date, ticker]
                if pd.notna(price):
                    total += pos.shares * price
        return total

    def _get_price(self, ticker: str, date: pd.Timestamp) -> float:
        """Get price for a ticker on a date, with fallback.

        Args:
            ticker: Stock ticker.
            date: Date to look up.

        Returns:
            Price as float. Returns NaN if not available.
        """
        if ticker in self.prices.columns:
            price = self.prices.loc[date, ticker]
            if pd.notna(price):
                return float(price)
        return np.nan

    def _get_regime(self, date: pd.Timestamp) -> str:
        """Get VIX regime for a date.

        Uses the monthly average VIX (first of month) for regime classification.

        Args:
            date: Date to classify.

        Returns:
            Regime string: 'GREEN', 'YELLOW', or 'RED'.
        """
        # Find the most recent VIX value on or before this date
        valid_vix = self.vix[:date]
        if valid_vix.empty:
            return 'YELLOW'  # Default to defensive
        # Use monthly average for regime determination
        month_start = date.replace(day=1)
        month_vix = self.vix[
            (self.vix.index >= month_start) &
            (self.vix.index <= date)
        ]
        if month_vix.empty:
            # Fall back to most recent available
            vix_val = float(valid_vix.iloc[-1])
        else:
            vix_val = float(month_vix.mean())
        return RegimeClassifier.classify(vix_val)

    def _get_monthly_regime(self, date: pd.Timestamp) -> Tuple[str, float]:
        """Get regime using the full month's VIX average.

        Args:
            date: Any date in the month.

        Returns:
            Tuple of (regime_string, vix_average).
        """
        month_start = date.replace(day=1)
        # Get all VIX values for this month up to and including this date
        mask = (self.vix.index >= month_start) & (self.vix.index <= date)
        month_vix = self.vix[mask]
        if month_vix.empty:
            # Try previous month
            prev = month_start - pd.Timedelta(days=1)
            prev_start = prev.replace(day=1)
            mask2 = (self.vix.index >= prev_start) & (self.vix.index <= prev)
            month_vix = self.vix[mask2]
        if month_vix.empty:
            return 'YELLOW', 20.0
        avg = float(month_vix.mean())
        return RegimeClassifier.classify(avg), avg

    def _is_first_trading_day_of_month(self, date: pd.Timestamp, idx: int) -> bool:
        """Check if a date is the first trading day of its month.

        Args:
            date: Date to check.
            idx: Index in self.dates.

        Returns:
            True if this is the first trading day of the month.
        """
        if idx == 0:
            return True
        prev_date = self.dates[idx - 1]
        return date.month != prev_date.month

    def _is_quarter_start(self, date: pd.Timestamp, idx: int) -> bool:
        """Check if date is first trading day of a new quarter.

        Args:
            date: Date to check.
            idx: Index in self.dates.

        Returns:
            True if this is Q1 start (Jan/Apr/Jul/Oct).
        """
        if not self._is_first_trading_day_of_month(date, idx):
            return False
        return date.month in (1, 4, 7, 10)

    # ------------------------------------------------------------------
    # Strategy 1: Buy & Hold
    # ------------------------------------------------------------------

    def run_buy_and_hold(self) -> pd.DataFrame:
        """Run buy-and-hold strategy.

        Simply holds the initial positions without any trading.

        Returns:
            DataFrame with 'Date' and 'BuyHold' columns for daily portfolio values.
        """
        positions = {}
        for ticker, shares in self.holdings.items():
            first_price = np.nan
            for d in self.dates:
                p = self._get_price(ticker, d)
                if not np.isnan(p):
                    first_price = p
                    break
            positions[ticker] = Position(
                ticker=ticker,
                shares=shares,
                avg_cost=first_price if not np.isnan(first_price) else 0,
            )

        values = []
        for date in self.dates:
            val = self._portfolio_value(positions, self.cash, date)
            values.append({'Date': date, 'BuyHold': val})

        return pd.DataFrame(values).set_index('Date')

    # ------------------------------------------------------------------
    # Strategy 2: V1 Active
    # ------------------------------------------------------------------

    def run_v1_active(self) -> Tuple[pd.DataFrame, List[Trade]]:
        """Run V1 Active strategy.

        Rules applied on the first trading day of each month:
        - YELLOW/RED: Trim 50% of positions with >50% gain
        - Any regime: Cut 100% of positions with >25% loss
        - YELLOW/RED: Redeploy freed cash into oversold quality stocks (RSI < 35)

        Returns:
            Tuple of (daily values DataFrame, trade log list).
        """
        positions: Dict[str, Position] = {}
        for ticker, shares in self.holdings.items():
            first_price = np.nan
            for d in self.dates:
                p = self._get_price(ticker, d)
                if not np.isnan(p):
                    first_price = p
                    break
            positions[ticker] = Position(
                ticker=ticker,
                shares=shares,
                avg_cost=first_price if not np.isnan(first_price) else 0,
                entry_date=self.dates[0],
            )

        cash = self.cash
        trades: List[Trade] = []
        values = []

        for idx, date in enumerate(self.dates):
            if self._is_first_trading_day_of_month(date, idx) and idx > 0:
                regime, _ = self._get_monthly_regime(date)
                cash, positions, new_trades = self._apply_v1_rules(
                    date, regime, positions, cash,
                )
                trades.extend(new_trades)

            val = self._portfolio_value(positions, cash, date)
            values.append({'Date': date, 'V1_Active': val})

        return pd.DataFrame(values).set_index('Date'), trades

    def _apply_v1_rules(
        self,
        date: pd.Timestamp,
        regime: str,
        positions: Dict[str, Position],
        cash: float,
    ) -> Tuple[float, Dict[str, Position], List[Trade]]:
        """Apply V1 active management rules.

        Args:
            date: Current date.
            regime: Current VIX regime.
            positions: Current positions.
            cash: Current cash.

        Returns:
            Tuple of (updated cash, updated positions, new trades).
        """
        new_trades: List[Trade] = []
        to_remove: List[str] = []

        # --- Trim winners in YELLOW/RED ---
        if regime in ('YELLOW', 'RED'):
            for ticker, pos in list(positions.items()):
                price = self._get_price(ticker, date)
                if np.isnan(price) or pos.avg_cost <= 0:
                    continue
                gain = (price - pos.avg_cost) / pos.avg_cost
                if gain > config.TRIM_GAIN_THRESHOLD:
                    trim_shares = int(pos.shares * config.TRIM_SELL_FRACTION)
                    if trim_shares > 0:
                        trade_val = trim_shares * price
                        cash += trade_val
                        pos.shares -= trim_shares
                        action_name = f"TRIM_{regime}"
                        new_trades.append(Trade(
                            date=date,
                            action=action_name,
                            ticker=ticker,
                            shares=trim_shares,
                            price=round(price, 2),
                            value=round(trade_val, 2),
                            reason=f"{regime} regime trim: {gain:.1%} gain",
                            regime=regime,
                        ))
                        if pos.shares <= 0:
                            to_remove.append(ticker)

        # --- Cut losers (any regime) ---
        for ticker, pos in list(positions.items()):
            if ticker in to_remove:
                continue
            price = self._get_price(ticker, date)
            if np.isnan(price) or pos.avg_cost <= 0:
                continue
            loss = (price - pos.avg_cost) / pos.avg_cost
            if loss < config.CUT_LOSS_THRESHOLD:
                trade_val = pos.shares * price
                cash += trade_val
                new_trades.append(Trade(
                    date=date,
                    action='CUT_LOSER',
                    ticker=ticker,
                    shares=int(pos.shares),
                    price=round(price, 2),
                    value=round(trade_val, 2),
                    reason=f"Cut loser: {loss:.1%} loss",
                    regime=regime,
                ))
                to_remove.append(ticker)

        for ticker in to_remove:
            if ticker in positions:
                del positions[ticker]

        # --- Redeploy into oversold quality (YELLOW/RED with freed cash) ---
        if regime in ('YELLOW', 'RED') and new_trades:
            freed_cash = sum(t.value for t in new_trades)
            if freed_cash > 100:
                cash, positions, redeploy_trades = self._redeploy_quality(
                    date, regime, positions, cash, freed_cash,
                )
                new_trades.extend(redeploy_trades)

        return cash, positions, new_trades

    def _redeploy_quality(
        self,
        date: pd.Timestamp,
        regime: str,
        positions: Dict[str, Position],
        cash: float,
        budget: float,
    ) -> Tuple[float, Dict[str, Position], List[Trade]]:
        """Redeploy freed cash into oversold quality stocks.

        Looks for quality large-caps with RSI < 35 and buys them
        with the freed cash, roughly equal-weighted.

        Args:
            date: Current date.
            regime: Current regime.
            positions: Current positions.
            cash: Current cash.
            budget: Amount to redeploy.

        Returns:
            Tuple of (updated cash, updated positions, redeploy trades).
        """
        redeploy_trades: List[Trade] = []
        candidates = []

        # Check quality universe for oversold conditions
        for ticker in config.QUALITY_REDEPLOY_UNIVERSE:
            if ticker not in self.prices.columns:
                continue
            price_hist = self.prices[ticker][:date].dropna()
            if len(price_hist) < 20:
                continue
            rsi_series = compute_rsi(price_hist)
            if rsi_series.empty:
                continue
            current_rsi = float(rsi_series.iloc[-1])
            if current_rsi < config.REDEPLOY_RSI_THRESHOLD:
                price = self._get_price(ticker, date)
                if not np.isnan(price) and price > 0:
                    candidates.append({
                        'ticker': ticker,
                        'rsi': current_rsi,
                        'price': price,
                    })

        if not candidates:
            return cash, positions, redeploy_trades

        # Sort by RSI (most oversold first)
        candidates.sort(key=lambda x: x['rsi'])

        # Allocate budget roughly equally
        per_pick = budget / min(len(candidates), 3)

        for cand in candidates[:3]:
            ticker = cand['ticker']
            price = cand['price']
            shares = int(per_pick / price)
            if shares <= 0:
                continue
            cost = shares * price
            if cost > cash:
                shares = int(cash / price)
                cost = shares * price
            if shares <= 0:
                continue

            cash -= cost

            if ticker in positions:
                # Add to existing position
                old = positions[ticker]
                total_shares = old.shares + shares
                old_cost_total = old.shares * old.avg_cost
                new_avg = (old_cost_total + cost) / total_shares
                old.shares = total_shares
                old.avg_cost = new_avg
            else:
                sector = 'Unknown'
                fund_row = self.fundamentals[self.fundamentals['ticker'] == ticker]
                if not fund_row.empty:
                    sector = fund_row.iloc[0].get('sector', 'Unknown')
                positions[ticker] = Position(
                    ticker=ticker,
                    shares=shares,
                    avg_cost=price,
                    entry_date=date,
                    sector=sector,
                )

            redeploy_trades.append(Trade(
                date=date,
                action='REDEPLOY_QUALITY',
                ticker=ticker,
                shares=shares,
                price=round(price, 2),
                value=round(cost, 2),
                reason=f"Quality redeploy: RSI={cand['rsi']:.0f} (oversold)",
                regime=regime,
            ))

        return cash, positions, redeploy_trades

    # ------------------------------------------------------------------
    # Strategy 3: V2 Value + Active (no guardrails)
    # ------------------------------------------------------------------

    def run_v2_value_active(self) -> Tuple[pd.DataFrame, List[Trade]]:
        """Run V2 Value+Active strategy.

        V1 rules plus a simple value scanner without guardrails.
        Picks stocks by highest composite score (PE rank + FCF rank + momentum rank)
        with no sector limits or momentum filters.

        Returns:
            Tuple of (daily values DataFrame, trade log list).
        """
        positions: Dict[str, Position] = {}
        for ticker, shares in self.holdings.items():
            first_price = np.nan
            for d in self.dates:
                p = self._get_price(ticker, d)
                if not np.isnan(p):
                    first_price = p
                    break
            positions[ticker] = Position(
                ticker=ticker,
                shares=shares,
                avg_cost=first_price if not np.isnan(first_price) else 0,
                entry_date=self.dates[0],
            )

        cash = self.cash
        trades: List[Trade] = []
        values = []

        for idx, date in enumerate(self.dates):
            if self._is_first_trading_day_of_month(date, idx) and idx > 0:
                regime, _ = self._get_monthly_regime(date)

                # V1 rules first
                cash, positions, v1_trades = self._apply_v1_rules(
                    date, regime, positions, cash,
                )
                trades.extend(v1_trades)

                # V2 value picks (no guardrails, GREEN or YELLOW)
                if regime in ('GREEN', 'YELLOW'):
                    portfolio_val = self._portfolio_value(positions, cash, date)
                    cash, positions, v2_trades = self._v2_value_picks(
                        date, regime, positions, cash, portfolio_val,
                    )
                    trades.extend(v2_trades)

            val = self._portfolio_value(positions, cash, date)
            values.append({'Date': date, 'V2_Value': val})

        return pd.DataFrame(values).set_index('Date'), trades

    def _v2_value_picks(
        self,
        date: pd.Timestamp,
        regime: str,
        positions: Dict[str, Position],
        cash: float,
        portfolio_value: float,
    ) -> Tuple[float, Dict[str, Position], List[Trade]]:
        """Simple value picks without guardrails (V2).

        Scores universe by composite PE+FCF+momentum rank and buys top 3,
        no sector cap, no RSI filter, no trailing stop.

        Args:
            date: Current date.
            regime: Current regime.
            positions: Current positions.
            cash: Current cash.
            portfolio_value: Current total portfolio value.

        Returns:
            Tuple of (updated cash, updated positions, new trades).
        """
        new_trades: List[Trade] = []
        candidates = []

        for _, row in self.fundamentals.iterrows():
            ticker = row['ticker']
            if ticker in positions:
                continue
            if ticker not in self.prices.columns:
                continue

            price_hist = self.prices[ticker][:date].dropna()
            if len(price_hist) < 65:
                continue

            price = self._get_price(ticker, date)
            if np.isnan(price) or price <= 0:
                continue

            pe = row.get('forwardPE', np.nan)
            fcf_yield = row.get('fcfYield', 0)
            if np.isnan(pe) or pe <= 0:
                continue

            mom = compute_momentum(price_hist, 63)
            if np.isnan(mom):
                continue

            candidates.append({
                'ticker': ticker,
                'pe': pe,
                'fcf_yield': fcf_yield,
                'momentum': mom,
                'price': price,
                'sector': row.get('sector', 'Unknown'),
            })

        if not candidates:
            return cash, positions, new_trades

        cdf = pd.DataFrame(candidates)
        cdf['pe_rank'] = cdf['pe'].rank(ascending=True)
        cdf['fcf_rank'] = cdf['fcf_yield'].rank(ascending=False)
        cdf['mom_rank'] = cdf['momentum'].rank(ascending=False)
        n = len(cdf)
        cdf['score'] = (
            (1 - (cdf['pe_rank'] - 1) / max(n - 1, 1)) +
            (1 - (cdf['fcf_rank'] - 1) / max(n - 1, 1)) +
            (1 - (cdf['mom_rank'] - 1) / max(n - 1, 1))
        ) / 3.0
        cdf = cdf.sort_values('score', ascending=False)

        for _, pick in cdf.head(3).iterrows():
            ticker = pick['ticker']
            price = pick['price']
            position_size = portfolio_value * config.VALUE_POSITION_SIZE_PCT
            shares = int(position_size / price)
            if shares <= 0:
                continue
            cost = shares * price
            if cost > cash:
                shares = int(cash / price)
                cost = shares * price
            if shares <= 0:
                continue

            cash -= cost
            positions[ticker] = Position(
                ticker=ticker,
                shares=shares,
                avg_cost=price,
                entry_date=date,
                sector=pick['sector'],
            )
            new_trades.append(Trade(
                date=date,
                action='BUY_VALUE',
                ticker=ticker,
                shares=shares,
                price=round(price, 2),
                value=round(cost, 2),
                reason=f"V2 value pick [{pick['sector']}]",
                regime=regime,
            ))

        return cash, positions, new_trades

    # ------------------------------------------------------------------
    # Strategy 4: V3 Guarded Value (the main strategy)
    # ------------------------------------------------------------------

    def run_v3_guarded_value(self) -> Tuple[pd.DataFrame, List[Trade]]:
        """Run V3 Guarded Value strategy.

        V1 rules plus value scanner WITH 5 guardrails:
        1. Sector cap: max 1 new pick per sector per month
        2. Rolling fundamentals: re-score quarterly with noise
        3. Regime-aware: only BUY_VALUE in GREEN
        4. Tighter momentum: 3m momentum > 0 AND RSI < 50
        5. Trailing stop: exit value picks if >10% drop from high watermark within 90 days

        Returns:
            Tuple of (daily values DataFrame, trade log list).
        """
        positions: Dict[str, Position] = {}
        for ticker, shares in self.holdings.items():
            first_price = np.nan
            for d in self.dates:
                p = self._get_price(ticker, d)
                if not np.isnan(p):
                    first_price = p
                    break
            sector = 'Unknown'
            fund_row = self.fundamentals[self.fundamentals['ticker'] == ticker]
            if not fund_row.empty:
                sector = fund_row.iloc[0].get('sector', 'Unknown')
            positions[ticker] = Position(
                ticker=ticker,
                shares=shares,
                avg_cost=first_price if not np.isnan(first_price) else 0,
                entry_date=self.dates[0],
                is_value_pick=False,
                high_watermark=first_price if not np.isnan(first_price) else 0,
                sector=sector,
            )

        cash = self.cash
        trades: List[Trade] = []
        values = []
        current_fundamentals = self.fundamentals.copy()
        month_sectors_used: Set[str] = set()
        current_month = None

        for idx, date in enumerate(self.dates):
            # Reset monthly sector tracker on new month
            if current_month is None or date.month != current_month:
                current_month = date.month
                month_sectors_used = set()

            # --- Quarterly re-score with noise ---
            if self._is_quarter_start(date, idx):
                current_fundamentals = self._rescore_fundamentals(current_fundamentals)

            # --- Monthly rebalance on first trading day ---
            if self._is_first_trading_day_of_month(date, idx) and idx > 0:
                regime, vix_avg = self._get_monthly_regime(date)

                # V1 rules: trim winners (YELLOW/RED), cut losers (any)
                cash, positions, v1_trades = self._apply_v1_rules_v3(
                    date, regime, positions, cash,
                )
                trades.extend(v1_trades)

                # V3 value picks (GREEN only)
                if regime == 'GREEN':
                    portfolio_val = self._portfolio_value(positions, cash, date)
                    cash, positions, v3_trades, new_sectors = self._v3_value_picks(
                        date, regime, positions, cash, portfolio_val,
                        current_fundamentals, month_sectors_used,
                    )
                    trades.extend(v3_trades)
                    month_sectors_used.update(new_sectors)

            # --- Daily: check trailing stops ---
            cash, positions, stop_trades = self._check_trailing_stops(
                date, positions, cash,
            )
            trades.extend(stop_trades)

            # --- Daily: update high watermarks ---
            for ticker, pos in positions.items():
                price = self._get_price(ticker, date)
                if not np.isnan(price) and price > pos.high_watermark:
                    pos.high_watermark = price

            val = self._portfolio_value(positions, cash, date)
            values.append({'Date': date, 'V3_Guarded': val})

        return pd.DataFrame(values).set_index('Date'), trades

    def _apply_v1_rules_v3(
        self,
        date: pd.Timestamp,
        regime: str,
        positions: Dict[str, Position],
        cash: float,
    ) -> Tuple[float, Dict[str, Position], List[Trade]]:
        """Apply V1 rules adapted for V3 (preserves value pick metadata).

        Same as _apply_v1_rules but preserves Position objects and metadata.

        Args:
            date: Current date.
            regime: Current regime.
            positions: Current positions.
            cash: Current cash.

        Returns:
            Tuple of (updated cash, updated positions, trades).
        """
        new_trades: List[Trade] = []
        to_remove: List[str] = []

        # --- Trim winners in YELLOW/RED ---
        if regime in ('YELLOW', 'RED'):
            for ticker, pos in sorted(positions.items()):
                price = self._get_price(ticker, date)
                if np.isnan(price) or pos.avg_cost <= 0:
                    continue
                gain = (price - pos.avg_cost) / pos.avg_cost
                if gain > config.TRIM_GAIN_THRESHOLD:
                    trim_shares = int(pos.shares * config.TRIM_SELL_FRACTION)
                    if trim_shares > 0:
                        trade_val = trim_shares * price
                        cash += trade_val
                        pos.shares -= trim_shares
                        action_name = f"TRIM_{regime}"
                        new_trades.append(Trade(
                            date=date,
                            action=action_name,
                            ticker=ticker,
                            shares=trim_shares,
                            price=round(price, 2),
                            value=round(trade_val, 2),
                            reason=f"{regime} regime trim: {gain:.1%} gain",
                            regime=regime,
                        ))
                        if pos.shares <= 0:
                            to_remove.append(ticker)

        # --- Cut losers (any regime) ---
        for ticker, pos in sorted(positions.items()):
            if ticker in to_remove:
                continue
            price = self._get_price(ticker, date)
            if np.isnan(price) or pos.avg_cost <= 0:
                continue
            loss = (price - pos.avg_cost) / pos.avg_cost
            if loss < config.CUT_LOSS_THRESHOLD:
                trade_val = pos.shares * price
                cash += trade_val
                new_trades.append(Trade(
                    date=date,
                    action='CUT_LOSER',
                    ticker=ticker,
                    shares=int(pos.shares),
                    price=round(price, 2),
                    value=round(trade_val, 2),
                    reason=f"Cut loser: {loss:.1%} loss",
                    regime=regime,
                ))
                to_remove.append(ticker)

        for ticker in to_remove:
            if ticker in positions:
                del positions[ticker]

        # --- Redeploy into oversold quality ---
        if regime in ('YELLOW', 'RED') and new_trades:
            freed_cash = sum(t.value for t in new_trades)
            if freed_cash > 100:
                cash, positions, redeploy_trades = self._redeploy_quality_v3(
                    date, regime, positions, cash, freed_cash,
                )
                new_trades.extend(redeploy_trades)

        return cash, positions, new_trades

    def _redeploy_quality_v3(
        self,
        date: pd.Timestamp,
        regime: str,
        positions: Dict[str, Position],
        cash: float,
        budget: float,
    ) -> Tuple[float, Dict[str, Position], List[Trade]]:
        """Redeploy freed cash into oversold quality stocks (V3 version).

        Args:
            date: Current date.
            regime: Current regime.
            positions: Current positions.
            cash: Current cash.
            budget: Amount to redeploy.

        Returns:
            Tuple of (updated cash, updated positions, trades).
        """
        redeploy_trades: List[Trade] = []
        candidates = []

        for ticker in config.QUALITY_REDEPLOY_UNIVERSE:
            if ticker not in self.prices.columns:
                continue
            price_hist = self.prices[ticker][:date].dropna()
            if len(price_hist) < 20:
                continue
            rsi_series = compute_rsi(price_hist)
            if rsi_series.empty:
                continue
            current_rsi = float(rsi_series.iloc[-1])
            if current_rsi < config.REDEPLOY_RSI_THRESHOLD:
                price = self._get_price(ticker, date)
                if not np.isnan(price) and price > 0:
                    candidates.append({
                        'ticker': ticker,
                        'rsi': current_rsi,
                        'price': price,
                    })

        if not candidates:
            return cash, positions, redeploy_trades

        candidates.sort(key=lambda x: x['rsi'])
        per_pick = budget / min(len(candidates), 3)

        for cand in candidates[:3]:
            ticker = cand['ticker']
            price = cand['price']
            shares = int(per_pick / price)
            if shares <= 0:
                continue
            cost = shares * price
            if cost > cash:
                shares = int(cash / price)
                cost = shares * price
            if shares <= 0:
                continue

            cash -= cost
            sector = 'Unknown'
            fund_row = self.fundamentals[self.fundamentals['ticker'] == ticker]
            if not fund_row.empty:
                sector = fund_row.iloc[0].get('sector', 'Unknown')

            if ticker in positions:
                old = positions[ticker]
                total_shares = old.shares + shares
                old_cost_total = old.shares * old.avg_cost
                new_avg = (old_cost_total + cost) / total_shares
                old.shares = total_shares
                old.avg_cost = new_avg
            else:
                positions[ticker] = Position(
                    ticker=ticker,
                    shares=shares,
                    avg_cost=price,
                    entry_date=date,
                    is_value_pick=False,
                    high_watermark=price,
                    sector=sector,
                )

            redeploy_trades.append(Trade(
                date=date,
                action='REDEPLOY_QUALITY',
                ticker=ticker,
                shares=shares,
                price=round(price, 2),
                value=round(cost, 2),
                reason=f"Quality redeploy: RSI={cand['rsi']:.0f} (oversold)",
                regime=regime,
            ))

        return cash, positions, redeploy_trades

    def _v3_value_picks(
        self,
        date: pd.Timestamp,
        regime: str,
        positions: Dict[str, Position],
        cash: float,
        portfolio_value: float,
        fundamentals: pd.DataFrame,
        month_sectors: Set[str],
    ) -> Tuple[float, Dict[str, Position], List[Trade], Set[str]]:
        """V3 guarded value picks with all 5 guardrails.

        Args:
            date: Current date.
            regime: Must be 'GREEN'.
            positions: Current positions.
            cash: Current cash.
            portfolio_value: Total portfolio value.
            fundamentals: Current fundamentals (may have been re-scored with noise).
            month_sectors: Sectors already picked this month.

        Returns:
            Tuple of (updated cash, positions, trades, new sectors used).
        """
        new_trades: List[Trade] = []
        new_sectors: Set[str] = set()

        if regime != 'GREEN':
            return cash, positions, new_trades, new_sectors

        candidates = []
        for _, row in fundamentals.iterrows():
            ticker = row['ticker']
            if ticker in positions:
                continue
            if ticker not in self.prices.columns:
                continue

            price_hist = self.prices[ticker][:date].dropna()
            if len(price_hist) < 65:
                continue

            price = self._get_price(ticker, date)
            if np.isnan(price) or price <= 0:
                continue

            pe = row.get('forwardPE', np.nan)
            fcf_yield = row.get('fcfYield', 0)
            sector = row.get('sector', 'Unknown')

            if np.isnan(pe) or pe <= 0 or pe >= config.MAX_FORWARD_PE:
                continue
            if fcf_yield < config.MIN_FCF_YIELD:
                continue

            # Tighter momentum: 3m > 0 AND RSI < 50
            mom = compute_momentum(price_hist, 63)
            if np.isnan(mom) or mom <= config.MIN_MOM_3M:
                continue

            rsi_series = compute_rsi(price_hist)
            if rsi_series.empty:
                continue
            current_rsi = float(rsi_series.iloc[-1])
            if current_rsi >= config.MAX_RSI_ENTRY:
                continue

            # Sector cap check
            if sector in month_sectors:
                continue

            candidates.append({
                'ticker': ticker,
                'pe': pe,
                'fcf_yield': fcf_yield,
                'momentum': mom,
                'rsi': current_rsi,
                'price': price,
                'sector': sector,
            })

        if not candidates:
            return cash, positions, new_trades, new_sectors

        # Score candidates
        cdf = pd.DataFrame(candidates)
        cdf['pe_rank'] = cdf['pe'].rank(ascending=True)
        cdf['fcf_rank'] = cdf['fcf_yield'].rank(ascending=False)
        cdf['mom_rank'] = cdf['momentum'].rank(ascending=False)
        n = len(cdf)
        cdf['score'] = (
            (1 - (cdf['pe_rank'] - 1) / max(n - 1, 1)) +
            (1 - (cdf['fcf_rank'] - 1) / max(n - 1, 1)) +
            (1 - (cdf['mom_rank'] - 1) / max(n - 1, 1))
        ) / 3.0
        cdf = cdf.sort_values('score', ascending=False)

        # Enforce sector cap: max 1 per sector
        seen_sectors: Set[str] = set()
        picked = []
        for _, row in cdf.iterrows():
            sector = row['sector']
            if sector in seen_sectors or sector in month_sectors:
                continue
            seen_sectors.add(sector)
            picked.append(row)
            if len(picked) >= config.MAX_VALUE_PICKS_PER_MONTH:
                break

        for pick in picked:
            ticker = pick['ticker']
            price = pick['price']
            position_size = portfolio_value * config.VALUE_POSITION_SIZE_PCT
            shares = int(position_size / price)
            if shares <= 0:
                continue
            cost = shares * price
            if cost > cash:
                shares = int(cash / price)
                cost = shares * price
            if shares <= 0:
                continue

            cash -= cost
            sector = pick['sector']
            positions[ticker] = Position(
                ticker=ticker,
                shares=shares,
                avg_cost=price,
                entry_date=date,
                is_value_pick=True,
                high_watermark=price,
                sector=sector,
            )
            new_sectors.add(sector)

            new_trades.append(Trade(
                date=date,
                action='BUY_VALUE',
                ticker=ticker,
                shares=shares,
                price=round(price, 2),
                value=round(cost, 2),
                reason=f"V3 value pick [{sector}]",
                regime=regime,
            ))

        return cash, positions, new_trades, new_sectors

    def _check_trailing_stops(
        self,
        date: pd.Timestamp,
        positions: Dict[str, Position],
        cash: float,
    ) -> Tuple[float, Dict[str, Position], List[Trade]]:
        """Check trailing stops on value picks daily.

        If a value pick drops >10% from its high watermark within 90 days
        of entry, sell 100%.

        Args:
            date: Current date.
            positions: Current positions.
            cash: Current cash.

        Returns:
            Tuple of (updated cash, positions, stop trades).
        """
        stop_trades: List[Trade] = []
        to_remove: List[str] = []

        for ticker, pos in list(positions.items()):
            if not pos.is_value_pick:
                continue
            if pos.entry_date is None:
                continue

            days_held = (date - pos.entry_date).days
            if days_held > config.TRAILING_STOP_DAYS:
                continue

            price = self._get_price(ticker, date)
            if np.isnan(price):
                continue

            if pos.high_watermark <= 0:
                continue

            drop_from_high = (price - pos.high_watermark) / pos.high_watermark
            drop_from_entry = (price - pos.avg_cost) / pos.avg_cost

            if drop_from_entry < -config.TRAILING_STOP_PCT:
                trade_val = pos.shares * price
                cash += trade_val
                regime, _ = self._get_monthly_regime(date)
                stop_trades.append(Trade(
                    date=date,
                    action='TRAILING_STOP',
                    ticker=ticker,
                    shares=int(pos.shares),
                    price=round(price, 2),
                    value=round(trade_val, 2),
                    reason=f"Value pick stopped out: {drop_from_entry:.1%} from entry",
                    regime=regime,
                ))
                to_remove.append(ticker)

        for ticker in to_remove:
            if ticker in positions:
                del positions[ticker]

        return cash, positions, stop_trades

    def _rescore_fundamentals(self, fundamentals: pd.DataFrame) -> pd.DataFrame:
        """Re-score fundamentals with small random noise (quarterly).

        Simulates real-world fundamental data updates by adding
        +/- 5% noise to PE and FCF yield values.

        Args:
            fundamentals: Current fundamentals DataFrame.

        Returns:
            New DataFrame with noisy fundamental values.
        """
        df = fundamentals.copy()
        noise_scale = config.QUARTERLY_RESCORE_NOISE

        for col in ['forwardPE', 'fcfYield']:
            if col in df.columns:
                noise = self.rng.uniform(
                    1 - noise_scale,
                    1 + noise_scale,
                    size=len(df),
                )
                mask = df[col].notna()
                df.loc[mask, col] = df.loc[mask, col] * noise[mask]

        return df

    # ------------------------------------------------------------------
    # Run all strategies
    # ------------------------------------------------------------------

    def run_all(self) -> Dict[str, Any]:
        """Run all four strategies and combine results.

        Returns:
            Dictionary with keys:
            - 'daily': Combined DataFrame of daily portfolio values
            - 'metrics': Dict of strategy -> metrics dict
            - 'trades': Dict of strategy -> trade list
        """
        print("Running Buy & Hold...")
        bh_df = self.run_buy_and_hold()

        print("Running V1 Active...")
        v1_df, v1_trades = self.run_v1_active()

        print("Running V2 Value+Active...")
        v2_df, v2_trades = self.run_v2_value_active()

        print("Running V3 Guarded Value...")
        v3_df, v3_trades = self.run_v3_guarded_value()

        # Combine daily values
        daily = bh_df.join(v1_df).join(v2_df).join(v3_df)

        # Compute metrics for each strategy
        metrics = {}
        for col_name, trade_count in [
            ('BuyHold', '-'),
            ('V1_Active', len(v1_trades)),
            ('V2_Value', len(v2_trades)),
            ('V3_Guarded', len(v3_trades)),
        ]:
            if col_name in daily.columns:
                m = self.compute_metrics(daily[col_name])
                m['Trades'] = trade_count
                metrics[col_name] = m

        return {
            'daily': daily,
            'metrics': metrics,
            'trades': {
                'V1_Active': v1_trades,
                'V2_Value': v2_trades,
                'V3_Guarded': v3_trades,
            },
        }

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------

    def compute_metrics(self, daily_values: pd.Series) -> Dict[str, Any]:
        """Compute performance metrics for a strategy.

        Args:
            daily_values: Series of daily portfolio values.

        Returns:
            Dictionary with Starting, Ending, Total Return, Max Drawdown,
            Sharpe, Sortino, Calmar, Win Rate.
        """
        start_val = float(daily_values.iloc[0])
        end_val = float(daily_values.iloc[-1])
        total_return = (end_val - start_val) / start_val

        daily_returns = daily_values.pct_change().dropna()
        max_dd = compute_max_drawdown(daily_values)
        sharpe = compute_sharpe(daily_returns)
        sortino = compute_sortino(daily_returns)
        calmar = compute_calmar(total_return, max_dd)
        win_rate = compute_win_rate(daily_returns)

        return {
            'Starting': start_val,
            'Ending': end_val,
            'Total Return': total_return,
            'Max Drawdown': max_dd,
            'Sharpe': sharpe,
            'Sortino': sortino,
            'Calmar': calmar,
            'Win Rate': win_rate,
        }

    # ------------------------------------------------------------------
    # Chart generation
    # ------------------------------------------------------------------

    def generate_charts(
        self,
        results_df: pd.DataFrame,
        vix_monthly: Optional[pd.DataFrame] = None,
        save_dir: str = './output',
    ) -> List[str]:
        """Generate 4 comparison charts for all strategies.

        Charts:
        1. Portfolio value with VIX regime shading
        2. Cumulative returns (4 lines)
        3. Monthly returns grouped bar chart
        4. Drawdown comparison

        Args:
            results_df: Combined daily values DataFrame from run_all().
            vix_monthly: Optional monthly VIX DataFrame with 'Regime' column.
            save_dir: Directory to save chart images.

        Returns:
            List of saved file paths.
        """
        os.makedirs(save_dir, exist_ok=True)
        saved_files = []
        strategy_names = {
            'BuyHold': 'Buy & Hold',
            'V1_Active': 'V1 Active',
            'V2_Value': 'V2 Value+Active',
            'V3_Guarded': 'V3 Guarded Value',
        }
        colors = {
            'BuyHold': '#6c757d',
            'V1_Active': '#007bff',
            'V2_Value': '#ffc107',
            'V3_Guarded': '#28a745',
        }

        # --- Chart 1: Portfolio Value with Regime Shading ---
        fig, ax = plt.subplots(figsize=(14, 7))
        if vix_monthly is not None and 'Regime' in vix_monthly.columns:
            for i in range(len(vix_monthly)):
                start = vix_monthly.index[i]
                end = vix_monthly.index[i + 1] if i + 1 < len(vix_monthly) else results_df.index[-1]
                regime = vix_monthly.iloc[i]['Regime']
                color = RegimeClassifier.get_regime_color(regime)
                ax.axvspan(start, end, alpha=0.3, color=color, label=None)

        for col in results_df.columns:
            label = strategy_names.get(col, col)
            ax.plot(results_df.index, results_df[col], label=label,
                    color=colors.get(col, None), linewidth=1.5)

        ax.set_title('Portfolio Value - V3 Value Engine Backtest', fontsize=14, fontweight='bold')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_xlabel('Date')
        ax.legend(loc='upper left')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        fig.autofmt_xdate()
        ax.grid(True, alpha=0.3)
        path = os.path.join(save_dir, 'v3_portfolio_value.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files.append(path)

        # --- Chart 2: Cumulative Returns ---
        fig, ax = plt.subplots(figsize=(14, 7))
        for col in results_df.columns:
            returns = (results_df[col] / results_df[col].iloc[0] - 1) * 100
            label = strategy_names.get(col, col)
            ax.plot(results_df.index, returns, label=label,
                    color=colors.get(col, None), linewidth=1.5)

        ax.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Cumulative Return (%)')
        ax.set_xlabel('Date')
        ax.legend(loc='upper left')
        ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        fig.autofmt_xdate()
        ax.grid(True, alpha=0.3)
        path = os.path.join(save_dir, 'v3_cumulative_returns.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files.append(path)

        # --- Chart 3: Monthly Returns Bar Chart ---
        fig, ax = plt.subplots(figsize=(14, 7))
        monthly_returns = results_df.resample('MS').last().pct_change().dropna() * 100
        x = np.arange(len(monthly_returns))
        width = 0.2
        cols = [c for c in results_df.columns if c in monthly_returns.columns]

        for i, col in enumerate(cols):
            label = strategy_names.get(col, col)
            ax.bar(x + i * width, monthly_returns[col], width,
                   label=label, color=colors.get(col, None), alpha=0.8)

        ax.set_title('Monthly Returns by Strategy', fontsize=14, fontweight='bold')
        ax.set_ylabel('Monthly Return (%)')
        ax.set_xticks(x + width * (len(cols) - 1) / 2)
        labels = [d.strftime('%Y-%m') for d in monthly_returns.index]
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend(loc='upper left')
        ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
        ax.grid(True, alpha=0.3, axis='y')
        path = os.path.join(save_dir, 'v3_monthly_returns.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files.append(path)

        # --- Chart 4: Drawdown Comparison ---
        fig, ax = plt.subplots(figsize=(14, 7))
        for col in results_df.columns:
            dd = compute_drawdown(results_df[col]) * 100
            label = strategy_names.get(col, col)
            ax.fill_between(results_df.index, dd, 0, alpha=0.15, color=colors.get(col, None))
            ax.plot(results_df.index, dd, label=label,
                    color=colors.get(col, None), linewidth=1.0)

        ax.set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Drawdown (%)')
        ax.set_xlabel('Date')
        ax.legend(loc='lower left')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        fig.autofmt_xdate()
        ax.grid(True, alpha=0.3)
        path = os.path.join(save_dir, 'v3_drawdown.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files.append(path)

        return saved_files

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def trades_to_dataframe(self, trades: List[Trade]) -> pd.DataFrame:
        """Convert trade list to a DataFrame.

        Args:
            trades: List of Trade objects.

        Returns:
            DataFrame with trade details.
        """
        if not trades:
            return pd.DataFrame(columns=[
                'Date', 'Action', 'Ticker', 'Shares', 'Price', 'Value', 'Reason', 'Regime',
            ])
        records = []
        for t in trades:
            records.append({
                'Date': t.date.strftime('%Y-%m-%d') if hasattr(t.date, 'strftime') else str(t.date),
                'Action': t.action,
                'Ticker': t.ticker,
                'Shares': int(t.shares),
                'Price': t.price,
                'Value': t.value,
                'Reason': t.reason,
                'Regime': t.regime,
            })
        return pd.DataFrame(records)

    def format_metrics_table(self, metrics: Dict[str, Dict]) -> str:
        """Format metrics as a printable table.

        Args:
            metrics: Dict of strategy_name -> metrics_dict.

        Returns:
            Formatted multi-line string.
        """
        strategy_labels = {
            'BuyHold': 'Buy & Hold',
            'V1_Active': 'V1 Active',
            'V2_Value': 'V2 Value+Active',
            'V3_Guarded': 'V3 Guarded Value',
        }

        lines = [
            "=" * 90,
            f"{'Strategy':<20} {'Starting':>10} {'Ending':>10} {'Return':>9} "
            f"{'MaxDD':>8} {'Sharpe':>7} {'Sortino':>8} {'Calmar':>7} {'WinRate':>8} {'Trades':>7}",
            "-" * 90,
        ]

        for key, m in metrics.items():
            label = strategy_labels.get(key, key)
            lines.append(
                f"{label:<20} "
                f"${m['Starting']:>9,.0f} "
                f"${m['Ending']:>9,.0f} "
                f"{m['Total Return']:>+8.2%} "
                f"{-m['Max Drawdown']:>7.2%} "
                f"{m['Sharpe']:>7.3f} "
                f"{m['Sortino']:>8.3f} "
                f"{m['Calmar']:>7.3f} "
                f"{m['Win Rate']:>7.1%} "
                f"{str(m.get('Trades', '-')):>7}"
            )

        lines.append("=" * 90)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"BacktestEngine(tickers={len(self.prices.columns)}, "
            f"dates={len(self.dates)}, "
            f"holdings={len(self.holdings)})"
        )
