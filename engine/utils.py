"""V3 Value Engine - Utility Functions.

Technical indicators, performance metrics, and helper functions
used across the engine modules.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Technical Indicators
# ---------------------------------------------------------------------------

def compute_rsi(prices: pd.Series, window: int = 14) -> float:
    """Compute the Relative Strength Index for the most recent bar.

    Args:
        prices: Series of closing prices (at least *window + 1* bars).
        window: Lookback period (default 14).

    Returns:
        RSI value between 0 and 100.  Returns 50.0 if insufficient data.
    """
    if len(prices) < window + 1:
        return 50.0

    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    # Use Wilder smoothing after the initial SMA
    for i in range(window, len(avg_gain)):
        if not np.isnan(avg_gain.iloc[i - 1]):
            avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (window - 1) + gain.iloc[i]) / window
            avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (window - 1) + loss.iloc[i]) / window

    last_gain = avg_gain.iloc[-1]
    last_loss = avg_loss.iloc[-1]

    if last_loss == 0:
        return 100.0
    if last_gain == 0:
        return 0.0

    rs = last_gain / last_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_rsi_series(prices: pd.Series, window: int = 14) -> pd.Series:
    """Compute a full RSI series for every bar.

    Args:
        prices: Series of closing prices.
        window: Lookback period.

    Returns:
        Series of RSI values aligned to the input index.
    """
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_momentum(prices: pd.Series, period: int = 63) -> float:
    """Compute price momentum (percent return) over *period* trading days.

    Args:
        prices: Series of closing prices.
        period: Number of trading days (default 63 ~ 3 months).

    Returns:
        Fractional return (e.g. 0.15 for +15%).  Returns NaN if
        insufficient data.
    """
    if len(prices) < period + 1:
        return float("nan")
    return (prices.iloc[-1] / prices.iloc[-period - 1]) - 1.0


# ---------------------------------------------------------------------------
# Performance Metrics
# ---------------------------------------------------------------------------

def compute_sharpe(
    returns: pd.Series,
    risk_free_rate: float = 0.04,
    trading_days: int = 252,
) -> float:
    """Annualised Sharpe ratio.

    Args:
        returns: Daily simple returns.
        risk_free_rate: Annual risk-free rate (default 4%).
        trading_days: Trading days per year.

    Returns:
        Sharpe ratio (float).  Returns 0.0 if std is zero.
    """
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / trading_days
    std = excess.std()
    if std == 0:
        return 0.0
    return float(np.sqrt(trading_days) * excess.mean() / std)


def compute_sortino(
    returns: pd.Series,
    risk_free_rate: float = 0.04,
    trading_days: int = 252,
) -> float:
    """Annualised Sortino ratio (downside deviation only).

    Args:
        returns: Daily simple returns.
        risk_free_rate: Annual risk-free rate.
        trading_days: Trading days per year.

    Returns:
        Sortino ratio (float).  Returns 0.0 if downside std is zero.
    """
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / trading_days
    downside = excess[excess < 0]
    if len(downside) == 0:
        return 0.0
    downside_std = downside.std()
    if downside_std == 0:
        return 0.0
    return float(np.sqrt(trading_days) * excess.mean() / downside_std)


def compute_calmar(
    total_return: float,
    max_drawdown: float,
    years: float = 1.0,
) -> float:
    """Calmar ratio: annualised return / max drawdown.

    Args:
        total_return: Cumulative fractional return.
        max_drawdown: Max drawdown as a *positive* fraction (e.g. 0.15).
        years: Investment period in years.

    Returns:
        Calmar ratio.  Returns 0.0 if max_drawdown is zero.
    """
    if max_drawdown == 0:
        return 0.0
    annualised = (1 + total_return) ** (1 / years) - 1
    return float(annualised / max_drawdown)


def compute_drawdown(values: pd.Series) -> pd.Series:
    """Compute the running drawdown series from portfolio values.

    Args:
        values: Series of portfolio values (or cumulative returns).

    Returns:
        Series of drawdown fractions (non-positive, e.g. -0.12 = 12% DD).
    """
    cummax = values.cummax()
    return (values - cummax) / cummax


def compute_max_drawdown(values: pd.Series) -> float:
    """Maximum drawdown as a positive fraction.

    Args:
        values: Series of portfolio values.

    Returns:
        Max drawdown (e.g. 0.17 for a 17% peak-to-trough decline).
    """
    dd = compute_drawdown(values)
    return float(-dd.min()) if len(dd) > 0 else 0.0


def compute_win_rate(returns: pd.Series) -> float:
    """Fraction of positive-return days.

    Args:
        returns: Daily returns series.

    Returns:
        Win rate as a fraction (0.0 to 1.0).
    """
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).sum() / len(returns))


# ---------------------------------------------------------------------------
# Formatting Helpers
# ---------------------------------------------------------------------------

def fmt_currency(value: float) -> str:
    """Format a number as USD currency string."""
    return f"${value:,.0f}"


def fmt_pct(value: float, decimals: int = 2) -> str:
    """Format a fraction as a percentage string."""
    return f"{value * 100:+.{decimals}f}%"


def fmt_regime(regime: str) -> str:
    """Return a coloured emoji prefix for regime display."""
    icons = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}
    return f"{icons.get(regime, '⚪')} {regime}"
