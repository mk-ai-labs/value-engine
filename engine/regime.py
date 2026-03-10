"""V3 Value Engine - VIX Regime Classifier.

Classifies market conditions into three regimes based on VIX levels:
- GREEN (VIX < 18):  Risk-on, allow new value buys
- YELLOW (18 <= VIX < 25):  Caution, trim winners, no new buys
- RED (VIX >= 25):  Risk-off, cut losers aggressively
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from engine import config


class RegimeClassifier:
    """Static VIX-based regime classifier."""

    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"

    _COLORS = {
        "GREEN": "#2ecc71",
        "YELLOW": "#f1c40f",
        "RED": "#e74c3c",
    }

    @staticmethod
    def classify(vix_value: float) -> str:
        """Classify a single VIX value into a regime."""
        if vix_value < config.VIX_GREEN:
            return RegimeClassifier.GREEN
        elif vix_value < config.VIX_YELLOW:
            return RegimeClassifier.YELLOW
        else:
            return RegimeClassifier.RED

    @staticmethod
    def classify_series(vix_series: pd.Series) -> pd.Series:
        """Classify an entire VIX series into regimes."""
        return vix_series.apply(RegimeClassifier.classify)

    @classmethod
    def get_current_regime(cls) -> Dict[str, Any]:
        """Fetch the current VIX and return regime information."""
        try:
            vix_ticker = yf.Ticker("^VIX")
            hist = vix_ticker.history(period="5d")
            if hist.empty:
                return cls._fallback("No VIX data returned")
            vix_val = float(hist["Close"].iloc[-1])
        except Exception as exc:
            return cls._fallback(str(exc))

        regime = cls.classify(vix_val)
        return {
            "vix": round(vix_val, 2),
            "regime": regime,
            "color": cls._COLORS.get(regime, "#95a5a6"),
            "description": cls._describe(regime, vix_val),
            "thresholds": {
                "green_below": config.VIX_GREEN,
                "yellow_below": config.VIX_YELLOW,
            },
        }

    @staticmethod
    def get_regime_color(regime: str) -> str:
        """Return the hex colour for a regime string."""
        return RegimeClassifier._COLORS.get(regime, "#95a5a6")

    @classmethod
    def _fallback(cls, reason: str) -> Dict[str, Any]:
        """Return a cautious YELLOW regime when data is unavailable."""
        return {
            "vix": None,
            "regime": cls.YELLOW,
            "color": cls._COLORS[cls.YELLOW],
            "description": f"Regime unknown ({reason}). Defaulting to YELLOW (caution).",
            "thresholds": {
                "green_below": config.VIX_GREEN,
                "yellow_below": config.VIX_YELLOW,
            },
        }

    @staticmethod
    def _describe(regime: str, vix: float) -> str:
        """Human-readable regime description."""
        if regime == "GREEN":
            return (
                f"VIX at {vix:.1f} -- Risk-on. Safe to deploy capital "
                f"into new value picks."
            )
        elif regime == "YELLOW":
            return (
                f"VIX at {vix:.1f} -- Caution zone. Trim winners, "
                f"hold cash, no new value buys."
            )
        else:
            return (
                f"VIX at {vix:.1f} -- Risk-off. Cut losers, raise cash, "
                f"protect capital."
            )
