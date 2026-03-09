"""V3 Value Engine - Automated Value Investing with VIX Regime Detection.

A production-grade value investing engine featuring:
- VIX-based regime classification (GREEN/YELLOW/RED)
- 5-guardrail value stock scanner
- 4-strategy backtest engine (Buy&Hold, V1 Active, V2 Value+Active, V3 Guarded Value)
- Hourly US market analyzer

Usage:
    from engine.regime import RegimeClassifier
    from engine.scanner import ValueScanner
    from engine.backtest import BacktestEngine
    from engine.market_analyzer import MarketAnalyzer
"""

__version__ = "3.0.0"
__author__ = "Madhur Kapadia"

# Lazy imports to avoid circular dependencies at module level.
# Users import directly from submodules:
#   from engine.regime import RegimeClassifier
#   from engine.scanner import ValueScanner
#   from engine.backtest import BacktestEngine
#   from engine.market_analyzer import MarketAnalyzer

__all__ = [
    "RegimeClassifier",
    "ValueScanner",
    "BacktestEngine",
    "MarketAnalyzer",
]


def __getattr__(name: str):
    """Lazy-load classes on first access from the package."""
    if name == "RegimeClassifier":
        from engine.regime import RegimeClassifier
        return RegimeClassifier
    elif name == "ValueScanner":
        from engine.scanner import ValueScanner
        return ValueScanner
    elif name == "BacktestEngine":
        from engine.backtest import BacktestEngine
        return BacktestEngine
    elif name == "MarketAnalyzer":
        from engine.market_analyzer import MarketAnalyzer
        return MarketAnalyzer
    raise AttributeError(f"module 'engine' has no attribute {name!r}")
