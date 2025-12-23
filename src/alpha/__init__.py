"""
Alpha: Position Sizing and Portfolio Construction Module.

This module provides production-quality position sizing, optimization,
and backtesting infrastructure for the triple-barrier momentum strategy.

Key Components:
    - cv: Purged walk-forward cross-validation with embargo
    - sizing: Position sizing rules and constraints
    - backtest: Monday-entry/Friday-rebalance backtest engine
    - reporting: Tear sheets and performance visualization

Trading Protocol:
    - Signal generation: Monday close
    - Rebalance decision: Friday close
    - Holding horizon: 20 trading days (unless barrier hit earlier)

Example:
    >>> from src.alpha import run_backtest
    >>> results = run_backtest(predictions, config)
"""

__version__ = "0.1.0"

# Lazy imports to avoid circular dependencies
__all__ = [
    "cv",
    "sizing",
    "backtest",
    "reporting",
    "models",
    "data",
]
