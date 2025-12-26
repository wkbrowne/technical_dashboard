"""Sizing diagnostics module.

Provides comprehensive diagnostics for position sizing and backtesting:
- Data integrity checks (missing predictions, NaNs, duplicates)
- Signal distribution and drift monitoring (PSI, KS tests)
- Portfolio construction metrics (exposure, concentration, turnover)
- Execution risk proxies (liquidity, gap risk)
- Outcome sanity checks (backtest mode)
- Warning system with configurable thresholds

Usage:
    from src.diagnostics.sizing import SizingDiagnosticsRunner

    runner = SizingDiagnosticsRunner(weighted_signals, config)
    report = runner.run()
    runner.save_report("artifacts/diagnostics")
"""

from .schema import (
    SizingDiagnosticReport,
    WeeklyDiagnostics,
    DataIntegrityMetrics,
    SignalDistributionMetrics,
    PortfolioMetrics,
    ExecutionRiskMetrics,
    BacktestMetrics,
    SizingWarning,
    Severity,
)
from .thresholds import SizingDiagnosticThresholds, DEFAULT_SIZING_THRESHOLDS
from .warnings import WarningEngine
from .baselines import BaselineManager
from .runner import SizingDiagnosticsRunner

__all__ = [
    # Schema
    'SizingDiagnosticReport',
    'WeeklyDiagnostics',
    'DataIntegrityMetrics',
    'SignalDistributionMetrics',
    'PortfolioMetrics',
    'ExecutionRiskMetrics',
    'BacktestMetrics',
    'SizingWarning',
    'Severity',
    # Thresholds
    'SizingDiagnosticThresholds',
    'DEFAULT_SIZING_THRESHOLDS',
    # Core
    'WarningEngine',
    'BaselineManager',
    'SizingDiagnosticsRunner',
]
