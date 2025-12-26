"""
Diagnostic system for the 4-model LightGBM trading system.

Provides comprehensive model diagnostics including:
- Data quality checks (NaN rates, constant features, duplicates)
- Leakage detection (CV overlap, label leakage)
- CV stability analysis
- Calibration and ranking sanity checks
- Sample weighting validation
- Hyperopt parameter analysis
- Feature importance and story consistency

Usage:
    python run_diagnostics.py --model long_normal
    python run_diagnostics.py --all-models
"""

from .core import DiagnosticFlag, Severity, DiagnosticResult, DiagnosticThresholds
from .runner import ModelDiagnosticRunner
from .checks_data_quality import run_data_quality_checks

__all__ = [
    'DiagnosticFlag',
    'Severity',
    'DiagnosticResult',
    'DiagnosticThresholds',
    'ModelDiagnosticRunner',
    'run_data_quality_checks',
]
