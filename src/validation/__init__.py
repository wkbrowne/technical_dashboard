"""
Centralized data validation for the trading pipeline.

This module provides consistent validation logic for target generation,
ensuring all filtering thresholds are defined in one place.
"""

from .target_data import (
    TargetDataValidator,
    DEFAULT_VALIDATION_CONFIG,
    filter_symbols_by_price,
    filter_invalid_entries,
    filter_extreme_mfe_mae,
    filter_extreme_returns,
    validate_target_input_data,
)

__all__ = [
    'TargetDataValidator',
    'DEFAULT_VALIDATION_CONFIG',
    'filter_symbols_by_price',
    'filter_invalid_entries',
    'filter_extreme_mfe_mae',
    'filter_extreme_returns',
    'validate_target_input_data',
]
