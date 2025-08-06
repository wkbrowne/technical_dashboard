"""
Configuration package for the technical dashboard.

This package contains centralized configuration for all models including:
- Regime configuration (trend and volatility thresholds)
- Model parameters
- Global settings
"""

from .regime_config import (
    REGIME_CONFIG,
    GlobalRegimeConfig,
    TrendRegimeConfig, 
    VolatilityRegimeConfig,
    get_regime_config,
    update_trend_thresholds,
    update_volatility_thresholds,
    get_all_combined_regimes,
    ModelRegimeConfig
)

# Import basic configuration settings
import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Cache file location
CACHE_FILE = PROJECT_ROOT / "cache" / "stock_data.pkl"

# Make sure cache directory exists
CACHE_FILE.parent.mkdir(exist_ok=True)

# Environment variables and API settings
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

# Other configuration settings
DEFAULT_RATE_LIMIT = 1.0
DEFAULT_INTERVAL = "1d"

__all__ = [
    'REGIME_CONFIG',
    'GlobalRegimeConfig',
    'TrendRegimeConfig',
    'VolatilityRegimeConfig', 
    'get_regime_config',
    'update_trend_thresholds',
    'update_volatility_thresholds',
    'get_all_combined_regimes',
    'ModelRegimeConfig',
    'CACHE_FILE',
    'PROJECT_ROOT',
    'RAPIDAPI_KEY',
    'DEFAULT_RATE_LIMIT',
    'DEFAULT_INTERVAL'
]