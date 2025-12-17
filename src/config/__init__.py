"""
Configuration package for the technical dashboard.

This package contains centralized configuration for all models including:
- Regime configuration (trend and volatility thresholds)
- Feature configuration (feature specs, toggles, parameters)
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

from .features import (
    FeatureCategory,
    Timeframe,
    FeatureSpec,
    FeatureConfig,
    get_default_feature_config,
)

from .parallel import ParallelConfig

# Import basic configuration settings
import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Cache directory structure
CACHE_DIR = PROJECT_ROOT / "cache"
STOCKS_CACHE_DIR = CACHE_DIR / "stocks"
ETFS_CACHE_DIR = CACHE_DIR / "etfs"

# Legacy cache file location (for backward compatibility)
CACHE_FILE = CACHE_DIR / "stock_data.pkl"

# Make sure cache directories exist
CACHE_DIR.mkdir(exist_ok=True)
STOCKS_CACHE_DIR.mkdir(exist_ok=True)
ETFS_CACHE_DIR.mkdir(exist_ok=True)

# Environment variables and API settings
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

# Other configuration settings
DEFAULT_RATE_LIMIT = 1.0
DEFAULT_INTERVAL = "1d"

__all__ = [
    # Regime configuration
    'REGIME_CONFIG',
    'GlobalRegimeConfig',
    'TrendRegimeConfig',
    'VolatilityRegimeConfig',
    'get_regime_config',
    'update_trend_thresholds',
    'update_volatility_thresholds',
    'get_all_combined_regimes',
    'ModelRegimeConfig',
    # Feature configuration
    'FeatureCategory',
    'Timeframe',
    'FeatureSpec',
    'FeatureConfig',
    'get_default_feature_config',
    # Parallel configuration
    'ParallelConfig',
    # Global settings
    'CACHE_DIR',
    'STOCKS_CACHE_DIR',
    'ETFS_CACHE_DIR',
    'CACHE_FILE',
    'PROJECT_ROOT',
    'RAPIDAPI_KEY',
    'DEFAULT_RATE_LIMIT',
    'DEFAULT_INTERVAL'
]