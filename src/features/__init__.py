"""
Feature computation modules for technical analysis.

This package contains modules for computing various technical indicators and features:

Core Components:
- base: BaseFeature and CrossSectionalFeature abstract base classes
- registry: Feature registration and factory
- timeframe: Unified D/W/M resampling and merge
- single_stock: Orchestrates single-stock feature computation
- cross_sectional: Orchestrates cross-sectional feature computation

Feature Modules:
- trend: Moving average slopes, RSI, MACD
- volatility: Multi-scale volatility regime features
- distance: Distance to moving average features
- range_breakout: Range and breakout features, ATR
- volume: Volume-based features
- alpha: Alpha momentum features
- relstrength: Relative strength features
- breadth: Market breadth features
- xsec: Cross-sectional momentum features

Support:
- assemble: Core data assembly functions
- hurst: Hurst exponent features
- postprocessing: NaN interpolation, lags, outlier handling
- target_generation: Triple barrier target labels
"""

from .single_stock import compute_single_stock_features, compute_single_stock_features_safe
from .cross_sectional import compute_cross_sectional_features, compute_cross_sectional_features_safe
from .base import BaseFeature, CrossSectionalFeature, FeatureError, ValidationError
from .registry import (
    FEATURE_REGISTRY,
    register_feature,
    get_feature,
    create_feature,
    list_features,
    list_all_features,
)
from .timeframe import (
    TimeframeType,
    TimeframeConfig,
    TimeframeResampler,
    partition_by_symbol,
    combine_to_long,
)

__all__ = [
    # Core functions
    'compute_single_stock_features',
    'compute_single_stock_features_safe',
    'compute_cross_sectional_features',
    'compute_cross_sectional_features_safe',
    # Base classes
    'BaseFeature',
    'CrossSectionalFeature',
    'FeatureError',
    'ValidationError',
    # Registry
    'FEATURE_REGISTRY',
    'register_feature',
    'get_feature',
    'create_feature',
    'list_features',
    'list_all_features',
    # Timeframe
    'TimeframeType',
    'TimeframeConfig',
    'TimeframeResampler',
    'partition_by_symbol',
    'combine_to_long',
]