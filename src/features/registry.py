"""
Feature registry and factory for creating feature instances.

This module provides a centralized registry of all available features
and a factory function to create feature instances from configuration.

NOTE: This registry is primarily for EXTENSIBILITY and future use.
The main pipeline (orchestrator.py, single_stock.py) calls feature
functions DIRECTLY rather than going through the registry.

If you're adding new features, see CLAUDE.md for the recommended workflow:
1. Create feature function in src/features/
2. Call it from single_stock.py or cross_sectional.py
3. Add to BASE_FEATURES in base_features.py if selected by model

The registry pattern is preserved for:
- Dynamic feature discovery
- Configuration-driven feature selection
- Potential future plugin architecture
"""

from typing import Dict, List, Type, Optional, Any, Callable
import logging

from .base import BaseFeature, CrossSectionalFeature

logger = logging.getLogger(__name__)


class FeatureRegistry:
    """Registry for feature classes.

    This class maintains a mapping from feature names to their implementation
    classes, enabling dynamic feature instantiation from configuration.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._features: Dict[str, Type[BaseFeature]] = {}
        self._cross_sectional: Dict[str, Type[CrossSectionalFeature]] = {}

    def register(
        self,
        name: str,
        feature_class: Type[BaseFeature],
        cross_sectional: bool = False
    ) -> None:
        """Register a feature class.

        Args:
            name: Unique name for the feature
            feature_class: The feature class to register
            cross_sectional: True if this is a cross-sectional feature
        """
        if cross_sectional:
            self._cross_sectional[name] = feature_class
            logger.debug(f"Registered cross-sectional feature: {name}")
        else:
            self._features[name] = feature_class
            logger.debug(f"Registered single-stock feature: {name}")

    def get(self, name: str) -> Optional[Type[BaseFeature]]:
        """Get a feature class by name.

        Args:
            name: Feature name

        Returns:
            Feature class or None if not found
        """
        return self._features.get(name) or self._cross_sectional.get(name)

    def create(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[BaseFeature]:
        """Create a feature instance by name.

        Args:
            name: Feature name
            params: Parameters to pass to the feature constructor

        Returns:
            Feature instance or None if not found
        """
        feature_class = self.get(name)
        if feature_class is None:
            logger.warning(f"Feature not found in registry: {name}")
            return None
        return feature_class(params=params)

    def list_features(self) -> List[str]:
        """List all registered single-stock feature names."""
        return list(self._features.keys())

    def list_cross_sectional(self) -> List[str]:
        """List all registered cross-sectional feature names."""
        return list(self._cross_sectional.keys())

    def list_all(self) -> List[str]:
        """List all registered feature names."""
        return self.list_features() + self.list_cross_sectional()

    def is_cross_sectional(self, name: str) -> bool:
        """Check if a feature is cross-sectional."""
        return name in self._cross_sectional

    def __contains__(self, name: str) -> bool:
        """Check if a feature is registered."""
        return name in self._features or name in self._cross_sectional

    def __len__(self) -> int:
        """Return total number of registered features."""
        return len(self._features) + len(self._cross_sectional)


# Global registry instance
FEATURE_REGISTRY = FeatureRegistry()


def register_feature(
    name: str,
    cross_sectional: bool = False
) -> Callable[[Type[BaseFeature]], Type[BaseFeature]]:
    """Decorator to register a feature class.

    Usage:
        @register_feature('my_feature')
        class MyFeature(BaseFeature):
            ...

    Args:
        name: Name to register the feature under
        cross_sectional: True if this is a cross-sectional feature

    Returns:
        Decorator function
    """
    def decorator(cls: Type[BaseFeature]) -> Type[BaseFeature]:
        FEATURE_REGISTRY.register(name, cls, cross_sectional)
        return cls
    return decorator


def get_feature(name: str) -> Optional[Type[BaseFeature]]:
    """Get a feature class from the global registry.

    Args:
        name: Feature name

    Returns:
        Feature class or None
    """
    return FEATURE_REGISTRY.get(name)


def create_feature(
    name: str,
    params: Optional[Dict[str, Any]] = None
) -> Optional[BaseFeature]:
    """Create a feature instance from the global registry.

    Args:
        name: Feature name
        params: Parameters for the feature

    Returns:
        Feature instance or None
    """
    return FEATURE_REGISTRY.create(name, params)


def list_features() -> List[str]:
    """List all registered single-stock features."""
    return FEATURE_REGISTRY.list_features()


def list_cross_sectional_features() -> List[str]:
    """List all registered cross-sectional features."""
    return FEATURE_REGISTRY.list_cross_sectional()


def list_all_features() -> List[str]:
    """List all registered features."""
    return FEATURE_REGISTRY.list_all()


# =============================================================================
# Legacy Feature Wrappers
# =============================================================================
# These classes wrap the existing feature functions to conform to the new
# BaseFeature interface. They allow gradual migration while maintaining
# compatibility with existing code.

class LegacyFeatureWrapper(BaseFeature):
    """Wrapper to adapt legacy feature functions to the BaseFeature interface.

    This allows existing feature functions (like add_trend_features) to be
    used with the new registry system without rewriting them.
    """

    def __init__(
        self,
        name: str,
        compute_func: Callable,
        params: Optional[Dict[str, Any]] = None,
        category: str = "legacy"
    ):
        """Initialize wrapper.

        Args:
            name: Feature name
            compute_func: The legacy function to wrap
            params: Parameters to pass to the function
            category: Feature category
        """
        super().__init__(params)
        self.name = name
        self.category = category
        self._compute_func = compute_func

    def compute(self, df, **kwargs):
        """Call the wrapped function with stored params and kwargs."""
        # Merge stored params with runtime kwargs
        all_params = {**self.params, **kwargs}
        return self._compute_func(df, **all_params)


class LegacyCrossSectionalWrapper(CrossSectionalFeature):
    """Wrapper for legacy cross-sectional functions."""

    def __init__(
        self,
        name: str,
        compute_func: Callable,
        params: Optional[Dict[str, Any]] = None,
        category: str = "legacy"
    ):
        """Initialize wrapper.

        Args:
            name: Feature name
            compute_func: The legacy function to wrap
            params: Parameters to pass to the function
            category: Feature category
        """
        super().__init__(params)
        self.name = name
        self.category = category
        self._compute_func = compute_func

    def compute_panel(self, indicators_by_symbol, **kwargs):
        """Call the wrapped function."""
        all_params = {**self.params, **kwargs}
        return self._compute_func(indicators_by_symbol, **all_params)


def register_legacy_feature(
    name: str,
    compute_func: Callable,
    params: Optional[Dict[str, Any]] = None,
    category: str = "legacy",
    cross_sectional: bool = False
) -> None:
    """Register a legacy feature function.

    Args:
        name: Feature name
        compute_func: The legacy function
        params: Default parameters
        category: Feature category
        cross_sectional: Whether this is a cross-sectional feature
    """
    if cross_sectional:
        wrapper_class = type(
            f"Legacy_{name}",
            (LegacyCrossSectionalWrapper,),
            {}
        )
        # Create a factory that produces properly configured instances
        def factory(p=None, cf=compute_func, cat=category, n=name):
            return LegacyCrossSectionalWrapper(n, cf, p or params, cat)
        FEATURE_REGISTRY._cross_sectional[name] = factory
    else:
        def factory(p=None, cf=compute_func, cat=category, n=name):
            return LegacyFeatureWrapper(n, cf, p or params, cat)
        FEATURE_REGISTRY._features[name] = factory


# =============================================================================
# Initialize Registry with Existing Features
# =============================================================================

def _initialize_registry():
    """Initialize the registry with all existing feature functions.

    This is called at module load time to populate the registry with
    wrapped versions of all existing feature functions.
    """
    try:
        # Import existing feature functions
        from .trend import add_trend_features, add_rsi_features, add_macd_features
        from .volatility import add_multiscale_vol_regime, add_vol_regime_cs_context
        from .distance import add_distance_to_ma_features
        from .range_breakout import add_range_breakout_features
        from .volume import add_volume_features, add_volume_shock_features
        from .alpha import add_alpha_momentum_features
        from .relstrength import add_relative_strength
        from .breadth import add_breadth_series
        from .xsec import add_xsec_momentum_panel

        # Register single-stock features
        register_legacy_feature(
            'trend_ma',
            add_trend_features,
            category='trend'
        )
        register_legacy_feature(
            'rsi',
            add_rsi_features,
            category='momentum'
        )
        register_legacy_feature(
            'macd',
            add_macd_features,
            category='momentum'
        )
        register_legacy_feature(
            'volatility',
            add_multiscale_vol_regime,
            category='volatility'
        )
        register_legacy_feature(
            'distance_ma',
            add_distance_to_ma_features,
            category='distance'
        )
        register_legacy_feature(
            'range_breakout',
            add_range_breakout_features,
            category='range'
        )
        register_legacy_feature(
            'volume',
            add_volume_features,
            category='volume'
        )
        register_legacy_feature(
            'volume_shock',
            add_volume_shock_features,
            category='volume'
        )

        # Register cross-sectional features
        register_legacy_feature(
            'volatility_cs',
            add_vol_regime_cs_context,
            category='cross_sectional',
            cross_sectional=True
        )
        register_legacy_feature(
            'alpha',
            add_alpha_momentum_features,
            category='alpha',
            cross_sectional=True
        )
        register_legacy_feature(
            'relative_strength',
            add_relative_strength,
            category='relative_strength',
            cross_sectional=True
        )
        register_legacy_feature(
            'breadth',
            add_breadth_series,
            category='breadth',
            cross_sectional=True
        )
        register_legacy_feature(
            'xsec_momentum',
            add_xsec_momentum_panel,
            category='cross_sectional',
            cross_sectional=True
        )

        logger.debug(f"Registry initialized with {len(FEATURE_REGISTRY)} features")

    except ImportError as e:
        logger.warning(f"Could not import all feature modules: {e}")


# Initialize on module load
_initialize_registry()
