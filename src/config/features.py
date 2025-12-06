"""
Feature configuration for the technical dashboard pipeline.

This module provides dataclass-based configuration for feature specifications,
enabling easy on/off toggling, parameter customization, and YAML loading.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FeatureCategory(str, Enum):
    """Categories of features for organization and filtering."""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    DISTANCE = "distance"
    RANGE = "range"
    CROSS_SECTIONAL = "cross_sectional"
    RELATIVE_STRENGTH = "relative_strength"
    BREADTH = "breadth"
    ALPHA = "alpha"


class Timeframe(str, Enum):
    """Supported timeframes for feature computation."""
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"

    @classmethod
    def from_string(cls, s: str) -> 'Timeframe':
        """Parse timeframe from string."""
        mapping = {
            'D': cls.DAILY, 'daily': cls.DAILY, 'DAILY': cls.DAILY,
            'W': cls.WEEKLY, 'weekly': cls.WEEKLY, 'WEEKLY': cls.WEEKLY,
            'M': cls.MONTHLY, 'monthly': cls.MONTHLY, 'MONTHLY': cls.MONTHLY,
        }
        if s in mapping:
            return mapping[s]
        raise ValueError(f"Unknown timeframe: {s}")


@dataclass
class FeatureSpec:
    """Specification for a single feature or feature group.

    Attributes:
        name: Unique identifier for this feature
        category: Category for grouping (trend, momentum, etc.)
        timeframes: List of timeframes where this feature applies
        enabled: Whether this feature should be computed
        params: Feature-specific parameters
        requires_cross_sectional: True if feature needs data from multiple stocks
        depends_on: List of feature names that must be computed first
        description: Human-readable description
    """
    name: str
    category: FeatureCategory
    timeframes: List[Timeframe] = field(default_factory=lambda: [Timeframe.DAILY])
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    requires_cross_sectional: bool = False
    depends_on: List[str] = field(default_factory=list)
    description: str = ""

    def __post_init__(self):
        # Convert string category to enum if needed
        if isinstance(self.category, str):
            self.category = FeatureCategory(self.category)
        # Convert string timeframes to enum if needed
        self.timeframes = [
            Timeframe.from_string(t) if isinstance(t, str) else t
            for t in self.timeframes
        ]


@dataclass
class FeatureConfig:
    """Complete feature configuration for the pipeline.

    This class manages all feature specifications and provides methods
    for filtering, loading from YAML, and saving configurations.

    Attributes:
        features: Dictionary mapping feature name to FeatureSpec
        nan_handling: How to handle NaN values ('preserve', 'interpolate_inside', 'drop')
        min_valid_ratio: Minimum ratio of valid values to keep a row (0-1)
    """
    features: Dict[str, FeatureSpec] = field(default_factory=dict)
    nan_handling: str = "preserve"
    min_valid_ratio: float = 0.5

    @classmethod
    def default(cls) -> 'FeatureConfig':
        """Create default configuration with all standard features."""
        config = cls()

        # === SINGLE-STOCK FEATURES ===

        # Trend features
        config.features['trend_ma'] = FeatureSpec(
            name='trend_ma',
            category=FeatureCategory.TREND,
            timeframes=[Timeframe.DAILY, Timeframe.WEEKLY, Timeframe.MONTHLY],
            enabled=True,
            params={
                'ma_periods': (10, 20, 30, 50, 75, 100, 150, 200),
                'slope_window': 20,
                'eps': 1e-5,
            },
            description="Moving average slopes, agreement, trend scores"
        )

        # RSI features
        config.features['rsi'] = FeatureSpec(
            name='rsi',
            category=FeatureCategory.MOMENTUM,
            timeframes=[Timeframe.DAILY, Timeframe.WEEKLY],
            enabled=True,
            params={
                'periods': (14, 21, 30),
            },
            description="RSI momentum oscillator at multiple periods"
        )

        # MACD features
        config.features['macd'] = FeatureSpec(
            name='macd',
            category=FeatureCategory.MOMENTUM,
            timeframes=[Timeframe.DAILY, Timeframe.WEEKLY],
            enabled=True,
            params={
                'fast': 12,
                'slow': 26,
                'signal': 9,
                'derivative_ema_span': 3,
            },
            description="MACD histogram and derivative"
        )

        # Volatility features
        config.features['volatility'] = FeatureSpec(
            name='volatility',
            category=FeatureCategory.VOLATILITY,
            timeframes=[Timeframe.DAILY],
            enabled=True,
            params={
                'short_windows': (10, 20),
                'long_windows': (60, 100),
                'z_window': 60,
                'ema_span': 10,
                'slope_win': 20,
            },
            description="Multi-scale volatility regime features"
        )

        # Distance to MA features
        config.features['distance_ma'] = FeatureSpec(
            name='distance_ma',
            category=FeatureCategory.DISTANCE,
            timeframes=[Timeframe.DAILY, Timeframe.WEEKLY],
            enabled=True,
            params={
                'ma_lengths': (20, 50, 100, 200),
                'z_window': 60,
            },
            description="Distance to moving averages with z-scores"
        )

        # Range and breakout features
        config.features['range_breakout'] = FeatureSpec(
            name='range_breakout',
            category=FeatureCategory.RANGE,
            timeframes=[Timeframe.DAILY],
            enabled=True,
            params={
                'win_list': (5, 10, 20),
            },
            description="Range, breakout, and ATR features"
        )

        # Volume features
        config.features['volume'] = FeatureSpec(
            name='volume',
            category=FeatureCategory.VOLUME,
            timeframes=[Timeframe.DAILY],
            enabled=True,
            params={},
            description="Volume ratios and patterns"
        )

        # Volume shock features
        config.features['volume_shock'] = FeatureSpec(
            name='volume_shock',
            category=FeatureCategory.VOLUME,
            timeframes=[Timeframe.DAILY],
            enabled=True,
            params={
                'lookback': 20,
                'ema_span': 10,
            },
            description="Volume shock and price alignment features"
        )

        # === CROSS-SECTIONAL FEATURES ===

        # Volatility cross-sectional context
        config.features['volatility_cs'] = FeatureSpec(
            name='volatility_cs',
            category=FeatureCategory.CROSS_SECTIONAL,
            timeframes=[Timeframe.DAILY],
            enabled=True,
            requires_cross_sectional=True,
            depends_on=['volatility'],
            params={},
            description="Cross-sectional volatility regime context"
        )

        # Alpha momentum features
        config.features['alpha'] = FeatureSpec(
            name='alpha',
            category=FeatureCategory.ALPHA,
            timeframes=[Timeframe.DAILY],
            enabled=True,
            requires_cross_sectional=True,
            params={
                'beta_win': 60,
                'windows': (20, 60, 120),
                'ema_span': 10,
            },
            description="Alpha-momentum vs market and sectors"
        )

        # Relative strength features
        config.features['relative_strength'] = FeatureSpec(
            name='relative_strength',
            category=FeatureCategory.RELATIVE_STRENGTH,
            timeframes=[Timeframe.DAILY, Timeframe.WEEKLY],
            enabled=True,
            requires_cross_sectional=True,
            params={},
            description="Relative strength vs SPY, sectors, subsectors"
        )

        # Market breadth features
        config.features['breadth'] = FeatureSpec(
            name='breadth',
            category=FeatureCategory.BREADTH,
            timeframes=[Timeframe.DAILY],
            enabled=True,
            requires_cross_sectional=True,
            params={},
            description="Market breadth (advancing/declining, highs/lows)"
        )

        # Cross-sectional momentum
        config.features['xsec_momentum'] = FeatureSpec(
            name='xsec_momentum',
            category=FeatureCategory.CROSS_SECTIONAL,
            timeframes=[Timeframe.DAILY],
            enabled=True,
            requires_cross_sectional=True,
            params={
                'lookbacks': (5, 20, 60),
            },
            description="Cross-sectional momentum rankings"
        )

        return config

    @classmethod
    def from_yaml(cls, path: str) -> 'FeatureConfig':
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            FeatureConfig instance
        """
        import yaml

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        config = cls()
        config.nan_handling = data.get('nan_handling', 'preserve')
        config.min_valid_ratio = data.get('min_valid_ratio', 0.5)

        features_data = data.get('features', {})
        for name, spec_data in features_data.items():
            config.features[name] = FeatureSpec(
                name=name,
                category=spec_data.get('category', 'trend'),
                timeframes=spec_data.get('timeframes', ['D']),
                enabled=spec_data.get('enabled', True),
                params=spec_data.get('params', {}),
                requires_cross_sectional=spec_data.get('requires_cross_sectional', False),
                depends_on=spec_data.get('depends_on', []),
                description=spec_data.get('description', ''),
            )

        return config

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save YAML file
        """
        import yaml

        data = {
            'nan_handling': self.nan_handling,
            'min_valid_ratio': self.min_valid_ratio,
            'features': {}
        }

        for name, spec in self.features.items():
            data['features'][name] = {
                'category': spec.category.value,
                'timeframes': [t.value for t in spec.timeframes],
                'enabled': spec.enabled,
                'params': spec.params,
                'requires_cross_sectional': spec.requires_cross_sectional,
                'depends_on': spec.depends_on,
                'description': spec.description,
            }

        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def get_enabled_features(
        self,
        timeframe: Optional[Timeframe] = None,
        category: Optional[FeatureCategory] = None,
        cross_sectional_only: bool = False,
        single_stock_only: bool = False,
    ) -> List[FeatureSpec]:
        """Get list of enabled features with optional filtering.

        Args:
            timeframe: Filter by timeframe (optional)
            category: Filter by category (optional)
            cross_sectional_only: Only return cross-sectional features
            single_stock_only: Only return single-stock features

        Returns:
            List of matching FeatureSpec objects
        """
        result = []
        for spec in self.features.values():
            if not spec.enabled:
                continue
            if timeframe and timeframe not in spec.timeframes:
                continue
            if category and spec.category != category:
                continue
            if cross_sectional_only and not spec.requires_cross_sectional:
                continue
            if single_stock_only and spec.requires_cross_sectional:
                continue
            result.append(spec)
        return result

    def enable_feature(self, name: str) -> None:
        """Enable a feature by name."""
        if name in self.features:
            self.features[name].enabled = True
        else:
            logger.warning(f"Feature '{name}' not found in configuration")

    def disable_feature(self, name: str) -> None:
        """Disable a feature by name."""
        if name in self.features:
            self.features[name].enabled = False
        else:
            logger.warning(f"Feature '{name}' not found in configuration")

    def enable_category(self, category: FeatureCategory) -> None:
        """Enable all features in a category."""
        for spec in self.features.values():
            if spec.category == category:
                spec.enabled = True

    def disable_category(self, category: FeatureCategory) -> None:
        """Disable all features in a category."""
        for spec in self.features.values():
            if spec.category == category:
                spec.enabled = False

    def enable_all(self) -> None:
        """Enable all features."""
        for spec in self.features.values():
            spec.enabled = True

    def disable_all(self) -> None:
        """Disable all features."""
        for spec in self.features.values():
            spec.enabled = False

    def set_param(self, feature_name: str, param_name: str, value: Any) -> None:
        """Set a parameter for a specific feature.

        Args:
            feature_name: Name of the feature
            param_name: Name of the parameter
            value: New value for the parameter
        """
        if feature_name not in self.features:
            raise ValueError(f"Feature '{feature_name}' not found")
        self.features[feature_name].params[param_name] = value

    def get_param(self, feature_name: str, param_name: str, default: Any = None) -> Any:
        """Get a parameter value for a specific feature.

        Args:
            feature_name: Name of the feature
            param_name: Name of the parameter
            default: Default value if not found

        Returns:
            Parameter value or default
        """
        if feature_name not in self.features:
            return default
        return self.features[feature_name].params.get(param_name, default)

    def summary(self) -> str:
        """Get a summary of the configuration."""
        enabled = [s for s in self.features.values() if s.enabled]
        disabled = [s for s in self.features.values() if not s.enabled]

        lines = [
            f"Feature Configuration Summary",
            f"=" * 40,
            f"Total features: {len(self.features)}",
            f"Enabled: {len(enabled)}",
            f"Disabled: {len(disabled)}",
            f"NaN handling: {self.nan_handling}",
            f"Min valid ratio: {self.min_valid_ratio}",
            f"",
            f"Enabled features by category:",
        ]

        by_category = {}
        for spec in enabled:
            cat = spec.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(spec.name)

        for cat, names in sorted(by_category.items()):
            lines.append(f"  {cat}: {', '.join(names)}")

        return '\n'.join(lines)

    def __repr__(self) -> str:
        enabled = sum(1 for s in self.features.values() if s.enabled)
        return f"FeatureConfig({len(self.features)} features, {enabled} enabled)"


# Convenience function for getting default config
def get_default_feature_config() -> FeatureConfig:
    """Get the default feature configuration."""
    return FeatureConfig.default()
