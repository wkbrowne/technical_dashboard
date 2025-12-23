"""
Model keys for the 4-model system.

This module defines the model keys used to distinguish different trading strategies:
- LONG_NORMAL: Long positions in normal trending conditions (1.5 ATR style)
- LONG_PARABOLIC: Long positions capturing extended moves / persistence
- SHORT_NORMAL: Short positions for breakdown / fragility setups
- SHORT_PARABOLIC: Short positions for panic / regime shift scenarios

Each model key corresponds to a distinct feature set:
- CORE_FEATURES: Shared backbone features across all models
- HEAD_FEATURES[model_key]: Model-specific additive features

Usage:
    from src.config.model_keys import ModelKey

    # Get features for a specific model
    from src.feature_selection.base_features import get_featureset
    features = get_featureset(ModelKey.LONG_NORMAL)
"""

import json
import logging
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


class ModelKey(str, Enum):
    """
    Enumeration of model keys for the 4-model trading system.

    Each model targets a specific market regime / trade type:
    - LONG_NORMAL: Standard long momentum (impulse/transition/gap behavior)
    - LONG_PARABOLIC: Extended momentum / trend persistence plays
    - SHORT_NORMAL: Breakdown / fragility / liquidity stress shorts
    - SHORT_PARABOLIC: Panic / regime shift / vol-of-vol shorts
    """

    LONG_NORMAL = "long_normal"
    LONG_PARABOLIC = "long_parabolic"
    SHORT_NORMAL = "short_normal"
    SHORT_PARABOLIC = "short_parabolic"

    @classmethod
    def all_keys(cls) -> List["ModelKey"]:
        """Return all model keys as a list."""
        return list(cls)

    @classmethod
    def long_keys(cls) -> List["ModelKey"]:
        """Return only long-side model keys."""
        return [cls.LONG_NORMAL, cls.LONG_PARABOLIC]

    @classmethod
    def short_keys(cls) -> List["ModelKey"]:
        """Return only short-side model keys."""
        return [cls.SHORT_NORMAL, cls.SHORT_PARABOLIC]

    @classmethod
    def normal_keys(cls) -> List["ModelKey"]:
        """Return only normal (non-parabolic) model keys."""
        return [cls.LONG_NORMAL, cls.SHORT_NORMAL]

    @classmethod
    def parabolic_keys(cls) -> List["ModelKey"]:
        """Return only parabolic model keys."""
        return [cls.LONG_PARABOLIC, cls.SHORT_PARABOLIC]

    def is_long(self) -> bool:
        """Check if this is a long-side model."""
        return self in (ModelKey.LONG_NORMAL, ModelKey.LONG_PARABOLIC)

    def is_short(self) -> bool:
        """Check if this is a short-side model."""
        return self in (ModelKey.SHORT_NORMAL, ModelKey.SHORT_PARABOLIC)

    def is_parabolic(self) -> bool:
        """Check if this is a parabolic model."""
        return self in (ModelKey.LONG_PARABOLIC, ModelKey.SHORT_PARABOLIC)

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"ModelKey.{self.name}"


# Convenient aliases for common use cases
LONG_NORMAL = ModelKey.LONG_NORMAL
LONG_PARABOLIC = ModelKey.LONG_PARABOLIC
SHORT_NORMAL = ModelKey.SHORT_NORMAL
SHORT_PARABOLIC = ModelKey.SHORT_PARABOLIC

# Default model key (for backwards compatibility)
DEFAULT_MODEL_KEY = ModelKey.LONG_NORMAL


# =============================================================================
# TARGET CONFIGURATIONS - Model-specific triple barrier ATR multiples
# =============================================================================

# Each model has different ATR multiples for target generation:
# - Long models: profit target is UP barrier (up_mult), stop loss is DOWN barrier (dn_mult)
# - Short models: profit target is DOWN barrier (dn_mult), stop loss is UP barrier (up_mult)

TARGET_CONFIGS: Dict[ModelKey, Dict[str, Any]] = {
    # LONG_NORMAL: Standard long momentum, 1.5 ATR profit target
    ModelKey.LONG_NORMAL: {
        "up_mult": 1.5,      # Profit target: 1.5x ATR above entry
        "dn_mult": 1.5,      # Stop loss: 1.5x ATR below entry
        "max_horizon": 20,
        "start_every": 3,
        "target_col": "target_long_normal",
    },
    # LONG_PARABOLIC: Extended momentum, 2.5 ATR profit target (runners)
    ModelKey.LONG_PARABOLIC: {
        "up_mult": 2.5,      # Profit target: 2.5x ATR above entry
        "dn_mult": 1.5,      # Stop loss: 1.5x ATR below entry
        "max_horizon": 20,
        "start_every": 3,
        "target_col": "target_long_parabolic",
    },
    # SHORT_NORMAL: Breakdown/fragility, 2.0 ATR profit target
    ModelKey.SHORT_NORMAL: {
        "up_mult": 1.5,      # Stop loss: 1.5x ATR above entry
        "dn_mult": 2.0,      # Profit target: 2.0x ATR below entry
        "max_horizon": 20,
        "start_every": 3,
        "target_col": "target_short_normal",
    },
    # SHORT_PARABOLIC: Panic/capitulation, 2.5 ATR profit target
    ModelKey.SHORT_PARABOLIC: {
        "up_mult": 1.5,      # Stop loss: 1.5x ATR above entry
        "dn_mult": 2.5,      # Profit target: 2.5x ATR below entry
        "max_horizon": 20,
        "start_every": 3,
        "target_col": "target_short_parabolic",
    },
}


def get_target_config(model_key: ModelKey) -> Dict[str, Any]:
    """Get the target configuration for a specific model.

    Args:
        model_key: The model key to get configuration for

    Returns:
        Dictionary with target configuration (up_mult, dn_mult, max_horizon, etc.)
    """
    return TARGET_CONFIGS[model_key].copy()


def get_all_target_configs() -> Dict[ModelKey, Dict[str, Any]]:
    """Get all target configurations.

    Returns:
        Dictionary mapping ModelKey to target configuration
    """
    return {k: v.copy() for k, v in TARGET_CONFIGS.items()}


def load_target_configs_from_file(
    config_path: Union[str, Path]
) -> Dict[ModelKey, Dict[str, Any]]:
    """
    Load target configurations from a calibration JSON file.

    This allows using custom barrier thresholds from calibrate_barriers.py output
    instead of the hardcoded TARGET_CONFIGS defaults.

    Args:
        config_path: Path to barrier_calibration.json file

    Returns:
        Dictionary mapping ModelKey to target configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid

    Example:
        >>> configs = load_target_configs_from_file("artifacts/targets/barrier_calibration.json")
        >>> # Use with target generation
        >>> for model_key, config in configs.items():
        ...     print(f"{model_key}: up={config['up_mult']}, dn={config['dn_mult']}")
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Target config file not found: {config_path}")

    with open(config_path, 'r') as f:
        data = json.load(f)

    if 'target_configs' not in data:
        raise ValueError(f"Invalid calibration file: missing 'target_configs' key in {config_path}")

    raw_configs = data['target_configs']
    configs: Dict[ModelKey, Dict[str, Any]] = {}

    # Required keys for each config
    required_keys = {'up_mult', 'dn_mult', 'max_horizon', 'start_every'}

    for model_key_str, config in raw_configs.items():
        # Validate config has required keys
        missing = required_keys - set(config.keys())
        if missing:
            raise ValueError(f"Config for {model_key_str} missing required keys: {missing}")

        # Convert string key to ModelKey enum
        try:
            model_key = ModelKey(model_key_str)
        except ValueError:
            logger.warning(f"Unknown model key in config file: {model_key_str}, skipping")
            continue

        # Extract only the standard config keys (exclude calibration_info metadata)
        clean_config = {
            'up_mult': float(config['up_mult']),
            'dn_mult': float(config['dn_mult']),
            'max_horizon': int(config['max_horizon']),
            'start_every': int(config['start_every']),
            'target_col': config.get('target_col', f"target_{model_key_str}"),
        }

        configs[model_key] = clean_config

    if not configs:
        raise ValueError(f"No valid model configurations found in {config_path}")

    # Log loaded configs
    logger.info(f"Loaded {len(configs)} target configs from {config_path}")
    for mk, cfg in configs.items():
        logger.info(f"  {mk.value}: up_mult={cfg['up_mult']}, dn_mult={cfg['dn_mult']}")

    return configs


def override_target_configs(
    config_path: Optional[Union[str, Path]] = None
) -> Dict[ModelKey, Dict[str, Any]]:
    """
    Get target configs, optionally overriding with values from a file.

    This is the recommended way to get target configs when you want to support
    both default and calibrated configurations.

    Args:
        config_path: Optional path to barrier_calibration.json.
                     If None, returns default TARGET_CONFIGS.

    Returns:
        Dictionary mapping ModelKey to target configuration

    Example:
        >>> # Using default configs
        >>> configs = override_target_configs()

        >>> # Using calibrated configs
        >>> configs = override_target_configs("artifacts/targets/barrier_calibration.json")
    """
    if config_path is None:
        return get_all_target_configs()

    return load_target_configs_from_file(config_path)
