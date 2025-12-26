"""
Search space refinement for multi-round HPO.

Provides:
- SearchSpace representation compatible with Optuna
- Quantile-based narrowing of numeric parameters
- Optional importance-based parameter freezing
"""

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
import numpy as np


class ParamType(Enum):
    """Parameter type for search space."""
    INT = "int"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    INT_LOG = "int_log"
    FLOAT_LOG = "float_log"


@dataclass
class SearchSpaceParam:
    """
    Definition of a single hyperparameter in the search space.

    Attributes:
        name: Parameter name
        param_type: Type of parameter (int, float, categorical, etc.)
        low: Lower bound (for numeric params)
        high: Upper bound (for numeric params)
        choices: List of choices (for categorical params)
        log: Whether to sample in log space (for numeric params)
        frozen_value: If set, parameter is frozen to this value
        original_low: Original lower bound before refinement
        original_high: Original upper bound before refinement
    """
    name: str
    param_type: ParamType
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    log: bool = False
    frozen_value: Optional[Any] = None
    original_low: Optional[float] = None
    original_high: Optional[float] = None

    def __post_init__(self):
        """Store original bounds on first creation."""
        if self.original_low is None and self.low is not None:
            self.original_low = self.low
        if self.original_high is None and self.high is not None:
            self.original_high = self.high

    def is_frozen(self) -> bool:
        """Check if parameter is frozen."""
        return self.frozen_value is not None

    def is_numeric(self) -> bool:
        """Check if parameter is numeric."""
        return self.param_type in (
            ParamType.INT, ParamType.FLOAT, ParamType.INT_LOG, ParamType.FLOAT_LOG
        )

    def is_log_scale(self) -> bool:
        """Check if parameter uses log scale."""
        return self.log or self.param_type in (ParamType.INT_LOG, ParamType.FLOAT_LOG)

    def get_range(self) -> float:
        """Get the range of a numeric parameter."""
        if not self.is_numeric() or self.low is None or self.high is None:
            return 0.0
        if self.is_log_scale():
            return math.log(self.high) - math.log(self.low)
        return self.high - self.low

    def get_original_range(self) -> float:
        """Get the original range before refinement."""
        if not self.is_numeric():
            return 0.0
        low = self.original_low if self.original_low is not None else self.low
        high = self.original_high if self.original_high is not None else self.high
        if low is None or high is None:
            return 0.0
        if self.is_log_scale():
            return math.log(high) - math.log(low)
        return high - low

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = {
            'name': self.name,
            'param_type': self.param_type.value,
            'log': self.log,
        }
        if self.low is not None:
            d['low'] = self.low
        if self.high is not None:
            d['high'] = self.high
        if self.choices is not None:
            d['choices'] = self.choices
        if self.frozen_value is not None:
            d['frozen_value'] = self.frozen_value
        if self.original_low is not None:
            d['original_low'] = self.original_low
        if self.original_high is not None:
            d['original_high'] = self.original_high
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SearchSpaceParam":
        """Create from dictionary."""
        return cls(
            name=d['name'],
            param_type=ParamType(d['param_type']),
            low=d.get('low'),
            high=d.get('high'),
            choices=d.get('choices'),
            log=d.get('log', False),
            frozen_value=d.get('frozen_value'),
            original_low=d.get('original_low'),
            original_high=d.get('original_high'),
        )


@dataclass
class SearchSpace:
    """
    Collection of hyperparameters defining the search space.

    Supports serialization and refinement operations.
    """
    params: Dict[str, SearchSpaceParam] = field(default_factory=dict)

    def add_int(
        self,
        name: str,
        low: int,
        high: int,
        log: bool = False,
    ) -> "SearchSpace":
        """Add an integer parameter."""
        self.params[name] = SearchSpaceParam(
            name=name,
            param_type=ParamType.INT_LOG if log else ParamType.INT,
            low=float(low),
            high=float(high),
            log=log,
        )
        return self

    def add_float(
        self,
        name: str,
        low: float,
        high: float,
        log: bool = False,
    ) -> "SearchSpace":
        """Add a float parameter."""
        self.params[name] = SearchSpaceParam(
            name=name,
            param_type=ParamType.FLOAT_LOG if log else ParamType.FLOAT,
            low=low,
            high=high,
            log=log,
        )
        return self

    def add_categorical(
        self,
        name: str,
        choices: List[Any],
    ) -> "SearchSpace":
        """Add a categorical parameter."""
        self.params[name] = SearchSpaceParam(
            name=name,
            param_type=ParamType.CATEGORICAL,
            choices=choices,
        )
        return self

    def freeze(self, name: str, value: Any) -> "SearchSpace":
        """Freeze a parameter to a specific value."""
        if name in self.params:
            self.params[name].frozen_value = value
        return self

    def unfreeze(self, name: str) -> "SearchSpace":
        """Unfreeze a parameter."""
        if name in self.params:
            self.params[name].frozen_value = None
        return self

    def get_frozen_params(self) -> Dict[str, Any]:
        """Get dictionary of frozen parameters and their values."""
        return {
            name: param.frozen_value
            for name, param in self.params.items()
            if param.is_frozen()
        }

    def get_active_params(self) -> Dict[str, SearchSpaceParam]:
        """Get parameters that are not frozen."""
        return {
            name: param
            for name, param in self.params.items()
            if not param.is_frozen()
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'params': {name: param.to_dict() for name, param in self.params.items()}
        }

    def to_json(self, path: str) -> None:
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SearchSpace":
        """Create from dictionary."""
        space = cls()
        for name, param_dict in d.get('params', {}).items():
            space.params[name] = SearchSpaceParam.from_dict(param_dict)
        return space

    @classmethod
    def from_json(cls, path: str) -> "SearchSpace":
        """Load from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def copy(self) -> "SearchSpace":
        """Create a deep copy of the search space."""
        return SearchSpace.from_dict(self.to_dict())


def compute_elite_quantiles(
    values: np.ndarray,
    q_low: float = 0.1,
    q_high: float = 0.9,
    log_scale: bool = False,
) -> Tuple[float, float]:
    """
    Compute quantile bounds from elite trial values.

    Args:
        values: Array of parameter values from elite trials
        q_low: Lower quantile (default 0.1 = 10th percentile)
        q_high: Upper quantile (default 0.9 = 90th percentile)
        log_scale: Whether to compute quantiles in log space

    Returns:
        Tuple of (low_bound, high_bound)
    """
    if len(values) == 0:
        raise ValueError("Cannot compute quantiles from empty values")

    if log_scale:
        # Transform to log space
        log_values = np.log(values[values > 0])
        if len(log_values) == 0:
            raise ValueError("All values <= 0, cannot compute log quantiles")
        low = np.exp(np.quantile(log_values, q_low))
        high = np.exp(np.quantile(log_values, q_high))
    else:
        low = np.quantile(values, q_low)
        high = np.quantile(values, q_high)

    return float(low), float(high)


def refine_search_space(
    search_space: SearchSpace,
    elite_params: List[Dict[str, Any]],
    q_low: float = 0.1,
    q_high: float = 0.9,
    padding: float = 0.10,
    min_range_frac: float = 0.20,
    param_importance: Optional[Dict[str, float]] = None,
    importance_threshold: float = 0.02,
    freeze_low_importance: bool = False,
    allow_singleton_cats: bool = False,
) -> SearchSpace:
    """
    Refine search space based on elite trial parameters.

    For numeric parameters:
    - Compute quantile bounds from elite values
    - Add padding to avoid premature collapse
    - Enforce minimum range relative to original

    For categorical parameters:
    - Keep options that appear in elite trials (unless allow_singleton_cats=False)

    Args:
        search_space: Current search space to refine
        elite_params: List of parameter dictionaries from elite trials
        q_low: Lower quantile for bounds
        q_high: Upper quantile for bounds
        padding: Padding factor to expand range
        min_range_frac: Minimum range as fraction of original
        param_importance: Optional parameter importance scores
        importance_threshold: Threshold below which params may be frozen
        freeze_low_importance: Whether to freeze low-importance params
        allow_singleton_cats: Whether to allow reducing categorical to single option

    Returns:
        Refined SearchSpace
    """
    if not elite_params:
        return search_space.copy()

    refined = search_space.copy()

    for name, param in refined.params.items():
        if param.is_frozen():
            continue  # Skip already frozen params

        # Get values from elite trials
        values = [p[name] for p in elite_params if name in p]
        if not values:
            continue

        # Check importance-based freezing
        if freeze_low_importance and param_importance is not None:
            importance = param_importance.get(name, 0.0)
            if importance < importance_threshold:
                # Freeze to most common value among elite
                if param.param_type == ParamType.CATEGORICAL:
                    # Most common choice
                    from collections import Counter
                    most_common = Counter(values).most_common(1)[0][0]
                    refined.freeze(name, most_common)
                else:
                    # Median value for numeric
                    refined.freeze(name, float(np.median(values)))
                continue

        if param.is_numeric():
            # Numeric parameter refinement
            values_arr = np.array(values, dtype=float)

            # Compute quantile bounds
            try:
                new_low, new_high = compute_elite_quantiles(
                    values_arr,
                    q_low=q_low,
                    q_high=q_high,
                    log_scale=param.is_log_scale(),
                )
            except ValueError:
                continue  # Skip if quantile computation fails

            # Apply padding
            if param.is_log_scale():
                log_range = math.log(new_high) - math.log(new_low)
                padded_range = log_range * (1 + 2 * padding)
                center = (math.log(new_high) + math.log(new_low)) / 2
                new_low = math.exp(center - padded_range / 2)
                new_high = math.exp(center + padded_range / 2)
            else:
                range_size = new_high - new_low
                pad_amount = range_size * padding
                new_low -= pad_amount
                new_high += pad_amount

            # Enforce minimum range
            original_range = param.get_original_range()
            if original_range > 0:
                min_range = original_range * min_range_frac
                current_range = (
                    math.log(new_high) - math.log(new_low) if param.is_log_scale()
                    else new_high - new_low
                )
                if current_range < min_range:
                    # Expand to minimum range
                    expansion = (min_range - current_range) / 2
                    if param.is_log_scale():
                        center = (math.log(new_high) + math.log(new_low)) / 2
                        new_low = math.exp(center - min_range / 2)
                        new_high = math.exp(center + min_range / 2)
                    else:
                        center = (new_high + new_low) / 2
                        new_low = center - min_range / 2
                        new_high = center + min_range / 2

            # Clamp to original bounds
            orig_low = param.original_low if param.original_low is not None else param.low
            orig_high = param.original_high if param.original_high is not None else param.high
            if orig_low is not None:
                new_low = max(new_low, orig_low)
            if orig_high is not None:
                new_high = min(new_high, orig_high)

            # Ensure low < high
            if new_low >= new_high:
                # Revert to original if refinement fails
                new_low = orig_low
                new_high = orig_high

            # Update bounds
            if param.param_type in (ParamType.INT, ParamType.INT_LOG):
                param.low = float(int(new_low))
                param.high = float(int(new_high))
            else:
                param.low = new_low
                param.high = new_high

        elif param.param_type == ParamType.CATEGORICAL:
            # Categorical parameter refinement
            # In simple quantile mode, we don't drop categorical options
            if param_importance is not None and freeze_low_importance:
                # Only drop options if using importance-based refinement
                from collections import Counter
                value_counts = Counter(values)
                present_choices = set(value_counts.keys())

                if param.choices:
                    # Keep choices that appear in elite trials
                    new_choices = [c for c in param.choices if c in present_choices]

                    # Ensure we keep at least 2 choices (or 1 if allow_singleton_cats)
                    min_choices = 1 if allow_singleton_cats else 2
                    if len(new_choices) < min_choices:
                        # Keep top choices by frequency
                        top_choices = [c for c, _ in value_counts.most_common(min_choices)]
                        new_choices = [c for c in param.choices if c in top_choices]

                    if len(new_choices) >= min_choices:
                        param.choices = new_choices

    return refined


def get_default_lgbm_search_space() -> SearchSpace:
    """
    Get the default LightGBM search space.

    Matches the current run_model_tuning.py search space.
    """
    space = SearchSpace()

    # num_leaves via exponent (log-ish sampling)
    space.add_float('num_leaves_exp', low=4.0, high=8.0)

    # min_child_samples (log scale)
    space.add_int('min_child_samples', low=50, high=3000, log=True)

    # max_depth (categorical including -1 for unlimited)
    space.add_categorical('max_depth', choices=[-1, 4, 6, 8, 10])

    # Learning rate (log scale)
    space.add_float('learning_rate', low=0.01, high=0.15, log=True)

    # Regularization (log scale)
    space.add_float('reg_alpha', low=1e-4, high=10.0, log=True)
    space.add_float('reg_lambda', low=1e-4, high=10.0, log=True)

    # min_split_gain (linear)
    space.add_float('min_split_gain', low=0.0, high=0.2)

    # Sampling
    space.add_float('subsample', low=0.5, high=1.0)
    space.add_int('subsample_freq', low=1, high=10)
    space.add_float('colsample_bytree', low=0.5, high=1.0)

    # max_bin (categorical)
    space.add_categorical('max_bin', choices=[63, 127, 255])

    # Class balancing
    space.add_categorical('use_balanced', choices=[True, False])

    return space
