"""
Base classes for feature computation.

This module provides abstract base classes that all feature implementations
should inherit from. It ensures a consistent interface across all features.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BaseFeature(ABC):
    """Abstract base class for single-stock feature computation.

    All single-stock feature implementations should inherit from this class
    and implement the `compute` method.

    Attributes:
        name: Unique identifier for this feature
        category: Category for grouping (e.g., 'trend', 'momentum')
        output_columns: List of column names this feature produces
    """

    name: str = "base_feature"
    category: str = "uncategorized"
    output_columns: List[str] = []

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the feature with optional parameters.

        Args:
            params: Dictionary of feature-specific parameters
        """
        self.params = params or {}

    @abstractmethod
    def compute(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Compute features for a single stock's DataFrame.

        This method should add new columns to the DataFrame and return it.
        It should NOT modify the input DataFrame in place.

        Args:
            df: Input DataFrame with OHLCV data and DatetimeIndex
            **kwargs: Additional arguments (e.g., price_col, ret_col)

        Returns:
            DataFrame with new feature columns added
        """
        pass

    def validate_input(self, df: pd.DataFrame) -> bool:
        """Validate that input DataFrame has required columns.

        Override this method to add custom validation logic.

        Args:
            df: Input DataFrame to validate

        Returns:
            True if valid, raises ValueError otherwise
        """
        return True

    def get_output_columns(self) -> List[str]:
        """Return list of column names this feature produces.

        Override this method if output columns are dynamic based on params.

        Returns:
            List of column names
        """
        return self.output_columns

    def get_param(self, name: str, default: Any = None) -> Any:
        """Get a parameter value with optional default.

        Args:
            name: Parameter name
            default: Default value if not found

        Returns:
            Parameter value or default
        """
        return self.params.get(name, default)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, params={self.params})"


class CrossSectionalFeature(BaseFeature):
    """Abstract base class for cross-sectional feature computation.

    Cross-sectional features require data from multiple stocks to compute.
    They typically add context about how a stock compares to its peers.

    The key difference from BaseFeature is that compute_panel() receives
    a dictionary of all symbols' DataFrames and modifies them in place.
    """

    requires_cross_sectional: bool = True

    @abstractmethod
    def compute_panel(
        self,
        indicators_by_symbol: Dict[str, pd.DataFrame],
        **kwargs
    ) -> None:
        """Compute cross-sectional features across all symbols.

        This method modifies the DataFrames in indicators_by_symbol in place.

        Args:
            indicators_by_symbol: Dict mapping symbol -> DataFrame
            **kwargs: Additional arguments (e.g., sectors, sector_to_etf)
        """
        pass

    def compute(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Single-stock compute is not applicable for cross-sectional features.

        Raises:
            NotImplementedError: Always, as cross-sectional features need panel data
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} is a cross-sectional feature. "
            "Use compute_panel() with all symbols' data instead."
        )


class CompositeFeature(BaseFeature):
    """A feature that combines multiple other features.

    Use this to create feature groups that should be computed together
    or to build complex features from simpler ones.
    """

    def __init__(
        self,
        features: List[BaseFeature],
        params: Optional[Dict[str, Any]] = None
    ):
        """Initialize with a list of component features.

        Args:
            features: List of BaseFeature instances to combine
            params: Additional parameters for the composite
        """
        super().__init__(params)
        self.features = features
        self.name = f"composite_{len(features)}"

    def compute(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Compute all component features in sequence.

        Args:
            df: Input DataFrame
            **kwargs: Additional arguments passed to each feature

        Returns:
            DataFrame with all component features added
        """
        result = df.copy()
        for feature in self.features:
            try:
                result = feature.compute(result, **kwargs)
            except Exception as e:
                logger.error(f"Error computing {feature.name}: {e}")
                raise
        return result

    def get_output_columns(self) -> List[str]:
        """Get combined output columns from all component features."""
        columns = []
        for feature in self.features:
            columns.extend(feature.get_output_columns())
        return columns


class FeatureError(Exception):
    """Exception raised when feature computation fails."""
    pass


class ValidationError(FeatureError):
    """Exception raised when input validation fails."""
    pass


def validate_ohlcv(df: pd.DataFrame) -> bool:
    """Validate that DataFrame has standard OHLCV columns.

    Args:
        df: DataFrame to validate

    Returns:
        True if valid

    Raises:
        ValidationError: If required columns are missing
    """
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValidationError(f"Missing required OHLCV columns: {missing}")

    return True


def validate_datetime_index(df: pd.DataFrame) -> bool:
    """Validate that DataFrame has DatetimeIndex.

    Args:
        df: DataFrame to validate

    Returns:
        True if valid

    Raises:
        ValidationError: If index is not DatetimeIndex
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValidationError(
            f"DataFrame index must be DatetimeIndex, got {type(df.index)}"
        )
    return True


def validate_returns(df: pd.DataFrame, ret_col: str = 'ret') -> bool:
    """Validate that DataFrame has returns column.

    Args:
        df: DataFrame to validate
        ret_col: Name of returns column

    Returns:
        True if valid

    Raises:
        ValidationError: If returns column is missing
    """
    if ret_col not in df.columns:
        raise ValidationError(f"Missing returns column: {ret_col}")
    return True
