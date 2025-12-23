"""Feature transforms for position sizing.

All transforms are designed to be fit on training data only and
then applied to validation/test data, preventing lookahead bias.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


class BaseTransform(ABC):
    """Abstract base class for feature transforms."""

    @abstractmethod
    def fit(self, X: pd.DataFrame) -> "BaseTransform":
        """Fit transform on training data."""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply transform to data."""
        pass

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


@dataclass
class ZScoreTransform(BaseTransform):
    """Z-score normalization (mean=0, std=1).

    Attributes:
        columns: Columns to transform.
        clip_std: Number of std devs to clip at (None = no clipping).
        means_: Fitted means per column.
        stds_: Fitted stds per column.
    """
    columns: Optional[List[str]] = None
    clip_std: Optional[float] = 3.0
    means_: Dict[str, float] = field(default_factory=dict)
    stds_: Dict[str, float] = field(default_factory=dict)

    def fit(self, X: pd.DataFrame) -> "ZScoreTransform":
        """Fit z-score parameters on training data."""
        cols = self.columns if self.columns else X.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col in X.columns:
                self.means_[col] = X[col].mean()
                self.stds_[col] = X[col].std()
                # Avoid division by zero
                if self.stds_[col] == 0 or pd.isna(self.stds_[col]):
                    self.stds_[col] = 1.0

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply z-score normalization."""
        X = X.copy()

        for col, mean in self.means_.items():
            if col in X.columns:
                std = self.stds_[col]
                X[col] = (X[col] - mean) / std

                if self.clip_std is not None:
                    X[col] = X[col].clip(-self.clip_std, self.clip_std)

        return X


@dataclass
class RankTransform(BaseTransform):
    """Rank transform (converts to percentile ranks).

    Attributes:
        columns: Columns to transform.
        method: Ranking method ('average', 'min', 'max', 'first').
    """
    columns: Optional[List[str]] = None
    method: str = "average"

    def fit(self, X: pd.DataFrame) -> "RankTransform":
        """No fitting needed for rank transform."""
        if self.columns is None:
            self.columns = X.select_dtypes(include=[np.number]).columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply rank transform (per-date cross-sectional ranks)."""
        X = X.copy()

        if "date" in X.columns or "week_monday" in X.columns:
            # Cross-sectional rank per date
            date_col = "date" if "date" in X.columns else "week_monday"

            for col in self.columns:
                if col in X.columns:
                    X[col] = X.groupby(date_col)[col].rank(
                        method=self.method, pct=True
                    )
        else:
            # Simple rank
            for col in self.columns:
                if col in X.columns:
                    X[col] = X[col].rank(method=self.method, pct=True)

        return X


@dataclass
class WinsorizeTransform(BaseTransform):
    """Winsorize extreme values.

    Attributes:
        columns: Columns to transform.
        lower_quantile: Lower quantile threshold.
        upper_quantile: Upper quantile threshold.
        bounds_: Fitted bounds per column.
    """
    columns: Optional[List[str]] = None
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99
    bounds_: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def fit(self, X: pd.DataFrame) -> "WinsorizeTransform":
        """Fit winsorization bounds on training data."""
        cols = self.columns if self.columns else X.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col in X.columns:
                lower = X[col].quantile(self.lower_quantile)
                upper = X[col].quantile(self.upper_quantile)
                self.bounds_[col] = (lower, upper)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply winsorization."""
        X = X.copy()

        for col, (lower, upper) in self.bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lower, upper)

        return X


@dataclass
class FeatureTransformer:
    """Composite transformer that applies multiple transforms in sequence.

    Attributes:
        transforms: List of transforms to apply.
        feature_columns: Columns to transform (auto-detected if None).
    """
    transforms: List[BaseTransform] = field(default_factory=list)
    feature_columns: Optional[List[str]] = None
    _fitted: bool = False

    def add_transform(self, transform: BaseTransform) -> "FeatureTransformer":
        """Add a transform to the pipeline."""
        self.transforms.append(transform)
        return self

    def fit(self, X: pd.DataFrame) -> "FeatureTransformer":
        """Fit all transforms on training data."""
        # Detect feature columns if not specified
        if self.feature_columns is None:
            exclude = ["symbol", "date", "week_monday", "target", "probability",
                       "entry_price", "exit_price", "hit", "sector"]
            self.feature_columns = [
                col for col in X.select_dtypes(include=[np.number]).columns
                if col not in exclude
            ]

        # Set columns for each transform if not set
        for transform in self.transforms:
            if hasattr(transform, "columns") and transform.columns is None:
                transform.columns = self.feature_columns

        # Fit transforms sequentially
        X_temp = X.copy()
        for transform in self.transforms:
            transform.fit(X_temp)
            X_temp = transform.transform(X_temp)

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all transforms."""
        if not self._fitted:
            raise RuntimeError("FeatureTransformer must be fit before transform")

        X = X.copy()
        for transform in self.transforms:
            X = transform.transform(X)

        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def get_state(self) -> dict:
        """Get fitted state for serialization."""
        state = {
            "feature_columns": self.feature_columns,
            "transforms": [],
        }

        for transform in self.transforms:
            t_state = {
                "type": type(transform).__name__,
            }
            if hasattr(transform, "means_"):
                t_state["means_"] = transform.means_
            if hasattr(transform, "stds_"):
                t_state["stds_"] = transform.stds_
            if hasattr(transform, "bounds_"):
                t_state["bounds_"] = transform.bounds_

            state["transforms"].append(t_state)

        return state


def create_default_transformer() -> FeatureTransformer:
    """Create default transformer pipeline."""
    return FeatureTransformer(
        transforms=[
            WinsorizeTransform(lower_quantile=0.01, upper_quantile=0.99),
            ZScoreTransform(clip_std=3.0),
        ]
    )
