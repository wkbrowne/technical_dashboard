"""Time-series cross-validation with purging and embargo.

This module implements time-series aware CV strategies that prevent
lookahead bias and handle overlapping events properly.
"""

from typing import Generator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .config import CVConfig, CVScheme


class TimeSeriesCV:
    """Time-series cross-validation splitter with purging and embargo.

    Implements rolling and expanding window CV with optional purging
    (removing samples at the end of training that might overlap with test)
    and embargo (gap between train and test).

    This follows the Lopez de Prado approach for handling overlapping
    labels in financial ML applications.

    Attributes:
        config: CVConfig object with split parameters.
    """

    def __init__(self, config: CVConfig):
        """Initialize the CV splitter.

        Args:
            config: CVConfig specifying split parameters.
        """
        self.config = config

    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/test indices for each fold.

        Args:
            X: Feature matrix (used only for length).
            y: Target vector (optional, not used for splitting).
            groups: Group labels (optional, not used for splitting).

        Yields:
            Tuple of (train_indices, test_indices) for each fold.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate sizes
        test_size = self._get_test_size(n_samples)
        train_size = self._get_train_size(n_samples, test_size)

        # Generate splits
        for fold_idx in range(self.config.n_splits):
            train_idx, test_idx = self._get_fold_indices(
                n_samples, fold_idx, train_size, test_size
            )

            if train_idx is not None and test_idx is not None:
                yield train_idx, test_idx

    def _get_test_size(self, n_samples: int) -> int:
        """Calculate test set size."""
        if self.config.test_size is not None:
            if isinstance(self.config.test_size, float):
                return int(n_samples * self.config.test_size)
            return self.config.test_size

        # Default: divide remaining data after min_train among folds
        available = n_samples - self.config.min_train_samples
        return max(1, available // (self.config.n_splits + 1))

    def _get_train_size(self, n_samples: int, test_size: int) -> int:
        """Calculate training set size (for rolling window)."""
        if self.config.train_size is not None:
            if isinstance(self.config.train_size, float):
                return int(n_samples * self.config.train_size)
            return self.config.train_size

        # Default for rolling: use data up to first test set
        if self.config.scheme == CVScheme.ROLLING:
            total_test = test_size * self.config.n_splits
            total_gaps = (self.config.gap + self.config.purge_window) * self.config.n_splits
            return max(
                self.config.min_train_samples,
                n_samples - total_test - total_gaps
            )

        # For expanding: not directly used
        return self.config.min_train_samples

    def _get_fold_indices(
        self,
        n_samples: int,
        fold_idx: int,
        train_size: int,
        test_size: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get train and test indices for a specific fold.

        Args:
            n_samples: Total number of samples.
            fold_idx: Index of the current fold (0-based).
            train_size: Size of training window.
            test_size: Size of test window.

        Returns:
            Tuple of (train_indices, test_indices) or (None, None) if invalid.
        """
        gap = self.config.gap
        purge = self.config.purge_window

        if self.config.scheme == CVScheme.EXPANDING:
            # Expanding window: train on all data before test
            # Test sets are non-overlapping chunks at the end
            test_start = n_samples - (self.config.n_splits - fold_idx) * test_size
            test_end = test_start + test_size

            # Training ends before gap and purge
            train_end = test_start - gap - purge
            train_start = 0

        else:  # Rolling window
            # Fixed-size training window that slides forward
            test_start = n_samples - (self.config.n_splits - fold_idx) * test_size
            test_end = test_start + test_size

            train_end = test_start - gap - purge
            train_start = max(0, train_end - train_size)

        # Validate
        if train_end <= train_start:
            return None, None
        if train_end - train_start < self.config.min_train_samples:
            return None, None
        if test_start >= n_samples:
            return None, None

        train_indices = np.arange(train_start, train_end)
        test_indices = np.arange(test_start, min(test_end, n_samples))

        return train_indices, test_indices

    def get_n_splits(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """Return the number of splits.

        Args:
            X, y, groups: Optional, not used but kept for sklearn compatibility.

        Returns:
            Number of CV splits.
        """
        return self.config.n_splits


class PurgedKFold:
    """K-Fold cross-validation with purging for overlapping labels.

    This is an alternative implementation that focuses on handling
    overlapping labels explicitly via a provided label overlap matrix
    or time-based overlap detection.

    Attributes:
        n_splits: Number of folds.
        purge_window: Number of periods to purge.
        embargo_pct: Percentage of test data to use as embargo.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_window: int = 0,
        embargo_pct: float = 0.0
    ):
        """Initialize PurgedKFold.

        Args:
            n_splits: Number of cross-validation folds.
            purge_window: Number of samples to purge from training end.
            embargo_pct: Percentage of test size to use as embargo after test.
        """
        self.n_splits = n_splits
        self.purge_window = purge_window
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        times: Optional[pd.Series] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate indices for train/test splits with purging.

        Args:
            X: Feature matrix.
            y: Target vector (optional).
            times: Timestamp series for time-based purging (optional).

        Yields:
            Tuple of (train_indices, test_indices).
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Basic fold boundaries (time-ordered)
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits)
        fold_sizes[:n_samples % self.n_splits] += 1

        current = 0
        for fold_idx in range(self.n_splits):
            test_start = current
            test_end = current + fold_sizes[fold_idx]
            current = test_end

            test_indices = indices[test_start:test_end]

            # Build train indices with purging
            train_indices = self._get_train_indices_with_purge(
                indices, test_start, test_end, n_samples
            )

            if len(train_indices) > 0:
                yield train_indices, test_indices

    def _get_train_indices_with_purge(
        self,
        indices: np.ndarray,
        test_start: int,
        test_end: int,
        n_samples: int
    ) -> np.ndarray:
        """Get training indices with purging applied.

        Args:
            indices: All sample indices.
            test_start: Start of test period.
            test_end: End of test period.
            n_samples: Total samples.

        Returns:
            Array of training indices.
        """
        # Calculate embargo size
        test_size = test_end - test_start
        embargo_size = int(test_size * self.embargo_pct)

        # Training before test (with purge)
        train_before_end = max(0, test_start - self.purge_window)
        train_before = indices[:train_before_end]

        # Training after test (with embargo)
        train_after_start = min(n_samples, test_end + embargo_size)
        train_after = indices[train_after_start:]

        return np.concatenate([train_before, train_after])

    def get_n_splits(self) -> int:
        """Return number of splits."""
        return self.n_splits


def create_cv_splitter(config: CVConfig) -> TimeSeriesCV:
    """Factory function to create a CV splitter from config.

    Args:
        config: CVConfig object.

    Returns:
        TimeSeriesCV instance configured according to config.
    """
    return TimeSeriesCV(config)


def get_fold_info(
    cv_splitter: TimeSeriesCV,
    X: pd.DataFrame
) -> List[dict]:
    """Get information about each CV fold.

    Args:
        cv_splitter: CV splitter instance.
        X: Feature DataFrame with DateTimeIndex.

    Returns:
        List of dicts with fold information (train/test sizes, date ranges).
    """
    fold_info = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X)):
        info = {
            'fold': fold_idx,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
        }

        if isinstance(X.index, pd.DatetimeIndex):
            info['train_start'] = X.index[train_idx[0]]
            info['train_end'] = X.index[train_idx[-1]]
            info['test_start'] = X.index[test_idx[0]]
            info['test_end'] = X.index[test_idx[-1]]

        fold_info.append(info)

    return fold_info
