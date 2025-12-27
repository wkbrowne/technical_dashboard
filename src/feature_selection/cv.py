"""Time-series cross-validation with purging and embargo.

This module implements time-series aware CV strategies that prevent
lookahead bias and handle overlapping events properly.

CRITICAL: For panel data (multiple symbols per date), we split by DATE
to prevent cross-sectional leakage. All symbols on a given date go to
train OR test, never both.
"""

from typing import Generator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .config import CVConfig, CVScheme


class TimeSeriesCV:
    """Time-series cross-validation splitter with purging and embargo.

    For panel data (multiple rows per date), this splitter groups by date
    to prevent cross-sectional leakage.

    Attributes:
        config: CVConfig object with split parameters.
    """

    def __init__(self, config: CVConfig, verbose: bool = True):
        """Initialize the CV splitter.

        Args:
            config: CVConfig specifying split parameters.
            verbose: Whether to print CV split information.
        """
        self.config = config
        self.verbose = verbose
        self._logged_split_info = False

    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/test indices for each fold.

        For DataFrames with date index, splits by DATE to prevent
        cross-sectional leakage. All symbols on a given date go to train OR test.

        Args:
            X: Feature matrix. If DataFrame with date index, splits by date.
            y: Target vector (optional, not used for splitting).
            groups: Optional explicit group labels for splitting.

        Yields:
            Tuple of (train_indices, test_indices) for each fold.
        """
        if isinstance(X, pd.DataFrame):
            # Check if index looks like dates (has duplicates = panel data)
            n_unique = X.index.nunique()
            n_total = len(X)

            if n_unique < n_total * 0.9:  # Many duplicates = panel data
                yield from self._split_by_date_groups(X)
            else:
                # Single time series or nearly unique indices
                yield from self._split_sequential(n_total)
        else:
            yield from self._split_sequential(len(X))

    def _split_by_date_groups(
        self,
        X: pd.DataFrame
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Walk-forward split by date groups for panel data.

        This is the correct approach for panel data:
        1. Group all rows by their date
        2. Split DATES into train/test periods (not rows)
        3. Apply embargo gap in terms of dates
        4. Map back to row indices
        """
        # Build date -> row indices mapping
        date_groups = {}
        for row_idx, date_val in enumerate(X.index):
            if date_val not in date_groups:
                date_groups[date_val] = []
            date_groups[date_val].append(row_idx)

        # Get sorted unique dates
        unique_dates = sorted(date_groups.keys())
        n_dates = len(unique_dates)

        if self.verbose and not self._logged_split_info:
            avg_rows_per_date = len(X) / n_dates
            print(f"\n  CV: Panel data detected - splitting by DATE")
            print(f"  CV: {n_dates} unique dates, {len(X)} rows ({avg_rows_per_date:.0f} rows/date)")
            print(f"  CV: Embargo gap = {self.config.gap} dates, Purge = {self.config.purge_window} dates")
            self._logged_split_info = True

        # Calculate test size in dates
        test_size_dates = n_dates // (self.config.n_splits + 1)
        if test_size_dates < 1:
            test_size_dates = 1

        gap = self.config.gap
        purge = self.config.purge_window

        # Generate walk-forward folds
        for fold_idx in range(self.config.n_splits):
            # Test period: non-overlapping chunks at the end
            test_end_idx = n_dates - (self.config.n_splits - fold_idx - 1) * test_size_dates
            test_start_idx = test_end_idx - test_size_dates

            # Training ends before gap and purge
            train_end_idx = test_start_idx - gap - purge

            if self.config.scheme == CVScheme.EXPANDING:
                train_start_idx = 0
            else:  # Rolling
                # For rolling, use a fixed training window
                min_train = max(self.config.min_train_samples // (len(X) // n_dates), 50)
                train_start_idx = max(0, train_end_idx - min_train)

            # Validate
            if train_end_idx <= train_start_idx:
                continue
            if test_start_idx >= n_dates or test_end_idx > n_dates:
                continue

            # Get date ranges
            train_dates = unique_dates[train_start_idx:train_end_idx]
            test_dates = unique_dates[test_start_idx:test_end_idx]

            # Map dates to row indices
            train_rows = []
            for d in train_dates:
                train_rows.extend(date_groups[d])

            test_rows = []
            for d in test_dates:
                test_rows.extend(date_groups[d])

            if len(train_rows) > 0 and len(test_rows) > 0:
                # Verify no overlap
                train_set = set(train_rows)
                test_set = set(test_rows)
                assert train_set.isdisjoint(test_set), "Train/test row overlap detected!"

                if self.verbose and fold_idx == 0:
                    print(f"  CV: Fold 0: train_dates={len(train_dates)} ({train_dates[0]} to {train_dates[-1]})")
                    print(f"  CV: Fold 0: test_dates={len(test_dates)} ({test_dates[0]} to {test_dates[-1]})")
                    print(f"  CV: Fold 0: train_rows={len(train_rows)}, test_rows={len(test_rows)}")

                yield np.array(train_rows), np.array(test_rows)

    def _split_sequential(
        self,
        n_samples: int
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Sequential walk-forward split for single time series."""
        if self.verbose and not self._logged_split_info:
            print(f"\n  CV: Sequential splitting ({n_samples} samples)")
            self._logged_split_info = True

        test_size = n_samples // (self.config.n_splits + 1)
        if test_size < 1:
            test_size = 1

        gap = self.config.gap
        purge = self.config.purge_window

        for fold_idx in range(self.config.n_splits):
            test_end = n_samples - (self.config.n_splits - fold_idx - 1) * test_size
            test_start = test_end - test_size

            train_end = test_start - gap - purge

            if self.config.scheme == CVScheme.EXPANDING:
                train_start = 0
            else:
                train_start = max(0, train_end - self.config.min_train_samples)

            if train_end <= train_start:
                continue
            if test_start >= n_samples:
                continue

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, min(test_end, n_samples))

            if len(train_indices) >= self.config.min_train_samples and len(test_indices) > 0:
                yield train_indices, test_indices

    def get_n_splits(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """Return the number of splits."""
        return self.config.n_splits


class PurgedGroupTimeSeriesSplit:
    """Walk-forward CV with explicit group-based splitting.

    Use this when you want explicit control over grouping.
    Pass groups=dates to split by date.
    """

    def __init__(
        self,
        n_splits: int = 5,
        gap: int = 0,
        purge: int = 0,
        expanding: bool = True
    ):
        self.n_splits = n_splits
        self.gap = gap
        self.purge = purge
        self.expanding = expanding

    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Split with explicit groups."""
        if groups is None:
            raise ValueError("groups must be provided (e.g., dates)")

        # Get unique groups in order
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        # Map groups to row indices
        group_to_rows = {g: [] for g in unique_groups}
        for row_idx, g in enumerate(groups):
            group_to_rows[g].append(row_idx)

        test_size = n_groups // (self.n_splits + 1)

        for fold_idx in range(self.n_splits):
            test_end_idx = n_groups - (self.n_splits - fold_idx - 1) * test_size
            test_start_idx = test_end_idx - test_size
            train_end_idx = test_start_idx - self.gap - self.purge

            if self.expanding:
                train_start_idx = 0
            else:
                train_start_idx = max(0, train_end_idx - test_size * 2)

            if train_end_idx <= train_start_idx:
                continue

            train_groups = unique_groups[train_start_idx:train_end_idx]
            test_groups = unique_groups[test_start_idx:test_end_idx]

            train_rows = []
            for g in train_groups:
                train_rows.extend(group_to_rows[g])

            test_rows = []
            for g in test_groups:
                test_rows.extend(group_to_rows[g])

            if len(train_rows) > 0 and len(test_rows) > 0:
                yield np.array(train_rows), np.array(test_rows)


class PurgedKFold:
    """K-Fold cross-validation with purging for overlapping labels.

    This is an alternative implementation that focuses on handling
    overlapping labels explicitly via a provided label overlap matrix
    or time-based overlap detection.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_window: int = 0,
        embargo_pct: float = 0.0
    ):
        self.n_splits = n_splits
        self.purge_window = purge_window
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        times: Optional[pd.Series] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate indices for train/test splits with purging."""
        n_samples = len(X)
        indices = np.arange(n_samples)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits)
        fold_sizes[:n_samples % self.n_splits] += 1

        current = 0
        for fold_idx in range(self.n_splits):
            test_start = current
            test_end = current + fold_sizes[fold_idx]
            current = test_end

            test_indices = indices[test_start:test_end]

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
        """Get training indices with purging applied."""
        test_size = test_end - test_start
        embargo_size = int(test_size * self.embargo_pct)

        train_before_end = max(0, test_start - self.purge_window)
        train_before = indices[:train_before_end]

        train_after_start = min(n_samples, test_end + embargo_size)
        train_after = indices[train_after_start:]

        return np.concatenate([train_before, train_after])

    def get_n_splits(self) -> int:
        return self.n_splits


def create_cv_splitter(config: CVConfig, verbose: bool = True) -> TimeSeriesCV:
    """Factory function to create a CV splitter from config."""
    return TimeSeriesCV(config, verbose=verbose)


def get_fold_info(
    cv_splitter: TimeSeriesCV,
    X: pd.DataFrame
) -> List[dict]:
    """Get information about each CV fold."""
    fold_info = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X)):
        info = {
            'fold': fold_idx,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
        }

        # Get date ranges
        train_dates = X.index[train_idx]
        test_dates = X.index[test_idx]

        info['train_start'] = train_dates.min()
        info['train_end'] = train_dates.max()
        info['test_start'] = test_dates.min()
        info['test_end'] = test_dates.max()
        info['train_unique_dates'] = train_dates.nunique()
        info['test_unique_dates'] = test_dates.nunique()

        fold_info.append(info)

    return fold_info


def verify_no_leakage(X: pd.DataFrame, cv_splitter: TimeSeriesCV) -> bool:
    """Verify that train and test sets don't share any dates.

    Returns True if no leakage detected.
    """
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X)):
        train_dates = set(X.index[train_idx])
        test_dates = set(X.index[test_idx])

        overlap = train_dates & test_dates
        if overlap:
            print(f"LEAKAGE DETECTED in fold {fold_idx}!")
            print(f"  Overlapping dates: {sorted(list(overlap))[:5]}...")
            return False

        # Also verify temporal ordering
        if max(train_dates) >= min(test_dates):
            print(f"TEMPORAL VIOLATION in fold {fold_idx}!")
            print(f"  Train max: {max(train_dates)}, Test min: {min(test_dates)}")
            return False

    print("No leakage detected - train/test dates are disjoint and properly ordered.")
    return True
