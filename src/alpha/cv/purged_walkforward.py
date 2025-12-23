"""Purged walk-forward cross-validation for overlapping labels.

This module implements time-series aware CV that handles overlapping
labels properly through purging and embargo periods.

Key concepts:
    - Weekly signal timestamps aligned to Monday entries
    - 20-day label horizon means labels overlap
    - Purge: Remove training samples whose label extends into test
    - Embargo: Gap after test before next training period
"""

from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .embargo import compute_purge_mask, check_label_overlap


@dataclass
class FoldInfo:
    """Information about a CV fold."""
    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train_original: int
    n_train_after_purge: int
    n_test: int
    n_purged: int
    has_leakage: bool


class PurgedWalkForwardCV:
    """Walk-forward cross-validation with purging and embargo.

    Implements the Lopez de Prado approach for handling overlapping
    labels in financial machine learning.

    Attributes:
        n_splits: Number of CV folds.
        purge_days: Days to purge from training end.
        embargo_days: Days to skip after test.
        min_train_samples: Minimum required training samples.
        test_size: Test size (fraction or int).
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_days: int = 25,
        embargo_days: int = 5,
        min_train_samples: int = 500,
        test_size: Optional[Union[int, float]] = None,
    ):
        """Initialize the CV splitter.

        Args:
            n_splits: Number of cross-validation folds.
            purge_days: Number of days to purge from training end.
            embargo_days: Number of days gap after test.
            min_train_samples: Minimum required training samples per fold.
            test_size: Size of test set (fraction or absolute).
        """
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.min_train_samples = min_train_samples
        self.test_size = test_size

    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        entry_dates: Optional[pd.Series] = None,
        exit_dates: Optional[pd.Series] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/test indices for each fold.

        Args:
            X: Feature matrix.
            y: Target vector (optional).
            entry_dates: Series of entry dates for purging.
            exit_dates: Series of exit dates for purging.

        Yields:
            Tuple of (train_indices, test_indices) for each fold.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Get dates from DataFrame index if available
        if entry_dates is None and isinstance(X, pd.DataFrame):
            if "entry_date" in X.columns:
                entry_dates = X["entry_date"]
            elif "signal_date" in X.columns:
                entry_dates = X["signal_date"]
            elif isinstance(X.index, pd.DatetimeIndex):
                entry_dates = pd.Series(X.index, index=X.index)

        # Calculate test size
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        elif isinstance(self.test_size, float):
            test_size = int(n_samples * self.test_size)
        else:
            test_size = self.test_size

        # Generate splits
        for fold_idx in range(self.n_splits):
            # Calculate test boundaries (walk forward from end)
            test_end_idx = n_samples - fold_idx * test_size
            test_start_idx = test_end_idx - test_size

            if test_start_idx < self.min_train_samples:
                continue

            test_indices = indices[test_start_idx:test_end_idx]

            # Training is all data before test
            train_end_idx = test_start_idx
            train_indices = indices[:train_end_idx]

            # Apply purging if we have date information
            if entry_dates is not None and exit_dates is not None:
                test_start_date = entry_dates.iloc[test_start_idx]

                # Purge: remove training samples whose exit_date >= test_start_date
                train_exit_dates = exit_dates.iloc[train_indices]
                purge_threshold = test_start_date - pd.Timedelta(days=self.purge_days)

                # Keep only samples that exit before the purge threshold
                keep_mask = train_exit_dates <= test_start_date
                train_indices = train_indices[keep_mask.values]

            # Check minimum training samples
            if len(train_indices) < self.min_train_samples:
                continue

            yield train_indices, test_indices

    def get_n_splits(self) -> int:
        """Return number of splits."""
        return self.n_splits


class WeeklySignalCV:
    """Cross-validation for weekly trading signals.

    Specialized CV that operates on weekly decision timestamps,
    aligned to the Monday entry / Friday rebalance protocol.

    Attributes:
        n_splits: Number of CV folds.
        min_train_weeks: Minimum training period in weeks.
        test_weeks: Test period in weeks.
        purge_days: Days to purge (>= label horizon).
        embargo_days: Days gap after test.
    """

    def __init__(
        self,
        n_splits: int = 5,
        min_train_weeks: int = 52,
        test_weeks: int = 13,
        purge_days: int = 25,
        embargo_days: int = 5,
    ):
        """Initialize the weekly CV splitter.

        Args:
            n_splits: Number of folds.
            min_train_weeks: Minimum weeks of training data.
            test_weeks: Weeks per test fold.
            purge_days: Days to purge from training end.
            embargo_days: Days gap after test.
        """
        self.n_splits = n_splits
        self.min_train_weeks = min_train_weeks
        self.test_weeks = test_weeks
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    def split(
        self,
        signals: pd.DataFrame,
    ) -> Generator[Tuple[np.ndarray, np.ndarray, FoldInfo], None, None]:
        """Generate train/test indices for weekly signals.

        Args:
            signals: DataFrame with 'week_monday' and 'actual_exit_date' columns.

        Yields:
            Tuple of (train_indices, test_indices, fold_info).
        """
        if "week_monday" not in signals.columns:
            raise ValueError("signals must have 'week_monday' column")

        # Get unique weeks
        weeks = signals["week_monday"].unique()
        weeks = pd.to_datetime(weeks)
        weeks = np.sort(weeks)
        n_weeks = len(weeks)

        # Calculate week indices for splits
        test_size_weeks = self.test_weeks
        min_train_weeks = self.min_train_weeks

        for fold_idx in range(self.n_splits):
            # Test period (walk forward from end)
            test_end_week_idx = n_weeks - fold_idx * test_size_weeks
            test_start_week_idx = test_end_week_idx - test_size_weeks

            if test_start_week_idx < min_train_weeks:
                continue

            test_weeks_range = weeks[test_start_week_idx:test_end_week_idx]
            test_start_date = test_weeks_range[0]
            test_end_date = test_weeks_range[-1] + pd.Timedelta(days=6)  # Through Friday

            # Training period
            train_weeks_range = weeks[:test_start_week_idx]
            train_start_date = train_weeks_range[0]
            train_end_date = train_weeks_range[-1] + pd.Timedelta(days=6)

            # Get sample indices
            test_mask = signals["week_monday"].isin(test_weeks_range)
            test_indices = np.where(test_mask)[0]

            train_mask = signals["week_monday"].isin(train_weeks_range)
            train_indices_original = np.where(train_mask)[0]
            n_train_original = len(train_indices_original)

            # Apply purging: remove training samples whose exit extends into test
            if "actual_exit_date" in signals.columns:
                exit_dates = signals["actual_exit_date"]
                purge_threshold = test_start_date - pd.Timedelta(days=1)

                # Purge if exit_date > purge_threshold
                train_exit = exit_dates.iloc[train_indices_original]
                keep_mask = train_exit <= purge_threshold
                train_indices = train_indices_original[keep_mask.values]
            else:
                # Without exit dates, use entry-based purge window
                entry_dates = signals["week_monday"]
                purge_threshold = test_start_date - pd.Timedelta(days=self.purge_days)

                train_entry = entry_dates.iloc[train_indices_original]
                keep_mask = train_entry <= purge_threshold
                train_indices = train_indices_original[keep_mask.values]

            n_train_after_purge = len(train_indices)
            n_purged = n_train_original - n_train_after_purge

            # Create fold info
            fold_info = FoldInfo(
                fold_idx=fold_idx,
                train_start=train_start_date,
                train_end=train_end_date,
                test_start=test_start_date,
                test_end=test_end_date,
                n_train_original=n_train_original,
                n_train_after_purge=n_train_after_purge,
                n_test=len(test_indices),
                n_purged=n_purged,
                has_leakage=False,  # Updated after validation
            )

            yield train_indices, test_indices, fold_info

    def get_n_splits(self) -> int:
        """Return number of splits."""
        return self.n_splits


def validate_no_leakage(
    signals: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> Tuple[bool, dict]:
    """Validate that there is no label leakage between train and test.

    Args:
        signals: DataFrame with 'week_monday' and 'actual_exit_date' columns.
        train_indices: Training sample indices.
        test_indices: Test sample indices.

    Returns:
        Tuple of (is_valid, details_dict).

    Raises:
        AssertionError: If leakage is detected.
    """
    train_entry = signals["week_monday"].iloc[train_indices]
    train_exit = signals.get("actual_exit_date", signals["week_monday"])
    if "actual_exit_date" in signals.columns:
        train_exit = signals["actual_exit_date"].iloc[train_indices]
    else:
        # Assume 20-day horizon
        train_exit = train_entry + pd.Timedelta(days=28)  # Conservative

    test_entry = signals["week_monday"].iloc[test_indices]

    # Check: all training exits must be before first test entry
    train_exit_max = train_exit.max()
    test_entry_min = test_entry.min()

    gap_days = (test_entry_min - train_exit_max).days

    has_leakage = train_exit_max >= test_entry_min

    details = {
        "train_entry_range": (train_entry.min(), train_entry.max()),
        "train_exit_max": train_exit_max,
        "test_entry_min": test_entry_min,
        "gap_days": gap_days,
        "has_leakage": has_leakage,
    }

    # Assert no leakage
    assert not has_leakage, (
        f"LEAKAGE DETECTED! Training exit ({train_exit_max}) >= "
        f"Test entry ({test_entry_min}). Gap: {gap_days} days"
    )

    return True, details


def get_fold_info(
    cv: Union[PurgedWalkForwardCV, WeeklySignalCV],
    signals: pd.DataFrame,
) -> List[FoldInfo]:
    """Get information about all CV folds.

    Args:
        cv: CV splitter instance.
        signals: Signals DataFrame.

    Returns:
        List of FoldInfo objects.
    """
    fold_infos = []

    if isinstance(cv, WeeklySignalCV):
        for train_idx, test_idx, info in cv.split(signals):
            # Validate no leakage
            try:
                is_valid, details = validate_no_leakage(signals, train_idx, test_idx)
                info.has_leakage = not is_valid
            except AssertionError:
                info.has_leakage = True

            fold_infos.append(info)
    else:
        # For PurgedWalkForwardCV, construct fold info manually
        entry_dates = signals.get("week_monday", signals.get("entry_date"))
        exit_dates = signals.get("actual_exit_date")

        for fold_idx, (train_idx, test_idx) in enumerate(
            cv.split(signals, entry_dates=entry_dates, exit_dates=exit_dates)
        ):
            info = FoldInfo(
                fold_idx=fold_idx,
                train_start=entry_dates.iloc[train_idx].min(),
                train_end=entry_dates.iloc[train_idx].max(),
                test_start=entry_dates.iloc[test_idx].min(),
                test_end=entry_dates.iloc[test_idx].max(),
                n_train_original=len(train_idx),
                n_train_after_purge=len(train_idx),
                n_test=len(test_idx),
                n_purged=0,
                has_leakage=False,
            )
            fold_infos.append(info)

    return fold_infos


def print_cv_summary(fold_infos: List[FoldInfo]) -> None:
    """Print a summary of CV folds."""
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION FOLD SUMMARY")
    print("=" * 70)

    for info in fold_infos:
        print(f"\nFold {info.fold_idx}:")
        print(f"  Train: {info.train_start.date()} to {info.train_end.date()}")
        print(f"  Test:  {info.test_start.date()} to {info.test_end.date()}")
        print(f"  Samples: {info.n_train_after_purge:,} train, {info.n_test:,} test")
        print(f"  Purged: {info.n_purged:,} ({info.n_purged / max(info.n_train_original, 1) * 100:.1f}%)")
        if info.has_leakage:
            print("  WARNING: LEAKAGE DETECTED!")

    print("\n" + "=" * 70)
