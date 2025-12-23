"""Embargo and purge utilities for handling overlapping labels.

This module provides functions to compute embargo and purge masks
that prevent information leakage when labels have overlapping windows.
"""

from typing import Tuple
import numpy as np
import pandas as pd


def compute_purge_mask(
    train_end_date: pd.Timestamp,
    entry_dates: pd.Series,
    exit_dates: pd.Series,
    train_indices: np.ndarray,
) -> np.ndarray:
    """Compute mask for samples to purge from training.

    Purges training samples whose label window overlaps with the
    validation/test period. A sample is purged if its exit_date
    extends beyond the training end date.

    Args:
        train_end_date: Last date in training period.
        entry_dates: Series of entry dates for all samples.
        exit_dates: Series of exit dates for all samples.
        train_indices: Indices of training samples.

    Returns:
        Boolean mask (True = keep, False = purge).
    """
    # Get dates for training samples
    train_entry = entry_dates.iloc[train_indices]
    train_exit = exit_dates.iloc[train_indices]

    # Purge if exit_date > train_end_date
    # This sample's label window extends into the test period
    keep_mask = train_exit <= train_end_date

    return keep_mask.values


def compute_embargo_mask(
    test_start_date: pd.Timestamp,
    test_end_date: pd.Timestamp,
    entry_dates: pd.Series,
    embargo_days: int,
) -> np.ndarray:
    """Compute embargo mask for samples after test period.

    Applies an embargo period after the test set before the next
    training period can begin.

    Args:
        test_start_date: First date in test period.
        test_end_date: Last date in test period.
        entry_dates: Series of entry dates for all samples.
        embargo_days: Number of days to embargo after test.

    Returns:
        Boolean mask (True = in embargo, False = not in embargo).
    """
    embargo_end = test_end_date + pd.Timedelta(days=embargo_days)

    # Samples entered during embargo period
    in_embargo = (entry_dates > test_end_date) & (entry_dates <= embargo_end)

    return in_embargo.values


def check_label_overlap(
    entry_dates: pd.Series,
    exit_dates: pd.Series,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> dict:
    """Check for label overlap between train and test sets.

    Args:
        entry_dates: Series of entry dates.
        exit_dates: Series of exit dates.
        train_indices: Indices of training samples.
        test_indices: Indices of test samples.

    Returns:
        Dict with overlap statistics.
    """
    train_entry = entry_dates.iloc[train_indices].values
    train_exit = exit_dates.iloc[train_indices].values
    test_entry = entry_dates.iloc[test_indices].values
    test_exit = exit_dates.iloc[test_indices].values

    # Find overlaps
    # Training window overlaps with test if:
    # train_entry <= test_exit AND train_exit >= test_entry
    n_overlap = 0
    overlap_indices = []

    test_entry_min = test_entry.min() if len(test_entry) > 0 else None
    test_exit_max = test_exit.max() if len(test_exit) > 0 else None

    if test_entry_min is not None:
        for i, (t_entry, t_exit) in enumerate(zip(train_entry, train_exit)):
            # Check if training label window overlaps with any test window
            if t_exit >= test_entry_min:
                n_overlap += 1
                overlap_indices.append(train_indices[i])

    results = {
        "n_train": len(train_indices),
        "n_test": len(test_indices),
        "n_overlap": n_overlap,
        "overlap_pct": n_overlap / len(train_indices) * 100 if len(train_indices) > 0 else 0,
        "overlap_indices": overlap_indices,
        "has_leakage": n_overlap > 0,
    }

    # Add date ranges
    if len(train_entry) > 0:
        results["train_entry_range"] = (pd.Timestamp(train_entry.min()),
                                         pd.Timestamp(train_entry.max()))
        results["train_exit_range"] = (pd.Timestamp(train_exit.min()),
                                        pd.Timestamp(train_exit.max()))
    if len(test_entry) > 0:
        results["test_entry_range"] = (pd.Timestamp(test_entry.min()),
                                        pd.Timestamp(test_entry.max()))
        results["test_exit_range"] = (pd.Timestamp(test_exit.min()),
                                       pd.Timestamp(test_exit.max()))

    return results


def get_overlap_matrix(
    entry_dates: pd.Series,
    exit_dates: pd.Series,
    sample_size: int = 1000,
) -> np.ndarray:
    """Compute label overlap matrix for a sample of data.

    This is useful for understanding the extent of label overlap
    in the dataset.

    Args:
        entry_dates: Series of entry dates.
        exit_dates: Series of exit dates.
        sample_size: Number of samples to use (for performance).

    Returns:
        Boolean matrix where overlap[i, j] = True if labels overlap.
    """
    n = min(len(entry_dates), sample_size)

    # Sample indices
    if n < len(entry_dates):
        indices = np.random.choice(len(entry_dates), n, replace=False)
        indices = np.sort(indices)
    else:
        indices = np.arange(n)

    entry = entry_dates.iloc[indices].values
    exit = exit_dates.iloc[indices].values

    # Compute pairwise overlap
    # Labels i and j overlap if: entry[i] <= exit[j] AND exit[i] >= entry[j]
    overlap = np.zeros((n, n), dtype=bool)

    for i in range(n):
        for j in range(n):
            if i != j:
                overlap[i, j] = (entry[i] <= exit[j]) and (exit[i] >= entry[j])

    return overlap


def estimate_overlap_density(
    entry_dates: pd.Series,
    exit_dates: pd.Series,
) -> pd.DataFrame:
    """Estimate label overlap density over time.

    Computes the average number of overlapping labels per date.

    Args:
        entry_dates: Series of entry dates.
        exit_dates: Series of exit dates.

    Returns:
        DataFrame with date and overlap count.
    """
    # Get date range
    min_date = entry_dates.min()
    max_date = exit_dates.max()

    dates = pd.date_range(min_date, max_date, freq="D")
    counts = []

    for date in dates:
        # Count labels active on this date
        active = ((entry_dates <= date) & (exit_dates >= date)).sum()
        counts.append({"date": date, "n_active_labels": active})

    df = pd.DataFrame(counts)
    return df
