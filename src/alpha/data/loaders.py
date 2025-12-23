"""Data loading utilities for alpha module.

This module provides functions to load and prepare data for the
position sizing and backtesting pipeline.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .schemas import SignalRecord


def load_targets(
    path: Union[str, Path] = "artifacts/targets_triple_barrier.parquet",
    exclude_neutral: bool = True,
) -> pd.DataFrame:
    """Load triple barrier targets.

    Args:
        path: Path to targets parquet file.
        exclude_neutral: Whether to exclude neutral (hit=0) labels.

    Returns:
        DataFrame with columns:
            - symbol, t0 (entry date), t_hit (exit date)
            - hit (-1, 0, 1), entry_px, price_hit
            - ret_from_entry, h_used (holding days)
            - weight_final
    """
    df = pd.read_parquet(path)

    # Standardize column names
    df = df.rename(columns={
        "t0": "entry_date",
        "t_hit": "exit_date",
        "entry_px": "entry_price",
        "price_hit": "exit_price",
    })

    # Convert dates
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])

    # Exclude neutral if requested
    if exclude_neutral:
        df = df[df["hit"] != 0].copy()

    # Binary target (1 = upper barrier hit)
    df["target"] = (df["hit"] == 1).astype(int)

    # Sort by date
    df = df.sort_values(["entry_date", "symbol"]).reset_index(drop=True)

    return df


def load_features(
    path: Union[str, Path] = "artifacts/features_complete.parquet",
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load features.

    Args:
        path: Path to features parquet file.
        feature_names: Optional list of specific features to load.

    Returns:
        DataFrame with features.
    """
    df = pd.read_parquet(path)

    # Standardize date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # Filter to specific features if requested
    if feature_names is not None:
        required_cols = ["symbol", "date"]
        available = [f for f in feature_names if f in df.columns]
        missing = set(feature_names) - set(available)
        if missing:
            print(f"Warning: {len(missing)} features not found: {list(missing)[:5]}")
        df = df[required_cols + available].copy()

    return df


def load_predictions(
    path: Union[str, Path] = "artifacts/predictions/predictions_latest.parquet",
) -> pd.DataFrame:
    """Load model predictions.

    Args:
        path: Path to predictions file.

    Returns:
        DataFrame with predictions.
    """
    df = pd.read_parquet(path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # Ensure probability column exists
    if "probability" not in df.columns and "prob" in df.columns:
        df["probability"] = df["prob"]

    return df


def load_prices(
    path: Union[str, Path] = "cache/stock_data_combined.parquet",
    symbols: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load OHLCV price data.

    Args:
        path: Path to price data.
        symbols: Optional list of symbols to filter.
        start_date: Optional start date filter.
        end_date: Optional end date filter.

    Returns:
        DataFrame with OHLCV data.
    """
    df = pd.read_parquet(path)

    # Lowercase columns
    df.columns = [c.lower() for c in df.columns]

    # Standardize date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # Apply filters
    if symbols is not None:
        df = df[df["symbol"].isin(symbols)]
    if start_date is not None:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    return df


def get_monday_dates(
    start_date: Union[str, pd.Timestamp],
    end_date: Union[str, pd.Timestamp],
) -> pd.DatetimeIndex:
    """Get all Monday dates in a range.

    Args:
        start_date: Start of range.
        end_date: End of range.

    Returns:
        DatetimeIndex of Mondays.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="W-MON")
    return dates


def get_friday_dates(
    start_date: Union[str, pd.Timestamp],
    end_date: Union[str, pd.Timestamp],
) -> pd.DatetimeIndex:
    """Get all Friday dates in a range.

    Args:
        start_date: Start of range.
        end_date: End of range.

    Returns:
        DatetimeIndex of Fridays.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="W-FRI")
    return dates


def get_week_bounds(
    date: pd.Timestamp,
    prices: pd.DataFrame,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Get actual trading week bounds for a given date.

    Handles holidays by finding nearest actual trading days.

    Args:
        date: Any date within the week.
        prices: Price data to check for trading days.

    Returns:
        Tuple of (monday, friday) trading days.
    """
    # Get Monday of the week
    days_since_monday = date.weekday()
    monday = date - pd.Timedelta(days=days_since_monday)

    # Get Friday of the week
    friday = monday + pd.Timedelta(days=4)

    # Get available trading dates
    trading_dates = prices["date"].unique()
    trading_dates = pd.to_datetime(trading_dates)

    # Find actual trading Monday (or next day)
    week_start = monday
    while week_start not in trading_dates and week_start <= friday:
        week_start += pd.Timedelta(days=1)

    # Find actual trading Friday (or previous day)
    week_end = friday
    while week_end not in trading_dates and week_end >= monday:
        week_end -= pd.Timedelta(days=1)

    return week_start, week_end


def prepare_weekly_signals(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
    features: Optional[pd.DataFrame] = None,
    sector_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Prepare weekly trading signals from predictions.

    Aligns predictions with targets to create tradeable signals.

    IMPORTANT: Only predictions with matching (symbol, date) in targets are
    included. This is by design - backtesting requires known outcomes from
    the targets file. Predictions beyond the last target date are naturally
    excluded since they have no ground truth for evaluation.

    Args:
        predictions: Model predictions with probability scores.
            Required columns: symbol, date (or signal_date), probability
        targets: Triple barrier targets (from load_targets).
            Must have entry_date column with known outcomes.
        features: Optional features for sizing (ATR, volume, etc.).
        sector_map: Optional symbol -> sector mapping.

    Returns:
        DataFrame with weekly signals (empty if no date overlap):
            - week_monday: Signal date (Monday)
            - symbol, probability
            - entry_price, target_price, stop_price
            - actual_exit_date, actual_exit_price, actual_return
            - hit (outcome: 1, -1, or 0)
    """
    # Merge predictions with targets
    predictions = predictions.copy()
    targets = targets.copy()

    # Rename for clarity
    if "date" in predictions.columns:
        predictions["signal_date"] = predictions["date"]

    if "entry_date" not in targets.columns and "t0" in targets.columns:
        targets["entry_date"] = targets["t0"]

    # Drop columns from predictions that will come from targets to avoid conflicts
    target_provided_cols = ["target_price", "stop_price", "entry_price"]
    for col in target_provided_cols:
        if col in predictions.columns:
            predictions = predictions.drop(columns=[col])

    # Merge on symbol and date
    merged = predictions.merge(
        targets[["symbol", "entry_date", "exit_date", "entry_price", "top", "bot",
                 "exit_price", "ret_from_entry", "hit", "h_used", "weight_final"]],
        left_on=["symbol", "signal_date"],
        right_on=["symbol", "entry_date"],
        how="inner",
    )

    # Rename columns
    merged = merged.rename(columns={
        "top": "target_price",
        "bot": "stop_price",
        "exit_date": "actual_exit_date",
        "exit_price": "actual_exit_price",
        "ret_from_entry": "actual_return",
        "h_used": "holding_days",
    })

    # Get week Monday for each signal
    merged["week_monday"] = merged["signal_date"].apply(
        lambda x: x - pd.Timedelta(days=x.weekday())
    )

    # Add ATR - check predictions first, then features
    if "atr_percent" in merged.columns:
        # ATR already in predictions/merged data
        merged["atr_pct"] = merged["atr_percent"]
    elif features is not None and "atr_percent" in features.columns:
        # Merge ATR from features
        atr_data = features[["symbol", "date", "atr_percent"]].copy()
        merged = merged.merge(
            atr_data,
            left_on=["symbol", "signal_date"],
            right_on=["symbol", "date"],
            how="left",
            suffixes=("", "_feat"),
        )
        # Use the feature ATR (may be named atr_percent or atr_percent_feat)
        atr_col = "atr_percent_feat" if "atr_percent_feat" in merged.columns else "atr_percent"
        merged["atr_pct"] = merged[atr_col]
    else:
        # Estimate ATR from barrier levels
        merged["atr_pct"] = (merged["target_price"] / merged["entry_price"] - 1) / 3.0

    # Add sector if provided
    if sector_map is not None:
        merged["sector"] = merged["symbol"].map(sector_map)
    else:
        merged["sector"] = "Unknown"

    # Select and order columns
    output_cols = [
        "week_monday", "signal_date", "symbol", "probability",
        "entry_price", "target_price", "stop_price",
        "actual_exit_date", "actual_exit_price", "actual_return",
        "hit", "holding_days", "atr_pct", "sector", "weight_final",
    ]

    available_cols = [c for c in output_cols if c in merged.columns]
    result = merged[available_cols].copy()

    # Sort by week and probability
    result = result.sort_values(
        ["week_monday", "probability"],
        ascending=[True, False]
    ).reset_index(drop=True)

    return result


def merge_features_for_sizing(
    signals: pd.DataFrame,
    features: pd.DataFrame,
    sizing_features: List[str],
) -> pd.DataFrame:
    """Merge sizing-relevant features with signals.

    Args:
        signals: Weekly signals DataFrame.
        features: Full features DataFrame.
        sizing_features: List of feature names to merge.

    Returns:
        Signals with sizing features added.
    """
    # Check which features are available
    available = [f for f in sizing_features if f in features.columns]

    if not available:
        return signals

    # Subset features
    feat_subset = features[["symbol", "date"] + available].copy()

    # Merge on signal date
    merged = signals.merge(
        feat_subset,
        left_on=["symbol", "signal_date"],
        right_on=["symbol", "date"],
        how="left",
    )

    # Drop duplicate date column if exists
    if "date" in merged.columns:
        merged = merged.drop(columns=["date"])

    return merged


def validate_signal_data(signals: pd.DataFrame) -> Dict[str, any]:
    """Validate signal data for common issues.

    Args:
        signals: Prepared signals DataFrame.

    Returns:
        Dict with validation results.
    """
    results = {
        "n_signals": len(signals),
        "n_symbols": signals["symbol"].nunique(),
        "date_range": (signals["week_monday"].min(), signals["week_monday"].max()),
        "issues": [],
    }

    # Check for missing values
    critical_cols = ["symbol", "week_monday", "probability", "entry_price"]
    for col in critical_cols:
        if col in signals.columns:
            n_missing = signals[col].isna().sum()
            if n_missing > 0:
                results["issues"].append(f"{col}: {n_missing} missing values")

    # Check probability range
    if "probability" in signals.columns:
        probs = signals["probability"]
        if probs.min() < 0 or probs.max() > 1:
            results["issues"].append(
                f"probability out of range: [{probs.min():.3f}, {probs.max():.3f}]"
            )

    # Check for duplicate signals
    if "symbol" in signals.columns and "week_monday" in signals.columns:
        n_dupes = signals.duplicated(subset=["symbol", "week_monday"]).sum()
        if n_dupes > 0:
            results["issues"].append(f"{n_dupes} duplicate (symbol, week) pairs")

    # Check price validity
    if "entry_price" in signals.columns:
        n_zero = (signals["entry_price"] <= 0).sum()
        if n_zero > 0:
            results["issues"].append(f"{n_zero} signals with entry_price <= 0")

    results["is_valid"] = len(results["issues"]) == 0

    return results
