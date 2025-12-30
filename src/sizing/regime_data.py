"""Regime data loading and joining utilities.

This module provides utilities for:
- Loading regime features from feature files
- Joining regime features to predictions
- Validating required regime columns
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from .config import REGIME_FEATURE_COLS, RegimeGatingConfig

logger = logging.getLogger(__name__)


class MissingRegimeFeaturesError(Exception):
    """Raised when required regime features are missing and strict=True."""
    def __init__(self, missing_cols: List[str], available_cols: List[str]):
        self.missing_cols = missing_cols
        self.available_cols = available_cols
        msg = (
            f"Missing required regime features: {missing_cols}. "
            f"Available columns: {available_cols[:20]}..."
        )
        super().__init__(msg)


def find_regime_column(
    df: pd.DataFrame,
    feature_type: str,
    fallback_cols: Optional[List[str]] = None,
) -> Optional[str]:
    """Find the best available column for a regime feature type.

    Args:
        df: DataFrame with regime features.
        feature_type: One of "vix_percentile", "credit_spread", "breadth".
        fallback_cols: Optional list of column names to try.

    Returns:
        Column name if found, None otherwise.
    """
    # Get candidate column names
    candidates = REGIME_FEATURE_COLS.get(feature_type, [])
    if fallback_cols:
        candidates = fallback_cols + candidates

    for col in candidates:
        if col in df.columns:
            return col

    return None


def validate_regime_features(
    df: pd.DataFrame,
    config: RegimeGatingConfig,
    strict: bool = False,
) -> Dict[str, Optional[str]]:
    """Validate that required regime features are present.

    Args:
        df: DataFrame with regime features.
        config: Regime gating configuration.
        strict: If True, raise error for missing features.

    Returns:
        Dict mapping feature_type -> column_name (or None if missing).

    Raises:
        MissingRegimeFeaturesError: If strict=True and features are missing.
    """
    found_cols = {}
    missing_types = []

    # Check VIX column
    vix_col = find_regime_column(df, "vix_percentile")
    found_cols["vix_percentile"] = vix_col
    if vix_col is None and config.vix_high_threshold < 100:
        missing_types.append("vix_percentile")

    # Check credit spread column
    credit_col = find_regime_column(df, "credit_spread")
    found_cols["credit_spread"] = credit_col
    if credit_col is None and config.credit_risk_threshold < 100:
        missing_types.append("credit_spread")

    # Check breadth column
    breadth_col = find_regime_column(df, "breadth")
    found_cols["breadth"] = breadth_col
    if breadth_col is None and config.breadth_poor_threshold > 0:
        missing_types.append("breadth")

    if missing_types:
        if strict or config.strict_missing_features:
            raise MissingRegimeFeaturesError(
                missing_cols=missing_types,
                available_cols=list(df.columns),
            )
        else:
            logger.warning(
                f"Missing regime features (rules will use mult=1.0): {missing_types}"
            )

    return found_cols


def load_regime_features(
    features_path: Union[str, Path],
    date_col: str = "date",
    required_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load regime features from a features file.

    Args:
        features_path: Path to features parquet file.
        date_col: Name of date column.
        required_cols: Optional list of columns to load (loads all if None).

    Returns:
        DataFrame with regime features indexed by date.
    """
    features_path = Path(features_path)

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    # Load features
    df = pd.read_parquet(features_path)
    df.columns = [c.lower() for c in df.columns]

    # Standardize date column
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])

    # If we have symbol column, get unique date-level features
    # (regime features are typically market-level, not stock-level)
    if "symbol" in df.columns:
        # Get regime columns that exist
        regime_cols = []
        for feature_type, candidates in REGIME_FEATURE_COLS.items():
            for col in candidates:
                if col in df.columns:
                    regime_cols.append(col)
                    break

        if regime_cols:
            # Group by date and take first (regime features should be same for all symbols)
            df = df.groupby(date_col)[regime_cols].first().reset_index()

    return df


def join_regime_features(
    predictions: pd.DataFrame,
    regime_df: pd.DataFrame,
    date_col: str = "date",
    strict: bool = False,
) -> pd.DataFrame:
    """Join regime features to predictions DataFrame.

    Args:
        predictions: Predictions DataFrame with date, symbol, p_* columns.
        regime_df: Regime features DataFrame with date and regime columns.
        date_col: Name of date column.
        strict: If True, raise error if regime features are missing for any date.

    Returns:
        Merged DataFrame with predictions and regime features.
    """
    predictions = predictions.copy()
    regime_df = regime_df.copy()

    # Ensure date columns are datetime
    predictions[date_col] = pd.to_datetime(predictions[date_col])
    regime_df[date_col] = pd.to_datetime(regime_df[date_col])

    # Get regime columns (exclude date)
    regime_cols = [c for c in regime_df.columns if c != date_col]

    if not regime_cols:
        logger.warning("No regime columns found in regime_df")
        return predictions

    # Merge on date
    merged = predictions.merge(
        regime_df[[date_col] + regime_cols],
        on=date_col,
        how="left",
    )

    # Check for missing regime data
    n_missing = merged[regime_cols[0]].isna().sum()
    if n_missing > 0:
        pct_missing = 100 * n_missing / len(merged)
        msg = f"{n_missing} rows ({pct_missing:.1f}%) missing regime features"
        if strict:
            raise ValueError(msg)
        else:
            logger.warning(msg)

    return merged


def prepare_regime_row(
    row: pd.Series,
    found_cols: Dict[str, Optional[str]],
) -> Dict[str, float]:
    """Extract regime feature values from a row.

    Args:
        row: Row from DataFrame with regime features.
        found_cols: Dict mapping feature_type -> column_name.

    Returns:
        Dict with regime feature values (NaN if missing).
    """
    values = {}

    for feature_type, col in found_cols.items():
        if col is not None and col in row.index:
            values[feature_type] = row[col]
        else:
            values[feature_type] = float("nan")

    return values


def load_and_join_regime_features(
    predictions: pd.DataFrame,
    features_path: Union[str, Path],
    config: RegimeGatingConfig,
    strict: Optional[bool] = None,
) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    """Load regime features and join to predictions.

    This is the main entry point for regime data preparation.

    Args:
        predictions: Predictions DataFrame.
        features_path: Path to features parquet file.
        config: Regime gating configuration.
        strict: Override config's strict_missing_features setting.

    Returns:
        Tuple of:
        - Merged DataFrame with predictions and regime features
        - Dict mapping feature_type -> column_name
    """
    strict = strict if strict is not None else config.strict_missing_features

    # Load regime features
    regime_df = load_regime_features(features_path)

    # Validate features are present
    found_cols = validate_regime_features(regime_df, config, strict=strict)

    # Join to predictions
    merged = join_regime_features(predictions, regime_df, strict=strict)

    return merged, found_cols


def get_regime_summary(
    df: pd.DataFrame,
    found_cols: Dict[str, Optional[str]],
    config: RegimeGatingConfig,
) -> pd.DataFrame:
    """Get summary of regime feature values and rule activations.

    Args:
        df: DataFrame with regime features.
        found_cols: Dict mapping feature_type -> column_name.
        config: Regime gating configuration.

    Returns:
        DataFrame with per-date regime summary.
    """
    if "date" not in df.columns:
        return pd.DataFrame()

    summary_rows = []

    for date, group in df.groupby("date"):
        row = group.iloc[0]
        summary = {"date": date}

        # VIX
        vix_col = found_cols.get("vix_percentile")
        if vix_col and vix_col in row.index:
            vix_val = row[vix_col]
            summary["vix_percentile"] = vix_val
            summary["vix_triggered"] = vix_val > config.vix_high_threshold

        # Credit
        credit_col = found_cols.get("credit_spread")
        if credit_col and credit_col in row.index:
            credit_val = row[credit_col]
            summary["credit_zscore"] = credit_val
            summary["credit_triggered"] = credit_val > config.credit_risk_threshold

        # Breadth
        breadth_col = found_cols.get("breadth")
        if breadth_col and breadth_col in row.index:
            breadth_val = row[breadth_col]
            summary["breadth_percentile"] = breadth_val
            summary["breadth_triggered"] = breadth_val < config.breadth_poor_threshold

        summary_rows.append(summary)

    return pd.DataFrame(summary_rows)
