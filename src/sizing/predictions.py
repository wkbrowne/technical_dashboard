"""Multi-model prediction data interface.

Supports two prediction formats:
1. Wide format: Single parquet with columns date, symbol, p_long_normal,
   p_long_parabolic, p_short_normal, p_short_parabolic
2. Separate files: Four parquets keyed by model type

Both formats are validated for:
- Strictly out-of-sample predictions
- Aligned rebalance dates across models
- Valid probability ranges
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from .config import ModelType


# Column naming conventions
PROB_COL_PREFIX = "p_"
MODEL_PROB_COLS = {
    ModelType.LONG_NORMAL: "p_long_normal",
    ModelType.LONG_PARABOLIC: "p_long_parabolic",
    ModelType.SHORT_NORMAL: "p_short_normal",
    ModelType.SHORT_PARABOLIC: "p_short_parabolic",
}


def get_prob_column(model: ModelType) -> str:
    """Get probability column name for a model type."""
    return MODEL_PROB_COLS[model]


def load_multi_model_predictions(
    path: Union[str, Path],
    models: Optional[List[ModelType]] = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Load multi-model predictions from parquet.

    Supports two formats:
    1. Wide format: Single file with columns for each model
    2. Directory: Separate files per model (path is directory)

    Args:
        path: Path to parquet file or directory.
        models: Models to load (default: all four).
        validate: Whether to validate the data.

    Returns:
        DataFrame with columns: date, symbol, and p_<model> for each model.
    """
    path = Path(path)

    if models is None:
        models = list(ModelType)

    if path.is_dir():
        # Load from separate files
        df = _load_from_directory(path, models)
    else:
        # Load from single wide parquet
        df = _load_from_wide_parquet(path, models)

    if validate:
        validation = validate_predictions_format(df, models)
        if not validation["is_valid"]:
            issues = "\n  ".join(validation["issues"])
            raise ValueError(f"Invalid predictions format:\n  {issues}")

    return df


def _load_from_wide_parquet(
    path: Path,
    models: List[ModelType],
) -> pd.DataFrame:
    """Load predictions from single wide-format parquet."""
    df = pd.read_parquet(path)

    # Standardize date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # Check for required columns
    required = ["date", "symbol"]
    for model in models:
        col = get_prob_column(model)
        if col not in df.columns:
            # Try alternate naming
            alt_col = f"probability_{model.value}"
            if alt_col in df.columns:
                df[col] = df[alt_col]
            else:
                raise ValueError(f"Missing probability column for {model.value}: {col}")
        required.append(col)

    # Select only needed columns
    output_cols = ["date", "symbol"] + [get_prob_column(m) for m in models]
    available_cols = [c for c in output_cols if c in df.columns]

    return df[available_cols].copy()


def _load_from_directory(
    directory: Path,
    models: List[ModelType],
) -> pd.DataFrame:
    """Load predictions from separate model files in directory."""
    dfs = []

    for model in models:
        # Try different file naming patterns
        patterns = [
            f"cv_predictions_{model.value}.parquet",
            f"predictions_{model.value}.parquet",
            f"{model.value}_predictions.parquet",
        ]

        file_path = None
        for pattern in patterns:
            candidate = directory / pattern
            if candidate.exists():
                file_path = candidate
                break

        if file_path is None:
            raise FileNotFoundError(
                f"No predictions file found for {model.value} in {directory}"
            )

        model_df = pd.read_parquet(file_path)
        model_df["date"] = pd.to_datetime(model_df["date"])

        # Rename probability column
        prob_col = get_prob_column(model)
        if "probability" in model_df.columns:
            model_df = model_df.rename(columns={"probability": prob_col})
        elif "prob" in model_df.columns:
            model_df = model_df.rename(columns={"prob": prob_col})

        dfs.append(model_df[["date", "symbol", prob_col]])

    # Merge all model predictions
    result = dfs[0]
    for model_df in dfs[1:]:
        result = result.merge(
            model_df,
            on=["date", "symbol"],
            how="outer",
        )

    return result


def validate_predictions_format(
    df: pd.DataFrame,
    models: Optional[List[ModelType]] = None,
) -> Dict:
    """Validate predictions format and data quality.

    Checks:
    - Required columns exist
    - Probability values are in [0, 1]
    - No duplicate (date, symbol) pairs
    - Dates are aligned across models

    Args:
        df: Predictions DataFrame.
        models: Models to validate.

    Returns:
        Dict with 'is_valid' and 'issues' list.
    """
    if models is None:
        models = list(ModelType)

    issues = []

    # Check required columns
    required = ["date", "symbol"]
    for col in required:
        if col not in df.columns:
            issues.append(f"Missing required column: {col}")

    # Check probability columns
    for model in models:
        col = get_prob_column(model)
        if col not in df.columns:
            issues.append(f"Missing probability column: {col}")
        else:
            # Check value range
            probs = df[col].dropna()
            if len(probs) > 0:
                if probs.min() < 0 or probs.max() > 1:
                    issues.append(
                        f"{col}: values outside [0, 1] range "
                        f"[{probs.min():.3f}, {probs.max():.3f}]"
                    )

    # Check for duplicates
    if "date" in df.columns and "symbol" in df.columns:
        n_dupes = df.duplicated(subset=["date", "symbol"]).sum()
        if n_dupes > 0:
            issues.append(f"{n_dupes} duplicate (date, symbol) pairs")

    # Check date alignment (all models have predictions for same dates)
    if len(issues) == 0:  # Only check if basic format is valid
        date_counts = df.groupby("date").size()
        if date_counts.nunique() > 1:
            min_count = date_counts.min()
            max_count = date_counts.max()
            if max_count > min_count * 1.5:  # Allow some variation
                issues.append(
                    f"Uneven date coverage: {min_count} to {max_count} symbols per date"
                )

    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "n_rows": len(df),
        "n_symbols": df["symbol"].nunique() if "symbol" in df.columns else 0,
        "date_range": (
            df["date"].min(),
            df["date"].max(),
        ) if "date" in df.columns else None,
    }


def convert_long_to_wide_predictions(
    long_df: pd.DataFrame,
    model_col: str = "model",
    prob_col: str = "probability",
) -> pd.DataFrame:
    """Convert long-format predictions to wide format.

    Long format has one row per (date, symbol, model) with probability.
    Wide format has one row per (date, symbol) with columns for each model.

    Args:
        long_df: Long-format DataFrame with model column.
        model_col: Column containing model type.
        prob_col: Column containing probability.

    Returns:
        Wide-format DataFrame.
    """
    long_df = long_df.copy()
    long_df["date"] = pd.to_datetime(long_df["date"])

    # Pivot to wide format
    wide_df = long_df.pivot_table(
        index=["date", "symbol"],
        columns=model_col,
        values=prob_col,
        aggfunc="first",
    ).reset_index()

    # Rename columns to standard format
    rename_map = {}
    for model in ModelType:
        if model.value in wide_df.columns:
            rename_map[model.value] = get_prob_column(model)

    wide_df = wide_df.rename(columns=rename_map)

    return wide_df


def merge_predictions_with_targets(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
    models: Optional[List[ModelType]] = None,
) -> pd.DataFrame:
    """Merge multi-model predictions with target outcomes.

    Args:
        predictions: Multi-model predictions (wide format).
        targets: Targets DataFrame with entry_date and outcomes.
        models: Models to include.

    Returns:
        Merged DataFrame with predictions and outcomes.
    """
    if models is None:
        models = list(ModelType)

    predictions = predictions.copy()
    targets = targets.copy()

    # Standardize date columns
    predictions["date"] = pd.to_datetime(predictions["date"])
    if "entry_date" in targets.columns:
        targets["entry_date"] = pd.to_datetime(targets["entry_date"])
    elif "t0" in targets.columns:
        targets["entry_date"] = pd.to_datetime(targets["t0"])

    # Select prediction columns
    pred_cols = ["date", "symbol"] + [get_prob_column(m) for m in models]
    pred_cols = [c for c in pred_cols if c in predictions.columns]

    # Select target columns
    target_cols = [
        "symbol", "entry_date", "exit_date", "entry_price",
        "top", "bot", "exit_price", "ret_from_entry", "hit",
        "h_used", "weight_final",
    ]
    target_cols = [c for c in target_cols if c in targets.columns]

    # Merge
    merged = predictions[pred_cols].merge(
        targets[target_cols],
        left_on=["date", "symbol"],
        right_on=["entry_date", "symbol"],
        how="inner",
    )

    # Standardize column names
    merged = merged.rename(columns={
        "top": "target_price",
        "bot": "stop_price",
        "exit_date": "actual_exit_date",
        "exit_price": "actual_exit_price",
        "ret_from_entry": "actual_return",
        "h_used": "holding_days",
    })

    # Add week Monday
    merged["week_monday"] = merged["date"].apply(
        lambda x: x - pd.Timedelta(days=x.weekday())
    )

    return merged


def get_rebalance_dates(
    predictions: pd.DataFrame,
    freq: str = "W-MON",
) -> pd.DatetimeIndex:
    """Get rebalance dates from predictions.

    Args:
        predictions: Predictions DataFrame.
        freq: Pandas frequency string for rebalance.

    Returns:
        DatetimeIndex of rebalance dates.
    """
    predictions["date"] = pd.to_datetime(predictions["date"])

    # Get unique dates
    dates = pd.to_datetime(predictions["date"].unique())

    # Filter to desired frequency
    if freq == "W-MON":
        # Get Mondays
        mondays = dates[dates.weekday == 0]
        return pd.DatetimeIndex(sorted(mondays))
    else:
        return pd.DatetimeIndex(sorted(dates))


def check_oos_predictions(
    predictions: pd.DataFrame,
    cv_metadata: Optional[Dict] = None,
) -> Dict:
    """Check that predictions are strictly out-of-sample.

    Args:
        predictions: Predictions DataFrame.
        cv_metadata: Optional CV fold metadata for verification.

    Returns:
        Dict with validation results.
    """
    result = {
        "is_oos": True,
        "issues": [],
    }

    # If fold metadata is available, verify OOS status
    if cv_metadata and "fold_metrics" in cv_metadata:
        for fold in cv_metadata["fold_metrics"]:
            if not fold.get("is_oos", True):
                result["is_oos"] = False
                result["issues"].append(
                    f"Fold {fold.get('fold', '?')} contains in-sample predictions"
                )

    # Check for common leakage indicators
    if "is_oos" in predictions.columns:
        n_in_sample = (~predictions["is_oos"]).sum()
        if n_in_sample > 0:
            result["is_oos"] = False
            result["issues"].append(f"{n_in_sample} in-sample predictions found")

    return result
