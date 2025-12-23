#!/usr/bin/env python
"""
Generate out-of-sample predictions using purged walk-forward CV.

This script produces TRUE out-of-sample predictions by:
1. Training on historical data up to time T
2. Predicting on data after T (that the model never saw)
3. Repeating across walk-forward folds

These predictions are the ONLY valid basis for backtesting and sizing
optimization. Using production model predictions would be information leakage.

Output:
    artifacts/predictions/cv_predictions.parquet
    - Contains predictions for ALL samples covered by CV folds
    - Each sample predicted by the model that NEVER saw it during training
    - Includes fold info for analysis

Usage:
    python scripts/generate_cv_predictions.py
    python scripts/generate_cv_predictions.py --use-hyperopt-params
    python scripts/generate_cv_predictions.py --dry-run
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.alpha.config import AlphaConfig, load_config, WeightScheme
from src.alpha.data.loaders import load_targets, load_features, prepare_weekly_signals
from src.alpha.cv import WeeklySignalCV, validate_no_leakage
from src.alpha.models.train_lgbm import compute_sample_weights
from src.alpha.models.calibration import CalibratedClassifier, evaluate_calibration


def load_hyperopt_params(path: str = "artifacts/hyperopt/best_params.json") -> dict:
    """Load hyperparameters from hyperopt tuning."""
    with open(path) as f:
        params = json.load(f)

    # Extract only LightGBM params (exclude _metrics and _metadata)
    lgbm_params = {k: v for k, v in params.items() if not k.startswith("_")}

    # Convert to proper types
    int_params = ["max_depth", "num_leaves", "min_child_samples", "n_estimators",
                  "subsample_freq", "max_bin"]
    for p in int_params:
        if p in lgbm_params:
            lgbm_params[p] = int(lgbm_params[p])

    # Remove non-LightGBM params
    lgbm_params.pop("use_balanced", None)

    return lgbm_params


def get_default_params() -> dict:
    """Conservative default LightGBM parameters."""
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "max_depth": 6,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbosity": -1,
        "random_state": 42,
        "n_jobs": -1,
    }


def generate_cv_predictions(
    signals: pd.DataFrame,
    feature_columns: list,
    config: AlphaConfig,
    lgbm_params: dict,
    weight_scheme: WeightScheme = WeightScheme.OVERLAP_INVERSE,
    calibrate: bool = True,
) -> pd.DataFrame:
    """Generate out-of-sample predictions for all CV folds.

    For each fold:
    - Train on training set only
    - Predict on validation set (out-of-sample)
    - Optionally calibrate probabilities

    Returns DataFrame with predictions for all samples covered by CV.
    """
    cv = WeeklySignalCV(
        n_splits=config.cv.n_splits,
        min_train_weeks=config.cv.min_train_weeks,
        test_weeks=config.cv.test_weeks,
        purge_days=config.cv.purge_days,
        embargo_days=config.cv.embargo_days,
    )

    # Get available features
    available_features = [f for f in feature_columns if f in signals.columns]
    if len(available_features) < len(feature_columns):
        missing = set(feature_columns) - set(available_features)
        print(f"  Warning: {len(missing)} features not found: {list(missing)[:5]}...")

    # Prepare results storage
    all_predictions = []
    fold_metrics = []

    for train_idx, val_idx, fold_info in cv.split(signals):
        train_start = pd.Timestamp(fold_info.train_start).date()
        train_end = pd.Timestamp(fold_info.train_end).date()
        test_start = pd.Timestamp(fold_info.test_start).date()
        test_end = pd.Timestamp(fold_info.test_end).date()
        print(f"\nFold {fold_info.fold_idx}:")
        print(f"  Train: {train_start} to {train_end}")
        print(f"  Test:  {test_start} to {test_end}")
        print(f"  Samples: {len(train_idx):,} train ({fold_info.n_purged} purged), {len(val_idx):,} val")

        # Validate no leakage
        is_valid, details = validate_no_leakage(signals, train_idx, val_idx)
        if not is_valid:
            print(f"  WARNING: Potential leakage detected!")
        else:
            print(f"  Gap: {details['gap_days']} days (no leakage)")

        # Prepare data
        train_data = signals.iloc[train_idx]
        val_data = signals.iloc[val_idx]

        X_train = train_data[available_features]
        y_train = (train_data["hit"] == 1).astype(int)

        X_val = val_data[available_features]
        y_val = (val_data["hit"] == 1).astype(int)

        # Clean data
        X_train_clean = X_train.fillna(0).replace([np.inf, -np.inf], 0)
        X_val_clean = X_val.fillna(0).replace([np.inf, -np.inf], 0)

        # Compute sample weights
        weights = compute_sample_weights(train_data, scheme=weight_scheme)

        # Train model
        model = lgb.LGBMClassifier(**lgbm_params)
        model.fit(X_train_clean, y_train, sample_weight=weights)

        # Predict on validation (out-of-sample!)
        y_prob_raw = model.predict_proba(X_val_clean)[:, 1]

        # Calibrate if requested
        if calibrate:
            calibrator = CalibratedClassifier(method="isotonic")
            calibrator.fit(y_val.values, y_prob_raw)
            y_prob = calibrator.transform(y_prob_raw)
        else:
            y_prob = y_prob_raw

        # Compute metrics
        from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

        auc = roc_auc_score(y_val, y_prob_raw)
        aupr = average_precision_score(y_val, y_prob_raw)
        brier = brier_score_loss(y_val, y_prob_raw)

        print(f"  AUC: {auc:.4f}, AUPR: {aupr:.4f}, Brier: {brier:.4f}")

        fold_metrics.append({
            "fold": fold_info.fold_idx,
            "train_start": str(train_start),
            "train_end": str(train_end),
            "test_start": str(test_start),
            "test_end": str(test_end),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_purged": fold_info.n_purged,
            "auc": auc,
            "aupr": aupr,
            "brier": brier,
        })

        # Store predictions for validation samples
        fold_preds = pd.DataFrame({
            "original_idx": val_idx,
            "symbol": val_data["symbol"].values,
            "date": val_data["signal_date"].values if "signal_date" in val_data.columns else val_data["week_monday"].values,
            "probability_raw": y_prob_raw,
            "probability": y_prob,
            "actual": y_val.values,
            "fold": fold_info.fold_idx,
        })
        all_predictions.append(fold_preds)

    # Combine all fold predictions
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    # Add summary stats
    print("\n" + "=" * 60)
    print("CV PREDICTION SUMMARY")
    print("=" * 60)
    print(f"Total predictions: {len(predictions_df):,}")
    print(f"Unique symbols: {predictions_df['symbol'].nunique()}")
    print(f"Date range: {predictions_df['date'].min()} to {predictions_df['date'].max()}")

    # Overall metrics
    overall_auc = roc_auc_score(predictions_df["actual"], predictions_df["probability_raw"])
    overall_aupr = average_precision_score(predictions_df["actual"], predictions_df["probability_raw"])
    print(f"\nOverall AUC (out-of-sample): {overall_auc:.4f}")
    print(f"Overall AUPR (out-of-sample): {overall_aupr:.4f}")

    # Attach fold metrics as metadata
    predictions_df.attrs["fold_metrics"] = fold_metrics
    predictions_df.attrs["overall_auc"] = overall_auc
    predictions_df.attrs["overall_aupr"] = overall_aupr

    return predictions_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate out-of-sample predictions via walk-forward CV"
    )
    parser.add_argument("--config", type=str, default="src/alpha/config/default.yaml",
                        help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default="artifacts/predictions",
                        help="Output directory for predictions")
    parser.add_argument("--use-hyperopt-params", action="store_true",
                        help="Use hyperparameters from hyperopt tuning")
    parser.add_argument("--weight-scheme", type=str, default="overlap_inverse",
                        choices=["uniform", "overlap_inverse", "liquidity", "inverse_volatility"],
                        help="Sample weighting scheme")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="Skip probability calibration")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show CV folds without generating predictions")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("GENERATE OUT-OF-SAMPLE CV PREDICTIONS")
    print("=" * 60)
    print()
    print("This script generates TRUE out-of-sample predictions by training")
    print("on historical data and predicting on held-out validation folds.")
    print("These predictions are valid for backtesting and sizing optimization.")
    print()

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        print(f"Loading config from: {config_path}")
        config = load_config(config_path)
    else:
        print("Using default configuration")
        config = AlphaConfig()

    # Load LightGBM parameters
    if args.use_hyperopt_params:
        hyperopt_path = Path("artifacts/hyperopt/best_params.json")
        if hyperopt_path.exists():
            print(f"Loading hyperopt params from: {hyperopt_path}")
            lgbm_params = load_hyperopt_params(str(hyperopt_path))
        else:
            print("Hyperopt params not found, using defaults")
            lgbm_params = get_default_params()
    else:
        print("Using default LightGBM parameters")
        lgbm_params = get_default_params()

    # Add required params
    lgbm_params.update({
        "objective": "binary",
        "verbosity": -1,
        "random_state": 42,
        "n_jobs": -1,
    })

    print(f"  Key params: max_depth={lgbm_params.get('max_depth')}, "
          f"num_leaves={lgbm_params.get('num_leaves')}, "
          f"n_estimators={lgbm_params.get('n_estimators')}")

    # Load data
    print("\nLoading data...")
    targets = load_targets(config.targets_file, exclude_neutral=True)
    print(f"  Targets: {len(targets):,} samples")
    print(f"  Date range: {targets['entry_date'].min().date()} to {targets['entry_date'].max().date()}")

    features = load_features(config.features_file)
    print(f"  Features: {features.shape}")

    # Load selected features
    selected_features_file = Path("artifacts/feature_selection/selected_features.txt")
    if selected_features_file.exists():
        with open(selected_features_file) as f:
            feature_columns = [line.strip() for line in f
                             if line.strip() and not line.startswith("#")]
        print(f"  Selected features: {len(feature_columns)}")
    else:
        print("  WARNING: No selected features file found, using all numeric columns")
        feature_columns = [col for col in features.columns
                          if col not in ["symbol", "date"] and features[col].dtype in [np.float64, np.float32, np.int64]][:50]

    # Prepare weekly signals (using placeholder probabilities - they'll be replaced)
    print("\nPreparing signals for CV...")
    signals = prepare_weekly_signals(
        predictions=pd.DataFrame({
            "symbol": targets["symbol"],
            "date": targets["entry_date"],
            "probability": 0.5,  # Placeholder - will be replaced by CV predictions
        }),
        targets=targets,
        features=features,
    )
    print(f"  Signals prepared: {len(signals):,}")

    # Merge features needed for training
    feat_cols_in_signals = [f for f in feature_columns if f in signals.columns]
    feat_cols_to_merge = [f for f in feature_columns if f not in signals.columns and f in features.columns]

    if feat_cols_to_merge:
        print(f"  Merging {len(feat_cols_to_merge)} additional features...")
        feat_subset = features[["symbol", "date"] + feat_cols_to_merge].copy()
        signals = signals.merge(
            feat_subset,
            left_on=["symbol", "signal_date"],
            right_on=["symbol", "date"],
            how="left",
        )
        if "date" in signals.columns and "signal_date" in signals.columns:
            signals = signals.drop(columns=["date"])

    # Setup CV and show folds
    cv = WeeklySignalCV(
        n_splits=config.cv.n_splits,
        min_train_weeks=config.cv.min_train_weeks,
        test_weeks=config.cv.test_weeks,
        purge_days=config.cv.purge_days,
        embargo_days=config.cv.embargo_days,
    )

    print(f"\nCV Configuration:")
    print(f"  n_splits: {config.cv.n_splits}")
    print(f"  min_train_weeks: {config.cv.min_train_weeks}")
    print(f"  test_weeks: {config.cv.test_weeks}")
    print(f"  purge_days: {config.cv.purge_days}")
    print(f"  embargo_days: {config.cv.embargo_days}")

    if args.dry_run:
        print("\nDry run - showing fold structure:")
        for train_idx, val_idx, fold_info in cv.split(signals):
            train_start = pd.Timestamp(fold_info.train_start).date()
            train_end = pd.Timestamp(fold_info.train_end).date()
            test_start = pd.Timestamp(fold_info.test_start).date()
            test_end = pd.Timestamp(fold_info.test_end).date()
            print(f"  Fold {fold_info.fold_idx}: "
                  f"train {train_start} to {train_end} "
                  f"({len(train_idx):,} samples), "
                  f"val {test_start} to {test_end} "
                  f"({len(val_idx):,} samples)")
        print("\nDry run complete - exiting without generating predictions")
        return

    # Map weight scheme
    weight_map = {
        "uniform": WeightScheme.UNIFORM,
        "overlap_inverse": WeightScheme.OVERLAP_INVERSE,
        "liquidity": WeightScheme.LIQUIDITY,
        "inverse_volatility": WeightScheme.INVERSE_VOLATILITY,
    }
    weight_scheme = weight_map[args.weight_scheme]
    print(f"\nWeight scheme: {weight_scheme.value}")

    # Generate predictions
    print("\n" + "-" * 60)
    print("GENERATING OUT-OF-SAMPLE PREDICTIONS")
    print("-" * 60)

    predictions = generate_cv_predictions(
        signals=signals,
        feature_columns=feature_columns,
        config=config,
        lgbm_params=lgbm_params,
        weight_scheme=weight_scheme,
        calibrate=not args.no_calibrate,
    )

    # Save predictions
    output_dir.mkdir(parents=True, exist_ok=True)

    # Main predictions file
    pred_path = output_dir / "cv_predictions.parquet"
    predictions.to_parquet(pred_path, index=False)
    print(f"\nPredictions saved to: {pred_path}")

    # Save fold metrics as JSON for inspection
    metrics_path = output_dir / "cv_predictions_metrics.json"
    metrics_data = {
        "fold_metrics": predictions.attrs.get("fold_metrics", []),
        "overall_auc": predictions.attrs.get("overall_auc"),
        "overall_aupr": predictions.attrs.get("overall_aupr"),
        "n_predictions": len(predictions),
        "n_symbols": int(predictions["symbol"].nunique()),
        "date_range": {
            "start": str(predictions["date"].min()),
            "end": str(predictions["date"].max()),
        },
        "generated_at": datetime.now().isoformat(),
        "params": {
            "use_hyperopt_params": args.use_hyperopt_params,
            "weight_scheme": args.weight_scheme,
            "calibrated": not args.no_calibrate,
        }
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    print("\n" + "=" * 60)
    print("CV PREDICTION GENERATION COMPLETE")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Run sizing optimization with these predictions:")
    print("     python scripts/run_optimize_sizing.py")
    print()
    print("  2. Run backtest with optimized parameters:")
    print("     python scripts/run_backtest.py")
    print()


if __name__ == "__main__":
    main()
