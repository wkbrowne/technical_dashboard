#!/usr/bin/env python
"""
Optimize position sizing parameters using Optuna.

This script optimizes sizing rule parameters using purged walk-forward CV,
ensuring the objective is computed only on validation folds.

IMPORTANT: This script requires out-of-sample CV predictions to produce valid
results. Run `generate_cv_predictions.py` first to create these predictions.

Using placeholder predictions (based on actual outcomes) will give you
artificially good results due to information leakage!

Usage:
    # First generate proper out-of-sample predictions
    python scripts/generate_cv_predictions.py --use-hyperopt-params

    # Then optimize sizing
    python scripts/run_optimize_sizing.py [--config config.yaml] [--n-trials 100]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.alpha.config import AlphaConfig, SizingMethod, load_config
from src.alpha.data.loaders import load_targets, load_features, prepare_weekly_signals
from src.alpha.sizing.optimizer import SizingOptimizer, print_optimization_summary
from src.alpha.models.artifacts import save_sizing_params


def load_cv_predictions(path: Path) -> pd.DataFrame:
    """Load CV predictions with validation.

    Args:
        path: Path to cv_predictions.parquet

    Returns:
        DataFrame with columns: symbol, date, probability
    """
    df = pd.read_parquet(path)

    # Ensure required columns
    required = ["symbol", "date", "probability"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CV predictions missing columns: {missing}")

    # Use calibrated probability if available
    if "probability" not in df.columns and "probability_raw" in df.columns:
        df["probability"] = df["probability_raw"]

    df["date"] = pd.to_datetime(df["date"])

    return df[["symbol", "date", "probability"]].copy()


def main():
    parser = argparse.ArgumentParser(description="Optimize sizing parameters")
    parser.add_argument("--config", type=str, default="src/alpha/config/default.yaml",
                        help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default="artifacts/sizing",
                        help="Output directory for sizing parameters")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Number of optimization trials")
    parser.add_argument("--method", type=str, default="monotone_probability",
                        choices=["monotone_probability", "rank_bucket", "threshold_confidence"],
                        help="Sizing method to optimize")
    parser.add_argument("--metric", type=str, default="penalized_return",
                        help="Metric to optimize")
    parser.add_argument("--predictions", type=str, default="artifacts/predictions/cv_predictions.parquet",
                        help="Path to CV predictions file (from generate_cv_predictions.py)")
    parser.add_argument("--allow-placeholder", action="store_true",
                        help="Allow placeholder predictions (WARNING: causes information leakage)")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("Position Sizing Optimization")
    print("=" * 60)
    print()

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        print(f"Loading config from: {config_path}")
        config = load_config(config_path)
    else:
        print("Using default configuration")
        config = AlphaConfig()

    # Load data
    print("\nLoading data...")
    targets = load_targets(config.targets_file, exclude_neutral=True)
    print(f"  Targets: {len(targets):,} samples")

    features = load_features(config.features_file)
    print(f"  Features: {features.shape}")

    # Load CV predictions (the correct way)
    cv_predictions_file = Path(args.predictions)
    use_cv_predictions = False
    use_placeholder = False

    if cv_predictions_file.exists():
        print(f"\n  Loading CV predictions from: {cv_predictions_file}")
        predictions = load_cv_predictions(cv_predictions_file)
        print(f"    Predictions: {len(predictions):,} rows")

        # Check date overlap with targets
        pred_dates = predictions["date"]
        target_dates = targets["entry_date"]
        n_overlap = pred_dates.isin(target_dates).sum()

        if n_overlap > 0:
            use_cv_predictions = True
            print(f"    Date range: {pred_dates.min().date()} to {pred_dates.max().date()}")
            print(f"    Overlap with targets: {n_overlap:,} dates")
            print(f"    Using OUT-OF-SAMPLE CV predictions (valid for backtesting)")
        else:
            print(f"    No date overlap with targets")

    if not use_cv_predictions:
        # Check for legacy predictions file
        legacy_predictions_file = Path(config.predictions_file)
        if legacy_predictions_file.exists():
            print(f"\n  Checking legacy predictions file: {legacy_predictions_file}")
            legacy_preds = pd.read_parquet(legacy_predictions_file)
            if "date" in legacy_preds.columns:
                pred_dates = pd.to_datetime(legacy_preds["date"])
                if pred_dates.isin(targets["entry_date"]).any():
                    print("    WARNING: Legacy predictions may contain in-sample data!")
                    print("    Use generate_cv_predictions.py for proper out-of-sample predictions.")

        if args.allow_placeholder:
            # Use placeholder (for development/testing only)
            print("\n  WARNING: Using placeholder predictions!")
            print("  These are based on actual outcomes and will show artificially good results.")
            print("  This is INFORMATION LEAKAGE - do not use for production backtesting!")
            predictions = pd.DataFrame({
                "symbol": targets["symbol"],
                "date": targets["entry_date"],
                "probability": 0.5 + (targets["hit"] == 1).astype(float) * 0.3,
            })
            use_placeholder = True
        else:
            print("\n" + "=" * 60)
            print("ERROR: No valid CV predictions found!")
            print("=" * 60)
            print()
            print("To generate proper out-of-sample predictions, run:")
            print("  python scripts/generate_cv_predictions.py --use-hyperopt-params")
            print()
            print("Or use --allow-placeholder to use placeholder predictions")
            print("(WARNING: placeholder predictions cause information leakage)")
            sys.exit(1)

    # Prepare weekly signals
    # Note: Signals are limited to dates present in targets (inner join)
    # This naturally enforces that we only backtest on dates with known outcomes
    print("\nPreparing weekly signals...")
    print(f"  Targets date range: {targets['entry_date'].min().date()} to {targets['entry_date'].max().date()}")

    signals = prepare_weekly_signals(
        predictions=predictions,
        targets=targets,
        features=features,
    )

    if len(signals) == 0:
        print("  ERROR: No signals generated (predictions don't overlap with targets)")
        print("  Predictions need dates within the targets date range for backtesting")
        sys.exit(1)

    print(f"  Signals: {len(signals):,}")
    print(f"  Signal date range: {signals['week_monday'].min().date()} to {signals['week_monday'].max().date()}")

    # Map sizing method
    method_map = {
        "monotone_probability": SizingMethod.MONOTONE_PROBABILITY,
        "rank_bucket": SizingMethod.RANK_BUCKET,
        "threshold_confidence": SizingMethod.THRESHOLD_CONFIDENCE,
    }
    method = method_map[args.method]

    # Create optimizer
    print(f"\nOptimizing {method.value} sizing with {args.n_trials} trials...")
    print(f"Objective metric: {args.metric}")

    optimizer = SizingOptimizer(
        config=config,
        method=method,
        n_trials=args.n_trials,
        metric=args.metric,
    )

    # Run optimization
    result = optimizer.optimize(signals, show_progress=True)

    # Print summary
    print_optimization_summary(result)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Best params
    params_path = output_dir / f"best_params_{method.value}.json"
    save_sizing_params(
        result.best_params,
        params_path,
        metadata={
            "method": method.value,
            "objective": args.metric,
            "best_value": result.best_value,
            "n_trials": args.n_trials,
            "cv_metrics": result.cv_metrics,
            "predictions_source": "cv_predictions" if use_cv_predictions else "placeholder",
            "predictions_file": str(cv_predictions_file) if use_cv_predictions else None,
            "warning": None if use_cv_predictions else "PLACEHOLDER PREDICTIONS - results may be overfitted",
        }
    )
    print(f"\nBest params saved to: {params_path}")

    # Trials history
    trials_path = output_dir / f"trials_{method.value}.csv"
    result.trials_df.to_csv(trials_path, index=False)
    print(f"Trials saved to: {trials_path}")

    # Parameter importance
    if result.param_importance:
        importance_path = output_dir / f"param_importance_{method.value}.json"
        import json
        with open(importance_path, "w") as f:
            json.dump(result.param_importance, f, indent=2)
        print(f"Parameter importance saved to: {importance_path}")

    print("\n" + "=" * 60)
    print("Optimization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
