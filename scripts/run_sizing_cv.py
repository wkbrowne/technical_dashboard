#!/usr/bin/env python
"""
Train sizing models with purged walk-forward CV and calibration.

This script trains per-fold LightGBM classifiers for the position sizing
pipeline. Unlike run_training.py (which trains a single production model),
this produces fold-level artifacts needed for:
- Sizing parameter optimization (Optuna)
- Out-of-fold probability calibration
- Backtest validation

Uses hardcoded conservative hyperparameters by default. The sizing optimizer
evaluates position sizing rules, not model hyperparameters.

Usage:
    python scripts/run_sizing_cv.py [--config config.yaml] [--output-dir artifacts/models]
    python scripts/run_sizing_cv.py --dry-run  # Preview CV folds
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.alpha.config import AlphaConfig, load_config, WeightScheme
from src.alpha.data.loaders import load_targets, load_features, prepare_weekly_signals
from src.alpha.cv import WeeklySignalCV, print_cv_summary, get_fold_info
from src.alpha.models import train_with_cv, save_cv_results


def main():
    parser = argparse.ArgumentParser(description="Train sizing models with purged CV")
    parser.add_argument("--config", type=str, default="src/alpha/config/default.yaml",
                        help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default="artifacts/models",
                        help="Output directory for model artifacts")
    parser.add_argument("--weight-scheme", type=str, default="overlap_inverse",
                        choices=["uniform", "overlap_inverse", "liquidity", "inverse_volatility"],
                        help="Sample weighting scheme")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print CV folds without training")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("Sizing Model Training with Purged Walk-Forward CV")
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

    # Prepare weekly signals
    print("\nPreparing weekly signals...")
    signals = prepare_weekly_signals(
        predictions=pd.DataFrame({
            "symbol": targets["symbol"],
            "date": targets["entry_date"],
            "probability": 0.5,  # Placeholder - actual model predictions would go here
        }),
        targets=targets,
        features=features,
    )
    print(f"  Signals: {len(signals):,}")

    # Get feature columns from selected features file
    selected_features_file = Path("artifacts/feature_selection/selected_features.txt")
    if selected_features_file.exists():
        with open(selected_features_file) as f:
            feature_columns = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        print(f"  Using {len(feature_columns)} selected features")
    else:
        # Use default features
        feature_columns = [col for col in features.columns
                          if col not in ["symbol", "date", "close", "open", "high", "low", "volume"]][:50]
        print(f"  No selected features file, using {len(feature_columns)} columns")

    # Setup CV
    cv = WeeklySignalCV(
        n_splits=config.cv.n_splits,
        min_train_weeks=config.cv.min_train_weeks,
        test_weeks=config.cv.test_weeks,
        purge_days=config.cv.purge_days,
        embargo_days=config.cv.embargo_days,
    )

    # Print CV summary
    fold_infos = get_fold_info(cv, signals)
    print_cv_summary(fold_infos)

    if args.dry_run:
        print("\nDry run - stopping before training")
        return

    # Map weight scheme
    weight_map = {
        "uniform": WeightScheme.UNIFORM,
        "overlap_inverse": WeightScheme.OVERLAP_INVERSE,
        "liquidity": WeightScheme.LIQUIDITY,
        "inverse_volatility": WeightScheme.INVERSE_VOLATILITY,
    }
    weight_scheme = weight_map[args.weight_scheme]
    print(f"\nUsing weight scheme: {weight_scheme.value}")

    # Train with CV
    print("\nTraining models...")
    results = train_with_cv(
        signals=signals,
        feature_columns=feature_columns,
        config=config,
        weight_scheme=weight_scheme,
    )

    # Save results
    print("\nSaving artifacts...")
    saved = save_cv_results(results, output_dir, prefix="sizing_model")
    for name, path in saved.items():
        print(f"  {name}: {path}")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    val_aucs = [r.val_metrics["auc"] for r in results]
    print(f"\nValidation AUC: {sum(val_aucs)/len(val_aucs):.4f} ± {pd.Series(val_aucs).std():.4f}")
    print(f"Folds: {len(results)}")

    print("\nTraining complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
