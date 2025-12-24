#!/usr/bin/env python
"""
Optimize multi-model position sizing parameters.

This script optimizes sizing parameters for the 4-model system:
- LONG_NORMAL
- LONG_PARABOLIC
- SHORT_NORMAL
- SHORT_PARABOLIC

Supports optional regime gating optimization.

Usage:
    # Optimize sizing for all 4 models
    python scripts/run_multi_model_sizing.py \\
        --prediction-path artifacts/predictions/cv_predictions_multi.parquet \\
        --n-trials 100

    # With regime gating optimization
    python scripts/run_multi_model_sizing.py \\
        --prediction-path artifacts/predictions/ \\
        --regime-gating on \\
        --n-trials 200

    # Single model backward-compatible mode
    python scripts/run_multi_model_sizing.py \\
        --models LONG_NORMAL \\
        --prediction-path artifacts/predictions/cv_predictions.parquet
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.sizing import (
    ModelType,
    MultiModelSizingConfig,
    load_multi_model_predictions,
    MultiModelSizingEngine,
)
from src.sizing.optimizer import (
    MultiModelSizingOptimizer,
    print_optimization_summary,
    save_optimization_result,
)
from src.sizing.predictions import merge_predictions_with_targets


def parse_models(models_str: str) -> list:
    """Parse comma-separated model names."""
    if not models_str:
        return list(ModelType)

    model_names = [m.strip().upper() for m in models_str.split(",")]
    models = []
    for name in model_names:
        try:
            models.append(ModelType(name.lower()))
        except ValueError:
            print(f"Warning: Unknown model '{name}', skipping")
    return models or list(ModelType)


def load_targets(path: str = "artifacts/targets_triple_barrier.parquet") -> pd.DataFrame:
    """Load targets file."""
    df = pd.read_parquet(path)

    # Standardize column names
    if "t0" in df.columns:
        df = df.rename(columns={"t0": "entry_date"})
    if "t_hit" in df.columns:
        df = df.rename(columns={"t_hit": "exit_date"})

    df["entry_date"] = pd.to_datetime(df["entry_date"])

    # Exclude neutral if present
    if "hit" in df.columns:
        df = df[df["hit"] != 0].copy()

    return df


def load_regime_features(
    features_path: str = "artifacts/features_complete.parquet",
) -> pd.DataFrame:
    """Load regime-relevant features."""
    df = pd.read_parquet(features_path)
    df["date"] = pd.to_datetime(df["date"])

    # Select regime-relevant columns
    regime_cols = [
        "date", "symbol",
        "vix_percentile_252d", "d_vix_percentile_252d",
        "vix_zscore_60d", "d_vix_zscore_60d",
        "fred_bamlh0a0hym2_z60", "d_fred_bamlh0a0hym2_z60",
        "sector_breadth_pct_above_ma200", "d_sector_breadth_pct_above_ma200",
        "w_equity_bond_corr_60d",
    ]

    available = [c for c in regime_cols if c in df.columns]
    return df[available].copy()


def main():
    parser = argparse.ArgumentParser(
        description="Optimize multi-model position sizing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model selection
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated list of models (default: all four). "
             "Options: LONG_NORMAL,LONG_PARABOLIC,SHORT_NORMAL,SHORT_PARABOLIC",
    )

    # Data paths
    parser.add_argument(
        "--prediction-path",
        type=str,
        required=True,
        help="Path to multi-model predictions (parquet file or directory)",
    )
    parser.add_argument(
        "--targets-path",
        type=str,
        default="artifacts/targets_triple_barrier.parquet",
        help="Path to targets file",
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default="artifacts/features_complete.parquet",
        help="Path to features file (for regime gating)",
    )

    # Optimization settings
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of optimization trials",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="penalized_return",
        choices=["penalized_return", "portfolio_return", "hit_rate"],
        help="Metric to optimize",
    )

    # Regime gating
    parser.add_argument(
        "--regime-gating",
        type=str,
        default="off",
        choices=["on", "off"],
        help="Whether to optimize regime gating parameters",
    )

    # Combining policy
    parser.add_argument(
        "--combine-policy",
        type=str,
        default="mode_priority",
        choices=["mode_priority", "blend"],
        help="Policy for combining model signals",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/sizing",
        help="Output directory for results",
    )

    # Constraints
    parser.add_argument(
        "--max-gross-exposure",
        type=float,
        default=1.0,
        help="Maximum gross exposure",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=0.10,
        help="Maximum weight per position",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="Maximum number of positions",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Multi-Model Position Sizing Optimization")
    print("=" * 60)
    print()

    # Parse models
    models = parse_models(args.models)
    print(f"Models: {[m.value for m in models]}")
    print(f"Combine policy: {args.combine_policy}")
    print(f"Regime gating: {args.regime_gating}")
    print()

    # Load predictions
    print("Loading predictions...")
    prediction_path = Path(args.prediction_path)

    try:
        predictions = load_multi_model_predictions(prediction_path, models)
        print(f"  Loaded {len(predictions):,} prediction rows")
        print(f"  Date range: {predictions['date'].min().date()} to {predictions['date'].max().date()}")
    except Exception as e:
        print(f"Error loading predictions: {e}")
        print("\nExpected format:")
        print("  - Wide parquet with columns: date, symbol, p_long_normal, p_long_parabolic, ...")
        print("  - OR directory with per-model parquets")
        sys.exit(1)

    # Load targets
    print("\nLoading targets...")
    targets = load_targets(args.targets_path)
    print(f"  Loaded {len(targets):,} target rows")
    print(f"  Date range: {targets['entry_date'].min().date()} to {targets['entry_date'].max().date()}")

    # Merge predictions with targets
    print("\nMerging predictions with targets...")
    signals = merge_predictions_with_targets(predictions, targets, models)
    print(f"  Merged signals: {len(signals):,}")

    if len(signals) == 0:
        print("ERROR: No overlapping dates between predictions and targets")
        sys.exit(1)

    print(f"  Signal date range: {signals['date'].min().date()} to {signals['date'].max().date()}")

    # Load regime features if gating enabled
    regime_features = None
    if args.regime_gating == "on":
        print("\nLoading regime features...")
        regime_features = load_regime_features(args.features_path)
        print(f"  Loaded {len(regime_features):,} feature rows")

    # Create base config
    base_config = MultiModelSizingConfig(
        models=[m.value for m in models],
        combine_policy=args.combine_policy,
        max_gross_exposure=args.max_gross_exposure,
        max_weight_per_name=args.max_weight,
        max_positions=args.max_positions,
    )

    # Create optimizer
    print(f"\nOptimizing with {args.n_trials} trials...")
    print(f"Objective metric: {args.metric}")

    optimizer = MultiModelSizingOptimizer(
        models=models,
        n_trials=args.n_trials,
        metric=args.metric,
        optimize_gating=(args.regime_gating == "on"),
        base_config=base_config,
    )

    # Run optimization
    result = optimizer.optimize(
        signals,
        regime_features=regime_features,
        show_progress=True,
    )

    # Print summary
    print_optimization_summary(result)

    # Save results
    output_dir = Path(args.output_dir)
    suffix = "multi_model"
    if len(models) == 1:
        suffix = models[0].value

    paths = save_optimization_result(result, output_dir, name=suffix)

    print("\nResults saved:")
    for file_type, path in paths.items():
        print(f"  {file_type}: {path}")

    print("\n" + "=" * 60)
    print("Optimization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
