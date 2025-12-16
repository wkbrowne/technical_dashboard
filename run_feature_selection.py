#!/usr/bin/env python
"""Run Loose-then-Tight feature selection on the latest feature data.

This script:
1. Loads features and triple barrier targets
2. Merges them and prepares data for weekly prediction
3. Runs the LOOSE-THEN-TIGHT feature selection pipeline:
   - Start with base features (seed, not forced)
   - Loose forward selection (high recall)
   - Strict backward elimination (all features removable)
   - Light interaction pass
   - Hill climbing / pair swapping
   - Final cleanup
   - Return best subset from any stage
4. Saves results to artifacts/
"""

import gc
import os
import sys
import time
from pathlib import Path

# Set joblib temp folder to main disk (not tmpfs /tmp which is only 23GB)
# This prevents "No space left on device" errors during parallel processing
JOBLIB_TEMP = Path(__file__).parent / ".joblib_temp"
JOBLIB_TEMP.mkdir(exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = str(JOBLIB_TEMP)

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.feature_selection import (
    # Loose-then-tight pipeline
    LooseTightPipeline,
    LooseTightConfig,
    run_loose_tight_selection,
    # Base features
    get_base_features,
    get_expansion_candidates,
    validate_features,
    # Configuration
    ModelConfig, ModelType, TaskType,
    CVConfig, CVScheme,
    SearchConfig,
    MetricConfig, MetricType,
    # Utilities
    validate_data,
    filter_features_by_nan_rate,
    filter_correlated_features,
    print_feature_summary,
)


def compute_scale_pos_weight(y: pd.Series) -> float:
    """Compute scale_pos_weight for LightGBM class balancing.

    scale_pos_weight = n_negative / n_positive

    Args:
        y: Binary target series (0/1)

    Returns:
        scale_pos_weight value for LightGBM
    """
    n_positive = (y == 1).sum()
    n_negative = (y == 0).sum()
    return n_negative / n_positive


def load_and_prepare_data(
    max_symbols: int = 200,
    binary_target: bool = True,
    min_samples_per_symbol: int = 100
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load and prepare data for feature selection.

    Args:
        max_symbols: Maximum number of symbols to include (for memory).
        binary_target: If True, convert to binary (up=1 vs down=0), excluding neutral.
        min_samples_per_symbol: Minimum samples required per symbol.

    Returns:
        Tuple of (X, y, regime) DataFrames/Series.
    """
    print("Loading features...")
    features = pd.read_parquet('artifacts/features_daily.parquet')
    print(f"  Features shape: {features.shape}")

    print("Loading targets...")
    targets = pd.read_parquet('artifacts/targets_triple_barrier.parquet')
    print(f"  Targets shape: {targets.shape}")

    # Merge features with targets
    print("Merging features with targets...")
    targets = targets.rename(columns={'t0': 'date'})

    # Select symbols with enough data
    symbol_counts = targets.groupby('symbol').size()
    valid_symbols = symbol_counts[symbol_counts >= min_samples_per_symbol].index.tolist()
    print(f"  Symbols with >= {min_samples_per_symbol} samples: {len(valid_symbols)}")

    # Limit symbols for memory
    if len(valid_symbols) > max_symbols:
        # Pick most liquid symbols (by count)
        valid_symbols = symbol_counts.loc[valid_symbols].nlargest(max_symbols).index.tolist()
        print(f"  Limited to top {max_symbols} symbols by sample count")

    # Filter data
    features = features[features['symbol'].isin(valid_symbols)].copy()
    targets = targets[targets['symbol'].isin(valid_symbols)].copy()

    # Merge
    merged = features.merge(
        targets[['symbol', 'date', 'hit', 'weight_final']],
        on=['symbol', 'date'],
        how='inner'
    )
    print(f"  Merged shape: {merged.shape}")

    # Handle target
    if binary_target:
        # Binary: upper barrier (1) vs lower barrier (0), exclude neutral
        # This gives cleaner signal - definitive outcomes only
        merged = merged[merged['hit'] != 0].copy()
        merged['target'] = (merged['hit'] == 1).astype(int)
        print(f"  Binary target (upper vs lower, excluding neutral): {merged.shape[0]} samples")
        print(f"    Positive (hit upper): {(merged['target'] == 1).sum()}")
        print(f"    Negative (hit lower): {(merged['target'] == 0).sum()}")
    else:
        # Multi-class: -1, 0, 1 -> 0, 1, 2
        merged['target'] = merged['hit'] + 1

    # Sort by date for time-series CV
    merged = merged.sort_values(['date', 'symbol']).reset_index(drop=True)

    # Define feature columns (exclude metadata and target-related)
    exclude_cols = {
        'symbol', 'date', 'close', 'open', 'high', 'low', 'volume', 'adjclose',
        'hit', 'weight_final', 'target'
    }
    feature_cols = [c for c in merged.columns if c not in exclude_cols]

    # Extract regime (volatility regime if available)
    regime = None
    if 'vol_regime' in merged.columns:
        regime = merged['vol_regime'].copy()

    # Create X and y
    X = merged[feature_cols].copy()
    y = merged['target'].copy()

    # Set date as index for time-series CV
    X.index = merged['date']
    y.index = merged['date']
    if regime is not None:
        regime.index = merged['date']

    print(f"\nFinal data:")
    print(f"  X shape: {X.shape}")
    print(f"  y distribution: {y.value_counts().to_dict()}")
    print(f"  Date range: {X.index.min()} to {X.index.max()}")

    # Clean up
    del features, targets, merged
    gc.collect()

    return X, y, regime


def prefilter_features(
    X: pd.DataFrame,
    max_nan_rate: float = 0.3,
    correlation_threshold: float = 0.95
) -> list[str]:
    """Pre-filter features before selection.

    Args:
        X: Feature DataFrame.
        max_nan_rate: Maximum acceptable NaN rate.
        correlation_threshold: Correlation threshold for deduplication.

    Returns:
        List of filtered feature names.
    """
    print("\nPre-filtering features...")

    # Filter by NaN rate
    valid_features = filter_features_by_nan_rate(X, max_nan_rate)
    print(f"  After NaN filter (max {max_nan_rate:.0%}): {len(valid_features)} features")

    return valid_features


def run_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    regime: pd.Series = None,
    n_folds: int = 5,
    n_jobs: int = 8,
    scale_pos_weight: float = None,
    prune_base: bool = False
) -> LooseTightPipeline:
    """Run the LOOSE-THEN-TIGHT feature selection pipeline.

    Pipeline steps:
    1. Base features (seed, not forced)
    1b. (Optional) Base feature elimination - quick prune of base features
    2. Loose forward selection (high recall)
    3. Strict backward elimination (all features removable)
    4. Light interaction pass
    5. Hill climbing / pair swapping
    6. Final cleanup
    7. Return best subset from any stage

    Args:
        X: Feature DataFrame.
        y: Target Series.
        features: List of feature names to consider.
        regime: Optional regime labels (unused in loose-tight).
        n_folds: Number of CV folds.
        n_jobs: Number of parallel jobs.
        scale_pos_weight: LightGBM class weight (n_neg/n_pos). None for no weighting.
        prune_base: If True, run quick reverse elimination on BASE_FEATURES before
                   forward selection to prune weak base features early.

    Returns:
        Fitted LooseTightPipeline object.
    """
    print(f"\n{'='*60}")
    print("Running LOOSE-THEN-TIGHT Feature Selection Pipeline")
    print(f"{'='*60}")
    print(f"Available features: {len(features)}")
    print(f"Samples: {len(X)}")
    print(f"CV Folds: {n_folds}")
    print(f"Parallel Jobs: {n_jobs}")
    if scale_pos_weight is not None:
        print(f"Class Balancing: scale_pos_weight={scale_pos_weight:.3f}")
    print()

    # Validate base features
    base_features = get_base_features()
    base_in_data = [f for f in base_features if f in X.columns]
    print(f"Base features: {len(base_in_data)}/{len(base_features)} available")

    # Expansion candidates
    expansion = get_expansion_candidates(flat=True)
    expansion_in_data = [f for f in expansion if f in X.columns]
    print(f"Expansion candidates: {len(expansion_in_data)}/{len(expansion)} available")

    # Select only valid features
    print("\nCopying feature subset...")
    X_subset = X[features].copy()
    print(f"  X_subset shape: {X_subset.shape}")

    # Configure model for LightGBM classification
    # With 64 cores: 8 parallel candidate evaluations × 8 threads per model = 64 total
    model_params = {
        'learning_rate': 0.05,
        'max_depth': 6,
        'num_leaves': 31,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
    }

    # Add class weighting if specified
    if scale_pos_weight is not None:
        model_params['scale_pos_weight'] = scale_pos_weight

    model_config = ModelConfig(
        model_type=ModelType.LIGHTGBM,
        task_type=TaskType.CLASSIFICATION,
        params=model_params,
        num_threads=8,  # 8 threads per model (8 jobs × 8 threads = 64 cores)
        early_stopping_rounds=50,
        num_boost_round=300,
    )

    # Configure CV with embargo
    # gap=20 matches max_horizon to prevent label overlap between train/test
    cv_config = CVConfig(
        n_splits=n_folds,
        scheme=CVScheme.EXPANDING,
        gap=20,  # Match max_horizon to fully separate outcome windows
        purge_window=0,
        min_train_samples=1000,
    )

    # Configure search (for evaluator)
    search_config = SearchConfig(
        n_jobs=1,  # Evaluator parallelizes internally
        random_state=42,
    )

    # Configure metrics
    metric_config = MetricConfig(
        primary_metric=MetricType.AUC,
        secondary_metrics=[MetricType.LOG_LOSS],
        tail_quantile=0.1,
    )

    # Configure the loose-then-tight pipeline
    pipeline_config = LooseTightConfig(
        # Base feature elimination (optional quick pruning)
        run_base_elimination=prune_base,
        epsilon_remove_base=0.0005,  # Slightly more lenient than strict

        # LOOSE forward selection (high recall)
        epsilon_add_loose=0.0002,  # Very permissive - allow weak features
        min_fold_improvement_ratio_loose=0.6,  # 60% of folds must improve
        max_features_loose=80,  # Hard cap after loose FS

        # STRICT backward elimination (high precision)
        epsilon_remove_strict=0.0,  # Remove anything that doesn't help

        # Interactions
        run_interactions=True,
        max_interactions=8,
        epsilon_add_interaction=0.001,  # Stricter than loose FS
        n_top_features_for_interactions=20,

        # Swapping
        epsilon_swap=0.0005,
        max_swap_iterations=50,
        n_borderline_features=10,

        # Parallelization
        n_jobs=n_jobs,
    )

    # Create pipeline
    pipeline = LooseTightPipeline(
        model_config=model_config,
        cv_config=cv_config,
        search_config=search_config,
        metric_config=metric_config,
        pipeline_config=pipeline_config,
    )

    print("\nStarting pipeline training...")
    start_time = time.time()
    pipeline.run(X_subset, y, verbose=True)
    elapsed = time.time() - start_time

    print(f"\nFeature selection completed in {elapsed/60:.1f} minutes")

    return pipeline


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='LOOSE-THEN-TIGHT Feature Selection')
    parser.add_argument('--balanced', action='store_true',
                        help='Use class weights (scale_pos_weight) for balanced training')
    parser.add_argument('--max-symbols', type=int, default=5000,
                        help='Maximum number of symbols to include')
    parser.add_argument('--n-jobs', type=int, default=8,
                        help='Number of parallel jobs')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--prune-base', action='store_true',
                        help='Run quick reverse elimination on BASE_FEATURES before forward selection')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint if available')
    parser.add_argument('--checkpoint-info', action='store_true',
                        help='Show checkpoint info and exit')
    args = parser.parse_args()

    # Handle checkpoint info request
    if args.checkpoint_info:
        info = LooseTightPipeline.checkpoint_info()
        if info:
            print("Checkpoint found:")
            print(info)
        else:
            print("No checkpoint found.")
        return None

    print("="*60)
    print("LOOSE-THEN-TIGHT Feature Selection")
    if args.balanced:
        print("  MODE: BALANCED (using class weights)")
    else:
        print("  MODE: IMBALANCED (no class weights)")
    if args.prune_base:
        print("  BASE FEATURE PRUNING: ENABLED")
    print("="*60)
    print()

    # Load and prepare data - use all symbols
    X, y, regime = load_and_prepare_data(
        max_symbols=args.max_symbols,
        binary_target=True,
        min_samples_per_symbol=5
    )

    # Compute scale_pos_weight if balanced mode
    scale_pos_weight = None
    if args.balanced:
        scale_pos_weight = compute_scale_pos_weight(y)
        print(f"\nClass balancing enabled:")
        print(f"  Positive samples: {(y == 1).sum():,}")
        print(f"  Negative samples: {(y == 0).sum():,}")
        print(f"  scale_pos_weight: {scale_pos_weight:.3f}")

    # Validate data
    is_valid, issues = validate_data(X, y, min_samples=1000)
    if not is_valid:
        print("\nData validation issues:")
        for issue in issues:
            print(f"  - {issue}")
        if len(issues) > 3:
            print("\nProceeding with warnings...")

    # Pre-filter features
    valid_features = prefilter_features(X, max_nan_rate=0.3, correlation_threshold=0.95)

    # Check for resume
    if args.resume and LooseTightPipeline.has_checkpoint():
        print("\n" + "="*60)
        print("RESUMING FROM CHECKPOINT")
        print("="*60)
        info = LooseTightPipeline.checkpoint_info()
        print(info)
        print()

        # Create pipeline and resume
        pipeline = LooseTightPipeline()
        pipeline.resume_from_checkpoint(X, y, verbose=True)
    else:
        if args.resume:
            print("\nNo checkpoint found, starting fresh...")

        # Run loose-then-tight feature selection
        pipeline = run_feature_selection(
            X, y,
            features=valid_features,
            regime=regime,
            n_folds=args.n_folds,
            n_jobs=args.n_jobs,
            scale_pos_weight=scale_pos_weight,
            prune_base=args.prune_base
        )

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    # Get stage summary
    summary = pipeline.get_stage_summary()
    print("\nStage Summary:")
    print(summary.to_string(index=False))

    # Get best features
    best_features = pipeline.get_best_features()
    best_metric = pipeline.get_best_metric()

    print(f"\nBest Subset:")
    print(f"  Stage: {pipeline.best_snapshot.stage}")
    print(f"  Features: {len(best_features)}")
    print(f"  AUC: {best_metric[0]:.4f} +/- {best_metric[1]:.4f}")
    print()
    print("Selected features:")
    for f in sorted(best_features):
        print(f"  - {f}")

    # Print feature summary
    print_feature_summary(best_features, "Best Subset Summary")

    # Show all snapshots
    print("\nAll Pipeline Snapshots:")
    for s in pipeline.snapshots:
        marker = " <-- BEST" if s == pipeline.best_snapshot else ""
        print(f"  {s.stage}: {len(s.features)} features, AUC={s.metric_mean:.4f}±{s.metric_std:.4f}{marker}")

    # Save results
    output_dir = Path('artifacts/feature_selection')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save stage summary
    summary.to_csv(output_dir / 'stage_summary.csv', index=False)
    print(f"\nStage summary saved to {output_dir}/stage_summary.csv")

    # Save best features
    with open(output_dir / 'selected_features.txt', 'w') as f:
        f.write(f"# Best features from Loose-then-Tight pipeline\n")
        f.write(f"# Stage: {pipeline.best_snapshot.stage}\n")
        f.write(f"# AUC: {best_metric[0]:.4f} +/- {best_metric[1]:.4f}\n\n")
        for feat in sorted(best_features):
            f.write(f"{feat}\n")
    print(f"Selected features list saved to {output_dir}/selected_features.txt")

    return pipeline


if __name__ == '__main__':
    main()
