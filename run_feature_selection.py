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
import json
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
    min_samples_per_symbol: int = 100,
    use_filtered_features: bool = True
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Load and prepare data for feature selection.

    Args:
        max_symbols: Maximum number of symbols to include (for memory).
        binary_target: If True, convert to binary (up=1 vs down=0), excluding neutral.
        min_samples_per_symbol: Minimum samples required per symbol.
        use_filtered_features: If True, load pre-filtered features from features_filtered.parquet.

    Returns:
        Tuple of (X, y, sample_weight, sector, symbol) DataFrames/Series.
        - sector: Sector labels for each sample (for sector-stratified evaluation)
        - symbol: Symbol labels for each sample (for debugging)
    """
    # Choose feature file
    if use_filtered_features:
        feature_file = 'artifacts/features_filtered.parquet'
        print(f"Loading filtered features from {feature_file}...")
    else:
        feature_file = 'artifacts/features_daily.parquet'
        print(f"Loading raw features from {feature_file}...")

    features = pd.read_parquet(feature_file)
    print(f"  Features shape: {features.shape}")

    print("Loading targets...")
    targets = pd.read_parquet('artifacts/targets_triple_barrier.parquet')
    print(f"  Targets shape: {targets.shape}")

    # Check for sample weights
    has_weights = 'weight_final' in targets.columns
    if not has_weights:
        print("  WARNING: 'weight_final' column not found in targets!")
        print("           Sample weighting will be DISABLED.")
        print("           Re-run target generation to enable overlap inverse weighting.")

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

    # Merge - include weight_final only if available
    target_cols = ['symbol', 'date', 'hit']
    if has_weights:
        target_cols.append('weight_final')

    merged = features.merge(
        targets[target_cols],
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
        'hit', 'weight_final', 'target', 'sector'
    }
    feature_cols = [c for c in merged.columns if c not in exclude_cols]

    # Load sector labels from universe CSV for sector-stratified evaluation
    sector = None
    symbol_series = merged['symbol'].copy()
    universe_files = list(Path('cache').glob('US universe*.csv'))
    if universe_files:
        universe_file = universe_files[0]
        print(f"Loading sector labels from {universe_file}...")
        try:
            universe = pd.read_csv(universe_file)
            if 'Symbol' in universe.columns and 'Sector' in universe.columns:
                sector_map = dict(zip(universe['Symbol'], universe['Sector']))
                merged['sector'] = merged['symbol'].map(sector_map)
                # Fill missing sectors with 'Unknown'
                merged['sector'] = merged['sector'].fillna('Unknown')
                sector = merged['sector'].copy()
                sector_counts = sector.value_counts()
                print(f"  Sectors loaded: {len(sector_counts)} unique sectors")
                print(f"  Sector distribution:")
                for sec, count in sector_counts.head(5).items():
                    print(f"    {sec}: {count:,} samples ({100*count/len(sector):.1f}%)")
                if len(sector_counts) > 5:
                    print(f"    ... and {len(sector_counts) - 5} more sectors")
            else:
                print(f"  WARNING: Universe file missing Symbol/Sector columns")
        except Exception as e:
            print(f"  WARNING: Could not load sector labels: {e}")
    else:
        print("  WARNING: No universe CSV found in cache/, sector-stratified evaluation disabled")

    # Create X, y, and sample_weight
    X = merged[feature_cols].copy()
    y = merged['target'].copy()

    # Extract sample weights if available
    sample_weight = None
    if has_weights:
        sample_weight = merged['weight_final'].copy()

    # Set date as index for time-series CV
    X.index = merged['date']
    y.index = merged['date']
    if sample_weight is not None:
        sample_weight.index = merged['date']
    if sector is not None:
        sector.index = merged['date']
    symbol_series.index = merged['date']

    print(f"\nFinal data:")
    print(f"  X shape: {X.shape}")
    print(f"  y distribution: {y.value_counts().to_dict()}")
    print(f"  Date range: {X.index.min()} to {X.index.max()}")
    if sample_weight is not None:
        print(f"  Sample weight range: [{sample_weight.min():.3f}, {sample_weight.max():.3f}]")
        print(f"  Sample weight mean: {sample_weight.mean():.3f}")
    else:
        print(f"  Sample weights: DISABLED (no weight_final in targets)")
    if sector is not None:
        print(f"  Sector-stratified evaluation: ENABLED ({sector.nunique()} sectors)")
    else:
        print(f"  Sector-stratified evaluation: DISABLED (no sector labels)")

    # Clean up
    del features, targets, merged
    gc.collect()

    return X, y, sample_weight, sector, symbol_series


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
    sample_weight: pd.Series = None,
    sector: pd.Series = None,
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
        sample_weight: Sample weights from triple barrier (overlap inverse weighting).
        sector: Sector labels for sector-stratified evaluation.
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
    if sample_weight is not None:
        print(f"Sample Weighting: ENABLED (overlap inverse)")
    if scale_pos_weight is not None:
        print(f"Class Balancing: scale_pos_weight={scale_pos_weight:.3f}")
    if sector is not None:
        print(f"Sector-Stratified Evaluation: ENABLED ({sector.nunique()} sectors)")
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

    # Configure metrics - track multiple metrics for comprehensive evaluation
    metric_config = MetricConfig(
        primary_metric=MetricType.AUC,
        secondary_metrics=[MetricType.AUPR, MetricType.LOG_LOSS],
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

    # Subset sample weights to match X_subset index
    sample_weight_subset = None
    if sample_weight is not None:
        sample_weight_subset = sample_weight.loc[X_subset.index]

    print("\nStarting pipeline training...")
    start_time = time.time()
    pipeline.run(X_subset, y, verbose=True, sample_weight=sample_weight_subset)
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
    X, y, sample_weight, sector, symbol = load_and_prepare_data(
        max_symbols=args.max_symbols,
        binary_target=True,
        min_samples_per_symbol=5,
        use_filtered_features=True
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
            sample_weight=sample_weight,
            sector=sector,
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

    # Re-evaluate best features to get extended metrics for final summary
    print("\n  Evaluating final metrics on best subset...")
    from src.feature_selection.evaluation import SubsetEvaluator
    from src.feature_selection.config import SearchConfig

    # Use the pipeline's config to re-evaluate (with sector for stratified metrics)
    final_evaluator = SubsetEvaluator(
        X=X[valid_features].copy(),
        y=y,
        model_config=pipeline.model_config,
        cv_config=pipeline.cv_config,
        metric_config=pipeline.metric_config,
        search_config=SearchConfig(),
        sample_weight=sample_weight.loc[X.index] if sample_weight is not None else None,
        regime=sector.loc[X.index] if sector is not None else None,  # Pass sector for stratified metrics
    )
    final_result = final_evaluator.evaluate(best_features)

    # Print extended metrics table
    print()
    print("  ┌────────────────────────────────────────────────────────────┐")
    print("  │              FINAL METRICS SUMMARY                         │")
    print("  ├────────────┬────────────┬────────────┬────────────────────┤")
    print("  │  Metric    │    Mean    │    Std     │   Interpretation   │")
    print("  ├────────────┼────────────┼────────────┼────────────────────┤")

    # Print each metric
    metrics_to_show = [
        ('auc', 'AUC', '↑ Discrimination'),
        ('aupr', 'AUPR', '↑ Prec-Recall'),
        ('brier', 'Brier', '↓ Calibration'),
        ('log_loss', 'LogLoss', '↓ Likelihood'),
    ]

    for key, display_name, interpretation in metrics_to_show:
        if key in final_result.secondary_metrics:
            mean, std = final_result.secondary_metrics[key]
            print(f"  │ {display_name:^10} │ {mean:^10.4f} │ {std:^10.4f} │ {interpretation:^18} │")
        else:
            print(f"  │ {display_name:^10} │ {'N/A':^10} │ {'N/A':^10} │ {interpretation:^18} │")

    print("  └────────────┴────────────┴────────────┴────────────────────┘")

    # Print sector-stratified metrics with concentration warning
    if final_result.regime_metrics and sector is not None:
        sector_auc_values = list(final_result.regime_metrics.values())
        sector_auc_std = np.std(sector_auc_values) if len(sector_auc_values) > 1 else 0.0
        sector_auc_mean = np.mean(sector_auc_values)

        print()
        print("  ┌────────────────────────────────────────────────────────────┐")
        print("  │              SECTOR-STRATIFIED AUC                         │")
        print("  ├─────────────────────────────┬──────────────────────────────┤")
        print(f"  │ {'Sector':<27} │ {'AUC':^28} │")
        print("  ├─────────────────────────────┼──────────────────────────────┤")

        # Sort sectors by AUC (descending)
        sorted_sectors = sorted(final_result.regime_metrics.items(), key=lambda x: x[1], reverse=True)
        for sec_name, sec_auc in sorted_sectors:
            # Highlight if significantly different from mean
            indicator = ''
            if sec_auc > sector_auc_mean + sector_auc_std:
                indicator = ' ▲'  # Above average
            elif sec_auc < sector_auc_mean - sector_auc_std:
                indicator = ' ▼'  # Below average
            print(f"  │ {sec_name:<27} │ {sec_auc:^26.4f}{indicator} │")

        print("  ├─────────────────────────────┴──────────────────────────────┤")
        print(f"  │ {'Mean':27} │ {sector_auc_mean:^28.4f} │")
        print(f"  │ {'Std Dev':27} │ {sector_auc_std:^28.4f} │")
        print("  └────────────────────────────────────────────────────────────┘")

        # Concentration warning
        CONCENTRATION_THRESHOLD = 0.03
        if sector_auc_std > CONCENTRATION_THRESHOLD:
            print()
            print("  ⚠️  WARNING: High sector AUC variance detected!")
            print(f"     Sector AUC Std Dev ({sector_auc_std:.4f}) > threshold ({CONCENTRATION_THRESHOLD})")
            print("     This may indicate the model concentrates on specific sectors.")
            print("     Consider adding more cross-sectional features to BASE_FEATURES.")

            # Show best/worst sectors
            if len(sorted_sectors) >= 2:
                best_sec, best_auc = sorted_sectors[0]
                worst_sec, worst_auc = sorted_sectors[-1]
                print(f"     Best sector:  {best_sec} (AUC: {best_auc:.4f})")
                print(f"     Worst sector: {worst_sec} (AUC: {worst_auc:.4f})")
                print(f"     Spread: {best_auc - worst_auc:.4f}")
        else:
            print()
            print(f"  ✓ Sector coverage looks consistent (Std Dev: {sector_auc_std:.4f} < {CONCENTRATION_THRESHOLD})")

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

    # Build extended metrics dict for JSON output
    extended_metrics = {}
    for key in ['auc', 'aupr', 'brier', 'log_loss']:
        if key in final_result.secondary_metrics:
            mean, std = final_result.secondary_metrics[key]
            extended_metrics[key] = {"mean": round(mean, 6), "std": round(std, 6)}

    # Build sector metrics dict for JSON output
    sector_metrics = {}
    sector_concentration_warning = False
    if final_result.regime_metrics and sector is not None:
        sector_auc_values = list(final_result.regime_metrics.values())
        sector_auc_std = np.std(sector_auc_values) if len(sector_auc_values) > 1 else 0.0
        sector_auc_mean = np.mean(sector_auc_values)
        CONCENTRATION_THRESHOLD = 0.03

        sector_metrics = {
            "by_sector": {sec: round(auc, 6) for sec, auc in final_result.regime_metrics.items()},
            "mean": round(sector_auc_mean, 6),
            "std": round(sector_auc_std, 6),
            "concentration_threshold": CONCENTRATION_THRESHOLD,
            "concentration_warning": sector_auc_std > CONCENTRATION_THRESHOLD
        }
        sector_concentration_warning = sector_auc_std > CONCENTRATION_THRESHOLD

    # Save selection_summary.json with complete results
    selection_summary = {
        "best_stage": pipeline.best_snapshot.stage,
        "n_features": len(best_features),
        "metric_mean": best_metric[0],
        "metric_std": best_metric[1],
        "extended_metrics": extended_metrics,
        "sector_metrics": sector_metrics,
        "sector_concentration_warning": sector_concentration_warning,
        "features": sorted(best_features),
        "stages": [
            {
                "stage": s.stage,
                "n_features": len(s.features),
                "metric_mean": s.metric_mean,
                "metric_std": s.metric_std,
                "is_best": s == pipeline.best_snapshot
            }
            for s in pipeline.snapshots
        ],
        "config": {
            "n_folds": args.n_folds,
            "n_jobs": args.n_jobs,
            "balanced": args.balanced,
            "prune_base": args.prune_base,
            "max_symbols": args.max_symbols,
            "sample_weighting": "overlap_inverse" if sample_weight is not None else "disabled",
            "sector_stratified": sector is not None
        }
    }
    with open(output_dir / 'selection_summary.json', 'w') as f:
        json.dump(selection_summary, f, indent=2)
    print(f"Selection summary saved to {output_dir}/selection_summary.json")

    return pipeline


if __name__ == '__main__':
    main()
