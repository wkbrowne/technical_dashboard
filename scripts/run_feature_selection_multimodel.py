#!/usr/bin/env python
"""
Multi-Model Feature Selection Runner.

This script runs feature selection for all 4 model targets (LONG_NORMAL,
LONG_PARABOLIC, SHORT_NORMAL, SHORT_PARABOLIC) in a single run, sharing:
- The same CV splits for comparability
- The same feature universe
- A single data load

Outputs:
- Per-model: artifacts/feature_selection/{model_key}/selected_features.json|txt
- Global: artifacts/feature_selection/summary.json
- (Optional) Auto-updates src/feature_selection/base_features.py

Usage:
    # Run all 4 models (default)
    python scripts/run_feature_selection_multimodel.py

    # Run specific models
    python scripts/run_feature_selection_multimodel.py --model-keys LONG_NORMAL,SHORT_NORMAL

    # Limit max features
    python scripts/run_feature_selection_multimodel.py --max-features 50

    # Auto-update base_features.py with new CORE/HEAD features
    python scripts/run_feature_selection_multimodel.py --update-registry

    # Dry-run to preview registry changes without modifying
    python scripts/run_feature_selection_multimodel.py --update-registry --dry-run

    # Use cached data
    python scripts/run_feature_selection_multimodel.py --use-cache --cache-path /tmp/fs_cache.pkl
"""

import argparse
import gc
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Set joblib temp folder before imports
JOBLIB_TEMP = Path(__file__).parent.parent / ".joblib_temp"
JOBLIB_TEMP.mkdir(exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = str(JOBLIB_TEMP)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.model_keys import ModelKey
from src.feature_selection import (
    get_all_selectable_features,
    get_base_features,
    get_expansion_candidates,
    filter_features_by_nan_rate,
    validate_data,
)
from src.feature_selection.config import (
    CVConfig,
    CVScheme,
    MetricConfig,
    MetricType,
    ModelConfig,
    ModelType,
    TaskType,
)
from src.feature_selection.pipeline import LooseTightConfig
from src.feature_selection.multimodel import (
    FeatureSelectionResult,
    MultiModelSelectionSummary,
    compute_overlap_analysis,
    run_single_model_selection,
    validate_selected_features,
    write_global_summary,
    write_per_model_artifacts,
    # Auto-update functions
    update_base_features_file,
    compute_feature_diff,
    print_feature_diff,
)
from src.feature_selection.base_features import CORE_FEATURES, HEAD_FEATURES


# =============================================================================
# Data Loading
# =============================================================================

def load_and_prepare_data(
    max_symbols: int = 5000,
    min_samples_per_symbol: int = 5,
    use_filtered_features: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
    """Load features and targets, prepare for multi-model selection.

    Args:
        max_symbols: Maximum number of symbols to include.
        min_samples_per_symbol: Minimum samples required per symbol.
        use_filtered_features: Whether to use pre-filtered features.
        verbose: Whether to print progress.

    Returns:
        Tuple of (X, targets_df, sample_weight, sector)
        - X: Feature DataFrame
        - targets_df: DataFrame with columns [symbol, date, hit, weight_final, ...]
        - sample_weight: Sample weight Series (or None)
        - sector: Sector labels Series (or None)
    """
    if verbose:
        print("Loading data...")

    # Load features
    if use_filtered_features:
        feature_file = 'artifacts/features_filtered.parquet'
    else:
        feature_file = 'artifacts/features_complete.parquet'

    if not Path(feature_file).exists():
        # Fallback to complete if filtered doesn't exist
        feature_file = 'artifacts/features_complete.parquet'

    features = pd.read_parquet(feature_file)
    if verbose:
        print(f"  Features: {features.shape} from {feature_file}")

    # Load targets
    targets = pd.read_parquet('artifacts/targets_triple_barrier.parquet')
    if verbose:
        print(f"  Targets: {targets.shape}")

    # Rename t0 to date for consistency
    targets = targets.rename(columns={'t0': 'date'})

    # Filter symbols by sample count
    symbol_counts = targets.groupby('symbol').size()
    valid_symbols = symbol_counts[symbol_counts >= min_samples_per_symbol].index.tolist()
    if verbose:
        print(f"  Symbols with >= {min_samples_per_symbol} samples: {len(valid_symbols)}")

    if len(valid_symbols) > max_symbols:
        valid_symbols = symbol_counts.loc[valid_symbols].nlargest(max_symbols).index.tolist()
        if verbose:
            print(f"  Limited to top {max_symbols} symbols")

    # Filter data
    features = features[features['symbol'].isin(valid_symbols)].copy()
    targets = targets[targets['symbol'].isin(valid_symbols)].copy()

    # Check for sample weights
    has_weights = 'weight_final' in targets.columns
    sample_weight = None

    # Merge features with targets (inner join on symbol, date)
    merged = features.merge(
        targets,
        on=['symbol', 'date'],
        how='inner'
    )
    if verbose:
        print(f"  Merged: {merged.shape}")

    # Sort by date for time-series CV
    merged = merged.sort_values(['date', 'symbol']).reset_index(drop=True)

    # Load sector labels
    sector = None
    universe_files = list(Path('cache').glob('US universe*.csv'))
    if universe_files:
        try:
            universe = pd.read_csv(universe_files[0])
            if 'Symbol' in universe.columns and 'Sector' in universe.columns:
                sector_map = dict(zip(universe['Symbol'], universe['Sector']))
                merged['sector'] = merged['symbol'].map(sector_map).fillna('Unknown')
                sector = merged['sector'].copy()
                if verbose:
                    print(f"  Sectors: {sector.nunique()} unique")
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not load sector labels: {e}")

    # Define feature columns (exclude metadata and target-related)
    exclude_cols = {
        'symbol', 'date', 'close', 'open', 'high', 'low', 'volume', 'adjclose',
        'hit', 'weight_final', 'target', 'sector',
        # Triple barrier target columns (per-model hit columns)
        'hit_long_normal', 'hit_long_parabolic', 'hit_short_normal', 'hit_short_parabolic',
        'ret_long_normal', 'ret_long_parabolic', 'ret_short_normal', 'ret_short_parabolic',
        'h_used_long_normal', 'h_used_long_parabolic', 'h_used_short_normal', 'h_used_short_parabolic',
        't0', 't_hit', 'entry_px', 'atr_at_entry',
        'side', 'h_used', 'pnl', 'n_overlapping_trajs',
        'weight_overlap', 'weight_class_balance', 'is_last_traj',
    }
    feature_cols = [c for c in merged.columns if c not in exclude_cols]

    # Create X
    X = merged[feature_cols].copy()
    X.index = merged['date']

    # Create targets DataFrame with model-specific hit columns
    target_cols = ['symbol', 'date']
    # Add model-specific hit columns if they exist
    for model_hit_col in ['hit_long_normal', 'hit_long_parabolic', 'hit_short_normal', 'hit_short_parabolic']:
        if model_hit_col in merged.columns:
            target_cols.append(model_hit_col)
    # Fallback to 'hit' if no model-specific columns
    if 'hit' in merged.columns and not any(c.startswith('hit_') for c in target_cols):
        target_cols.append('hit')
    if has_weights:
        target_cols.append('weight_final')
    targets_df = merged[target_cols].copy()

    # Extract sample weights
    if has_weights:
        sample_weight = merged['weight_final'].copy()
        sample_weight.index = merged['date']

    # Align sector index
    if sector is not None:
        sector.index = merged['date']

    if verbose:
        print(f"\nPrepared data:")
        print(f"  X shape: {X.shape}")
        print(f"  Date range: {X.index.min()} to {X.index.max()}")
        if sample_weight is not None:
            print(f"  Sample weights: mean={sample_weight.mean():.3f}")
        if sector is not None:
            print(f"  Sectors: {sector.nunique()} unique")

    # Clean up
    del features, merged
    gc.collect()

    return X, targets_df, sample_weight, sector


def get_model_labels(
    targets_df: pd.DataFrame,
    model_key: ModelKey,
    binary: bool = True,
) -> pd.Series:
    """Get target labels for a specific model key.

    Uses model-specific hit columns from triple barrier targets:
    - hit_long_normal: 1 if upper barrier hit, -1 if lower, 0 if timeout
    - hit_long_parabolic: same for extended upper barrier
    - hit_short_normal: same for lower barrier target
    - hit_short_parabolic: same for extended lower barrier

    Args:
        targets_df: DataFrame with target columns (hit_long_normal, etc.)
        model_key: Model key to get labels for
        binary: If True, convert to binary (exclude neutral/timeout)

    Returns:
        Target Series with binary labels (1=success, 0=failure)
    """
    # Map model key to corresponding hit column
    label_column_map = {
        ModelKey.LONG_NORMAL: 'hit_long_normal',
        ModelKey.LONG_PARABOLIC: 'hit_long_parabolic',
        ModelKey.SHORT_NORMAL: 'hit_short_normal',
        ModelKey.SHORT_PARABOLIC: 'hit_short_parabolic',
    }

    label_col = label_column_map.get(model_key)

    # Fallback to 'hit' if model-specific column doesn't exist
    if label_col not in targets_df.columns:
        if 'hit' in targets_df.columns:
            label_col = 'hit'
        else:
            raise ValueError(
                f"No target column found for {model_key}. "
                f"Expected '{label_column_map[model_key]}' or 'hit'. "
                f"Available columns: {list(targets_df.columns)}"
            )

    df = targets_df.copy()

    if binary:
        # Binary: hit barrier (1) vs missed/opposite (-1), exclude timeout (0)
        df = df[df[label_col] != 0].copy()

        if model_key.is_long():
            # Long models: upper barrier hit = success (1), lower = failure (0)
            df['target'] = (df[label_col] == 1).astype(int)
        else:
            # Short models: lower barrier hit = success (1), upper = failure (0)
            df['target'] = (df[label_col] == -1).astype(int)
    else:
        # Multi-class: -1, 0, 1 -> 0, 1, 2
        df['target'] = df[label_col] + 1

    y = df['target'].copy()
    y.index = df['date']

    return y


# =============================================================================
# Multi-Model Runner
# =============================================================================

def run_multimodel_selection(
    model_keys: List[ModelKey],
    X: pd.DataFrame,
    targets_df: pd.DataFrame,
    sample_weight: Optional[pd.Series] = None,
    candidate_features: Optional[List[str]] = None,
    max_features: Optional[int] = None,
    output_dir: Path = Path("artifacts/feature_selection"),
    cv_config: Optional[CVConfig] = None,
    model_config: Optional[ModelConfig] = None,
    pipeline_config: Optional[LooseTightConfig] = None,
    verbose: bool = True,
) -> MultiModelSelectionSummary:
    """Run feature selection for multiple models.

    Args:
        model_keys: List of ModelKey enums to run selection for
        X: Feature DataFrame
        targets_df: DataFrame with target labels
        sample_weight: Optional sample weights
        candidate_features: Optional list of candidate features (default: all selectable)
        max_features: Optional max features limit
        output_dir: Output directory for artifacts
        cv_config: CV configuration (shared across all models)
        model_config: Model configuration
        pipeline_config: Pipeline configuration
        verbose: Whether to print progress

    Returns:
        MultiModelSelectionSummary with all results
    """
    start_time = time.time()

    # Get candidate features
    if candidate_features is None:
        candidate_features = get_all_selectable_features()

    # Filter to features present in X
    available_features = set(X.columns)
    candidate_features = [f for f in candidate_features if f in available_features]

    if verbose:
        print(f"\n{'='*70}")
        print(f"MULTI-MODEL FEATURE SELECTION")
        print(f"{'='*70}")
        print(f"Models: {', '.join(mk.value for mk in model_keys)}")
        print(f"Candidate features: {len(candidate_features)}")
        print(f"Output: {output_dir}")
        print()

    # Pre-filter features by NaN rate
    valid_features = filter_features_by_nan_rate(X[candidate_features], max_rate=0.3)
    if verbose:
        print(f"After NaN filter (max 30%): {len(valid_features)} features")

    # Use shared CV config for all models
    if cv_config is None:
        cv_config = CVConfig(
            n_splits=5,
            scheme=CVScheme.EXPANDING,
            gap=20,
            purge_window=0,
            min_train_samples=1000,
        )

    # Apply max_features to pipeline config
    if pipeline_config is None:
        pipeline_config = LooseTightConfig(
            run_base_elimination=False,
            epsilon_add_loose=0.0002,
            min_fold_improvement_ratio_loose=0.6,
            max_features_loose=max_features or 80,
            epsilon_remove_strict=0.0,
            run_interactions=True,
            max_interactions=8,
            epsilon_add_interaction=0.001,
            epsilon_swap=0.0005,
            max_swap_iterations=50,
            n_jobs=8,
        )
    elif max_features:
        pipeline_config.max_features_loose = max_features

    # Run selection for each model
    results: Dict[str, FeatureSelectionResult] = {}

    for i, model_key in enumerate(model_keys):
        if verbose:
            print(f"\n[{i+1}/{len(model_keys)}] Running {model_key.value.upper()}")
            print("-" * 50)

        # Get labels for this model
        y = get_model_labels(targets_df, model_key, binary=True)

        # Align X with y (binary target excludes neutral samples)
        X_aligned = X.loc[y.index].copy()
        sample_weight_aligned = sample_weight.loc[y.index] if sample_weight is not None else None

        if verbose:
            pos_rate = y.mean()
            print(f"Samples: {len(y)}, Positive rate: {pos_rate:.1%}")

        # Run selection
        result = run_single_model_selection(
            X=X_aligned,
            y=y,
            model_key=model_key,
            features=valid_features,
            sample_weight=sample_weight_aligned,
            cv_config=cv_config,
            model_config=model_config,
            pipeline_config=pipeline_config,
            verbose=verbose,
        )

        results[model_key.value] = result

        # Write per-model artifacts
        model_output_dir = output_dir / model_key.value
        write_per_model_artifacts(result, model_output_dir)

        if verbose:
            print(f"Artifacts written to {model_output_dir}")

        # Clean up
        gc.collect()

    # Compute overlap analysis
    summary = compute_overlap_analysis(results)

    # Write global summary
    write_global_summary(summary, output_dir)

    elapsed = time.time() - start_time

    if verbose:
        print_summary(summary, elapsed)

    return summary


def print_summary(summary: MultiModelSelectionSummary, elapsed_seconds: float) -> None:
    """Print a formatted summary of multi-model selection results."""
    print(f"\n{'='*70}")
    print(f"MULTI-MODEL SELECTION SUMMARY")
    print(f"{'='*70}")

    # Per-model results
    print("\nPER-MODEL RESULTS:")
    print("-" * 50)
    print(f"{'Model':<20} {'Features':>10} {'AUC':>12} {'Stage':<20}")
    print("-" * 50)
    for mk, result in summary.results.items():
        auc_str = f"{result.cv_auc_mean:.4f}±{result.cv_auc_std:.4f}"
        print(f"{mk:<20} {result.n_features:>10} {auc_str:>12} {result.best_stage:<20}")

    # Core features
    print(f"\nCORE FEATURES ({len(summary.core_features)} shared by ALL models):")
    print("-" * 50)
    for i, feat in enumerate(summary.core_features[:10], 1):
        print(f"  {i}. {feat}")
    if len(summary.core_features) > 10:
        print(f"  ... and {len(summary.core_features) - 10} more")

    # Head features per model
    print(f"\nHEAD FEATURES (model-specific):")
    print("-" * 50)
    for mk, head_feats in summary.head_features.items():
        print(f"\n  {mk}: {len(head_feats)} unique features")
        for i, feat in enumerate(head_feats[:5], 1):
            print(f"    {i}. {feat}")
        if len(head_feats) > 5:
            print(f"    ... and {len(head_feats) - 5} more")

    # Overlap matrix
    print(f"\nOVERLAP MATRIX (Jaccard similarity):")
    print("-" * 50)
    model_keys = list(summary.overlap_matrix.keys())
    header = f"{'':20}" + "".join(f"{mk:>15}" for mk in model_keys)
    print(header)
    for mk1 in model_keys:
        row = f"{mk1:20}"
        for mk2 in model_keys:
            jaccard = summary.overlap_matrix[mk1][mk2]
            row += f"{jaccard:>15.3f}"
        print(row)

    # Top shared features
    print(f"\nTOP SHARED FEATURES (by model count):")
    print("-" * 50)
    for feat, count in summary.top_shared_features[:10]:
        print(f"  {count}/4 models: {feat}")

    # Statistics
    print(f"\nSTATISTICS:")
    print("-" * 50)
    print(f"  Total unique features (union): {summary.union_size}")
    print(f"  Core features (intersection): {len(summary.core_features)}")
    print(f"  Run signature: {summary.run_signature}")
    print(f"  Total elapsed time: {elapsed_seconds/60:.1f} minutes")

    print(f"\n{'='*70}")


# =============================================================================
# Cache Support
# =============================================================================

def save_cache(
    cache_path: Path,
    X: pd.DataFrame,
    targets_df: pd.DataFrame,
    sample_weight: Optional[pd.Series],
    sector: Optional[pd.Series],
    valid_features: List[str],
) -> None:
    """Save data to cache for reuse."""
    cache = {
        'X': X,
        'targets_df': targets_df,
        'sample_weight': sample_weight,
        'sector': sector,
        'valid_features': valid_features,
        'timestamp': datetime.utcnow().isoformat(),
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)


def load_cache(cache_path: Path) -> Optional[dict]:
    """Load cached data if available."""
    if not cache_path.exists():
        return None
    with open(cache_path, 'rb') as f:
        return pickle.load(f)


# =============================================================================
# CLI
# =============================================================================

def parse_model_keys(keys_str: str) -> List[ModelKey]:
    """Parse comma-separated model key string."""
    keys = [k.strip().upper() for k in keys_str.split(',')]
    result = []
    for k in keys:
        try:
            result.append(ModelKey[k])
        except KeyError:
            raise ValueError(f"Unknown model key: {k}. Valid: {[m.name for m in ModelKey]}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Model Feature Selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all 4 models
    python scripts/run_feature_selection_multimodel.py

    # Run specific models
    python scripts/run_feature_selection_multimodel.py --model-keys LONG_NORMAL,SHORT_NORMAL

    # Limit max features
    python scripts/run_feature_selection_multimodel.py --max-features 50

    # Use cached data
    python scripts/run_feature_selection_multimodel.py --use-cache
        """
    )

    parser.add_argument(
        '--model-keys',
        type=str,
        default='LONG_NORMAL,LONG_PARABOLIC,SHORT_NORMAL,SHORT_PARABOLIC',
        help='Comma-separated model keys (default: all 4)'
    )
    parser.add_argument(
        '--max-features',
        type=int,
        default=None,
        help='Maximum features to select per model (default: 80)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='artifacts/feature_selection',
        help='Output directory for artifacts'
    )
    parser.add_argument(
        '--use-cache',
        action='store_true',
        help='Use cached data if available'
    )
    parser.add_argument(
        '--cache-path',
        type=str,
        default='.cache/feature_selection_data.pkl',
        help='Path to cache file'
    )
    parser.add_argument(
        '--max-symbols',
        type=int,
        default=5000,
        help='Maximum symbols to include'
    )
    parser.add_argument(
        '--n-folds',
        type=int,
        default=5,
        help='Number of CV folds'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=8,
        help='Number of parallel workers'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce verbosity'
    )
    parser.add_argument(
        '--update-registry',
        action='store_true',
        help='Auto-update base_features.py with new CORE/HEAD features'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview registry changes without modifying files (requires --update-registry)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup when updating registry'
    )

    args = parser.parse_args()

    verbose = not args.quiet
    output_dir = Path(args.output_dir)
    cache_path = Path(args.cache_path)

    # Parse model keys
    model_keys = parse_model_keys(args.model_keys)

    if verbose:
        print(f"Multi-Model Feature Selection")
        print(f"Models: {', '.join(mk.name for mk in model_keys)}")

    # Load data (from cache or fresh)
    if args.use_cache and cache_path.exists():
        if verbose:
            print(f"\nLoading cached data from {cache_path}")
        cache = load_cache(cache_path)
        X = cache['X']
        targets_df = cache['targets_df']
        sample_weight = cache['sample_weight']
        sector = cache['sector']
        if verbose:
            print(f"  Loaded: X={X.shape}, cached at {cache['timestamp']}")
    else:
        X, targets_df, sample_weight, sector = load_and_prepare_data(
            max_symbols=args.max_symbols,
            verbose=verbose,
        )

        # Save cache for reuse
        if args.use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Pre-filter features for cache
            candidate_features = get_all_selectable_features()
            available = set(X.columns)
            candidate_features = [f for f in candidate_features if f in available]
            valid_features = filter_features_by_nan_rate(X[candidate_features], max_rate=0.3)
            save_cache(cache_path, X, targets_df, sample_weight, sector, valid_features)
            if verbose:
                print(f"Data cached to {cache_path}")

    # Create CV config
    cv_config = CVConfig(
        n_splits=args.n_folds,
        scheme=CVScheme.EXPANDING,
        gap=20,
        purge_window=0,
        min_train_samples=1000,
    )

    # Create pipeline config
    pipeline_config = LooseTightConfig(
        run_base_elimination=False,
        epsilon_add_loose=0.0002,
        min_fold_improvement_ratio_loose=0.6,
        max_features_loose=args.max_features or 80,
        epsilon_remove_strict=0.0,
        run_interactions=True,
        max_interactions=8,
        epsilon_add_interaction=0.001,
        epsilon_swap=0.0005,
        max_swap_iterations=50,
        n_jobs=args.n_jobs,
    )

    # Run multi-model selection
    summary = run_multimodel_selection(
        model_keys=model_keys,
        X=X,
        targets_df=targets_df,
        sample_weight=sample_weight,
        max_features=args.max_features,
        output_dir=output_dir,
        cv_config=cv_config,
        pipeline_config=pipeline_config,
        verbose=verbose,
    )

    if verbose:
        print(f"\nArtifacts written to {output_dir}")

    # Auto-update base_features.py if requested
    if args.update_registry:
        if verbose:
            print("\n" + "=" * 70)
            print("AUTO-UPDATE FEATURE REGISTRY")
            print("=" * 70)

        # Compute diff from current registry
        old_head = {mk.value: HEAD_FEATURES.get(mk, []) for mk in ModelKey}
        diff = compute_feature_diff(CORE_FEATURES, old_head, summary)

        if verbose:
            print_feature_diff(diff)

        # Perform update
        if args.dry_run:
            if verbose:
                print("\n[DRY RUN] No changes made to base_features.py")
                print("Run without --dry-run to apply changes.")
        else:
            result = update_base_features_file(
                summary=summary,
                backup=not args.no_backup,
                dry_run=False,
                verbose=verbose,
            )
            if result['success']:
                if verbose:
                    print("\nbase_features.py updated successfully!")
                    if result['backup_path']:
                        print(f"Backup saved to: {result['backup_path']}")
            else:
                print("Warning: Failed to update base_features.py")

    if verbose:
        print("\nDone!")

    return summary


if __name__ == '__main__':
    main()
