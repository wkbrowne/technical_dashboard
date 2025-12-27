#!/usr/bin/env python
"""Run group-first feature selection on the latest feature data.

This script implements group-first feature selection where entire hypothesis
groups are the atomic units of selection rather than individual features.

Pipeline stages:
1. Start with baseline groups (CORE + HEAD)
2. Forward selection from candidate groups
3. Deterministic swaps between groups
4. Backward elimination of non-essential groups
5. Add curated interaction groups

Usage:
    python run_group_selection.py --model long_normal
    python run_group_selection.py --model all --allow-demotions
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

# Set joblib temp folder
JOBLIB_TEMP = Path(__file__).parent / ".joblib_temp"
JOBLIB_TEMP.mkdir(exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = str(JOBLIB_TEMP)

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.model_keys import ModelKey
from src.feature_selection import (
    # Group-first selection
    GroupSelectionConfig,
    run_group_selection,
    select_groups_for_model,
    # Group data structures
    CORE_GROUPS, HEAD_GROUPS, CANDIDATE_GROUPS, INTERACTION_GROUPS,
    get_baseline_groups, get_all_groups,
    validate_group_sizes,
    # Configuration
    ModelConfig, ModelType, TaskType,
    CVConfig, CVScheme,
    SearchConfig,
    MetricConfig, MetricType,
    # Utilities
    validate_data,
    filter_features_by_nan_rate,
)


def compute_scale_pos_weight(y: pd.Series) -> float:
    """Compute scale_pos_weight for LightGBM class balancing."""
    n_positive = (y == 1).sum()
    n_negative = (y == 0).sum()
    return n_negative / n_positive


def split_by_date_holdout(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series | None,
    holdout_pct: float = 0.05,
) -> tuple:
    """Split data temporally, reserving final holdout_pct of dates for evaluation.

    Args:
        X: Feature matrix with date index
        y: Target series
        sample_weight: Optional sample weights
        holdout_pct: Fraction of dates to reserve (0.05 = 5%)

    Returns:
        Tuple of (X_train, y_train, sw_train, X_holdout, y_holdout, sw_holdout, holdout_dates)
    """
    # Get unique dates sorted
    unique_dates = sorted(X.index.unique())
    n_dates = len(unique_dates)

    # Calculate split point
    n_holdout_dates = max(1, int(n_dates * holdout_pct))
    split_idx = n_dates - n_holdout_dates

    train_dates = set(unique_dates[:split_idx])
    holdout_dates = unique_dates[split_idx:]
    holdout_dates_set = set(holdout_dates)

    # Create masks
    train_mask = X.index.isin(train_dates)
    holdout_mask = X.index.isin(holdout_dates_set)

    # Split data
    X_train = X[train_mask].copy()
    y_train = y[train_mask].copy()
    X_holdout = X[holdout_mask].copy()
    y_holdout = y[holdout_mask].copy()

    sw_train = sample_weight[train_mask].copy() if sample_weight is not None else None
    sw_holdout = sample_weight[holdout_mask].copy() if sample_weight is not None else None

    return X_train, y_train, sw_train, X_holdout, y_holdout, sw_holdout, holdout_dates


def evaluate_on_holdout(
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    selected_features: list[str],
    model_threads: int = 1,
    scale_pos_weight: float | None = None,
) -> dict:
    """Evaluate selected features on holdout set.

    Trains a fresh model on all non-holdout data and evaluates on holdout.
    Returns dict with AUC and other metrics.
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

    # Filter to selected features
    available_features = [f for f in selected_features if f in X_holdout.columns]
    if len(available_features) < len(selected_features):
        missing = set(selected_features) - set(available_features)
        print(f"  WARNING: {len(missing)} features missing from holdout: {list(missing)[:5]}...")

    X_eval = X_holdout[available_features].copy()

    # Handle NaNs
    X_eval = X_eval.fillna(0)

    # Simple LightGBM model (same params as selection)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'max_depth': 6,
        'min_data_in_leaf': 100,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': -1,
        'num_threads': model_threads,
    }
    if scale_pos_weight is not None:
        params['scale_pos_weight'] = scale_pos_weight

    # Note: For proper holdout eval, we'd train on train data and predict on holdout.
    # But here we just use the holdout for a simple eval (no training, just predict).
    # For a more rigorous approach, we'd need access to the training data.

    # Since we only have holdout here, we'll do a simple sanity check:
    # Just report class distribution and return NaN for metrics
    # The proper evaluation happens in the main loop where we have access to train data.

    results = {
        'n_samples': len(y_holdout),
        'n_features': len(available_features),
        'positive_rate': y_holdout.mean(),
        'holdout_date_range': f"{X_holdout.index.min()} to {X_holdout.index.max()}",
    }

    return results


def train_and_evaluate_holdout(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    selected_features: list[str],
    model_threads: int = 1,
    scale_pos_weight: float | None = None,
) -> dict:
    """Train on train set, evaluate on holdout set.

    Returns dict with holdout AUC and other metrics.
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

    # Filter to selected features that exist
    available_features = [f for f in selected_features if f in X_train.columns and f in X_holdout.columns]

    X_tr = X_train[available_features].fillna(0)
    X_ho = X_holdout[available_features].fillna(0)

    # LightGBM params (same as selection)
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_data_in_leaf': 100,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': -1,
        'n_jobs': model_threads,
    }
    if scale_pos_weight is not None:
        params['scale_pos_weight'] = scale_pos_weight

    # Train model
    model = lgb.LGBMClassifier(**params, n_estimators=200)
    model.fit(X_tr, y_train)

    # Predict on holdout
    y_pred_proba = model.predict_proba(X_ho)[:, 1]

    # Calculate metrics
    auc = roc_auc_score(y_holdout, y_pred_proba)
    aupr = average_precision_score(y_holdout, y_pred_proba)
    brier = brier_score_loss(y_holdout, y_pred_proba)

    return {
        'holdout_auc': auc,
        'holdout_aupr': aupr,
        'holdout_brier': brier,
        'holdout_n_samples': len(y_holdout),
        'holdout_positive_rate': float(y_holdout.mean()),
        'holdout_date_range': f"{X_holdout.index.min()} to {X_holdout.index.max()}",
        'n_features_used': len(available_features),
    }


def load_and_prepare_data(
    model_key: ModelKey,
    max_symbols: int = 5000,
    min_samples_per_symbol: int = 100,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load and prepare data for group selection.

    Args:
        model_key: The model key to load target for (determines which hit_* column to use)
        max_symbols: Maximum number of symbols to include
        min_samples_per_symbol: Minimum samples per symbol (unused currently)

    Returns:
        Tuple of (X, y, sample_weight)
    """
    print("Loading data...")

    # Load features
    feature_file = 'artifacts/features_filtered.parquet'
    if not Path(feature_file).exists():
        feature_file = 'artifacts/features_complete.parquet'

    features = pd.read_parquet(feature_file)
    print(f"  Loaded features: {features.shape}")

    # Load targets
    targets = pd.read_parquet('artifacts/targets_triple_barrier.parquet')
    print(f"  Loaded targets: {targets.shape}")

    # Rename t0 -> date for merge (targets use t0, features use date)
    if 't0' in targets.columns and 'date' not in targets.columns:
        targets = targets.rename(columns={'t0': 'date'})

    # Determine target column based on model key
    # Target columns are: hit_long_normal, hit_long_parabolic, hit_short_normal, hit_short_parabolic
    target_col = f"hit_{model_key.value}"
    if target_col not in targets.columns:
        raise ValueError(f"Target column '{target_col}' not found in targets. Available: {[c for c in targets.columns if c.startswith('hit_')]}")

    print(f"  Using target column: {target_col}")

    # Merge on date and symbol
    merged = features.merge(targets, on=['date', 'symbol'], how='inner')
    print(f"  Merged: {merged.shape}")

    # Filter to definitive outcomes only (exclude neutral/timeout hit=0)
    # Keep: hit=1 (upper barrier) and hit=-1 (lower barrier)
    merged = merged[merged[target_col] != 0]
    print(f"  After excluding neutral: {len(merged)} samples")

    # Sample if too many symbols
    symbols = merged['symbol'].unique()
    if len(symbols) > max_symbols:
        np.random.seed(42)
        symbols = np.random.choice(symbols, max_symbols, replace=False)
        merged = merged[merged['symbol'].isin(symbols)]
        print(f"  Sampled to {len(symbols)} symbols: {len(merged)} samples")

    # Get feature columns (exclude meta columns and target columns)
    meta_cols = ['date', 'symbol', 'sector']
    target_cols_list = [c for c in merged.columns if c.startswith('hit_') or c.startswith('ret_')
                        or c.startswith('h_used_') or c.startswith('weight') or c.startswith('t_hit')
                        or c in ['entry_px', 'atr_at_entry', 'n_overlapping_trajs']]
    feature_cols = [c for c in merged.columns if c not in meta_cols and c not in target_cols_list]

    # Create X, y
    X = merged[feature_cols].copy()
    # Convert to binary target based on model type:
    # - Long models: hit=1 (upper barrier) -> success
    # - Short models: hit=-1 (lower barrier) -> success
    if model_key.is_short():
        y = (merged[target_col] == -1).astype(int)
    else:
        y = (merged[target_col] == 1).astype(int)

    # Sample weights
    sample_weight = None
    if 'weight_final' in merged.columns:
        sample_weight = merged['weight_final'].copy()

    # Set date as index
    X.index = merged['date']
    y.index = merged['date']
    if sample_weight is not None:
        sample_weight.index = merged['date']

    print(f"\nData prepared for {model_key.value}:")
    print(f"  X shape: {X.shape}")
    print(f"  y distribution: {y.value_counts().to_dict()}")
    print(f"  Date range: {X.index.min()} to {X.index.max()}")

    return X, y, sample_weight


def run_selection_for_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_key: ModelKey,
    sample_weight: pd.Series = None,
    scale_pos_weight: float = None,
    n_folds: int = 5,
    epsilon_add: float = 0.002,
    epsilon_swap: float = 0.001,
    allow_baseline_demotions: bool = False,
    max_groups: int = 20,
    max_interactions: int = 5,
    n_jobs: int = 4,
    model_threads: int = 1,
    verbose: bool = True,
    enable_k_of_n: bool = False,
    group_k_default: int = 5,
    epsilon_add_feature: float = 0.0005,
) -> dict:
    """Run group selection for a single model.

    Returns:
        Dict with selection results
    """
    print(f"\n{'='*70}")
    print(f"GROUP SELECTION FOR: {model_key.value}")
    print(f"{'='*70}")

    # Show baseline info
    baseline_groups = get_baseline_groups(model_key)
    candidate_groups = CANDIDATE_GROUPS
    print(f"\nGroup Configuration:")
    print(f"  Baseline groups: {len(baseline_groups)}")
    print(f"  Candidate groups: {len(candidate_groups)}")
    print(f"  Interaction groups: {len(INTERACTION_GROUPS)}")

    # Configure model
    lgbm_params = {
        'learning_rate': 0.03,
        'max_depth': 5,
        'num_leaves': 31,
        'min_data_in_leaf': 100,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': -1,
    }
    if scale_pos_weight is not None:
        lgbm_params['scale_pos_weight'] = scale_pos_weight

    model_config = ModelConfig(
        model_type=ModelType.LIGHTGBM,
        task_type=TaskType.CLASSIFICATION,
        params=lgbm_params,
        num_threads=model_threads,
    )

    cv_config = CVConfig(
        n_splits=n_folds,
        scheme=CVScheme.EXPANDING,
        gap=20,
        purge_window=2,
    )

    metric_config = MetricConfig(
        primary_metric=MetricType.AUC,
        secondary_metrics=[MetricType.AUPR, MetricType.BRIER, MetricType.LOG_LOSS, MetricType.PRECISION_AT_K],
        tail_quantile=0.1,
    )

    # Print CV and metric configuration
    if verbose:
        print(f"\nCross-Validation Configuration:")
        print(f"  Scheme: {cv_config.scheme.value} (walk-forward)")
        print(f"  Folds: {cv_config.n_splits}")
        print(f"  Embargo gap: {cv_config.gap} DATES (not rows)")
        print(f"  Purge window: {cv_config.purge_window} DATES")
        print(f"\nMetrics Tracked:")
        print(f"  Primary: {metric_config.primary_metric.value}")
        secondary_names = [m.value for m in metric_config.secondary_metrics]
        print(f"  Secondary: {', '.join(secondary_names)}")
        print(f"  Tail quantile: {metric_config.tail_quantile:.0%} (precision@{int(metric_config.tail_quantile*100)})")

    search_config = SearchConfig(
        epsilon_add=epsilon_add,
        epsilon_swap=epsilon_swap,
        max_features=200,  # Group selection doesn't use this directly
        n_jobs=n_jobs,
        forward_selection_n_jobs=n_jobs,
        forward_selection_model_threads=model_threads,
        interaction_n_jobs=n_jobs,
        interaction_num_threads_model=model_threads,
    )

    config = GroupSelectionConfig(
        epsilon_add=epsilon_add,
        epsilon_swap=epsilon_swap,
        epsilon_remove=epsilon_swap,
        allow_baseline_demotions=allow_baseline_demotions,
        max_groups=max_groups,
        max_interaction_groups=max_interactions,
        max_search_iterations=100,  # More thorough swap search
        enable_add_drop_moves=True,  # Enable add/drop during swap phase
        enable_caching=True,  # Cache evaluations for efficiency
        n_jobs=n_jobs,  # Pass parallelism setting
        verbose=verbose,
        # K-of-N feature selection within groups
        enable_k_of_n=enable_k_of_n,
        group_k_default=group_k_default,
        epsilon_add_feature=epsilon_add_feature,
    )

    # Run selection
    result = run_group_selection(
        X=X,
        y=y,
        model_key=model_key,
        model_config=model_config,
        cv_config=cv_config,
        metric_config=metric_config,
        search_config=search_config,
        config=config,
    )

    return result


def save_results(results: dict, output_dir: Path):
    """Save selection results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, result in results.items():
        # Save JSON summary
        result_dict = result.to_dict()
        json_path = output_dir / f"group_selection_{model_name}.json"
        with open(json_path, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        print(f"Saved: {json_path}")

        # Save feature list
        features_path = output_dir / f"selected_features_{model_name}.txt"
        with open(features_path, 'w') as f:
            for feat in result.selected_features:
                f.write(feat + '\n')
        print(f"Saved: {features_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Group-first Feature Selection')

    # Model selection
    parser.add_argument('--model', type=str, default='long_normal',
                        help='Model key: long_normal, long_parabolic, short_normal, short_parabolic, or all')

    # Selection parameters
    parser.add_argument('--epsilon-add', type=float, default=0.002,
                        help='Minimum improvement to add a group (default: 0.002)')
    parser.add_argument('--epsilon-swap', type=float, default=0.001,
                        help='Minimum improvement for swaps (default: 0.001)')
    parser.add_argument('--allow-demotions', action='store_true',
                        help='Allow dropping baseline groups during backward elimination')
    parser.add_argument('--max-groups', type=int, default=20,
                        help='Maximum total groups to select (default: 20)')
    parser.add_argument('--max-interactions', type=int, default=5,
                        help='Maximum interaction groups to add (default: 5)')

    # K-of-N feature selection within groups (enabled by default)
    parser.add_argument('--disable-k-of-n', action='store_true',
                        help='Disable K-of-N selection (use all features per group)')
    parser.add_argument('--group-k', type=int, default=2,
                        help='Default K for K-of-N selection (default: 2)')
    parser.add_argument('--epsilon-add-feature', type=float, default=0.0005,
                        help='Min improvement to add feature within group (default: 0.0005)')

    # Data options
    parser.add_argument('--max-symbols', type=int, default=5000,
                        help='Maximum number of symbols')
    parser.add_argument('--balanced', action='store_true',
                        help='Use class weights (scale_pos_weight)')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--holdout-pct', type=float, default=0.05,
                        help='Fraction of dates to hold out for unbiased evaluation (default: 0.05 = 5%%)')

    # Parallelism
    parser.add_argument('--n-jobs', type=int, default=4,
                        help='Number of parallel jobs (default: 4)')
    parser.add_argument('--model-threads', type=int, default=1,
                        help='Threads per model (default: 1)')

    # Output
    parser.add_argument('--output-dir', type=str, default='artifacts/group_selection',
                        help='Output directory for results')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')

    args = parser.parse_args()

    # Validate group sizes before running
    print("\nValidating group configurations...")
    validation = validate_group_sizes()
    if not validation['valid']:
        print("WARNING: Some groups have invalid sizes:")
        for issue in validation['issues'][:5]:
            print(f"  - {issue}")
        print()

    # Print parallelism config
    total_threads = args.n_jobs * args.model_threads
    print(f"\nParallelism: {args.n_jobs} jobs × {args.model_threads} threads/model = {total_threads} total threads")

    # Print holdout config
    if args.holdout_pct > 0:
        print(f"Holdout: {args.holdout_pct:.1%} of dates reserved for unbiased evaluation")
    else:
        print("Holdout: DISABLED (CV metrics may be optimistically biased)")

    # Determine models to run
    if args.model.lower() == 'all':
        model_keys = list(ModelKey.all_keys())
    else:
        model_name_map = {
            'long_normal': ModelKey.LONG_NORMAL,
            'long_parabolic': ModelKey.LONG_PARABOLIC,
            'short_normal': ModelKey.SHORT_NORMAL,
            'short_parabolic': ModelKey.SHORT_PARABOLIC,
        }
        if args.model.lower() not in model_name_map:
            print(f"ERROR: Unknown model '{args.model}'")
            print(f"Valid options: {', '.join(model_name_map.keys())}, all")
            sys.exit(1)
        model_keys = [model_name_map[args.model.lower()]]

    # Run selection for each model
    results = {}
    holdout_results = {}

    for model_key in model_keys:
        # Load data for this model (each model has different target column)
        X_full, y_full, sample_weight_full = load_and_prepare_data(
            model_key=model_key,
            max_symbols=args.max_symbols,
        )

        # Split into train/holdout if holdout enabled
        X_holdout, y_holdout = None, None
        if args.holdout_pct > 0:
            X, y, sample_weight, X_holdout, y_holdout, sw_holdout, holdout_dates = split_by_date_holdout(
                X_full, y_full, sample_weight_full, holdout_pct=args.holdout_pct
            )
            print(f"\n  Data split: {len(X)} train rows, {len(X_holdout)} holdout rows")
            print(f"  Holdout dates: {holdout_dates[0]} to {holdout_dates[-1]} ({len(holdout_dates)} dates)")
        else:
            X, y, sample_weight = X_full, y_full, sample_weight_full

        # Compute class weight if requested
        scale_pos_weight = None
        if args.balanced:
            scale_pos_weight = compute_scale_pos_weight(y)
            print(f"Class balancing: scale_pos_weight = {scale_pos_weight:.3f}")

        result = run_selection_for_model(
            X=X,
            y=y,
            model_key=model_key,
            sample_weight=sample_weight,
            scale_pos_weight=scale_pos_weight,
            n_folds=args.n_folds,
            epsilon_add=args.epsilon_add,
            epsilon_swap=args.epsilon_swap,
            allow_baseline_demotions=args.allow_demotions,
            max_groups=args.max_groups,
            max_interactions=args.max_interactions,
            n_jobs=args.n_jobs,
            model_threads=args.model_threads,
            verbose=not args.quiet,
            enable_k_of_n=not args.disable_k_of_n,
            group_k_default=args.group_k,
            epsilon_add_feature=args.epsilon_add_feature,
        )
        results[model_key.value] = result

        # Evaluate on holdout if enabled
        if args.holdout_pct > 0 and X_holdout is not None:
            print(f"\n  Evaluating on holdout set...")
            holdout_eval = train_and_evaluate_holdout(
                X_train=X,
                y_train=y,
                X_holdout=X_holdout,
                y_holdout=y_holdout,
                selected_features=result.selected_features,
                model_threads=args.model_threads,
                scale_pos_weight=scale_pos_weight,
            )
            holdout_results[model_key.value] = holdout_eval
            print(f"  Holdout AUC: {holdout_eval['holdout_auc']:.4f} (vs CV: {result.final_metric:.4f})")
            print(f"  Holdout samples: {holdout_eval['holdout_n_samples']}, positive rate: {holdout_eval['holdout_positive_rate']:.1%}")

        # Free memory between models
        del X, y, sample_weight, X_full, y_full, sample_weight_full
        if X_holdout is not None:
            del X_holdout, y_holdout
        gc.collect()

    # Save results
    output_dir = Path(args.output_dir)
    save_results(results, output_dir)

    # Save holdout results if available
    if holdout_results:
        holdout_path = output_dir / "holdout_evaluation.json"
        with open(holdout_path, 'w') as f:
            json.dump(holdout_results, f, indent=2, default=str)
        print(f"Saved holdout results: {holdout_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("SELECTION SUMMARY")
    print(f"{'='*70}")
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Groups: {len(result.selected_groups)}")
        print(f"  Features: {len(result.selected_features)}")
        print()
        print(f"  {'Metric':<15} {'Baseline':>12} {'Final':>12} {'Δ':>10}")
        print(f"  {'-'*51}")
        print(f"  {'AUC (CV)':<15} {result.baseline_metric:>12.4f} {result.final_metric:>12.4f} {result.final_metric - result.baseline_metric:>+10.4f}")

        # Show holdout AUC if available
        if model_name in holdout_results:
            ho = holdout_results[model_name]
            cv_ho_diff = result.final_metric - ho['holdout_auc']
            print(f"  {'AUC (HOLDOUT)':<15} {'-':>12} {ho['holdout_auc']:>12.4f} {'':>10}")
            print(f"  {'CV-Holdout gap':<15} {'-':>12} {'-':>12} {cv_ho_diff:>+10.4f}")

        # Show secondary metrics
        for metric_name in sorted(result.final_secondary_metrics.keys()):
            base_val = result.baseline_secondary_metrics.get(metric_name, (0, 0))[0]
            final_val = result.final_secondary_metrics.get(metric_name, (0, 0))[0]
            delta = final_val - base_val
            print(f"  {metric_name:<15} {base_val:>12.4f} {final_val:>12.4f} {delta:>+10.4f}")

    # Print holdout warning/interpretation
    if holdout_results:
        print(f"\n{'='*70}")
        print("HOLDOUT INTERPRETATION")
        print(f"{'='*70}")
        for model_name, ho in holdout_results.items():
            cv_auc = results[model_name].final_metric
            holdout_auc = ho['holdout_auc']
            gap = cv_auc - holdout_auc
            print(f"\n{model_name}:")
            print(f"  CV AUC:      {cv_auc:.4f}")
            print(f"  Holdout AUC: {holdout_auc:.4f}")
            print(f"  Gap:         {gap:+.4f}")
            if gap > 0.10:
                print(f"  WARNING: Large CV-holdout gap suggests significant selection bias")
            elif gap > 0.05:
                print(f"  NOTICE: Moderate CV-holdout gap - some selection bias present")
            else:
                print(f"  OK: Small gap - feature selection appears robust")

    print(f"\nResults saved to: {output_dir}/")


if __name__ == '__main__':
    main()
