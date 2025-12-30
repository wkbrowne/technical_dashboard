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
    # Outer CV for robustness
    OuterCVResult,
    StabilityAggregationResult,
    run_outer_cv_with_finalization,
    # Group data structures
    CORE_GROUPS, HEAD_GROUPS, CANDIDATE_GROUPS, INTERACTION_TEMPLATES,
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
from src.features.registry import (
    build_registry_from_selection,
    save_registry,
    get_registry_summary,
)
from src.feature_selection.base_features import generate_interaction_feature_names


def validate_feature_availability(X: pd.DataFrame, model_key: ModelKey) -> bool:
    """Validate that all declared features exist in the data.

    Prints a BIG FAT WARNING if any features are missing.

    Args:
        X: Feature DataFrame
        model_key: Model key for HEAD_GROUPS lookup

    Returns:
        True if all features present, False if any missing
    """
    available = set(X.columns)
    missing_by_source = {}

    # Check CORE_GROUPS
    for group_name, features in CORE_GROUPS.items():
        missing = [f for f in features if f not in available]
        if missing:
            missing_by_source[f"CORE_GROUPS['{group_name}']"] = missing

    # Check HEAD_GROUPS for this model
    head_groups = HEAD_GROUPS.get(model_key, {})
    for group_name, features in head_groups.items():
        missing = [f for f in features if f not in available]
        if missing:
            missing_by_source[f"HEAD_GROUPS[{model_key.value}]['{group_name}']"] = missing

    # Check CANDIDATE_GROUPS
    for group_name, features in CANDIDATE_GROUPS.items():
        missing = [f for f in features if f not in available]
        if missing:
            missing_by_source[f"CANDIDATE_GROUPS['{group_name}']"] = missing

    # Check INTERACTION_TEMPLATES (base/gate features, not generated names)
    for template_name, template in INTERACTION_TEMPLATES.items():
        base_feats = template.get("base_features", [])
        gate_feats = template.get("gate_features", [])
        missing_base = [f for f in base_feats if f not in available]
        missing_gate = [f for f in gate_feats if f not in available]
        if missing_base:
            missing_by_source[f"INTERACTION_TEMPLATES['{template_name}'].base_features"] = missing_base
        if missing_gate:
            missing_by_source[f"INTERACTION_TEMPLATES['{template_name}'].gate_features"] = missing_gate

    if missing_by_source:
        # Count totals
        total_missing = sum(len(v) for v in missing_by_source.values())
        unique_missing = set()
        for feats in missing_by_source.values():
            unique_missing.update(feats)

        print()
        print("!" * 80)
        print("!" * 80)
        print("!!  FEATURE AVAILABILITY WARNING")
        print("!" * 80)
        print(f"!!  {len(unique_missing)} UNIQUE FEATURES MISSING FROM DATA")
        print(f"!!  {len(missing_by_source)} groups/templates affected")
        print("!" * 80)
        print()

        for source, missing in sorted(missing_by_source.items()):
            print(f"  {source}:")
            for feat in missing[:10]:
                print(f"    - {feat}")
            if len(missing) > 10:
                print(f"    ... and {len(missing) - 10} more")
            print()

        print("!" * 80)
        print("!!  FIX: Update base_features.py or regenerate features_complete.parquet")
        print("!" * 80)
        print()

        return False

    return True


def compute_scale_pos_weight(y: pd.Series) -> float:
    """Compute scale_pos_weight for LightGBM class balancing."""
    n_positive = (y == 1).sum()
    n_negative = (y == 0).sum()
    return n_negative / n_positive


def split_by_date_holdout(
    X: pd.DataFrame,
    y: pd.Series,
    holdout_pct: float = 0.05,
) -> tuple:
    """Split data temporally, reserving final holdout_pct of dates for evaluation.

    Args:
        X: Feature matrix with date index
        y: Target series
        holdout_pct: Fraction of dates to reserve (0.05 = 5%)

    Returns:
        Tuple of (X_train, y_train, X_holdout, y_holdout, holdout_dates)
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

    return X_train, y_train, X_holdout, y_holdout, holdout_dates


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
) -> tuple[pd.DataFrame, pd.Series]:
    """Load and prepare data for group selection.

    Args:
        model_key: The model key to load target for (determines which hit_* column to use)
        max_symbols: Maximum number of symbols to include
        min_samples_per_symbol: Minimum samples per symbol (unused currently)

    Returns:
        Tuple of (X, y)
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

    # Set date as index
    X.index = merged['date']
    y.index = merged['date']

    print(f"\nData prepared for {model_key.value}:")
    print(f"  X shape: {X.shape}")
    print(f"  y distribution: {y.value_counts().to_dict()}")
    print(f"  Date range: {X.index.min()} to {X.index.max()}")

    return X, y


def run_selection_for_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_key: ModelKey,
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
    print(f"  Interaction templates: {len(INTERACTION_TEMPLATES)}")

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


def run_outer_cv_for_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_key: ModelKey,
    scale_pos_weight: float = None,
    n_folds: int = 5,
    n_outer_folds: int = 3,
    outer_test_frac: float = 0.10,
    holdout_frac: float = 0.05,
    # Selection params: None = use GroupSelectionConfig defaults
    epsilon_add: float = None,
    epsilon_swap: float = None,
    allow_baseline_demotions: bool = False,
    max_groups: int = None,
    max_interactions: int = None,
    n_jobs: int = 4,
    model_threads: int = 1,
    verbose: bool = True,
    inner_verbose: bool = False,
    debug_acceptance: bool = False,
    enable_k_of_n: bool = False,
    group_k_default: int = None,
    epsilon_add_feature: float = None,
):
    """Run outer CV selection with stability-based aggregation.

    This runs feature selection N times (default 3), each on a different
    temporal split, then keeps groups selected in >= 2/3 of the folds.

    Returns:
        Tuple of (GroupSelectionResult-like object, OuterCVResult, StabilityAggregationResult)
    """
    from src.feature_selection.group_selection import GroupSelectionResult
    import math

    print(f"\n{'='*70}")
    print(f"OUTER CV SELECTION FOR: {model_key.value}")
    print(f"{'='*70}")

    # Show baseline info
    baseline_groups = get_baseline_groups(model_key)
    print(f"\nGroup Configuration:")
    print(f"  Baseline groups: {len(baseline_groups)}")
    print(f"  Candidate groups: {len(CANDIDATE_GROUPS)}")
    print(f"  Outer CV folds: {n_outer_folds}")
    print(f"  Stability threshold: >= {math.ceil(n_outer_folds * 2 / 3)}/{n_outer_folds} folds")

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
        secondary_metrics=[MetricType.AUPR, MetricType.BRIER, MetricType.LOG_LOSS],
        tail_quantile=0.1,
    )

    search_config = SearchConfig(
        n_jobs=n_jobs,
        random_state=42,
    )

    # Build config kwargs, only including non-None values so config defaults apply
    config_kwargs = {
        'allow_baseline_demotions': allow_baseline_demotions,
        'max_search_iterations': 100,
        'enable_add_drop_moves': True,
        'enable_caching': True,
        'n_jobs': n_jobs,
        'verbose': inner_verbose,  # Controlled by --verbose flag
        'debug_acceptance': debug_acceptance,  # Controlled by --debug-acceptance flag
        'enable_k_of_n': enable_k_of_n,
    }
    # Only set these if explicitly provided (otherwise use GroupSelectionConfig defaults)
    if epsilon_add is not None:
        config_kwargs['epsilon_add'] = epsilon_add
    if epsilon_swap is not None:
        config_kwargs['epsilon_swap'] = epsilon_swap
        config_kwargs['epsilon_remove'] = epsilon_swap  # Use same as swap
    if max_groups is not None:
        config_kwargs['max_groups'] = max_groups
    if max_interactions is not None:
        config_kwargs['max_interaction_groups'] = max_interactions
    if group_k_default is not None:
        config_kwargs['group_k_default'] = group_k_default
    if epsilon_add_feature is not None:
        config_kwargs['epsilon_add_feature'] = epsilon_add_feature

    group_config = GroupSelectionConfig(**config_kwargs)

    # Print effective config values
    print(f"\nSelection Config (from GroupSelectionConfig defaults + CLI overrides):")
    print(f"  epsilon_add: {group_config.epsilon_add:.4f}")
    print(f"  epsilon_swap: {group_config.epsilon_swap:.4f}")
    print(f"  max_groups: {group_config.max_groups}")
    print(f"  max_interaction_groups: {group_config.max_interaction_groups}")
    print(f"  K-of-N: {'enabled' if group_config.enable_k_of_n else 'disabled'}"
          f"{f' (K={group_config.group_k_default})' if group_config.enable_k_of_n else ''}")

    start_time = time.time()

    # Run outer CV with finalization
    outer_cv_result, agg_result = run_outer_cv_with_finalization(
        X=X,
        y=y,
        model_key=model_key,
        model_config=model_config,
        cv_config=cv_config,
        metric_config=metric_config,
        search_config=search_config,
        group_config=group_config,
        n_outer_splits=n_outer_folds,
        test_frac=outer_test_frac,
        final_holdout_frac=holdout_frac,
        allow_baseline_demotions=allow_baseline_demotions,
        verbose=verbose,
    )

    total_time = time.time() - start_time

    # Convert chosen_groups list to dict format {group_name: [features]}
    all_groups = get_all_groups(model_key)
    selected_groups_dict = {}
    for group_name in agg_result.chosen_groups:
        if group_name in all_groups:
            # Get features that exist in the data
            group_features = [f for f in all_groups[group_name] if f in X.columns]
            selected_groups_dict[group_name] = group_features

    # Get baseline AUC from first outer fold (as reference)
    baseline_auc = outer_cv_result.outer_fold_results[0].baseline_auc if outer_cv_result.outer_fold_results else 0.5

    # Create a compatible result object
    result = GroupSelectionResult(
        selected_groups=selected_groups_dict,
        selected_features=agg_result.chosen_features,
        final_metric=agg_result.holdout_auc,
        baseline_metric=baseline_auc,
        final_secondary_metrics={},  # Not tracked in outer CV mode
        baseline_secondary_metrics={},
        total_time_seconds=total_time,
    )

    # Add outer CV specific info to result for later use
    result._outer_cv_result = outer_cv_result
    result._stability_result = agg_result

    return result, outer_cv_result, agg_result


def save_results(results: dict, output_dir: Path, holdout_results: dict = None):
    """Save selection results to disk, including feature registries."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, result in results.items():
        # Save JSON summary (legacy format for backwards compatibility)
        result_dict = result.to_dict()
        json_path = output_dir / f"group_selection_{model_name}.json"
        with open(json_path, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        print(f"Saved: {json_path}")

        # Save feature list (legacy format)
        features_path = output_dir / f"selected_features_{model_name}.txt"
        with open(features_path, 'w') as f:
            for feat in result.selected_features:
                f.write(feat + '\n')
        print(f"Saved: {features_path}")

        # Build and save feature registry
        # Include selection metadata for traceability
        selection_metadata = {
            "final_metric": result.final_metric,
            "baseline_metric": result.baseline_metric,
            "improvement": result.final_metric - result.baseline_metric,
            "n_groups": len(result.selected_groups),
            "total_groups_evaluated": result.total_groups_evaluated,
            "total_time_seconds": result.total_time_seconds,
        }

        # Add holdout metrics if available
        if holdout_results and model_name in holdout_results:
            selection_metadata["holdout_auc"] = holdout_results[model_name].get("holdout_auc")
            selection_metadata["holdout_aupr"] = holdout_results[model_name].get("holdout_aupr")

        registry = build_registry_from_selection(
            model_name=model_name,
            selected_groups=result.selected_groups,
            groups=None,  # Don't include full groups to keep file compact
            selection_metadata=selection_metadata,
        )

        # Save to model-specific directory
        registry_dir = Path("artifacts") / model_name
        registry_dir.mkdir(parents=True, exist_ok=True)
        registry_path = registry_dir / "features.json"
        save_registry(registry, registry_path)

        # Print summary
        summary = get_registry_summary(registry)
        print(f"Registry: {summary}")
        print(f"Saved: {registry_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Group-first Feature Selection')

    # Model selection
    parser.add_argument('--model', type=str, default='long_normal',
                        help='Model key: long_normal, long_parabolic, short_normal, short_parabolic, or all')

    # Selection parameters (defaults from GroupSelectionConfig if not specified)
    parser.add_argument('--epsilon-add', type=float, default=None,
                        help='Minimum improvement to add a group (config default: 0.0003)')
    parser.add_argument('--epsilon-swap', type=float, default=None,
                        help='Minimum improvement for swaps (config default: 0.0003)')
    parser.add_argument('--allow-demotions', action='store_true',
                        help='Allow dropping baseline groups during backward elimination')
    parser.add_argument('--max-groups', type=int, default=None,
                        help='Maximum total groups to select (config default: 20)')
    parser.add_argument('--max-interactions', type=int, default=None,
                        help='Maximum interaction groups to add (config default: 3)')

    # K-of-N feature selection within groups (enabled by default)
    parser.add_argument('--disable-k-of-n', action='store_true',
                        help='Disable K-of-N selection (use all features per group)')
    parser.add_argument('--group-k', type=int, default=None,
                        help='Default K for K-of-N selection (config default: 2)')
    parser.add_argument('--epsilon-add-feature', type=float, default=None,
                        help='Min improvement to add feature within group (config default: 0.0005)')

    # Data options
    parser.add_argument('--max-symbols', type=int, default=5000,
                        help='Maximum number of symbols')
    parser.add_argument('--balanced', action='store_true',
                        help='Use class weights (scale_pos_weight)')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of CV folds (inner CV)')
    parser.add_argument('--holdout-pct', type=float, default=0.05,
                        help='Fraction of dates to hold out for unbiased evaluation (default: 0.05 = 5%%)')

    # Outer CV for stability (enabled by default)
    parser.add_argument('--disable-outer-cv', action='store_true',
                        help='Disable outer CV stability selection (use single-run selection instead)')
    parser.add_argument('--n-outer-folds', type=int, default=3,
                        help='Number of outer CV folds (default: 3)')
    parser.add_argument('--outer-test-frac', type=float, default=0.10,
                        help='Fraction of data per outer test fold (default: 0.10)')

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
    parser.add_argument('--verbose', action='store_true',
                        help='Enable detailed progress logging (shows each group evaluation and accept/reject reasons)')
    parser.add_argument('--debug-acceptance', action='store_true',
                        help='Show detailed SNR acceptance diagnostics for each move')

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

    # Print outer CV config
    if not args.disable_outer_cv:
        print(f"\nOuter CV: ENABLED ({args.n_outer_folds} folds, {args.outer_test_frac:.0%} test per fold)")
        print(f"  - Groups selected in >= {(args.n_outer_folds * 2 + 2) // 3}/{args.n_outer_folds} folds kept (2/3 consensus)")
        print(f"  - Final holdout: {args.holdout_pct:.1%} of dates for unbiased evaluation")
    else:
        print(f"\nOuter CV: DISABLED (single-run selection)")
        # Print holdout config for non-outer-cv mode
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
    outer_cv_results = {}  # Store outer CV results when enabled

    for model_key in model_keys:
        # Load data for this model (each model has different target column)
        X_full, y_full = load_and_prepare_data(
            model_key=model_key,
            max_symbols=args.max_symbols,
        )

        # Validate all declared features exist in the data
        print(f"\nValidating feature availability for {model_key.value}...")
        features_valid = validate_feature_availability(X_full, model_key)
        if features_valid:
            print(f"  ✓ All declared features present in data")
        else:
            # Ask user if they want to continue
            print("WARNING: Continuing with missing features may cause errors or silent failures.")
            if not args.quiet:
                response = input("Continue anyway? [y/N]: ").strip().lower()
                if response != 'y':
                    print("Aborting. Fix feature definitions or regenerate data.")
                    sys.exit(1)

        # Compute class weight if requested
        scale_pos_weight = None
        if args.balanced:
            scale_pos_weight = compute_scale_pos_weight(y_full)
            print(f"Class balancing: scale_pos_weight = {scale_pos_weight:.3f}")

        if not args.disable_outer_cv:
            # OUTER CV MODE (default): Run 3-fold outer CV with stability aggregation
            # Outer CV handles holdout internally
            result, outer_cv_result, stability_result = run_outer_cv_for_model(
                X=X_full,
                y=y_full,
                model_key=model_key,
                scale_pos_weight=scale_pos_weight,
                n_folds=args.n_folds,
                n_outer_folds=args.n_outer_folds,
                outer_test_frac=args.outer_test_frac,
                holdout_frac=args.holdout_pct,
                epsilon_add=args.epsilon_add,
                epsilon_swap=args.epsilon_swap,
                allow_baseline_demotions=args.allow_demotions,
                max_groups=args.max_groups,
                max_interactions=args.max_interactions,
                n_jobs=args.n_jobs,
                model_threads=args.model_threads,
                verbose=not args.quiet,
                inner_verbose=args.verbose,
                debug_acceptance=args.debug_acceptance,
                enable_k_of_n=not args.disable_k_of_n,
                group_k_default=args.group_k,
                epsilon_add_feature=args.epsilon_add_feature,
            )
            results[model_key.value] = result
            outer_cv_results[model_key.value] = {
                'outer_cv': outer_cv_result.to_dict(),
                'stability': stability_result.to_dict(),
            }

            # Holdout is already evaluated internally by outer CV
            holdout_results[model_key.value] = {
                'holdout_auc': stability_result.holdout_auc,
                'chosen_set': stability_result.chosen_set_name,
                'decision_reason': stability_result.decision_reason,
                'group_frequency': stability_result.group_frequency,
            }

        else:
            # SINGLE-RUN MODE: Original behavior with manual holdout
            X_holdout, y_holdout = None, None
            if args.holdout_pct > 0:
                X, y, X_holdout, y_holdout, holdout_dates = split_by_date_holdout(
                    X_full, y_full, holdout_pct=args.holdout_pct
                )
                print(f"\n  Data split: {len(X)} train rows, {len(X_holdout)} holdout rows")
                print(f"  Holdout dates: {holdout_dates[0]} to {holdout_dates[-1]} ({len(holdout_dates)} dates)")
            else:
                X, y = X_full, y_full

            result = run_selection_for_model(
                X=X,
                y=y,
                model_key=model_key,
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

            if X_holdout is not None:
                del X_holdout, y_holdout

        # Free memory between models
        del X_full, y_full
        gc.collect()

    # Save results (including feature registries)
    output_dir = Path(args.output_dir)
    save_results(results, output_dir, holdout_results=holdout_results)

    # Save holdout results if available
    if holdout_results:
        holdout_path = output_dir / "holdout_evaluation.json"
        with open(holdout_path, 'w') as f:
            json.dump(holdout_results, f, indent=2, default=str)
        print(f"Saved holdout results: {holdout_path}")

    # Save outer CV results if available
    if outer_cv_results:
        outer_cv_path = output_dir / "outer_cv_results.json"
        with open(outer_cv_path, 'w') as f:
            json.dump(outer_cv_results, f, indent=2, default=str)
        print(f"Saved outer CV results: {outer_cv_path}")

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
        if outer_cv_results:
            print("OUTER CV STABILITY RESULTS")
        else:
            print("HOLDOUT INTERPRETATION")
        print(f"{'='*70}")
        for model_name, ho in holdout_results.items():
            cv_auc = results[model_name].final_metric
            holdout_auc = ho['holdout_auc']
            gap = cv_auc - holdout_auc
            print(f"\n{model_name}:")

            # Show outer CV specific info
            if 'chosen_set' in ho:
                print(f"  Selection: {ho['chosen_set']} set (2/3 consensus)")
                print(f"  Reason: {ho['decision_reason']}")
                print(f"  Group frequency across outer folds:")
                for group, freq in sorted(ho['group_frequency'].items(), key=lambda x: -x[1]):
                    print(f"    {group}: {freq}/{args.n_outer_folds} folds")
                print()

            print(f"  Holdout AUC: {holdout_auc:.4f}")
            if not outer_cv_results:
                # Only show CV-holdout gap for single-run mode
                print(f"  CV AUC:      {cv_auc:.4f}")
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
