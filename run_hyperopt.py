#!/usr/bin/env python
"""
LightGBM Hyperparameter Optimization using Optuna.

Uses the selected features from feature selection and the same CV configuration
(expanding window with 20-day embargo) to find optimal hyperparameters.

Usage:
    python run_hyperopt.py [--n-trials 100] [--timeout 3600]
"""

import gc
import os
import sys
import time
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime

# Suppress LightGBM feature name warnings
warnings.filterwarnings('ignore', message='.*feature_name.*')
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
# Suppress sklearn feature name validation warning
warnings.filterwarnings('ignore', message='.*does not have valid feature names.*')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.validation')

# Set joblib temp folder to main disk
JOBLIB_TEMP = Path(__file__).parent / ".joblib_temp"
JOBLIB_TEMP.mkdir(exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = str(JOBLIB_TEMP)

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def load_selected_features() -> list[str]:
    """Load the selected features from feature selection output."""
    features_file = Path('artifacts/feature_selection/selected_features.txt')
    if not features_file.exists():
        raise FileNotFoundError(
            f"Selected features file not found: {features_file}\n"
            "Run feature selection first: python run_feature_selection.py"
        )

    features = []
    with open(features_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                features.append(line)

    return features


def load_and_prepare_data(
    selected_features: list[str],
    max_symbols: int = 5000,
    min_samples_per_symbol: int = 5
) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """Load and prepare data for hyperopt.

    Returns:
        Tuple of (X, y, dates) where dates is used for CV splitting.
    """
    print("Loading features...")
    features = pd.read_parquet('artifacts/features_daily.parquet')
    print(f"  Features shape: {features.shape}")

    print("Loading targets...")
    targets = pd.read_parquet('artifacts/targets_triple_barrier.parquet')
    print(f"  Targets shape: {targets.shape}")

    # Merge
    targets = targets.rename(columns={'t0': 'date'})

    # Filter symbols
    symbol_counts = targets.groupby('symbol').size()
    valid_symbols = symbol_counts[symbol_counts >= min_samples_per_symbol].index.tolist()

    if len(valid_symbols) > max_symbols:
        valid_symbols = symbol_counts.loc[valid_symbols].nlargest(max_symbols).index.tolist()

    features = features[features['symbol'].isin(valid_symbols)].copy()
    targets = targets[targets['symbol'].isin(valid_symbols)].copy()

    # Merge
    merged = features.merge(
        targets[['symbol', 'date', 'hit']],
        on=['symbol', 'date'],
        how='inner'
    )

    # Binary target (exclude neutral)
    merged = merged[merged['hit'] != 0].copy()
    merged['target'] = (merged['hit'] == 1).astype(int)

    # Sort by date
    merged = merged.sort_values(['date', 'symbol']).reset_index(drop=True)

    # Check which selected features are available
    available_features = [f for f in selected_features if f in merged.columns]
    missing = set(selected_features) - set(available_features)
    if missing:
        print(f"  Warning: {len(missing)} selected features not in data: {list(missing)[:5]}...")

    print(f"  Using {len(available_features)} features")

    X = merged[available_features].copy()
    y = merged['target'].copy()
    dates = pd.to_datetime(merged['date'])

    print(f"\nFinal data:")
    print(f"  Samples: {len(X):,}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Positive class: {y.sum():,} ({y.mean()*100:.1f}%)")
    print(f"  Date range: {dates.min()} to {dates.max()}")

    del features, targets, merged
    gc.collect()

    return X, y, dates


def get_expanding_cv_splits(
    dates: pd.DatetimeIndex,
    n_splits: int = 5,
    gap: int = 20,
    min_train_samples: int = 1000
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate expanding window CV splits with embargo gap.

    Args:
        dates: DatetimeIndex of all samples
        n_splits: Number of CV folds
        gap: Embargo gap in trading days between train and test
        min_train_samples: Minimum samples required in training set

    Returns:
        List of (train_idx, test_idx) tuples
    """
    unique_dates = np.sort(dates.unique())
    n_dates = len(unique_dates)

    # Calculate test size (equal portions for each fold)
    # Reserve first portion for minimum training
    min_train_dates = max(min_train_samples // 100, 50)  # Rough estimate
    available_dates = n_dates - min_train_dates
    test_size = available_dates // (n_splits + 1)

    splits = []

    for fold in range(n_splits):
        # Test window: from the end, moving backwards
        test_end_idx = n_dates - 1 - fold * test_size
        test_start_idx = test_end_idx - test_size + 1

        # Train window: everything before test, minus gap
        train_end_idx = test_start_idx - gap - 1
        train_start_idx = 0  # Expanding window

        if train_end_idx < min_train_dates:
            continue

        train_dates = unique_dates[train_start_idx:train_end_idx + 1]
        test_dates = unique_dates[test_start_idx:test_end_idx + 1]

        train_mask = dates.isin(train_dates)
        test_mask = dates.isin(test_dates)

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        if len(train_idx) >= min_train_samples and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    # Reverse to get chronological order (oldest train first)
    return splits[::-1]


def objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    n_jobs: int = 8,
    min_fold_auc: float = 0.55,
    variance_penalty: float = 1.0
) -> float:
    """Optuna objective function for LightGBM hyperparameter optimization.

    Uses a stability-adjusted score: mean_auc - variance_penalty * std_auc
    Also requires all folds to exceed min_fold_auc threshold.

    Args:
        trial: Optuna trial object
        X: Feature matrix
        y: Target vector
        cv_splits: List of (train_idx, test_idx) tuples
        n_jobs: Threads per model
        min_fold_auc: Minimum AUC required on each fold (default 0.55)
        variance_penalty: Weight for std penalty (default 1.0)

    Returns:
        Stability-adjusted AUC score (mean - penalty * std)
    """

    # Hyperparameter search space
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 42,
        'num_threads': n_jobs,

        # Tree structure
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 500),

        # Learning
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),

        # Regularization
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),

        # Subsampling
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),

        # Additional
        'max_bin': trial.suggest_categorical('max_bin', [63, 127, 255]),
    }

    # Constraint: num_leaves <= 2^max_depth
    max_leaves = 2 ** params['max_depth']
    if params['num_leaves'] > max_leaves:
        params['num_leaves'] = max_leaves

    # Cross-validation
    fold_aucs = []
    fold_auprs = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Handle NaN/inf
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        # Train
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )

        # Predict
        y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        aupr = average_precision_score(y_test, y_pred)
        fold_aucs.append(auc)
        fold_auprs.append(aupr)

        # Early rejection: if any fold is below minimum threshold, prune
        if auc < min_fold_auc:
            # Store the failure reason for analysis
            trial.set_user_attr('failed_fold', fold_idx + 1)
            trial.set_user_attr('failed_auc', auc)
            raise optuna.TrialPruned()

        # Report intermediate result for pruning
        trial.report(np.mean(fold_aucs), fold_idx)

        # Prune if not promising
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Calculate stability-adjusted score
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    min_auc = np.min(fold_aucs)
    mean_aupr = np.mean(fold_auprs)
    std_aupr = np.std(fold_auprs)

    # Store metrics for analysis
    trial.set_user_attr('mean_auc', mean_auc)
    trial.set_user_attr('std_auc', std_auc)
    trial.set_user_attr('min_auc', min_auc)
    trial.set_user_attr('fold_aucs', fold_aucs)
    trial.set_user_attr('mean_aupr', mean_aupr)
    trial.set_user_attr('std_aupr', std_aupr)
    trial.set_user_attr('fold_auprs', fold_auprs)

    # Stability-adjusted score: penalize high variance
    # This encourages consistent performance across all folds
    stability_score = mean_auc - variance_penalty * std_auc

    return stability_score


def trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Callback to print trial results as they complete."""
    if trial.state == optuna.trial.TrialState.COMPLETE:
        mean_auc = trial.user_attrs.get('mean_auc', trial.value)
        std_auc = trial.user_attrs.get('std_auc', 0)
        mean_aupr = trial.user_attrs.get('mean_aupr', 0)
        std_aupr = trial.user_attrs.get('std_aupr', 0)

        # Check if this is a new best
        is_best = study.best_trial.number == trial.number

        best_marker = " ** NEW BEST **" if is_best else ""
        print(f"Trial {trial.number:3d}: AUC={mean_auc:.4f}±{std_auc:.4f}  AUPR={mean_aupr:.4f}±{std_aupr:.4f}  Score={trial.value:.4f}{best_marker}")
    elif trial.state == optuna.trial.TrialState.PRUNED:
        failed_fold = trial.user_attrs.get('failed_fold', '?')
        failed_auc = trial.user_attrs.get('failed_auc', 0)
        print(f"Trial {trial.number:3d}: PRUNED (fold {failed_fold}, AUC={failed_auc:.4f})")


def run_hyperopt(
    n_trials: int = 100,
    timeout: int = None,
    n_jobs_model: int = 8,
    n_jobs_optuna: int = 1,
    n_folds: int = 5,
    gap: int = 20,
    min_fold_auc: float = 0.55,
    variance_penalty: float = 1.0
) -> dict:
    """Run hyperparameter optimization.

    Args:
        n_trials: Number of Optuna trials
        timeout: Timeout in seconds (None for no timeout)
        n_jobs_model: Threads per LightGBM model
        n_jobs_optuna: Parallel Optuna trials (1 recommended for reproducibility)
        n_folds: Number of CV folds
        gap: Embargo gap in days
        min_fold_auc: Minimum AUC required on each fold
        variance_penalty: Weight for std penalty in objective

    Returns:
        Dictionary with best params and results
    """
    print("=" * 60)
    print("LightGBM Hyperparameter Optimization")
    print("=" * 60)
    print()

    # Load selected features
    print("Loading selected features...")
    selected_features = load_selected_features()
    print(f"  {len(selected_features)} features from feature selection")

    # Load data
    X, y, dates = load_and_prepare_data(selected_features)

    # Convert to numpy for speed
    feature_names = X.columns.tolist()
    X_np = X.values.astype(np.float32)
    y_np = y.values

    # Generate CV splits
    print(f"\nGenerating {n_folds}-fold expanding CV splits with {gap}-day embargo...")
    cv_splits = get_expanding_cv_splits(dates, n_splits=n_folds, gap=gap)
    print(f"  Generated {len(cv_splits)} valid splits")

    for i, (train_idx, test_idx) in enumerate(cv_splits):
        print(f"    Fold {i+1}: {len(train_idx):,} train, {len(test_idx):,} test")

    # Create Optuna study
    print(f"\nStarting Optuna optimization...")
    print(f"  Trials: {n_trials}")
    print(f"  Timeout: {timeout}s" if timeout else "  Timeout: None")
    print(f"  Model threads: {n_jobs_model}")
    print(f"  Min fold AUC: {min_fold_auc}")
    print(f"  Variance penalty: {variance_penalty}")
    print(f"  Objective: mean_auc - {variance_penalty} * std_auc")
    print()

    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=2)

    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name='lightgbm_hyperopt'
    )

    start_time = time.time()

    study.optimize(
        lambda trial: objective(
            trial, X_np, y_np, cv_splits, n_jobs_model,
            min_fold_auc=min_fold_auc,
            variance_penalty=variance_penalty
        ),
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs_optuna,
        show_progress_bar=False,  # Disabled since we have custom callback output
        gc_after_trial=True,
        callbacks=[trial_callback]
    )

    elapsed = time.time() - start_time

    # Results
    print()
    print("=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print()
    print(f"Completed trials: {len(study.trials)}")
    print(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print()

    # Get best trial's actual metrics
    best_trial = study.best_trial
    best_mean_auc = best_trial.user_attrs.get('mean_auc', study.best_value)
    best_std_auc = best_trial.user_attrs.get('std_auc', 0)
    best_min_auc = best_trial.user_attrs.get('min_auc', 0)
    best_fold_aucs = best_trial.user_attrs.get('fold_aucs', [])

    print(f"Best Stability Score: {study.best_value:.4f}")
    print(f"  (mean_auc - {variance_penalty} * std_auc)")
    print()
    print(f"Best Trial Metrics:")
    print(f"  Mean AUC: {best_mean_auc:.4f}")
    print(f"  Std AUC:  {best_std_auc:.4f}")
    print(f"  Min AUC:  {best_min_auc:.4f}")
    if best_fold_aucs:
        print(f"  Fold AUCs: {[f'{auc:.4f}' for auc in best_fold_aucs]}")
    print()
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    # Save results
    output_dir = Path('artifacts/hyperopt')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save best params
    best_params = study.best_params.copy()
    best_params['best_auc'] = study.best_value
    best_params['n_trials'] = len(study.trials)
    best_params['elapsed_seconds'] = elapsed
    best_params['timestamp'] = datetime.now().isoformat()
    best_params['features'] = feature_names

    params_file = output_dir / 'best_params.json'
    with open(params_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest params saved to: {params_file}")

    # Save trial history
    trials_df = study.trials_dataframe()
    trials_file = output_dir / 'trials_history.csv'
    trials_df.to_csv(trials_file, index=False)
    print(f"Trial history saved to: {trials_file}")

    # Parameter importance
    try:
        importance = optuna.importance.get_param_importances(study)
        print("\nParameter importance:")
        for param, imp in sorted(importance.items(), key=lambda x: -x[1])[:10]:
            print(f"  {param}: {imp:.3f}")

        importance_file = output_dir / 'param_importance.json'
        with open(importance_file, 'w') as f:
            json.dump(importance, f, indent=2)
    except Exception as e:
        print(f"  Could not compute importance: {e}")

    # Final validation with best params
    print()
    print("=" * 60)
    print("FINAL VALIDATION WITH BEST PARAMS")
    print("=" * 60)

    # Reconstruct full params
    final_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 42,
        'num_threads': n_jobs_model,
        **study.best_params
    }

    # Ensure num_leaves constraint
    max_leaves = 2 ** final_params['max_depth']
    if final_params['num_leaves'] > max_leaves:
        final_params['num_leaves'] = max_leaves

    # Run final CV with verbose output
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        X_train, X_test = X_np[train_idx], X_np[test_idx]
        y_train, y_test = y_np[train_idx], y_np[test_idx]

        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        model = lgb.LGBMClassifier(**final_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )

        # Train metrics
        y_train_pred = model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_pred)
        train_aupr = average_precision_score(y_train, y_train_pred)

        # Test metrics
        y_test_pred = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_test_pred)
        test_aupr = average_precision_score(y_test, y_test_pred)

        fold_results.append({
            'fold': fold_idx + 1,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'train_aupr': train_aupr,
            'test_aupr': test_aupr,
            'gap': train_auc - test_auc,
            'best_iteration': model.best_iteration_
        })

        print(f"  Fold {fold_idx + 1}: Train AUC={train_auc:.4f} AUPR={train_aupr:.4f}, Test AUC={test_auc:.4f} AUPR={test_aupr:.4f}, Gap={train_auc - test_auc:.4f}")

    mean_train_auc = np.mean([r['train_auc'] for r in fold_results])
    mean_test_auc = np.mean([r['test_auc'] for r in fold_results])
    std_test_auc = np.std([r['test_auc'] for r in fold_results])
    mean_train_aupr = np.mean([r['train_aupr'] for r in fold_results])
    mean_test_aupr = np.mean([r['test_aupr'] for r in fold_results])
    std_test_aupr = np.std([r['test_aupr'] for r in fold_results])
    mean_gap = np.mean([r['gap'] for r in fold_results])

    print()
    print(f"Final Results:")
    print(f"  Train AUC:  {mean_train_auc:.4f}")
    print(f"  Test AUC:   {mean_test_auc:.4f} ± {std_test_auc:.4f}")
    print(f"  Train AUPR: {mean_train_aupr:.4f}")
    print(f"  Test AUPR:  {mean_test_aupr:.4f} ± {std_test_aupr:.4f}")
    print(f"  Gap (AUC):  {mean_gap:.4f}")

    # Save final results
    final_results = {
        'best_params': study.best_params,
        'fold_results': fold_results,
        'mean_train_auc': mean_train_auc,
        'mean_test_auc': mean_test_auc,
        'std_test_auc': std_test_auc,
        'mean_train_aupr': mean_train_aupr,
        'mean_test_aupr': mean_test_aupr,
        'std_test_aupr': std_test_aupr,
        'mean_gap': mean_gap,
    }

    final_file = output_dir / 'final_results.json'
    with open(final_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    print(f"\nFinal results saved to: {final_file}")

    return final_results


def main():
    parser = argparse.ArgumentParser(description='LightGBM Hyperparameter Optimization')
    parser.add_argument('--n-trials', type=int, default=100, help='Number of Optuna trials')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds')
    parser.add_argument('--n-jobs-model', type=int, default=8, help='Threads per model')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--gap', type=int, default=20, help='Embargo gap in days')
    parser.add_argument('--min-fold-auc', type=float, default=0.55,
                        help='Minimum AUC required on each fold (trials below this are pruned)')
    parser.add_argument('--variance-penalty', type=float, default=1.0,
                        help='Penalty weight for std in objective (score = mean - penalty * std)')

    args = parser.parse_args()

    results = run_hyperopt(
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs_model=args.n_jobs_model,
        n_folds=args.n_folds,
        gap=args.gap,
        min_fold_auc=args.min_fold_auc,
        variance_penalty=args.variance_penalty
    )

    return results


if __name__ == '__main__':
    main()
