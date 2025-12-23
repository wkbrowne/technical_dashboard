#!/usr/bin/env python
"""
LightGBM Hyperparameter Tuning using Optuna.

Optimizes a composite objective of financial metrics:
- Discrimination: AUC + AUPR
- Calibration: Brier score
- Tail performance: Precision@10%, Spread
- Stability: CV coefficient

Uses sample weights from triple barrier overlap inverse weighting.

Usage:
    python run_model_tuning.py [--n-trials 200] [--n-jobs 8]
    python run_model_tuning.py --objective auc  # AUC-only mode
    python run_model_tuning.py --resume  # Resume interrupted study
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
from dataclasses import dataclass
from typing import Optional

warnings.filterwarnings('ignore', message='.*feature_name.*')
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
warnings.filterwarnings('ignore', message='.*does not have valid feature names.*')

# Set joblib temp folder
JOBLIB_TEMP = Path(__file__).parent / ".joblib_temp"
JOBLIB_TEMP.mkdir(exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = str(JOBLIB_TEMP)

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

sys.path.insert(0, str(Path(__file__).parent))


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ObjectiveWeights:
    """Weights for composite objective function."""
    auc: float = 0.20
    aupr: float = 0.15
    calibration: float = 0.15
    tail: float = 0.20
    spread: float = 0.10
    stability: float = 0.20


@dataclass
class PruningConfig:
    """Thresholds for early trial pruning."""
    min_auc: float = 0.54
    max_brier: float = 0.26
    max_cv_coef: float = 0.20


DEFAULT_WEIGHTS = ObjectiveWeights()
DEFAULT_PRUNING = PruningConfig()


# =============================================================================
# Data Loading
# =============================================================================

def load_selected_features() -> list[str]:
    """Load selected features from feature selection output."""
    features_file = Path('artifacts/feature_selection/selected_features.txt')
    if not features_file.exists():
        raise FileNotFoundError(
            f"Selected features not found: {features_file}\n"
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex, list[str]]:
    """
    Load features, targets, and sample weights.

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Binary target array
        sample_weight: Overlap inverse weights
        dates: DatetimeIndex for CV splitting
        feature_names: List of feature names
    """
    print("Loading features...")
    features = pd.read_parquet('artifacts/features_complete.parquet')
    print(f"  Features shape: {features.shape}")

    print("Loading targets...")
    targets = pd.read_parquet('artifacts/targets_triple_barrier.parquet')
    print(f"  Targets shape: {targets.shape}")

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
        targets[['symbol', 'date', 'hit', 'weight_final']],
        on=['symbol', 'date'],
        how='inner'
    )

    # Binary target (exclude neutral hit=0)
    merged = merged[merged['hit'] != 0].copy()
    merged['target'] = (merged['hit'] == 1).astype(int)
    merged = merged.sort_values(['date', 'symbol']).reset_index(drop=True)

    # Check available features
    available_features = [f for f in selected_features if f in merged.columns]
    missing = set(selected_features) - set(available_features)
    if missing:
        print(f"  Warning: {len(missing)} features not in data: {list(missing)[:5]}...")

    print(f"  Using {len(available_features)} features")

    X = merged[available_features].values.astype(np.float32)
    y = merged['target'].values
    sample_weight = merged['weight_final'].values if 'weight_final' in merged.columns else None
    dates = pd.to_datetime(merged['date'])

    print(f"\nData prepared:")
    print(f"  Samples: {len(X):,}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Positive rate: {y.mean()*100:.1f}%")
    print(f"  Date range: {dates.min().date()} to {dates.max().date()}")
    if sample_weight is not None:
        print(f"  Sample weights: min={sample_weight.min():.3f}, max={sample_weight.max():.3f}")

    del features, targets, merged
    gc.collect()

    return X, y, sample_weight, dates, available_features


# =============================================================================
# Cross-Validation
# =============================================================================

def get_expanding_cv_splits(
    dates: pd.DatetimeIndex,
    n_splits: int = 5,
    gap: int = 20,
    min_train_samples: int = 3000
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate expanding window CV splits with embargo gap.

    gap=20 matches max_horizon from triple barrier targets.
    """
    unique_dates = np.sort(dates.unique())
    n_dates = len(unique_dates)

    min_train_dates = max(min_train_samples // 100, 50)
    available_dates = n_dates - min_train_dates
    test_size = available_dates // (n_splits + 1)

    splits = []
    for fold in range(n_splits):
        test_end_idx = n_dates - 1 - fold * test_size
        test_start_idx = test_end_idx - test_size + 1
        train_end_idx = test_start_idx - gap - 1

        if train_end_idx < min_train_dates:
            continue

        train_dates = unique_dates[:train_end_idx + 1]
        test_dates = unique_dates[test_start_idx:test_end_idx + 1]

        train_mask = dates.isin(train_dates)
        test_mask = dates.isin(test_dates)

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        if len(train_idx) >= min_train_samples and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits[::-1]  # Chronological order


# =============================================================================
# Metrics
# =============================================================================

def compute_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all financial metrics for one CV fold."""
    n = len(y_pred)
    k = max(1, int(n * 0.10))

    sorted_idx = np.argsort(y_pred)
    top_idx = sorted_idx[-k:]
    bottom_idx = sorted_idx[:k]

    return {
        'auc': roc_auc_score(y_true, y_pred),
        'aupr': average_precision_score(y_true, y_pred),
        'brier': brier_score_loss(y_true, np.clip(y_pred, 0, 1)),
        'precision_top_10': y_true[top_idx].mean(),
        'precision_bottom_10': y_true[bottom_idx].mean(),
    }


def compute_composite_score(fold_metrics: list[dict], weights: ObjectiveWeights) -> float:
    """Compute weighted composite objective from fold metrics."""
    auc_mean = np.mean([m['auc'] for m in fold_metrics])
    auc_std = np.std([m['auc'] for m in fold_metrics])
    aupr_mean = np.mean([m['aupr'] for m in fold_metrics])
    brier_mean = np.mean([m['brier'] for m in fold_metrics])
    prec_top = np.mean([m['precision_top_10'] for m in fold_metrics])
    prec_bot = np.mean([m['precision_bottom_10'] for m in fold_metrics])

    score = 0.0

    # Discrimination (35%)
    score += weights.auc * auc_mean
    score += weights.aupr * aupr_mean

    # Calibration (15%) - Brier in [0, 0.25], invert
    score += weights.calibration * (1 - brier_mean / 0.25)

    # Tail performance (30%)
    score += weights.tail * prec_top
    score += weights.spread * (prec_top - prec_bot)

    # Stability (20%)
    cv_coef = auc_std / (auc_mean + 1e-8)
    stability = np.clip(1 - cv_coef * 5, 0, 1)
    score += weights.stability * stability

    return score


def compute_auc_score(fold_metrics: list[dict], variance_penalty: float) -> float:
    """Compute AUC-only objective with variance penalty."""
    auc_mean = np.mean([m['auc'] for m in fold_metrics])
    auc_std = np.std([m['auc'] for m in fold_metrics])
    return auc_mean - variance_penalty * auc_std


# =============================================================================
# Optuna Objective
# =============================================================================

def create_objective(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
    cv_splits: list,
    objective_mode: str,
    weights: ObjectiveWeights,
    pruning: PruningConfig,
    variance_penalty: float,
    num_threads: int,
):
    """Create Optuna objective function."""

    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'seed': 42,
            'num_threads': num_threads,

            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'max_bin': trial.suggest_categorical('max_bin', [63, 127, 255]),
        }

        # Constraint: num_leaves <= 2^max_depth
        max_leaves = 2 ** params['max_depth']
        if params['num_leaves'] > max_leaves:
            params['num_leaves'] = max_leaves

        # Optional class balancing
        use_balanced = trial.suggest_categorical('use_balanced', [True, False])
        if use_balanced:
            params['scale_pos_weight'] = (y == 0).sum() / (y == 1).sum()

        # Cross-validation
        all_fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            X_train = np.nan_to_num(X[train_idx], nan=0.0, posinf=0.0, neginf=0.0)
            X_test = np.nan_to_num(X[test_idx], nan=0.0, posinf=0.0, neginf=0.0)
            y_train, y_test = y[train_idx], y[test_idx]
            w_train = sample_weight[train_idx] if sample_weight is not None else None

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                sample_weight=w_train,
                eval_set=[(X_test, y_test)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )

            y_pred = model.predict_proba(X_test)[:, 1]
            fold_metrics = compute_fold_metrics(y_test, y_pred)
            all_fold_metrics.append(fold_metrics)

            # Custom pruning (composite mode only)
            if objective_mode == 'composite':
                if fold_metrics['auc'] < pruning.min_auc:
                    trial.set_user_attr('prune_reason', f'low_auc_fold_{fold_idx}')
                    raise optuna.TrialPruned()

                if fold_metrics['brier'] > pruning.max_brier:
                    trial.set_user_attr('prune_reason', f'poor_brier_fold_{fold_idx}')
                    raise optuna.TrialPruned()

                if len(all_fold_metrics) >= 3:
                    aucs = [m['auc'] for m in all_fold_metrics]
                    cv = np.std(aucs) / np.mean(aucs)
                    if cv > pruning.max_cv_coef:
                        trial.set_user_attr('prune_reason', 'high_variance')
                        raise optuna.TrialPruned()

            # Optuna intermediate reporting
            trial.report(fold_metrics['auc'], fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Store metrics
        for metric in ['auc', 'aupr', 'brier', 'precision_top_10', 'precision_bottom_10']:
            values = [m[metric] for m in all_fold_metrics]
            trial.set_user_attr(f'{metric}_mean', np.mean(values))
            trial.set_user_attr(f'{metric}_std', np.std(values))

        # Compute objective
        if objective_mode == 'composite':
            return compute_composite_score(all_fold_metrics, weights)
        else:
            return compute_auc_score(all_fold_metrics, variance_penalty)

    return objective


# =============================================================================
# Trial Callback
# =============================================================================

def trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Print trial results."""
    if trial.state == optuna.trial.TrialState.COMPLETE:
        auc_mean = trial.user_attrs.get('auc_mean', 0)
        auc_std = trial.user_attrs.get('auc_std', 0)
        brier_mean = trial.user_attrs.get('brier_mean', 0)
        prec_top = trial.user_attrs.get('precision_top_10_mean', 0)

        is_best = study.best_trial.number == trial.number
        marker = " ** BEST **" if is_best else ""

        print(f"Trial {trial.number:3d}: Score={trial.value:.4f}  "
              f"AUC={auc_mean:.4f}±{auc_std:.4f}  "
              f"Brier={brier_mean:.4f}  "
              f"Prec@10={prec_top:.4f}{marker}")

    elif trial.state == optuna.trial.TrialState.PRUNED:
        reason = trial.user_attrs.get('prune_reason', 'optuna_pruner')
        print(f"Trial {trial.number:3d}: PRUNED ({reason})")


# =============================================================================
# Main
# =============================================================================

def run_hyperopt(
    n_trials: int = 200,
    timeout: Optional[int] = None,
    n_jobs: int = 8,
    n_folds: int = 5,
    gap: int = 20,
    objective_mode: str = 'composite',
    variance_penalty: float = 1.0,
    resume: bool = False,
) -> dict:
    """
    Run hyperparameter optimization.

    Args:
        n_trials: Number of Optuna trials
        timeout: Timeout in seconds
        n_jobs: LightGBM threads per model
        n_folds: Number of CV folds
        gap: Embargo gap in days
        objective_mode: 'composite' or 'auc'
        variance_penalty: Penalty for AUC std (auc mode only)
        resume: Resume from saved study

    Returns:
        Dict with best params and results
    """
    print("=" * 70)
    print("LightGBM Hyperparameter Optimization")
    print(f"  Objective: {objective_mode}")
    print("=" * 70)
    print()

    # Load data
    selected_features = load_selected_features()
    print(f"  {len(selected_features)} features from feature selection\n")

    X, y, sample_weight, dates, feature_names = load_and_prepare_data(selected_features)

    # CV splits
    print(f"\nGenerating {n_folds}-fold expanding CV with {gap}-day embargo...")
    cv_splits = get_expanding_cv_splits(dates, n_splits=n_folds, gap=gap)
    print(f"  Generated {len(cv_splits)} folds")
    for i, (train_idx, test_idx) in enumerate(cv_splits):
        print(f"    Fold {i+1}: {len(train_idx):,} train, {len(test_idx):,} test")

    # Create study
    output_dir = Path('artifacts/hyperopt')
    output_dir.mkdir(parents=True, exist_ok=True)

    storage = f'sqlite:///{output_dir}/study.db'

    sampler = TPESampler(seed=42, n_startup_trials=15, multivariate=True)
    pruner = HyperbandPruner(min_resource=2, max_resource=n_folds, reduction_factor=2)

    if resume:
        print("\nResuming from saved study...")
        study = optuna.load_study(
            study_name='lgbm_hyperopt',
            storage=storage,
            sampler=sampler,
            pruner=pruner,
        )
        print(f"  Found {len(study.trials)} existing trials")
    else:
        # Delete existing study if present (fresh start)
        try:
            optuna.delete_study(study_name='lgbm_hyperopt', storage=storage)
            print("\nDeleted existing study for fresh start...")
        except KeyError:
            pass  # No existing study

        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name='lgbm_hyperopt',
            storage=storage,
        )

    # Run optimization
    print(f"\nStarting optimization...")
    print(f"  Trials: {n_trials}")
    print(f"  Timeout: {timeout}s" if timeout else "  Timeout: None")
    print(f"  LightGBM threads: {n_jobs}")
    print(f"  Sample weights: {'enabled' if sample_weight is not None else 'disabled'}")
    print()

    objective = create_objective(
        X=X,
        y=y,
        sample_weight=sample_weight,
        cv_splits=cv_splits,
        objective_mode=objective_mode,
        weights=DEFAULT_WEIGHTS,
        pruning=DEFAULT_PRUNING,
        variance_penalty=variance_penalty,
        num_threads=n_jobs,
    )

    start_time = time.time()

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=1,  # Sequential for TPE effectiveness
        show_progress_bar=False,
        gc_after_trial=True,
        callbacks=[trial_callback],
    )

    elapsed = time.time() - start_time

    # Results
    print()
    print("=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)

    n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

    print(f"\nTrials: {n_completed} completed, {n_pruned} pruned")
    print(f"Time: {elapsed/60:.1f} minutes")

    best = study.best_trial
    print(f"\nBest Score: {study.best_value:.4f}")
    print(f"\nBest Metrics:")
    print(f"  AUC:           {best.user_attrs.get('auc_mean', 0):.4f} ± {best.user_attrs.get('auc_std', 0):.4f}")
    print(f"  AUPR:          {best.user_attrs.get('aupr_mean', 0):.4f}")
    print(f"  Brier:         {best.user_attrs.get('brier_mean', 0):.4f}")
    print(f"  Precision@10%: {best.user_attrs.get('precision_top_10_mean', 0):.4f}")
    print(f"  Spread@10%:    {best.user_attrs.get('precision_top_10_mean', 0) - best.user_attrs.get('precision_bottom_10_mean', 0):.4f}")

    print(f"\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    # Save results
    best_params = study.best_params.copy()
    best_params['_metrics'] = {
        'composite_score': study.best_value,
        'auc_mean': best.user_attrs.get('auc_mean'),
        'auc_std': best.user_attrs.get('auc_std'),
        'aupr_mean': best.user_attrs.get('aupr_mean'),
        'brier_mean': best.user_attrs.get('brier_mean'),
        'precision_top_10_mean': best.user_attrs.get('precision_top_10_mean'),
        'precision_bottom_10_mean': best.user_attrs.get('precision_bottom_10_mean'),
    }
    best_params['_metadata'] = {
        'n_trials': len(study.trials),
        'n_completed': n_completed,
        'n_pruned': n_pruned,
        'elapsed_seconds': elapsed,
        'timestamp': datetime.now().isoformat(),
        'objective_mode': objective_mode,
        'sample_weights_used': sample_weight is not None,
        'features': feature_names,
    }

    with open(output_dir / 'best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest params saved to: {output_dir / 'best_params.json'}")

    # Save trial history
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / 'trials_history.csv', index=False)

    # Parameter importance
    try:
        importance = optuna.importance.get_param_importances(study)
        print("\nParameter Importance:")
        for param, imp in sorted(importance.items(), key=lambda x: -x[1])[:8]:
            print(f"  {param}: {imp:.3f}")
        with open(output_dir / 'param_importance.json', 'w') as f:
            json.dump(importance, f, indent=2)
    except Exception as e:
        print(f"  Could not compute importance: {e}")

    print()
    print("=" * 70)
    print("Next step: python run_training.py")
    print("=" * 70)

    return best_params


def main():
    parser = argparse.ArgumentParser(description='LightGBM Hyperparameter Optimization')
    parser.add_argument('--n-trials', type=int, default=200, help='Number of trials')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds')
    parser.add_argument('--n-jobs', type=int, default=8, help='LightGBM threads')
    parser.add_argument('--n-folds', type=int, default=5, help='CV folds')
    parser.add_argument('--gap', type=int, default=20, help='Embargo gap days')
    parser.add_argument('--objective', type=str, default='composite',
                        choices=['composite', 'auc'], help='Objective mode')
    parser.add_argument('--variance-penalty', type=float, default=1.0,
                        help='Variance penalty (auc mode only)')
    parser.add_argument('--resume', action='store_true', help='Resume from saved study')

    args = parser.parse_args()

    run_hyperopt(
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        n_folds=args.n_folds,
        gap=args.gap,
        objective_mode=args.objective,
        variance_penalty=args.variance_penalty,
        resume=args.resume,
    )


if __name__ == '__main__':
    main()
