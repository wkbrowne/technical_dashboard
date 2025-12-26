"""
Hyperopt sanity checks.

Checks:
- Over-pruning rate
- Degenerate parameter combinations
- Parameter importance stability
- Learning rate extremes
"""

from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import json
import numpy as np
import pandas as pd

from .core import (
    DiagnosticFlag, Severity,
    DiagnosticThresholds, DEFAULT_THRESHOLDS,
)


def load_hyperopt_artifacts(
    model_key: str,
    hyperopt_dir: Path = Path('artifacts/hyperopt'),
) -> Tuple[Optional[Dict], Optional[pd.DataFrame], Optional[Dict]]:
    """
    Load hyperopt artifacts for a model.

    Args:
        model_key: Model key string
        hyperopt_dir: Base hyperopt directory

    Returns:
        Tuple of (best_params, trials_df, param_importance)
    """
    model_dir = hyperopt_dir / model_key

    best_params = None
    trials_df = None
    param_importance = None

    # Load best params
    best_params_file = model_dir / 'best_params.json'
    if best_params_file.exists():
        with open(best_params_file) as f:
            best_params = json.load(f)

    # Load trials history
    trials_file = model_dir / 'trials_history.csv'
    if trials_file.exists():
        trials_df = pd.read_csv(trials_file)

    # Load param importance
    importance_file = model_dir / 'param_importance.json'
    if importance_file.exists():
        with open(importance_file) as f:
            param_importance = json.load(f)

    return best_params, trials_df, param_importance


def check_pruning_rate(
    trials_df: Optional[pd.DataFrame],
    thresholds: DiagnosticThresholds = DEFAULT_THRESHOLDS,
) -> List[DiagnosticFlag]:
    """
    Check for over-pruning in hyperopt trials.

    Args:
        trials_df: Trials history DataFrame
        thresholds: Diagnostic thresholds

    Returns:
        List of diagnostic flags
    """
    flags = []

    if trials_df is None or len(trials_df) == 0:
        return flags

    # Check for state column (Optuna format)
    if 'state' in trials_df.columns:
        n_total = len(trials_df)
        n_pruned = (trials_df['state'] == 'PRUNED').sum()
        n_complete = (trials_df['state'] == 'COMPLETE').sum()
    else:
        # Fallback: check for NaN values in objective
        n_total = len(trials_df)
        value_col = 'value' if 'value' in trials_df.columns else trials_df.columns[0]
        n_complete = trials_df[value_col].notna().sum()
        n_pruned = n_total - n_complete

    if n_total > 0:
        pruning_rate = n_pruned / n_total

        if pruning_rate > thresholds.pruning_rate_warn:
            flags.append(DiagnosticFlag(
                severity=Severity.WARN,
                check_name='high_pruning_rate',
                symptom=f"{pruning_rate*100:.1f}% of trials were pruned ({n_pruned}/{n_total})",
                why_it_matters="High pruning rate suggests search space may be too aggressive or pruning thresholds too strict",
                suggested_fix="Consider relaxing pruning thresholds or adjusting search space bounds",
                evidence={
                    'n_total': n_total,
                    'n_complete': n_complete,
                    'n_pruned': n_pruned,
                    'pruning_rate': round(pruning_rate * 100, 1),
                    'threshold': thresholds.pruning_rate_warn * 100,
                },
            ))

    return flags


def check_degenerate_params(
    best_params: Optional[Dict],
    thresholds: DiagnosticThresholds = DEFAULT_THRESHOLDS,
) -> List[DiagnosticFlag]:
    """
    Check for potentially degenerate hyperparameter combinations.

    Args:
        best_params: Best parameters dictionary
        thresholds: Diagnostic thresholds

    Returns:
        List of diagnostic flags
    """
    flags = []

    if best_params is None:
        return flags

    # Extract params (handle nested structure from best_params.json)
    params = {k: v for k, v in best_params.items() if not k.startswith('_')}

    # Compute num_leaves from exponent if present
    num_leaves = params.get('num_leaves')
    if 'num_leaves_exp' in params:
        num_leaves = int(2 ** params['num_leaves_exp'])

    min_child_samples = params.get('min_child_samples')
    max_depth = params.get('max_depth')
    learning_rate = params.get('learning_rate')

    # Check huge leaves + tiny min_child
    if num_leaves and min_child_samples:
        if num_leaves > thresholds.degenerate_num_leaves_threshold and \
           min_child_samples < thresholds.degenerate_min_child_threshold:
            flags.append(DiagnosticFlag(
                severity=Severity.WARN,
                check_name='degenerate_tree_params',
                symptom=f"num_leaves={num_leaves} with min_child_samples={min_child_samples}",
                why_it_matters="Many leaves with few samples per leaf leads to overfitting",
                suggested_fix="Constrain num_leaves or increase min_child_samples in search space",
                evidence={
                    'num_leaves': num_leaves,
                    'min_child_samples': min_child_samples,
                    'num_leaves_threshold': thresholds.degenerate_num_leaves_threshold,
                    'min_child_threshold': thresholds.degenerate_min_child_threshold,
                },
            ))

    # Check learning rate extremes
    if learning_rate:
        if learning_rate < thresholds.learning_rate_low:
            flags.append(DiagnosticFlag(
                severity=Severity.INFO,
                check_name='low_learning_rate',
                symptom=f"Learning rate is {learning_rate:.4f} (below {thresholds.learning_rate_low})",
                why_it_matters="Very low learning rate may indicate need for more trees or underfitting",
                suggested_fix="Consider if more boosting rounds would help; check n_estimators",
                evidence={
                    'learning_rate': learning_rate,
                    'threshold': thresholds.learning_rate_low,
                },
            ))
        elif learning_rate > thresholds.learning_rate_high:
            flags.append(DiagnosticFlag(
                severity=Severity.INFO,
                check_name='high_learning_rate',
                symptom=f"Learning rate is {learning_rate:.4f} (above {thresholds.learning_rate_high})",
                why_it_matters="High learning rate may cause overfitting or training instability",
                suggested_fix="Monitor training/validation curves for signs of overfitting",
                evidence={
                    'learning_rate': learning_rate,
                    'threshold': thresholds.learning_rate_high,
                },
            ))

    # Check for unlimited depth with many leaves
    if max_depth == -1 and num_leaves and num_leaves > 64:
        flags.append(DiagnosticFlag(
            severity=Severity.INFO,
            check_name='unlimited_depth',
            symptom=f"max_depth=-1 (unlimited) with num_leaves={num_leaves}",
            why_it_matters="Unlimited depth can lead to very deep trees and potential overfitting",
            suggested_fix="Consider bounded max_depth if overfitting is observed",
            evidence={
                'max_depth': max_depth,
                'num_leaves': num_leaves,
            },
        ))

    return flags


def check_param_importance(
    param_importance: Optional[Dict],
    trials_df: Optional[pd.DataFrame],
) -> List[DiagnosticFlag]:
    """
    Analyze parameter importance from hyperopt.

    Args:
        param_importance: Parameter importance dictionary
        trials_df: Trials history for variance analysis

    Returns:
        List of diagnostic flags
    """
    flags = []

    if param_importance is None or len(param_importance) == 0:
        flags.append(DiagnosticFlag(
            severity=Severity.INFO,
            check_name='no_param_importance',
            symptom="Parameter importance not available",
            why_it_matters="Cannot analyze which hyperparameters most affect performance",
            suggested_fix="Ensure optuna.importance.get_param_importances is called after optimization",
            evidence={},
        ))
        return flags

    # Sort by importance
    sorted_importance = sorted(param_importance.items(), key=lambda x: -x[1])

    # Check for single dominant parameter
    if len(sorted_importance) >= 2:
        top_importance = sorted_importance[0][1]
        second_importance = sorted_importance[1][1]

        if top_importance > 0.5 and top_importance > second_importance * 3:
            flags.append(DiagnosticFlag(
                severity=Severity.INFO,
                check_name='dominant_param',
                symptom=f"'{sorted_importance[0][0]}' dominates with {top_importance*100:.1f}% importance",
                why_it_matters="Single dominant parameter suggests other params may not matter much",
                suggested_fix="Consider fixing less important params to reduce search space",
                evidence={
                    'top_param': sorted_importance[0][0],
                    'top_importance': round(top_importance, 4),
                    'param_importance': {k: round(v, 4) for k, v in sorted_importance[:5]},
                },
            ))

    return flags


def compute_hyperopt_summary(
    best_params: Optional[Dict],
    trials_df: Optional[pd.DataFrame],
    param_importance: Optional[Dict],
) -> Dict[str, Any]:
    """
    Compute summary of hyperopt results.

    Args:
        best_params: Best parameters dictionary
        trials_df: Trials history DataFrame
        param_importance: Parameter importance dictionary

    Returns:
        Summary dictionary
    """
    summary = {
        'best_params_available': best_params is not None,
        'trials_available': trials_df is not None,
        'importance_available': param_importance is not None,
    }

    if best_params:
        # Extract clean params and metrics
        params = {k: v for k, v in best_params.items() if not k.startswith('_')}
        metrics = best_params.get('_metrics', {})
        metadata = best_params.get('_metadata', {})

        summary['best_params'] = params
        summary['metrics'] = metrics
        summary['n_features'] = metadata.get('feature_count')

    if trials_df is not None and len(trials_df) > 0:
        if 'state' in trials_df.columns:
            summary['n_trials'] = len(trials_df)
            summary['n_complete'] = (trials_df['state'] == 'COMPLETE').sum()
            summary['n_pruned'] = (trials_df['state'] == 'PRUNED').sum()
        else:
            summary['n_trials'] = len(trials_df)

    if param_importance:
        sorted_importance = sorted(param_importance.items(), key=lambda x: -x[1])
        summary['param_importance'] = {k: round(v, 4) for k, v in sorted_importance}

    return summary


def run_hyperopt_checks(
    model_key: str,
    hyperopt_dir: Path = Path('artifacts/hyperopt'),
    thresholds: DiagnosticThresholds = DEFAULT_THRESHOLDS,
) -> Tuple[List[DiagnosticFlag], Dict[str, Any]]:
    """
    Run all hyperopt sanity checks.

    Args:
        model_key: Model key string
        hyperopt_dir: Base hyperopt directory
        thresholds: Diagnostic thresholds

    Returns:
        Tuple of (flags, summary_dict)
    """
    flags = []

    # Load artifacts
    best_params, trials_df, param_importance = load_hyperopt_artifacts(
        model_key, hyperopt_dir
    )

    if best_params is None:
        flags.append(DiagnosticFlag(
            severity=Severity.WARN,
            check_name='no_hyperopt_results',
            symptom=f"No hyperopt results found for {model_key}",
            why_it_matters="Cannot validate hyperparameters without hyperopt results",
            suggested_fix=f"Run: python run_model_tuning.py --model {model_key}",
            evidence={
                'model_key': model_key,
                'hyperopt_dir': str(hyperopt_dir),
            },
        ))
        return flags, {'hyperopt_available': False}

    # Run checks
    flags.extend(check_pruning_rate(trials_df, thresholds))
    flags.extend(check_degenerate_params(best_params, thresholds))
    flags.extend(check_param_importance(param_importance, trials_df))

    # Summary
    summary = compute_hyperopt_summary(best_params, trials_df, param_importance)

    return flags, summary
