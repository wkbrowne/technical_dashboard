#!/usr/bin/env python
"""
Production Model Training Script.

Supports the 4-model system with per-model features and configurations:
- LONG_NORMAL: Standard long momentum
- LONG_PARABOLIC: Extended momentum / trend persistence
- SHORT_NORMAL: Breakdown / fragility setups
- SHORT_PARABOLIC: Panic / regime shift scenarios

Trains LightGBM models using:
- Model-specific features from base_features.py (CORE + HEAD)
- Per-model hyperparameters from hyperopt (model_configs.json)
- Model-specific target labels from triple barrier targets

Outputs:
- artifacts/models/{model_key}/production_model.pkl
- artifacts/models/{model_key}/feature_importance.csv
- artifacts/models/{model_key}/model_metadata.json
- artifacts/models/training_registry.json (combined registry)

Usage:
    # Train all 4 models
    python run_training.py --all-models

    # Train specific model
    python run_training.py --model long_normal

    # Train default model (LONG_NORMAL)
    python run_training.py
"""

import gc
import os
import sys
import json
import pickle
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

# Suppress warnings
warnings.filterwarnings('ignore', message='.*feature_name.*')
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
warnings.filterwarnings('ignore', message='.*does not have valid feature names.*')

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.model_keys import ModelKey, TARGET_CONFIGS
from src.feature_selection.base_features import get_featureset, CORE_FEATURES, HEAD_FEATURES
from src.features.registry import load_registry, registry_exists, get_registry_path


def load_model_features(
    model_key: ModelKey,
    use_registry: bool = False
) -> Tuple[List[str], Optional[str]]:
    """
    Load features for a specific model.

    Args:
        model_key: The model to load features for
        use_registry: If True, load from feature registry (artifacts/<model>/features.json).
                      If False (default), use CORE + HEAD from base_features.py.

    Returns:
        Tuple of (feature_list, feature_signature)
        - feature_signature is None if not using registry
    """
    if use_registry:
        if registry_exists(model_key.value):
            registry_path = get_registry_path(model_key.value)
            registry = load_registry(registry_path)
            features = registry["resolved_features"]
            signature = registry.get("feature_signature")
            print(f"  Loaded {len(features)} features from registry: {registry_path}")
            print(f"  Feature signature: {signature[:30]}..." if signature else "  No signature")
            return features, signature
        else:
            print(f"  WARNING: --use-registry specified but no registry found for {model_key.value}")
            print(f"  Falling back to base_features.py (CORE + HEAD)")

    # Default: use base_features.py
    features = get_featureset(model_key, include_expansion=False, flat=True)
    print(f"  Using {len(features)} features from base_features.py (CORE + HEAD)")
    return features, None


def load_selected_features_legacy() -> List[str]:
    """
    Load selected features from legacy feature selection output.

    DEPRECATED: Use load_model_features(model_key) for 4-model system.
    """
    features_file = Path('artifacts/feature_selection/selected_features.txt')
    if not features_file.exists():
        raise FileNotFoundError(
            f"Selected features file not found: {features_file}\n"
            "Run feature selection first or use --model flag for 4-model system."
        )

    features = []
    with open(features_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                features.append(line)

    return features


def load_model_configs() -> Dict[str, Any]:
    """Load the combined model configs from hyperopt."""
    config_file = Path('artifacts/hyperopt/model_configs.json')
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    return {'models': {}}


def load_best_params(model_key: Optional[ModelKey] = None) -> Tuple[dict, bool, float]:
    """
    Load the best hyperparameters from hyperopt output.

    Args:
        model_key: If provided, load params for this specific model.
                   Otherwise, try legacy path.

    Returns:
        Tuple of (params, balanced, scale_pos_weight)
    """
    param_keys = [
        'max_depth', 'num_leaves', 'min_child_samples', 'learning_rate',
        'n_estimators', 'reg_alpha', 'reg_lambda', 'min_split_gain',
        'subsample', 'subsample_freq', 'colsample_bytree', 'max_bin'
    ]

    # Try model-specific path first
    if model_key is not None:
        # First try the combined config registry
        model_configs = load_model_configs()
        if model_key.value in model_configs.get('models', {}):
            model_config = model_configs['models'][model_key.value]
            params = model_config.get('hyperparameters', {})
            params = {k: v for k, v in params.items() if k in param_keys}
            balanced = params.get('use_balanced', False)
            scale_pos_weight = params.get('scale_pos_weight', None)
            print(f"  Loaded hyperparameters from model_configs.json [{model_key.value}]")
            return params, balanced, scale_pos_weight

        # Fall back to per-model best_params.json
        params_file = Path(f'artifacts/hyperopt/{model_key.value}/best_params.json')
        if params_file.exists():
            with open(params_file) as f:
                data = json.load(f)
            params = {k: v for k, v in data.items() if k in param_keys}
            balanced = data.get('use_balanced', False)
            scale_pos_weight = data.get('scale_pos_weight', None)
            print(f"  Loaded hyperparameters from {params_file}")
            return params, balanced, scale_pos_weight

    # Legacy path
    params_file = Path('artifacts/hyperopt/best_params.json')
    if params_file.exists():
        with open(params_file) as f:
            data = json.load(f)
        params = {k: v for k, v in data.items() if k in param_keys}
        balanced = data.get('use_balanced', False)
        scale_pos_weight = data.get('scale_pos_weight', None)
        print(f"  Loaded hyperparameters from {params_file} (legacy)")
        return params, balanced, scale_pos_weight

    # Default conservative parameters
    print("  Warning: No hyperopt results found, using default parameters")
    return {
        'max_depth': 6,
        'num_leaves': 31,
        'min_child_samples': 100,
        'learning_rate': 0.05,
        'n_estimators': 300,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'min_split_gain': 0.01,
        'subsample': 0.8,
        'subsample_freq': 5,
        'colsample_bytree': 0.8,
        'max_bin': 255,
    }, False, None


def get_model_target_column(model_key: ModelKey) -> str:
    """Get the target column name for a specific model."""
    target_col_map = {
        ModelKey.LONG_NORMAL: 'hit_long_normal',
        ModelKey.LONG_PARABOLIC: 'hit_long_parabolic',
        ModelKey.SHORT_NORMAL: 'hit_short_normal',
        ModelKey.SHORT_PARABOLIC: 'hit_short_parabolic',
    }
    return target_col_map.get(model_key, 'hit')


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


def load_training_data(
    selected_features: List[str],
    model_key: Optional[ModelKey] = None,
    min_samples_per_symbol: int = 5,
    use_sample_weights: bool = True
) -> Tuple[pd.DataFrame, pd.Series, Optional[np.ndarray], pd.DataFrame]:
    """Load and prepare training data.

    Args:
        selected_features: List of feature column names to use
        model_key: Optional ModelKey for model-specific target labels
        min_samples_per_symbol: Minimum samples required per symbol
        use_sample_weights: Whether to load and return sample weights

    Returns:
        Tuple of (X, y, sample_weight, metadata)
        - X: Feature DataFrame
        - y: Binary target Series
        - sample_weight: Sample weights array (or None if disabled)
        - metadata: DataFrame with symbol, date, entry_px, etc.
    """
    # Determine target column first
    print("Loading targets...")
    targets = pd.read_parquet('artifacts/targets_triple_barrier.parquet')
    print(f"  Targets shape: {targets.shape}")

    # Rename columns to standard names
    targets = targets.rename(columns={
        't0': 'date',
        'top': 'target_price',
        'bot': 'stop_price',
        'entry_px': 'entry_price'
    })

    # Get the appropriate hit column for this model
    if model_key is not None:
        hit_col = get_model_target_column(model_key)
        if hit_col not in targets.columns:
            hit_col = 'hit'
            print(f"  Warning: {get_model_target_column(model_key)} not found, using 'hit'")
        else:
            print(f"  Using model-specific target: {hit_col}")
    else:
        hit_col = 'hit'

    # Filter symbols with enough samples (on targets - smaller than features)
    symbol_counts = targets.groupby('symbol').size()
    valid_symbols = set(symbol_counts[symbol_counts >= min_samples_per_symbol].index)
    targets = targets[targets['symbol'].isin(valid_symbols)]
    print(f"  Valid symbols: {len(valid_symbols)}")

    # Determine which columns to load from features
    # Only load what we need: symbol, date, and selected features
    feature_cols_needed = ['symbol', 'date'] + list(selected_features)

    print("Loading features (selected columns only)...")
    # Check which columns actually exist in the parquet
    import pyarrow.parquet as pq
    pq_file = pq.ParquetFile('artifacts/features_complete.parquet')
    available_cols = set(pq_file.schema.names)
    cols_to_load = [c for c in feature_cols_needed if c in available_cols]
    missing_features = set(selected_features) - available_cols
    if missing_features:
        print(f"  Warning: {len(missing_features)} features not in parquet: {list(missing_features)[:5]}...")

    features = pd.read_parquet('artifacts/features_complete.parquet', columns=cols_to_load)
    print(f"  Features shape: {features.shape}")

    # Filter features to valid symbols (no .copy() needed - boolean indexing returns new df)
    features = features[features['symbol'].isin(valid_symbols)]

    # Determine which columns to merge from targets
    base_cols = ['symbol', 'date', 'entry_price', 'target_price', 'stop_price']
    target_cols = base_cols + [hit_col]
    if use_sample_weights and 'weight_final' in targets.columns:
        target_cols.append('weight_final')

    # Merge features with targets (inner join filters to matching rows)
    merged = features.merge(
        targets[target_cols],
        on=['symbol', 'date'],
        how='inner'
    )
    del features, targets  # Free memory

    # Binary target (exclude neutral hit=0)
    merged = merged[merged[hit_col] != 0]

    # For long models: upper barrier hit (1) = success
    # For short models: lower barrier hit (-1) = success
    if model_key is not None and model_key.is_short():
        merged['target'] = (merged[hit_col] == -1).astype(int)
    else:
        merged['target'] = (merged[hit_col] == 1).astype(int)

    # Sort by date
    merged = merged.sort_values(['date', 'symbol']).reset_index(drop=True)

    # Check which selected features are available
    available_features = [f for f in selected_features if f in merged.columns]
    print(f"  Using {len(available_features)} features")

    # Prepare output (use .values for numpy arrays to avoid DataFrame overhead)
    X = merged[available_features]
    y = merged['target']

    # Sample weights
    sample_weight = None
    if use_sample_weights and 'weight_final' in merged.columns:
        sample_weight = merged['weight_final'].values
        print(f"  Sample weights: min={sample_weight.min():.3f}, max={sample_weight.max():.3f}, mean={sample_weight.mean():.3f}")
    elif use_sample_weights:
        print("  Warning: Sample weights requested but 'weight_final' not in targets")

    # Metadata for later use
    metadata_cols = ['symbol', 'date', 'entry_price', 'target_price', 'stop_price', hit_col]
    metadata = merged[metadata_cols]
    if hit_col != 'hit':
        metadata = metadata.rename(columns={hit_col: 'hit'})

    print(f"\nTraining data:")
    print(f"  Samples: {len(X):,}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Positive class: {y.sum():,} ({y.mean()*100:.1f}%)")
    print(f"  Date range: {merged['date'].min()} to {merged['date'].max()}")

    del features, targets, merged
    gc.collect()

    return X, y, sample_weight, metadata


def train_production_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    n_jobs: int = 8,
    sample_weight: np.ndarray | None = None,
    scale_pos_weight: float | None = None
) -> lgb.LGBMClassifier:
    """Train the production model on all data.

    Args:
        X: Feature matrix
        y: Target vector
        params: LightGBM hyperparameters
        n_jobs: Number of threads
        sample_weight: Per-sample weights from triple barrier overlap inverse
        scale_pos_weight: Class weight for balancing (n_neg/n_pos). None for no weighting.

    Returns:
        Trained LGBMClassifier
    """
    print("\nTraining production model...")

    # Build full params
    full_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 42,
        'num_threads': n_jobs,
        **params
    }

    # Add class weighting if specified
    if scale_pos_weight is not None:
        full_params['scale_pos_weight'] = scale_pos_weight
        print(f"  Using class balancing: scale_pos_weight={scale_pos_weight:.3f}")

    # Ensure num_leaves constraint
    max_leaves = 2 ** full_params.get('max_depth', 6)
    if full_params.get('num_leaves', 31) > max_leaves:
        full_params['num_leaves'] = max_leaves

    # Handle NaN
    X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)

    # Train with sample weights
    model = lgb.LGBMClassifier(**full_params)
    if sample_weight is not None:
        print(f"  Using sample weights (overlap inverse)")
        model.fit(X_clean, y, sample_weight=sample_weight)
    else:
        model.fit(X_clean, y)

    # Evaluate on training data (sanity check)
    y_pred = model.predict_proba(X_clean)[:, 1]
    train_auc = roc_auc_score(y, y_pred)
    train_aupr = average_precision_score(y, y_pred)

    print(f"  Training complete: {model.n_estimators_} trees")
    print(f"  Train AUC: {train_auc:.4f}")
    print(f"  Train AUPR: {train_aupr:.4f}")

    return model


def save_model_artifacts(
    model: lgb.LGBMClassifier,
    feature_names: List[str],
    params: dict,
    output_dir: Path,
    train_metrics: dict,
    model_key: Optional[ModelKey] = None,
    feature_signature: Optional[str] = None
):
    """Save model and associated artifacts.

    Args:
        model: Trained model
        feature_names: List of feature names
        params: Hyperparameters used
        output_dir: Output directory
        train_metrics: Training metrics
        model_key: Optional ModelKey for additional metadata
        feature_signature: Optional signature from feature registry for traceability
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_file = output_dir / 'production_model.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {model_file}")

    # Save feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_,
        'importance_pct': model.feature_importances_ / model.feature_importances_.sum() * 100
    }).sort_values('importance', ascending=False)

    importance_file = output_dir / 'feature_importance.csv'
    importance_df.to_csv(importance_file, index=False)
    print(f"Feature importance saved to: {importance_file}")

    # Print top features
    print("\nTop 10 features by importance:")
    for _, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance_pct']:.1f}%")

    # Save metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'model_key': model_key.value if model_key else None,
        'n_features': len(feature_names),
        'features': feature_names,
        'feature_signature': feature_signature,  # Registry signature for reproducibility
        'params': params,
        'n_estimators': model.n_estimators_,
        'train_auc': train_metrics['train_auc'],
        'train_aupr': train_metrics['train_aupr'],
        'n_samples': train_metrics['n_samples'],
        'positive_rate': train_metrics['positive_rate'],
        'date_range': train_metrics['date_range'],
        'balanced': train_metrics.get('balanced', False),
        'scale_pos_weight': train_metrics.get('scale_pos_weight'),
        'sample_weights_used': train_metrics.get('sample_weights_used', False),
    }

    # Add model-specific metadata
    if model_key is not None:
        target_config = TARGET_CONFIGS.get(model_key, {})
        metadata['target_config'] = {
            'up_mult': target_config.get('up_mult'),
            'dn_mult': target_config.get('dn_mult'),
            'max_horizon': target_config.get('max_horizon'),
        }
        metadata['feature_breakdown'] = {
            'core_count': len(CORE_FEATURES),
            'head_count': len(HEAD_FEATURES.get(model_key, [])),
        }

    metadata_file = output_dir / 'model_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Metadata saved to: {metadata_file}")


def update_training_registry(
    model_key: ModelKey,
    output_dir: Path,
    train_metrics: dict
) -> None:
    """
    Update the combined training registry with results for a model.

    Args:
        model_key: The model key
        output_dir: Directory where model was saved
        train_metrics: Training metrics
    """
    registry_file = Path('artifacts/models/training_registry.json')
    registry_file.parent.mkdir(parents=True, exist_ok=True)

    if registry_file.exists():
        with open(registry_file) as f:
            registry = json.load(f)
    else:
        registry = {'models': {}, 'updated': None}

    registry['models'][model_key.value] = {
        'model_path': str(output_dir / 'production_model.pkl'),
        'metadata_path': str(output_dir / 'model_metadata.json'),
        'train_auc': train_metrics['train_auc'],
        'train_aupr': train_metrics['train_aupr'],
        'n_samples': train_metrics['n_samples'],
        'positive_rate': train_metrics['positive_rate'],
        'training_date': datetime.now().isoformat(),
    }
    registry['updated'] = datetime.now().isoformat()

    with open(registry_file, 'w') as f:
        json.dump(registry, f, indent=2)
    print(f"Training registry updated: {registry_file}")


def train_single_model(
    model_key: ModelKey,
    base_output_dir: Path,
    n_jobs: int = 8,
    use_sample_weights: bool = True,
    force_balanced: Optional[bool] = None,
    use_registry: bool = False,
) -> dict:
    """
    Train a single model.

    Args:
        model_key: The model to train
        base_output_dir: Base output directory (model-specific subdir created)
        n_jobs: Number of threads
        use_sample_weights: Whether to use sample weights
        force_balanced: True/False to force, None to use hyperopt setting
        use_registry: If True, load features from registry instead of base_features.py

    Returns:
        Dict with training metrics
    """
    print(f"\nTraining {model_key.value.upper()}")
    print("-" * 50)

    # Load features for this model
    selected_features, feature_signature = load_model_features(model_key, use_registry=use_registry)

    # Load hyperparameters for this model
    params, hyperopt_balanced, hyperopt_scale_pos_weight = load_best_params(model_key)

    # Load data with model-specific target
    X, y, sample_weight, metadata = load_training_data(
        selected_features,
        model_key=model_key,
        use_sample_weights=use_sample_weights
    )

    # Determine balanced training
    if force_balanced is False:
        scale_pos_weight = None
    elif force_balanced is True:
        scale_pos_weight = compute_scale_pos_weight(y)
    elif hyperopt_balanced:
        scale_pos_weight = compute_scale_pos_weight(y)
    else:
        scale_pos_weight = None

    # Train
    model = train_production_model(
        X, y, params,
        n_jobs=n_jobs,
        sample_weight=sample_weight,
        scale_pos_weight=scale_pos_weight
    )

    # Calculate metrics
    X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
    y_pred = model.predict_proba(X_clean)[:, 1]

    train_metrics = {
        'train_auc': roc_auc_score(y, y_pred),
        'train_aupr': average_precision_score(y, y_pred),
        'n_samples': len(y),
        'positive_rate': float(y.mean()),
        'date_range': [str(metadata['date'].min()), str(metadata['date'].max())],
        'balanced': scale_pos_weight is not None,
        'scale_pos_weight': scale_pos_weight,
        'sample_weights_used': sample_weight is not None,
        'feature_signature': feature_signature,  # From registry if available
    }

    # Save to model-specific directory
    output_dir = base_output_dir / model_key.value
    save_model_artifacts(
        model=model,
        feature_names=X.columns.tolist(),
        params=params,
        output_dir=output_dir,
        train_metrics=train_metrics,
        model_key=model_key,
        feature_signature=feature_signature
    )

    # Update training registry
    update_training_registry(model_key, output_dir, train_metrics)

    return train_metrics


def train_all_models(
    base_output_dir: Path,
    n_jobs: int = 8,
    use_sample_weights: bool = True,
    force_balanced: Optional[bool] = None,
    use_registry: bool = False,
) -> Dict[str, dict]:
    """
    Train all 4 models.

    Args:
        base_output_dir: Base output directory
        n_jobs: Number of threads
        use_sample_weights: Whether to use sample weights
        force_balanced: True/False to force, None to use hyperopt setting
        use_registry: If True, load features from registry instead of base_features.py

    Returns:
        Dict mapping model_key to training metrics
    """
    print("=" * 70)
    print("MULTI-MODEL PRODUCTION TRAINING")
    print(f"  Models: {', '.join(mk.value for mk in ModelKey.all_keys())}")
    print(f"  Feature source: {'registry' if use_registry else 'base_features.py'}")
    print("=" * 70)

    results = {}

    for i, model_key in enumerate(ModelKey.all_keys()):
        print(f"\n[{i+1}/4] Training {model_key.value.upper()}")
        print("=" * 70)

        metrics = train_single_model(
            model_key=model_key,
            base_output_dir=base_output_dir,
            n_jobs=n_jobs,
            use_sample_weights=use_sample_weights,
            force_balanced=force_balanced,
            use_registry=use_registry,
        )
        results[model_key.value] = metrics

        gc.collect()

    # Print summary
    print("\n" + "=" * 70)
    print("MULTI-MODEL TRAINING COMPLETE")
    print("=" * 70)
    print("\nResults per model:")
    print("-" * 60)
    print(f"{'Model':<20} {'Train AUC':>12} {'Train AUPR':>12} {'Samples':>12}")
    print("-" * 60)

    for model_key in ModelKey.all_keys():
        metrics = results.get(model_key.value, {})
        auc = metrics.get('train_auc', 0)
        aupr = metrics.get('train_aupr', 0)
        n = metrics.get('n_samples', 0)
        print(f"{model_key.value:<20} {auc:>12.4f} {aupr:>12.4f} {n:>12,}")

    print(f"\nTraining registry: {base_output_dir}/training_registry.json")
    print("Next step: python run_predict.py --all-models")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train Production Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train all 4 models
    python run_training.py --all-models

    # Train specific model
    python run_training.py --model long_normal

    # Train default model (LONG_NORMAL)
    python run_training.py
        """
    )
    parser.add_argument('--model', type=str, default=None,
                        choices=['long_normal', 'long_parabolic', 'short_normal', 'short_parabolic'],
                        help='Model key to train (default: long_normal)')
    parser.add_argument('--all-models', action='store_true',
                        help='Train all 4 models')
    parser.add_argument('--output-dir', type=str, default='artifacts/models',
                        help='Output directory for model artifacts')
    parser.add_argument('--n-jobs', type=int, default=8,
                        help='Number of threads for training')
    parser.add_argument('--balanced', action='store_true',
                        help='Use class weights for balanced training')
    parser.add_argument('--no-balanced', action='store_true',
                        help='Force disable balanced training')
    parser.add_argument('--no-sample-weights', action='store_true',
                        help='Disable sample weights from triple barrier overlap inverse')
    parser.add_argument('--use-registry', action='store_true',
                        help='Load features from registry (artifacts/<model>/features.json) instead of base_features.py')

    args = parser.parse_args()
    base_output_dir = Path(args.output_dir)
    use_sample_weights = not args.no_sample_weights
    use_registry = args.use_registry

    # Determine balanced setting
    force_balanced = None
    if args.no_balanced:
        force_balanced = False
    elif args.balanced:
        force_balanced = True

    if args.all_models:
        train_all_models(
            base_output_dir=base_output_dir,
            n_jobs=args.n_jobs,
            use_sample_weights=use_sample_weights,
            force_balanced=force_balanced,
            use_registry=use_registry,
        )
    else:
        # Single model training
        model_key = ModelKey(args.model) if args.model else ModelKey.LONG_NORMAL

        print("=" * 60)
        print("Production Model Training")
        print(f"  Model: {model_key.value.upper()}")
        print("=" * 60)

        train_single_model(
            model_key=model_key,
            base_output_dir=base_output_dir,
            n_jobs=args.n_jobs,
            use_sample_weights=use_sample_weights,
            force_balanced=force_balanced,
            use_registry=use_registry,
        )

        print()
        print("=" * 60)
        print("Training complete!")
        print("=" * 60)


if __name__ == '__main__':
    main()
