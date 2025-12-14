#!/usr/bin/env python
"""
Production Model Training Script.

Trains a LightGBM model using:
- Fixed features from feature selection (selected_features.txt)
- Fixed hyperparameters from hyperopt (best_params.json)
- All available historical data

Outputs:
- Trained model (production_model.pkl)
- Feature importance (feature_importance.csv)
- Model metadata (model_metadata.json)

Usage:
    python run_training.py [--output-dir artifacts/models]
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


def load_best_params() -> dict:
    """Load the best hyperparameters from hyperopt output."""
    params_file = Path('artifacts/hyperopt/best_params.json')

    if params_file.exists():
        with open(params_file) as f:
            data = json.load(f)
        # Extract just the hyperparameters (exclude metadata like 'best_auc', 'features', etc.)
        param_keys = [
            'max_depth', 'num_leaves', 'min_child_samples', 'learning_rate',
            'n_estimators', 'reg_alpha', 'reg_lambda', 'min_split_gain',
            'subsample', 'subsample_freq', 'colsample_bytree', 'max_bin'
        ]
        params = {k: v for k, v in data.items() if k in param_keys}
        print(f"  Loaded hyperparameters from {params_file}")
        return params
    else:
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
        }


def load_training_data(
    selected_features: list[str],
    min_samples_per_symbol: int = 5
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load and prepare training data.

    Returns:
        Tuple of (X, y, metadata) where metadata contains symbol, date for predictions.
    """
    print("Loading features...")
    features = pd.read_parquet('artifacts/features_daily.parquet')
    print(f"  Features shape: {features.shape}")

    print("Loading targets...")
    targets = pd.read_parquet('artifacts/targets_triple_barrier.parquet')
    print(f"  Targets shape: {targets.shape}")

    # Merge
    targets = targets.rename(columns={'t0': 'date'})

    # Filter symbols with enough samples
    symbol_counts = targets.groupby('symbol').size()
    valid_symbols = symbol_counts[symbol_counts >= min_samples_per_symbol].index.tolist()

    features = features[features['symbol'].isin(valid_symbols)].copy()
    targets = targets[targets['symbol'].isin(valid_symbols)].copy()

    # Merge features with targets
    merged = features.merge(
        targets[['symbol', 'date', 'hit', 'target_price', 'stop_price']],
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

    # Prepare output
    X = merged[available_features].copy()
    y = merged['target'].copy()

    # Metadata for later use
    metadata = merged[['symbol', 'date', 'close', 'target_price', 'stop_price', 'hit']].copy()

    print(f"\nTraining data:")
    print(f"  Samples: {len(X):,}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Positive class: {y.sum():,} ({y.mean()*100:.1f}%)")
    print(f"  Date range: {merged['date'].min()} to {merged['date'].max()}")

    del features, targets, merged
    gc.collect()

    return X, y, metadata


def train_production_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    n_jobs: int = 8
) -> lgb.LGBMClassifier:
    """Train the production model on all data.

    Args:
        X: Feature matrix
        y: Target vector
        params: LightGBM hyperparameters
        n_jobs: Number of threads

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

    # Ensure num_leaves constraint
    max_leaves = 2 ** full_params.get('max_depth', 6)
    if full_params.get('num_leaves', 31) > max_leaves:
        full_params['num_leaves'] = max_leaves

    # Handle NaN
    X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)

    # Train
    model = lgb.LGBMClassifier(**full_params)
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
    feature_names: list[str],
    params: dict,
    output_dir: Path,
    train_metrics: dict
):
    """Save model and associated artifacts.

    Args:
        model: Trained model
        feature_names: List of feature names
        params: Hyperparameters used
        output_dir: Output directory
        train_metrics: Training metrics
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
        'n_features': len(feature_names),
        'features': feature_names,
        'params': params,
        'n_estimators': model.n_estimators_,
        'train_auc': train_metrics['train_auc'],
        'train_aupr': train_metrics['train_aupr'],
        'n_samples': train_metrics['n_samples'],
        'positive_rate': train_metrics['positive_rate'],
        'date_range': train_metrics['date_range'],
    }

    metadata_file = output_dir / 'model_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Metadata saved to: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description='Train Production Model')
    parser.add_argument('--output-dir', type=str, default='artifacts/models',
                        help='Output directory for model artifacts')
    parser.add_argument('--n-jobs', type=int, default=8,
                        help='Number of threads for training')

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("Production Model Training")
    print("=" * 60)
    print()

    # Load configuration
    print("Loading configuration...")
    selected_features = load_selected_features()
    print(f"  {len(selected_features)} selected features")

    params = load_best_params()

    # Load data
    X, y, metadata = load_training_data(selected_features)

    # Train model
    model = train_production_model(X, y, params, n_jobs=args.n_jobs)

    # Calculate metrics for metadata
    X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
    y_pred = model.predict_proba(X_clean)[:, 1]

    train_metrics = {
        'train_auc': roc_auc_score(y, y_pred),
        'train_aupr': average_precision_score(y, y_pred),
        'n_samples': len(y),
        'positive_rate': y.mean(),
        'date_range': [str(metadata['date'].min()), str(metadata['date'].max())],
    }

    # Save artifacts
    save_model_artifacts(
        model=model,
        feature_names=X.columns.tolist(),
        params=params,
        output_dir=output_dir,
        train_metrics=train_metrics
    )

    print()
    print("=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
