#!/usr/bin/env python
"""
Prediction Script.

Loads the production model and generates predictions for the universe.
Outputs rankings with probability scores, targets, and stop-losses.

Usage:
    python run_predict.py [--date 2024-01-15] [--top-n 50]
"""

import os
import sys
import json
import pickle
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timedelta

# Suppress warnings
warnings.filterwarnings('ignore', message='.*feature_name.*')
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
warnings.filterwarnings('ignore', message='.*does not have valid feature names.*')

import numpy as np
import pandas as pd
import lightgbm as lgb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def load_model(model_dir: Path = Path('artifacts/models')) -> tuple[lgb.LGBMClassifier, dict]:
    """Load the production model and metadata.

    Returns:
        Tuple of (model, metadata)
    """
    model_file = model_dir / 'production_model.pkl'
    metadata_file = model_dir / 'model_metadata.json'

    if not model_file.exists():
        raise FileNotFoundError(
            f"Model not found: {model_file}\n"
            "Run training first: python run_training.py"
        )

    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    metadata = {}
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)

    return model, metadata


def find_latest_valid_date(
    features: pd.DataFrame,
    min_coverage: float = 0.8,
    required_col: str = 'close'
) -> pd.Timestamp:
    """Find the most recent date with sufficient valid data.

    Args:
        features: Features DataFrame with 'date' column
        min_coverage: Minimum fraction of symbols with valid required_col (default 80%)
        required_col: Column that must be non-null (default 'close')

    Returns:
        Most recent date with sufficient coverage
    """
    # Calculate coverage by date
    coverage = features.groupby('date')[required_col].apply(
        lambda x: x.notna().sum() / len(x)
    )

    # Find dates with sufficient coverage
    valid_dates = coverage[coverage >= min_coverage]

    if len(valid_dates) == 0:
        # Fall back to date with best coverage
        best_date = coverage.idxmax()
        print(f"  Warning: No date has {min_coverage:.0%} coverage. Using best: {best_date.date()} ({coverage[best_date]:.1%})")
        return best_date

    return valid_dates.index.max()


def load_latest_features(
    feature_names: list[str],
    target_date: str = None,
    min_coverage: float = 0.8
) -> pd.DataFrame:
    """Load features for the latest available date or specified date.

    Args:
        feature_names: List of required feature names
        target_date: Specific date to load (YYYY-MM-DD) or None for latest with valid data
        min_coverage: Minimum data coverage required (default 80%)

    Returns:
        DataFrame with features for the target date
    """
    print("Loading features...")
    features = pd.read_parquet('artifacts/features_daily.parquet')

    # Convert date if needed
    if 'date' in features.columns:
        features['date'] = pd.to_datetime(features['date'])

    # Get target date
    if target_date:
        target_dt = pd.to_datetime(target_date)
        print(f"  Requested date: {target_dt.date()}")
    else:
        # Find most recent date with valid data
        target_dt = find_latest_valid_date(features, min_coverage=min_coverage)
        max_date = features['date'].max()
        if target_dt < max_date:
            days_stale = (max_date - target_dt).days
            print(f"  WARNING: Data is {days_stale} days stale!")
            print(f"  Latest date in file: {max_date.date()}")
            print(f"  Latest date with valid data: {target_dt.date()}")

    print(f"  Using date: {target_dt.date()}")

    # Filter to target date
    df = features[features['date'] == target_dt].copy()

    if len(df) == 0:
        # Try finding nearest available date
        available_dates = features['date'].unique()
        nearest = min(available_dates, key=lambda x: abs(x - target_dt))
        print(f"  Warning: No data for {target_dt.date()}, using {nearest.date()}")
        df = features[features['date'] == nearest].copy()

    # Filter to symbols with valid close prices
    valid_close = df['close'].notna()
    n_valid = valid_close.sum()
    n_total = len(df)
    if n_valid < n_total:
        print(f"  Filtering: {n_valid}/{n_total} symbols have valid close price")
        df = df[valid_close].copy()

    print(f"  Symbols available: {len(df)}")

    # Check feature availability
    available = [f for f in feature_names if f in df.columns]
    missing = set(feature_names) - set(available)
    if missing:
        print(f"  Warning: {len(missing)} features missing: {list(missing)[:5]}...")

    return df


def calculate_targets_and_stops(
    df: pd.DataFrame,
    up_mult: float = 3.0,
    dn_mult: float = 1.5,
    atr_col: str = 'atr_percent'
) -> pd.DataFrame:
    """Calculate target prices and stop-losses based on ATR.

    Args:
        df: DataFrame with 'close' and ATR columns
        up_mult: ATR multiplier for upside target
        dn_mult: ATR multiplier for downside stop
        atr_col: Name of ATR column (as percentage)

    Returns:
        DataFrame with target_price, stop_price, reward_risk columns added
    """
    df = df.copy()

    # ATR as decimal (e.g., 0.02 = 2%)
    if atr_col in df.columns:
        atr_pct = df[atr_col]  # Already in decimal form
    else:
        # Fallback: estimate from recent volatility
        atr_pct = 0.02  # Default 2%

    # Calculate levels
    df['target_price'] = df['close'] * (1 + up_mult * atr_pct)
    df['stop_price'] = df['close'] * (1 - dn_mult * atr_pct)

    # Target/stop as percentages from current price
    df['target_pct'] = (df['target_price'] / df['close'] - 1) * 100
    df['stop_pct'] = (1 - df['stop_price'] / df['close']) * 100

    # Reward/risk ratio
    df['reward_risk'] = df['target_pct'] / df['stop_pct'].clip(lower=0.1)

    return df


def generate_predictions(
    model: lgb.LGBMClassifier,
    df: pd.DataFrame,
    feature_names: list[str]
) -> pd.DataFrame:
    """Generate predictions for all symbols.

    Args:
        model: Trained model
        df: DataFrame with features
        feature_names: List of feature names model expects

    Returns:
        DataFrame with predictions added
    """
    print("\nGenerating predictions...")

    # Prepare features
    available_features = [f for f in feature_names if f in df.columns]
    X = df[available_features].copy()

    # Handle missing features (fill with 0)
    for f in feature_names:
        if f not in X.columns:
            X[f] = 0

    # Reorder to match model
    X = X[feature_names]

    # Handle NaN/inf
    X = X.fillna(0).replace([np.inf, -np.inf], 0)

    # Predict
    probabilities = model.predict_proba(X)[:, 1]

    # Add to dataframe
    df = df.copy()
    df['probability'] = probabilities
    df['rank'] = df['probability'].rank(ascending=False, method='first').astype(int)

    print(f"  Predictions generated for {len(df)} symbols")
    print(f"  Probability range: {probabilities.min():.3f} - {probabilities.max():.3f}")

    return df


def create_rankings(
    df: pd.DataFrame,
    top_n: int = 50
) -> pd.DataFrame:
    """Create final rankings output.

    Args:
        df: DataFrame with predictions
        top_n: Number of top candidates to include

    Returns:
        Rankings DataFrame
    """
    # Select columns for output
    output_cols = [
        'rank', 'symbol', 'probability',
        'close', 'target_price', 'stop_price',
        'target_pct', 'stop_pct', 'reward_risk',
        'date'
    ]

    # Add optional columns if available
    optional_cols = [
        'sector', 'rsi_14', 'vol_regime', 'trend_score_sign',
        'alpha_mom_spy_20_ema10', 'vix_regime'
    ]
    for col in optional_cols:
        if col in df.columns:
            output_cols.append(col)

    # Filter to available columns
    output_cols = [c for c in output_cols if c in df.columns]

    # Sort by probability and take top N
    rankings = df.nlargest(top_n, 'probability')[output_cols].copy()
    rankings = rankings.reset_index(drop=True)

    return rankings


def save_predictions(
    rankings: pd.DataFrame,
    full_predictions: pd.DataFrame,
    output_dir: Path,
    date_str: str
):
    """Save prediction outputs.

    Args:
        rankings: Top N rankings
        full_predictions: All predictions
        output_dir: Output directory
        date_str: Date string for filenames
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save rankings (CSV for easy viewing)
    rankings_file = output_dir / f'rankings_{date_str}.csv'
    rankings.to_csv(rankings_file, index=False)
    print(f"\nRankings saved to: {rankings_file}")

    # Save full predictions (parquet for efficiency)
    predictions_file = output_dir / f'predictions_{date_str}.parquet'
    full_predictions.to_parquet(predictions_file, index=False)
    print(f"Full predictions saved to: {predictions_file}")

    # Save latest symlink/copy
    latest_rankings = output_dir / 'rankings_latest.csv'
    rankings.to_csv(latest_rankings, index=False)

    latest_predictions = output_dir / 'predictions_latest.parquet'
    full_predictions.to_parquet(latest_predictions, index=False)


def print_top_picks(rankings: pd.DataFrame, n: int = 20):
    """Print top picks in a nice format."""
    print("\n" + "=" * 80)
    print(f"TOP {n} MOMENTUM CANDIDATES")
    print("=" * 80)
    print()

    print(f"{'Rank':>4} {'Symbol':<6} {'Prob':>6} {'Price':>8} {'Target':>8} {'Stop':>8} "
          f"{'↑%':>6} {'↓%':>6} {'R/R':>5}")
    print("-" * 80)

    for _, row in rankings.head(n).iterrows():
        print(f"{int(row['rank']):>4} {row['symbol']:<6} {row['probability']:>6.1%} "
              f"{row['close']:>8.2f} {row['target_price']:>8.2f} {row['stop_price']:>8.2f} "
              f"{row['target_pct']:>5.1f}% {row['stop_pct']:>5.1f}% {row['reward_risk']:>5.1f}")

    print()


def main():
    parser = argparse.ArgumentParser(description='Generate Predictions')
    parser.add_argument('--date', type=str, default=None,
                        help='Target date (YYYY-MM-DD), default is latest')
    parser.add_argument('--top-n', type=int, default=50,
                        help='Number of top candidates to output')
    parser.add_argument('--output-dir', type=str, default='artifacts/predictions',
                        help='Output directory')
    parser.add_argument('--up-mult', type=float, default=3.0,
                        help='ATR multiplier for upside target')
    parser.add_argument('--dn-mult', type=float, default=1.5,
                        help='ATR multiplier for downside stop')

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("Prediction Generation")
    print("=" * 60)
    print()

    # Load model
    print("Loading model...")
    model, metadata = load_model()
    feature_names = metadata.get('features', [])
    print(f"  Model loaded: {len(feature_names)} features, {model.n_estimators_} trees")
    print(f"  Training AUC: {metadata.get('train_auc', 'N/A'):.4f}")

    # Load features
    df = load_latest_features(feature_names, args.date)

    # Get actual date from data
    pred_date = df['date'].iloc[0] if 'date' in df.columns else datetime.now()
    date_str = pd.to_datetime(pred_date).strftime('%Y%m%d')

    # Calculate targets and stops
    df = calculate_targets_and_stops(df, up_mult=args.up_mult, dn_mult=args.dn_mult)

    # Generate predictions
    df = generate_predictions(model, df, feature_names)

    # Create rankings
    rankings = create_rankings(df, top_n=args.top_n)

    # Print top picks
    print_top_picks(rankings, n=20)

    # Save outputs
    save_predictions(rankings, df, output_dir, date_str)

    # Summary stats
    print("\nPrediction Summary:")
    print(f"  Date: {pred_date}")
    print(f"  Total symbols: {len(df)}")
    print(f"  Mean probability: {df['probability'].mean():.3f}")
    print(f"  High conviction (>60%): {(df['probability'] > 0.6).sum()}")
    print(f"  Very high conviction (>70%): {(df['probability'] > 0.7).sum()}")

    print()
    print("=" * 60)
    print("Predictions complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
