#!/usr/bin/env python3
"""
Diagnostic analysis for barrier calibration quality.

Checks:
1. Positives per CV fold for each label - ensures stable AUC estimates
2. Symbol concentration in parabolic positives - identifies if ultra-vol names dominate
3. ATR% distribution of positive labels - flags potential outlier-driven calibration

Usage:
    python scripts/diagnose_calibration.py --target-config artifacts/targets/barrier_calibration.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.model_keys import ModelKey, TARGET_CONFIGS
from src.validation.target_data import TargetDataValidator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(
    input_path: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load OHLCV + ATR data."""
    logger.info(f"Loading data from {input_path}")
    df = pd.read_parquet(input_path)
    df.columns = [c.lower() for c in df.columns]

    # Normalize ATR column
    if 'atr14' in df.columns and 'atr' not in df.columns:
        df['atr'] = df['atr14']

    # Date filter
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]

    # Basic validation
    validator = TargetDataValidator()
    df, _ = validator.validate_input(df)

    # Compute ATR percent
    df['atr_pct'] = 100 * df['atr'] / df['close']

    return df


def load_calibration(config_path: Path) -> Dict:
    """Load calibration config."""
    with open(config_path, 'r') as f:
        return json.load(f)


def compute_target_labels(
    df: pd.DataFrame,
    target_configs: Dict,
    horizon: int = 20,
    start_every: int = 3,
    stop_mult: float = 1.0,
) -> pd.DataFrame:
    """Compute target labels for each model key."""
    results = []

    symbols = df['symbol'].unique()
    logger.info(f"Computing labels for {len(symbols)} symbols...")

    for symbol in symbols:
        sym_df = df[df['symbol'] == symbol].sort_values('date').reset_index(drop=True)

        if len(sym_df) < horizon + 2:
            continue

        close = sym_df['close'].values
        high = sym_df['high'].values
        low = sym_df['low'].values
        atr = sym_df['atr'].values
        dates = sym_df['date'].values
        atr_pct = sym_df['atr_pct'].values
        n = len(sym_df)

        pos = 0
        while pos < n - horizon - 1:
            c0 = close[pos]
            a0 = atr[pos]

            if a0 <= 0 or not np.isfinite(a0):
                pos += 1
                continue

            start_idx = pos + 1
            end_idx = start_idx + horizon

            entry = {
                'symbol': symbol,
                'date': dates[pos],
                'close': c0,
                'atr': a0,
                'atr_pct': atr_pct[pos],
            }

            # Compute labels for each model
            for model_key, config in target_configs.items():
                up_mult = config['up_mult']
                dn_mult = config['dn_mult']

                # Target and stop prices
                target_price_long = c0 + up_mult * a0
                stop_price_long = c0 - dn_mult * a0
                target_price_short = c0 - dn_mult * a0
                stop_price_short = c0 + up_mult * a0

                # Find first hit day
                target_days_long = np.where(high[start_idx:end_idx] >= target_price_long)[0]
                stop_days_long = np.where(low[start_idx:end_idx] <= stop_price_long)[0]
                target_days_short = np.where(low[start_idx:end_idx] <= target_price_short)[0]
                stop_days_short = np.where(high[start_idx:end_idx] >= stop_price_short)[0]

                target_day_long = target_days_long[0] if len(target_days_long) > 0 else horizon + 1
                stop_day_long = stop_days_long[0] if len(stop_days_long) > 0 else horizon + 1
                target_day_short = target_days_short[0] if len(target_days_short) > 0 else horizon + 1
                stop_day_short = stop_days_short[0] if len(stop_days_short) > 0 else horizon + 1

                # Determine label
                if 'long' in model_key:
                    if target_day_long < stop_day_long and target_day_long <= horizon:
                        label = 1
                    elif stop_day_long < target_day_long and stop_day_long <= horizon:
                        label = -1
                    else:
                        label = 0
                else:  # short
                    if target_day_short < stop_day_short and target_day_short <= horizon:
                        label = 1
                    elif stop_day_short < target_day_short and stop_day_short <= horizon:
                        label = -1
                    else:
                        label = 0

                entry[f'label_{model_key}'] = label

            results.append(entry)
            pos += start_every

    return pd.DataFrame(results)


def analyze_positives_per_fold(
    labels_df: pd.DataFrame,
    target_configs: Dict,
    n_folds: int = 5,
) -> Dict[str, Dict]:
    """Analyze positives per CV fold for each model."""
    print("\n" + "=" * 70)
    print("1. POSITIVES PER CV FOLD")
    print("=" * 70)
    print(f"\nUsing {n_folds}-fold time-series split (non-shuffled)")

    # Sort by date for time-series split
    labels_df = labels_df.sort_values('date').reset_index(drop=True)
    n = len(labels_df)
    fold_size = n // n_folds

    results = {}

    for model_key in target_configs.keys():
        col = f'label_{model_key}'
        if col not in labels_df.columns:
            continue

        positives_per_fold = []
        total_per_fold = []

        for fold in range(n_folds):
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < n_folds - 1 else n

            fold_df = labels_df.iloc[start_idx:end_idx]
            n_pos = (fold_df[col] == 1).sum()
            n_total = len(fold_df)

            positives_per_fold.append(n_pos)
            total_per_fold.append(n_total)

        total_pos = (labels_df[col] == 1).sum()
        prevalence = total_pos / len(labels_df) * 100

        results[model_key] = {
            'positives_per_fold': positives_per_fold,
            'total_per_fold': total_per_fold,
            'total_positives': total_pos,
            'prevalence': prevalence,
            'min_positives': min(positives_per_fold),
            'max_positives': max(positives_per_fold),
            'cv_of_positives': np.std(positives_per_fold) / np.mean(positives_per_fold) if np.mean(positives_per_fold) > 0 else float('inf'),
        }

        print(f"\n  {model_key}:")
        print(f"    Total positives: {total_pos:,} ({prevalence:.1f}%)")
        print(f"    Per fold: {positives_per_fold}")
        print(f"    Min/Max: {min(positives_per_fold):,} / {max(positives_per_fold):,}")
        print(f"    CV of positives: {results[model_key]['cv_of_positives']:.2f}")

        # Warning thresholds
        if min(positives_per_fold) < 100:
            print(f"    ⚠️  WARNING: Min fold has <100 positives - AUC will be unstable!")
        elif min(positives_per_fold) < 500:
            print(f"    ⚠️  CAUTION: Min fold has <500 positives - consider higher prevalence")

    return results


def analyze_symbol_concentration(
    labels_df: pd.DataFrame,
    target_configs: Dict,
    top_n: int = 20,
) -> Dict[str, Dict]:
    """Analyze which symbols dominate positive labels."""
    print("\n" + "=" * 70)
    print("2. SYMBOL CONCENTRATION IN POSITIVES")
    print("=" * 70)

    results = {}

    for model_key in target_configs.keys():
        col = f'label_{model_key}'
        if col not in labels_df.columns:
            continue

        # Get positive samples
        pos_df = labels_df[labels_df[col] == 1]

        if len(pos_df) == 0:
            print(f"\n  {model_key}: No positive samples!")
            continue

        # Symbol counts
        symbol_counts = pos_df['symbol'].value_counts()
        total_pos = len(pos_df)
        n_symbols_with_pos = len(symbol_counts)

        # Concentration metrics
        top_10_pct = symbol_counts.head(10).sum() / total_pos * 100
        top_20_pct = symbol_counts.head(20).sum() / total_pos * 100

        # Gini coefficient (measure of inequality)
        counts = symbol_counts.values
        n = len(counts)
        if n > 1:
            sorted_counts = np.sort(counts)
            cumulative = np.cumsum(sorted_counts)
            gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
        else:
            gini = 0

        results[model_key] = {
            'n_symbols_with_positives': n_symbols_with_pos,
            'top_10_concentration': top_10_pct,
            'top_20_concentration': top_20_pct,
            'gini_coefficient': gini,
            'top_symbols': symbol_counts.head(top_n).to_dict(),
        }

        print(f"\n  {model_key}:")
        print(f"    Symbols with positives: {n_symbols_with_pos:,}")
        print(f"    Top 10 symbols: {top_10_pct:.1f}% of positives")
        print(f"    Top 20 symbols: {top_20_pct:.1f}% of positives")
        print(f"    Gini coefficient: {gini:.3f} (0=equal, 1=concentrated)")

        # Warning
        if top_10_pct > 30:
            print(f"    ⚠️  WARNING: Top 10 symbols have >{top_10_pct:.0f}% of positives - highly concentrated!")

        print(f"\n    Top {min(10, len(symbol_counts))} symbols:")
        for sym, count in symbol_counts.head(10).items():
            pct = count / total_pos * 100
            print(f"      {sym}: {count:,} ({pct:.1f}%)")

    return results


def analyze_atr_distribution(
    labels_df: pd.DataFrame,
    target_configs: Dict,
) -> Dict[str, Dict]:
    """Analyze ATR% distribution of positive labels."""
    print("\n" + "=" * 70)
    print("3. ATR% DISTRIBUTION OF POSITIVES")
    print("=" * 70)
    print("\n  ATR% = ATR / Close * 100 (daily volatility as % of price)")

    results = {}

    # Overall ATR% distribution
    print(f"\n  Overall ATR% distribution (all samples):")
    print(f"    p10={labels_df['atr_pct'].quantile(0.10):.2f}%")
    print(f"    p25={labels_df['atr_pct'].quantile(0.25):.2f}%")
    print(f"    p50={labels_df['atr_pct'].quantile(0.50):.2f}%")
    print(f"    p75={labels_df['atr_pct'].quantile(0.75):.2f}%")
    print(f"    p90={labels_df['atr_pct'].quantile(0.90):.2f}%")

    for model_key in target_configs.keys():
        col = f'label_{model_key}'
        if col not in labels_df.columns:
            continue

        pos_df = labels_df[labels_df[col] == 1]
        neg_df = labels_df[labels_df[col] == -1]

        if len(pos_df) == 0:
            continue

        # ATR% stats for positives
        pos_atr_pct = pos_df['atr_pct']

        # Check for ultra-high/low volatility concentration
        ultra_low = (pos_atr_pct < 0.5).sum()
        ultra_high = (pos_atr_pct > 20).sum()
        low_vol = (pos_atr_pct < 2).sum()
        high_vol = (pos_atr_pct > 10).sum()

        results[model_key] = {
            'mean_atr_pct': pos_atr_pct.mean(),
            'median_atr_pct': pos_atr_pct.median(),
            'p10': pos_atr_pct.quantile(0.10),
            'p90': pos_atr_pct.quantile(0.90),
            'ultra_low_count': ultra_low,
            'ultra_high_count': ultra_high,
            'ultra_low_pct': ultra_low / len(pos_df) * 100,
            'ultra_high_pct': ultra_high / len(pos_df) * 100,
        }

        print(f"\n  {model_key} (positives only):")
        print(f"    Mean ATR%: {pos_atr_pct.mean():.2f}%")
        print(f"    Median ATR%: {pos_atr_pct.median():.2f}%")
        print(f"    p10-p90: {pos_atr_pct.quantile(0.10):.2f}% - {pos_atr_pct.quantile(0.90):.2f}%")
        print(f"    Ultra-low (<0.5%): {ultra_low:,} ({ultra_low/len(pos_df)*100:.1f}%)")
        print(f"    Ultra-high (>20%): {ultra_high:,} ({ultra_high/len(pos_df)*100:.1f}%)")
        print(f"    Low vol (<2%): {low_vol:,} ({low_vol/len(pos_df)*100:.1f}%)")
        print(f"    High vol (>10%): {high_vol:,} ({high_vol/len(pos_df)*100:.1f}%)")

        # Warnings
        if ultra_high / len(pos_df) > 0.2:
            print(f"    ⚠️  WARNING: >{ultra_high/len(pos_df)*100:.0f}% from ultra-high vol stocks!")
        if ultra_low / len(pos_df) > 0.1:
            print(f"    ⚠️  CAUTION: {ultra_low/len(pos_df)*100:.0f}% from ultra-low vol stocks")

    return results


def suggest_filters(
    fold_results: Dict,
    concentration_results: Dict,
    atr_results: Dict,
) -> None:
    """Suggest filtering based on analysis."""
    print("\n" + "=" * 70)
    print("4. RECOMMENDED FILTERS")
    print("=" * 70)

    suggestions = []

    # Check for low positive counts
    for model_key, data in fold_results.items():
        if data['min_positives'] < 100:
            suggestions.append(
                f"• {model_key}: Consider increasing target_prevalence - "
                f"min fold has only {data['min_positives']} positives"
            )

    # Check for high concentration
    for model_key, data in concentration_results.items():
        if data['top_10_concentration'] > 30:
            suggestions.append(
                f"• {model_key}: Consider symbol-level caps - "
                f"top 10 symbols have {data['top_10_concentration']:.0f}% of positives"
            )

    # Check for ATR% extremes
    for model_key, data in atr_results.items():
        if data['ultra_high_pct'] > 20:
            suggestions.append(
                f"• {model_key}: Consider ATR% filter (< 20%) - "
                f"{data['ultra_high_pct']:.0f}% from ultra-high vol stocks"
            )
        if data['ultra_low_pct'] > 10:
            suggestions.append(
                f"• {model_key}: Consider ATR% filter (> 0.5%) - "
                f"{data['ultra_low_pct']:.0f}% from ultra-low vol stocks"
            )

    if suggestions:
        print("\n  Suggested filters:")
        for s in suggestions:
            print(f"    {s}")
    else:
        print("\n  ✓ No major issues detected - calibration looks healthy!")

    print("\n  General recommendations:")
    print("    • ATR% filter: 0.5% < atr_pct < 20% (excludes illiquid and ultra-vol)")
    print("    • Symbol cap: max 5% of positives per symbol")
    print("    • Min positives per fold: 500+ for stable AUC")


def main():
    parser = argparse.ArgumentParser(
        description='Diagnostic analysis for barrier calibration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--input', type=str, default='artifacts/features_complete.parquet',
                       help='Input parquet file')
    parser.add_argument('--target-config', type=str, default='artifacts/targets/barrier_calibration.json',
                       help='Calibration config file')
    parser.add_argument('--start', type=str, default=None,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--n-folds', type=int, default=5,
                       help='Number of CV folds')

    args = parser.parse_args()

    # Load data
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    df = load_data(input_path, args.start, args.end)
    logger.info(f"Loaded {len(df):,} rows, {df['symbol'].nunique():,} symbols")

    # Load calibration
    config_path = Path(args.target_config)
    if config_path.exists():
        calibration = load_calibration(config_path)
        target_configs = calibration['target_configs']
        logger.info(f"Loaded calibration from {config_path}")
    else:
        logger.warning(f"Calibration file not found, using defaults")
        target_configs = {k.value: v for k, v in TARGET_CONFIGS.items()}

    # Compute labels
    logger.info("Computing target labels...")
    labels_df = compute_target_labels(
        df, target_configs,
        horizon=target_configs.get('long_normal', {}).get('max_horizon', 20),
        start_every=target_configs.get('long_normal', {}).get('start_every', 3),
    )
    logger.info(f"Computed {len(labels_df):,} labeled samples")

    # Run analyses
    fold_results = analyze_positives_per_fold(labels_df, target_configs, args.n_folds)
    concentration_results = analyze_symbol_concentration(labels_df, target_configs)
    atr_results = analyze_atr_distribution(labels_df, target_configs)

    # Suggestions
    suggest_filters(fold_results, concentration_results, atr_results)

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
