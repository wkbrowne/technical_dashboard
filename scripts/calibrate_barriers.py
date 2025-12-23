#!/usr/bin/env python3
"""
Calibrate triple barrier thresholds from empirical return quantiles.

This script computes ATR-normalized barrier levels based on historical MFE/MAE
distributions, producing a frozen calibration config for target generation.

The calibration:
1. Computes forward Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE)
   in ATR units for each entry point
2. Determines target thresholds at specified quantiles to achieve desired label prevalence
3. Outputs a barrier_calibration.json with frozen thresholds for reproducible target generation

Usage:
    python scripts/calibrate_barriers.py --start 2018-01-01 --end 2024-12-31 --horizon 20

Output:
    artifacts/targets/barrier_calibration.json
"""

import argparse
import json
import hashlib
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.model_keys import ModelKey, TARGET_CONFIGS
from src.validation.target_data import (
    TargetDataValidator,
    ValidationConfig,
    validate_target_input_data,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_minimal_data(
    input_path: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    validator: Optional[TargetDataValidator] = None
) -> pd.DataFrame:
    """
    Load minimal OHLCV + ATR data required for calibration.

    Args:
        input_path: Path to parquet file with features
        start_date: Start date for calibration window (inclusive)
        end_date: End date for calibration window (inclusive)
        symbols: Optional list of symbols to include
        validator: Optional TargetDataValidator for validation

    Returns:
        DataFrame with columns: symbol, date, close, high, low, atr
    """
    required_cols = ['symbol', 'date', 'close', 'high', 'low']
    atr_cols = ['atr14', 'atr']  # Try both column names

    logger.info(f"Loading data from {input_path}")

    # Read only required columns for efficiency
    df = pd.read_parquet(input_path)

    # Normalize column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    # Check for required columns
    missing = set(required_cols) - set(df.columns)
    if missing:
        # Try adjclose -> close
        if 'close' in missing and 'adjclose' in df.columns:
            df['close'] = df['adjclose']
            missing.discard('close')

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Find ATR column
    atr_col = None
    for col in atr_cols:
        if col in df.columns:
            atr_col = col
            break

    if atr_col is None:
        raise ValueError(f"Missing ATR column. Expected one of: {atr_cols}")

    # Select and rename columns
    df = df[['symbol', 'date', 'close', 'high', 'low', atr_col]].copy()
    df = df.rename(columns={atr_col: 'atr'})

    # Convert date column
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    # Apply date filter
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]

    # Apply symbol filter
    if symbols:
        df = df[df['symbol'].isin(symbols)]

    # Use centralized validation
    if validator is None:
        validator = TargetDataValidator()

    df, summary = validator.validate_input(df)

    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

    return df


def compute_mfe_mae_distributions(
    df: pd.DataFrame,
    horizon: int = 20,
    start_every: int = 3,
    stop_mult: float = 1.0,
    validator: Optional[TargetDataValidator] = None
) -> Dict[str, np.ndarray]:
    """
    Compute forward MFE/MAE distributions in ATR units.

    Also computes stop_day_long/short which records when the stop was hit
    (or horizon+1 if never hit). This allows computing target-first rates
    for any candidate target threshold.

    Args:
        df: DataFrame with symbol, date, close, high, low, atr
        horizon: Maximum forward horizon in trading days
        start_every: Spacing between entry points
        stop_mult: Stop loss multiplier (for conditional analysis)
        validator: Optional TargetDataValidator for filtering

    Returns:
        Dictionary with:
        - mfe_long: MFE for long trades (upside in ATR units)
        - mae_long: MAE for long trades (downside in ATR units, negative)
        - mfe_short: MFE for short trades (downside in ATR units)
        - mae_short: MAE for short trades (upside in ATR units, negative)
        - stop_first_long: Boolean array, True if stop hit before 1.5 ATR proxy target
        - stop_first_short: Boolean array, True if stop hit before 1.5 ATR proxy target
        - stop_day_long: Day when stop was hit (or horizon+1 if never)
        - stop_day_short: Day when stop was hit (or horizon+1 if never)
        - high_by_day_long: Array of shape (n_samples, horizon) with cumulative max high
        - low_by_day_short: Array of shape (n_samples, horizon) with cumulative min low
        - atr_at_entry: ATR at each entry point (for computing target prices)
        - close_at_entry: Close at each entry point
    """
    if validator is None:
        validator = TargetDataValidator()

    mfe_long_list = []
    mae_long_list = []
    mfe_short_list = []
    mae_short_list = []
    stop_first_long_list = []
    stop_first_short_list = []
    stop_day_long_list = []
    stop_day_short_list = []
    atr_at_entry_list = []
    close_at_entry_list = []
    # Store per-day high/low for precise target-first computation
    high_by_day_list = []
    low_by_day_list = []

    symbols = df['symbol'].unique()
    logger.info(f"Computing MFE/MAE for {len(symbols)} symbols with horizon={horizon}, start_every={start_every}")

    processed = 0
    skipped_outliers = 0
    for symbol in symbols:
        sym_df = df[df['symbol'] == symbol].sort_values('date').reset_index(drop=True)

        if len(sym_df) < horizon + 2:
            continue

        # Extract numpy arrays for speed
        close = sym_df['close'].values
        high = sym_df['high'].values
        low = sym_df['low'].values
        atr = sym_df['atr'].values
        n = len(sym_df)

        pos = 0
        while pos < n - horizon - 1:
            # Use centralized validation for MFE/MAE computation
            result = validator.compute_mfe_mae(close, high, low, atr, pos, horizon)

            if result is None:
                # Invalid entry or outlier MFE/MAE - skip
                pos += 1
                skipped_outliers += 1
                continue

            mfe_long = result['mfe_long']
            mae_long = result['mae_long']
            mfe_short = result['mfe_short']
            mae_short = result['mae_short']

            mfe_long_list.append(mfe_long)
            mae_long_list.append(mae_long)
            mfe_short_list.append(mfe_short)
            mae_short_list.append(mae_short)

            c0 = close[pos]
            a0 = atr[pos]
            atr_at_entry_list.append(a0)
            close_at_entry_list.append(c0)

            start_idx = pos + 1
            end_idx = start_idx + horizon

            # Store daily high/low for this trajectory (in ATR units from entry)
            daily_highs = (high[start_idx:end_idx] - c0) / a0
            daily_lows = (low[start_idx:end_idx] - c0) / a0
            high_by_day_list.append(daily_highs)
            low_by_day_list.append(daily_lows)

            # Long: stop at -stop_mult ATR
            stop_price_long = c0 - stop_mult * a0
            stop_hit_days_long = np.where(low[start_idx:end_idx] <= stop_price_long)[0]
            stop_day_long = stop_hit_days_long[0] if len(stop_hit_days_long) > 0 else horizon + 1
            stop_day_long_list.append(stop_day_long)

            # Check if stop hits before a 1.5 ATR target (proxy)
            target_price_long = c0 + 1.5 * a0
            target_hit_days_long = np.where(high[start_idx:end_idx] >= target_price_long)[0]
            target_day_long = target_hit_days_long[0] if len(target_hit_days_long) > 0 else horizon + 1

            stop_first_long_list.append(stop_day_long < target_day_long)

            # Short: stop at +stop_mult ATR
            stop_price_short = c0 + stop_mult * a0
            stop_hit_days_short = np.where(high[start_idx:end_idx] >= stop_price_short)[0]
            stop_day_short = stop_hit_days_short[0] if len(stop_hit_days_short) > 0 else horizon + 1
            stop_day_short_list.append(stop_day_short)

            target_price_short = c0 - 1.5 * a0
            target_hit_days_short = np.where(low[start_idx:end_idx] <= target_price_short)[0]
            target_day_short = target_hit_days_short[0] if len(target_hit_days_short) > 0 else horizon + 1

            stop_first_short_list.append(stop_day_short < target_day_short)

            pos += start_every

        processed += 1
        if processed % 500 == 0:
            logger.info(f"  Processed {processed}/{len(symbols)} symbols, {len(mfe_long_list):,} samples")

    logger.info(f"Completed: {len(mfe_long_list):,} entry points from {processed} symbols "
               f"(skipped {skipped_outliers:,} invalid/outlier entries)")

    # Stack daily arrays - each row is a trajectory, each column is a day
    high_by_day = np.array(high_by_day_list)  # shape (n_samples, horizon)
    low_by_day = np.array(low_by_day_list)

    return {
        'mfe_long': np.array(mfe_long_list),
        'mae_long': np.array(mae_long_list),
        'mfe_short': np.array(mfe_short_list),
        'mae_short': np.array(mae_short_list),
        'stop_first_long': np.array(stop_first_long_list),
        'stop_first_short': np.array(stop_first_short_list),
        'stop_day_long': np.array(stop_day_long_list),
        'stop_day_short': np.array(stop_day_short_list),
        'high_by_day': high_by_day,
        'low_by_day': low_by_day,
        'atr_at_entry': np.array(atr_at_entry_list),
        'close_at_entry': np.array(close_at_entry_list),
    }


def estimate_trades_per_week(
    label_prevalence: float,
    symbols_universe: int = 500,
    top_n: int = 5,
    prob_threshold: float = 0.6
) -> Tuple[float, float]:
    """
    Estimate trades per week from label prevalence and portfolio parameters.

    This is an approximate mapping. Actual trades depend on model quality.

    Args:
        label_prevalence: Fraction of entries labeled as positive
        symbols_universe: Number of symbols in trading universe
        top_n: Number of top-ranked signals selected per rebalance
        prob_threshold: Minimum model probability threshold

    Returns:
        Tuple of (lower_bound, upper_bound) trades per week
    """
    # Expected signals per day = universe * prevalence * P(model_prob > threshold)
    # Assuming model is decent, P(prob > threshold | positive) ~ 0.3-0.5
    # and P(prob > threshold | negative) ~ 0.05-0.1

    # Conservative estimate: actual positives with high confidence
    daily_signals_low = symbols_universe * label_prevalence * 0.2
    daily_signals_high = symbols_universe * label_prevalence * 0.5

    # With weekly rebalance and start_every=3, we get ~1-2 decision points per week
    # If we pick top_n from available signals:
    trades_low = min(top_n, daily_signals_low) * 1  # 1 rebalance per week
    trades_high = min(top_n, daily_signals_high) * 2  # up to 2 decision points

    return (trades_low, trades_high)


def calibrate_thresholds(
    distributions: Dict[str, np.ndarray],
    target_prevalence_normal: float = 0.12,
    parabolic_ratio: float = 0.5,
    stop_mult: float = 1.0
) -> Dict[str, Dict[str, Any]]:
    """
    Calibrate barrier thresholds to achieve desired target-first (win) rates.

    The key insight is that we want to find the profit target where ~target_prevalence
    of ALL trajectories hit the TARGET before hitting the STOP. This is the actual
    label prevalence we'll see in training data.

    We use the stored high_by_day/low_by_day arrays to compute the exact target-first
    rate for any candidate threshold, then use binary search to find the threshold
    that achieves the desired prevalence.

    Args:
        distributions: Output from compute_mfe_mae_distributions (includes high_by_day, etc.)
        target_prevalence_normal: Target label prevalence for NORMAL models (~8-15%)
                                  This is the actual target-first rate across ALL trajectories.
        parabolic_ratio: PARABOLIC positives as fraction of NORMAL (default 0.5)
        stop_mult: Stop loss multiplier (fixed at 1.0 ATR)

    Returns:
        Dictionary with calibrated configs for each model key
    """
    mfe_long = distributions['mfe_long']
    mfe_short = distributions['mfe_short']
    mae_long = distributions['mae_long']
    mae_short = distributions['mae_short']
    high_by_day = distributions['high_by_day']  # shape (n_samples, horizon), in ATR units
    low_by_day = distributions['low_by_day']
    stop_day_long = distributions['stop_day_long']
    stop_day_short = distributions['stop_day_short']

    n_samples = len(mfe_long)
    horizon = high_by_day.shape[1]

    logger.info(f"MFE distribution (all trajectories):")
    logger.info(f"  Long: p50={np.median(mfe_long):.2f}, p75={np.percentile(mfe_long, 75):.2f}, "
                f"p90={np.percentile(mfe_long, 90):.2f}")
    logger.info(f"  Short: p50={np.median(mfe_short):.2f}, p75={np.percentile(mfe_short, 75):.2f}, "
                f"p90={np.percentile(mfe_short, 90):.2f}")

    def compute_target_first_rate_long(target_mult: float) -> float:
        """Compute fraction of trajectories where target is hit before stop (LONG)."""
        # For each trajectory, find first day high >= target_mult
        # Target is hit if any day's high >= target_mult
        # We need: first day where high >= target_mult < stop_day_long
        target_hit_mask = high_by_day >= target_mult  # shape (n_samples, horizon)
        # First day where target is hit (or horizon if never)
        target_day = np.argmax(target_hit_mask, axis=1)
        # If never hit, argmax returns 0, so we need to check
        never_hit = ~np.any(target_hit_mask, axis=1)
        target_day = np.where(never_hit, horizon + 1, target_day)

        # Target first if target_day < stop_day_long
        target_first = target_day < stop_day_long
        return np.mean(target_first)

    def compute_target_first_rate_short(target_mult: float) -> float:
        """Compute fraction of trajectories where target is hit before stop (SHORT)."""
        # For short, target is hit when low <= -target_mult (going down)
        target_hit_mask = low_by_day <= -target_mult
        target_day = np.argmax(target_hit_mask, axis=1)
        never_hit = ~np.any(target_hit_mask, axis=1)
        target_day = np.where(never_hit, horizon + 1, target_day)

        target_first = target_day < stop_day_short
        return np.mean(target_first)

    from scipy.optimize import brentq

    def find_threshold_long(target_prev: float, min_threshold: float = 1.5) -> float:
        """Find threshold where target_prev fraction hit target before stop (LONG)."""
        def objective(threshold):
            return compute_target_first_rate_long(threshold) - target_prev

        # Bounds
        low = min_threshold
        high = np.percentile(mfe_long, 99)

        rate_at_low = compute_target_first_rate_long(low)
        if rate_at_low < target_prev:
            logger.warning(f"Long: Cannot achieve target prevalence {target_prev:.1%} - "
                          f"max achievable at {low:.1f} ATR is {rate_at_low:.1%}")
            return low

        rate_at_high = compute_target_first_rate_long(high)
        if rate_at_high > target_prev:
            return high

        try:
            return brentq(objective, low, high, xtol=0.01)
        except ValueError:
            return min_threshold

    def find_threshold_short(target_prev: float, min_threshold: float = 1.5) -> float:
        """Find threshold where target_prev fraction hit target before stop (SHORT)."""
        def objective(threshold):
            return compute_target_first_rate_short(threshold) - target_prev

        low = min_threshold
        high = np.percentile(mfe_short, 99)

        rate_at_low = compute_target_first_rate_short(low)
        if rate_at_low < target_prev:
            logger.warning(f"Short: Cannot achieve target prevalence {target_prev:.1%} - "
                          f"max achievable at {low:.1f} ATR is {rate_at_low:.1%}")
            return low

        rate_at_high = compute_target_first_rate_short(high)
        if rate_at_high > target_prev:
            return high

        try:
            return brentq(objective, low, high, xtol=0.01)
        except ValueError:
            return min_threshold

    # Minimum target: at least 1.5 ATR (gives reasonable risk/reward with 1 ATR stop)
    min_target = max(stop_mult + 0.5, 1.5)

    # Find thresholds using target-first rate
    long_normal_up = find_threshold_long(target_prevalence_normal, min_target)
    short_normal_dn = find_threshold_short(target_prevalence_normal, min_target)

    parabolic_target = target_prevalence_normal * parabolic_ratio
    long_parabolic_up = find_threshold_long(parabolic_target, min_target)
    short_parabolic_dn = find_threshold_short(parabolic_target, min_target)

    # Ensure parabolic > normal
    long_parabolic_up = max(long_parabolic_up, long_normal_up + 0.3)
    short_parabolic_dn = max(short_parabolic_dn, short_normal_dn + 0.3)

    # Compute actual prevalences
    prev_long_normal = compute_target_first_rate_long(long_normal_up)
    prev_long_parabolic = compute_target_first_rate_long(long_parabolic_up)
    prev_short_normal = compute_target_first_rate_short(short_normal_dn)
    prev_short_parabolic = compute_target_first_rate_short(short_parabolic_dn)

    logger.info(f"Calibrated thresholds (target-first rates):")
    logger.info(f"  Long normal: {long_normal_up:.2f} ATR -> {prev_long_normal:.1%} target-first")
    logger.info(f"  Long parabolic: {long_parabolic_up:.2f} ATR -> {prev_long_parabolic:.1%} target-first")
    logger.info(f"  Short normal: {short_normal_dn:.2f} ATR -> {prev_short_normal:.1%} target-first")
    logger.info(f"  Short parabolic: {short_parabolic_dn:.2f} ATR -> {prev_short_parabolic:.1%} target-first")

    # Compute prevalence helper
    def compute_prevalence(rate: float) -> float:
        """Return the target-first rate as prevalence."""
        return rate

    # Quantile targets (for reporting)
    q_normal = 1.0 - target_prevalence_normal
    q_parabolic = 1.0 - parabolic_target

    # Build calibrated configs
    calibrated = {
        ModelKey.LONG_NORMAL.value: {
            "up_mult": round(long_normal_up, 2),
            "dn_mult": stop_mult,
            "max_horizon": 20,
            "start_every": 3,
            "target_col": "target_long_normal",
            "calibration_info": {
                "quantile": round(q_normal, 3),
                "estimated_prevalence": round(prev_long_normal, 4),
                "mfe_p50": round(np.percentile(mfe_long, 50), 2),
                "mfe_p75": round(np.percentile(mfe_long, 75), 2),
                "mfe_p90": round(np.percentile(mfe_long, 90), 2),
            }
        },
        ModelKey.LONG_PARABOLIC.value: {
            "up_mult": round(long_parabolic_up, 2),
            "dn_mult": stop_mult,
            "max_horizon": 20,
            "start_every": 3,
            "target_col": "target_long_parabolic",
            "calibration_info": {
                "quantile": round(q_parabolic, 3),
                "estimated_prevalence": round(prev_long_parabolic, 4),
            }
        },
        ModelKey.SHORT_NORMAL.value: {
            "up_mult": stop_mult,
            "dn_mult": round(short_normal_dn, 2),
            "max_horizon": 20,
            "start_every": 3,
            "target_col": "target_short_normal",
            "calibration_info": {
                "quantile": round(q_normal, 3),
                "estimated_prevalence": round(prev_short_normal, 4),
                "mfe_p50": round(np.percentile(mfe_short, 50), 2),
                "mfe_p75": round(np.percentile(mfe_short, 75), 2),
                "mfe_p90": round(np.percentile(mfe_short, 90), 2),
            }
        },
        ModelKey.SHORT_PARABOLIC.value: {
            "up_mult": stop_mult,
            "dn_mult": round(short_parabolic_dn, 2),
            "max_horizon": 20,
            "start_every": 3,
            "target_col": "target_short_parabolic",
            "calibration_info": {
                "quantile": round(q_parabolic, 3),
                "estimated_prevalence": round(prev_short_parabolic, 4),
            }
        },
    }

    return calibrated


def compute_universe_hash(df: pd.DataFrame) -> str:
    """Compute a hash of the symbol universe for reproducibility tracking."""
    symbols = sorted(df['symbol'].unique())
    return hashlib.md5(','.join(symbols).encode()).hexdigest()[:12]


def generate_calibration_report(
    distributions: Dict[str, np.ndarray],
    calibrated_configs: Dict[str, Dict],
    stop_mult: float = 1.0
) -> str:
    """Generate human-readable calibration report."""
    lines = []
    lines.append("=" * 70)
    lines.append("BARRIER CALIBRATION REPORT")
    lines.append("=" * 70)
    lines.append("")

    n = len(distributions['mfe_long'])
    lines.append(f"Total entry points analyzed: {n:,}")
    lines.append(f"Fixed stop multiplier: {stop_mult} ATR")
    lines.append("")

    # MFE/MAE statistics
    lines.append("MFE/MAE Distribution Statistics (in ATR units):")
    lines.append("-" * 50)

    for side in ['long', 'short']:
        mfe = distributions[f'mfe_{side}']
        mae = distributions[f'mae_{side}']
        stop_first = distributions[f'stop_first_{side}']
        n_eligible = int(np.sum(~stop_first))

        lines.append(f"  {side.upper()}:")
        lines.append(f"    MFE: mean={np.mean(mfe):.2f}, median={np.median(mfe):.2f}, "
                    f"p75={np.percentile(mfe, 75):.2f}, p90={np.percentile(mfe, 90):.2f}")
        lines.append(f"    MAE: mean={np.mean(mae):.2f}, median={np.median(mae):.2f}, "
                    f"p25={np.percentile(mae, 25):.2f}, p10={np.percentile(mae, 10):.2f}")
        lines.append(f"    Stop-first rate: {100 * np.mean(stop_first):.1f}% "
                    f"({n_eligible:,} eligible for profit target)")

    lines.append("")
    lines.append("Calibrated Barrier Multipliers:")
    lines.append("-" * 50)
    lines.append("  Targets set at MFE quantiles to achieve desired label prevalence.")
    lines.append("  Note: Actual win rate depends on target vs stop race dynamics.")
    lines.append("")

    for model_key, config in calibrated_configs.items():
        info = config.get('calibration_info', {})
        lines.append(f"  {model_key}:")
        lines.append(f"    up_mult: {config['up_mult']:.2f}, dn_mult: {config['dn_mult']:.2f}")

        prev = info.get('estimated_prevalence', 0)
        lines.append(f"    Est. prevalence: {prev * 100:.1f}% (MFE >= target)")

        # Estimate trades per week
        trades_low, trades_high = estimate_trades_per_week(prev)
        lines.append(f"    Est. trades/week: {trades_low:.1f} - {trades_high:.1f} (with top-5 selection)")

    lines.append("")
    lines.append("Suggested TARGET_CONFIGS update:")
    lines.append("-" * 50)
    lines.append("# Copy this to src/config/model_keys.py or use --target-config flag")
    lines.append("")

    for model_key, config in calibrated_configs.items():
        enum_name = model_key.upper().replace('_', '_')
        lines.append(f"    ModelKey.{enum_name}: {{")
        lines.append(f'        "up_mult": {config["up_mult"]},')
        lines.append(f'        "dn_mult": {config["dn_mult"]},')
        lines.append(f'        "max_horizon": {config["max_horizon"]},')
        lines.append(f'        "start_every": {config["start_every"]},')
        lines.append(f'        "target_col": "{config["target_col"]}",')
        lines.append("    },")

    lines.append("")
    lines.append("=" * 70)

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Calibrate triple barrier thresholds from empirical data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard calibration on 2018-2024 data
    python scripts/calibrate_barriers.py --start 2018-01-01 --end 2024-12-31

    # Calibrate for fewer trades per week
    python scripts/calibrate_barriers.py --target-prevalence 0.08

    # Use custom input data
    python scripts/calibrate_barriers.py --input artifacts/features_complete.parquet
        """
    )

    parser.add_argument('--input', type=str, default='artifacts/features_complete.parquet',
                       help='Input parquet file with OHLCV + ATR data')
    parser.add_argument('--output', type=str, default='artifacts/targets/barrier_calibration.json',
                       help='Output calibration JSON file')
    parser.add_argument('--start', type=str, default=None,
                       help='Start date for calibration window (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                       help='End date for calibration window (YYYY-MM-DD)')
    parser.add_argument('--horizon', type=int, default=20,
                       help='Maximum forward horizon in trading days')
    parser.add_argument('--start-every', type=int, default=3,
                       help='Spacing between entry points')
    parser.add_argument('--stop-mult', type=float, default=1.0,
                       help='Stop loss multiplier (fixed for risk normalization)')
    parser.add_argument('--target-prevalence', type=float, default=0.16,
                       help='Target label prevalence for NORMAL models (0.10-0.20 recommended)')
    parser.add_argument('--parabolic-ratio', type=float, default=0.5,
                       help='PARABOLIC positives as fraction of NORMAL')
    parser.add_argument('--symbols', type=str, default=None,
                       help='Comma-separated list of symbols to include (default: all)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load data
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]

    df = load_minimal_data(
        input_path,
        start_date=args.start,
        end_date=args.end,
        symbols=symbols
    )

    if len(df) == 0:
        logger.error("No data available for calibration")
        sys.exit(1)

    # Compute MFE/MAE distributions
    distributions = compute_mfe_mae_distributions(
        df,
        horizon=args.horizon,
        start_every=args.start_every,
        stop_mult=args.stop_mult
    )

    if len(distributions['mfe_long']) < 1000:
        logger.warning(f"Only {len(distributions['mfe_long'])} samples - results may be unstable")

    # Calibrate thresholds
    calibrated = calibrate_thresholds(
        distributions,
        target_prevalence_normal=args.target_prevalence,
        parabolic_ratio=args.parabolic_ratio,
        stop_mult=args.stop_mult
    )

    # Generate report
    report = generate_calibration_report(distributions, calibrated, args.stop_mult)
    print(report)

    # Build output structure
    output = {
        "calibration_metadata": {
            "created_at": datetime.now().astimezone().isoformat(),
            "input_file": str(input_path),
            "date_range": {
                "start": args.start or str(df['date'].min().date()),
                "end": args.end or str(df['date'].max().date()),
            },
            "horizon": args.horizon,
            "start_every": args.start_every,
            "stop_mult": args.stop_mult,
            "target_prevalence_normal": args.target_prevalence,
            "parabolic_ratio": args.parabolic_ratio,
            "n_symbols": int(df['symbol'].nunique()),
            "n_samples": int(len(distributions['mfe_long'])),
            "universe_hash": compute_universe_hash(df),
        },
        "mfe_mae_statistics": {
            "long": {
                "mfe_mean": round(float(np.mean(distributions['mfe_long'])), 3),
                "mfe_median": round(float(np.median(distributions['mfe_long'])), 3),
                "mfe_p75": round(float(np.percentile(distributions['mfe_long'], 75)), 3),
                "mfe_p90": round(float(np.percentile(distributions['mfe_long'], 90)), 3),
                "mae_mean": round(float(np.mean(distributions['mae_long'])), 3),
                "mae_median": round(float(np.median(distributions['mae_long'])), 3),
                "stop_first_rate": round(float(np.mean(distributions['stop_first_long'])), 4),
            },
            "short": {
                "mfe_mean": round(float(np.mean(distributions['mfe_short'])), 3),
                "mfe_median": round(float(np.median(distributions['mfe_short'])), 3),
                "mfe_p75": round(float(np.percentile(distributions['mfe_short'], 75)), 3),
                "mfe_p90": round(float(np.percentile(distributions['mfe_short'], 90)), 3),
                "mae_mean": round(float(np.mean(distributions['mae_short'])), 3),
                "mae_median": round(float(np.median(distributions['mae_short'])), 3),
                "stop_first_rate": round(float(np.mean(distributions['stop_first_short'])), 4),
            },
        },
        "target_configs": calibrated,
        "warnings": [],
    }

    # Add warnings if needed
    for model_key, config in calibrated.items():
        info = config.get('calibration_info', {})
        prev = info.get('estimated_prevalence', 0)

        # Check if positives are too rare
        n_positives = int(prev * len(distributions['mfe_long']))
        if n_positives < 5000:
            output["warnings"].append(
                f"{model_key}: Only ~{n_positives:,} positive samples (prevalence {prev:.1%}). "
                f"Consider increasing target_prevalence."
            )

        # Check if positives are too common
        if prev > 0.25:
            output["warnings"].append(
                f"{model_key}: Prevalence {prev:.1%} is high. Trades will be frequent."
            )

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Calibration saved to {output_path}")

    # Print warnings
    if output["warnings"]:
        print("\nWARNINGS:")
        for w in output["warnings"]:
            print(f"  - {w}")

    print(f"\nCalibration complete. To use:")
    print(f"  python scripts/recompute_targets.py --target-config {output_path}")


if __name__ == '__main__':
    main()
