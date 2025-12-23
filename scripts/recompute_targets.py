#!/usr/bin/env python3
"""
Recompute triple barrier targets WITHOUT running the full feature pipeline.

This script loads minimal OHLCV + ATR data and regenerates targets using either:
1. Default TARGET_CONFIGS from src/config/model_keys.py
2. Custom calibration from a barrier_calibration.json file

This is much faster than running the full feature pipeline when you only need to
update targets (e.g., after recalibrating barrier thresholds).

Usage:
    # Using default configs
    python scripts/recompute_targets.py

    # Using calibrated thresholds
    python scripts/recompute_targets.py --target-config artifacts/targets/barrier_calibration.json

    # Specific model keys only
    python scripts/recompute_targets.py --model-keys long_normal,short_normal

Output:
    artifacts/targets/targets.parquet (or custom path via --output)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.model_keys import ModelKey, TARGET_CONFIGS, get_target_config
from src.features.target_generation import generate_multi_targets_parallel
from src.validation.target_data import TargetDataValidator, validate_target_input_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_calibration_config(config_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load target configs from calibration JSON file.

    Args:
        config_path: Path to barrier_calibration.json

    Returns:
        Dictionary mapping model_key string -> config dict
    """
    with open(config_path, 'r') as f:
        data = json.load(f)

    if 'target_configs' not in data:
        raise ValueError(f"Invalid calibration file: missing 'target_configs' key")

    configs = data['target_configs']

    # Validate required keys
    required_keys = {'up_mult', 'dn_mult', 'max_horizon', 'start_every'}
    for model_key, config in configs.items():
        missing = required_keys - set(config.keys())
        if missing:
            raise ValueError(f"Config for {model_key} missing keys: {missing}")

    logger.info(f"Loaded calibration from {config_path}")
    logger.info(f"  Created: {data.get('calibration_metadata', {}).get('created_at', 'unknown')}")
    logger.info(f"  Models: {list(configs.keys())}")

    return configs


def load_minimal_data(
    input_path: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    validator: Optional[TargetDataValidator] = None
) -> pd.DataFrame:
    """
    Load minimal OHLCV + ATR data required for target generation.

    Args:
        input_path: Path to parquet file with features or OHLCV data
        start_date: Optional start date filter (inclusive)
        end_date: Optional end date filter (inclusive)
        validator: Optional TargetDataValidator for centralized validation

    Returns:
        DataFrame with columns: symbol, date, close, high, low, atr
    """
    logger.info(f"Loading data from {input_path}")

    # Try to read only required columns for efficiency
    try:
        # Check available columns first
        schema = pd.read_parquet(input_path, columns=[]).columns.tolist()

        # Determine which columns to load
        cols_to_load = ['symbol', 'date']

        # Price columns (prefer adjclose for close)
        if 'adjclose' in schema:
            cols_to_load.append('adjclose')
        elif 'close' in schema:
            cols_to_load.append('close')
        else:
            raise ValueError("No close or adjclose column found")

        if 'high' in schema:
            cols_to_load.append('high')
        else:
            raise ValueError("No high column found")

        if 'low' in schema:
            cols_to_load.append('low')
        else:
            raise ValueError("No low column found")

        # ATR column
        if 'atr14' in schema:
            cols_to_load.append('atr14')
        elif 'atr' in schema:
            cols_to_load.append('atr')
        else:
            raise ValueError("No atr14 or atr column found")

        df = pd.read_parquet(input_path, columns=cols_to_load)

    except Exception as e:
        logger.warning(f"Efficient column loading failed: {e}. Loading full file.")
        df = pd.read_parquet(input_path)

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]

    # Rename columns to standard names
    rename_map = {}
    if 'adjclose' in df.columns and 'close' not in df.columns:
        rename_map['adjclose'] = 'close'
    if 'atr14' in df.columns and 'atr' not in df.columns:
        rename_map['atr14'] = 'atr'

    if rename_map:
        df = df.rename(columns=rename_map)

    # Validate required columns
    required = {'symbol', 'date', 'close', 'high', 'low', 'atr'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Select only required columns
    df = df[['symbol', 'date', 'close', 'high', 'low', 'atr']].copy()

    # Convert date
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    # Apply date filters
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]

    # Use centralized validation (handles price filtering, NaN removal, etc.)
    if validator is None:
        validator = TargetDataValidator()

    df, summary = validator.validate_input(df)

    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

    return df


def generate_targets_with_config(
    df: pd.DataFrame,
    target_configs: Dict[str, Dict[str, Any]],
    model_keys: Optional[List[str]] = None,
    n_jobs: int = -1,
    weight_min_clip: float = 0.01,
    weight_max_clip: float = 10.0,
) -> pd.DataFrame:
    """
    Generate multi-target labels using custom config.

    This is a wrapper around generate_multi_targets_parallel that allows
    using custom barrier configurations instead of the defaults.

    Args:
        df: DataFrame with symbol, date, close, high, low, atr
        target_configs: Dictionary mapping model_key string -> config dict
        model_keys: List of model keys to generate (default: all in config)
        n_jobs: Number of parallel workers
        weight_min_clip: Minimum weight value
        weight_max_clip: Maximum weight value

    Returns:
        DataFrame with multi-target columns
    """
    # Filter model keys if specified
    if model_keys:
        target_configs = {k: v for k, v in target_configs.items() if k in model_keys}

    if not target_configs:
        raise ValueError("No valid model keys specified")

    logger.info(f"Generating targets for {len(target_configs)} models: {list(target_configs.keys())}")
    for model_key, config in target_configs.items():
        logger.info(f"  {model_key}: up_mult={config['up_mult']}, dn_mult={config['dn_mult']}")

    # Temporarily override TARGET_CONFIGS for generation
    # This is a bit hacky but avoids modifying the target_generation module
    from src.config import model_keys as mk_module

    original_configs = mk_module.TARGET_CONFIGS.copy()

    try:
        # Update TARGET_CONFIGS with custom values
        for model_key_str, config in target_configs.items():
            try:
                model_key = ModelKey(model_key_str)
                mk_module.TARGET_CONFIGS[model_key] = config
            except ValueError:
                logger.warning(f"Unknown model key: {model_key_str}, skipping")

        # Convert model key strings to ModelKey enums
        model_key_enums = []
        for key_str in target_configs.keys():
            try:
                model_key_enums.append(ModelKey(key_str))
            except ValueError:
                pass

        if not model_key_enums:
            raise ValueError("No valid ModelKey enums found")

        # Generate targets
        targets_df = generate_multi_targets_parallel(
            df,
            model_keys=model_key_enums,
            n_jobs=n_jobs,
            weight_min_clip=weight_min_clip,
            weight_max_clip=weight_max_clip,
        )

        return targets_df

    finally:
        # Restore original configs
        mk_module.TARGET_CONFIGS = original_configs


def print_target_summary(targets_df: pd.DataFrame, model_keys: List[str]) -> None:
    """Print summary statistics for generated targets."""
    print("\n" + "=" * 60)
    print("TARGET GENERATION SUMMARY")
    print("=" * 60)

    print(f"\nTotal trajectories: {len(targets_df):,}")
    print(f"Symbols: {targets_df['symbol'].nunique():,}")
    print(f"Date range: {targets_df['t0'].min()} to {targets_df['t0'].max()}")

    print("\nClass distributions:")
    print("-" * 40)

    for model_key in model_keys:
        hit_col = f'hit_{model_key}'
        if hit_col in targets_df.columns:
            counts = targets_df[hit_col].value_counts().sort_index()
            total = len(targets_df)
            print(f"\n  {model_key}:")
            print(f"    +1 (target hit):  {counts.get(1, 0):>8,} ({100*counts.get(1, 0)/total:>5.1f}%)")
            print(f"     0 (time expiry): {counts.get(0, 0):>8,} ({100*counts.get(0, 0)/total:>5.1f}%)")
            print(f"    -1 (stop hit):    {counts.get(-1, 0):>8,} ({100*counts.get(-1, 0)/total:>5.1f}%)")

    if 'weight_final' in targets_df.columns:
        print("\nWeight statistics:")
        print(f"  min: {targets_df['weight_final'].min():.4f}")
        print(f"  max: {targets_df['weight_final'].max():.4f}")
        print(f"  sum: {targets_df['weight_final'].sum():.2f}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Recompute triple barrier targets without running full feature pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Recompute with default configs
    python scripts/recompute_targets.py

    # Use calibrated thresholds
    python scripts/recompute_targets.py --target-config artifacts/targets/barrier_calibration.json

    # Only specific models
    python scripts/recompute_targets.py --model-keys long_normal,long_parabolic

    # Custom date range
    python scripts/recompute_targets.py --start 2020-01-01 --end 2024-12-31
        """
    )

    parser.add_argument('--input', type=str, default='artifacts/features_complete.parquet',
                       help='Input parquet file with OHLCV + ATR data')
    parser.add_argument('--output', type=str, default='artifacts/targets/targets.parquet',
                       help='Output parquet file for targets')
    parser.add_argument('--target-config', type=str, default=None,
                       help='Path to barrier_calibration.json (default: use built-in TARGET_CONFIGS)')
    parser.add_argument('--model-keys', type=str, default=None,
                       help='Comma-separated list of model keys to generate (default: all)')
    parser.add_argument('--start', type=str, default=None,
                       help='Start date filter (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                       help='End date filter (YYYY-MM-DD)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of parallel workers (-1 for all cores)')
    parser.add_argument('--weight-min-clip', type=float, default=0.01,
                       help='Minimum weight value')
    parser.add_argument('--weight-max-clip', type=float, default=10.0,
                       help='Maximum weight value')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load input data
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    df = load_minimal_data(input_path, args.start, args.end)

    if len(df) == 0:
        logger.error("No data available for target generation")
        sys.exit(1)

    # Determine target configs
    if args.target_config:
        config_path = Path(args.target_config)
        if not config_path.exists():
            logger.error(f"Target config file not found: {config_path}")
            sys.exit(1)
        target_configs = load_calibration_config(config_path)
    else:
        # Use default TARGET_CONFIGS
        target_configs = {k.value: v.copy() for k, v in TARGET_CONFIGS.items()}
        logger.info("Using default TARGET_CONFIGS from src/config/model_keys.py")

    # Filter model keys if specified
    model_keys = None
    if args.model_keys:
        model_keys = [k.strip() for k in args.model_keys.split(',')]
        invalid_keys = [k for k in model_keys if k not in target_configs]
        if invalid_keys:
            logger.warning(f"Unknown model keys (ignored): {invalid_keys}")
        model_keys = [k for k in model_keys if k in target_configs]
        if not model_keys:
            logger.error("No valid model keys specified")
            sys.exit(1)
    else:
        model_keys = list(target_configs.keys())

    # Generate targets
    targets_df = generate_targets_with_config(
        df,
        target_configs,
        model_keys=model_keys,
        n_jobs=args.n_jobs,
        weight_min_clip=args.weight_min_clip,
        weight_max_clip=args.weight_max_clip,
    )

    if targets_df.empty:
        logger.error("No targets generated")
        sys.exit(1)

    # Print summary
    print_target_summary(targets_df, model_keys)

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    targets_df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(targets_df):,} targets to {output_path}")

    # Also save to standard location if using non-default output
    standard_path = Path('artifacts/targets_triple_barrier.parquet')
    if output_path != standard_path:
        print(f"\nNote: To use these targets with training, either:")
        print(f"  1. Copy to: {standard_path}")
        print(f"  2. Or update your training script to use: {output_path}")


if __name__ == '__main__':
    main()
