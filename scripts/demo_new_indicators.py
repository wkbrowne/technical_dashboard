#!/usr/bin/env python
"""
Demo script for the new indicators: CHOP, ADX, Bollinger Bandwidth, Squeeze, Gap/Overnight.

This script:
1. Loads sample data (from cache or generates synthetic)
2. Computes all new indicator features
3. Prints the last 5 rows for inspection

Usage:
    python scripts/demo_new_indicators.py [SYMBOL]

    SYMBOL: Optional stock symbol (default: AAPL)
            If cache exists, loads real data; otherwise generates synthetic.

Examples:
    python scripts/demo_new_indicators.py
    python scripts/demo_new_indicators.py MSFT
"""
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_days: int = 252) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)

    dates = pd.date_range(end=pd.Timestamp.now().normalize(), periods=n_days, freq='B')

    # Generate realistic price path
    returns = np.random.randn(n_days) * 0.015 + 0.0003  # ~1.5% daily vol, small drift
    log_prices = np.cumsum(returns)
    close = 150 * np.exp(log_prices)

    # Generate OHLC from close
    daily_range_pct = np.random.uniform(0.005, 0.025, n_days)
    high = close * (1 + daily_range_pct * np.random.uniform(0.3, 1.0, n_days))
    low = close * (1 - daily_range_pct * np.random.uniform(0.3, 1.0, n_days))
    open_price = low + (high - low) * np.random.uniform(0.1, 0.9, n_days)

    # Generate volume
    volume = np.random.lognormal(mean=16, sigma=0.5, size=n_days).astype(int)

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'adjclose': close,
        'volume': volume,
    }, index=dates)

    return df


def load_data(symbol: str = 'AAPL') -> pd.DataFrame:
    """Load data from cache or generate synthetic."""
    cache_path = project_root / 'cache' / 'stock_data_combined.parquet'

    if cache_path.exists():
        try:
            logger.info(f"Loading {symbol} from cache...")
            df = pd.read_parquet(cache_path)

            # Filter to symbol
            if 'symbol' in df.columns:
                df = df[df['symbol'] == symbol].copy()
                df = df.set_index('date')
            else:
                # Wide format - need different handling
                logger.warning("Cache is wide format, using synthetic data")
                df = generate_synthetic_data()

            if len(df) < 60:
                logger.warning(f"Insufficient data for {symbol}, using synthetic")
                df = generate_synthetic_data()

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}, using synthetic data")
            df = generate_synthetic_data()
    else:
        logger.info("No cache found, using synthetic data")
        df = generate_synthetic_data()

    # Ensure lowercase columns
    df.columns = [c.lower() for c in df.columns]

    return df


def demo_new_indicators(symbol: str = 'AAPL') -> None:
    """Demonstrate the new indicators."""
    print("=" * 80)
    print(f"NEW INDICATORS DEMO - {symbol}")
    print("=" * 80)

    # Load data
    df = load_data(symbol)
    print(f"\nLoaded {len(df)} rows of data")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Import feature functions
    from src.features.trend import add_chop_features, add_adx_features
    from src.features.volatility import add_bollinger_bandwidth_features, add_squeeze_features
    from src.features.range_breakout import add_range_breakout_features

    # Compute features
    print("\nComputing features...")

    # 1. CHOP
    df = add_chop_features(df, length=14)
    print("  - CHOP (chop_14)")

    # 2. ADX
    df = add_adx_features(df, length=14)
    print("  - ADX (adx_14, di_plus_14, di_minus_14)")

    # 3. Bollinger Bandwidth
    df = add_bollinger_bandwidth_features(df, length=20, std=2.0, z_window=60)
    print("  - Bollinger Bandwidth (bb_width_20_2, bb_width_20_2_z60)")

    # 4. Squeeze
    df = add_squeeze_features(df, length=20)
    print("  - Squeeze (squeeze_on_20, squeeze_intensity_20, days_in_squeeze_20, ...)")

    # 5. Range/Breakout (includes gap features)
    df = add_range_breakout_features(df)
    print("  - Gap/Overnight (overnight_ret, gap_atr_ratio_raw, gap_fill_frac, atr_percent_chg_5)")

    # Display results
    print("\n" + "=" * 80)
    print("LAST 5 ROWS OF NEW INDICATORS")
    print("=" * 80)

    # Group 1: Trend Quality
    print("\n1. TREND QUALITY / CHOP FILTER:")
    trend_cols = ['chop_14', 'adx_14', 'di_plus_14', 'di_minus_14']
    print(df[trend_cols].tail())

    # Interpretation
    last_chop = df['chop_14'].iloc[-1]
    last_adx = df['adx_14'].iloc[-1]
    print(f"\n   Interpretation:")
    if pd.notna(last_chop):
        if last_chop > 61.8:
            print(f"   - CHOP {last_chop:.1f} > 61.8: Choppy/sideways market")
        elif last_chop < 38.2:
            print(f"   - CHOP {last_chop:.1f} < 38.2: Strong trending market")
        else:
            print(f"   - CHOP {last_chop:.1f}: Neutral")
    if pd.notna(last_adx):
        if last_adx < 20:
            print(f"   - ADX {last_adx:.1f} < 20: Weak trend")
        elif last_adx > 40:
            print(f"   - ADX {last_adx:.1f} > 40: Strong trend")
        else:
            print(f"   - ADX {last_adx:.1f}: Developing trend")

    # Group 2: Volatility Compression
    print("\n2. VOLATILITY COMPRESSION / SQUEEZE:")
    vol_cols = ['bb_width_20_2', 'bb_width_20_2_z60', 'squeeze_on_20', 'squeeze_intensity_20']
    print(df[vol_cols].tail())

    # Interpretation
    last_squeeze = df['squeeze_on_20'].iloc[-1]
    last_days = df['days_in_squeeze_20'].iloc[-1]
    if pd.notna(last_squeeze):
        if last_squeeze == 1:
            print(f"\n   Interpretation: SQUEEZE ON (BB inside KC) - {int(last_days)} consecutive days")
            print("   -> Volatility compression detected, breakout may be imminent")
        else:
            print(f"\n   Interpretation: No squeeze (normal volatility)")

    # Group 3: Squeeze Details
    print("\n   SQUEEZE DETAILS:")
    squeeze_cols = ['squeeze_on_wide_20', 'squeeze_release_20', 'days_in_squeeze_20']
    print(df[squeeze_cols].tail())

    # Group 4: Gap/Overnight
    print("\n3. GAP / OVERNIGHT FEATURES:")
    gap_cols = ['overnight_ret', 'gap_atr_ratio_raw', 'gap_fill_frac', 'atr_percent_chg_5']
    print(df[gap_cols].tail())

    # Interpretation
    last_overnight = df['overnight_ret'].iloc[-1]
    last_gap_atr = df['gap_atr_ratio_raw'].iloc[-1]
    if pd.notna(last_overnight):
        print(f"\n   Interpretation:")
        print(f"   - Last overnight return: {last_overnight*100:.2f}%")
        if pd.notna(last_gap_atr):
            print(f"   - Gap size: {last_gap_atr:.2f} ATRs")

    # Summary stats
    print("\n" + "=" * 80)
    print("FEATURE STATISTICS")
    print("=" * 80)

    all_new_features = [
        'chop_14', 'adx_14', 'di_plus_14', 'di_minus_14',
        'bb_width_20_2', 'bb_width_20_2_z60',
        'squeeze_on_20', 'squeeze_on_wide_20', 'squeeze_intensity_20',
        'squeeze_release_20', 'days_in_squeeze_20',
        'overnight_ret', 'gap_atr_ratio_raw', 'gap_fill_frac', 'atr_percent_chg_5'
    ]

    stats = []
    for feat in all_new_features:
        if feat in df.columns:
            col = df[feat]
            nan_pct = col.isna().mean() * 100
            non_nan = col.dropna()
            if len(non_nan) > 0:
                stats.append({
                    'Feature': feat,
                    'NaN%': f"{nan_pct:.1f}%",
                    'Min': f"{non_nan.min():.3f}",
                    'Max': f"{non_nan.max():.3f}",
                    'Mean': f"{non_nan.mean():.3f}",
                })

    stats_df = pd.DataFrame(stats)
    print(stats_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'
    demo_new_indicators(symbol)
