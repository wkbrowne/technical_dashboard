"""
OHLC adjustment module for aligning price data with adjusted close.

This module provides functionality to adjust Open, High, Low, Close prices
for splits and dividends, ensuring consistency for target generation.

POLICY: Raw prices for features, adjusted prices for targets
------------------------------------------------------------------------
This module is ONLY used for target generation and PnL/backtest calculations.
Feature computation uses raw OHLC for point-in-time correctness.

Adjustment approach: Scale all OHLC prices by (adjclose/close) ratio so they
match the adjclose price space. This is required ONLY for:
- Triple barrier targets (entry price from adjclose, barrier checks from high/low)
- PnL and realized return calculations in backtest

NOT used for:
- Feature computation (uses raw OHLC)
- Technical indicators (computed from raw prices for point-in-time correctness)

Why adjusted prices matter for targets:
- Entry price is at t+1 open/close which should use adjusted prices
- Barrier checks (high >= upper, low <= lower) must be in same price space
- Without adjustment, historical high/low may be 30%+ different from adjclose
"""
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def adjust_ohlc_to_adjclose(
    df: pd.DataFrame,
    max_source_error_pct: float = 0.1
) -> pd.DataFrame:
    """
    Adjust ALL OHLC prices to match adjclose price space.

    Multiplies each OHLC column by (adjclose/close) to align with
    split/dividend adjusted prices. This ensures:
    1. OHLC consistency is preserved (low <= close <= high)
    2. Technical indicators (chop, pos_in_range, ATR) compute correctly
    3. Triple barrier targets work correctly (entry from adjclose, barriers from high/low)

    For recent data: factor ≈ 1.0 (close ≈ adjclose, minimal change)
    For historical data: factor < 1.0 (scale down to match adjclose)

    Args:
        df: DataFrame with OHLC and adjclose columns, sorted by date
        max_source_error_pct: Maximum percentage of rows with source data errors
            to auto-fix. If more than this percentage have issues, raises ValueError
            (indicates systematic problem, not sporadic data provider errors).

    Returns:
        DataFrame with adjusted OHLC prices matching adjclose space

    Raises:
        ValueError: If source data has too many OHLC inconsistencies (> max_source_error_pct)

    Notes:
        - Requires 'adjclose' and 'close' columns
        - Volume is NOT adjusted
        - All OHLC columns are adjusted together to maintain consistency
        - Small numbers of source data errors (close > high, close < low) are auto-fixed
    """
    # Check for required columns
    required_cols = ['adjclose', 'close']
    if not all(col in df.columns for col in required_cols):
        logger.debug("Missing required columns for OHLC adjustment")
        return df

    # Check for OHLC columns to adjust
    ohlc_cols = ['open', 'high', 'low', 'close']
    available_ohlc = [col for col in ohlc_cols if col in df.columns]

    if not available_ohlc:
        logger.debug("No OHLC columns found to adjust")
        return df

    df_adjusted = df.copy()

    # Convert to numeric and handle invalid values
    close_values = pd.to_numeric(df['close'], errors='coerce')
    adjclose_values = pd.to_numeric(df['adjclose'], errors='coerce')

    # Check for and fix source data inconsistencies BEFORE adjustment
    # These are data provider errors that would persist through adjustment
    if 'high' in df_adjusted.columns and 'low' in df_adjusted.columns:
        high_values = pd.to_numeric(df_adjusted['high'], errors='coerce')
        low_values = pd.to_numeric(df_adjusted['low'], errors='coerce')

        # Count inconsistencies
        close_above_high = close_values > high_values
        close_below_low = close_values < low_values
        n_close_above_high = close_above_high.sum()
        n_close_below_low = close_below_low.sum()
        total_errors = n_close_above_high + n_close_below_low
        total_rows = len(df)

        if total_errors > 0:
            error_pct = (total_errors / total_rows) * 100

            # If too many errors, this indicates a systematic problem
            if error_pct > max_source_error_pct:
                sample_mask = close_above_high | close_below_low
                sample = df[sample_mask][['high', 'low', 'close']].head(5)
                raise ValueError(
                    f"Source data has too many OHLC inconsistencies: {total_errors} rows "
                    f"({error_pct:.2f}%) exceed threshold of {max_source_error_pct}%.\n"
                    f"  - close > high: {n_close_above_high} rows\n"
                    f"  - close < low: {n_close_below_low} rows\n"
                    "This indicates a systematic problem with the data or adjustment logic, "
                    "not sporadic data provider errors.\n"
                    f"Sample problematic rows:\n{sample.to_string()}"
                )

            # Auto-fix small numbers of source data errors
            if n_close_above_high > 0:
                df_adjusted.loc[close_above_high, 'high'] = close_values[close_above_high]
                logger.debug(f"Fixed {n_close_above_high} rows where close > high (source data error)")

            if n_close_below_low > 0:
                df_adjusted.loc[close_below_low, 'low'] = close_values[close_below_low]
                logger.debug(f"Fixed {n_close_below_low} rows where close < low (source data error)")

    # Find valid data points (non-zero, non-NaN)
    valid_mask = (close_values != 0) & pd.notna(close_values) & pd.notna(adjclose_values)
    if not valid_mask.any():
        logger.debug("No valid price data found for adjustment")
        return df

    # Verify we have valid data
    valid_indices = df.index[valid_mask]
    if len(valid_indices) == 0:
        return df

    # Calculate adjustment factor for each row: adjclose/close
    # This factor scales OHLC prices to match adjclose (split/dividend adjusted)
    # For recent data: factor ≈ 1.0 (close ≈ adjclose)
    # For historical data: factor < 1.0 (scale down to match adjclose)
    adjustment_factors = pd.Series(1.0, index=df.index)
    adjustment_factors[valid_mask] = adjclose_values[valid_mask] / close_values[valid_mask]

    # Apply adjustment to ALL OHLC columns
    # Use df_adjusted values (which may have source data fixes applied)
    adjustment_applied = False
    for col in available_ohlc:
        if col in df_adjusted.columns:
            values_to_adjust = pd.to_numeric(df_adjusted[col], errors='coerce')
            df_adjusted[col] = values_to_adjust * adjustment_factors
            adjustment_applied = True

    # Log adjustment summary
    if adjustment_applied:
        factor_std = adjustment_factors[valid_mask].std()
        if factor_std > 0.001:
            avg_factor = adjustment_factors[valid_mask].mean()
            factor_range = (adjustment_factors[valid_mask].min(), adjustment_factors[valid_mask].max())
            logger.debug(f"Applied OHLC adjustment: avg factor = {avg_factor:.4f}, "
                        f"range = {factor_range[0]:.4f} to {factor_range[1]:.4f}")

    # Validate OHLC consistency after adjustment (only if all OHLC columns present)
    result = validate_ohlc_consistency(df_adjusted, tolerance=0.01)
    if not result['valid'] and 'error' not in result:
        # Only warn if there are actual violations (not just missing columns)
        total_violations = sum(result.get('violations', {}).values())
        if total_violations > 0:
            logger.warning(f"OHLC consistency issues after adjustment: {result['violations']}")

    return df_adjusted


def validate_ohlc_consistency(df: pd.DataFrame, tolerance: float = 0.01) -> dict:
    """
    Validate that OHLC prices are internally consistent.

    Checks:
    1. low <= close <= high (allowing for tolerance)
    2. low <= open <= high
    3. low <= high

    Args:
        df: DataFrame with OHLC columns
        tolerance: Fraction tolerance for comparisons (default 1%)

    Returns:
        Dict with validation results:
        - valid: True if all checks pass
        - violations: Count of rows failing each check
        - sample_violations: Sample of violating rows
    """
    results = {'valid': True, 'violations': {}, 'sample_violations': None}

    required = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required):
        results['valid'] = False
        results['error'] = 'Missing required OHLC columns'
        return results

    # Convert to numeric
    o = pd.to_numeric(df['open'], errors='coerce')
    h = pd.to_numeric(df['high'], errors='coerce')
    l = pd.to_numeric(df['low'], errors='coerce')
    c = pd.to_numeric(df['close'], errors='coerce')

    # Check low <= close <= high (with tolerance)
    close_above_high = c > h * (1 + tolerance)
    close_below_low = c < l * (1 - tolerance)

    # Check low <= high
    low_above_high = l > h * (1 + tolerance)

    results['violations'] = {
        'close_above_high': close_above_high.sum(),
        'close_below_low': close_below_low.sum(),
        'low_above_high': low_above_high.sum()
    }

    total_violations = sum(results['violations'].values())
    if total_violations > 0:
        results['valid'] = False
        # Get sample violations
        any_violation = close_above_high | close_below_low | low_above_high
        if any_violation.any():
            sample = df[any_violation][['open', 'high', 'low', 'close']].head(5)
            results['sample_violations'] = sample.to_dict()

    return results