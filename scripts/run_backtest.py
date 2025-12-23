#!/usr/bin/env python
"""
Run backtest with position sizing.

This script runs a complete backtest with the configured sizing rules,
comparing against baseline strategies.

Usage:
    python scripts/run_backtest.py [--config config.yaml] [--compare]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.alpha.config import AlphaConfig, load_config
from src.alpha.data.loaders import load_targets, load_features, prepare_weekly_signals
from src.alpha.sizing.rules import create_sizing_rule
from src.alpha.backtest import run_backtest, compare_strategies
from src.alpha.backtest.metrics import compute_performance_metrics
from src.alpha.reporting import save_tearsheet


def main():
    parser = argparse.ArgumentParser(description="Run backtest with sizing")
    parser.add_argument("--config", type=str, default="src/alpha/config/default.yaml",
                        help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default="artifacts/backtests",
                        help="Output directory for backtest results")
    parser.add_argument("--compare", action="store_true",
                        help="Compare against baseline strategies")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None,
                        help="Backtest end date (YYYY-MM-DD)")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("Position Sizing Backtest")
    print("=" * 60)
    print()

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        print(f"Loading config from: {config_path}")
        config = load_config(config_path)
    else:
        print("Using default configuration")
        config = AlphaConfig()

    print(f"  Sizing method: {config.sizing.method.value}")
    print(f"  Max exposure: {config.sizing.max_gross_exposure:.0%}")
    print(f"  Max weight: {config.sizing.max_name_weight:.0%}")
    print(f"  Top N: {config.sizing.top_n}")

    # Load data
    print("\nLoading data...")
    targets = load_targets(config.targets_file, exclude_neutral=True)
    print(f"  Targets: {len(targets):,} samples")

    features = load_features(config.features_file)
    print(f"  Features: {features.shape}")

    # Load predictions - prefer CV predictions for backtesting
    cv_predictions_file = Path("artifacts/predictions/cv_predictions.parquet")
    legacy_predictions_file = Path(config.predictions_file)

    if cv_predictions_file.exists():
        predictions = pd.read_parquet(cv_predictions_file)
        print(f"  CV Predictions: {len(predictions):,} rows (out-of-sample)")
        # Rename columns if needed
        if "probability" not in predictions.columns and "probability_raw" in predictions.columns:
            predictions["probability"] = predictions["probability_raw"]
    elif legacy_predictions_file.exists():
        predictions = pd.read_parquet(legacy_predictions_file)
        print(f"  Legacy Predictions: {len(predictions):,} rows")
        print("  WARNING: Legacy predictions may contain in-sample data!")
    else:
        # Use hit as proxy for probability (for testing)
        predictions = pd.DataFrame({
            "symbol": targets["symbol"],
            "date": targets["entry_date"],
            "probability": 0.5 + (targets["hit"] == 1).astype(float) * 0.3,
        })
        print("  WARNING: Using placeholder predictions (information leakage!)")

    # Prepare weekly signals
    print("\nPreparing weekly signals...")
    signals = prepare_weekly_signals(
        predictions=predictions,
        targets=targets,
        features=features,
    )
    print(f"  Signals: {len(signals):,}")

    # Apply date filters
    if args.start_date:
        signals = signals[signals["week_monday"] >= pd.to_datetime(args.start_date)]
    if args.end_date:
        signals = signals[signals["week_monday"] <= pd.to_datetime(args.end_date)]

    print(f"  Date range: {signals['week_monday'].min().date()} to {signals['week_monday'].max().date()}")

    if args.compare:
        # Run comparison against baselines
        print("\nRunning strategy comparison...")
        results = compare_strategies(signals, config)

        # Print comparison
        print("\n" + "=" * 80)
        print("STRATEGY COMPARISON")
        print("=" * 80)

        for name, result in results.items():
            print(f"\n--- {name.upper()} ---")
            if hasattr(result, 'metrics') and isinstance(result.metrics, dict):
                metrics = result.metrics
                print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
                print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")

        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save individual tearsheets
        for name, result in results.items():
            save_tearsheet(result, output_dir, name=name)

    else:
        # Run single backtest
        print("\nRunning backtest...")
        result = run_backtest(signals, config)

        # Calculate backtest duration for annualization
        start_date = signals['week_monday'].min()
        end_date = signals['week_monday'].max()
        n_weeks = len(signals['week_monday'].unique())
        years = n_weeks / 52.0

        # Print metrics
        if hasattr(result.metrics, '__getitem__'):
            print("\n" + "=" * 60)
            print("PERFORMANCE METRICS")
            print("=" * 60)
            metrics = result.metrics

            total_return = metrics.get('total_return', 0)
            volatility = metrics.get('volatility', 0)
            max_dd = metrics.get('max_drawdown', 0)

            # Annualized metrics
            cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
            ann_volatility = volatility * (52 ** 0.5)  # Weekly to annual
            ann_sharpe = (cagr / ann_volatility) if ann_volatility > 0 else 0

            print(f"\nBacktest Period: {start_date.date()} to {end_date.date()}")
            print(f"Duration: {n_weeks} weeks ({years:.2f} years)")

            print(f"\n--- RAW RETURNS (over {years:.2f} years) ---")
            print(f"  Total Return:    {total_return:>10.2%}")
            print(f"  Best Week:       {metrics.get('best_week', 0):>10.2%}")
            print(f"  Worst Week:      {metrics.get('worst_week', 0):>10.2%}")

            print(f"\n--- ANNUALIZED METRICS ---")
            print(f"  CAGR:            {cagr:>10.2%}")
            print(f"  Volatility:      {ann_volatility:>10.2%}")
            print(f"  Sharpe Ratio:    {ann_sharpe:>10.2f}")
            print(f"  Max Drawdown:    {max_dd:>10.2%}")
            print(f"  Calmar Ratio:    {(cagr / abs(max_dd)) if max_dd != 0 else 0:>10.2f}")

            print(f"\n--- TRADING STATISTICS ---")
            print(f"  Hit Rate:        {metrics.get('hit_rate', 0):>10.2%}")
            print(f"  Profit Factor:   {metrics.get('profit_factor', 0):>10.2f}")
            print(f"  Avg Turnover:    {metrics.get('avg_turnover', 0):>10.2%}")
            print(f"  Sortino Ratio:   {metrics.get('sortino_ratio', 0):>10.2f}")

            # Add annualized metrics to result for saving
            metrics['cagr'] = cagr
            metrics['ann_volatility'] = ann_volatility
            metrics['ann_sharpe'] = ann_sharpe
            metrics['backtest_years'] = years
            metrics['n_weeks'] = n_weeks

        # Save tearsheet
        output_dir.mkdir(parents=True, exist_ok=True)
        save_tearsheet(result, output_dir, name="backtest")

    print("\n" + "=" * 60)
    print("Backtest complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
