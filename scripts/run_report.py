#!/usr/bin/env python
"""
Generate tear sheet and reports from backtest results.

Usage:
    python scripts/run_report.py [--backtest-dir artifacts/backtests] [--output-dir artifacts/reports]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd

from src.alpha.data.schemas import BacktestResult, WeeklySnapshot
from src.alpha.backtest.metrics import compute_performance_metrics, compute_monthly_returns
from src.alpha.reporting import generate_tearsheet, save_tearsheet


def load_backtest_result(backtest_dir: Path) -> BacktestResult:
    """Load backtest result from saved files."""
    # Load equity curve
    equity_path = backtest_dir / "backtest_equity_curve.csv"
    if not equity_path.exists():
        # Try finding any equity curve file
        equity_files = list(backtest_dir.glob("*_equity_curve.csv"))
        if equity_files:
            equity_path = equity_files[0]
        else:
            raise FileNotFoundError(f"No equity curve found in {backtest_dir}")

    equity_curve = pd.read_csv(equity_path)

    # Load metrics
    metrics_path = backtest_dir / "backtest_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        # Recompute metrics
        metrics = compute_performance_metrics(equity_curve).to_dict()

    # Load config
    config_path = backtest_dir / "backtest_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}

    return BacktestResult(
        config=config,
        snapshots=[],  # Not loading full snapshots
        all_trades=[],
        metrics=metrics,
        equity_curve=equity_curve,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate backtest reports")
    parser.add_argument("--backtest-dir", type=str, default="artifacts/backtests",
                        help="Directory containing backtest results")
    parser.add_argument("--output-dir", type=str, default="artifacts/reports",
                        help="Output directory for reports")
    parser.add_argument("--title", type=str, default="Backtest Report",
                        help="Report title")

    args = parser.parse_args()
    backtest_dir = Path(args.backtest_dir)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("Report Generation")
    print("=" * 60)
    print()

    if not backtest_dir.exists():
        print(f"Error: Backtest directory not found: {backtest_dir}")
        print("Run backtest first: python scripts/run_backtest.py")
        sys.exit(1)

    # Load backtest result
    print(f"Loading backtest from: {backtest_dir}")
    try:
        result = load_backtest_result(backtest_dir)
        print(f"  Loaded equity curve: {len(result.equity_curve)} rows")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Generate reports
    print("\nGenerating reports...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full tearsheet
    outputs = save_tearsheet(result, output_dir, name="report")

    # Monthly returns table
    try:
        monthly = compute_monthly_returns(result.equity_curve)
        monthly_path = output_dir / "monthly_returns.csv"
        monthly.to_csv(monthly_path)
        print(f"  Monthly returns: {monthly_path}")

        # Print monthly returns
        print("\nMonthly Returns:")
        print(monthly.to_string())
    except Exception as e:
        print(f"  Could not compute monthly returns: {e}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)

    metrics = result.metrics
    print(f"\nTotal Return:    {metrics.get('total_return', 0):>10.2%}")
    print(f"CAGR:            {metrics.get('cagr', 0):>10.2%}")
    print(f"Volatility:      {metrics.get('volatility', 0):>10.2%}")
    print(f"Sharpe Ratio:    {metrics.get('sharpe_ratio', 0):>10.2f}")
    print(f"Max Drawdown:    {metrics.get('max_drawdown', 0):>10.2%}")
    print(f"Calmar Ratio:    {metrics.get('calmar_ratio', 0):>10.2f}")

    print("\n" + "=" * 60)
    print("Reports generated!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
