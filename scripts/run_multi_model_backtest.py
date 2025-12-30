#!/usr/bin/env python
"""
Run backtest with multi-model position sizing.

This script runs a complete backtest using the 4-model sizing system:
- LONG_NORMAL, LONG_PARABOLIC, SHORT_NORMAL, SHORT_PARABOLIC

Features:
- Multi-model prediction loading
- Configurable combining policy
- Direction-aware regime gating
- Decision log output for dashboard display
- Detailed diagnostics (longs vs shorts, model contributions, gating effects)
- Sizing diagnostics with warnings (data integrity, drift, concentration, etc.)

Usage:
    # Basic multi-model backtest
    python scripts/run_multi_model_backtest.py \\
        --prediction-path artifacts/predictions/cv_predictions_multi.parquet \\
        --sizing-config artifacts/sizing/best_config_multi_model.json

    # With regime gating
    python scripts/run_multi_model_backtest.py \\
        --prediction-path artifacts/predictions/ \\
        --sizing-config artifacts/sizing/best_config_multi_model.json \\
        --regime-gating on

    # Export decision log for dashboard
    python scripts/run_multi_model_backtest.py \\
        --prediction-path artifacts/predictions/ \\
        --sizing-config artifacts/sizing/best_config_multi_model.json \\
        --decision-log artifacts/backtests/decision_log.parquet

    # With diagnostics
    python scripts/run_multi_model_backtest.py \\
        --prediction-path artifacts/predictions/cv_predictions_multi.parquet \\
        --sizing-config artifacts/sizing/best_config_multi_model.json \\
        --diagnostics on

    # Compute diagnostic baselines from historical data
    python scripts/run_multi_model_backtest.py \\
        --prediction-path artifacts/predictions/cv_predictions_multi.parquet \\
        --compute-diagnostic-baselines

    # Single model backward-compatible mode
    python scripts/run_multi_model_backtest.py \\
        --models LONG_NORMAL \\
        --prediction-path artifacts/predictions/cv_predictions.parquet
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.sizing import (
    ModelType,
    MultiModelSizingConfig,
    MultiModelSizingEngine,
    load_multi_model_config,
    load_multi_model_predictions,
)
from src.sizing.predictions import merge_predictions_with_targets, get_prob_column
from src.sizing.regime_gating import get_gating_diagnostics
from src.sizing.regime_data import load_regime_features as load_regime_features_util
from src.sizing.multi_model import create_decision_log


def parse_models(models_str: str) -> List[ModelType]:
    """Parse comma-separated model names."""
    if not models_str:
        return list(ModelType)

    model_names = [m.strip().upper() for m in models_str.split(",")]
    models = []
    for name in model_names:
        try:
            models.append(ModelType(name.lower()))
        except ValueError:
            print(f"Warning: Unknown model '{name}', skipping")
    return models or list(ModelType)


def load_targets(path: str = "artifacts/targets_triple_barrier.parquet") -> pd.DataFrame:
    """Load targets file."""
    df = pd.read_parquet(path)

    # Standardize column names
    if "t0" in df.columns:
        df = df.rename(columns={"t0": "entry_date"})
    if "t_hit" in df.columns:
        df = df.rename(columns={"t_hit": "exit_date"})

    df["entry_date"] = pd.to_datetime(df["entry_date"])

    # Exclude neutral if present
    if "hit" in df.columns:
        df = df[df["hit"] != 0].copy()

    return df


def load_regime_features(
    features_path: str = "artifacts/features_complete.parquet",
) -> pd.DataFrame:
    """Load regime-relevant features using the regime_data utility.

    This function leverages the new regime_data module which:
    - Automatically finds the correct column names for regime features
    - Handles market-level vs stock-level regime data
    - Groups by date for unique date-level features
    """
    try:
        return load_regime_features_util(features_path)
    except FileNotFoundError:
        print(f"Warning: Features file not found at {features_path}")
        print("Continuing without regime features...")
        return pd.DataFrame()


def compute_weekly_metrics(signals: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics per week."""
    if "week_monday" not in signals.columns:
        return pd.DataFrame()

    weekly = signals.groupby("week_monday").agg({
        "final_weight": lambda x: x.abs().sum(),  # gross exposure
        "actual_return": lambda x: (x * signals.loc[x.index, "final_weight"]).sum(),
        "symbol": "count",
    }).rename(columns={
        "final_weight": "gross_exposure",
        "actual_return": "portfolio_return",
        "symbol": "n_signals",
    })

    # Add position counts
    for week in weekly.index:
        week_data = signals[signals["week_monday"] == week]
        weekly.loc[week, "n_longs"] = (week_data["final_weight"] > 0).sum()
        weekly.loc[week, "n_shorts"] = (week_data["final_weight"] < 0).sum()
        weekly.loc[week, "n_positions"] = weekly.loc[week, "n_longs"] + weekly.loc[week, "n_shorts"]

    # Add gating info if available
    if "gating_multiplier" in signals.columns:
        for week in weekly.index:
            week_data = signals[signals["week_monday"] == week]
            weekly.loc[week, "gating_multiplier"] = week_data["gating_multiplier"].iloc[0]
            weekly.loc[week, "gating_triggered"] = week_data["gating_triggered"].iloc[0]

    # Cumulative return
    weekly["cumulative_return"] = (1 + weekly["portfolio_return"]).cumprod() - 1

    return weekly.reset_index()


def compute_model_contributions(signals: pd.DataFrame) -> Dict:
    """Compute how often each model contributed to positions."""
    if "contributing_model" not in signals.columns:
        return {}

    # Overall counts
    contrib = signals["contributing_model"].value_counts()
    total = len(signals)

    result = {
        "counts": contrib.to_dict(),
        "fractions": (contrib / total).to_dict() if total > 0 else {},
    }

    # By direction
    if "final_weight" in signals.columns:
        longs = signals[signals["final_weight"] > 0]
        shorts = signals[signals["final_weight"] < 0]

        if len(longs) > 0:
            result["long_contributions"] = longs["contributing_model"].value_counts().to_dict()
        if len(shorts) > 0:
            result["short_contributions"] = shorts["contributing_model"].value_counts().to_dict()

    return result


def compute_backtest_metrics(weekly: pd.DataFrame) -> Dict:
    """Compute overall backtest metrics from weekly data."""
    if len(weekly) == 0:
        return {}

    returns = weekly["portfolio_return"].values
    n_weeks = len(returns)
    years = n_weeks / 52.0

    # Basic metrics
    total_return = (1 + returns).prod() - 1
    mean_return = returns.mean()
    volatility = returns.std()

    # Risk metrics
    downside = returns[returns < 0]
    downside_vol = downside.std() if len(downside) > 0 else 0

    # Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = drawdowns.min()

    # Win rate
    hit_rate = (returns > 0).mean()

    # Profit factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else float("inf")

    # Annualized
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    ann_vol = volatility * np.sqrt(52)
    sharpe = cagr / ann_vol if ann_vol > 0 else 0
    sortino = cagr / (downside_vol * np.sqrt(52)) if downside_vol > 0 else 0
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

    # Position stats
    avg_gross = weekly["gross_exposure"].mean()
    avg_longs = weekly["n_longs"].mean()
    avg_shorts = weekly["n_shorts"].mean()

    # Gating stats
    gating_frac = 0.0
    if "gating_triggered" in weekly.columns:
        gating_frac = weekly["gating_triggered"].mean()

    return {
        "n_weeks": n_weeks,
        "years": years,
        "total_return": total_return,
        "cagr": cagr,
        "volatility": volatility,
        "ann_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "hit_rate": hit_rate,
        "profit_factor": profit_factor,
        "avg_gross_exposure": avg_gross,
        "avg_longs": avg_longs,
        "avg_shorts": avg_shorts,
        "gating_triggered_frac": gating_frac,
        "best_week": returns.max(),
        "worst_week": returns.min(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-model position sizing backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model selection
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated list of models (default: all four)",
    )

    # Data paths
    parser.add_argument(
        "--prediction-path",
        type=str,
        required=True,
        help="Path to multi-model predictions",
    )
    parser.add_argument(
        "--targets-path",
        type=str,
        default="artifacts/targets_triple_barrier.parquet",
        help="Path to targets file",
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default="artifacts/features_complete.parquet",
        help="Path to features file (for regime gating)",
    )

    # Sizing config
    parser.add_argument(
        "--sizing-config",
        type=str,
        default=None,
        help="Path to sizing config JSON (from optimization)",
    )

    # Override options
    parser.add_argument(
        "--combine-policy",
        type=str,
        default=None,
        choices=["mode_priority", "blend"],
        help="Override combine policy from config",
    )
    parser.add_argument(
        "--regime-gating",
        type=str,
        default=None,
        choices=["on", "off"],
        help="Override regime gating setting",
    )
    parser.add_argument(
        "--max-gross-exposure",
        type=float,
        default=None,
        help="Override max gross exposure",
    )

    # Date filters
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Backtest start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Backtest end date (YYYY-MM-DD)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/backtests",
        help="Output directory for results",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="multi_model",
        help="Name prefix for output files",
    )
    parser.add_argument(
        "--decision-log",
        type=str,
        default=None,
        help="Path to save decision log parquet (contains justification for each signal)",
    )

    # Diagnostics
    parser.add_argument(
        "--diagnostics",
        type=str,
        default="off",
        choices=["on", "off"],
        help="Enable sizing diagnostics with warnings",
    )
    parser.add_argument(
        "--diagnostics-output-dir",
        type=str,
        default="artifacts/diagnostics",
        help="Output directory for diagnostics",
    )
    parser.add_argument(
        "--compute-diagnostic-baselines",
        action="store_true",
        help="Compute and save baseline distributions for drift metrics",
    )
    parser.add_argument(
        "--diagnostics-thresholds",
        type=str,
        default=None,
        help="Path to custom diagnostics thresholds JSON",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Multi-Model Position Sizing Backtest")
    print("=" * 60)
    print()

    # Parse models
    models = parse_models(args.models)
    print(f"Models: {[m.value for m in models]}")

    # Load or create config
    if args.sizing_config and Path(args.sizing_config).exists():
        print(f"\nLoading sizing config from: {args.sizing_config}")
        config = load_multi_model_config(args.sizing_config)
    else:
        print("\nUsing default sizing configuration")
        config = MultiModelSizingConfig(
            models=[m.value for m in models],
        )

    # Apply overrides
    if args.combine_policy:
        config.combine_policy = args.combine_policy
    if args.regime_gating:
        config.regime_gating.enabled = (args.regime_gating == "on")
    if args.max_gross_exposure:
        config.max_gross_exposure = args.max_gross_exposure

    print(f"  Combine policy: {config.combine_policy}")
    print(f"  Regime gating: {'enabled' if config.regime_gating.enabled else 'disabled'}")
    print(f"  Max gross exposure: {config.max_gross_exposure:.0%}")

    # Load predictions
    print("\nLoading predictions...")
    prediction_path = Path(args.prediction_path)

    try:
        predictions = load_multi_model_predictions(prediction_path, models)
        print(f"  Loaded {len(predictions):,} prediction rows")
    except Exception as e:
        print(f"Error loading predictions: {e}")
        sys.exit(1)

    # Load targets
    print("\nLoading targets...")
    targets = load_targets(args.targets_path)
    print(f"  Loaded {len(targets):,} target rows")

    # Merge predictions with targets
    print("\nMerging predictions with targets...")
    signals = merge_predictions_with_targets(predictions, targets, models)
    print(f"  Merged signals: {len(signals):,}")

    if len(signals) == 0:
        print("ERROR: No overlapping dates between predictions and targets")
        sys.exit(1)

    # Apply date filters
    if args.start_date:
        signals = signals[signals["week_monday"] >= pd.to_datetime(args.start_date)]
    if args.end_date:
        signals = signals[signals["week_monday"] <= pd.to_datetime(args.end_date)]

    print(f"  Date range: {signals['week_monday'].min().date()} to {signals['week_monday'].max().date()}")
    print(f"  Weeks: {signals['week_monday'].nunique()}")

    # Load regime features if gating enabled
    regime_features = None
    if config.regime_gating.enabled:
        print("\nLoading regime features...")
        regime_features = load_regime_features(args.features_path)
        print(f"  Loaded {len(regime_features):,} feature rows")

    # Create sizing engine
    engine = MultiModelSizingEngine(config)

    # Compute weights for all weeks
    print("\nComputing weekly weights...")
    weighted_signals = engine.compute_weekly_weights(signals, regime_features)
    print(f"  Weighted {len(weighted_signals):,} signals")

    # Compute weekly metrics
    print("\nComputing weekly metrics...")
    weekly = compute_weekly_metrics(weighted_signals)

    # Compute overall metrics
    metrics = compute_backtest_metrics(weekly)

    # Compute model contributions
    contributions = compute_model_contributions(weighted_signals)

    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    print(f"\nPeriod: {weekly['week_monday'].min().date()} to {weekly['week_monday'].max().date()}")
    print(f"Duration: {metrics['n_weeks']} weeks ({metrics['years']:.2f} years)")

    print("\n--- RETURNS ---")
    print(f"  Total Return:     {metrics['total_return']:>10.2%}")
    print(f"  CAGR:             {metrics['cagr']:>10.2%}")
    print(f"  Best Week:        {metrics['best_week']:>10.2%}")
    print(f"  Worst Week:       {metrics['worst_week']:>10.2%}")

    print("\n--- RISK ---")
    print(f"  Volatility (ann): {metrics['ann_volatility']:>10.2%}")
    print(f"  Max Drawdown:     {metrics['max_drawdown']:>10.2%}")
    print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:>10.2f}")
    print(f"  Sortino Ratio:    {metrics['sortino_ratio']:>10.2f}")
    print(f"  Calmar Ratio:     {metrics['calmar_ratio']:>10.2f}")

    print("\n--- TRADING ---")
    print(f"  Hit Rate:         {metrics['hit_rate']:>10.2%}")
    print(f"  Profit Factor:    {metrics['profit_factor']:>10.2f}")
    print(f"  Avg Gross Exp:    {metrics['avg_gross_exposure']:>10.2%}")
    print(f"  Avg Longs:        {metrics['avg_longs']:>10.1f}")
    print(f"  Avg Shorts:       {metrics['avg_shorts']:>10.1f}")

    if config.regime_gating.enabled:
        print("\n--- REGIME GATING ---")
        print(f"  Weeks w/ gating:  {metrics['gating_triggered_frac']:>10.2%}")

    if contributions.get("counts"):
        print("\n--- MODEL CONTRIBUTIONS ---")
        for model, count in contributions["counts"].items():
            frac = contributions["fractions"].get(model, 0)
            print(f"  {model:20s}: {count:>6} ({frac:>6.1%})")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save equity curve
    equity_path = output_dir / f"{args.name}_equity_curve.csv"
    weekly.to_csv(equity_path, index=False)
    print(f"\nEquity curve saved to: {equity_path}")

    # Save metrics
    metrics_path = output_dir / f"{args.name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            **metrics,
            "model_contributions": contributions,
            "config": config.to_dict(),
        }, f, indent=2, default=str)
    print(f"Metrics saved to: {metrics_path}")

    # Save weighted signals for analysis
    signals_path = output_dir / f"{args.name}_weighted_signals.parquet"
    weighted_signals.to_parquet(signals_path)
    print(f"Weighted signals saved to: {signals_path}")

    # Generate decision log if requested
    if args.decision_log:
        print("\nGenerating decision log...")

        # Recompute with justification if not already included
        if "j_winning_model" not in weighted_signals.columns:
            weighted_signals = engine.compute_weekly_weights(
                signals,
                regime_features=regime_features,
                include_justification=True,
            )

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        decision_log = create_decision_log(weighted_signals, run_id=run_id)

        decision_log_path = Path(args.decision_log)
        decision_log_path.parent.mkdir(parents=True, exist_ok=True)
        decision_log.to_parquet(decision_log_path, index=False)

        print(f"  Decision log saved to: {decision_log_path}")
        print(f"  Rows: {len(decision_log):,}")
        j_cols = [c for c in decision_log.columns if c.startswith("j_")]
        print(f"  Justification columns: {len(j_cols)} columns")

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    # Handle baseline computation (separate mode)
    if args.compute_diagnostic_baselines:
        print("\n" + "=" * 60)
        print("COMPUTING DIAGNOSTIC BASELINES")
        print("=" * 60)

        from src.diagnostics.sizing import BaselineManager

        baseline_manager = BaselineManager(
            baseline_dir=str(Path(args.diagnostics_output_dir) / "baselines")
        )

        print("\nComputing baselines from predictions...")
        baselines = baseline_manager.compute_baselines(
            predictions=weighted_signals,
            models=[m.value for m in models],
            metrics=["probability", "edge"],
            overwrite=True,
        )

        print(f"\nComputed {len(baselines)} baselines")
        print(f"Baselines saved to: {args.diagnostics_output_dir}/baselines/")

    # Run diagnostics if enabled
    if args.diagnostics == "on":
        print("\n" + "=" * 60)
        print("RUNNING SIZING DIAGNOSTICS")
        print("=" * 60)

        from src.diagnostics.sizing import (
            SizingDiagnosticsRunner,
            SizingDiagnosticThresholds,
        )

        # Load custom thresholds if provided
        thresholds = None
        if args.diagnostics_thresholds:
            thresholds = SizingDiagnosticThresholds.from_json(args.diagnostics_thresholds)
            print(f"\nLoaded custom thresholds from: {args.diagnostics_thresholds}")

        # Create runner
        diag_runner = SizingDiagnosticsRunner(
            weighted_signals=weighted_signals,
            sizing_config=config.to_dict(),
            models=[m.value for m in models],
            thresholds=thresholds,
            baseline_dir=str(Path(args.diagnostics_output_dir) / "baselines"),
            backtest_mode=True,
            run_id=args.name,
        )

        # Run diagnostics
        diag_report = diag_runner.run()

        # Save reports
        diag_paths = diag_runner.save_report(
            output_dir=args.diagnostics_output_dir,
            save_json=True,
            save_csv=True,
            save_md=True,
        )

        # Print summary
        diag_runner.print_summary()

        print(f"\nDiagnostics saved to:")
        for fmt, path in diag_paths.items():
            print(f"  {fmt.upper()}: {path}")

    print("\n" + "=" * 60)
    print("Backtest complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
