#!/usr/bin/env python
"""
Unified Position Sizing Pipeline.

This script orchestrates the complete position sizing workflow:
1. Generate out-of-sample predictions via walk-forward CV
2. Optimize sizing parameters using those predictions
3. Run backtest with optimized parameters

This is the CORRECT way to develop and evaluate a trading strategy.
Using production model predictions for backtesting would be information leakage.

Usage:
    # Full pipeline
    python scripts/run_sizing_pipeline.py

    # Just generate predictions (if you want to run optimization separately)
    python scripts/run_sizing_pipeline.py --predictions-only

    # Skip predictions (use existing cv_predictions.parquet)
    python scripts/run_sizing_pipeline.py --skip-predictions

    # Quick test run
    python scripts/run_sizing_pipeline.py --n-trials 20
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print("\n" + "=" * 60)
    print(f"STEP: {description}")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)

    if result.returncode != 0:
        print(f"\nERROR: {description} failed with return code {result.returncode}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run complete position sizing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline with 100 optimization trials
    python scripts/run_sizing_pipeline.py --n-trials 100

    # Quick test run
    python scripts/run_sizing_pipeline.py --n-trials 20

    # Just generate CV predictions
    python scripts/run_sizing_pipeline.py --predictions-only

    # Run optimization using existing predictions
    python scripts/run_sizing_pipeline.py --skip-predictions --n-trials 200
        """
    )
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Number of Optuna optimization trials")
    parser.add_argument("--method", type=str, default="monotone_probability",
                        choices=["monotone_probability", "rank_bucket", "threshold_confidence"],
                        help="Sizing method to optimize")
    parser.add_argument("--weight-scheme", type=str, default="overlap_inverse",
                        choices=["uniform", "overlap_inverse", "liquidity", "inverse_volatility"],
                        help="Sample weighting scheme for training")
    parser.add_argument("--use-hyperopt-params", action="store_true", default=True,
                        help="Use hyperparameters from hyperopt tuning (default: True)")
    parser.add_argument("--no-hyperopt-params", action="store_true",
                        help="Use default conservative parameters instead of hyperopt")
    parser.add_argument("--predictions-only", action="store_true",
                        help="Only generate CV predictions, skip optimization")
    parser.add_argument("--skip-predictions", action="store_true",
                        help="Skip prediction generation, use existing cv_predictions.parquet")
    parser.add_argument("--skip-backtest", action="store_true",
                        help="Skip backtest step")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be run without executing")

    args = parser.parse_args()

    print("=" * 60)
    print("POSITION SIZING PIPELINE")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("This pipeline generates valid out-of-sample predictions and")
    print("optimizes position sizing parameters WITHOUT information leakage.")
    print()

    # Check for required files
    required_files = [
        "artifacts/features_complete.parquet",
        "artifacts/targets_triple_barrier.parquet",
        "artifacts/feature_selection/selected_features.txt",
    ]

    missing = [f for f in required_files if not Path(f).exists()]
    if missing:
        print("ERROR: Missing required files:")
        for f in missing:
            print(f"  - {f}")
        print()
        print("Run the feature pipeline first:")
        print("  python -m src.cli.compute")
        sys.exit(1)

    # Build commands
    use_hyperopt = args.use_hyperopt_params and not args.no_hyperopt_params

    # Step 1: Generate CV predictions
    if not args.skip_predictions:
        pred_cmd = [
            sys.executable, "scripts/generate_cv_predictions.py",
            "--weight-scheme", args.weight_scheme,
        ]
        if use_hyperopt:
            pred_cmd.append("--use-hyperopt-params")

        if args.dry_run:
            print(f"Would run: {' '.join(pred_cmd)}")
        else:
            if not run_command(pred_cmd, "Generate Out-of-Sample CV Predictions"):
                sys.exit(1)

        if args.predictions_only:
            if not args.dry_run:
                print("\n" + "=" * 60)
                print("Predictions generated. Stopping as requested.")
                print("=" * 60)
            else:
                print("\nDry run complete (predictions-only mode).")
            return

    # Check that predictions exist
    cv_pred_file = Path("artifacts/predictions/cv_predictions.parquet")
    if not cv_pred_file.exists() and not args.dry_run:
        print(f"\nERROR: CV predictions not found at {cv_pred_file}")
        print("Run without --skip-predictions to generate them.")
        sys.exit(1)

    # Step 2: Optimize sizing parameters
    opt_cmd = [
        sys.executable, "scripts/run_optimize_sizing.py",
        "--n-trials", str(args.n_trials),
        "--method", args.method,
        "--predictions", str(cv_pred_file),
    ]

    if args.dry_run:
        print(f"Would run: {' '.join(opt_cmd)}")
    else:
        if not run_command(opt_cmd, "Optimize Sizing Parameters"):
            sys.exit(1)

    # Step 3: Run backtest (if script exists)
    if not args.skip_backtest:
        backtest_script = Path("scripts/run_backtest.py")
        if backtest_script.exists():
            backtest_cmd = [
                sys.executable, str(backtest_script),
                "--config", f"artifacts/sizing/best_params_{args.method}.json",
            ]

            if args.dry_run:
                print(f"Would run: {' '.join(backtest_cmd)}")
            else:
                if not run_command(backtest_cmd, "Run Backtest"):
                    print("Backtest failed (non-critical)")
        else:
            print("\nNote: Backtest script not found, skipping.")

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Output files:")
    print(f"  - CV Predictions: artifacts/predictions/cv_predictions.parquet")
    print(f"  - Sizing Params:  artifacts/sizing/best_params_{args.method}.json")
    print()
    print("Key metrics are in the terminal output above.")
    print()
    print("IMPORTANT: These results are based on OUT-OF-SAMPLE predictions.")
    print("The model never saw the test data during training - no leakage!")
    print()


if __name__ == "__main__":
    main()
