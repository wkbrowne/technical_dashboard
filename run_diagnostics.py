#!/usr/bin/env python
"""
Model Diagnostic Report Generator.

Runs comprehensive diagnostics for the 4-model LightGBM trading system:
- Data quality and leakage detection
- CV stability analysis
- Calibration and ranking sanity
- Sample weighting validation
- Hyperopt parameter analysis
- Feature importance and story consistency

Usage:
    # Run diagnostics for a specific model
    python run_diagnostics.py --model long_normal

    # Run diagnostics for all 4 models
    python run_diagnostics.py --all-models

    # With model fitting (if no trained model available)
    python run_diagnostics.py --model long_normal --fit

Output:
    artifacts/diagnostics/{model_key}/diagnostic_report.json
    artifacts/diagnostics/{model_key}/diagnostic_report.md
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.diagnostics import ModelDiagnosticRunner, DiagnosticResult


MODEL_KEYS = ['long_normal', 'long_parabolic', 'short_normal', 'short_parabolic']


def run_single_model(
    model_key: str,
    fit_model: bool = False,
    n_folds: int = 5,
    gap: int = 20,
    n_jobs: int = 8,
) -> DiagnosticResult:
    """
    Run diagnostics for a single model.

    Args:
        model_key: Model key string
        fit_model: Whether to fit model for diagnostics
        n_folds: Number of CV folds
        gap: Embargo gap in days
        n_jobs: Number of threads for model fitting

    Returns:
        DiagnosticResult
    """
    runner = ModelDiagnosticRunner(
        model_key=model_key,
        fit_model=fit_model,
        n_folds=n_folds,
        gap=gap,
        n_jobs=n_jobs,
    )

    result = runner.run()
    runner.save_report()

    return result


def run_all_models(
    fit_model: bool = False,
    n_folds: int = 5,
    gap: int = 20,
    n_jobs: int = 8,
) -> Dict[str, DiagnosticResult]:
    """
    Run diagnostics for all 4 models.

    Args:
        fit_model: Whether to fit models for diagnostics
        n_folds: Number of CV folds
        gap: Embargo gap in days
        n_jobs: Number of threads for model fitting

    Returns:
        Dict mapping model_key to DiagnosticResult
    """
    results = {}
    start_time = time.time()

    print("=" * 70)
    print("MULTI-MODEL DIAGNOSTIC REPORT")
    print(f"  Models: {', '.join(MODEL_KEYS)}")
    print("=" * 70)
    print()

    for i, model_key in enumerate(MODEL_KEYS):
        print(f"\n[{i+1}/{len(MODEL_KEYS)}] Running diagnostics for {model_key.upper()}")
        print("-" * 70)

        result = run_single_model(
            model_key=model_key,
            fit_model=fit_model,
            n_folds=n_folds,
            gap=gap,
            n_jobs=n_jobs,
        )
        results[model_key] = result

    elapsed = time.time() - start_time

    # Print summary
    print("\n" + "=" * 70)
    print("MULTI-MODEL DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print("\nResults per model:")
    print("-" * 60)
    print(f"{'Model':<20} {'Status':>10} {'Critical':>10} {'Warn':>10} {'Info':>10}")
    print("-" * 60)

    for model_key, result in results.items():
        status = "PASS" if result.passed else "FAIL"
        print(f"{model_key:<20} {status:>10} {result.n_critical:>10} "
              f"{result.n_warn:>10} {result.n_info:>10}")

    print("-" * 60)

    # Overall status
    all_passed = all(r.passed for r in results.values())
    total_critical = sum(r.n_critical for r in results.values())
    total_warn = sum(r.n_warn for r in results.values())

    print(f"\nOverall: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")
    print(f"Total critical issues: {total_critical}")
    print(f"Total warnings: {total_warn}")
    print(f"\nReports saved to: artifacts/diagnostics/")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Model Diagnostic Report Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run for specific model
    python run_diagnostics.py --model long_normal

    # Run for all 4 models
    python run_diagnostics.py --all-models

    # With model fitting (slower, but doesn't require trained model)
    python run_diagnostics.py --model long_normal --fit

    # Custom CV settings
    python run_diagnostics.py --model long_normal --n-folds 3 --gap 20
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=MODEL_KEYS,
        help='Model key to run diagnostics for'
    )
    parser.add_argument(
        '--all-models',
        action='store_true',
        help='Run diagnostics for all 4 models'
    )
    parser.add_argument(
        '--fit',
        action='store_true',
        help='Fit model during diagnostics (slower, but works without trained model)'
    )
    parser.add_argument(
        '--n-folds',
        type=int,
        default=5,
        help='Number of CV folds (default: 5)'
    )
    parser.add_argument(
        '--gap',
        type=int,
        default=20,
        help='Embargo gap in days (default: 20)'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=8,
        help='Number of threads for model fitting (default: 8)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.model and not args.all_models:
        parser.error("Must specify either --model or --all-models")

    if args.model and args.all_models:
        parser.error("Cannot specify both --model and --all-models")

    # Run diagnostics
    if args.all_models:
        run_all_models(
            fit_model=args.fit,
            n_folds=args.n_folds,
            gap=args.gap,
            n_jobs=args.n_jobs,
        )
    else:
        run_single_model(
            model_key=args.model,
            fit_model=args.fit,
            n_folds=args.n_folds,
            gap=args.gap,
            n_jobs=args.n_jobs,
        )


if __name__ == '__main__':
    main()
