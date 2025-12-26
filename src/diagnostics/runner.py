"""
Main diagnostic runner that orchestrates all checks.
"""

import gc
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd

from .core import (
    DiagnosticFlag, DiagnosticResult, Severity,
    DiagnosticThresholds, DEFAULT_THRESHOLDS,
    convert_to_json_serializable,
)
from .checks_data import run_leakage_checks
from .checks_data_quality import run_data_quality_checks
from .checks_cv import run_cv_checks
from .checks_calibration import run_calibration_checks
from .checks_weights import run_weight_checks
from .checks_hyperopt import run_hyperopt_checks
from .checks_features import run_feature_checks

# Suppress warnings during model fitting
warnings.filterwarnings('ignore', message='.*feature_name.*')
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')


class ModelDiagnosticRunner:
    """
    Orchestrates all diagnostic checks for a model.

    Usage:
        runner = ModelDiagnosticRunner(model_key='long_normal')
        result = runner.run()
        runner.save_report()
    """

    def __init__(
        self,
        model_key: str,
        features_path: Path = Path('artifacts/features_complete.parquet'),
        targets_path: Path = Path('artifacts/targets_triple_barrier.parquet'),
        hyperopt_dir: Path = Path('artifacts/hyperopt'),
        models_dir: Path = Path('artifacts/models'),
        output_dir: Optional[Path] = None,
        thresholds: DiagnosticThresholds = DEFAULT_THRESHOLDS,
        n_folds: int = 5,
        gap: int = 20,
        fit_model: bool = False,
        n_jobs: int = 8,
    ):
        """
        Initialize diagnostic runner.

        Args:
            model_key: Model key (long_normal, long_parabolic, short_normal, short_parabolic)
            features_path: Path to features parquet
            targets_path: Path to targets parquet
            hyperopt_dir: Path to hyperopt artifacts
            models_dir: Path to trained model artifacts
            output_dir: Output directory for reports (default: artifacts/diagnostics/{model_key})
            thresholds: Diagnostic thresholds
            n_folds: Number of CV folds
            gap: Embargo gap in days
            fit_model: Whether to fit model for diagnostics if not available
            n_jobs: Number of threads for model fitting
        """
        self.model_key = model_key
        self.features_path = Path(features_path)
        self.targets_path = Path(targets_path)
        self.hyperopt_dir = Path(hyperopt_dir)
        self.models_dir = Path(models_dir)
        self.output_dir = output_dir or Path('artifacts/diagnostics') / model_key
        self.thresholds = thresholds
        self.n_folds = n_folds
        self.gap = gap
        self.fit_model = fit_model
        self.n_jobs = n_jobs

        # Will be populated during run
        self.result: Optional[DiagnosticResult] = None
        self._data_loaded = False

    def _load_model_config(self) -> Tuple[Optional[Dict], List[str]]:
        """Load model config and feature list."""
        # Try to import from src
        try:
            from src.config.model_keys import ModelKey
            from src.feature_selection.base_features import get_featureset

            model_key_enum = ModelKey(self.model_key)
            features = get_featureset(model_key_enum, include_expansion=False, flat=True)
        except (ImportError, ValueError):
            # Fallback: load from hyperopt config
            config_file = self.hyperopt_dir / 'model_configs.json'
            if config_file.exists():
                with open(config_file) as f:
                    configs = json.load(f)
                    model_config = configs.get('models', {}).get(self.model_key, {})
                    features = model_config.get('features', {}).get('names', [])
            else:
                features = []

        # Load best params
        best_params = None
        params_file = self.hyperopt_dir / self.model_key / 'best_params.json'
        if params_file.exists():
            with open(params_file) as f:
                best_params = json.load(f)

        return best_params, features

    def _get_model_target_column(self) -> str:
        """Get target column name for this model."""
        target_col_map = {
            'long_normal': 'hit_long_normal',
            'long_parabolic': 'hit_long_parabolic',
            'short_normal': 'hit_short_normal',
            'short_parabolic': 'hit_short_parabolic',
        }
        return target_col_map.get(self.model_key, 'hit')

    def _is_short_model(self) -> bool:
        """Check if this is a short-side model."""
        return self.model_key in ('short_normal', 'short_parabolic')

    def _load_data(
        self,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray, List[str]]:
        """
        Load and prepare data for diagnostics.

        Returns:
            X, y, sample_weight, dates, symbols, available_features
        """
        print("Loading features...")
        features = pd.read_parquet(self.features_path)
        print(f"  Features shape: {features.shape}")

        print("Loading targets...")
        targets = pd.read_parquet(self.targets_path)
        targets = targets.rename(columns={'t0': 'date'})
        print(f"  Targets shape: {targets.shape}")

        # Get target column
        hit_col = self._get_model_target_column()
        if hit_col not in targets.columns:
            hit_col = 'hit'
            print(f"  Warning: Using fallback target column 'hit'")

        # Merge
        target_cols = ['symbol', 'date', 'weight_final', hit_col]
        target_cols = [c for c in target_cols if c in targets.columns]

        merged = features.merge(
            targets[target_cols],
            on=['symbol', 'date'],
            how='inner'
        )

        # Filter neutral
        merged = merged[merged[hit_col] != 0].copy()

        # Binary target
        if self._is_short_model():
            merged['target'] = (merged[hit_col] == -1).astype(int)
        else:
            merged['target'] = (merged[hit_col] == 1).astype(int)

        merged = merged.sort_values(['date', 'symbol']).reset_index(drop=True)

        # Available features
        available_features = [f for f in feature_names if f in merged.columns]
        missing = set(feature_names) - set(available_features)
        if missing:
            print(f"  Warning: {len(missing)} features not in data")

        # Extract arrays
        X = merged[available_features].values.astype(np.float32)
        y = merged['target'].values
        sample_weight = merged['weight_final'].values if 'weight_final' in merged.columns else None
        dates = pd.to_datetime(merged['date'])
        symbols = merged['symbol'].values

        print(f"\nData prepared:")
        print(f"  Samples: {len(X):,}")
        print(f"  Features: {len(available_features)}")
        print(f"  Positive rate: {y.mean()*100:.1f}%")

        del features, targets, merged
        gc.collect()

        self._data_loaded = True

        return X, y, sample_weight, dates, symbols, available_features

    def _get_cv_splits(
        self,
        dates: pd.DatetimeIndex,
        min_train_samples: int = 3000,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate expanding CV splits with embargo."""
        unique_dates = np.sort(dates.unique())
        n_dates = len(unique_dates)

        min_train_dates = max(min_train_samples // 100, 50)
        available_dates = n_dates - min_train_dates
        test_size = available_dates // (self.n_folds + 1)

        splits = []
        for fold in range(self.n_folds):
            test_end_idx = n_dates - 1 - fold * test_size
            test_start_idx = test_end_idx - test_size + 1
            train_end_idx = test_start_idx - self.gap - 1

            if train_end_idx < min_train_dates:
                continue

            train_dates = unique_dates[:train_end_idx + 1]
            test_dates = unique_dates[test_start_idx:test_end_idx + 1]

            train_mask = dates.isin(train_dates)
            test_mask = dates.isin(test_dates)

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            if len(train_idx) >= min_train_samples and len(test_idx) > 0:
                splits.append((train_idx, test_idx))

        return splits[::-1]  # Chronological order

    def _compute_fold_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Compute metrics for a single fold."""
        from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

        n = len(y_pred)
        k = max(1, int(n * 0.10))

        sorted_idx = np.argsort(y_pred)
        top_idx = sorted_idx[-k:]
        bottom_idx = sorted_idx[:k]

        return {
            'auc': roc_auc_score(y_true, y_pred),
            'aupr': average_precision_score(y_true, y_pred),
            'brier': brier_score_loss(y_true, np.clip(y_pred, 0, 1)),
            'precision_top_10': y_true[top_idx].mean(),
            'precision_bottom_10': y_true[bottom_idx].mean(),
        }

    def _run_cv_with_predictions(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray],
        cv_splits: List[Tuple[np.ndarray, np.ndarray]],
        params: Dict[str, Any],
    ) -> Tuple[List[Dict[str, float]], List[Tuple[np.ndarray, np.ndarray]], List[Dict[str, float]]]:
        """
        Run cross-validation and collect predictions.

        Returns:
            fold_metrics, fold_predictions (y_true, y_pred), fold_metrics_unweighted
        """
        import lightgbm as lgb

        fold_metrics = []
        fold_predictions = []
        fold_metrics_unweighted = []

        # Clean params for LightGBM
        lgb_params = {k: v for k, v in params.items() if not k.startswith('_')}

        # Handle num_leaves from exponent
        if 'num_leaves_exp' in lgb_params:
            lgb_params['num_leaves'] = int(2 ** lgb_params.pop('num_leaves_exp'))

        # Set defaults
        lgb_params.setdefault('objective', 'binary')
        lgb_params.setdefault('metric', 'auc')
        lgb_params.setdefault('boosting_type', 'gbdt')
        lgb_params.setdefault('verbosity', -1)
        lgb_params.setdefault('n_estimators', 500)
        lgb_params.setdefault('num_threads', self.n_jobs)

        # Handle scale_pos_weight
        if lgb_params.pop('use_balanced', False):
            lgb_params['scale_pos_weight'] = (y == 0).sum() / ((y == 1).sum() + 1)

        print(f"\nRunning {len(cv_splits)}-fold CV...")

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            X_train = np.nan_to_num(X[train_idx], nan=0.0, posinf=0.0, neginf=0.0)
            X_test = np.nan_to_num(X[test_idx], nan=0.0, posinf=0.0, neginf=0.0)
            y_train, y_test = y[train_idx], y[test_idx]
            w_train = sample_weight[train_idx] if sample_weight is not None else None

            # Fit model
            model = lgb.LGBMClassifier(**lgb_params)
            model.fit(
                X_train, y_train,
                sample_weight=w_train,
                eval_set=[(X_test, y_test)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )

            # Predict
            y_pred = model.predict_proba(X_test)[:, 1]

            # Store predictions
            fold_predictions.append((y_test, y_pred))

            # Compute metrics
            metrics = self._compute_fold_metrics(y_test, y_pred)
            fold_metrics.append(metrics)

            print(f"  Fold {fold_idx+1}: AUC={metrics['auc']:.4f}, "
                  f"Brier={metrics['brier']:.4f}, P@10={metrics['precision_top_10']:.4f}")

            # Also compute without weights for sensitivity analysis
            if sample_weight is not None:
                model_uw = lgb.LGBMClassifier(**lgb_params)
                model_uw.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50, verbose=False),
                        lgb.log_evaluation(period=0),
                    ],
                )
                y_pred_uw = model_uw.predict_proba(X_test)[:, 1]
                metrics_uw = self._compute_fold_metrics(y_test, y_pred_uw)
                fold_metrics_unweighted.append(metrics_uw)

        return fold_metrics, fold_predictions, fold_metrics_unweighted

    def run(self) -> DiagnosticResult:
        """
        Run all diagnostic checks.

        Returns:
            DiagnosticResult with all flags and summaries
        """
        print("=" * 70)
        print(f"DIAGNOSTIC REPORT: {self.model_key.upper()}")
        print("=" * 70)

        self.result = DiagnosticResult(model_key=self.model_key)

        # Load config and features
        print("\nLoading model configuration...")
        best_params, feature_names = self._load_model_config()

        if not feature_names:
            self.result.add_flag(DiagnosticFlag(
                severity=Severity.CRITICAL,
                check_name='no_features',
                symptom=f"No features found for model {self.model_key}",
                why_it_matters="Cannot run diagnostics without feature list",
                suggested_fix="Run hyperopt first or check feature_selection/base_features.py",
                evidence={'model_key': self.model_key},
            ))
            return self.result

        print(f"  Found {len(feature_names)} features")

        # Load data
        X, y, sample_weight, dates, symbols, available_features = self._load_data(feature_names)

        # Generate CV splits
        cv_splits = self._get_cv_splits(dates)
        print(f"  Generated {len(cv_splits)} CV folds")

        # Get params for CV
        if best_params is None:
            if not self.fit_model:
                self.result.add_flag(DiagnosticFlag(
                    severity=Severity.WARN,
                    check_name='no_hyperopt_params',
                    symptom="No hyperopt parameters found",
                    why_it_matters="Using default parameters may give suboptimal results",
                    suggested_fix=f"Run hyperopt: python run_model_tuning.py --model {self.model_key}",
                    evidence={},
                ))
            # Use reasonable defaults
            best_params = {
                'num_leaves': 64,
                'min_child_samples': 200,
                'max_depth': 6,
                'learning_rate': 0.05,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
            }

        # Run CV to get metrics and predictions
        fold_metrics, fold_predictions, fold_metrics_unweighted = self._run_cv_with_predictions(
            X, y, sample_weight, cv_splits, best_params
        )

        # Run all checks
        print("\nRunning diagnostic checks...")

        # 1. Data quality checks
        print("  Data quality checks...")
        dq_flags, dq_summary = run_data_quality_checks(
            X, available_features, feature_names, self.thresholds
        )
        for flag in dq_flags:
            self.result.add_flag(flag)
        self.result.metrics_summary['data_quality'] = dq_summary

        # 2. Leakage checks
        print("  Leakage detection checks...")
        leakage_flags, leakage_summary = run_leakage_checks(
            cv_splits, dates, symbols,
            fold_metrics, self.gap, self.thresholds
        )
        for flag in leakage_flags:
            self.result.add_flag(flag)
        self.result.metrics_summary['leakage'] = leakage_summary

        # 3. CV checks
        print("  CV stability checks...")
        cv_flags, cv_summary = run_cv_checks(
            cv_splits, y, fold_metrics, self.thresholds
        )
        for flag in cv_flags:
            self.result.add_flag(flag)
        self.result.metrics_summary['cv'] = cv_summary

        # 4. Calibration checks
        print("  Calibration & ranking checks...")
        base_rate = y.mean()
        cal_flags, cal_summary = run_calibration_checks(
            fold_predictions, base_rate, fold_metrics, self.thresholds
        )
        for flag in cal_flags:
            self.result.add_flag(flag)
        self.result.metrics_summary['calibration'] = cal_summary

        # 5. Weight checks
        print("  Sample weighting checks...")
        weight_flags, weight_summary = run_weight_checks(
            sample_weight, fold_metrics,
            fold_metrics_unweighted if fold_metrics_unweighted else None,
            self.thresholds
        )
        for flag in weight_flags:
            self.result.add_flag(flag)
        self.result.metrics_summary['weights'] = weight_summary

        # 6. Hyperopt checks
        print("  Hyperopt sanity checks...")
        hyperopt_flags, hyperopt_summary = run_hyperopt_checks(
            self.model_key, self.hyperopt_dir, self.thresholds
        )
        for flag in hyperopt_flags:
            self.result.add_flag(flag)
        self.result.metrics_summary['hyperopt'] = hyperopt_summary

        # 7. Feature checks
        print("  Feature sanity checks...")
        feature_flags, feature_summary = run_feature_checks(
            available_features, self.model_key, self.models_dir, self.thresholds
        )
        for flag in feature_flags:
            self.result.add_flag(flag)
        self.result.metrics_summary['features'] = feature_summary

        # Print summary
        print("\n" + "=" * 70)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 70)
        print(f"  CRITICAL: {self.result.n_critical}")
        print(f"  WARN:     {self.result.n_warn}")
        print(f"  INFO:     {self.result.n_info}")
        if self.result.passed:
            print(f"  STATUS:   PASSED")
        else:
            print(f"  STATUS:   FAILED ({self.result.n_critical} critical issue(s))")

        return self.result

    def save_report(self) -> Tuple[Path, Path]:
        """
        Save diagnostic report to JSON and Markdown files.

        Returns:
            Tuple of (json_path, markdown_path)
        """
        if self.result is None:
            raise ValueError("Must run diagnostics before saving report")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = self.output_dir / 'diagnostic_report.json'
        with open(json_path, 'w') as f:
            json.dump(self.result.to_dict(), f, indent=2)

        # Save Markdown
        md_path = self.output_dir / 'diagnostic_report.md'
        with open(md_path, 'w') as f:
            f.write(self._generate_markdown())

        print(f"\nReports saved to:")
        print(f"  JSON: {json_path}")
        print(f"  Markdown: {md_path}")

        return json_path, md_path

    def _generate_markdown(self) -> str:
        """Generate markdown report."""
        status_msg = '✅ PASSED' if self.result.passed else f'❌ FAILED ({self.result.n_critical} critical)'
        lines = [
            f"# Diagnostic Report: {self.model_key.upper()}",
            "",
            f"**Generated:** {self.result.timestamp}",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Status | {status_msg} |",
            f"| Critical Issues | {self.result.n_critical} |",
            f"| Warnings | {self.result.n_warn} |",
            f"| Info | {self.result.n_info} |",
            "",
        ]

        # Add key metrics if available
        if 'cv' in self.result.metrics_summary:
            cv_metrics = self.result.metrics_summary['cv'].get('metrics', {})
            if 'auc' in cv_metrics:
                auc = cv_metrics['auc']
                lines.extend([
                    "### CV Performance",
                    "",
                    f"| Metric | Mean | Std |",
                    f"|--------|------|-----|",
                    f"| AUC | {auc.get('mean', 'N/A')} | {auc.get('std', 'N/A')} |",
                ])
                if 'aupr' in cv_metrics:
                    aupr = cv_metrics['aupr']
                    lines.append(f"| AUPR | {aupr.get('mean', 'N/A')} | {aupr.get('std', 'N/A')} |")
                if 'brier' in cv_metrics:
                    brier = cv_metrics['brier']
                    lines.append(f"| Brier | {brier.get('mean', 'N/A')} | {brier.get('std', 'N/A')} |")
                if 'precision_top_10' in cv_metrics:
                    prec = cv_metrics['precision_top_10']
                    lines.append(f"| Precision@10% | {prec.get('mean', 'N/A')} | {prec.get('std', 'N/A')} |")
                lines.append("")

        # Group flags by severity
        lines.extend([
            "---",
            "",
            "## Diagnostic Flags",
            "",
        ])

        for severity in [Severity.CRITICAL, Severity.WARN, Severity.INFO]:
            severity_flags = [f for f in self.result.flags if f.severity == severity]

            if severity_flags:
                emoji = {"CRITICAL": "🔴", "WARN": "🟡", "INFO": "🔵"}[str(severity)]
                lines.extend([
                    f"### {emoji} {severity} ({len(severity_flags)})",
                    "",
                ])

                for flag in severity_flags:
                    lines.extend([
                        f"#### {flag.check_name}",
                        "",
                        f"**Symptom:** {flag.symptom}",
                        "",
                        f"**Why it matters:** {flag.why_it_matters}",
                        "",
                        f"**Suggested fix:** {flag.suggested_fix}",
                        "",
                        "<details>",
                        "<summary>Evidence</summary>",
                        "",
                        "```json",
                        json.dumps(convert_to_json_serializable(flag.evidence), indent=2),
                        "```",
                        "</details>",
                        "",
                        "---",
                        "",
                    ])

        # Add metrics details
        lines.extend([
            "## Detailed Metrics",
            "",
        ])

        for section, summary in self.result.metrics_summary.items():
            lines.extend([
                f"### {section.title()}",
                "",
                "<details>",
                "<summary>Click to expand</summary>",
                "",
                "```json",
                json.dumps(convert_to_json_serializable(summary), indent=2),
                "```",
                "</details>",
                "",
            ])

        return "\n".join(lines)
