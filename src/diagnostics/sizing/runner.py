"""Sizing diagnostics runner.

Orchestrates the complete diagnostic workflow:
1. Compute weekly metrics
2. Apply warning rules
3. Generate reports (JSON, CSV, Markdown)
4. Print terminal summary
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import uuid

import numpy as np
import pandas as pd

from .schema import (
    Severity,
    SizingWarning,
    SizingDiagnosticReport,
    WeeklyDiagnostics,
    DataIntegrityMetrics,
    SignalDistributionMetrics,
    PortfolioMetrics,
    ExecutionRiskMetrics,
    BacktestMetrics,
)
from .thresholds import SizingDiagnosticThresholds, DEFAULT_SIZING_THRESHOLDS
from .warnings import WarningEngine
from .baselines import BaselineManager
from .metrics import (
    compute_data_integrity_metrics,
    compute_signal_distribution_metrics,
    compute_portfolio_metrics,
    compute_execution_risk_metrics,
    compute_backtest_metrics,
    compute_aggregate_metrics,
)


class SizingDiagnosticsRunner:
    """Main runner for sizing diagnostics.

    Usage:
        runner = SizingDiagnosticsRunner(
            weighted_signals=df,
            sizing_config=config,
            models=['long_normal', 'short_normal'],
        )
        report = runner.run()
        runner.save_report("artifacts/diagnostics")
        runner.print_summary()
    """

    def __init__(
        self,
        weighted_signals: pd.DataFrame,
        sizing_config: Optional[Dict[str, Any]] = None,
        models: Optional[List[str]] = None,
        thresholds: Optional[SizingDiagnosticThresholds] = None,
        baseline_dir: str = "artifacts/diagnostics/baselines",
        backtest_mode: bool = False,
        run_id: Optional[str] = None,
    ):
        """Initialize diagnostics runner.

        Args:
            weighted_signals: DataFrame with weighted signals.
                Required columns: week_monday, symbol, final_weight
                Optional: p_<model>, edge_<model>, contributing_model,
                         actual_return, gating_multiplier, rdollar_vol_20, etc.
            sizing_config: Sizing configuration dict.
            models: List of model keys. Auto-detected if None.
            thresholds: Custom thresholds. Uses defaults if None.
            baseline_dir: Directory for baseline files.
            backtest_mode: If True, compute backtest metrics.
            run_id: Unique identifier. Auto-generated if None.
        """
        self.signals = weighted_signals.copy()
        self.sizing_config = sizing_config or {}
        self.thresholds = thresholds or DEFAULT_SIZING_THRESHOLDS
        self.backtest_mode = backtest_mode
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Auto-detect models from columns
        if models is None:
            prob_cols = [c for c in self.signals.columns if c.startswith('p_')]
            self.models = [c[2:] for c in prob_cols]
        else:
            self.models = models

        # Initialize baseline manager
        self.baseline_manager = BaselineManager(baseline_dir)

        # Initialize warning engine with baselines
        self.warning_engine = WarningEngine(
            thresholds=self.thresholds,
            baseline_extreme_rate=self._get_baseline_extreme_rate(),
            baseline_turnover=self._get_baseline_turnover(),
        )

        # Internal state
        self._report: Optional[SizingDiagnosticReport] = None
        self._weekly_diagnostics: List[WeeklyDiagnostics] = []

    def _get_baseline_extreme_rate(self) -> Optional[float]:
        """Get baseline extreme signal rate from stored baselines."""
        # Average across models
        rates = []
        for model in self.models:
            baseline = self.baseline_manager.load_baseline(model, 'probability')
            if baseline is not None:
                # Estimate from percentiles
                p10 = baseline.percentiles.get('p10', 0)
                p90 = baseline.percentiles.get('p90', 1)
                # Rough estimate: fraction below p10 + above p90
                rate = (1 - 0.80)  # By definition, ~20%
                rates.append(rate)
        return np.mean(rates) if rates else None

    def _get_baseline_turnover(self) -> Optional[float]:
        """Get baseline turnover - would need historical data."""
        # This would need to be stored separately
        return None

    def run(self) -> SizingDiagnosticReport:
        """Run complete diagnostics.

        Returns:
            SizingDiagnosticReport with all metrics and warnings.
        """
        # Validate inputs
        if 'week_monday' not in self.signals.columns:
            raise ValueError("signals must have week_monday column")

        # Sort by week
        self.signals['week_monday'] = pd.to_datetime(self.signals['week_monday'])
        weeks = sorted(self.signals['week_monday'].unique())

        # Track cumulative data for backtest mode
        cumulative_returns = pd.Series(dtype=float)
        trailing_returns = pd.Series(dtype=float)
        gating_consecutive_weeks = 0
        previous_week_signals = None
        previous_weights = None

        self._weekly_diagnostics = []

        for week in weeks:
            week_signals = self.signals[self.signals['week_monday'] == week].copy()
            week_str = str(week.date())

            # Compute data integrity metrics
            data_integrity = compute_data_integrity_metrics(
                signals=week_signals,
                models=self.models,
            )

            # Compute signal distribution metrics
            signal_distribution = compute_signal_distribution_metrics(
                signals=week_signals,
                models=self.models,
                baseline_manager=self.baseline_manager,
                previous_week_signals=previous_week_signals,
                sizing_intercept=self.sizing_config.get('sizing_params', {}).get('intercept', 0.5),
            )

            # Compute portfolio metrics
            portfolio = compute_portfolio_metrics(
                signals=week_signals,
                sizing_config=self.sizing_config,
                previous_weights=previous_weights,
            )

            # Track gating consecutive weeks
            if portfolio.gating_reduced_exposure:
                gating_consecutive_weeks += 1
            else:
                gating_consecutive_weeks = 0

            # Compute execution risk metrics
            execution_risk = compute_execution_risk_metrics(
                signals=week_signals,
                liquidity_threshold_millions=self.thresholds.liquidity_threshold_millions,
                gap_threshold=self.thresholds.gap_atr_threshold,
                slippage_budget_bps=self.thresholds.slippage_budget_bps,
            )

            # Compute backtest metrics if in backtest mode
            backtest = None
            if self.backtest_mode and 'actual_return' in week_signals.columns:
                # Compute weekly return
                weights = week_signals['final_weight']
                returns = week_signals['actual_return']
                weekly_return = (weights * returns).sum()

                # Long/short breakdown
                long_mask = weights > 0
                short_mask = weights < 0
                long_return = (weights[long_mask] * returns[long_mask]).sum() if long_mask.any() else 0
                short_return = (weights[short_mask] * returns[short_mask]).sum() if short_mask.any() else 0

                # Update cumulative
                trailing_returns = pd.concat([trailing_returns, pd.Series([weekly_return])])
                if len(cumulative_returns) == 0:
                    cumulative_returns = pd.Series([1 + weekly_return])
                else:
                    cumulative_returns = pd.concat([
                        cumulative_returns,
                        pd.Series([(1 + cumulative_returns.iloc[-1]) * (1 + weekly_return)])
                    ])

                backtest = compute_backtest_metrics(
                    weekly_return=weekly_return,
                    long_return=long_return,
                    short_return=short_return,
                    cumulative_returns=cumulative_returns,
                    trailing_returns=trailing_returns,
                    tail_percentile=self.thresholds.tail_week_percentile,
                    tail_absolute=self.thresholds.tail_week_absolute,
                )

            # Extract regime features if available
            regime_features = {}
            regime_cols = ['vix_percentile_252d', 'd_vix_percentile_252d']
            for col in regime_cols:
                if col in week_signals.columns:
                    regime_features[col] = float(week_signals[col].iloc[0])

            # Generate warnings
            warnings = self.warning_engine.check_all(
                date=week_str,
                data_integrity=data_integrity,
                signal_distribution=signal_distribution,
                portfolio=portfolio,
                execution_risk=execution_risk,
                backtest=backtest,
                sizing_config=self.sizing_config,
                regime_features=regime_features,
                gating_consecutive_weeks=gating_consecutive_weeks,
            )

            # Create weekly diagnostics
            wd = WeeklyDiagnostics(
                week_monday=week_str,
                data_integrity=data_integrity,
                signal_distribution=signal_distribution,
                portfolio=portfolio,
                execution_risk=execution_risk,
                backtest=backtest,
                warnings=warnings,
            )
            self._weekly_diagnostics.append(wd)

            # Update state for next week
            previous_week_signals = week_signals
            if 'symbol' in week_signals.columns and 'final_weight' in week_signals.columns:
                previous_weights = week_signals.set_index('symbol')['final_weight']

        # Collect all warnings
        all_warnings = []
        for wd in self._weekly_diagnostics:
            all_warnings.extend(wd.warnings)

        # Compute aggregate metrics
        weekly_dicts = [wd.to_dict() for wd in self._weekly_diagnostics]
        aggregate_metrics = compute_aggregate_metrics(weekly_dicts)

        # Build final report
        self._report = SizingDiagnosticReport(
            run_id=self.run_id,
            config_snapshot=self.sizing_config,
            n_weeks=len(weeks),
            date_range={
                'start': str(weeks[0].date()) if weeks else None,
                'end': str(weeks[-1].date()) if weeks else None,
            },
            weekly_diagnostics=self._weekly_diagnostics,
            aggregate_metrics=aggregate_metrics,
            all_warnings=all_warnings,
            summary={
                'models': self.models,
                'backtest_mode': self.backtest_mode,
            },
        )

        return self._report

    def save_report(
        self,
        output_dir: str = "artifacts/diagnostics",
        save_json: bool = True,
        save_csv: bool = True,
        save_md: bool = True,
    ) -> Dict[str, str]:
        """Save diagnostic report to files.

        Args:
            output_dir: Output directory.
            save_json: Save JSON report.
            save_csv: Save CSV weekly data.
            save_md: Save Markdown summary.

        Returns:
            Dict of file paths saved.
        """
        if self._report is None:
            raise RuntimeError("Must call run() before save_report()")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Save JSON
        if save_json:
            json_path = output_dir / f"sizing_diagnostics_{self.run_id}.json"
            with open(json_path, 'w') as f:
                f.write(self._report.to_json())
            paths['json'] = str(json_path)

        # Save CSV
        if save_csv:
            csv_path = output_dir / f"sizing_diagnostics_weekly_{self.run_id}.csv"
            rows = self._report.to_csv_rows()
            if rows:
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                paths['csv'] = str(csv_path)

        # Save Markdown
        if save_md:
            md_path = output_dir / f"sizing_diagnostics_{self.run_id}.md"
            with open(md_path, 'w') as f:
                f.write(self._report.to_markdown())
            paths['md'] = str(md_path)

        return paths

    def print_summary(self) -> None:
        """Print summary to terminal."""
        if self._report is None:
            raise RuntimeError("Must call run() before print_summary()")

        print("\n" + "=" * 60)
        print("SIZING DIAGNOSTICS SUMMARY")
        print("=" * 60)

        print(f"\nRun ID: {self._report.run_id}")
        print(f"Period: {self._report.date_range.get('start')} to {self._report.date_range.get('end')}")
        print(f"Weeks: {self._report.n_weeks}")
        print(f"Models: {', '.join(self.models)}")

        # Status
        status = "PASSED" if self._report.passed else "FAILED"
        status_color = "\033[92m" if self._report.passed else "\033[91m"
        print(f"\nStatus: {status_color}{status}\033[0m")

        # Warning counts
        print(f"\n--- WARNINGS ---")
        print(f"  ERRORS:   {self._report.n_errors}")
        print(f"  WARNINGS: {self._report.n_warns}")
        print(f"  INFO:     {self._report.n_infos}")

        # Show errors and warnings
        if self._report.n_errors > 0:
            print(f"\n\033[91m--- ERRORS ---\033[0m")
            for w in self._report.all_warnings:
                if w.severity == Severity.ERROR:
                    print(f"  [{w.code}] {w.message}")
                    print(f"    Value: {w.value:.4f} | Threshold: {w.threshold:.4f}")
                    print(f"    Action: {w.suggested_action}")

        if self._report.n_warns > 0:
            print(f"\n\033[93m--- WARNINGS ---\033[0m")
            # Show first 5 warnings
            warn_count = 0
            for w in self._report.all_warnings:
                if w.severity == Severity.WARN:
                    print(f"  [{w.code}] {w.message}")
                    warn_count += 1
                    if warn_count >= 5:
                        remaining = self._report.n_warns - 5
                        if remaining > 0:
                            print(f"  ... and {remaining} more warnings")
                        break

        # Aggregate metrics
        agg = self._report.aggregate_metrics
        if agg:
            print(f"\n--- AGGREGATE METRICS ---")
            print(f"  Mean Gross Exposure: {agg.get('mean_gross_exposure', 0):.2%}")
            print(f"  Mean Turnover:       {agg.get('mean_turnover', 0):.2%}")
            print(f"  Mean Positions:      {agg.get('mean_positions', 0):.1f}")
            print(f"  Mean HHI:            {agg.get('mean_hhi', 0):.4f}")

            if 'total_return' in agg:
                print(f"\n--- BACKTEST METRICS ---")
                print(f"  Total Return:  {agg.get('total_return', 0):.2%}")
                print(f"  Sharpe Ratio:  {agg.get('sharpe_ratio', 0):.2f}")
                print(f"  Max Drawdown:  {agg.get('max_drawdown', 0):.2%}")
                print(f"  Hit Rate:      {agg.get('hit_rate', 0):.2%}")

        print("\n" + "=" * 60)

    def get_warnings_by_week(self) -> Dict[str, List[SizingWarning]]:
        """Get warnings grouped by week.

        Returns:
            Dict mapping week_monday to list of warnings.
        """
        if self._report is None:
            raise RuntimeError("Must call run() first")

        result = {}
        for wd in self._weekly_diagnostics:
            result[wd.week_monday] = wd.warnings
        return result

    def get_latest_week_diagnostics(self) -> Optional[WeeklyDiagnostics]:
        """Get diagnostics for the most recent week.

        Returns:
            WeeklyDiagnostics for latest week, or None.
        """
        if not self._weekly_diagnostics:
            return None
        return self._weekly_diagnostics[-1]


def run_sizing_diagnostics(
    weighted_signals: pd.DataFrame,
    sizing_config: Optional[Dict[str, Any]] = None,
    models: Optional[List[str]] = None,
    output_dir: str = "artifacts/diagnostics",
    backtest_mode: bool = False,
    print_summary: bool = True,
) -> SizingDiagnosticReport:
    """Convenience function to run full sizing diagnostics.

    Args:
        weighted_signals: DataFrame with weighted signals.
        sizing_config: Sizing configuration.
        models: Model keys.
        output_dir: Output directory.
        backtest_mode: Whether running backtest.
        print_summary: Print summary to terminal.

    Returns:
        SizingDiagnosticReport.
    """
    runner = SizingDiagnosticsRunner(
        weighted_signals=weighted_signals,
        sizing_config=sizing_config,
        models=models,
        backtest_mode=backtest_mode,
    )

    report = runner.run()
    runner.save_report(output_dir)

    if print_summary:
        runner.print_summary()

    return report
