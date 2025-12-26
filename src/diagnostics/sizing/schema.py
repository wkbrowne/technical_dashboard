"""Schema definitions for sizing diagnostics.

Defines dataclasses for structured diagnostic outputs.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json


class Severity(str, Enum):
    """Severity level for diagnostic warnings."""
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"

    def __str__(self) -> str:
        return self.value


@dataclass
class SizingWarning:
    """A single sizing diagnostic warning.

    Attributes:
        date: Date when warning occurred (or None for aggregate).
        severity: INFO, WARN, or ERROR.
        code: Short identifier for the warning type.
        message: Human-readable description.
        value: Actual observed value.
        threshold: Threshold that was violated.
        suggested_action: Concrete recommendation.
    """
    date: Optional[str]
    severity: Severity
    code: str
    message: str
    value: float
    threshold: float
    suggested_action: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': self.date,
            'severity': str(self.severity),
            'code': self.code,
            'message': self.message,
            'value': self.value,
            'threshold': self.threshold,
            'suggested_action': self.suggested_action,
        }


@dataclass
class DataIntegrityMetrics:
    """Data integrity metrics for predictions.

    Attributes:
        pct_missing_overall: % of predictions missing across all models.
        pct_missing_by_model: % missing per model.
        pct_nan_overall: % of NaN predictions across all models.
        pct_nan_by_model: % NaN per model.
        n_duplicate_symbol_date: Count of duplicate (symbol, date) pairs.
        duplicate_symbols: List of symbols with duplicates.
        n_missing_weeks: Count of missing expected weeks.
        missing_weeks: List of missing week dates.
        date_gaps: List of unexpected date gaps.
        has_stale_predictions: Whether predictions are stale vs rebalance date.
        stale_days: Number of days predictions are stale.
    """
    pct_missing_overall: float = 0.0
    pct_missing_by_model: Dict[str, float] = field(default_factory=dict)
    pct_nan_overall: float = 0.0
    pct_nan_by_model: Dict[str, float] = field(default_factory=dict)
    n_duplicate_symbol_date: int = 0
    duplicate_symbols: List[str] = field(default_factory=list)
    n_missing_weeks: int = 0
    missing_weeks: List[str] = field(default_factory=list)
    date_gaps: List[Dict[str, Any]] = field(default_factory=list)
    has_stale_predictions: bool = False
    stale_days: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SignalDistributionMetrics:
    """Signal distribution and drift metrics.

    Attributes:
        prob_stats: Statistics for probability distributions by model.
        edge_stats: Statistics for edge distributions by model.
        psi_by_model: PSI values vs baseline by model.
        ks_by_model: KS statistic vs baseline by model.
        extreme_signal_rate: Fraction with p > 0.9 or p < 0.1.
        extreme_signal_rate_by_model: Per-model extreme rates.
        rank_correlation_vs_prev: Spearman correlation vs previous week.
    """
    prob_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    edge_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    psi_by_model: Dict[str, float] = field(default_factory=dict)
    ks_by_model: Dict[str, float] = field(default_factory=dict)
    extreme_signal_rate: float = 0.0
    extreme_signal_rate_by_model: Dict[str, float] = field(default_factory=dict)
    rank_correlation_vs_prev: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PortfolioMetrics:
    """Portfolio construction metrics.

    Attributes:
        gross_exposure: Total absolute exposure.
        net_exposure: Long - short exposure.
        cash_fraction: 1 - gross_exposure.
        n_positions: Total positions.
        n_longs: Long positions count.
        n_shorts: Short positions count.
        top1_concentration: Top 1 weight / gross exposure.
        top5_concentration: Top 5 weights / gross exposure.
        top10_concentration: Top 10 weights / gross exposure.
        hhi: Herfindahl-Hirschman Index of weights.
        max_weight: Maximum single-name weight.
        max_weight_pct_of_cap: Max weight as % of configured cap.
        turnover_total: Total turnover (sum of |delta_w|).
        turnover_one_way: One-way turnover (buys or sells).
        turnover_vs_penalty: Turnover relative to penalty threshold.
        model_counts: Position counts by contributing model.
        model_gross_exposure: Gross exposure by model.
        multi_model_conflict_count: Symbols where multiple models fired.
        multi_model_resolution: How conflicts were resolved.
        gating_multiplier: Applied gating multiplier.
        gating_modes_disabled: Which modes were disabled.
        gating_reduced_exposure: Whether gating reduced exposure.
        gating_trades_removed: Number of trades removed by gating.
        gating_exposure_reduced: Amount of exposure reduced by gating.
    """
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    cash_fraction: float = 1.0
    n_positions: int = 0
    n_longs: int = 0
    n_shorts: int = 0
    top1_concentration: float = 0.0
    top5_concentration: float = 0.0
    top10_concentration: float = 0.0
    hhi: float = 0.0
    max_weight: float = 0.0
    max_weight_pct_of_cap: float = 0.0
    turnover_total: float = 0.0
    turnover_one_way: float = 0.0
    turnover_vs_penalty: float = 0.0
    model_counts: Dict[str, int] = field(default_factory=dict)
    model_gross_exposure: Dict[str, float] = field(default_factory=dict)
    multi_model_conflict_count: int = 0
    multi_model_resolution: Dict[str, int] = field(default_factory=dict)
    gating_multiplier: float = 1.0
    gating_modes_disabled: List[str] = field(default_factory=list)
    gating_reduced_exposure: bool = False
    gating_trades_removed: int = 0
    gating_exposure_reduced: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionRiskMetrics:
    """Execution risk proxy metrics.

    Attributes:
        median_rdollar_vol: Median dollar volume for selected names.
        pct_below_liquidity_threshold: % of positions below threshold.
        liquidity_threshold_used: The threshold value.
        estimated_slippage: Estimated slippage in bps.
        slippage_budget_utilization: Slippage as % of budget.
        median_gap_atr_ratio: Median gap-to-ATR ratio.
        pct_above_gap_threshold: % above gap threshold.
        gap_threshold_used: The gap threshold value.
    """
    median_rdollar_vol: Optional[float] = None
    pct_below_liquidity_threshold: float = 0.0
    liquidity_threshold_used: float = 0.0
    estimated_slippage: float = 0.0
    slippage_budget_utilization: float = 0.0
    median_gap_atr_ratio: Optional[float] = None
    pct_above_gap_threshold: float = 0.0
    gap_threshold_used: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestMetrics:
    """Backtest outcome metrics (only populated in backtest mode).

    Attributes:
        weekly_return: Return for this week.
        weekly_return_long: Long sleeve contribution.
        weekly_return_short: Short sleeve contribution.
        cumulative_return: Cumulative return to date.
        rolling_drawdown: Current drawdown from peak.
        max_drawdown_trailing: Max drawdown over trailing window.
        rolling_sharpe_13w: 13-week rolling Sharpe.
        rolling_sharpe_26w: 26-week rolling Sharpe.
        is_tail_week: Whether this is a tail week (< p5 or < -X%).
        tail_week_threshold: The threshold used for tail detection.
    """
    weekly_return: float = 0.0
    weekly_return_long: float = 0.0
    weekly_return_short: float = 0.0
    cumulative_return: float = 0.0
    rolling_drawdown: float = 0.0
    max_drawdown_trailing: float = 0.0
    rolling_sharpe_13w: Optional[float] = None
    rolling_sharpe_26w: Optional[float] = None
    is_tail_week: bool = False
    tail_week_threshold: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WeeklyDiagnostics:
    """Complete diagnostics for a single week.

    Attributes:
        week_monday: Monday date of the week.
        data_integrity: Data integrity metrics.
        signal_distribution: Signal distribution metrics.
        portfolio: Portfolio construction metrics.
        execution_risk: Execution risk metrics.
        backtest: Backtest metrics (if in backtest mode).
        warnings: List of warnings for this week.
    """
    week_monday: str
    data_integrity: DataIntegrityMetrics = field(default_factory=DataIntegrityMetrics)
    signal_distribution: SignalDistributionMetrics = field(default_factory=SignalDistributionMetrics)
    portfolio: PortfolioMetrics = field(default_factory=PortfolioMetrics)
    execution_risk: ExecutionRiskMetrics = field(default_factory=ExecutionRiskMetrics)
    backtest: Optional[BacktestMetrics] = None
    warnings: List[SizingWarning] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'week_monday': self.week_monday,
            'data_integrity': self.data_integrity.to_dict(),
            'signal_distribution': self.signal_distribution.to_dict(),
            'portfolio': self.portfolio.to_dict(),
            'execution_risk': self.execution_risk.to_dict(),
            'warnings': [w.to_dict() for w in self.warnings],
        }
        if self.backtest is not None:
            d['backtest'] = self.backtest.to_dict()
        return d


@dataclass
class SizingDiagnosticReport:
    """Complete sizing diagnostic report.

    Attributes:
        run_id: Unique identifier for this run.
        timestamp: When the report was generated.
        config_snapshot: Snapshot of sizing configuration.
        n_weeks: Number of weeks analyzed.
        date_range: Start and end dates.
        weekly_diagnostics: Per-week diagnostic data.
        aggregate_metrics: Aggregated metrics across all weeks.
        all_warnings: All warnings across all weeks.
        summary: Summary statistics.
        passed: Whether all checks passed (no ERROR severity).
    """
    run_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    n_weeks: int = 0
    date_range: Dict[str, str] = field(default_factory=dict)
    weekly_diagnostics: List[WeeklyDiagnostics] = field(default_factory=list)
    aggregate_metrics: Dict[str, Any] = field(default_factory=dict)
    all_warnings: List[SizingWarning] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True if no ERROR severity warnings."""
        return not any(w.severity == Severity.ERROR for w in self.all_warnings)

    @property
    def n_errors(self) -> int:
        return sum(1 for w in self.all_warnings if w.severity == Severity.ERROR)

    @property
    def n_warns(self) -> int:
        return sum(1 for w in self.all_warnings if w.severity == Severity.WARN)

    @property
    def n_infos(self) -> int:
        return sum(1 for w in self.all_warnings if w.severity == Severity.INFO)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'run_id': self.run_id,
            'timestamp': self.timestamp,
            'config_snapshot': self.config_snapshot,
            'n_weeks': self.n_weeks,
            'date_range': self.date_range,
            'passed': self.passed,
            'summary': {
                'n_errors': self.n_errors,
                'n_warns': self.n_warns,
                'n_infos': self.n_infos,
                'total_warnings': len(self.all_warnings),
                **self.summary,
            },
            'aggregate_metrics': self.aggregate_metrics,
            'all_warnings': [w.to_dict() for w in self.all_warnings],
            'weekly_diagnostics': [wd.to_dict() for wd in self.weekly_diagnostics],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_csv_rows(self) -> List[Dict[str, Any]]:
        """Convert to flat rows for CSV export."""
        rows = []
        for wd in self.weekly_diagnostics:
            row = {
                'run_id': self.run_id,
                'week_monday': wd.week_monday,
                # Data integrity
                'pct_missing_overall': wd.data_integrity.pct_missing_overall,
                'pct_nan_overall': wd.data_integrity.pct_nan_overall,
                'n_duplicate_symbol_date': wd.data_integrity.n_duplicate_symbol_date,
                # Portfolio
                'gross_exposure': wd.portfolio.gross_exposure,
                'net_exposure': wd.portfolio.net_exposure,
                'n_positions': wd.portfolio.n_positions,
                'n_longs': wd.portfolio.n_longs,
                'n_shorts': wd.portfolio.n_shorts,
                'top5_concentration': wd.portfolio.top5_concentration,
                'hhi': wd.portfolio.hhi,
                'max_weight': wd.portfolio.max_weight,
                'turnover_total': wd.portfolio.turnover_total,
                'gating_multiplier': wd.portfolio.gating_multiplier,
                'gating_reduced_exposure': wd.portfolio.gating_reduced_exposure,
                # Execution risk
                'median_rdollar_vol': wd.execution_risk.median_rdollar_vol,
                'pct_below_liquidity_threshold': wd.execution_risk.pct_below_liquidity_threshold,
                # Signal
                'extreme_signal_rate': wd.signal_distribution.extreme_signal_rate,
                'rank_correlation_vs_prev': wd.signal_distribution.rank_correlation_vs_prev,
                # Warning counts
                'n_warnings': len(wd.warnings),
                'n_errors': sum(1 for w in wd.warnings if w.severity == Severity.ERROR),
                'n_warns': sum(1 for w in wd.warnings if w.severity == Severity.WARN),
            }
            # Add backtest metrics if available
            if wd.backtest is not None:
                row.update({
                    'weekly_return': wd.backtest.weekly_return,
                    'weekly_return_long': wd.backtest.weekly_return_long,
                    'weekly_return_short': wd.backtest.weekly_return_short,
                    'cumulative_return': wd.backtest.cumulative_return,
                    'rolling_drawdown': wd.backtest.rolling_drawdown,
                    'rolling_sharpe_13w': wd.backtest.rolling_sharpe_13w,
                    'is_tail_week': wd.backtest.is_tail_week,
                })
            rows.append(row)
        return rows

    def to_markdown(self) -> str:
        """Generate human-readable Markdown summary."""
        lines = [
            f"# Sizing Diagnostics Report",
            f"",
            f"**Run ID:** {self.run_id}",
            f"**Generated:** {self.timestamp}",
            f"**Period:** {self.date_range.get('start', 'N/A')} to {self.date_range.get('end', 'N/A')}",
            f"**Weeks Analyzed:** {self.n_weeks}",
            f"",
            f"## Summary",
            f"",
            f"| Status | {'PASSED' if self.passed else 'FAILED'} |",
            f"|--------|--------|",
            f"| Errors | {self.n_errors} |",
            f"| Warnings | {self.n_warns} |",
            f"| Info | {self.n_infos} |",
            f"",
        ]

        # Add aggregate metrics
        if self.aggregate_metrics:
            lines.extend([
                "## Aggregate Metrics",
                "",
            ])
            for key, value in self.aggregate_metrics.items():
                if isinstance(value, float):
                    lines.append(f"- **{key}:** {value:.4f}")
                else:
                    lines.append(f"- **{key}:** {value}")
            lines.append("")

        # Add warnings by severity
        if self.all_warnings:
            lines.extend([
                "## Warnings",
                "",
            ])

            # Group by severity
            for severity in [Severity.ERROR, Severity.WARN, Severity.INFO]:
                warnings = [w for w in self.all_warnings if w.severity == severity]
                if warnings:
                    lines.append(f"### {severity.value} ({len(warnings)})")
                    lines.append("")
                    for w in warnings[:10]:  # Limit to first 10
                        lines.append(f"- **[{w.code}]** {w.message}")
                        lines.append(f"  - Date: {w.date or 'aggregate'}")
                        lines.append(f"  - Value: {w.value:.4f} (threshold: {w.threshold:.4f})")
                        lines.append(f"  - Action: {w.suggested_action}")
                        lines.append("")
                    if len(warnings) > 10:
                        lines.append(f"*... and {len(warnings) - 10} more*")
                        lines.append("")

        # Add recent weekly summary
        lines.extend([
            "## Recent Weeks (Last 5)",
            "",
            "| Week | Gross Exp | Positions | Turnover | Warnings |",
            "|------|-----------|-----------|----------|----------|",
        ])
        for wd in self.weekly_diagnostics[-5:]:
            lines.append(
                f"| {wd.week_monday} | {wd.portfolio.gross_exposure:.2%} | "
                f"{wd.portfolio.n_positions} | {wd.portfolio.turnover_total:.2%} | "
                f"{len(wd.warnings)} |"
            )

        return "\n".join(lines)
