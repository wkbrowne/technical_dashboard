"""Warning rule engine for sizing diagnostics.

Implements a rule-based system that generates warnings based on
diagnostic metrics and configurable thresholds.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .schema import (
    Severity,
    SizingWarning,
    DataIntegrityMetrics,
    SignalDistributionMetrics,
    PortfolioMetrics,
    ExecutionRiskMetrics,
    BacktestMetrics,
)
from .thresholds import SizingDiagnosticThresholds


class WarningEngine:
    """Engine that applies warning rules to diagnostic metrics.

    Usage:
        engine = WarningEngine(thresholds)
        warnings = engine.check_all(
            date="2024-01-01",
            data_integrity=data_integrity_metrics,
            portfolio=portfolio_metrics,
            ...
        )
    """

    def __init__(
        self,
        thresholds: SizingDiagnosticThresholds,
        baseline_extreme_rate: Optional[float] = None,
        baseline_turnover: Optional[float] = None,
        baseline_cash_fraction: Optional[float] = None,
    ):
        """Initialize warning engine.

        Args:
            thresholds: Threshold configuration.
            baseline_extreme_rate: Historical baseline for extreme signal rate.
            baseline_turnover: Historical baseline for turnover.
            baseline_cash_fraction: Historical baseline for cash fraction.
        """
        self.thresholds = thresholds
        self.baseline_extreme_rate = baseline_extreme_rate
        self.baseline_turnover = baseline_turnover
        self.baseline_cash_fraction = baseline_cash_fraction

    def check_all(
        self,
        date: Optional[str],
        data_integrity: Optional[DataIntegrityMetrics] = None,
        signal_distribution: Optional[SignalDistributionMetrics] = None,
        portfolio: Optional[PortfolioMetrics] = None,
        execution_risk: Optional[ExecutionRiskMetrics] = None,
        backtest: Optional[BacktestMetrics] = None,
        sizing_config: Optional[Dict[str, Any]] = None,
        regime_features: Optional[Dict[str, float]] = None,
        gating_consecutive_weeks: int = 0,
    ) -> List[SizingWarning]:
        """Run all warning checks.

        Args:
            date: Week date for context.
            data_integrity: Data integrity metrics.
            signal_distribution: Signal distribution metrics.
            portfolio: Portfolio construction metrics.
            execution_risk: Execution risk metrics.
            backtest: Backtest outcome metrics (optional).
            sizing_config: Sizing configuration for context.
            regime_features: Current regime features.
            gating_consecutive_weeks: Consecutive weeks at max gating.

        Returns:
            List of SizingWarning objects.
        """
        warnings = []

        if data_integrity:
            warnings.extend(self._check_data_integrity(date, data_integrity))

        if signal_distribution:
            warnings.extend(self._check_signal_distribution(date, signal_distribution))

        if portfolio:
            warnings.extend(self._check_portfolio(
                date, portfolio, sizing_config, gating_consecutive_weeks
            ))

        if execution_risk:
            warnings.extend(self._check_execution_risk(date, execution_risk))

        if backtest:
            warnings.extend(self._check_backtest(date, backtest))

        # Behavioral risk checks (compound conditions)
        if portfolio and regime_features:
            warnings.extend(self._check_behavioral_risk(
                date, portfolio, regime_features
            ))

        return warnings

    def _check_data_integrity(
        self,
        date: Optional[str],
        metrics: DataIntegrityMetrics,
    ) -> List[SizingWarning]:
        """Check data integrity metrics."""
        warnings = []
        t = self.thresholds

        # Missing predictions
        if metrics.pct_missing_overall > t.missing_preds_error:
            warnings.append(SizingWarning(
                date=date,
                severity=Severity.ERROR,
                code="MISSING_PREDS_HIGH",
                message=f"Missing predictions rate critically high",
                value=metrics.pct_missing_overall,
                threshold=t.missing_preds_error,
                suggested_action="Check prediction pipeline; ensure all symbols have predictions",
            ))
        elif metrics.pct_missing_overall > t.missing_preds_warn:
            warnings.append(SizingWarning(
                date=date,
                severity=Severity.WARN,
                code="MISSING_PREDS",
                message=f"Missing predictions above threshold",
                value=metrics.pct_missing_overall,
                threshold=t.missing_preds_warn,
                suggested_action="Review prediction coverage; check for symbol filter issues",
            ))

        # NaN predictions
        if metrics.pct_nan_overall > t.nan_preds_error:
            warnings.append(SizingWarning(
                date=date,
                severity=Severity.ERROR,
                code="NAN_PREDS_HIGH",
                message=f"NaN prediction rate critically high",
                value=metrics.pct_nan_overall,
                threshold=t.nan_preds_error,
                suggested_action="Check model inference; likely feature data issues",
            ))
        elif metrics.pct_nan_overall > t.nan_preds_warn:
            warnings.append(SizingWarning(
                date=date,
                severity=Severity.WARN,
                code="NAN_PREDS",
                message=f"NaN predictions above threshold",
                value=metrics.pct_nan_overall,
                threshold=t.nan_preds_warn,
                suggested_action="Review model input features for NaN propagation",
            ))

        # Duplicate symbol-dates
        if metrics.n_duplicate_symbol_date > 0:
            warnings.append(SizingWarning(
                date=date,
                severity=Severity.ERROR,
                code="DUPLICATE_SYMBOL_DATE",
                message=f"Found {metrics.n_duplicate_symbol_date} duplicate (symbol, date) pairs",
                value=float(metrics.n_duplicate_symbol_date),
                threshold=0.0,
                suggested_action="Check prediction deduplication; ensure unique (symbol, date) keys",
            ))

        # Missing weeks
        if metrics.n_missing_weeks >= t.missing_weeks_error:
            warnings.append(SizingWarning(
                date=date,
                severity=Severity.ERROR,
                code="MISSING_WEEKS_HIGH",
                message=f"Multiple consecutive weeks missing",
                value=float(metrics.n_missing_weeks),
                threshold=float(t.missing_weeks_error),
                suggested_action="Check data pipeline; significant gap in prediction timeline",
            ))
        elif metrics.n_missing_weeks >= t.missing_weeks_warn:
            warnings.append(SizingWarning(
                date=date,
                severity=Severity.WARN,
                code="MISSING_WEEKS",
                message=f"Missing week(s) detected",
                value=float(metrics.n_missing_weeks),
                threshold=float(t.missing_weeks_warn),
                suggested_action="Review prediction schedule; check for holiday/market closures",
            ))

        # Stale predictions
        if metrics.has_stale_predictions:
            if metrics.stale_days >= t.stale_preds_error_days:
                warnings.append(SizingWarning(
                    date=date,
                    severity=Severity.ERROR,
                    code="STALE_PREDS_HIGH",
                    message=f"Predictions are {metrics.stale_days} days old",
                    value=float(metrics.stale_days),
                    threshold=float(t.stale_preds_error_days),
                    suggested_action="Regenerate predictions immediately; using outdated signals",
                ))
            elif metrics.stale_days >= t.stale_preds_warn_days:
                warnings.append(SizingWarning(
                    date=date,
                    severity=Severity.WARN,
                    code="STALE_PREDS",
                    message=f"Predictions are {metrics.stale_days} days old",
                    value=float(metrics.stale_days),
                    threshold=float(t.stale_preds_warn_days),
                    suggested_action="Consider refreshing predictions before trading",
                ))

        return warnings

    def _check_signal_distribution(
        self,
        date: Optional[str],
        metrics: SignalDistributionMetrics,
    ) -> List[SizingWarning]:
        """Check signal distribution metrics."""
        warnings = []
        t = self.thresholds

        # PSI drift
        for model, psi in metrics.psi_by_model.items():
            if psi > t.psi_error:
                warnings.append(SizingWarning(
                    date=date,
                    severity=Severity.ERROR,
                    code=f"PSI_HIGH_{model.upper()}",
                    message=f"PSI for {model} indicates major distribution shift",
                    value=psi,
                    threshold=t.psi_error,
                    suggested_action=f"Investigate regime change; consider retraining {model}",
                ))
            elif psi > t.psi_warn:
                warnings.append(SizingWarning(
                    date=date,
                    severity=Severity.WARN,
                    code=f"PSI_ELEVATED_{model.upper()}",
                    message=f"PSI for {model} shows moderate distribution shift",
                    value=psi,
                    threshold=t.psi_warn,
                    suggested_action=f"Monitor {model} performance; may need recalibration",
                ))

        # KS drift
        for model, ks in metrics.ks_by_model.items():
            if ks > t.ks_error:
                warnings.append(SizingWarning(
                    date=date,
                    severity=Severity.ERROR,
                    code=f"KS_HIGH_{model.upper()}",
                    message=f"KS statistic for {model} indicates significant shift",
                    value=ks,
                    threshold=t.ks_error,
                    suggested_action=f"Distribution for {model} has shifted significantly",
                ))
            elif ks > t.ks_warn:
                warnings.append(SizingWarning(
                    date=date,
                    severity=Severity.WARN,
                    code=f"KS_ELEVATED_{model.upper()}",
                    message=f"KS statistic for {model} elevated",
                    value=ks,
                    threshold=t.ks_warn,
                    suggested_action=f"Monitor {model} prediction distribution",
                ))

        # Extreme signal rate
        if metrics.extreme_signal_rate > t.extreme_signal_rate_warn:
            warnings.append(SizingWarning(
                date=date,
                severity=Severity.WARN,
                code="EXTREME_SIGNAL_RATE",
                message=f"High fraction of extreme predictions",
                value=metrics.extreme_signal_rate,
                threshold=t.extreme_signal_rate_warn,
                suggested_action="Check model calibration; many predictions near 0 or 1",
            ))

        # Extreme signal rate spike vs baseline
        if self.baseline_extreme_rate is not None:
            spike_ratio = metrics.extreme_signal_rate / (self.baseline_extreme_rate + 1e-10)
            if spike_ratio > t.extreme_signal_rate_spike_warn:
                warnings.append(SizingWarning(
                    date=date,
                    severity=Severity.WARN,
                    code="EXTREME_SIGNAL_SPIKE",
                    message=f"Extreme signal rate {spike_ratio:.1f}x baseline",
                    value=spike_ratio,
                    threshold=t.extreme_signal_rate_spike_warn,
                    suggested_action="Unusual concentration of extreme predictions",
                ))

        # Rank correlation
        if metrics.rank_correlation_vs_prev is not None:
            if metrics.rank_correlation_vs_prev < t.rank_correlation_error:
                warnings.append(SizingWarning(
                    date=date,
                    severity=Severity.ERROR,
                    code="RANK_INSTABILITY_HIGH",
                    message=f"Rank correlation with previous week very low",
                    value=metrics.rank_correlation_vs_prev,
                    threshold=t.rank_correlation_error,
                    suggested_action="Major regime shift or data issue; rankings unstable",
                ))
            elif metrics.rank_correlation_vs_prev < t.rank_correlation_warn:
                warnings.append(SizingWarning(
                    date=date,
                    severity=Severity.WARN,
                    code="RANK_INSTABILITY",
                    message=f"Rank correlation with previous week low",
                    value=metrics.rank_correlation_vs_prev,
                    threshold=t.rank_correlation_warn,
                    suggested_action="Rankings less stable than usual; monitor for regime shift",
                ))

        return warnings

    def _check_portfolio(
        self,
        date: Optional[str],
        metrics: PortfolioMetrics,
        sizing_config: Optional[Dict[str, Any]],
        gating_consecutive_weeks: int,
    ) -> List[SizingWarning]:
        """Check portfolio construction metrics."""
        warnings = []
        t = self.thresholds

        # Gross exposure violation
        max_gross = sizing_config.get('max_gross_exposure', 1.0) if sizing_config else 1.0
        if metrics.gross_exposure > max_gross + t.gross_exposure_tolerance:
            warnings.append(SizingWarning(
                date=date,
                severity=Severity.ERROR,
                code="GROSS_EXPOSURE_EXCEEDED",
                message=f"Gross exposure exceeds configured maximum",
                value=metrics.gross_exposure,
                threshold=max_gross,
                suggested_action="Portfolio constraint violation; check sizing logic",
            ))

        # Max weight violation
        max_weight = sizing_config.get('max_weight_per_name', 0.10) if sizing_config else 0.10
        if metrics.max_weight > max_weight + t.max_weight_tolerance:
            warnings.append(SizingWarning(
                date=date,
                severity=Severity.ERROR,
                code="MAX_WEIGHT_EXCEEDED",
                message=f"Position exceeds max weight cap",
                value=metrics.max_weight,
                threshold=max_weight,
                suggested_action="Weight constraint violation; check position sizing",
            ))

        # HHI concentration
        if metrics.hhi > t.hhi_error:
            warnings.append(SizingWarning(
                date=date,
                severity=Severity.ERROR,
                code="HHI_HIGH",
                message=f"Portfolio highly concentrated (HHI)",
                value=metrics.hhi,
                threshold=t.hhi_error,
                suggested_action="Reduce concentration; diversify positions",
            ))
        elif metrics.hhi > t.hhi_warn:
            warnings.append(SizingWarning(
                date=date,
                severity=Severity.WARN,
                code="HHI_ELEVATED",
                message=f"Portfolio moderately concentrated",
                value=metrics.hhi,
                threshold=t.hhi_warn,
                suggested_action="Consider diversification",
            ))

        # Top5 concentration
        if metrics.top5_concentration > t.top5_concentration_error:
            warnings.append(SizingWarning(
                date=date,
                severity=Severity.ERROR,
                code="TOP5_CONCENTRATION_HIGH",
                message=f"Top 5 positions dominate portfolio",
                value=metrics.top5_concentration,
                threshold=t.top5_concentration_error,
                suggested_action="Excessive concentration in top positions",
            ))
        elif metrics.top5_concentration > t.top5_concentration_warn:
            warnings.append(SizingWarning(
                date=date,
                severity=Severity.WARN,
                code="TOP5_CONCENTRATION",
                message=f"Top 5 positions are significant fraction",
                value=metrics.top5_concentration,
                threshold=t.top5_concentration_warn,
                suggested_action="Monitor concentration risk",
            ))

        # Turnover
        if metrics.turnover_total > t.turnover_error:
            warnings.append(SizingWarning(
                date=date,
                severity=Severity.ERROR,
                code="TURNOVER_HIGH",
                message=f"Turnover critically high",
                value=metrics.turnover_total,
                threshold=t.turnover_error,
                suggested_action="Review turnover penalty; trading costs may be excessive",
            ))
        elif metrics.turnover_total > t.turnover_warn:
            warnings.append(SizingWarning(
                date=date,
                severity=Severity.WARN,
                code="TURNOVER_ELEVATED",
                message=f"Turnover above threshold",
                value=metrics.turnover_total,
                threshold=t.turnover_warn,
                suggested_action="Monitor trading costs",
            ))

        # Turnover spike vs baseline
        if self.baseline_turnover is not None and self.baseline_turnover > 0:
            spike_ratio = metrics.turnover_total / self.baseline_turnover
            if spike_ratio > t.turnover_spike_ratio:
                warnings.append(SizingWarning(
                    date=date,
                    severity=Severity.WARN,
                    code="TURNOVER_SPIKE",
                    message=f"Turnover {spike_ratio:.1f}x historical average",
                    value=spike_ratio,
                    threshold=t.turnover_spike_ratio,
                    suggested_action="Unusually high turnover this week",
                ))

        # Position counts
        if metrics.n_positions < t.min_positions_warn and metrics.n_positions > 0:
            warnings.append(SizingWarning(
                date=date,
                severity=Severity.WARN,
                code="FEW_POSITIONS",
                message=f"Very few positions",
                value=float(metrics.n_positions),
                threshold=float(t.min_positions_warn),
                suggested_action="Check if model is finding sufficient opportunities",
            ))

        # Short dominance
        if metrics.n_shorts > 0 and metrics.gross_exposure > 0:
            short_frac = metrics.n_shorts / (metrics.n_longs + metrics.n_shorts + 1e-10)
            if short_frac > t.short_dominance_warn:
                warnings.append(SizingWarning(
                    date=date,
                    severity=Severity.WARN,
                    code="SHORT_DOMINANCE",
                    message=f"Shorts dominate portfolio",
                    value=short_frac,
                    threshold=t.short_dominance_warn,
                    suggested_action="Review short model signals; unusual bearish positioning",
                ))

        # Cash fraction
        if metrics.cash_fraction < t.cash_low_warn:
            warnings.append(SizingWarning(
                date=date,
                severity=Severity.WARN,
                code="CASH_LOW",
                message=f"Cash fraction unusually low",
                value=metrics.cash_fraction,
                threshold=t.cash_low_warn,
                suggested_action="Very high exposure; review risk tolerance",
            ))
        elif metrics.cash_fraction > t.cash_high_warn:
            warnings.append(SizingWarning(
                date=date,
                severity=Severity.WARN,
                code="CASH_HIGH",
                message=f"Cash fraction unusually high",
                value=metrics.cash_fraction,
                threshold=t.cash_high_warn,
                suggested_action="Very low exposure; check if intended",
            ))

        # Gating consecutive weeks
        if gating_consecutive_weeks >= t.gating_max_consecutive_warn:
            warnings.append(SizingWarning(
                date=date,
                severity=Severity.WARN,
                code="GATING_PROLONGED",
                message=f"Regime gating active for {gating_consecutive_weeks} consecutive weeks",
                value=float(gating_consecutive_weeks),
                threshold=float(t.gating_max_consecutive_warn),
                suggested_action="Prolonged risk-off regime; review gating thresholds",
            ))

        return warnings

    def _check_execution_risk(
        self,
        date: Optional[str],
        metrics: ExecutionRiskMetrics,
    ) -> List[SizingWarning]:
        """Check execution risk metrics."""
        warnings = []
        t = self.thresholds

        # Liquidity
        if metrics.pct_below_liquidity_threshold > t.pct_below_liquidity_error:
            warnings.append(SizingWarning(
                date=date,
                severity=Severity.ERROR,
                code="LIQUIDITY_POOR_HIGH",
                message=f"Many positions below liquidity threshold",
                value=metrics.pct_below_liquidity_threshold,
                threshold=t.pct_below_liquidity_error,
                suggested_action="Significant illiquidity risk; review position sizing",
            ))
        elif metrics.pct_below_liquidity_threshold > t.pct_below_liquidity_warn:
            warnings.append(SizingWarning(
                date=date,
                severity=Severity.WARN,
                code="LIQUIDITY_POOR",
                message=f"Some positions below liquidity threshold",
                value=metrics.pct_below_liquidity_threshold,
                threshold=t.pct_below_liquidity_warn,
                suggested_action="Monitor execution quality for illiquid names",
            ))

        # Gap risk
        if metrics.pct_above_gap_threshold > t.pct_above_gap_error:
            warnings.append(SizingWarning(
                date=date,
                severity=Severity.ERROR,
                code="GAP_RISK_HIGH",
                message=f"Many positions with high gap risk",
                value=metrics.pct_above_gap_threshold,
                threshold=t.pct_above_gap_error,
                suggested_action="Significant overnight gap risk; reduce position sizes",
            ))
        elif metrics.pct_above_gap_threshold > t.pct_above_gap_warn:
            warnings.append(SizingWarning(
                date=date,
                severity=Severity.WARN,
                code="GAP_RISK",
                message=f"Some positions with elevated gap risk",
                value=metrics.pct_above_gap_threshold,
                threshold=t.pct_above_gap_warn,
                suggested_action="Monitor overnight risk for gap-prone names",
            ))

        # Slippage utilization
        if metrics.slippage_budget_utilization > t.slippage_utilization_error:
            warnings.append(SizingWarning(
                date=date,
                severity=Severity.ERROR,
                code="SLIPPAGE_OVER_BUDGET",
                message=f"Estimated slippage exceeds budget",
                value=metrics.slippage_budget_utilization,
                threshold=t.slippage_utilization_error,
                suggested_action="Reduce position sizes or improve execution",
            ))
        elif metrics.slippage_budget_utilization > t.slippage_utilization_warn:
            warnings.append(SizingWarning(
                date=date,
                severity=Severity.WARN,
                code="SLIPPAGE_HIGH",
                message=f"Slippage approaching budget",
                value=metrics.slippage_budget_utilization,
                threshold=t.slippage_utilization_warn,
                suggested_action="Monitor execution costs",
            ))

        return warnings

    def _check_backtest(
        self,
        date: Optional[str],
        metrics: BacktestMetrics,
    ) -> List[SizingWarning]:
        """Check backtest outcome metrics."""
        warnings = []
        t = self.thresholds

        # Tail week
        if metrics.is_tail_week:
            warnings.append(SizingWarning(
                date=date,
                severity=Severity.WARN,
                code="TAIL_WEEK",
                message=f"Return below tail threshold",
                value=metrics.weekly_return,
                threshold=metrics.tail_week_threshold,
                suggested_action="Significant drawdown week; review positions",
            ))

        # Rolling Sharpe
        if metrics.rolling_sharpe_13w is not None:
            if metrics.rolling_sharpe_13w < t.sharpe_error:
                warnings.append(SizingWarning(
                    date=date,
                    severity=Severity.ERROR,
                    code="SHARPE_NEGATIVE",
                    message=f"13-week rolling Sharpe deeply negative",
                    value=metrics.rolling_sharpe_13w,
                    threshold=t.sharpe_error,
                    suggested_action="Strategy underperforming significantly; review model",
                ))
            elif metrics.rolling_sharpe_13w < t.sharpe_warn:
                warnings.append(SizingWarning(
                    date=date,
                    severity=Severity.WARN,
                    code="SHARPE_LOW",
                    message=f"13-week rolling Sharpe below zero",
                    value=metrics.rolling_sharpe_13w,
                    threshold=t.sharpe_warn,
                    suggested_action="Performance deteriorating; monitor closely",
                ))

        # Drawdown
        if metrics.rolling_drawdown < t.drawdown_error:
            warnings.append(SizingWarning(
                date=date,
                severity=Severity.ERROR,
                code="DRAWDOWN_SEVERE",
                message=f"Drawdown exceeds error threshold",
                value=metrics.rolling_drawdown,
                threshold=t.drawdown_error,
                suggested_action="Severe drawdown; consider reducing exposure",
            ))
        elif metrics.rolling_drawdown < t.drawdown_warn:
            warnings.append(SizingWarning(
                date=date,
                severity=Severity.WARN,
                code="DRAWDOWN_ELEVATED",
                message=f"Drawdown exceeds warning threshold",
                value=metrics.rolling_drawdown,
                threshold=t.drawdown_warn,
                suggested_action="Notable drawdown; monitor recovery",
            ))

        return warnings

    def _check_behavioral_risk(
        self,
        date: Optional[str],
        portfolio: PortfolioMetrics,
        regime_features: Dict[str, float],
    ) -> List[SizingWarning]:
        """Check compound behavioral risk conditions."""
        warnings = []
        t = self.thresholds

        # High exposure + high volatility regime
        vix_pct = regime_features.get('vix_percentile_252d') or regime_features.get('d_vix_percentile_252d', 0)
        if portfolio.gross_exposure > t.high_exposure_threshold and vix_pct > t.high_vol_vix_percentile:
            warnings.append(SizingWarning(
                date=date,
                severity=Severity.WARN,
                code="HIGH_EXPOSURE_HIGH_VOL",
                message=f"High exposure during elevated volatility regime",
                value=portfolio.gross_exposure,
                threshold=t.high_exposure_threshold,
                suggested_action="Consider reducing exposure given VIX level",
            ))

        # High concentration + high turnover
        if (portfolio.hhi > t.concentration_turnover_warn_hhi and
            portfolio.turnover_total > t.concentration_turnover_warn_turnover):
            warnings.append(SizingWarning(
                date=date,
                severity=Severity.WARN,
                code="FRAGILE_PORTFOLIO",
                message=f"High concentration with high turnover",
                value=portfolio.hhi,
                threshold=t.concentration_turnover_warn_hhi,
                suggested_action="Portfolio may be unstable; concentrated and churning",
            ))

        return warnings
