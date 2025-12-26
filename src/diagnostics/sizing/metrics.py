"""Core metric computation functions for sizing diagnostics.

Provides functions to compute each category of diagnostic metrics
from weighted signals and related data.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from .schema import (
    DataIntegrityMetrics,
    SignalDistributionMetrics,
    PortfolioMetrics,
    ExecutionRiskMetrics,
    BacktestMetrics,
)
from .baselines import BaselineManager


def compute_data_integrity_metrics(
    signals: pd.DataFrame,
    models: List[str],
    expected_symbols: Optional[List[str]] = None,
    expected_weeks: Optional[List[pd.Timestamp]] = None,
    prediction_timestamp: Optional[pd.Timestamp] = None,
    rebalance_date: Optional[pd.Timestamp] = None,
) -> DataIntegrityMetrics:
    """Compute data integrity metrics for a week's signals.

    Args:
        signals: DataFrame with prediction columns.
        models: List of model keys.
        expected_symbols: Expected symbols for completeness check.
        expected_weeks: Expected week dates for continuity check.
        prediction_timestamp: When predictions were generated.
        rebalance_date: When rebalance will occur.

    Returns:
        DataIntegrityMetrics object.
    """
    metrics = DataIntegrityMetrics()

    # Get probability columns
    prob_cols = [f"p_{m}" for m in models if f"p_{m}" in signals.columns]

    if len(prob_cols) == 0:
        return metrics

    # Missing predictions (symbol coverage)
    if expected_symbols is not None:
        actual_symbols = set(signals['symbol'].unique()) if 'symbol' in signals.columns else set()
        missing = set(expected_symbols) - actual_symbols
        metrics.pct_missing_overall = len(missing) / len(expected_symbols) if expected_symbols else 0

        # Per-model missing (if different models have different coverage)
        for model in models:
            col = f"p_{model}"
            if col in signals.columns:
                n_missing = signals[col].isna().sum()
                metrics.pct_missing_by_model[model] = n_missing / len(signals) if len(signals) > 0 else 0

    # NaN predictions
    for model in models:
        col = f"p_{model}"
        if col in signals.columns:
            nan_rate = signals[col].isna().mean()
            metrics.pct_nan_by_model[model] = float(nan_rate)

    # Overall NaN rate across all models
    if prob_cols:
        total_cells = len(signals) * len(prob_cols)
        nan_cells = sum(signals[col].isna().sum() for col in prob_cols)
        metrics.pct_nan_overall = nan_cells / total_cells if total_cells > 0 else 0

    # Duplicate (symbol, date) detection
    if 'symbol' in signals.columns and ('date' in signals.columns or 'week_monday' in signals.columns):
        date_col = 'week_monday' if 'week_monday' in signals.columns else 'date'
        duplicates = signals.duplicated(subset=['symbol', date_col], keep=False)
        n_dup = duplicates.sum()
        metrics.n_duplicate_symbol_date = int(n_dup)
        if n_dup > 0:
            metrics.duplicate_symbols = signals.loc[duplicates, 'symbol'].unique().tolist()[:10]

    # Missing weeks detection
    if expected_weeks is not None and 'week_monday' in signals.columns:
        actual_weeks = set(pd.to_datetime(signals['week_monday']).unique())
        missing_weeks = set(expected_weeks) - actual_weeks
        metrics.n_missing_weeks = len(missing_weeks)
        metrics.missing_weeks = [str(w.date()) for w in sorted(missing_weeks)][:5]

    # Date gaps (unexpected gaps larger than 7 days between weeks)
    if 'week_monday' in signals.columns:
        weeks = pd.to_datetime(signals['week_monday']).drop_duplicates().sort_values()
        if len(weeks) > 1:
            gaps = weeks.diff().dropna()
            large_gaps = gaps[gaps > pd.Timedelta(days=10)]
            for idx in large_gaps.index:
                metrics.date_gaps.append({
                    'after': str(weeks.loc[idx - 1].date()) if idx > 0 else None,
                    'gap_days': int(gaps.loc[idx].days),
                })

    # Stale predictions check
    if prediction_timestamp is not None and rebalance_date is not None:
        stale_days = (rebalance_date - prediction_timestamp).days
        if stale_days > 0:
            metrics.has_stale_predictions = True
            metrics.stale_days = int(stale_days)

    return metrics


def compute_signal_distribution_metrics(
    signals: pd.DataFrame,
    models: List[str],
    baseline_manager: Optional[BaselineManager] = None,
    previous_week_signals: Optional[pd.DataFrame] = None,
    sizing_intercept: float = 0.5,
    extreme_high: float = 0.90,
    extreme_low: float = 0.10,
) -> SignalDistributionMetrics:
    """Compute signal distribution and drift metrics.

    Args:
        signals: DataFrame with prediction columns.
        models: List of model keys.
        baseline_manager: For PSI/KS comparison.
        previous_week_signals: Previous week for rank correlation.
        sizing_intercept: Threshold for edge calculation.
        extreme_high: Upper extreme threshold.
        extreme_low: Lower extreme threshold.

    Returns:
        SignalDistributionMetrics object.
    """
    metrics = SignalDistributionMetrics()

    # Compute stats for each model
    for model in models:
        prob_col = f"p_{model}"
        if prob_col not in signals.columns:
            continue

        probs = signals[prob_col].dropna()
        if len(probs) == 0:
            continue

        # Probability stats
        metrics.prob_stats[model] = {
            'mean': float(probs.mean()),
            'median': float(probs.median()),
            'std': float(probs.std()),
            'p10': float(np.percentile(probs, 10)),
            'p50': float(np.percentile(probs, 50)),
            'p90': float(np.percentile(probs, 90)),
        }

        # Edge stats
        edges = probs - sizing_intercept
        metrics.edge_stats[model] = {
            'mean': float(edges.mean()),
            'median': float(edges.median()),
            'std': float(edges.std()),
            'p10': float(np.percentile(edges, 10)),
            'p50': float(np.percentile(edges, 50)),
            'p90': float(np.percentile(edges, 90)),
        }

        # Extreme signal rate per model
        extreme_rate = ((probs > extreme_high) | (probs < extreme_low)).mean()
        metrics.extreme_signal_rate_by_model[model] = float(extreme_rate)

        # PSI vs baseline
        if baseline_manager is not None:
            psi = baseline_manager.compute_psi_vs_baseline(model, 'probability', probs.values)
            if psi is not None:
                metrics.psi_by_model[model] = psi

            # KS vs baseline
            ks_result = baseline_manager.compute_ks_vs_baseline(model, 'probability', probs.values)
            if ks_result is not None:
                metrics.ks_by_model[model] = ks_result[0]

    # Overall extreme signal rate
    all_probs = []
    for model in models:
        prob_col = f"p_{model}"
        if prob_col in signals.columns:
            all_probs.extend(signals[prob_col].dropna().tolist())

    if all_probs:
        all_probs = np.array(all_probs)
        metrics.extreme_signal_rate = float(
            ((all_probs > extreme_high) | (all_probs < extreme_low)).mean()
        )

    # Rank correlation vs previous week
    if previous_week_signals is not None and 'symbol' in signals.columns:
        # Use average edge across models for ranking
        current_edges = []
        prev_edges = []

        for model in models:
            prob_col = f"p_{model}"
            if prob_col in signals.columns and prob_col in previous_week_signals.columns:
                current_edges.append(signals.set_index('symbol')[prob_col])
                prev_edges.append(previous_week_signals.set_index('symbol')[prob_col])

        if current_edges and prev_edges:
            current_avg = pd.concat(current_edges, axis=1).mean(axis=1)
            prev_avg = pd.concat(prev_edges, axis=1).mean(axis=1)

            common = current_avg.index.intersection(prev_avg.index)
            if len(common) > 10:
                corr, _ = stats.spearmanr(
                    current_avg.loc[common].values,
                    prev_avg.loc[common].values
                )
                metrics.rank_correlation_vs_prev = float(corr) if not np.isnan(corr) else None

    return metrics


def compute_portfolio_metrics(
    signals: pd.DataFrame,
    sizing_config: Optional[Dict[str, Any]] = None,
    previous_weights: Optional[pd.Series] = None,
) -> PortfolioMetrics:
    """Compute portfolio construction metrics.

    Args:
        signals: DataFrame with final_weight column.
        sizing_config: Sizing configuration for context.
        previous_weights: Previous week weights for turnover.

    Returns:
        PortfolioMetrics object.
    """
    metrics = PortfolioMetrics()

    if 'final_weight' not in signals.columns:
        return metrics

    weights = signals['final_weight']

    # Basic exposure
    metrics.gross_exposure = float(weights.abs().sum())
    metrics.net_exposure = float(weights.sum())
    metrics.cash_fraction = max(0, 1.0 - metrics.gross_exposure)

    # Position counts
    positions = weights[weights != 0]
    metrics.n_positions = len(positions)
    metrics.n_longs = int((weights > 0).sum())
    metrics.n_shorts = int((weights < 0).sum())

    # Concentration metrics
    if metrics.gross_exposure > 0:
        abs_weights = weights.abs()
        sorted_weights = abs_weights.sort_values(ascending=False)

        # Top-k concentration
        metrics.top1_concentration = float(sorted_weights.iloc[0] / metrics.gross_exposure) if len(sorted_weights) > 0 else 0
        metrics.top5_concentration = float(sorted_weights.iloc[:5].sum() / metrics.gross_exposure) if len(sorted_weights) >= 5 else metrics.top1_concentration
        metrics.top10_concentration = float(sorted_weights.iloc[:10].sum() / metrics.gross_exposure) if len(sorted_weights) >= 10 else metrics.top5_concentration

        # Max weight
        metrics.max_weight = float(abs_weights.max())

        # Max weight as % of cap
        max_cap = sizing_config.get('max_weight_per_name', 0.10) if sizing_config else 0.10
        metrics.max_weight_pct_of_cap = metrics.max_weight / max_cap if max_cap > 0 else 0

        # HHI (sum of squared weight shares)
        weight_shares = abs_weights / metrics.gross_exposure
        metrics.hhi = float((weight_shares ** 2).sum())

    # Turnover
    if previous_weights is not None and 'symbol' in signals.columns:
        current = signals.set_index('symbol')['final_weight']
        prev = previous_weights

        # Align indices
        all_symbols = current.index.union(prev.index)
        current_aligned = current.reindex(all_symbols, fill_value=0)
        prev_aligned = prev.reindex(all_symbols, fill_value=0)

        # Total turnover (sum of |delta|)
        delta = current_aligned - prev_aligned
        metrics.turnover_total = float(delta.abs().sum())

        # One-way turnover (buys OR sells)
        metrics.turnover_one_way = float(max(delta[delta > 0].sum(), delta[delta < 0].abs().sum()))

        # Turnover vs penalty (if configured)
        if sizing_config and sizing_config.get('turnover_penalty', 0) > 0:
            metrics.turnover_vs_penalty = metrics.turnover_total / sizing_config['turnover_penalty']

    # Model contribution metrics
    if 'contributing_model' in signals.columns:
        contrib_counts = signals[signals['final_weight'] != 0]['contributing_model'].value_counts()
        metrics.model_counts = contrib_counts.to_dict()

        # Gross exposure by model
        for model in signals['contributing_model'].dropna().unique():
            model_weights = signals[signals['contributing_model'] == model]['final_weight']
            metrics.model_gross_exposure[model] = float(model_weights.abs().sum())

    # Multi-model conflict detection
    # Check if multiple long/short models had signals for same symbol
    model_weight_cols = [c for c in signals.columns if c.startswith('w_') and c != 'w_final']
    if len(model_weight_cols) >= 2:
        conflicts = 0
        for idx in signals.index:
            row = signals.loc[idx]
            firing_models = [c for c in model_weight_cols if abs(row.get(c, 0)) > 0.001]
            if len(firing_models) > 1:
                conflicts += 1
        metrics.multi_model_conflict_count = conflicts

    # Gating metrics
    if 'gating_multiplier' in signals.columns:
        metrics.gating_multiplier = float(signals['gating_multiplier'].iloc[0])
        metrics.gating_reduced_exposure = metrics.gating_multiplier < 1.0

        if 'gating_rules' in signals.columns:
            rules = signals['gating_rules'].iloc[0]
            if rules and isinstance(rules, str):
                metrics.gating_modes_disabled = rules.split(',')

        # Estimate trades removed by gating
        if 'combined_weight' in signals.columns and metrics.gating_multiplier < 1.0:
            ungated_positions = (signals['combined_weight'].abs() > 0.001).sum()
            gated_positions = (signals['final_weight'].abs() > 0.001).sum()
            metrics.gating_trades_removed = max(0, ungated_positions - gated_positions)

            ungated_exposure = signals['combined_weight'].abs().sum()
            metrics.gating_exposure_reduced = max(0, ungated_exposure - metrics.gross_exposure)

    return metrics


def compute_execution_risk_metrics(
    signals: pd.DataFrame,
    liquidity_col: str = 'rdollar_vol_20',
    gap_col: str = 'gap_atr_ratio_raw',
    liquidity_threshold_millions: float = 1.0,
    gap_threshold: float = 1.0,
    slippage_budget_bps: float = 20.0,
) -> ExecutionRiskMetrics:
    """Compute execution risk proxy metrics.

    Args:
        signals: DataFrame with position weights and liquidity features.
        liquidity_col: Column name for dollar volume.
        gap_col: Column name for gap-to-ATR ratio.
        liquidity_threshold_millions: Threshold in millions.
        gap_threshold: Gap ratio threshold.
        slippage_budget_bps: Slippage budget in basis points.

    Returns:
        ExecutionRiskMetrics object.
    """
    metrics = ExecutionRiskMetrics()
    metrics.liquidity_threshold_used = liquidity_threshold_millions
    metrics.gap_threshold_used = gap_threshold

    # Filter to positions only
    if 'final_weight' in signals.columns:
        positions = signals[signals['final_weight'] != 0]
    else:
        positions = signals

    if len(positions) == 0:
        return metrics

    # Liquidity metrics
    if liquidity_col in positions.columns:
        liquidity = positions[liquidity_col].dropna()
        if len(liquidity) > 0:
            # Assuming rdollar_vol is in dollars, convert threshold to same units
            threshold = liquidity_threshold_millions * 1e6

            metrics.median_rdollar_vol = float(liquidity.median())
            metrics.pct_below_liquidity_threshold = float((liquidity < threshold).mean())

    # Alternative liquidity columns to try
    alt_liquidity_cols = ['d_rdollar_vol_20', 'volume_20d_avg', 'adv']
    if metrics.median_rdollar_vol is None:
        for col in alt_liquidity_cols:
            if col in positions.columns:
                liquidity = positions[col].dropna()
                if len(liquidity) > 0:
                    metrics.median_rdollar_vol = float(liquidity.median())
                    break

    # Gap risk metrics
    if gap_col in positions.columns:
        gaps = positions[gap_col].dropna()
        if len(gaps) > 0:
            metrics.median_gap_atr_ratio = float(gaps.median())
            metrics.pct_above_gap_threshold = float((gaps > gap_threshold).mean())

    # Alternative gap columns
    alt_gap_cols = ['d_gap_atr_ratio_raw', 'gap_risk', 'overnight_gap']
    if metrics.median_gap_atr_ratio is None:
        for col in alt_gap_cols:
            if col in positions.columns:
                gaps = positions[col].dropna()
                if len(gaps) > 0:
                    metrics.median_gap_atr_ratio = float(gaps.median())
                    break

    # Simple slippage estimate based on position sizes and liquidity
    # Assumes slippage ~ weight / liquidity (highly simplified)
    if 'final_weight' in positions.columns and metrics.median_rdollar_vol is not None:
        total_weight = positions['final_weight'].abs().sum()
        # Very rough estimate: 1 bp per 10% of volume traded
        estimated_slippage = total_weight * 10  # bps
        metrics.estimated_slippage = float(estimated_slippage)
        metrics.slippage_budget_utilization = estimated_slippage / slippage_budget_bps

    return metrics


def compute_backtest_metrics(
    weekly_return: float,
    long_return: float,
    short_return: float,
    cumulative_returns: pd.Series,
    trailing_returns: pd.Series,
    tail_percentile: float = 5.0,
    tail_absolute: float = -0.05,
) -> BacktestMetrics:
    """Compute backtest outcome metrics.

    Args:
        weekly_return: This week's return.
        long_return: Long sleeve contribution.
        short_return: Short sleeve contribution.
        cumulative_returns: Series of cumulative returns.
        trailing_returns: Series of trailing weekly returns.
        tail_percentile: Percentile for tail detection.
        tail_absolute: Absolute threshold for tail detection.

    Returns:
        BacktestMetrics object.
    """
    metrics = BacktestMetrics()

    metrics.weekly_return = float(weekly_return)
    metrics.weekly_return_long = float(long_return)
    metrics.weekly_return_short = float(short_return)

    if len(cumulative_returns) > 0:
        metrics.cumulative_return = float(cumulative_returns.iloc[-1])

        # Rolling drawdown
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / (peak + 1e-10)
        metrics.rolling_drawdown = float(drawdown.iloc[-1])
        metrics.max_drawdown_trailing = float(drawdown.min())

    # Rolling Sharpe
    if len(trailing_returns) >= 13:
        recent_13 = trailing_returns.iloc[-13:]
        sharpe_13 = recent_13.mean() / (recent_13.std() + 1e-10) * np.sqrt(52)
        metrics.rolling_sharpe_13w = float(sharpe_13)

    if len(trailing_returns) >= 26:
        recent_26 = trailing_returns.iloc[-26:]
        sharpe_26 = recent_26.mean() / (recent_26.std() + 1e-10) * np.sqrt(52)
        metrics.rolling_sharpe_26w = float(sharpe_26)

    # Tail week detection
    if len(trailing_returns) > 20:
        historical_p5 = np.percentile(trailing_returns, tail_percentile)
        threshold = min(historical_p5, tail_absolute)
        metrics.tail_week_threshold = float(threshold)
        metrics.is_tail_week = weekly_return < threshold
    else:
        metrics.tail_week_threshold = tail_absolute
        metrics.is_tail_week = weekly_return < tail_absolute

    return metrics


def compute_aggregate_metrics(
    weekly_diagnostics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute aggregate metrics across all weeks.

    Args:
        weekly_diagnostics: List of weekly diagnostic dicts.

    Returns:
        Dict of aggregate metrics.
    """
    if not weekly_diagnostics:
        return {}

    aggregate = {}

    # Extract arrays for aggregation
    gross_exposures = [w.get('portfolio', {}).get('gross_exposure', 0) for w in weekly_diagnostics]
    turnovers = [w.get('portfolio', {}).get('turnover_total', 0) for w in weekly_diagnostics]
    n_positions = [w.get('portfolio', {}).get('n_positions', 0) for w in weekly_diagnostics]
    hhis = [w.get('portfolio', {}).get('hhi', 0) for w in weekly_diagnostics]

    aggregate['mean_gross_exposure'] = float(np.mean(gross_exposures))
    aggregate['std_gross_exposure'] = float(np.std(gross_exposures))
    aggregate['mean_turnover'] = float(np.mean(turnovers))
    aggregate['std_turnover'] = float(np.std(turnovers))
    aggregate['mean_positions'] = float(np.mean(n_positions))
    aggregate['mean_hhi'] = float(np.mean(hhis))

    # PSI averages per model
    psi_by_model: Dict[str, List[float]] = {}
    for w in weekly_diagnostics:
        for model, psi in w.get('signal_distribution', {}).get('psi_by_model', {}).items():
            if model not in psi_by_model:
                psi_by_model[model] = []
            psi_by_model[model].append(psi)

    aggregate['mean_psi_by_model'] = {
        model: float(np.mean(values)) for model, values in psi_by_model.items()
    }

    # Backtest metrics if available
    weekly_returns = [
        w.get('backtest', {}).get('weekly_return', 0)
        for w in weekly_diagnostics
        if w.get('backtest') is not None
    ]

    if weekly_returns:
        returns = np.array(weekly_returns)
        aggregate['total_return'] = float((1 + returns).prod() - 1)
        aggregate['mean_weekly_return'] = float(returns.mean())
        aggregate['std_weekly_return'] = float(returns.std())
        aggregate['sharpe_ratio'] = float(
            returns.mean() / (returns.std() + 1e-10) * np.sqrt(52)
        )
        aggregate['hit_rate'] = float((returns > 0).mean())

        # Max drawdown
        cum = (1 + returns).cumprod()
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        aggregate['max_drawdown'] = float(dd.min())

    # Warning summary
    total_warnings = sum(len(w.get('warnings', [])) for w in weekly_diagnostics)
    aggregate['total_warnings'] = total_warnings

    return aggregate
