"""Statistics display for feature selection.

This module provides attractive formatted output for tracking metrics
during the feature selection process.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .config import MetricType, SubsetResult
from .metrics import (
    compute_auc, compute_aupr, compute_brier, compute_log_loss,
    compute_ic, compute_precision_at_k, compute_hit_rate, is_higher_better
)


# =============================================================================
# Display Constants
# =============================================================================

# Box drawing characters for tables
BOX_CHARS = {
    'top_left': '┌',
    'top_right': '┐',
    'bottom_left': '└',
    'bottom_right': '┘',
    'horizontal': '─',
    'vertical': '│',
    'cross': '┼',
    'top_tee': '┬',
    'bottom_tee': '┴',
    'left_tee': '├',
    'right_tee': '┤',
}

# Metric display names and formatting
METRIC_DISPLAY = {
    MetricType.AUC: ('AUC', '{:.4f}', '↑'),
    MetricType.AUPR: ('AUPR', '{:.4f}', '↑'),
    MetricType.BRIER: ('Brier', '{:.4f}', '↓'),
    MetricType.LOG_LOSS: ('LogLoss', '{:.4f}', '↓'),
    MetricType.IC: ('IC', '{:.4f}', '↑'),
    MetricType.PRECISION_AT_K: ('Prec@10%', '{:.4f}', '↑'),
    MetricType.HIT_RATE: ('HitRate', '{:.4f}', '↑'),
}


# =============================================================================
# Multi-Metric Result
# =============================================================================

@dataclass
class MultiMetricResult:
    """Result containing multiple metrics for a feature subset.

    Attributes:
        features: List of features in the subset.
        metrics: Dict mapping MetricType to (mean, std, fold_values).
        primary_metric: The main metric used for optimization.
    """
    features: List[str]
    metrics: Dict[MetricType, Tuple[float, float, List[float]]]
    primary_metric: MetricType = MetricType.AUC

    @property
    def n_features(self) -> int:
        return len(self.features)

    def get_primary(self) -> Tuple[float, float]:
        """Get primary metric (mean, std)."""
        if self.primary_metric in self.metrics:
            mean, std, _ = self.metrics[self.primary_metric]
            return mean, std
        return 0.0, 0.0


# =============================================================================
# Statistics Display Class
# =============================================================================

class StatsDisplay:
    """Attractive statistics display for feature selection.

    Provides formatted tables and progress indicators for tracking
    multiple metrics during the selection process.
    """

    def __init__(
        self,
        metrics_to_track: Optional[List[MetricType]] = None,
        primary_metric: MetricType = MetricType.AUC,
        show_folds: bool = False,
        width: int = 80,
    ):
        """Initialize the display.

        Args:
            metrics_to_track: List of metrics to track. If None, uses defaults.
            primary_metric: Primary metric for optimization.
            show_folds: Whether to show per-fold metrics.
            width: Display width in characters.
        """
        if metrics_to_track is None:
            # Default metrics suite
            self.metrics_to_track = [
                MetricType.AUC,
                MetricType.AUPR,
                MetricType.BRIER,
                MetricType.LOG_LOSS,
            ]
        else:
            self.metrics_to_track = metrics_to_track

        self.primary_metric = primary_metric
        self.show_folds = show_folds
        self.width = width

        # History tracking
        self.history: List[Dict] = []

    def _format_metric(
        self,
        metric_type: MetricType,
        value: float,
        std: Optional[float] = None,
        baseline: Optional[float] = None,
    ) -> str:
        """Format a metric value with optional std and delta from baseline."""
        _, fmt, direction = METRIC_DISPLAY.get(
            metric_type, (metric_type.value, '{:.4f}', '↑')
        )

        formatted = fmt.format(value)

        if std is not None:
            formatted += f"±{std:.4f}"

        if baseline is not None:
            delta = value - baseline
            # For lower-is-better metrics, flip the sign for display
            higher_better = is_higher_better(metric_type)

            if abs(delta) > 1e-6:
                sign = '+' if delta > 0 else ''
                # Color indicator based on improvement
                if higher_better:
                    indicator = '▲' if delta > 0 else '▼'
                else:
                    indicator = '▼' if delta < 0 else '▲'
                formatted += f" ({sign}{delta:.4f} {indicator})"

        return formatted

    def _draw_header(self, title: str) -> str:
        """Draw a header box."""
        b = BOX_CHARS
        inner_width = self.width - 2

        lines = [
            f"{b['top_left']}{b['horizontal'] * inner_width}{b['top_right']}",
            f"{b['vertical']} {title.center(inner_width - 2)} {b['vertical']}",
            f"{b['bottom_left']}{b['horizontal'] * inner_width}{b['bottom_right']}",
        ]
        return '\n'.join(lines)

    def _draw_table(
        self,
        headers: List[str],
        rows: List[List[str]],
        col_widths: Optional[List[int]] = None,
    ) -> str:
        """Draw a formatted table."""
        if col_widths is None:
            # Auto-calculate widths
            col_widths = []
            for i, h in enumerate(headers):
                max_width = len(h)
                for row in rows:
                    if i < len(row):
                        max_width = max(max_width, len(str(row[i])))
                col_widths.append(max_width + 2)

        b = BOX_CHARS

        # Top border
        top = b['top_left']
        for i, w in enumerate(col_widths):
            top += b['horizontal'] * w
            top += b['top_tee'] if i < len(col_widths) - 1 else ''
        top += b['top_right']

        # Header row
        header_row = b['vertical']
        for h, w in zip(headers, col_widths):
            header_row += f" {h.center(w - 2)} " + b['vertical']

        # Header separator
        sep = b['left_tee']
        for i, w in enumerate(col_widths):
            sep += b['horizontal'] * w
            sep += b['cross'] if i < len(col_widths) - 1 else ''
        sep += b['right_tee']

        # Data rows
        data_rows = []
        for row in rows:
            row_str = b['vertical']
            for val, w in zip(row, col_widths):
                row_str += f" {str(val).ljust(w - 2)} " + b['vertical']
            data_rows.append(row_str)

        # Bottom border
        bottom = b['bottom_left']
        for i, w in enumerate(col_widths):
            bottom += b['horizontal'] * w
            bottom += b['bottom_tee'] if i < len(col_widths) - 1 else ''
        bottom += b['bottom_right']

        lines = [top, header_row, sep] + data_rows + [bottom]
        return '\n'.join(lines)

    def format_metrics_table(
        self,
        result: MultiMetricResult,
        baseline: Optional[MultiMetricResult] = None,
        stage_name: str = "",
    ) -> str:
        """Format a comprehensive metrics table.

        Args:
            result: Current multi-metric result.
            baseline: Optional baseline for comparison.
            stage_name: Name of current pipeline stage.

        Returns:
            Formatted string with metrics table.
        """
        lines = []

        # Header
        if stage_name:
            lines.append(self._draw_header(f"📊 {stage_name}"))
        else:
            lines.append(self._draw_header("📊 Feature Selection Metrics"))

        lines.append("")
        lines.append(f"  Features: {result.n_features}")
        lines.append("")

        # Metrics table
        headers = ['Metric', 'Value', 'Std', 'Direction', 'Δ Baseline']
        rows = []

        for metric_type in self.metrics_to_track:
            if metric_type not in result.metrics:
                continue

            name, _, direction = METRIC_DISPLAY.get(
                metric_type, (metric_type.value, '{:.4f}', '↑')
            )

            mean, std, _ = result.metrics[metric_type]

            # Determine if this is the primary metric
            is_primary = metric_type == result.primary_metric
            name_display = f"★ {name}" if is_primary else f"  {name}"

            # Calculate delta from baseline
            delta_str = "-"
            if baseline and metric_type in baseline.metrics:
                baseline_mean, _, _ = baseline.metrics[metric_type]
                delta = mean - baseline_mean

                higher_better = is_higher_better(metric_type)
                if abs(delta) > 1e-6:
                    sign = '+' if delta > 0 else ''
                    improved = (delta > 0) if higher_better else (delta < 0)
                    indicator = '✓' if improved else '✗'
                    delta_str = f"{sign}{delta:.4f} {indicator}"
                else:
                    delta_str = "  0.0000 ─"

            rows.append([
                name_display,
                f"{mean:.4f}",
                f"{std:.4f}",
                direction,
                delta_str,
            ])

        lines.append(self._draw_table(headers, rows))

        return '\n'.join(lines)

    def format_stage_summary(
        self,
        stage_name: str,
        n_features: int,
        primary_value: float,
        primary_std: float,
        action: str = "",
        feature_changed: str = "",
    ) -> str:
        """Format a compact stage summary line.

        Args:
            stage_name: Name of the stage.
            n_features: Number of features.
            primary_value: Primary metric value.
            primary_std: Primary metric std.
            action: Action taken ('+', '-', '↔').
            feature_changed: Name of feature added/removed.

        Returns:
            Formatted summary line.
        """
        name, _, _ = METRIC_DISPLAY.get(
            self.primary_metric, ('AUC', '{:.4f}', '↑')
        )

        line = f"  {stage_name:20s} │ {n_features:3d} features │ {name}: {primary_value:.4f}±{primary_std:.4f}"

        if action and feature_changed:
            line += f" │ {action} {feature_changed}"

        return line

    def format_fold_breakdown(
        self,
        result: MultiMetricResult,
        metric_type: Optional[MetricType] = None,
    ) -> str:
        """Format per-fold metric breakdown.

        Args:
            result: Multi-metric result.
            metric_type: Metric to show folds for (default: primary).

        Returns:
            Formatted fold breakdown.
        """
        if metric_type is None:
            metric_type = result.primary_metric

        if metric_type not in result.metrics:
            return "  No fold data available"

        mean, std, fold_values = result.metrics[metric_type]
        name, _, _ = METRIC_DISPLAY.get(metric_type, (metric_type.value, '{:.4f}', '↑'))

        lines = [f"  {name} by Fold:"]

        for i, val in enumerate(fold_values):
            # Indicate if below mean
            indicator = '▼' if val < mean - std else ('▲' if val > mean + std else ' ')
            bar_len = int((val - 0.5) * 40) if val > 0.5 else 0  # Assuming 0.5-1.0 range
            bar = '█' * max(0, min(20, bar_len))
            lines.append(f"    Fold {i+1}: {val:.4f} {indicator} {bar}")

        return '\n'.join(lines)

    def format_progress_bar(
        self,
        current: int,
        total: int,
        prefix: str = "",
        width: int = 40,
    ) -> str:
        """Format a progress bar.

        Args:
            current: Current progress.
            total: Total items.
            prefix: Prefix text.
            width: Bar width in characters.

        Returns:
            Formatted progress bar.
        """
        if total == 0:
            pct = 100
        else:
            pct = int(100 * current / total)

        filled = int(width * current / max(total, 1))
        bar = '█' * filled + '░' * (width - filled)

        return f"  {prefix} [{bar}] {pct:3d}% ({current}/{total})"

    def print_pipeline_header(self, config_summary: str = "") -> None:
        """Print pipeline header with configuration summary."""
        print()
        print("=" * self.width)
        print("  FEATURE SELECTION PIPELINE - MULTI-METRIC TRACKING")
        print("=" * self.width)
        print()

        # Metrics being tracked
        metric_names = [
            METRIC_DISPLAY.get(m, (m.value, '', ''))[0]
            for m in self.metrics_to_track
        ]
        primary_name = METRIC_DISPLAY.get(
            self.primary_metric, (self.primary_metric.value, '', '')
        )[0]

        print(f"  Primary Metric: {primary_name}")
        print(f"  Tracking: {', '.join(metric_names)}")

        if config_summary:
            print()
            print(f"  {config_summary}")

        print()
        print("-" * self.width)

    def print_stage_header(self, stage_name: str, description: str = "") -> None:
        """Print a stage header."""
        print()
        print(f"┌{'─' * (self.width - 2)}┐")
        print(f"│ {stage_name.ljust(self.width - 4)} │")
        if description:
            print(f"│ {description.ljust(self.width - 4)} │")
        print(f"└{'─' * (self.width - 2)}┘")
        print()

    def print_metrics(
        self,
        result: MultiMetricResult,
        baseline: Optional[MultiMetricResult] = None,
        stage_name: str = "",
    ) -> None:
        """Print formatted metrics table."""
        print(self.format_metrics_table(result, baseline, stage_name))

    def print_summary_line(
        self,
        stage_name: str,
        n_features: int,
        primary_value: float,
        primary_std: float,
        action: str = "",
        feature_changed: str = "",
    ) -> None:
        """Print compact summary line."""
        print(self.format_stage_summary(
            stage_name, n_features, primary_value, primary_std,
            action, feature_changed
        ))

    def print_final_summary(
        self,
        best_result: MultiMetricResult,
        best_stage: str,
        all_stages: List[Dict],
    ) -> None:
        """Print final pipeline summary.

        Args:
            best_result: Best result found.
            best_stage: Stage where best was found.
            all_stages: List of stage summaries.
        """
        print()
        print("=" * self.width)
        print("  PIPELINE COMPLETE - FINAL RESULTS")
        print("=" * self.width)
        print()

        # Stage comparison table
        headers = ['Stage', 'Features', 'AUC', 'AUPR', 'Brier', 'Best']
        rows = []

        for stage in all_stages:
            is_best = stage.get('name', '') == best_stage
            rows.append([
                stage.get('name', ''),
                str(stage.get('n_features', 0)),
                f"{stage.get('auc', 0):.4f}",
                f"{stage.get('aupr', 0):.4f}",
                f"{stage.get('brier', 0):.4f}",
                '★' if is_best else '',
            ])

        print(self._draw_table(headers, rows))
        print()

        # Best result details
        print(f"  Best found at: {best_stage}")
        print(f"  Features: {best_result.n_features}")
        print()

        # Final metrics
        print("  Final Metrics:")
        for metric_type in self.metrics_to_track:
            if metric_type not in best_result.metrics:
                continue
            name, _, direction = METRIC_DISPLAY.get(
                metric_type, (metric_type.value, '{:.4f}', '↑')
            )
            mean, std, _ = best_result.metrics[metric_type]
            is_primary = metric_type == best_result.primary_metric
            marker = '★' if is_primary else ' '
            print(f"    {marker} {name:12s}: {mean:.4f} ± {std:.4f} ({direction} better)")

        print()
        print("=" * self.width)


# =============================================================================
# Metric Computation Helper
# =============================================================================

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: List[MetricType],
) -> Dict[MetricType, float]:
    """Compute multiple metrics for predictions.

    Args:
        y_true: True labels.
        y_pred: Predicted probabilities.
        metrics: List of metrics to compute.

    Returns:
        Dict mapping MetricType to value.
    """
    results = {}

    metric_funcs = {
        MetricType.AUC: compute_auc,
        MetricType.AUPR: compute_aupr,
        MetricType.BRIER: compute_brier,
        MetricType.LOG_LOSS: compute_log_loss,
        MetricType.IC: compute_ic,
        MetricType.HIT_RATE: compute_hit_rate,
        MetricType.PRECISION_AT_K: lambda y, p: compute_precision_at_k(y, p, k=0.1),
    }

    for metric in metrics:
        if metric in metric_funcs:
            try:
                results[metric] = metric_funcs[metric](y_true, y_pred)
            except Exception:
                results[metric] = np.nan

    return results


def aggregate_fold_metrics(
    fold_results: List[Dict[MetricType, float]],
) -> Dict[MetricType, Tuple[float, float, List[float]]]:
    """Aggregate metrics across CV folds.

    Args:
        fold_results: List of per-fold metric dicts.

    Returns:
        Dict mapping MetricType to (mean, std, fold_values).
    """
    aggregated = {}

    # Get all metrics present in any fold
    all_metrics = set()
    for fold in fold_results:
        all_metrics.update(fold.keys())

    for metric in all_metrics:
        values = [fold.get(metric, np.nan) for fold in fold_results]
        valid = [v for v in values if not np.isnan(v)]

        if valid:
            aggregated[metric] = (
                np.mean(valid),
                np.std(valid),
                values,
            )

    return aggregated
