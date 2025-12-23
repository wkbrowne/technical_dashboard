"""Tear sheet generation for backtest reports."""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union
import json
import numpy as np
import pandas as pd

from ..data.schemas import BacktestResult
from ..backtest.metrics import (
    compute_performance_metrics,
    compute_monthly_returns,
    PerformanceMetrics,
)

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from .plots import (
    plot_equity_curve,
    plot_drawdown,
    plot_exposure,
    plot_weekly_returns,
    plot_turnover,
    plot_monthly_heatmap,
)


def generate_tearsheet(
    result: BacktestResult,
    title: str = "Backtest Report",
    save_path: Optional[Union[str, Path]] = None,
) -> Optional[object]:
    """Generate a full tear sheet for backtest results.

    Args:
        result: BacktestResult from backtest engine.
        title: Report title.
        save_path: Optional path to save PDF.

    Returns:
        Matplotlib figure if not saving.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, generating text report only")
        return _generate_text_report(result, title)

    equity_curve = result.equity_curve

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 20))

    # Title
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    # 1. Equity Curve (top)
    ax1 = fig.add_subplot(4, 2, (1, 2))
    plot_equity_curve(equity_curve, ax=ax1)

    # 2. Drawdown
    ax2 = fig.add_subplot(4, 2, 3)
    plot_drawdown(equity_curve, ax=ax2)

    # 3. Weekly Returns Distribution
    ax3 = fig.add_subplot(4, 2, 4)
    plot_weekly_returns(equity_curve, ax=ax3)

    # 4. Exposure
    ax4 = fig.add_subplot(4, 2, 5)
    plot_exposure(equity_curve, ax=ax4)

    # 5. Turnover
    ax5 = fig.add_subplot(4, 2, 6)
    plot_turnover(equity_curve, ax=ax5)

    # 6. Metrics Table
    ax6 = fig.add_subplot(4, 2, (7, 8))
    _plot_metrics_table(ax6, result.metrics)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Tear sheet saved to: {save_path}")
        plt.close(fig)
        return None

    return fig


def _plot_metrics_table(ax, metrics: Dict[str, float]) -> None:
    """Plot metrics as a table."""
    ax.axis("off")

    # Format metrics for display
    table_data = []

    # Returns section
    table_data.append(["RETURNS", ""])
    table_data.append(["Total Return", f"{metrics.get('total_return', 0):.2%}"])
    table_data.append(["CAGR", f"{metrics.get('cagr', 0):.2%}"])
    table_data.append(["Best Week", f"{metrics.get('best_week', 0):.2%}"])
    table_data.append(["Worst Week", f"{metrics.get('worst_week', 0):.2%}"])

    # Risk section
    table_data.append(["", ""])
    table_data.append(["RISK", ""])
    table_data.append(["Volatility", f"{metrics.get('volatility', 0):.2%}"])
    table_data.append(["Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}"])
    table_data.append(["VaR (95%)", f"{metrics.get('var_95', 0):.2%}"])

    # Risk-adjusted section
    table_data.append(["", ""])
    table_data.append(["RISK-ADJUSTED", ""])
    table_data.append(["Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}"])
    table_data.append(["Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}"])
    table_data.append(["Calmar Ratio", f"{metrics.get('calmar_ratio', 0):.2f}"])

    # Trading section
    table_data.append(["", ""])
    table_data.append(["TRADING", ""])
    table_data.append(["Hit Rate", f"{metrics.get('hit_rate', 0):.2%}"])
    table_data.append(["Profit Factor", f"{metrics.get('profit_factor', 0):.2f}"])
    table_data.append(["Avg Turnover", f"{metrics.get('avg_turnover', 0):.2%}"])
    table_data.append(["Total Trades", f"{int(metrics.get('n_trades', 0))}"])

    # Create table
    table = ax.table(
        cellText=table_data,
        colWidths=[0.3, 0.2],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.5, 1.5)

    # Style header rows
    for i, row in enumerate(table_data):
        if row[1] == "":  # Header row
            table[(i, 0)].set_text_props(fontweight="bold")
            table[(i, 0)].set_facecolor("#e6e6e6")
            table[(i, 1)].set_facecolor("#e6e6e6")


def _generate_text_report(result: BacktestResult, title: str) -> str:
    """Generate text-based report when matplotlib unavailable."""
    metrics = result.metrics
    lines = []

    lines.append("=" * 60)
    lines.append(title)
    lines.append("=" * 60)
    lines.append("")

    lines.append("RETURNS:")
    lines.append(f"  Total Return:    {metrics.get('total_return', 0):>10.2%}")
    lines.append(f"  CAGR:            {metrics.get('cagr', 0):>10.2%}")
    lines.append(f"  Best Week:       {metrics.get('best_week', 0):>10.2%}")
    lines.append(f"  Worst Week:      {metrics.get('worst_week', 0):>10.2%}")
    lines.append("")

    lines.append("RISK:")
    lines.append(f"  Volatility:      {metrics.get('volatility', 0):>10.2%}")
    lines.append(f"  Max Drawdown:    {metrics.get('max_drawdown', 0):>10.2%}")
    lines.append(f"  VaR (95%):       {metrics.get('var_95', 0):>10.2%}")
    lines.append("")

    lines.append("RISK-ADJUSTED:")
    lines.append(f"  Sharpe Ratio:    {metrics.get('sharpe_ratio', 0):>10.2f}")
    lines.append(f"  Sortino Ratio:   {metrics.get('sortino_ratio', 0):>10.2f}")
    lines.append(f"  Calmar Ratio:    {metrics.get('calmar_ratio', 0):>10.2f}")
    lines.append("")

    lines.append("TRADING:")
    lines.append(f"  Hit Rate:        {metrics.get('hit_rate', 0):>10.2%}")
    lines.append(f"  Profit Factor:   {metrics.get('profit_factor', 0):>10.2f}")
    lines.append(f"  Avg Turnover:    {metrics.get('avg_turnover', 0):>10.2%}")
    lines.append(f"  Total Trades:    {int(metrics.get('n_trades', 0)):>10d}")
    lines.append("")

    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)
    return report


def save_tearsheet(
    result: BacktestResult,
    output_dir: Union[str, Path],
    name: str = "backtest",
) -> Dict[str, Path]:
    """Save complete tear sheet with all outputs.

    Args:
        result: BacktestResult from backtest.
        output_dir: Output directory.
        name: Base name for files.

    Returns:
        Dict of output_type -> path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    # 1. PDF tear sheet
    if MATPLOTLIB_AVAILABLE:
        pdf_path = output_dir / f"{name}_tearsheet.pdf"
        generate_tearsheet(result, title=f"Backtest Report: {name}", save_path=pdf_path)
        outputs["pdf"] = pdf_path

    # 2. Metrics JSON
    metrics_path = output_dir / f"{name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result.metrics, f, indent=2, default=lambda x: float(x))
    outputs["metrics"] = metrics_path

    # 3. Equity curve CSV
    equity_path = output_dir / f"{name}_equity_curve.csv"
    result.equity_curve.to_csv(equity_path, index=False)
    outputs["equity_curve"] = equity_path

    # 4. Monthly returns
    try:
        monthly = compute_monthly_returns(result.equity_curve)
        monthly_path = output_dir / f"{name}_monthly_returns.csv"
        monthly.to_csv(monthly_path)
        outputs["monthly_returns"] = monthly_path
    except Exception:
        pass

    # 5. Config
    config_path = output_dir / f"{name}_config.json"
    with open(config_path, "w") as f:
        json.dump(result.config, f, indent=2, default=str)
    outputs["config"] = config_path

    print(f"\nTear sheet saved to: {output_dir}")
    for output_type, path in outputs.items():
        print(f"  {output_type}: {path.name}")

    return outputs


def compare_strategies_tearsheet(
    results: Dict[str, BacktestResult],
    output_dir: Union[str, Path],
) -> Path:
    """Generate comparison tear sheet for multiple strategies.

    Args:
        results: Dict of strategy_name -> BacktestResult.
        output_dir: Output directory.

    Returns:
        Path to comparison report.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create comparison table
    comparison_data = []
    for name, result in results.items():
        metrics = result.metrics
        comparison_data.append({
            "Strategy": name,
            "Total Return": f"{metrics.get('total_return', 0):.2%}",
            "CAGR": f"{metrics.get('cagr', 0):.2%}",
            "Volatility": f"{metrics.get('volatility', 0):.2%}",
            "Sharpe": f"{metrics.get('sharpe_ratio', 0):.2f}",
            "Sortino": f"{metrics.get('sortino_ratio', 0):.2f}",
            "Max DD": f"{metrics.get('max_drawdown', 0):.2%}",
            "Calmar": f"{metrics.get('calmar_ratio', 0):.2f}",
            "Hit Rate": f"{metrics.get('hit_rate', 0):.2%}",
            "Turnover": f"{metrics.get('avg_turnover', 0):.2%}",
        })

    comparison_df = pd.DataFrame(comparison_data)

    # Save comparison
    comparison_path = output_dir / "strategy_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    # Print
    print("\n" + "=" * 100)
    print("STRATEGY COMPARISON")
    print("=" * 100)
    print(comparison_df.to_string(index=False))
    print("=" * 100)

    # Create comparison plot if matplotlib available
    if MATPLOTLIB_AVAILABLE:
        from .plots import create_strategy_comparison_plot

        equity_curves = {name: result.equity_curve for name, result in results.items()}
        fig = create_strategy_comparison_plot(equity_curves)

        plot_path = output_dir / "strategy_comparison.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nComparison plot saved to: {plot_path}")

    return comparison_path
