"""Plotting utilities for backtest analysis."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def _check_matplotlib():
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting")


def plot_equity_curve(
    equity_curve: pd.DataFrame,
    benchmark: Optional[pd.Series] = None,
    title: str = "Equity Curve",
    figsize: Tuple[int, int] = (12, 6),
    ax=None,
):
    """Plot equity curve.

    Args:
        equity_curve: DataFrame with 'week_end' and 'nav' columns.
        benchmark: Optional benchmark NAV series.
        title: Plot title.
        figsize: Figure size.
        ax: Optional matplotlib axis.

    Returns:
        Matplotlib axis.
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot strategy
    dates = pd.to_datetime(equity_curve["week_end"])
    nav = equity_curve["nav"]

    # Normalize to 100
    nav_normalized = nav / nav.iloc[0] * 100

    ax.plot(dates, nav_normalized, label="Strategy", linewidth=1.5, color="blue")

    # Plot benchmark if provided
    if benchmark is not None:
        bench_normalized = benchmark / benchmark.iloc[0] * 100
        ax.plot(benchmark.index, bench_normalized, label="Benchmark",
                linewidth=1.0, color="gray", alpha=0.7)

    ax.set_xlabel("Date")
    ax.set_ylabel("NAV (normalized to 100)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    plt.tight_layout()
    return ax


def plot_drawdown(
    equity_curve: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 4),
    ax=None,
):
    """Plot drawdown curve.

    Args:
        equity_curve: DataFrame with 'week_end' and 'nav' columns.
        figsize: Figure size.
        ax: Optional matplotlib axis.

    Returns:
        Matplotlib axis.
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    dates = pd.to_datetime(equity_curve["week_end"])
    nav = equity_curve["nav"]

    # Compute drawdown
    running_max = nav.cummax()
    drawdown = (nav - running_max) / running_max * 100

    ax.fill_between(dates, drawdown, 0, color="red", alpha=0.3)
    ax.plot(dates, drawdown, color="red", linewidth=0.5)

    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown")
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    plt.tight_layout()
    return ax


def plot_exposure(
    equity_curve: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 4),
    ax=None,
):
    """Plot exposure over time.

    Args:
        equity_curve: DataFrame with exposure columns.
        figsize: Figure size.
        ax: Optional matplotlib axis.

    Returns:
        Matplotlib axis.
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    dates = pd.to_datetime(equity_curve["week_end"])

    if "gross_exposure" in equity_curve.columns:
        exposure = equity_curve["gross_exposure"] * 100
        ax.fill_between(dates, exposure, 0, alpha=0.3, label="Gross Exposure")
        ax.plot(dates, exposure, linewidth=0.5)

    if "n_positions" in equity_curve.columns:
        ax2 = ax.twinx()
        ax2.plot(dates, equity_curve["n_positions"], color="green",
                 linewidth=1, label="# Positions")
        ax2.set_ylabel("# Positions", color="green")

    ax.set_xlabel("Date")
    ax.set_ylabel("Exposure (%)")
    ax.set_title("Portfolio Exposure")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return ax


def plot_weekly_returns(
    equity_curve: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 4),
    ax=None,
):
    """Plot weekly returns distribution.

    Args:
        equity_curve: DataFrame with 'weekly_return' column.
        figsize: Figure size.
        ax: Optional matplotlib axis.

    Returns:
        Matplotlib axis.
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    returns = equity_curve["weekly_return"] * 100

    # Histogram
    ax.hist(returns, bins=50, alpha=0.7, color="blue", edgecolor="black")

    # Add mean line
    mean_ret = returns.mean()
    ax.axvline(mean_ret, color="red", linestyle="--", linewidth=2,
               label=f"Mean: {mean_ret:.2f}%")

    ax.set_xlabel("Weekly Return (%)")
    ax.set_ylabel("Frequency")
    ax.set_title("Weekly Returns Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return ax


def plot_turnover(
    equity_curve: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 4),
    ax=None,
):
    """Plot turnover over time.

    Args:
        equity_curve: DataFrame with 'turnover' column.
        figsize: Figure size.
        ax: Optional matplotlib axis.

    Returns:
        Matplotlib axis.
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    dates = pd.to_datetime(equity_curve["week_end"])
    turnover = equity_curve["turnover"] * 100

    ax.bar(dates, turnover, width=5, alpha=0.7)

    # Rolling average
    rolling_avg = turnover.rolling(13).mean()  # 13-week rolling
    ax.plot(dates, rolling_avg, color="red", linewidth=2,
            label="13-week avg")

    ax.set_xlabel("Date")
    ax.set_ylabel("Turnover (%)")
    ax.set_title("Weekly Turnover")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return ax


def plot_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    figsize: Tuple[int, int] = (8, 6),
    ax=None,
):
    """Plot calibration curve.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.
        n_bins: Number of bins.
        figsize: Figure size.
        ax: Optional matplotlib axis.

    Returns:
        Matplotlib axis.
    """
    _check_matplotlib()
    from sklearn.calibration import calibration_curve

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    # Model calibration
    ax.plot(prob_pred, prob_true, "s-", color="blue", label="Model")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Reliability Diagram")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return ax


def plot_monthly_heatmap(
    equity_curve: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 8),
):
    """Plot monthly returns heatmap.

    Args:
        equity_curve: DataFrame with weekly returns.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    _check_matplotlib()
    from ..backtest.metrics import compute_monthly_returns

    monthly = compute_monthly_returns(equity_curve)

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap data (exclude Year column)
    data = monthly.iloc[:, :-1].values * 100  # Convert to percentage

    # Create heatmap
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=10)

    # Set ticks
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(monthly)))
    ax.set_yticklabels(monthly.index)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Return (%)")

    # Add text annotations
    for i in range(len(monthly)):
        for j in range(12):
            if not np.isnan(data[i, j]):
                text = ax.text(j, i, f"{data[i, j]:.1f}",
                               ha="center", va="center", fontsize=8)

    ax.set_title("Monthly Returns (%)")
    plt.tight_layout()

    return fig


def create_strategy_comparison_plot(
    results: Dict[str, pd.DataFrame],
    figsize: Tuple[int, int] = (12, 6),
):
    """Create comparison plot for multiple strategies.

    Args:
        results: Dict of strategy_name -> equity_curve DataFrame.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["blue", "orange", "green", "red", "purple"]

    for i, (name, equity_curve) in enumerate(results.items()):
        dates = pd.to_datetime(equity_curve["week_end"])
        nav = equity_curve["nav"]
        nav_normalized = nav / nav.iloc[0] * 100

        color = colors[i % len(colors)]
        ax.plot(dates, nav_normalized, label=name, linewidth=1.5, color=color)

    ax.set_xlabel("Date")
    ax.set_ylabel("NAV (normalized to 100)")
    ax.set_title("Strategy Comparison")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    plt.tight_layout()
    return fig
