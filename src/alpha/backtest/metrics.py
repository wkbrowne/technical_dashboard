"""Performance metrics computation."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    # Returns
    total_return: float
    cagr: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float

    # Risk
    max_drawdown: float
    calmar_ratio: float
    var_95: float
    cvar_95: float

    # Trade stats
    hit_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float

    # Turnover
    avg_turnover: float
    total_turnover: float

    # Other
    n_trades: int
    n_weeks: int
    best_week: float
    worst_week: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "cagr": self.cagr,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "hit_rate": self.hit_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "win_loss_ratio": self.win_loss_ratio,
            "avg_turnover": self.avg_turnover,
            "total_turnover": self.total_turnover,
            "n_trades": self.n_trades,
            "n_weeks": self.n_weeks,
            "best_week": self.best_week,
            "worst_week": self.worst_week,
        }


def compute_performance_metrics(
    equity_curve: pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 52.0,  # Weekly
) -> PerformanceMetrics:
    """Compute comprehensive performance metrics.

    Args:
        equity_curve: DataFrame with 'nav', 'weekly_return', 'turnover' columns.
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Number of periods per year (52 for weekly).

    Returns:
        PerformanceMetrics object.
    """
    returns = equity_curve["weekly_return"].values
    navs = equity_curve["nav"].values

    n_periods = len(returns)
    n_years = n_periods / periods_per_year

    # Total return
    total_return = (navs[-1] / navs[0] - 1) if len(navs) > 0 else 0

    # CAGR
    if n_years > 0 and navs[0] > 0:
        cagr = (navs[-1] / navs[0]) ** (1 / n_years) - 1
    else:
        cagr = 0

    # Volatility (annualized)
    volatility = np.std(returns) * np.sqrt(periods_per_year)

    # Sharpe ratio
    excess_returns = returns - risk_free_rate / periods_per_year
    if volatility > 0:
        sharpe_ratio = np.mean(excess_returns) * np.sqrt(periods_per_year) / volatility
    else:
        sharpe_ratio = 0

    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_vol = np.std(downside_returns) * np.sqrt(periods_per_year)
        if downside_vol > 0:
            sortino_ratio = np.mean(excess_returns) * np.sqrt(periods_per_year) / downside_vol
        else:
            sortino_ratio = np.inf
    else:
        sortino_ratio = np.inf

    # Maximum drawdown
    cumulative = (1 + pd.Series(returns)).cumprod()
    running_max = cumulative.cummax()
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = drawdowns.min()

    # Calmar ratio
    if max_drawdown < 0:
        calmar_ratio = cagr / abs(max_drawdown)
    else:
        calmar_ratio = np.inf

    # VaR and CVaR (95%)
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95

    # Hit rate
    wins = returns > 0
    hit_rate = wins.mean() if len(returns) > 0 else 0

    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Average win/loss
    avg_win = returns[returns > 0].mean() if wins.sum() > 0 else 0
    avg_loss = returns[returns < 0].mean() if (~wins).sum() > 0 else 0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

    # Turnover
    if "turnover" in equity_curve.columns:
        turnovers = equity_curve["turnover"].values
        avg_turnover = np.mean(turnovers)
        total_turnover = np.sum(turnovers)
    else:
        avg_turnover = 0
        total_turnover = 0

    # Trade count (from trades column if available)
    if "n_trades" in equity_curve.columns:
        n_trades = int(equity_curve["n_trades"].sum())
    else:
        n_trades = 0

    # Best/worst week
    best_week = returns.max() if len(returns) > 0 else 0
    worst_week = returns.min() if len(returns) > 0 else 0

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar_ratio,
        var_95=var_95,
        cvar_95=cvar_95,
        hit_rate=hit_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        win_loss_ratio=win_loss_ratio,
        avg_turnover=avg_turnover,
        total_turnover=total_turnover,
        n_trades=n_trades,
        n_weeks=n_periods,
        best_week=best_week,
        worst_week=worst_week,
    )


def compute_rolling_metrics(
    equity_curve: pd.DataFrame,
    window: int = 52,
) -> pd.DataFrame:
    """Compute rolling performance metrics.

    Args:
        equity_curve: DataFrame with 'weekly_return' column.
        window: Rolling window size.

    Returns:
        DataFrame with rolling metrics.
    """
    returns = equity_curve["weekly_return"]

    rolling = pd.DataFrame(index=equity_curve.index)

    # Rolling return
    rolling["rolling_return"] = (1 + returns).rolling(window).apply(
        lambda x: x.prod() - 1, raw=True
    )

    # Rolling volatility (annualized)
    rolling["rolling_volatility"] = returns.rolling(window).std() * np.sqrt(52)

    # Rolling Sharpe
    rolling_mean = returns.rolling(window).mean() * 52
    rolling_vol = rolling["rolling_volatility"]
    rolling["rolling_sharpe"] = rolling_mean / rolling_vol.replace(0, np.nan)

    # Rolling max drawdown
    cumulative = (1 + returns).cumprod()

    def rolling_max_dd(x):
        cum = (1 + x).cumprod()
        running_max = cum.cummax()
        dd = (cum - running_max) / running_max
        return dd.min()

    rolling["rolling_max_drawdown"] = returns.rolling(window).apply(
        rolling_max_dd, raw=False
    )

    # Rolling hit rate
    rolling["rolling_hit_rate"] = (returns > 0).rolling(window).mean()

    return rolling


def compute_drawdown_series(
    navs: pd.Series,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Compute drawdown time series and statistics.

    Args:
        navs: NAV time series.

    Returns:
        Tuple of (drawdown_series, drawdown_stats).
    """
    running_max = navs.cummax()
    drawdowns = (navs - running_max) / running_max

    # Find drawdown periods
    in_drawdown = drawdowns < 0
    drawdown_start = in_drawdown & ~in_drawdown.shift(1, fill_value=False)
    drawdown_end = ~in_drawdown & in_drawdown.shift(1, fill_value=False)

    # Compute stats for each drawdown period
    drawdown_stats = []
    current_start = None

    for i, (idx, is_start) in enumerate(drawdown_start.items()):
        if is_start:
            current_start = idx
        elif drawdown_end.iloc[i] and current_start is not None:
            period_dd = drawdowns.loc[current_start:idx]
            stats = {
                "start": current_start,
                "end": idx,
                "duration": (idx - current_start).days if hasattr(idx - current_start, 'days') else i,
                "max_drawdown": period_dd.min(),
                "recovery_time": None,  # Would need future data
            }
            drawdown_stats.append(stats)
            current_start = None

    return drawdowns, pd.DataFrame(drawdown_stats)


def compute_monthly_returns(
    equity_curve: pd.DataFrame,
    date_col: str = "week_end",
) -> pd.DataFrame:
    """Compute monthly return table.

    Args:
        equity_curve: DataFrame with date and NAV.
        date_col: Date column name.

    Returns:
        Pivot table with Year x Month returns.
    """
    df = equity_curve.copy()
    df["date"] = pd.to_datetime(df[date_col])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Get last NAV per month
    monthly = df.groupby(["year", "month"])["nav"].last().reset_index()

    # Compute monthly returns
    monthly["monthly_return"] = monthly["nav"].pct_change()

    # Pivot
    pivot = monthly.pivot(index="year", columns="month", values="monthly_return")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Add yearly total
    pivot["Year"] = df.groupby("year")["nav"].apply(
        lambda x: x.iloc[-1] / x.iloc[0] - 1 if len(x) > 1 else 0
    )

    return pivot


def print_metrics_summary(metrics: PerformanceMetrics) -> None:
    """Print formatted metrics summary."""
    print("\n" + "=" * 50)
    print("PERFORMANCE METRICS")
    print("=" * 50)

    print("\nReturns:")
    print(f"  Total Return:    {metrics.total_return:>10.2%}")
    print(f"  CAGR:            {metrics.cagr:>10.2%}")
    print(f"  Best Week:       {metrics.best_week:>10.2%}")
    print(f"  Worst Week:      {metrics.worst_week:>10.2%}")

    print("\nRisk:")
    print(f"  Volatility:      {metrics.volatility:>10.2%}")
    print(f"  Max Drawdown:    {metrics.max_drawdown:>10.2%}")
    print(f"  VaR (95%):       {metrics.var_95:>10.2%}")
    print(f"  CVaR (95%):      {metrics.cvar_95:>10.2%}")

    print("\nRisk-Adjusted:")
    print(f"  Sharpe Ratio:    {metrics.sharpe_ratio:>10.2f}")
    print(f"  Sortino Ratio:   {metrics.sortino_ratio:>10.2f}")
    print(f"  Calmar Ratio:    {metrics.calmar_ratio:>10.2f}")

    print("\nTrading:")
    print(f"  Hit Rate:        {metrics.hit_rate:>10.2%}")
    print(f"  Profit Factor:   {metrics.profit_factor:>10.2f}")
    print(f"  Win/Loss Ratio:  {metrics.win_loss_ratio:>10.2f}")
    print(f"  Avg Turnover:    {metrics.avg_turnover:>10.2%}")
    print(f"  Total Trades:    {metrics.n_trades:>10d}")

    print("=" * 50)
