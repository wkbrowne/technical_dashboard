"""Data schemas and types for alpha module.

This module defines the core data structures used throughout the
position sizing and backtesting pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class SignalRecord:
    """A single trading signal.

    Attributes:
        symbol: Ticker symbol.
        signal_date: Date signal was generated (Monday close).
        entry_date: Date of entry (next trading day).
        probability: Model probability score.
        calibrated_prob: Calibrated probability (if available).
        entry_price: Entry price.
        target_price: Upper barrier price.
        stop_price: Lower barrier price.
        atr_pct: ATR as percentage of price.
        sector: Sector classification (optional).
        features: Dict of feature values used for sizing.
    """
    symbol: str
    signal_date: datetime
    entry_date: datetime
    probability: float
    calibrated_prob: Optional[float] = None
    entry_price: float = 0.0
    target_price: float = 0.0
    stop_price: float = 0.0
    atr_pct: float = 0.0
    sector: Optional[str] = None
    features: Dict[str, float] = field(default_factory=dict)

    @property
    def reward_risk_ratio(self) -> float:
        """Calculate reward/risk ratio."""
        upside = (self.target_price / self.entry_price - 1) if self.entry_price > 0 else 0
        downside = (1 - self.stop_price / self.entry_price) if self.entry_price > 0 else 0
        return upside / max(downside, 0.001)


@dataclass
class PositionRecord:
    """A position in the portfolio.

    Attributes:
        symbol: Ticker symbol.
        entry_date: Date position was opened.
        entry_price: Entry price per share.
        shares: Number of shares.
        target_weight: Target portfolio weight.
        actual_weight: Current portfolio weight.
        current_price: Current price.
        unrealized_pnl: Unrealized P&L.
        exit_date: Exit date (if closed).
        exit_price: Exit price (if closed).
        exit_type: 'target', 'stop', 'horizon', or 'rebalance'.
        holding_days: Days held.
    """
    symbol: str
    entry_date: datetime
    entry_price: float
    shares: float
    target_weight: float
    actual_weight: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_type: Optional[str] = None
    holding_days: int = 0

    @property
    def is_open(self) -> bool:
        """Check if position is still open."""
        return self.exit_date is None

    @property
    def realized_pnl(self) -> float:
        """Calculate realized P&L for closed positions."""
        if not self.is_open and self.exit_price is not None:
            return (self.exit_price - self.entry_price) * self.shares
        return 0.0

    @property
    def return_pct(self) -> float:
        """Calculate percentage return."""
        if self.exit_price is not None and self.entry_price > 0:
            return (self.exit_price / self.entry_price - 1) * 100
        elif self.current_price > 0 and self.entry_price > 0:
            return (self.current_price / self.entry_price - 1) * 100
        return 0.0


@dataclass
class TradeRecord:
    """A single trade execution.

    Attributes:
        symbol: Ticker symbol.
        trade_date: Execution date.
        side: 'buy' or 'sell'.
        shares: Number of shares traded.
        price: Execution price.
        value: Total trade value (before costs).
        cost: Transaction costs.
        net_value: Trade value after costs.
        reason: Trade reason (entry, exit, rebalance).
    """
    symbol: str
    trade_date: datetime
    side: str  # 'buy' or 'sell'
    shares: float
    price: float
    value: float
    cost: float
    net_value: float
    reason: str  # 'entry', 'exit', 'rebalance'

    @property
    def cost_bps(self) -> float:
        """Cost as basis points of trade value."""
        if self.value > 0:
            return (self.cost / self.value) * 10000
        return 0.0


@dataclass
class WeeklySnapshot:
    """Weekly portfolio snapshot.

    Attributes:
        week_start: Monday date.
        week_end: Friday date.
        nav: Net asset value.
        cash: Cash balance.
        gross_exposure: Total position value / NAV.
        net_exposure: (Long - Short) / NAV.
        n_positions: Number of positions.
        positions: Dict of symbol -> PositionRecord.
        trades: List of trades executed this week.
        weekly_return: Return for the week.
        cumulative_return: Cumulative return since inception.
        realized_pnl: Realized P&L this week.
        unrealized_pnl: Unrealized P&L.
        turnover: Turnover as fraction of NAV.
        volatility_realized: Realized volatility (rolling).
    """
    week_start: datetime
    week_end: datetime
    nav: float
    cash: float
    gross_exposure: float
    net_exposure: float
    n_positions: int
    positions: Dict[str, PositionRecord] = field(default_factory=dict)
    trades: List[TradeRecord] = field(default_factory=list)
    weekly_return: float = 0.0
    cumulative_return: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    turnover: float = 0.0
    volatility_realized: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtest results.

    Attributes:
        config: Configuration used.
        snapshots: List of weekly snapshots.
        all_trades: List of all trades.
        metrics: Performance metrics dict.
        equity_curve: DataFrame with daily/weekly equity values.
        position_history: DataFrame with position history.
    """
    config: dict
    snapshots: List[WeeklySnapshot]
    all_trades: List[TradeRecord]
    metrics: Dict[str, float]
    equity_curve: pd.DataFrame
    position_history: Optional[pd.DataFrame] = None

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "config": self.config,
            "metrics": self.metrics,
            "n_weeks": len(self.snapshots),
            "n_trades": len(self.all_trades),
        }


def signals_to_dataframe(signals: List[SignalRecord]) -> pd.DataFrame:
    """Convert list of signals to DataFrame."""
    records = []
    for s in signals:
        record = {
            "symbol": s.symbol,
            "signal_date": s.signal_date,
            "entry_date": s.entry_date,
            "probability": s.probability,
            "calibrated_prob": s.calibrated_prob,
            "entry_price": s.entry_price,
            "target_price": s.target_price,
            "stop_price": s.stop_price,
            "atr_pct": s.atr_pct,
            "sector": s.sector,
            "reward_risk": s.reward_risk_ratio,
        }
        records.append(record)
    return pd.DataFrame(records)


def trades_to_dataframe(trades: List[TradeRecord]) -> pd.DataFrame:
    """Convert list of trades to DataFrame."""
    records = []
    for t in trades:
        record = {
            "symbol": t.symbol,
            "trade_date": t.trade_date,
            "side": t.side,
            "shares": t.shares,
            "price": t.price,
            "value": t.value,
            "cost": t.cost,
            "net_value": t.net_value,
            "reason": t.reason,
            "cost_bps": t.cost_bps,
        }
        records.append(record)
    return pd.DataFrame(records)


def snapshots_to_dataframe(snapshots: List[WeeklySnapshot]) -> pd.DataFrame:
    """Convert list of snapshots to DataFrame (equity curve)."""
    records = []
    for s in snapshots:
        record = {
            "week_start": s.week_start,
            "week_end": s.week_end,
            "nav": s.nav,
            "cash": s.cash,
            "gross_exposure": s.gross_exposure,
            "net_exposure": s.net_exposure,
            "n_positions": s.n_positions,
            "weekly_return": s.weekly_return,
            "cumulative_return": s.cumulative_return,
            "realized_pnl": s.realized_pnl,
            "unrealized_pnl": s.unrealized_pnl,
            "turnover": s.turnover,
            "volatility_realized": s.volatility_realized,
        }
        records.append(record)
    df = pd.DataFrame(records)
    if len(df) > 0:
        df["week_start"] = pd.to_datetime(df["week_start"])
        df["week_end"] = pd.to_datetime(df["week_end"])
    return df
