"""Backtest engine with Monday entry / Friday rebalance protocol.

This module implements a realistic backtesting engine that:
    - Enters positions on Monday close
    - Makes rebalance decisions on Friday close
    - Tracks 20-day holding horizons
    - Supports early exits via barrier hits
    - Accounts for transaction costs
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ..config import AlphaConfig, BacktestConfig, CostConfig
from ..data.schemas import (
    WeeklySnapshot,
    PositionRecord,
    TradeRecord,
    BacktestResult,
    snapshots_to_dataframe,
    trades_to_dataframe,
)
from ..sizing.rules import BaseSizingRule, create_sizing_rule
from ..sizing.constraints import PortfolioConstraints, apply_constraints, compute_turnover
from .costs import TransactionCostModel
from .metrics import compute_performance_metrics, PerformanceMetrics


@dataclass
class Position:
    """Internal position tracking."""
    symbol: str
    entry_date: datetime
    entry_price: float
    shares: float
    target_weight: float
    target_price: float
    stop_price: float
    expected_exit_date: Optional[datetime] = None
    actual_exit_date: Optional[datetime] = None
    actual_exit_price: Optional[float] = None

    def get_value(self, current_price: float) -> float:
        """Get current position value."""
        return self.shares * current_price

    def get_pnl(self, current_price: float) -> float:
        """Get unrealized P&L."""
        return self.shares * (current_price - self.entry_price)


class BacktestEngine:
    """Backtest engine for weekly rebalancing strategy.

    Implements the Monday entry / Friday rebalance protocol with
    20-day holding horizons and barrier-based exits.

    Attributes:
        config: Alpha configuration.
        sizing_rule: Position sizing rule.
        cost_model: Transaction cost model.
        constraints: Portfolio constraints.
    """

    def __init__(
        self,
        config: AlphaConfig,
        sizing_rule: Optional[BaseSizingRule] = None,
    ):
        """Initialize backtest engine.

        Args:
            config: Alpha configuration.
            sizing_rule: Optional custom sizing rule.
        """
        self.config = config
        self.sizing_rule = sizing_rule or create_sizing_rule(config.sizing)
        self.cost_model = TransactionCostModel.from_config(config.costs)
        self.constraints = PortfolioConstraints(
            max_gross_exposure=config.sizing.max_gross_exposure,
            max_name_weight=config.sizing.max_name_weight,
            min_name_weight=config.sizing.min_name_weight,
            cash_buffer=config.sizing.cash_buffer,
            sector_caps=config.sizing.sector_caps,
            max_positions=config.sizing.top_n,
        )

        # State
        self.nav = config.backtest.initial_capital
        self.cash = config.backtest.initial_capital
        self.positions: Dict[str, Position] = {}
        self.snapshots: List[WeeklySnapshot] = []
        self.all_trades: List[TradeRecord] = []

    def run(
        self,
        signals: pd.DataFrame,
        prices: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """Run backtest.

        Args:
            signals: Prepared weekly signals with outcomes.
            prices: Optional daily prices for MTM (not required if
                    signals have actual_exit_date and actual_exit_price).

        Returns:
            BacktestResult with full history.
        """
        # Reset state
        self.nav = self.config.backtest.initial_capital
        self.cash = self.config.backtest.initial_capital
        self.positions = {}
        self.snapshots = []
        self.all_trades = []

        # Get unique weeks
        weeks = sorted(signals["week_monday"].unique())

        print(f"Running backtest over {len(weeks)} weeks...")
        print(f"Initial capital: ${self.nav:,.0f}")

        for i, week_monday in enumerate(weeks):
            week_signals = signals[signals["week_monday"] == week_monday].copy()

            # Process the week
            self._process_week(week_monday, week_signals)

            if (i + 1) % 26 == 0:  # Every 6 months
                print(f"  Week {i+1}/{len(weeks)}: NAV = ${self.nav:,.0f}")

        # Final snapshot
        print(f"\nFinal NAV: ${self.nav:,.0f}")
        print(f"Total return: {(self.nav / self.config.backtest.initial_capital - 1) * 100:.1f}%")

        # Create result
        equity_curve = snapshots_to_dataframe(self.snapshots)

        # Compute metrics
        if len(equity_curve) > 0:
            metrics = compute_performance_metrics(equity_curve)
        else:
            metrics = {}

        return BacktestResult(
            config={"sizing": self.config.sizing.__dict__},
            snapshots=self.snapshots,
            all_trades=self.all_trades,
            metrics=metrics.to_dict() if hasattr(metrics, 'to_dict') else {},
            equity_curve=equity_curve,
        )

    def _process_week(
        self,
        week_monday: datetime,
        week_signals: pd.DataFrame,
    ) -> None:
        """Process a single week.

        Args:
            week_monday: Monday date of the week.
            week_signals: Signals for this week.
        """
        week_friday = week_monday + timedelta(days=4)

        # 1. Check for exits (barrier hits or horizon expiry)
        self._process_exits(week_monday)

        # 2. Mark to market existing positions
        position_value = self._mark_to_market(week_signals)

        # 3. Compute target weights for new signals
        if len(week_signals) > 0:
            target_weights = self._compute_target_weights(week_signals)
        else:
            target_weights = pd.Series(dtype=float)

        # 4. Rebalance portfolio
        trades = self._rebalance(week_signals, target_weights)

        # 5. Record snapshot
        self._record_snapshot(week_monday, week_friday, trades)

    def _process_exits(self, current_date: datetime) -> None:
        """Process position exits."""
        to_close = []

        for symbol, pos in self.positions.items():
            should_exit = False
            exit_type = None

            # Check if actual exit date has passed
            if pos.actual_exit_date is not None:
                if pd.Timestamp(current_date) >= pd.Timestamp(pos.actual_exit_date):
                    should_exit = True
                    exit_type = "barrier"

            # Check horizon expiry (20 days from entry)
            if pos.expected_exit_date is not None:
                if pd.Timestamp(current_date) >= pd.Timestamp(pos.expected_exit_date):
                    should_exit = True
                    exit_type = exit_type or "horizon"

            if should_exit:
                to_close.append((symbol, exit_type))

        # Close positions
        for symbol, exit_type in to_close:
            self._close_position(symbol, current_date, exit_type)

    def _close_position(
        self,
        symbol: str,
        date: datetime,
        exit_type: str,
    ) -> None:
        """Close a position."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]

        # Use actual exit price if available
        if pos.actual_exit_price is not None:
            exit_price = pos.actual_exit_price
        else:
            exit_price = pos.entry_price  # Fallback

        # Calculate proceeds
        gross_proceeds = pos.shares * exit_price
        cost = self.cost_model.compute_cost(gross_proceeds, "sell")
        net_proceeds = gross_proceeds - cost

        # Update cash
        self.cash += net_proceeds

        # Record trade
        trade = TradeRecord(
            symbol=symbol,
            trade_date=date,
            side="sell",
            shares=pos.shares,
            price=exit_price,
            value=gross_proceeds,
            cost=cost,
            net_value=net_proceeds,
            reason=f"exit_{exit_type}",
        )
        self.all_trades.append(trade)

        # Remove position
        del self.positions[symbol]

    def _mark_to_market(self, week_signals: pd.DataFrame) -> float:
        """Mark positions to market and return total value."""
        total_value = 0.0

        for symbol, pos in self.positions.items():
            # Get current price from signals if available
            symbol_signal = week_signals[week_signals["symbol"] == symbol]
            if len(symbol_signal) > 0:
                current_price = symbol_signal.iloc[0]["entry_price"]
            else:
                current_price = pos.entry_price  # Use entry as fallback

            total_value += pos.get_value(current_price)

        return total_value

    def _compute_target_weights(
        self,
        week_signals: pd.DataFrame,
    ) -> pd.Series:
        """Compute target weights for signals."""
        # Apply sizing rule
        raw_weights = self.sizing_rule.compute_weights(week_signals)

        # Apply constraints
        constrained_weights = apply_constraints(
            raw_weights, week_signals, self.constraints
        )

        return constrained_weights

    def _rebalance(
        self,
        week_signals: pd.DataFrame,
        target_weights: pd.Series,
    ) -> List[TradeRecord]:
        """Rebalance portfolio to target weights."""
        trades = []

        # Current position weights
        current_weights = {}
        for symbol, pos in self.positions.items():
            # Approximate weight
            if self.nav > 0:
                current_weights[symbol] = pos.get_value(pos.entry_price) / self.nav

        # Compute trades needed
        for idx in target_weights.index:
            if target_weights[idx] <= 0:
                continue

            signal = week_signals.loc[idx]
            symbol = signal["symbol"]
            target_weight = target_weights[idx]
            current_weight = current_weights.get(symbol, 0)

            # Weight difference
            weight_diff = target_weight - current_weight

            if abs(weight_diff) < 0.005:  # Skip small rebalances
                continue

            # Calculate trade
            trade_value = weight_diff * self.nav
            price = signal["entry_price"]

            if price <= 0:
                continue

            shares = abs(trade_value) / price

            if trade_value > 0:
                # Buy
                cost = self.cost_model.compute_cost(trade_value, "buy")

                if self.cash >= trade_value + cost:
                    self.cash -= (trade_value + cost)

                    # Create or update position
                    if symbol in self.positions:
                        pos = self.positions[symbol]
                        total_shares = pos.shares + shares
                        avg_price = (pos.entry_price * pos.shares + price * shares) / total_shares
                        pos.shares = total_shares
                        pos.entry_price = avg_price
                    else:
                        # New position
                        pos = Position(
                            symbol=symbol,
                            entry_date=signal["week_monday"],
                            entry_price=price,
                            shares=shares,
                            target_weight=target_weight,
                            target_price=signal.get("target_price", price * 1.1),
                            stop_price=signal.get("stop_price", price * 0.9),
                            expected_exit_date=signal["week_monday"] + timedelta(days=28),
                            actual_exit_date=signal.get("actual_exit_date"),
                            actual_exit_price=signal.get("actual_exit_price"),
                        )
                        self.positions[symbol] = pos

                    trade = TradeRecord(
                        symbol=symbol,
                        trade_date=signal["week_monday"],
                        side="buy",
                        shares=shares,
                        price=price,
                        value=trade_value,
                        cost=cost,
                        net_value=trade_value + cost,
                        reason="entry",
                    )
                    trades.append(trade)

        self.all_trades.extend(trades)
        return trades

    def _record_snapshot(
        self,
        week_start: datetime,
        week_end: datetime,
        trades: List[TradeRecord],
    ) -> None:
        """Record weekly snapshot."""
        # Calculate NAV
        position_value = sum(
            pos.get_value(pos.entry_price) for pos in self.positions.values()
        )
        self.nav = self.cash + position_value

        # Calculate exposure
        gross_exposure = position_value / self.nav if self.nav > 0 else 0

        # Calculate return
        if len(self.snapshots) > 0:
            prev_nav = self.snapshots[-1].nav
            weekly_return = (self.nav / prev_nav - 1) if prev_nav > 0 else 0
        else:
            weekly_return = 0

        # Cumulative return
        initial_nav = self.config.backtest.initial_capital
        cumulative_return = (self.nav / initial_nav - 1) if initial_nav > 0 else 0

        # Turnover
        trade_value = sum(abs(t.value) for t in trades)
        turnover = trade_value / self.nav if self.nav > 0 else 0

        snapshot = WeeklySnapshot(
            week_start=week_start,
            week_end=week_end,
            nav=self.nav,
            cash=self.cash,
            gross_exposure=gross_exposure,
            net_exposure=gross_exposure,  # Long-only
            n_positions=len(self.positions),
            positions={},  # Simplified
            trades=trades,
            weekly_return=weekly_return,
            cumulative_return=cumulative_return,
            realized_pnl=0,  # Would need more tracking
            unrealized_pnl=0,
            turnover=turnover,
        )
        self.snapshots.append(snapshot)


def run_backtest(
    signals: pd.DataFrame,
    config: AlphaConfig,
    sizing_rule: Optional[BaseSizingRule] = None,
) -> BacktestResult:
    """Convenience function to run backtest.

    Args:
        signals: Prepared weekly signals.
        config: Alpha configuration.
        sizing_rule: Optional custom sizing rule.

    Returns:
        BacktestResult.
    """
    engine = BacktestEngine(config, sizing_rule)
    return engine.run(signals)


def compare_strategies(
    signals: pd.DataFrame,
    config: AlphaConfig,
    strategies: Dict[str, BaseSizingRule],
) -> Dict[str, BacktestResult]:
    """Compare multiple sizing strategies.

    Args:
        signals: Prepared weekly signals.
        config: Alpha configuration.
        strategies: Dict of strategy_name -> sizing_rule.

    Returns:
        Dict of strategy_name -> BacktestResult.
    """
    results = {}

    for name, rule in strategies.items():
        print(f"\nRunning strategy: {name}")
        engine = BacktestEngine(config, rule)
        results[name] = engine.run(signals)

    return results


def run_baseline_comparison(
    signals: pd.DataFrame,
    config: AlphaConfig,
) -> Dict[str, BacktestResult]:
    """Run comparison with baseline strategies.

    Compares:
        - equal_weight: Equal weight all selected signals
        - prob_weight: Weight proportional to probability
        - optimized: Configured sizing rule

    Args:
        signals: Prepared weekly signals.
        config: Alpha configuration.

    Returns:
        Dict of strategy -> BacktestResult.
    """
    from ..sizing.rules import compute_equal_weights, compute_probability_weights

    # Create baseline sizing rules
    class EqualWeightRule(BaseSizingRule):
        def __init__(self, top_n, max_weight):
            self.top_n = top_n
            self.max_weight = max_weight

        def compute_weights(self, signals, **kwargs):
            return compute_equal_weights(signals, self.top_n, self.max_weight)

    class ProbWeightRule(BaseSizingRule):
        def __init__(self, top_n, max_weight):
            self.top_n = top_n
            self.max_weight = max_weight

        def compute_weights(self, signals, **kwargs):
            return compute_probability_weights(signals, self.top_n, self.max_weight)

    strategies = {
        "equal_weight": EqualWeightRule(config.sizing.top_n, config.sizing.max_name_weight),
        "prob_weight": ProbWeightRule(config.sizing.top_n, config.sizing.max_name_weight),
        "optimized": create_sizing_rule(config.sizing),
    }

    return compare_strategies(signals, config, strategies)
