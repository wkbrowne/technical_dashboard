"""Transaction cost models."""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

from ..config import CostConfig


@dataclass
class TransactionCostModel:
    """Transaction cost model.

    Computes costs as:
        cost = (spread + commission + slippage) * trade_value

    Attributes:
        spread_bps: Half-spread in basis points.
        commission_bps: Commission in basis points.
        slippage_bps: Market impact in basis points.
        min_cost: Minimum cost per trade.
    """
    spread_bps: float = 5.0
    commission_bps: float = 0.0
    slippage_bps: float = 5.0
    min_cost: float = 0.0

    @classmethod
    def from_config(cls, config: CostConfig) -> "TransactionCostModel":
        """Create from CostConfig."""
        return cls(
            spread_bps=config.spread_bps,
            commission_bps=config.commission_bps,
            slippage_bps=config.slippage_bps,
        )

    @property
    def one_way_bps(self) -> float:
        """Total one-way cost in basis points."""
        return self.spread_bps + self.commission_bps + self.slippage_bps

    @property
    def round_trip_bps(self) -> float:
        """Total round-trip cost in basis points."""
        return 2 * self.one_way_bps

    def compute_cost(
        self,
        trade_value: float,
        side: str = "buy",
    ) -> float:
        """Compute transaction cost for a trade.

        Args:
            trade_value: Absolute value of trade.
            side: 'buy' or 'sell'.

        Returns:
            Transaction cost in currency units.
        """
        cost = trade_value * (self.one_way_bps / 10000)
        return max(cost, self.min_cost)

    def compute_costs_series(
        self,
        trade_values: pd.Series,
    ) -> pd.Series:
        """Compute costs for multiple trades.

        Args:
            trade_values: Series of absolute trade values.

        Returns:
            Series of transaction costs.
        """
        costs = trade_values.abs() * (self.one_way_bps / 10000)
        return costs.clip(lower=self.min_cost)


def compute_transaction_cost(
    trade_value: float,
    config: CostConfig,
) -> float:
    """Convenience function to compute transaction cost.

    Args:
        trade_value: Absolute value of trade.
        config: Cost configuration.

    Returns:
        Transaction cost.
    """
    model = TransactionCostModel.from_config(config)
    return model.compute_cost(trade_value)


def estimate_market_impact(
    trade_value: float,
    adv: float,
    impact_coefficient: float = 0.1,
) -> float:
    """Estimate market impact for a trade.

    Uses square-root market impact model:
        impact = coefficient * sigma * sqrt(trade_size / ADV)

    Simplified version without volatility:
        impact = coefficient * sqrt(trade_fraction)

    Args:
        trade_value: Value of trade.
        adv: Average daily volume in dollars.
        impact_coefficient: Impact coefficient.

    Returns:
        Estimated market impact in basis points.
    """
    if adv <= 0:
        return 50.0  # High impact for illiquid

    trade_fraction = trade_value / adv
    impact = impact_coefficient * np.sqrt(trade_fraction)

    # Convert to basis points
    impact_bps = impact * 10000

    return min(impact_bps, 100.0)  # Cap at 1%


def compute_total_trading_cost(
    trades: pd.DataFrame,
    cost_model: TransactionCostModel,
    include_impact: bool = False,
) -> pd.DataFrame:
    """Compute total trading costs for a set of trades.

    Args:
        trades: DataFrame with 'value' and optionally 'adv' columns.
        cost_model: Transaction cost model.
        include_impact: Whether to include market impact.

    Returns:
        DataFrame with cost breakdown.
    """
    trades = trades.copy()

    # Base costs
    trades["spread_cost"] = trades["value"].abs() * (cost_model.spread_bps / 10000)
    trades["commission_cost"] = trades["value"].abs() * (cost_model.commission_bps / 10000)
    trades["slippage_cost"] = trades["value"].abs() * (cost_model.slippage_bps / 10000)

    # Market impact (if ADV available)
    if include_impact and "adv" in trades.columns:
        trades["impact_bps"] = trades.apply(
            lambda row: estimate_market_impact(row["value"], row.get("adv", 1e9)),
            axis=1
        )
        trades["impact_cost"] = trades["value"].abs() * (trades["impact_bps"] / 10000)
    else:
        trades["impact_cost"] = 0.0

    # Total
    trades["total_cost"] = (
        trades["spread_cost"] +
        trades["commission_cost"] +
        trades["slippage_cost"] +
        trades["impact_cost"]
    )

    return trades
