from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pandas as pd


# Standard CME Futures Contract Specifications
CONTRACT_SPECS: Dict[str, Dict[str, float]] = {
    "ES": {"point_value": 50.0, "tick_size": 0.25},
    "NQ": {"point_value": 20.0, "tick_size": 0.25},
    "MES": {"point_value": 5.0, "tick_size": 0.25},
    "MNQ": {"point_value": 2.0, "tick_size": 0.25},
}


@dataclass
class Position:
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    quantity: int
    stop_loss: float
    take_profit: float
    entry_time: pd.Timestamp
    initial_risk_points: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.initial_risk_points == 0.0 and self.stop_loss is not None:
            self.initial_risk_points = abs(self.entry_price - self.stop_loss)

    def unrealized_points(self, current_price: float) -> float:
        if self.direction == "LONG":
            return current_price - self.entry_price
        return self.entry_price - current_price


@dataclass
class TradeRecord:
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: int
    stop_loss: float
    take_profit: float
    exit_reason: str  # 'STOP_LOSS', 'TAKE_PROFIT', 'SIGNAL_EXIT', 'END_OF_DATA'
    points_pnl: float
    gross_pnl: float
    commission: float
    net_pnl: float
    r_multiple: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class Portfolio:
    """
    Manages account cash, open positions, realized/unrealized PnL,
    completed trade logs, and the bar-by-bar equity curve.
    """

    def __init__(self, initial_capital: float = 100_000.0):
        self.initial_capital: float = float(initial_capital)
        self.cash: float = float(initial_capital)
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[TradeRecord] = []
        self.equity_curve: List[Dict[str, Any]] = []

    @staticmethod
    def get_point_value(symbol: str) -> float:
        clean_sym = symbol.upper()
        if clean_sym in CONTRACT_SPECS:
            return CONTRACT_SPECS[clean_sym]["point_value"]
        return 1.0

    @staticmethod
    def get_tick_size(symbol: str) -> float:
        clean_sym = symbol.upper()
        if clean_sym in CONTRACT_SPECS:
            return CONTRACT_SPECS[clean_sym]["tick_size"]
        return 0.25

    def has_open_position(self, symbol: str) -> bool:
        return symbol in self.positions

    def open_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        quantity: int,
        stop_loss: float,
        take_profit: float,
        timestamp: pd.Timestamp,
        commission: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Position:
        if symbol in self.positions:
            raise ValueError(f"Position already open for symbol {symbol}.")

        pos = Position(
            symbol=symbol,
            direction=direction.upper(),
            entry_price=float(entry_price),
            quantity=int(quantity),
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            entry_time=timestamp,
            metadata=metadata or {},
        )
        self.positions[symbol] = pos
        self.cash -= float(commission)
        return pos

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        timestamp: pd.Timestamp,
        exit_reason: str,
        commission_per_side: float = 0.0,
    ) -> TradeRecord:
        if symbol not in self.positions:
            raise ValueError(f"No open position found for symbol {symbol}.")

        pos = self.positions.pop(symbol)
        point_val = self.get_point_value(symbol)

        points_pnl = pos.unrealized_points(float(exit_price))
        gross_pnl = points_pnl * point_val * pos.quantity
        total_commission = float(commission_per_side) * 2.0 * pos.quantity
        net_pnl = gross_pnl - total_commission

        # Deduct exit-side commission and add gross PnL (entry commission was deducted at open)
        self.cash += gross_pnl - (float(commission_per_side) * pos.quantity)

        r_mult = (
            points_pnl / pos.initial_risk_points
            if pos.initial_risk_points > 0
            else 0.0
        )

        record = TradeRecord(
            symbol=pos.symbol,
            direction=pos.direction,
            entry_time=pos.entry_time,
            exit_time=timestamp,
            entry_price=pos.entry_price,
            exit_price=float(exit_price),
            quantity=pos.quantity,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            exit_reason=exit_reason,
            points_pnl=points_pnl,
            gross_pnl=gross_pnl,
            commission=total_commission,
            net_pnl=net_pnl,
            r_multiple=r_mult,
            metadata=pos.metadata,
        )
        self.closed_trades.append(record)
        return record

    def update_mark_to_market(
        self, timestamp: pd.Timestamp, current_prices: Dict[str, float]
    ) -> float:
        """
        Calculates unrealized PnL across all open positions and appends
        a snapshot to the equity curve for Sharpe & Drawdown calculation.
        """
        unrealized_pnl = 0.0
        for sym, pos in self.positions.items():
            if sym in current_prices:
                pts = pos.unrealized_points(current_prices[sym])
                unrealized_pnl += pts * self.get_point_value(sym) * pos.quantity

        total_equity = self.cash + unrealized_pnl
        self.equity_curve.append(
            {
                "timestamp": timestamp,
                "cash": self.cash,
                "unrealized_pnl": unrealized_pnl,
                "equity": total_equity,
            }
        )
        return total_equity