from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pandas as pd
from engine.portfolio import Portfolio, Position, TradeRecord


@dataclass
class Order:
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    order_type: str  # 'MARKET', 'LIMIT', 'STOP'
    quantity: float
    timestamp: pd.Timestamp
    stop_loss: float
    take_profit: float
    limit_price: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimulatedBroker:
    """
    Executes market/limit orders with realistic slippage and commissions,
    and processes intrabar Stop-Loss, multi-stage Partial Take-Profits (TP1/TP2),
    and Full Take-Profit fills on every 1-minute candle.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        commission_per_contract: float = 2.00,
        slippage_ticks: int = 1,
    ):
        self.portfolio = portfolio
        self.commission_per_contract = float(commission_per_contract)
        self.slippage_ticks = int(slippage_ticks)
        self.pending_orders: List[Order] = []

    def submit_order(
        self,
        symbol: str,
        direction: str,
        order_type: str,
        quantity: float,
        stop_loss: float,
        take_profit: float,
        timestamp: pd.Timestamp,
        current_close: float,
        limit_price: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Position]:
        clean_dir = direction.upper()
        clean_type = order_type.upper()
        qty = float(quantity)
        if qty <= 0:
            return None

        tick_size = self.portfolio.get_tick_size(symbol)
        slip_pts = self.slippage_ticks * tick_size

        if clean_type == "MARKET":
            fill_price = current_close + slip_pts if clean_dir == "LONG" else current_close - slip_pts
            entry_comm = self.commission_per_contract * qty
            return self.portfolio.open_position(
                symbol=symbol,
                direction=clean_dir,
                entry_price=fill_price,
                quantity=qty,
                stop_loss=float(stop_loss),
                take_profit=float(take_profit),
                timestamp=timestamp,
                commission=entry_comm,
                metadata=metadata or {},
            )

        order = Order(
            symbol=symbol,
            direction=clean_dir,
            order_type=clean_type,
            quantity=qty,
            timestamp=timestamp,
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            limit_price=float(limit_price) if limit_price is not None else None,
            metadata=metadata or {},
        )
        self.pending_orders.append(order)
        return None

    def process_candle(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        open_p: float,
        high_p: float,
        low_p: float,
        close_p: float,
    ) -> List[TradeRecord]:
        closed_trades: List[TradeRecord] = []
        tick_size = self.portfolio.get_tick_size(symbol)
        slip_pts = self.slippage_ticks * tick_size

        # 1. Process any pending limit/stop entry orders
        if self.pending_orders and not self.portfolio.has_open_position(symbol):
            remaining_orders: List[Order] = []
            for order in self.pending_orders:
                if order.symbol != symbol:
                    remaining_orders.append(order)
                    continue
                filled = False
                fill_price = 0.0
                if order.order_type == "LIMIT" and order.limit_price is not None:
                    if order.direction == "LONG" and low_p <= order.limit_price:
                        fill_price = min(open_p, order.limit_price)
                        filled = True
                    elif order.direction == "SHORT" and high_p >= order.limit_price:
                        fill_price = max(open_p, order.limit_price)
                        filled = True
                if filled and not self.portfolio.has_open_position(symbol):
                    entry_comm = self.commission_per_contract * order.quantity
                    self.portfolio.open_position(
                        symbol=order.symbol,
                        direction=order.direction,
                        entry_price=fill_price,
                        quantity=order.quantity,
                        stop_loss=order.stop_loss,
                        take_profit=order.take_profit,
                        timestamp=timestamp,
                        commission=entry_comm,
                        metadata=order.metadata,
                    )
                else:
                    remaining_orders.append(order)
            self.pending_orders = remaining_orders

        # 2. Process open position Stop-Loss, Partial TPs (TP1 / TP2), and Full Take-Profit
        if not self.portfolio.has_open_position(symbol):
            return closed_trades

        pos = self.portfolio.positions[symbol]
        if timestamp <= pos.entry_time:
            return closed_trades

        if pos.direction == "LONG":
            # Conservative intrabar rule: check Stop-Loss first
            if low_p <= pos.stop_loss:
                exit_fill = pos.stop_loss - slip_pts
                rec = self.portfolio.close_position(
                    symbol=symbol,
                    exit_price=exit_fill,
                    timestamp=timestamp,
                    exit_reason="STOP_LOSS",
                    commission_per_side=self.commission_per_contract,
                )
                closed_trades.append(rec)
                return closed_trades

            # Check intermediate Partial Take-Profits (TP1, TP2)
            for pt in pos.partial_targets:
                if not pt.get("hit", False) and high_p >= float(pt["price"]):
                    pt["hit"] = True
                    self.portfolio.record_partial_exit(
                        symbol=symbol,
                        exit_price=float(pt["price"]),
                        weight=float(pt["weight"]),
                        timestamp=timestamp,
                        label=str(pt.get("label", "TP")),
                        target_type=str(pt.get("target_type", "")),
                        commission_per_side=self.commission_per_contract,
                    )

            # Check Full Take-Profit
            if high_p >= pos.take_profit:
                rec = self.portfolio.close_position(
                    symbol=symbol,
                    exit_price=pos.take_profit,
                    timestamp=timestamp,
                    exit_reason="TAKE_PROFIT",
                    commission_per_side=self.commission_per_contract,
                )
                closed_trades.append(rec)
                return closed_trades

        else:
            # SHORT position: check Stop-Loss first
            if high_p >= pos.stop_loss:
                exit_fill = pos.stop_loss + slip_pts
                rec = self.portfolio.close_position(
                    symbol=symbol,
                    exit_price=exit_fill,
                    timestamp=timestamp,
                    exit_reason="STOP_LOSS",
                    commission_per_side=self.commission_per_contract,
                )
                closed_trades.append(rec)
                return closed_trades

            # Check intermediate Partial Take-Profits (TP1, TP2)
            for pt in pos.partial_targets:
                if not pt.get("hit", False) and low_p <= float(pt["price"]):
                    pt["hit"] = True
                    self.portfolio.record_partial_exit(
                        symbol=symbol,
                        exit_price=float(pt["price"]),
                        weight=float(pt["weight"]),
                        timestamp=timestamp,
                        label=str(pt.get("label", "TP")),
                        target_type=str(pt.get("target_type", "")),
                        commission_per_side=self.commission_per_contract,
                    )

            # Check Full Take-Profit
            if low_p <= pos.take_profit:
                rec = self.portfolio.close_position(
                    symbol=symbol,
                    exit_price=pos.take_profit,
                    timestamp=timestamp,
                    exit_reason="TAKE_PROFIT",
                    commission_per_side=self.commission_per_contract,
                )
                closed_trades.append(rec)
                return closed_trades

        return closed_trades