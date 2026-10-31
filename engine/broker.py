from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pandas as pd
from engine.portfolio import Portfolio, Position, TradeRecord


@dataclass
class Order:
    order_id: int
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    order_type: str  # 'MARKET' or 'LIMIT'
    quantity: int
    stop_loss: float
    take_profit: float
    created_time: pd.Timestamp
    limit_price: Optional[float] = None
    status: str = "PENDING"  # 'PENDING', 'FILLED', 'CANCELLED'
    filled_time: Optional[pd.Timestamp] = None
    filled_price: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimulatedBroker:
    """
    Executes Market and Limit orders and monitors intrabar Stop-Loss (SL)
    and Take-Profit (TP) levels with strict Zero Look-Ahead protection.
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
        self.order_history: List[Order] = []
        self._next_order_id: int = 1

    def _apply_slippage(self, symbol: str, price: float, side: str) -> float:
        """
        Applies adverse slippage on market and stop executions:
        - Buying ('BUY') fills slightly higher.
        - Selling ('SELL') fills slightly lower.
        """
        tick_size = self.portfolio.get_tick_size(symbol)
        slip = self.slippage_ticks * tick_size
        if side.upper() == "BUY":
            return float(price + slip)
        return float(price - slip)

    def submit_order(
        self,
        symbol: str,
        direction: str,
        order_type: str,
        quantity: int,
        stop_loss: float,
        take_profit: float,
        timestamp: pd.Timestamp,
        limit_price: Optional[float] = None,
        current_close: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Order]:
        """
        Submits a new order.
        - If order_type == 'MARKET' and current_close is provided, fills immediately
          at current_close (with slippage) on the closed bar.
        - If order_type == 'LIMIT' (or MARKET without current_close), queues in
          pending_orders to be evaluated on subsequent candles.
        """
        direction = direction.upper()
        order_type = order_type.upper()

        if self.portfolio.has_open_position(symbol):
            return None

        # Validate SL/TP geometry before accepting order
        ref_price = limit_price if order_type == "LIMIT" and limit_price is not None else current_close
        if ref_price is not None:
            if direction == "LONG" and not (stop_loss < ref_price < take_profit):
                raise ValueError(
                    f"Invalid LONG bracket: stop_loss ({stop_loss}) must be < entry ({ref_price}) < take_profit ({take_profit})."
                )
            if direction == "SHORT" and not (take_profit < ref_price < stop_loss):
                raise ValueError(
                    f"Invalid SHORT bracket: take_profit ({take_profit}) must be < entry ({ref_price}) < stop_loss ({stop_loss})."
                )

        order = Order(
            order_id=self._next_order_id,
            symbol=symbol,
            direction=direction,
            order_type=order_type,
            quantity=int(quantity),
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            created_time=timestamp,
            limit_price=float(limit_price) if limit_price is not None else None,
            metadata=metadata or {},
        )
        self._next_order_id += 1

        if order_type == "MARKET" and current_close is not None:
            entry_side = "BUY" if direction == "LONG" else "SELL"
            fill_price = self._apply_slippage(symbol, current_close, entry_side)
            self._fill_order(order, fill_price, timestamp)
            return order

        self.pending_orders.append(order)
        return order

    def _fill_order(self, order: Order, fill_price: float, timestamp: pd.Timestamp) -> Position:
        order.status = "FILLED"
        order.filled_price = float(fill_price)
        order.filled_time = timestamp
        self.order_history.append(order)

        entry_commission = self.commission_per_contract * order.quantity
        return self.portfolio.open_position(
            symbol=order.symbol,
            direction=order.direction,
            entry_price=fill_price,
            quantity=order.quantity,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            timestamp=timestamp,
            commission=entry_commission,
            metadata=order.metadata,
        )

    def cancel_pending_orders(self, symbol: Optional[str] = None) -> None:
        remaining = []
        for order in self.pending_orders:
            if symbol is None or order.symbol == symbol:
                order.status = "CANCELLED"
                self.order_history.append(order)
            else:
                remaining.append(order)
        self.pending_orders = remaining

    def process_candle(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        open_p: float,
        high_p: float,
        low_p: float,
        close_p: float,
    ) -> List[TradeRecord]:
        """
        Evaluates a new OHLC candle for a given symbol:
        1. Checks existing open positions (entered strictly before `timestamp`)
           for Stop-Loss and Take-Profit hits, using conservative SL priority.
        2. Evaluates pending Market/Limit orders created strictly before `timestamp`.
        """
        closed_this_bar: List[TradeRecord] = []

        # Check SL / TP on existing open position
        if self.portfolio.has_open_position(symbol):
            pos = self.portfolio.positions[symbol]
            if timestamp > pos.entry_time:
                trade = self._check_position_exit(pos, timestamp, open_p, high_p, low_p)
                if trade is not None:
                    closed_this_bar.append(trade)

        # Check pending orders if no position is currently open for this symbol
        if not self.portfolio.has_open_position(symbol) and self.pending_orders:
            still_pending: List[Order] = []
            for order in self.pending_orders:
                if order.symbol != symbol or timestamp <= order.created_time:
                    still_pending.append(order)
                    continue

                if self.portfolio.has_open_position(symbol):
                    still_pending.append(order)
                    continue

                if order.order_type == "MARKET":
                    entry_side = "BUY" if order.direction == "LONG" else "SELL"
                    fill_price = self._apply_slippage(symbol, open_p, entry_side)
                    pos = self._fill_order(order, fill_price, timestamp)
                    # Check if the same bar immediately stops out the new position
                    trade = self._check_position_exit(pos, timestamp, open_p, high_p, low_p)
                    if trade is not None:
                        closed_this_bar.append(trade)

                elif order.order_type == "LIMIT" and order.limit_price is not None:
                    if order.direction == "LONG" and low_p <= order.limit_price:
                        fill_price = float(min(open_p, order.limit_price))
                        pos = self._fill_order(order, fill_price, timestamp)
                        trade = self._check_position_exit(pos, timestamp, open_p, high_p, low_p)
                        if trade is not None:
                            closed_this_bar.append(trade)
                    elif order.direction == "SHORT" and high_p >= order.limit_price:
                        fill_price = float(max(open_p, order.limit_price))
                        pos = self._fill_order(order, fill_price, timestamp)
                        trade = self._check_position_exit(pos, timestamp, open_p, high_p, low_p)
                        if trade is not None:
                            closed_this_bar.append(trade)
                    else:
                        still_pending.append(order)

            self.pending_orders = still_pending

        return closed_this_bar

    def _check_position_exit(
        self,
        pos: Position,
        timestamp: pd.Timestamp,
        open_p: float,
        high_p: float,
        low_p: float,
    ) -> Optional[TradeRecord]:
        """
        Checks if a candle triggers Stop-Loss or Take-Profit.
        Enforces conservative intrabar execution: if both SL and TP are touched
        within the same 1-minute candle, Stop-Loss takes priority.
        """
        symbol = pos.symbol

        if pos.direction == "LONG":
            sl_hit = low_p <= pos.stop_loss
            tp_hit = high_p >= pos.take_profit

            if sl_hit:
                # If candle opens below SL, fill at open_p (gap slippage); else at SL
                base_exit = min(open_p, pos.stop_loss)
                exit_price = self._apply_slippage(symbol, base_exit, "SELL")
                return self.portfolio.close_position(
                    symbol, exit_price, timestamp, "STOP_LOSS", self.commission_per_contract
                )
            if tp_hit:
                exit_price = max(open_p, pos.take_profit)
                return self.portfolio.close_position(
                    symbol, exit_price, timestamp, "TAKE_PROFIT", self.commission_per_contract
                )

        elif pos.direction == "SHORT":
            sl_hit = high_p >= pos.stop_loss
            tp_hit = low_p <= pos.take_profit

            if sl_hit:
                # If candle opens above SL, fill at open_p (gap slippage); else at SL
                base_exit = max(open_p, pos.stop_loss)
                exit_price = self._apply_slippage(symbol, base_exit, "BUY")
                return self.portfolio.close_position(
                    symbol, exit_price, timestamp, "STOP_LOSS", self.commission_per_contract
                )
            if tp_hit:
                exit_price = min(open_p, pos.take_profit)
                return self.portfolio.close_position(
                    symbol, exit_price, timestamp, "TAKE_PROFIT", self.commission_per_contract
                )

        return None