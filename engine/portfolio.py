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
    quantity: float
    stop_loss: float
    take_profit: float
    entry_time: pd.Timestamp
    initial_risk_points: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    initial_stop_loss: float = 0.0
    remaining_fraction: float = 1.0
    realized_points_pnl: float = 0.0
    realized_gross_pnl: float = 0.0
    realized_commission: float = 0.0
    partial_targets: List[Dict[str, Any]] = field(default_factory=list)
    partial_exits: List[Dict[str, Any]] = field(default_factory=list)
    sl_trail_history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if self.initial_stop_loss == 0.0 and self.stop_loss is not None:
            self.initial_stop_loss = float(self.stop_loss)
        if self.initial_risk_points == 0.0 and self.stop_loss is not None:
            self.initial_risk_points = abs(self.entry_price - self.initial_stop_loss)
        if not self.partial_targets and self.metadata and "partial_targets" in self.metadata:
            self.partial_targets = [dict(pt) for pt in self.metadata["partial_targets"]]

    def unrealized_points(self, current_price: float) -> float:
        if self.direction == "LONG":
            return current_price - self.entry_price
        return self.entry_price - current_price

    def update_stop_loss(self, new_sl: float, timestamp: pd.Timestamp, reason: str = "") -> bool:
        new_sl = float(new_sl)
        if self.direction == "LONG" and new_sl <= self.stop_loss + 1e-9:
            return False
        if self.direction == "SHORT" and new_sl >= self.stop_loss - 1e-9:
            return False

        old_sl = self.stop_loss
        self.stop_loss = new_sl
        self.sl_trail_history.append({
            "time": timestamp,
            "old_sl": float(old_sl),
            "new_sl": float(new_sl),
            "reason": reason,
        })
        return True


@dataclass
class TradeRecord:
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    exit_reason: str  # 'STOP_LOSS', 'TAKE_PROFIT', 'BE_TRAILED_SL', 'SIGNAL_EXIT', 'END_OF_DATA'
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
        quantity: float,
        stop_loss: float,
        take_profit: float,
        timestamp: pd.Timestamp,
        commission: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Position:
        if symbol in self.positions:
            raise ValueError(f"Position already open for symbol {symbol}.")

        eff_qty = int(quantity) if float(quantity).is_integer() else round(float(quantity), 2)
        pos = Position(
            symbol=symbol,
            direction=direction.upper(),
            entry_price=float(entry_price),
            quantity=eff_qty,
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            entry_time=timestamp,
            metadata=metadata or {},
        )
        self.positions[symbol] = pos
        self.cash -= float(commission)
        return pos

    def record_partial_exit(
        self,
        symbol: str,
        exit_price: float,
        weight: float,
        timestamp: pd.Timestamp,
        label: str,
        target_type: str = "",
        commission_per_side: float = 0.0,
    ) -> None:
        if symbol not in self.positions:
            raise ValueError(f"No open position found for symbol {symbol}.")

        pos = self.positions[symbol]
        eff_weight = min(float(weight), pos.remaining_fraction)
        if eff_weight <= 1e-9:
            return

        point_val = self.get_point_value(symbol)
        pts = pos.unrealized_points(float(exit_price))
        slice_gross = pts * point_val * pos.quantity * eff_weight
        slice_exit_comm = float(commission_per_side) * pos.quantity * eff_weight
        slice_roundtrip_comm = float(commission_per_side) * 2.0 * pos.quantity * eff_weight

        self.cash += slice_gross - slice_exit_comm

        pos.realized_points_pnl += pts * eff_weight
        pos.realized_gross_pnl += slice_gross
        pos.realized_commission += slice_roundtrip_comm
        pos.remaining_fraction = round(max(0.0, pos.remaining_fraction - eff_weight), 6)

        pos.partial_exits.append({
            "label": label,
            "target_type": target_type,
            "time": timestamp,
            "price": float(exit_price),
            "weight": eff_weight,
            "points_pnl": float(pts),
            "r_multiple": float(pts / pos.initial_risk_points) if pos.initial_risk_points > 0 else 0.0,
            "net_pnl": float(slice_gross - slice_roundtrip_comm),
        })

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
        rem = pos.remaining_fraction

        final_leg_pts = pos.unrealized_points(float(exit_price))
        weighted_final_pts = final_leg_pts * rem
        points_pnl = pos.realized_points_pnl + weighted_final_pts

        final_gross = weighted_final_pts * point_val * pos.quantity
        gross_pnl = pos.realized_gross_pnl + final_gross

        final_exit_comm = float(commission_per_side) * pos.quantity * rem
        final_roundtrip_comm = float(commission_per_side) * 2.0 * pos.quantity * rem
        total_commission = pos.realized_commission + final_roundtrip_comm
        net_pnl = gross_pnl - total_commission

        self.cash += final_gross - final_exit_comm

        r_mult = (
            points_pnl / pos.initial_risk_points
            if pos.initial_risk_points > 0
            else 0.0
        )

        sl_was_moved = abs(pos.stop_loss - pos.initial_stop_loss) > 1e-9
        had_partials = len(pos.partial_exits) > 0

        effective_exit_reason = exit_reason
        if exit_reason == "STOP_LOSS" and (sl_was_moved or (had_partials and net_pnl >= 0.0)):
            effective_exit_reason = "BE_TRAILED_SL"

        meta = dict(pos.metadata) if pos.metadata else {}
        if had_partials or sl_was_moved:
            meta["partial_exits"] = list(pos.partial_exits)
            meta["sl_trail_history"] = list(pos.sl_trail_history)
            meta["initial_stop_loss"] = float(pos.initial_stop_loss)
            meta["final_stop_loss"] = float(pos.stop_loss)
            meta["remaining_fraction_at_exit"] = float(rem)
            meta["management_summary"] = self._build_management_summary(
                pos, float(exit_price), timestamp, effective_exit_reason, net_pnl, r_mult
            )

        if effective_exit_reason == "BE_TRAILED_SL":
            meta["outcome"] = "BE"
        elif net_pnl > 0 or exit_reason == "TAKE_PROFIT":
            meta["outcome"] = "WIN"
        else:
            meta["outcome"] = "LOSS"

        record = TradeRecord(
            symbol=pos.symbol,
            direction=pos.direction,
            entry_time=pos.entry_time,
            exit_time=timestamp,
            entry_price=pos.entry_price,
            exit_price=float(exit_price),
            quantity=pos.quantity,
            stop_loss=pos.initial_stop_loss,
            take_profit=pos.take_profit,
            exit_reason=effective_exit_reason,
            points_pnl=points_pnl,
            gross_pnl=gross_pnl,
            commission=total_commission,
            net_pnl=net_pnl,
            r_multiple=r_mult,
            metadata=meta,
        )
        self.closed_trades.append(record)
        return record

    @staticmethod
    def _build_management_summary(
        pos: Position,
        exit_price: float,
        exit_time: pd.Timestamp,
        exit_reason: str,
        net_pnl: float,
        r_mult: float,
    ) -> str:
        timed_events = []
        for idx, pe in enumerate(pos.partial_exits):
            pct = int(round(pe["weight"] * 100))
            txt = (
                f"{pe['label']} hit @ {pe['price']:.2f} at {pe['time'].strftime('%H:%M')} "
                f"(cut {pct}%, {pe['r_multiple']:+.2f}R)"
            )
            timed_events.append((pe["time"], 0, idx, txt))

        for idx, tr in enumerate(pos.sl_trail_history):
            r_tag = f" [{tr['reason']}]" if tr.get("reason") else ""
            txt = f"Changed SL {tr['old_sl']:.2f} -> {tr['new_sl']:.2f}{r_tag} at {tr['time'].strftime('%H:%M')}"
            timed_events.append((tr["time"], 1, idx, txt))

        timed_events.sort(key=lambda x: (x[0], x[1], x[2]))
        steps: List[str] = [item[3] for item in timed_events]

        rem_pct = int(round(pos.remaining_fraction * 100))
        if exit_reason == "BE_TRAILED_SL":
            verb = "made" if net_pnl >= 0 else "lost"
            steps.append(
                f"Stopped out ({rem_pct}% rem) at changed SL @ {exit_price:.2f} at {exit_time.strftime('%H:%M')} "
                f"-> BE ({verb} ${abs(net_pnl):.2f} / {r_mult:+.2f}R)"
            )
        elif exit_reason == "TAKE_PROFIT":
            steps.append(
                f"FullTP hit ({rem_pct}% rem) @ {exit_price:.2f} at {exit_time.strftime('%H:%M')} "
                f"-> WIN (made ${net_pnl:+.2f} / {r_mult:+.2f}R)"
            )
        else:
            steps.append(
                f"Exited ({rem_pct}% rem) @ {exit_price:.2f} ({exit_reason}) -> Net PnL ${net_pnl:+.2f} ({r_mult:+.2f}R)"
            )
        return " -> ".join(steps)

    def update_mark_to_market(
        self, timestamp: pd.Timestamp, current_prices: Dict[str, float]
    ) -> float:
        unrealized_pnl = 0.0
        for sym, pos in self.positions.items():
            if sym in current_prices:
                pts = pos.unrealized_points(current_prices[sym])
                unrealized_pnl += pts * self.get_point_value(sym) * pos.quantity * pos.remaining_fraction

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