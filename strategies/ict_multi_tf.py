from typing import Optional
from domain.liquidity import SessionManager
from engine.event_bus import BarContext
from engine.broker import SimulatedBroker
from engine.portfolio import Portfolio
from strategies.base import BaseStrategy


class ICTMultiTimeframeStrategy(BaseStrategy):
    """
    Formalized ICT Multi-Confluence Strategy:
    Combines Session Killzones, Liquidity Sweeps / SMT Divergence, Fair Value Gaps (FVG),
    Single-Leg Equilibrium (Discount/Premium), and Fibonacci OTE zones.
    """
    name: str = "ICT_MultiConfluence_Strategy"

    def __init__(
        self,
        stop_loss_points: float = 10.0,
        reward_risk_ratio: float = 2.0,
        quantity: int = 1,
        allowed_sessions: Optional[tuple] = ("London", "New York"),
        require_killzone: bool = True,
    ):
        self.stop_loss_points = float(stop_loss_points)
        self.reward_risk_ratio = float(reward_risk_ratio)
        self.quantity = int(quantity)
        self.allowed_sessions = allowed_sessions
        self.require_killzone = require_killzone

    def on_bar(
        self,
        ctx: BarContext,
        broker: SimulatedBroker,
        portfolio: Portfolio,
    ) -> None:
        if portfolio.has_open_position(ctx.symbol):
            return

        session = SessionManager.get_session(ctx.timestamp)
        if self.require_killzone and self.allowed_sessions and session not in self.allowed_sessions:
            return

        price = ctx.close

        # 1. Evaluate Equilibrium (Discount / Premium) & Fibonacci OTE context
        eq_zone = ctx.active_dealing_range.get_zone(price) if ctx.active_dealing_range else "NEUTRAL"
        in_bullish_ote = (
            ctx.active_fib_range is not None
            and ctx.active_fib_range.direction == "BULLISH"
            and ctx.active_fib_range.is_in_ote(price)
        )
        in_bearish_ote = (
            ctx.active_fib_range is not None
            and ctx.active_fib_range.direction == "BEARISH"
            and ctx.active_fib_range.is_in_ote(price)
        )

        # 2. Check Liquidity Sweep or SMT Divergence Triggers
        ssl_swept = any(p.pool_type == "SSL" for p in ctx.recent_sweeps)
        bsl_swept = any(p.pool_type == "BSL" for p in ctx.recent_sweeps)

        bullish_smt = any("BULLISH" in str(ev.get("type", "")).upper() for ev in ctx.recent_smt_events)
        bearish_smt = any("BEARISH" in str(ev.get("type", "")).upper() for ev in ctx.recent_smt_events)

        # 3. Check Fair Value Gap (FVG) alignment near current price
        active_bisi = [
            f for f in ctx.active_fvgs
            if f.fvg_type == "BISI" and (f.bottom - self.stop_loss_points) <= price <= (f.top + self.stop_loss_points)
        ]
        active_sibi = [
            f for f in ctx.active_fvgs
            if f.fvg_type == "SIBI" and (f.bottom - self.stop_loss_points) <= price <= (f.top + self.stop_loss_points)
        ]

        # LONG ENTRY CONDITION:
        # (SSL Sweep OR Bullish SMT) + (Active BISI FVG OR Bullish OTE) + (Discount or Neutral EQ)
        if (ssl_swept or bullish_smt) and (active_bisi or in_bullish_ote) and eq_zone in ("DISCOUNT", "NEUTRAL"):
            if active_bisi:
                anchor_low = min(f.bottom for f in active_bisi)
                sl_dist = max(2.0, min(self.stop_loss_points, (price - anchor_low) + 1.0))
            else:
                sl_dist = self.stop_loss_points

            sl_price = price - sl_dist
            tp_price = price + (sl_dist * self.reward_risk_ratio)

            broker.submit_order(
                symbol=ctx.symbol,
                direction="LONG",
                order_type="MARKET",
                quantity=self.quantity,
                stop_loss=sl_price,
                take_profit=tp_price,
                timestamp=ctx.timestamp,
                current_close=price,
                metadata={
                    "session": session,
                    "eq_zone": eq_zone,
                    "ssl_swept": ssl_swept,
                    "bullish_smt": bullish_smt,
                    "bisi_fvg_count": len(active_bisi),
                    "in_bullish_ote": in_bullish_ote,
                },
            )
            return

        # SHORT ENTRY CONDITION:
        # (BSL Sweep OR Bearish SMT) + (Active SIBI FVG OR Bearish OTE) + (Premium or Neutral EQ)
        if (bsl_swept or bearish_smt) and (active_sibi or in_bearish_ote) and eq_zone in ("PREMIUM", "NEUTRAL"):
            if active_sibi:
                anchor_high = max(f.top for f in active_sibi)
                sl_dist = max(2.0, min(self.stop_loss_points, (anchor_high - price) + 1.0))
            else:
                sl_dist = self.stop_loss_points

            sl_price = price + sl_dist
            tp_price = price - (sl_dist * self.reward_risk_ratio)

            broker.submit_order(
                symbol=ctx.symbol,
                direction="SHORT",
                order_type="MARKET",
                quantity=self.quantity,
                stop_loss=sl_price,
                take_profit=tp_price,
                timestamp=ctx.timestamp,
                current_close=price,
                metadata={
                    "session": session,
                    "eq_zone": eq_zone,
                    "bsl_swept": bsl_swept,
                    "bearish_smt": bearish_smt,
                    "sibi_fvg_count": len(active_sibi),
                    "in_bearish_ote": in_bearish_ote,
                },
            )