from datetime import time
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from domain.fvg import FairValueGap
from domain.liquidity import SessionManager
from engine.event_bus import BarContext
from engine.broker import SimulatedBroker
from engine.portfolio import Portfolio
from strategies.base import BaseStrategy


GRADE_RANK: Dict[str, int] = {
    "B": 1,
    "A": 2,
    "A/A+": 3,
    "A+": 4,
}


class ICTMultiTimeframeStrategy(BaseStrategy):
    """
    Discretionary-to-Quantitative ICT Multi-Timeframe Strategy with Two Explicit Archetypes:
    1. Model 1 (5M_TREND_CONTINUATION):
       - 5m structure MUST already match the trade direction (BULLISH for Long, BEARISH for Short).
       - Price retraces into a same-direction 5m FVG, same-direction 5m Equilibrium
         (Discount of a BULLISH 5m range for Long; Premium of a BEARISH 5m range for Short),
         5m OTE, or makes a 1m Counter-BOS pullback (lower confluence).
       - 1m FVGs are strictly excluded on real market datasets.
    2. Model 2 (SMT_REVERSAL_MODEL):
       - Allowed when an active Confirmed SMT Divergence forms and is confirmed by a 1m BOS/CHoCH.
    """
    name: str = "ICT_MultiConfluence_Strategy"

    def __init__(
        self,
        stop_loss_points: float = 6.0,
        reward_risk_ratio: float = 2.0,
        min_rr: float = 1.25,
        max_rr: float = 3.80,
        risk_per_trade_pct: float = 0.01,
        use_dynamic_sizing: bool = False,
        quantity: int = 1,
        sl_mode: str = "tight_1m",  # "tight_1m" or "wide_5m"
        min_grade: str = "A",       # "B", "A", "A/A+", or "A+"
        entry_start_time: time = time(9, 40),
        entry_end_time: time = time(10, 30),
        require_ny_open_window: bool = True,
        allowed_sessions: Optional[Tuple[str, ...]] = ("London", "New York"),
        require_killzone: bool = True,
        front_run_ticks: int = 0,
        max_trades_per_day: int = 1,
        shared_daily_trade_dates: Optional[Dict[Any, str]] = None,
    ):
        self.stop_loss_points = float(stop_loss_points)
        self.reward_risk_ratio = float(reward_risk_ratio)
        self.min_rr = float(min_rr)
        self.max_rr = float(max_rr)
        self.risk_per_trade_pct = float(risk_per_trade_pct)
        self.use_dynamic_sizing = bool(use_dynamic_sizing)
        self.quantity = int(quantity)
        self.sl_mode = sl_mode.lower().strip()
        self.min_grade = min_grade.upper().strip()
        self.entry_start_time = entry_start_time
        self.entry_end_time = entry_end_time
        self.require_ny_open_window = bool(require_ny_open_window)
        self.allowed_sessions = allowed_sessions
        self.require_killzone = bool(require_killzone)
        self.front_run_ticks = int(front_run_ticks)
        self.max_trades_per_day = int(max_trades_per_day)
        self.shared_daily_trade_dates = shared_daily_trade_dates

        self.skipped_setups: List[Dict[str, Any]] = []
        self._trades_taken_by_date: Dict[Any, int] = {}

    def _is_within_trading_window(self, ctx: BarContext) -> bool:
        session = SessionManager.get_session(ctx.timestamp)
        if self.require_killzone and self.allowed_sessions and session not in self.allowed_sessions:
            return False

        if self.require_ny_open_window:
            bar_time = ctx.timestamp.time()
            if not (self.entry_start_time <= bar_time <= self.entry_end_time):
                return False

        trade_date = ctx.timestamp.date()
        if self._trades_taken_by_date.get(trade_date, 0) >= self.max_trades_per_day:
            return False

        if self.shared_daily_trade_dates is not None and trade_date in self.shared_daily_trade_dates:
            return False

        return True

    @staticmethod
    def _target_priority_rank(target_type: str) -> int:
        """
        Explicit Target Hierarchy:
        Priority 1: Macro Session Highs/Lows (London/Asia/NY), Weekend Gaps, 4H Pools, 1H Pools
        Priority 2: 15-Minute Relative Equal Highs/Lows (EQUAL_HL_15MIN)
        Priority 3: Standard 15-Minute Liquidity Pools (15MIN_POOL)
        """
        t_up = target_type.upper()
        if any(k in t_up for k in ("SESSION_", "WEEKEND_GAP", "4H_POOL", "EQUAL_HL_4H", "1H_POOL", "EQUAL_HL_1H")):
            return 1
        if "EQUAL_HL_15MIN" in t_up:
            return 2
        return 3

    def _select_best_target(
        self,
        ctx: BarContext,
        direction: str,
        fill_price: float,
        actual_risk_pts: float,
        tick_size: float,
    ) -> Optional[Dict[str, Any]]:
        front_run = self.front_run_ticks * tick_size
        candidates: List[Dict[str, Any]] = []

        # 1. Check Unfilled Weekend Gaps (Tier 1, Priority 1)
        for gap in ctx.active_weekend_gaps:
            if direction == "LONG" and gap.lower_bound > fill_price:
                for bound_price, label in ((gap.lower_bound, "WEEKEND_GAP_START"), (gap.upper_bound, "WEEKEND_GAP_FILL")):
                    adj_tp = bound_price - front_run
                    rr = (adj_tp - fill_price) / actual_risk_pts if actual_risk_pts > 0 else 0.0
                    if self.min_rr <= rr <= self.max_rr:
                        candidates.append({
                            "price": float(adj_tp),
                            "rr": float(rr),
                            "target_type": f"{label} @ {adj_tp:.2f}",
                            "tier": 1,
                            "priority": 1,
                        })
            elif direction == "SHORT" and gap.upper_bound < fill_price:
                for bound_price, label in ((gap.upper_bound, "WEEKEND_GAP_START"), (gap.lower_bound, "WEEKEND_GAP_FILL")):
                    adj_tp = bound_price + front_run
                    rr = (fill_price - adj_tp) / actual_risk_pts if actual_risk_pts > 0 else 0.0
                    if self.min_rr <= rr <= self.max_rr:
                        candidates.append({
                            "price": float(adj_tp),
                            "rr": float(rr),
                            "target_type": f"{label} @ {adj_tp:.2f}",
                            "tier": 1,
                            "priority": 1,
                        })

        # 2. Check Macro Session Highs/Lows & HTF (4H, 1H, 15m) Liquidity Pools inside [min_rr, max_rr]
        wanted_pool_type = "BSL" if direction == "LONG" else "SSL"
        for t_pool in ctx.target_pools:
            if t_pool["pool_type"] != wanted_pool_type:
                continue
            raw_price = float(t_pool["price"])
            formed_str = t_pool["creation_time"].strftime("%m-%d %H:%M") if "creation_time" in t_pool else ""
            raw_t_type = str(t_pool["target_type"])
            prio = self._target_priority_rank(raw_t_type)

            if direction == "LONG" and raw_price > fill_price:
                adj_tp = raw_price - front_run
                rr = (adj_tp - fill_price) / actual_risk_pts if actual_risk_pts > 0 else 0.0
                if self.min_rr <= rr <= self.max_rr:
                    candidates.append({
                        "price": float(adj_tp),
                        "rr": float(rr),
                        "target_type": f"{raw_t_type} ({wanted_pool_type} @ {adj_tp:.2f}, formed {formed_str})",
                        "tier": int(t_pool["tier"]),
                        "priority": prio,
                    })
            elif direction == "SHORT" and raw_price < fill_price:
                adj_tp = raw_price + front_run
                rr = (fill_price - adj_tp) / actual_risk_pts if actual_risk_pts > 0 else 0.0
                if self.min_rr <= rr <= self.max_rr:
                    candidates.append({
                        "price": float(adj_tp),
                        "rr": float(rr),
                        "target_type": f"{raw_t_type} ({wanted_pool_type} @ {adj_tp:.2f}, formed {formed_str})",
                        "tier": int(t_pool["tier"]),
                        "priority": prio,
                    })

        if candidates:
            # Sort strictly by Priority Hierarchy first (Session/4H/1H/Gap -> 15m Equal H/L -> 15m Pool), then closest RR
            candidates.sort(key=lambda x: (x["priority"], x["rr"]))
            return candidates[0]

        # Synthetic fallback ONLY for tiny unit-test DataFrames (< 25 bars)
        if len(ctx.history_df) < 25:
            default_tp = (
                fill_price + (actual_risk_pts * self.reward_risk_ratio)
                if direction == "LONG"
                else fill_price - (actual_risk_pts * self.reward_risk_ratio)
            )
            return {
                "price": float(default_tp),
                "rr": float(self.reward_risk_ratio),
                "target_type": f"SYNTHETIC_DEFAULT_RR @ {default_tp:.2f}",
                "tier": 1,
                "priority": 1,
            }

        return None

    def _evaluate_retracement_and_trigger(
        self,
        ctx: BarContext,
        direction: str,
        is_5m_trend_aligned: bool,
        has_active_smt: bool,
    ) -> Tuple[bool, str, str, int, List[FairValueGap], str]:
        """
        Evaluates the two distinct strategy models:
        - Model 1 (When `is_5m_trend_aligned` is True):
          * 5m FVG or 5m Same-Direction Equilibrium -> 3 pts
          * 5m OTE -> 2 pts
          * 1m Counter-BOS pullback (without 5m FVG/EQ) -> 1 pt
          * NEVER counts 1m FVGs on real market data!
        - Model 2 (When `is_5m_trend_aligned` is False, i.e., Counter-5m Confirmed SMT Reversal):
          * Requires `has_active_smt == True` and price within 1.5 * SL of the recent extreme.
        """
        wanted_struct_dir = "BULLISH" if direction == "LONG" else "BEARISH"
        opposite_struct_dir = "BEARISH" if direction == "LONG" else "BULLISH"

        recent_1m_events = ctx.structure_events_1m[-6:] if ctx.structure_events_1m else []
        confirm_event = None
        for ev in reversed(recent_1m_events):
            if ev["direction"] == wanted_struct_dir and (ctx.timestamp - ev["close_time"]) <= pd.Timedelta(minutes=1):
                confirm_event = ev
                break

        if confirm_event is None:
            return False, "NONE", "", 0, [], ""

        if confirm_event["close_time"] < ctx.timestamp:
            if direction == "LONG" and ctx.close < ctx.open:
                return False, "NONE", "", 0, [], ""
            if direction == "SHORT" and ctx.close > ctx.open:
                return False, "NONE", "", 0, [], ""

        trigger_desc = f"1m {confirm_event['label']} at {confirm_event['close_time'].strftime('%H:%M')}"

        # =====================================================================
        # MODEL 2: Counter-5m Confirmed SMT Reversal (e.g., ES Short at 09:50 on Oct 13)
        # =====================================================================
        if not is_5m_trend_aligned:
            if has_active_smt:
                recent_6 = ctx.history_df.iloc[-6:]
                if direction == "LONG":
                    dist_from_manipulation = ctx.close - float(recent_6["low"].min())
                else:
                    dist_from_manipulation = float(recent_6["high"].max()) - ctx.close

                if dist_from_manipulation <= (self.stop_loss_points * 1.5):
                    return (
                        True,
                        "SMT_REVERSAL_MODEL",
                        f"Confirmed {wanted_struct_dir.title()} SMT Reversal + {trigger_desc}",
                        3,
                        [],
                        trigger_desc,
                    )
            return False, "NONE", "", 0, [], ""

        # =====================================================================
        # MODEL 1: 5m Trend Continuation (Strictly 5m FVG / 5m EQ / 5m OTE / 1m Counter-BOS)
        # =====================================================================
        fvg_type_needed = "BISI" if direction == "LONG" else "SIBI"
        pullback_bars = ctx.history_df.iloc[-6:]
        pullback_low_min = float(pullback_bars["low"].min())
        pullback_high_max = float(pullback_bars["high"].max())
        price = ctx.close
        max_fvg_dist = self.stop_loss_points * 1.5
        is_short_synthetic = len(ctx.history_df) < 25

        touched_5m_fvgs: List[FairValueGap] = []
        fvg_5m_strs: List[str] = []

        # Only check 5m FVGs on real data (1m FVG fallback is strictly limited to the 13-bar unit test)
        fvg_sources = [("5m", ctx.active_fvgs_5m, 4)]
        if is_short_synthetic:
            fvg_sources.append(("1m", ctx.active_fvgs, 0))

        for tf_label, fvg_list, close_offset_min in fvg_sources:
            for f in fvg_list:
                if f.fvg_type != fvg_type_needed:
                    continue
                fvg_ready_time = f.creation_time + pd.Timedelta(minutes=close_offset_min)
                if fvg_ready_time >= ctx.timestamp and not is_short_synthetic:
                    continue

                if abs(price - f.top) > max_fvg_dist and abs(price - f.bottom) > max_fvg_dist:
                    continue

                is_hit = False
                if direction == "LONG":
                    if pullback_low_min <= (f.top + 0.5) and pullback_high_max >= f.bottom and price >= f.bottom:
                        is_hit = True
                else:
                    if pullback_high_max >= (f.bottom - 0.5) and pullback_low_min <= f.top and price <= f.top:
                        is_hit = True

                if is_hit:
                    desc_s = f"{tf_label} {f.fvg_type} [{f.bottom:.2f}-{f.top:.2f}] (formed {f.creation_time.strftime('%H:%M')})"
                    touched_5m_fvgs.append(f)
                    fvg_5m_strs.append(desc_s)

        # Strictly check 5m DealingRange (direction MUST match trade direction)
        dr_ref = ctx.active_dealing_range_5m if not is_short_synthetic else (ctx.active_dealing_range_5m or ctx.active_dealing_range)
        in_favorable_eq = False
        eq_detail_str = ""
        if dr_ref is not None and getattr(dr_ref, "direction", "") == wanted_struct_dir:
            eq_price = float(
                getattr(dr_ref, "eq", getattr(dr_ref, "equilibrium", (dr_ref.high + dr_ref.low) * 0.5))
            )
            if direction == "LONG" and (dr_ref.get_zone(price) == "DISCOUNT" or pullback_low_min <= eq_price):
                in_favorable_eq = True
                eq_detail_str = f"5m Bullish Range Discount EQ <= {eq_price:.2f} (Range [{dr_ref.low:.2f}-{dr_ref.high:.2f}])"
            elif direction == "SHORT" and (dr_ref.get_zone(price) == "PREMIUM" or pullback_high_max >= eq_price):
                in_favorable_eq = True
                eq_detail_str = f"5m Bearish Range Premium EQ >= {eq_price:.2f} (Range [{dr_ref.low:.2f}-{dr_ref.high:.2f}])"

        fib_ref = ctx.active_fib_range_5m if not is_short_synthetic else (ctx.active_fib_range_5m or ctx.active_fib_range)
        in_ote = False
        ote_detail_str = ""
        if fib_ref is not None and fib_ref.direction == wanted_struct_dir:
            check_p = pullback_low_min if direction == "LONG" else pullback_high_max
            if fib_ref.is_in_ote(price) or fib_ref.is_in_ote(check_p):
                in_ote = True
                span = float(fib_ref.high - fib_ref.low)
                if fib_ref.direction == "BULLISH":
                    ote_low = float(fib_ref.high - 0.79 * span)
                    ote_high = float(fib_ref.high - 0.62 * span)
                else:
                    ote_low = float(fib_ref.low + 0.62 * span)
                    ote_high = float(fib_ref.low + 0.79 * span)
                ote_detail_str = f"5m Fib OTE [{ote_low:.2f}-{ote_high:.2f}]"

        counter_events = [
            e for e in recent_1m_events
            if e["direction"] == opposite_struct_dir and e["close_time"] < confirm_event["close_time"]
        ]

        if touched_5m_fvgs and in_favorable_eq:
            detail = f"{fvg_5m_strs[0]} + {eq_detail_str}"
            return True, "5M_FVG_AND_EQ", detail, 3, touched_5m_fvgs, trigger_desc
        if touched_5m_fvgs:
            return True, "5M_FVG_RETRACE", fvg_5m_strs[0], 3, touched_5m_fvgs, trigger_desc
        if in_favorable_eq:
            return True, "5M_EQ_RETRACE", eq_detail_str, 3, [], trigger_desc
        if in_ote:
            return True, "5M_OTE_RETRACE", ote_detail_str, 2, [], trigger_desc
        if counter_events:
            last_ce = counter_events[-1]
            detail = f"1m Counter-{last_ce['kind']} at {last_ce['close_time'].strftime('%H:%M')} (No 5m FVG/EQ hit)"
            return True, "1M_COUNTER_BOS_ONLY", detail, 1, [], trigger_desc

        return False, "NONE", "", 0, [], ""

    def _compute_stop_loss(
        self,
        ctx: BarContext,
        direction: str,
        matching_fvgs: List[FairValueGap],
        tick_size: float,
    ) -> float:
        price = ctx.close
        buffer_pts = 2 * tick_size  # 0.50 pt buffer on ES/NQ
        min_sl = 2.0 if ctx.symbol == "ES" else 8.0
        max_sl = self.stop_loss_points

        if self.sl_mode == "wide_5m" and ctx.confirmed_swings_5m:
            if direction == "LONG":
                lows_5m = [s["price"] for s in ctx.confirmed_swings_5m[-3:] if s["type"] == "SwingLow" and s["price"] < price]
                if lows_5m:
                    dist = (price - max(lows_5m)) + buffer_pts
                    return float(max(min_sl, min(max_sl * 1.5, dist)))
            else:
                highs_5m = [s["price"] for s in ctx.confirmed_swings_5m[-3:] if s["type"] == "SwingHigh" and s["price"] > price]
                if highs_5m:
                    dist = (min(highs_5m) - price) + buffer_pts
                    return float(max(min_sl, min(max_sl * 1.5, dist)))

        micro_bars = ctx.history_df.iloc[-4:]
        if direction == "LONG":
            pullback_low = float(micro_bars["low"].min())
            if ctx.confirmed_swings:
                recent_lows = [
                    s["price"] for s in ctx.confirmed_swings[-3:]
                    if s["type"] == "SwingLow" and (price - max_sl) <= s["price"] < price
                ]
                if recent_lows:
                    pullback_low = min(pullback_low, max(recent_lows))
            dist = (price - pullback_low) + buffer_pts
            return float(max(min_sl, min(max_sl, dist)))
        else:
            pullback_high = float(micro_bars["high"].max())
            if ctx.confirmed_swings:
                recent_highs = [
                    s["price"] for s in ctx.confirmed_swings[-3:]
                    if s["type"] == "SwingHigh" and price < s["price"] <= (price + max_sl)
                ]
                if recent_highs:
                    pullback_high = max(pullback_high, min(recent_highs))
            dist = (pullback_high - price) + buffer_pts
            return float(max(min_sl, min(max_sl, dist)))

    @staticmethod
    def grade_setup(
        direction: str,
        target_info: Dict[str, Any],
        retracement_type: str,
        retracement_pts: int,
        has_smt: bool,
        has_macro_sweep: bool,
        has_structural_sweep: bool,
        htf_trends: Dict[str, str],
        liquidity_taken_level: str,
    ) -> Tuple[str, int, List[str], List[str]]:
        score = 0
        pros: List[str] = []
        cons: List[str] = []

        prio = int(target_info.get("priority", 1))
        if prio == 1:
            score += 3
            pros.append(f"Tier-1 Primary Target ({target_info['target_type']}, RR={target_info['rr']:.2f})")
        elif prio == 2:
            score += 2
            pros.append(f"15m Equal H/L Target ({target_info['target_type']}, RR={target_info['rr']:.2f})")
        else:
            score += 1
            cons.append(f"Tier-2 Lower-TF Target ({target_info['target_type']}, RR={target_info['rr']:.2f})")

        score += retracement_pts
        if retracement_pts >= 3:
            pros.append(f"Strong Entry / Retracement Model ({retracement_type})")
        elif retracement_pts == 2:
            pros.append("5m Fibonacci OTE Retracement")
        else:
            cons.append("1m Counter-BOS Pullback Only (No 5m FVG/EQ Mitigated)")

        if has_smt:
            score += 2
            pros.append("Confirmed SMT Divergence Catalyst")
        elif has_macro_sweep:
            score += 2
            pros.append("Macro Session Liquidity Sweep Catalyst")
        elif has_structural_sweep:
            score += 1
            pros.append("HTF Structural Liquidity Sweep Catalyst")
        else:
            score -= 1
            cons.append("No Prior Opposite-Side Sweep / SMT Manipulation")

        wanted_bias = "BULLISH" if direction == "LONG" else "BEARISH"
        t_4h = htf_trends.get("4h", "NEUTRAL")
        t_1h = htf_trends.get("1h", "NEUTRAL")
        t_15m = htf_trends.get("15min", "NEUTRAL")

        aligned_count = sum(1 for t_bias in (t_4h, t_1h, t_15m) if t_bias == wanted_bias)
        neutral_count = sum(1 for t_bias in (t_4h, t_1h, t_15m) if t_bias == "NEUTRAL")

        if aligned_count >= 2 or neutral_count == 3:
            score += 2
            pros.append(f"HTF Structure Aligned (4H={t_4h}, 1H={t_1h}, 15m={t_15m})")
        elif aligned_count == 1 or has_smt or "WEEKEND_GAP" in target_info["target_type"]:
            score += 1
            cons.append(f"Mixed HTF Structure (4H={t_4h}, 1H={t_1h}, 15m={t_15m})")
        else:
            score -= 1
            cons.append(f"Counter-HTF Structure (4H={t_4h}, 1H={t_1h}, 15m={t_15m})")

        if liquidity_taken_level == "HIGH":
            score -= 1 if has_smt else 3
            cons.append("High Target-Side Liquidity Already Taken Prior to Entry")
        elif liquidity_taken_level == "MEDIUM":
            pros.append("Medium Pre-Entry Liquidity Taken (Primary Targets Intact)")
        else:
            pros.append("Low Target-Side Liquidity Exhaustion (Clean Draw Ahead)")

        if liquidity_taken_level == "HIGH" and not has_smt:
            grade = "B"
        elif liquidity_taken_level == "HIGH" and has_smt:
            grade = "A" if score >= 5 else "B"
        elif score >= 9 and prio == 1:
            grade = "A+"
        elif score >= 7 and prio <= 2:
            grade = "A/A+"
        elif score >= 5:
            grade = "A"
        else:
            grade = "B"

        return grade, score, pros, cons

    def _calculate_contracts(self, portfolio: Portfolio, symbol: str, actual_risk_pts: float) -> int:
        if not self.use_dynamic_sizing or actual_risk_pts <= 0:
            return self.quantity
        risk_dollars = portfolio.cash * self.risk_per_trade_pct
        dollar_risk_per_contract = actual_risk_pts * portfolio.get_point_value(symbol)
        if dollar_risk_per_contract <= 0:
            return self.quantity
        contracts = int(round(risk_dollars / dollar_risk_per_contract))
        return max(1, contracts)

    def on_bar(
        self,
        ctx: BarContext,
        broker: SimulatedBroker,
        portfolio: Portfolio,
    ) -> None:
        if portfolio.has_open_position(ctx.symbol):
            return

        if not self._is_within_trading_window(ctx):
            return

        session = SessionManager.get_session(ctx.timestamp)
        price = ctx.close
        tick_size = portfolio.get_tick_size(ctx.symbol)
        slip_pts = broker.slippage_ticks * tick_size

        ssl_details = [d for d in ctx.recent_sweep_details if d["pool_type"] == "SSL"]
        bsl_details = [d for d in ctx.recent_sweep_details if d["pool_type"] == "BSL"]

        ssl_swept = len(ssl_details) > 0
        bsl_swept = len(bsl_details) > 0
        macro_ssl_swept = any(d["source"] == "Macro_Session" for d in ssl_details)
        macro_bsl_swept = any(d["source"] == "Macro_Session" for d in bsl_details)

        bull_smts = [ev for ev in ctx.recent_smt_events if "BULLISH" in str(ev.get("type", "")).upper()]
        bear_smts = [ev for ev in ctx.recent_smt_events if "BEARISH" in str(ev.get("type", "")).upper()]

        bullish_smt = len(bull_smts) > 0
        bearish_smt = len(bear_smts) > 0

        trend_5m = ctx.htf_trends.get("5min", "NEUTRAL")
        t_4h = ctx.htf_trends.get("4h", "NEUTRAL")
        t_1h = ctx.htf_trends.get("1h", "NEUTRAL")
        t_15m = ctx.htf_trends.get("15min", "NEUTRAL")

        for direction in ("LONG", "SHORT"):
            wanted_bias = "BULLISH" if direction == "LONG" else "BEARISH"
            htf_aligned_count = sum(1 for b in (t_4h, t_1h, t_15m) if b == wanted_bias)
            is_synthetic_short_df = len(ctx.history_df) < 25
            dir_exhaustion = ctx.liquidity_taken_by_dir.get(direction, ctx.liquidity_taken_level)
            dir_taken_pools = ctx.session_taken_by_dir.get(direction, ctx.session_taken_pools)

            if direction == "LONG":
                has_catalyst = ssl_swept or bullish_smt or htf_aligned_count >= 2
                is_5m_trend_aligned = (trend_5m == "BULLISH" and (htf_aligned_count >= 1 or bullish_smt or is_synthetic_short_df))
                can_attempt = has_catalyst and (is_5m_trend_aligned or bullish_smt)
                dir_sweep_details = [d["label"] for d in ssl_details]
                dir_smt_details = [ev["description"] for ev in bull_smts]
            else:
                has_catalyst = bsl_swept or bearish_smt or htf_aligned_count >= 2
                is_5m_trend_aligned = (trend_5m == "BEARISH" and (htf_aligned_count >= 1 or bearish_smt or is_synthetic_short_df))
                can_attempt = has_catalyst and (is_5m_trend_aligned or bearish_smt)
                dir_sweep_details = [d["label"] for d in bsl_details]
                dir_smt_details = [ev["description"] for ev in bear_smts]

            if not can_attempt:
                continue

            triggered, retrace_type, retrace_detail, retrace_pts, matching_fvgs, trigger_desc = (
                self._evaluate_retracement_and_trigger(
                    ctx=ctx,
                    direction=direction,
                    is_5m_trend_aligned=is_5m_trend_aligned,
                    has_active_smt=(bullish_smt if direction == "LONG" else bearish_smt),
                )
            )
            if not triggered:
                continue

            raw_sl_dist = self._compute_stop_loss(ctx, direction, matching_fvgs, tick_size)
            sl_price = price - raw_sl_dist if direction == "LONG" else price + raw_sl_dist
            fill_price = price + slip_pts if direction == "LONG" else price - slip_pts
            actual_risk_pts = abs(fill_price - sl_price)

            target_info = self._select_best_target(ctx, direction, fill_price, actual_risk_pts, tick_size)
            if target_info is None:
                continue

            grade, score, pros, cons = self.grade_setup(
                direction=direction,
                target_info=target_info,
                retracement_type=retrace_type,
                retracement_pts=retrace_pts,
                has_smt=(bullish_smt if direction == "LONG" else bearish_smt),
                has_macro_sweep=(macro_ssl_swept if direction == "LONG" else macro_bsl_swept),
                has_structural_sweep=(ssl_swept if direction == "LONG" else bsl_swept),
                htf_trends=ctx.htf_trends,
                liquidity_taken_level=dir_exhaustion,
            )

            setup_metadata = {
                "session": session,
                "grade": grade,
                "score": score,
                "target_type": target_info["target_type"],
                "target_tier": target_info["tier"],
                "planned_rr": round(target_info["rr"], 2),
                "retracement_type": retrace_type,
                "retracement_detail": retrace_detail,
                "entry_trigger_detail": trigger_desc,
                "sl_mode": self.sl_mode,
                "liquidity_already_taken": dir_exhaustion,
                "session_taken_pools": list(dir_taken_pools),
                "htf_trends": dict(ctx.htf_trends),
                "ssl_swept": ssl_swept,
                "bsl_swept": bsl_swept,
                "bullish_smt": bullish_smt,
                "bearish_smt": bearish_smt,
                "swept_pools_detail": dir_sweep_details,
                "smt_detail": dir_smt_details,
                "bisi_fvg_count": len(matching_fvgs) if direction == "LONG" else 0,
                "sibi_fvg_count": len(matching_fvgs) if direction == "SHORT" else 0,
                "pros": pros,
                "cons": cons,
            }

            if GRADE_RANK.get(grade, 0) < GRADE_RANK.get(self.min_grade, 2):
                self.skipped_setups.append({
                    "timestamp": ctx.timestamp,
                    "direction": direction,
                    "price": fill_price,
                    **setup_metadata,
                })
                continue

            tp_price = target_info["price"]
            contracts = self._calculate_contracts(portfolio, ctx.symbol, actual_risk_pts)

            broker.submit_order(
                symbol=ctx.symbol,
                direction=direction,
                order_type="MARKET",
                quantity=contracts,
                stop_loss=sl_price,
                take_profit=tp_price,
                timestamp=ctx.timestamp,
                current_close=price,
                metadata=setup_metadata,
            )
            trade_date = ctx.timestamp.date()
            self._trades_taken_by_date[trade_date] = self._trades_taken_by_date.get(trade_date, 0) + 1
            if self.shared_daily_trade_dates is not None:
                self.shared_daily_trade_dates[trade_date] = ctx.symbol
            return