from datetime import time
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from domain.fvg import FairValueGap
from domain.liquidity import SessionManager
from engine.event_bus import BarContext
from engine.broker import SimulatedBroker
from engine.portfolio import Portfolio, Position
from strategies.base import BaseStrategy


GRADE_RANK: Dict[str, int] = {
    "B": 1,
    "A": 2,
    "A/A+": 3,
    "A+": 4,
}


class ICTMultiTimeframeStrategy(BaseStrategy):
    """
    Discretionary-to-Quantitative ICT Multi-Timeframe Strategy with:
    - Two Entry Archetypes (5M_TREND_CONTINUATION and SMT_REVERSAL_MODEL)
    - ATH + Dual-Index HTF-Bullish Continuation Override & Bearish SMT Filter
    - Overextended Exhaustion-Leg Filter (HIGH Liquidity Exhaustion + Large Completed Leg)
    - Multi-Target Partial Take-Profit Engine (TP1 [30%], optional TP2 [30%], FullTP [70%/40%])
    - High-Confluence 3-TP Exception (TP1 >= 0.55R when TP2 >= 1.0R and FullTP >= 1.60R exist)
    - Homerun (>= 3.0R FullTP) 15m Liquidity Pool Partials
    - Natural 1m Structural Pullback, FVG Inversion & Liquidity Cluster Trailing Stop Engine
    """
    name: str = "ICT_MultiConfluence_Strategy"

    def __init__(
        self,
        stop_loss_points: float = 6.0,
        reward_risk_ratio: float = 2.0,
        min_rr: float = 1.60,
        min_tp1_rr: float = 0.80,
        max_rr: float = 3.85,
        partial_cut_pct: float = 0.30,
        enable_partials: bool = True,
        enable_trailing_sl: bool = True,
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
        shared_daily_trade_dates: Optional[Dict[Any, Any]] = None,
    ):
        self.stop_loss_points = float(stop_loss_points)
        self.reward_risk_ratio = float(reward_risk_ratio)
        self.min_rr = float(min_rr)
        self.min_tp1_rr = float(min_tp1_rr)
        self.max_rr = float(max_rr)
        self.partial_cut_pct = float(partial_cut_pct)
        self.enable_partials = bool(enable_partials)
        self.enable_trailing_sl = bool(enable_trailing_sl)
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
        self.daily_diagnostics: Dict[str, List[str]] = {}
        self.daily_window_summary: Dict[str, Dict[str, Any]] = {}
        self._trades_taken_by_date: Dict[Any, int] = {}

    def _log_diagnostic(self, date_str: str, message: str) -> None:
        lst = self.daily_diagnostics.setdefault(date_str, [])
        if message not in lst:
            lst.append(message)

    @staticmethod
    def _target_priority_rank(target_type: str) -> int:
        t_up = target_type.upper()
        if any(k in t_up for k in ("SESSION_", "WEEKEND_GAP", "4H_POOL", "EQUAL_HL_4H", "1H_POOL", "EQUAL_HL_1H")):
            return 1
        if "EQUAL_HL_15MIN" in t_up or "NY_OPENING_RANGE" in t_up:
            return 2
        return 3

    def _select_best_target(
        self,
        ctx: BarContext,
        direction: str,
        fill_price: float,
        actual_risk_pts: float,
        tick_size: float,
        allow_opening_range_target: bool = False,
        allow_exception_tp1: bool = False,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        front_run = self.front_run_ticks * tick_size
        all_directional_targets: List[Dict[str, Any]] = []

        # 1. Unfilled Weekend Gaps (Priority 1)
        for gap in ctx.active_weekend_gaps:
            if direction == "LONG" and gap.lower_bound > fill_price:
                for bound_price, label in ((gap.lower_bound, "WEEKEND_GAP_START"), (gap.upper_bound, "WEEKEND_GAP_FILL")):
                    adj_tp = bound_price - front_run
                    rr = (adj_tp - fill_price) / actual_risk_pts if actual_risk_pts > 0 else 0.0
                    all_directional_targets.append({
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
                    all_directional_targets.append({
                        "price": float(adj_tp),
                        "rr": float(rr),
                        "target_type": f"{label} @ {adj_tp:.2f}",
                        "tier": 1,
                        "priority": 1,
                    })

        # 2. Macro Session Highs/Lows & HTF (4H, 1H, 15m) Liquidity Pools
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
                all_directional_targets.append({
                    "price": float(adj_tp),
                    "rr": float(rr),
                    "target_type": f"{raw_t_type} ({wanted_pool_type} @ {adj_tp:.2f}, formed {formed_str})",
                    "tier": int(t_pool["tier"]),
                    "priority": prio,
                })
            elif direction == "SHORT" and raw_price < fill_price:
                adj_tp = raw_price + front_run
                rr = (fill_price - adj_tp) / actual_risk_pts if actual_risk_pts > 0 else 0.0
                all_directional_targets.append({
                    "price": float(adj_tp),
                    "rr": float(rr),
                    "target_type": f"{raw_t_type} ({wanted_pool_type} @ {adj_tp:.2f}, formed {formed_str})",
                    "tier": int(t_pool["tier"]),
                    "priority": prio,
                })

        # 3. Confirmed NY Opening Range Swing Target (Strictly when 5m + 1m SMT align, e.g., Oct 14 09:51 NQ Short)
        if allow_opening_range_target and ctx.confirmed_swings:
            ny_open_ts = ctx.timestamp.replace(hour=9, minute=30, second=0, microsecond=0)
            if direction == "LONG":
                post_open_highs = [
                    s for s in ctx.confirmed_swings
                    if s["type"] == "SwingHigh"
                    and ny_open_ts < s["time"] <= (ctx.timestamp - pd.Timedelta(minutes=6))
                    and float(s["price"]) > fill_price
                ]
                if post_open_highs:
                    best_sh = max(post_open_highs, key=lambda x: x["price"])
                    sh_p = float(best_sh["price"])
                    rr = (sh_p - fill_price) / actual_risk_pts if actual_risk_pts > 0 else 0.0
                    all_directional_targets.append({
                        "price": sh_p,
                        "rr": float(rr),
                        "target_type": f"NY_OPENING_RANGE_HIGH (BSL @ {sh_p:.2f}, formed {best_sh['time'].strftime('%m-%d %H:%M')})",
                        "tier": 2,
                        "priority": 2,
                    })
            else:
                post_open_lows = [
                    s for s in ctx.confirmed_swings
                    if s["type"] == "SwingLow"
                    and ny_open_ts < s["time"] <= (ctx.timestamp - pd.Timedelta(minutes=6))
                    and float(s["price"]) < fill_price
                ]
                if post_open_lows:
                    best_sl = min(post_open_lows, key=lambda x: x["price"])
                    sl_p = float(best_sl["price"])
                    rr = (fill_price - sl_p) / actual_risk_pts if actual_risk_pts > 0 else 0.0
                    all_directional_targets.append({
                        "price": sl_p,
                        "rr": float(rr),
                        "target_type": f"NY_OPENING_RANGE_LOW (SSL @ {sl_p:.2f}, formed {best_sl['time'].strftime('%m-%d %H:%M')})",
                        "tier": 2,
                        "priority": 2,
                    })

        eff_tp1_min_rr = 0.55 if allow_exception_tp1 else self.min_tp1_rr
        full_tp_min_rr = 1.60 if allow_exception_tp1 else (max(1.65, self.min_rr) if self.enable_partials else self.min_rr)

        prio1_targets = [c for c in all_directional_targets if c["priority"] == 1 and c["rr"] >= 0.25]
        if prio1_targets:
            prio1_targets.sort(key=lambda x: x["rr"])
            closest_prio1 = prio1_targets[0]

            if closest_prio1["rr"] < (eff_tp1_min_rr if self.enable_partials else self.min_rr):
                return (
                    None,
                    f"Closest Priority-1 target {closest_prio1['target_type']} only offered "
                    f"{closest_prio1['rr']:.2f}R (< min TP1 {eff_tp1_min_rr:.2f}R); skipping past a major target is forbidden.",
                )

            # MULTI-TP PATH A: Closest Priority-1 pool is in [eff_tp1_min_rr, full_tp_min_rr) -> use as TP1 (30%) and look for FullTP >= 1.60R
            if self.enable_partials and closest_prio1["rr"] < full_tp_min_rr:
                tp1_cand = closest_prio1
                further_cands = [
                    c for c in all_directional_targets
                    if c["priority"] <= 2
                    and (tp1_cand["rr"] + 0.35) <= c["rr"] <= self.max_rr
                ]
                further_cands.sort(key=lambda x: (x["rr"], x["priority"]))

                full_tp_cands = [c for c in further_cands if c["rr"] >= full_tp_min_rr]
                if not full_tp_cands:
                    return (
                        None,
                        f"Closest Priority-1 target {tp1_cand['target_type']} offered {tp1_cand['rr']:.2f}R (valid for TP1), "
                        f"but no unbroken FullTP target >= {full_tp_min_rr:.2f}R exists ahead.",
                    )

                full_tp_cands_prio = sorted(full_tp_cands, key=lambda x: (x["priority"], x["rr"]))
                full_tp = dict(full_tp_cands_prio[0])

                mid_cands = [
                    c for c in further_cands
                    if c["priority"] <= 2
                    and (tp1_cand["rr"] + 0.35) <= c["rr"] <= (full_tp["rr"] - 0.35)
                ]
                if not mid_cands and tp1_cand["rr"] >= 1.20:
                    higher_prio1 = [
                        c for c in full_tp_cands
                        if c["priority"] == 1 and (full_tp["rr"] + 0.50) <= c["rr"] <= self.max_rr
                    ]
                    if higher_prio1:
                        mid_cands = [full_tp]
                        full_tp = dict(higher_prio1[0])

                # If TP1 sacrificed below 0.80R (e.g. 0.57R on Oct 1 NQ Short),
                # strictly require BOTH a valid TP2 (>= 0.95R) and FullTP (>= 1.60R)!
                if tp1_cand["rr"] < self.min_tp1_rr and not mid_cands:
                    return (
                        None,
                        f"TP1 {tp1_cand['target_type']} is < {self.min_tp1_rr:.2f}R ({tp1_cand['rr']:.2f}R) "
                        f"and no intermediate Priority-1/2 TP2 exists between TP1 and FullTP.",
                    )

                partial_list: List[Dict[str, Any]] = [
                    {
                        "label": "TP1",
                        "price": float(tp1_cand["price"]),
                        "rr": round(float(tp1_cand["rr"]), 2),
                        "weight": self.partial_cut_pct,
                        "target_type": tp1_cand["target_type"],
                        "tier": tp1_cand["tier"],
                        "hit": False,
                    }
                ]
                if mid_cands:
                    tp2_cand = mid_cands[0]
                    partial_list.append({
                        "label": "TP2",
                        "price": float(tp2_cand["price"]),
                        "rr": round(float(tp2_cand["rr"]), 2),
                        "weight": self.partial_cut_pct,
                        "target_type": tp2_cand["target_type"],
                        "tier": tp2_cand["tier"],
                        "hit": False,
                    })

                if len(partial_list) >= 2 and all(self._target_priority_rank(pt["target_type"]) == 1 for pt in partial_list):
                    full_tp["priority"] = 1
                    full_tp["tier"] = 1

                full_tp["partial_targets"] = partial_list
                return full_tp, ""

            # MULTI-TP PATH B (HOMERUN TRADE >= 3.0R TO CLOSEST PRIORITY-1 POOL, e.g., Oct 6 ES Long to 7892.00 @ 3.23R):
            if self.enable_partials and 3.00 <= closest_prio1["rr"] <= self.max_rr:
                full_tp = dict(closest_prio1)
                intermediate_15m = [
                    c for c in all_directional_targets
                    if "15MIN" in c["target_type"].upper()
                    and 1.00 <= c["rr"] <= (full_tp["rr"] - 0.50)
                ]
                intermediate_15m.sort(key=lambda x: x["rr"])

                partial_list = []
                if intermediate_15m:
                    tp1_cand = intermediate_15m[0]
                    partial_list.append({
                        "label": "TP1",
                        "price": float(tp1_cand["price"]),
                        "rr": round(float(tp1_cand["rr"]), 2),
                        "weight": self.partial_cut_pct,
                        "target_type": tp1_cand["target_type"],
                        "tier": tp1_cand["tier"],
                        "hit": False,
                    })
                    tp2_cands = [
                        c for c in intermediate_15m[1:]
                        if c["rr"] >= (tp1_cand["rr"] + 0.50) and c["rr"] <= (full_tp["rr"] - 0.50)
                    ]
                    if tp2_cands:
                        tp2_cand = tp2_cands[0]
                        partial_list.append({
                            "label": "TP2",
                            "price": float(tp2_cand["price"]),
                            "rr": round(float(tp2_cand["rr"]), 2),
                            "weight": self.partial_cut_pct,
                            "target_type": tp2_cand["target_type"],
                            "tier": tp2_cand["tier"],
                            "hit": False,
                        })

                full_tp["partial_targets"] = partial_list
                return full_tp, ""

        # SINGLE-TP PATH: Closest major target already offers >= 1.60R (e.g., Oct 13 ES @ 2.32R, Oct 14 NQ @ 2.04R)
        valid_candidates = [
            c for c in all_directional_targets
            if c["priority"] <= 2 and full_tp_min_rr <= c["rr"] <= self.max_rr
        ]
        if valid_candidates:
            valid_candidates.sort(key=lambda x: (x["priority"], x["rr"]))
            best = dict(valid_candidates[0])
            best["partial_targets"] = []
            return best, ""

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
                "partial_targets": [],
            }, ""

        if all_directional_targets:
            best_rr = min(all_directional_targets, key=lambda x: abs(x["rr"] - 2.0))
            return (
                None,
                f"No unbroken target inside [{full_tp_min_rr:.2f}R, {self.max_rr:.2f}R] "
                f"(closest candidate was {best_rr['target_type']} at {best_rr['rr']:.2f}R).",
            )
        return None, "No unbroken Session/HTF target pools remaining ahead in trade direction."

    def _manage_open_position(
        self,
        ctx: BarContext,
        broker: SimulatedBroker,
        portfolio: Portfolio,
    ) -> None:
        if not self.enable_trailing_sl:
            return

        pos: Position = portfolio.positions[ctx.symbol]
        if ctx.timestamp <= pos.entry_time:
            return

        tick_size = portfolio.get_tick_size(ctx.symbol)
        buffer_pts = 2 * tick_size
        risk_pts = max(pos.initial_risk_points, tick_size)
        full_tp_dist = abs(pos.take_profit - pos.entry_price)

        if pos.direction == "LONG":
            fav_dist = ctx.high - pos.entry_price
        else:
            fav_dist = pos.entry_price - ctx.low

        prev_max_fav = float(pos.metadata.get("max_fav_dist", 0.0))
        max_fav_dist = max(prev_max_fav, fav_dist)
        pos.metadata["max_fav_dist"] = max_fav_dist

        progress_to_full_tp = (max_fav_dist / full_tp_dist) if full_tp_dist > 0 else 0.0
        num_partials_hit = len(pos.partial_exits)
        tp1_hit = num_partials_hit >= 1
        tp2_hit = num_partials_hit >= 2

        wanted_struct_dir = "BULLISH" if pos.direction == "LONG" else "BEARISH"
        opposite_struct_dir = "BEARISH" if pos.direction == "LONG" else "BULLISH"

        # 1. Straight-expansion Break-Even protection ONLY on single-TP trades (when >= 80% toward FullTP)
        has_planned_partials = len(pos.partial_targets) > 0
        if not has_planned_partials and progress_to_full_tp >= 0.80 and abs(pos.stop_loss - pos.initial_stop_loss) < 1e-9:
            if pos.direction == "LONG":
                be_swings = [
                    s for s in ctx.confirmed_swings
                    if s["type"] == "SwingLow"
                    and s["time"] >= pos.entry_time
                    and (pos.entry_price - 0.15 * risk_pts) <= float(s["price"]) <= (pos.entry_price + 0.35 * risk_pts)
                ]
                if be_swings:
                    best_be = max(be_swings, key=lambda x: x["time"])
                    cand_sl = float(best_be["price"]) - buffer_pts
                    if cand_sl < ctx.low - buffer_pts:
                        pos.update_stop_loss(
                            cand_sl,
                            ctx.timestamp,
                            f"below {best_be['time'].strftime('%H:%M')} BE swing low @ {best_be['price']:.2f}",
                        )
            else:
                be_swings = [
                    s for s in ctx.confirmed_swings
                    if s["type"] == "SwingHigh"
                    and s["time"] >= pos.entry_time
                    and (pos.entry_price - 0.35 * risk_pts) <= float(s["price"]) <= (pos.entry_price + 0.15 * risk_pts)
                ]
                if be_swings:
                    best_be = max(be_swings, key=lambda x: x["time"])
                    cand_sl = float(best_be["price"]) + buffer_pts
                    if cand_sl > ctx.high + buffer_pts:
                        pos.update_stop_loss(
                            cand_sl,
                            ctx.timestamp,
                            f"above {best_be['time'].strftime('%H:%M')} BE swing high @ {best_be['price']:.2f}",
                        )

        # 2. Check if the current 1m bar completed a structure break (or counter-FVG inversion) back in the trade direction
        recent_1m_events = ctx.structure_events_1m[-10:] if ctx.structure_events_1m else []
        cont_event = None
        for ev in reversed(recent_1m_events):
            if ev["direction"] == wanted_struct_dir and ev["close_time"] == ctx.timestamp:
                cont_event = ev
                break

        last_trail_ts = pos.metadata.get("last_trail_event_time", pos.entry_time)

        # Check if a counter-direction 1m FVG formed during a post-entry pullback
        # was inverted on the current bar (e.g., 09:53 BISI inverted at 09:56 after sweeping 30800.75 on Oct 1)
        counter_fvg_type = "SIBI" if pos.direction == "LONG" else "BISI"
        had_counter_fvg_inversion = False
        if cont_event is None:
            for rf in getattr(ctx, "recent_fvgs_all", []):
                if rf.fvg_type == counter_fvg_type and rf.creation_time > last_trail_ts:
                    for ev_name, ev_ts, _ in rf.events:
                        if ev_name == "INVERTED" and ev_ts == ctx.timestamp:
                            cont_event = {
                                "time": ctx.timestamp,
                                "close_time": ctx.timestamp,
                                "kind": "IFVG_BOS",
                                "direction": wanted_struct_dir,
                            }
                            had_counter_fvg_inversion = True
                            break
                    if cont_event is not None:
                        break

        if cont_event is None:
            return

        if cont_event["close_time"] <= last_trail_ts:
            return

        # Only advance `window_anchor_ts` to TP2's timestamp if TP2 occurred at least 2 minutes BEFORE
        # the current bar (so a TP2 hit on the same bar as `cont_event` never wipes out the pullback window)
        window_anchor_ts = last_trail_ts
        prior_tp2_exits = [
            pe for pe in pos.partial_exits
            if pe.get("label") == "TP2" and pe["time"] <= (ctx.timestamp - pd.Timedelta(minutes=2))
        ]
        if prior_tp2_exits:
            last_tp2_ts = prior_tp2_exits[-1]["time"]
            if last_tp2_ts > window_anchor_ts:
                window_anchor_ts = last_tp2_ts

        had_counter_bos = any(
            e["direction"] == opposite_struct_dir and window_anchor_ts < e["close_time"] < cont_event["close_time"]
            for e in recent_1m_events
        )

        fvg_type_needed = "BISI" if pos.direction == "LONG" else "SIBI"
        had_fvg_inversion = False
        for rf in getattr(ctx, "recent_fvgs_all", []):
            if rf.fvg_type == fvg_type_needed and rf.creation_time >= (pos.entry_time - pd.Timedelta(minutes=2)):
                for ev_name, ev_ts, _ in rf.events:
                    if ev_name == "INVERTED" and window_anchor_ts < ev_ts < cont_event["close_time"]:
                        had_fvg_inversion = True
                        break
                if had_fvg_inversion:
                    break

        window_df = ctx.history_df[ctx.history_df.index > window_anchor_ts]
        if len(window_df) < 2:
            return

        if pos.direction == "LONG":
            pullback_low = float(window_df["low"].min())
            pullback_time = window_df["low"].idxmin()
            prior_swings = [
                s for s in ctx.confirmed_swings
                if s["type"] == "SwingLow" and pos.entry_time <= s["time"] < pullback_time
            ]
            swept_cluster = sum(1 for s in prior_swings if pullback_low < float(s["price"])) >= 1
            cluster_touches = int((window_df["low"] <= (pullback_low + buffer_pts)).sum())
            if not (had_counter_bos or had_fvg_inversion or had_counter_fvg_inversion or swept_cluster or (tp2_hit and cluster_touches >= 2)):
                return

            raw_new_sl = pullback_low if (tp2_hit and cluster_touches >= 2) else (pullback_low - buffer_pts)
            if tp2_hit:
                tp1_price = float(pos.partial_exits[0]["price"])
                cand_sl = min(raw_new_sl, tp1_price + 0.15 * risk_pts)
            elif tp1_hit or progress_to_full_tp >= 0.75:
                cand_sl = min(raw_new_sl, pos.entry_price + 0.35 * risk_pts)
            else:
                if raw_new_sl > pos.entry_price:
                    return
                cand_sl = raw_new_sl

            if cand_sl > pos.stop_loss + tick_size and cand_sl < ctx.low - tick_size:
                if pos.update_stop_loss(
                    cand_sl,
                    ctx.timestamp,
                    f"below {pullback_time.strftime('%H:%M')} retrace cluster low @ {pullback_low:.2f}",
                ):
                    pos.metadata["last_trail_event_time"] = ctx.timestamp

        else:
            pullback_high = float(window_df["high"].max())
            pullback_time = window_df["high"].idxmax()
            prior_swings = [
                s for s in ctx.confirmed_swings
                if s["type"] == "SwingHigh" and pos.entry_time <= s["time"] < pullback_time
            ]
            swept_cluster = sum(1 for s in prior_swings if pullback_high > float(s["price"])) >= 1
            cluster_touches = int((window_df["high"] >= (pullback_high - buffer_pts)).sum())
            if not (had_counter_bos or had_fvg_inversion or had_counter_fvg_inversion or swept_cluster or (tp2_hit and cluster_touches >= 2)):
                return

            raw_new_sl = pullback_high if (tp2_hit and cluster_touches >= 2) else (pullback_high + buffer_pts)
            if tp2_hit:
                tp1_price = float(pos.partial_exits[0]["price"])
                cand_sl = max(raw_new_sl, tp1_price - 0.15 * risk_pts)
            elif tp1_hit or progress_to_full_tp >= 0.75:
                cand_sl = max(raw_new_sl, pos.entry_price - 0.35 * risk_pts)
            else:
                if raw_new_sl < pos.entry_price:
                    return
                cand_sl = raw_new_sl

            if cand_sl < pos.stop_loss - tick_size and cand_sl > ctx.high + tick_size:
                if pos.update_stop_loss(
                    cand_sl,
                    ctx.timestamp,
                    f"above {pullback_time.strftime('%H:%M')} retrace cluster high @ {pullback_high:.2f}",
                ):
                    pos.metadata["last_trail_event_time"] = ctx.timestamp

    def _evaluate_retracement_and_trigger(
        self,
        ctx: BarContext,
        direction: str,
        is_5m_trend_aligned: bool,
        has_intraday_post_open_smt: bool,
        is_ath_bullish_regime: bool = False,
    ) -> Tuple[bool, str, str, int, List[FairValueGap], str, str]:
        wanted_struct_dir = "BULLISH" if direction == "LONG" else "BEARISH"
        opposite_struct_dir = "BEARISH" if direction == "LONG" else "BULLISH"

        recent_1m_events = ctx.structure_events_1m[-6:] if ctx.structure_events_1m else []
        confirm_event = None
        for ev in reversed(recent_1m_events):
            if ev["direction"] == wanted_struct_dir and (ctx.timestamp - ev["close_time"]) <= pd.Timedelta(minutes=1):
                confirm_event = ev
                break

        if confirm_event is None:
            return False, "NONE", "", 0, [], "", ""

        trigger_desc = f"1m {confirm_event['label']} at {confirm_event['close_time'].strftime('%H:%M')}"

        if confirm_event["close_time"] < ctx.timestamp:
            if direction == "LONG" and ctx.close < ctx.open:
                return False, "NONE", "", 0, [], trigger_desc, "1m follow-through bar closed bearish against Long trigger."
            if direction == "SHORT" and ctx.close > ctx.open:
                return False, "NONE", "", 0, [], trigger_desc, "1m follow-through bar closed bullish against Short trigger."

        # Block counter-trend Bearish SMT shorts when in ATH + Dual-Index HTF-Bullish regime (e.g., Oct 6 10:03 ES Short)
        if direction == "SHORT" and is_ath_bullish_regime:
            return (
                False,
                "NONE",
                "",
                0,
                [],
                trigger_desc,
                "ATH + Dual-Index HTF-Bullish Regime active (ES & NQ 4H/1H/15m BULLISH): Counter-trend Bearish SMT shorts are forbidden at ATH.",
            )

        # MODEL 2: Counter-5m Intraday Post-Open SMT Reversal
        if not is_5m_trend_aligned:
            if has_intraday_post_open_smt:
                recent_6 = ctx.history_df.iloc[-6:]
                if direction == "LONG":
                    dist_from_manipulation = ctx.close - float(recent_6["low"].min())
                else:
                    dist_from_manipulation = float(recent_6["high"].max()) - ctx.close

                if dist_from_manipulation <= (self.stop_loss_points * 1.5):
                    return (
                        True,
                        "SMT_REVERSAL_MODEL",
                        f"Confirmed Intraday {wanted_struct_dir.title()} SMT Reversal + {trigger_desc}",
                        3,
                        [],
                        trigger_desc,
                        "",
                    )
                return (
                    False,
                    "NONE",
                    "",
                    0,
                    [],
                    trigger_desc,
                    f"Intraday SMT present, but price already moved {dist_from_manipulation:.2f} pts from manipulation extreme (> {self.stop_loss_points * 1.5:.2f} pts).",
                )
            return (
                False,
                "NONE",
                "",
                0,
                [],
                trigger_desc,
                f"5m trend is opposite ({ctx.htf_trends.get('5min')}) and no confirmed intraday post-09:30 {wanted_struct_dir.title()} SMT is active.",
            )

        # MODEL 1: 5m Trend Continuation
        fvg_type_needed = "BISI" if direction == "LONG" else "SIBI"
        pullback_bars = ctx.history_df.iloc[-6:]
        pullback_low_min = float(pullback_bars["low"].min())
        pullback_high_max = float(pullback_bars["high"].max())
        price = ctx.close
        max_fvg_dist = self.stop_loss_points * 1.5
        is_short_synthetic = len(ctx.history_df) < 25

        touched_5m_fvgs: List[FairValueGap] = []
        fvg_5m_strs: List[str] = []

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

        inverted_1m_fvg_desc = ""
        for rf in getattr(ctx, "recent_fvgs_all", []):
            if rf.fvg_type == fvg_type_needed:
                for ev_name, ev_ts, _ in rf.events:
                    if ev_name == "INVERTED" and (confirm_event["close_time"] - pd.Timedelta(minutes=6)) <= ev_ts < confirm_event["close_time"]:
                        inverted_1m_fvg_desc = (
                            f"1m {rf.fvg_type} [{rf.bottom:.2f}-{rf.top:.2f}] inverted at {ev_ts.strftime('%H:%M')}"
                        )
                        break

        if direction == "LONG" and is_ath_bullish_regime and ctx.liquidity_taken_level in ("MEDIUM", "HIGH"):
            if has_intraday_post_open_smt and (counter_events or inverted_1m_fvg_desc):
                retrace_label = (
                    inverted_1m_fvg_desc
                    if inverted_1m_fvg_desc
                    else f"1m Counter-{counter_events[-1]['kind']} at {counter_events[-1]['close_time'].strftime('%H:%M')}"
                )
                detail = (
                    f"ATH + Dual-Index HTF-Bullish Regime: {retrace_label} "
                    f"+ Confirmed 1m Bullish SMT + {trigger_desc}"
                )
                return True, "ATH_SMT_CONTINUATION", detail, 3, [], trigger_desc, ""
            return (
                False,
                "NONE",
                "",
                0,
                [],
                trigger_desc,
                "ATH regime with London BSL already swept requires 1m FVG inversion/counter-break + confirmed 1m Bullish SMT.",
            )

        if touched_5m_fvgs and in_favorable_eq:
            detail = f"{fvg_5m_strs[0]} + {eq_detail_str}"
            return True, "5M_FVG_AND_EQ", detail, 3, touched_5m_fvgs, trigger_desc, ""
        if touched_5m_fvgs:
            return True, "5M_FVG_RETRACE", fvg_5m_strs[0], 3, touched_5m_fvgs, trigger_desc, ""
        if in_favorable_eq:
            return True, "5M_EQ_RETRACE", eq_detail_str, 3, [], trigger_desc, ""
        if in_ote:
            return True, "5M_OTE_RETRACE", ote_detail_str, 2, [], trigger_desc, ""

        if direction == "LONG" and is_ath_bullish_regime and has_intraday_post_open_smt and (counter_events or inverted_1m_fvg_desc):
            retrace_label = (
                inverted_1m_fvg_desc
                if inverted_1m_fvg_desc
                else f"1m Counter-{counter_events[-1]['kind']} at {counter_events[-1]['close_time'].strftime('%H:%M')}"
            )
            detail = (
                f"ATH + Dual-Index HTF-Bullish Regime: {retrace_label} "
                f"+ Confirmed 1m Bullish SMT + {trigger_desc}"
            )
            return True, "ATH_SMT_CONTINUATION", detail, 3, [], trigger_desc, ""

        if counter_events:
            last_ce = counter_events[-1]
            detail = f"1m Counter-{last_ce['kind']} at {last_ce['close_time'].strftime('%H:%M')} (No 5m FVG/EQ hit)"
            return True, "1M_COUNTER_BOS_ONLY", detail, 1, [], trigger_desc, ""

        return (
            False,
            "NONE",
            "",
            0,
            [],
            trigger_desc,
            "No valid 5m FVG, 5m Equilibrium, 5m OTE, or prior 1m counter-BOS pullback occurred before this 1m break.",
        )

    def _compute_stop_loss(
        self,
        ctx: BarContext,
        direction: str,
        matching_fvgs: List[FairValueGap],
        tick_size: float,
        retrace_type: str = "",
        retrace_pts: int = 1,
    ) -> Tuple[Optional[float], float, str]:
        price = ctx.close
        buffer_pts = 2 * tick_size  # 0.50 pt buffer on ES/NQ
        min_sl = 2.0 if ctx.symbol == "ES" else 8.0

        # Allow structural Stop-Losses up to 102.0 pts on NQ whenever backed by a 3-point
        # 5m FVG/EQ or SMT entry model (regardless of whether liquidity exhaustion is LOW, MEDIUM, or HIGH).
        if ctx.symbol == "NQ" and retrace_pts >= 3:
            max_valid_sl = max(self.stop_loss_points * 1.6, 102.0)
        else:
            max_valid_sl = self.stop_loss_points * 1.6

        if self.sl_mode == "wide_5m" and ctx.confirmed_swings_5m:
            if direction == "LONG":
                lows_5m = [s for s in ctx.confirmed_swings_5m[-3:] if s["type"] == "SwingLow" and s["price"] < price]
                if lows_5m:
                    best_s = min(lows_5m, key=lambda x: x["price"])
                    dist = (price - float(best_s["price"])) + buffer_pts
                    anchor_str = f"below 5m SwingLow @ {best_s['price']:.2f} (formed {best_s['time'].strftime('%H:%M')})"
                    return (
                        (float(max(min_sl, dist)), float(dist), anchor_str)
                        if dist <= max_valid_sl * 1.25
                        else (None, float(dist), anchor_str)
                    )
            else:
                highs_5m = [s for s in ctx.confirmed_swings_5m[-3:] if s["type"] == "SwingHigh" and s["price"] > price]
                if highs_5m:
                    best_s = max(highs_5m, key=lambda x: x["price"])
                    dist = (float(best_s["price"]) - price) + buffer_pts
                    anchor_str = f"above 5m SwingHigh @ {best_s['price']:.2f} (formed {best_s['time'].strftime('%H:%M')})"
                    return (
                        (float(max(min_sl, dist)), float(dist), anchor_str)
                        if dist <= max_valid_sl * 1.25
                        else (None, float(dist), anchor_str)
                    )

        use_tight_smt_peak = (retrace_type == "SMT_REVERSAL_MODEL" and ctx.symbol == "ES")
        use_exact_cluster_low = (
            ctx.symbol == "ES"
            and direction == "LONG"
            and retrace_type == "ATH_SMT_CONTINUATION"
        )
        lookback_bars = 5 if use_tight_smt_peak else 8
        swing_age_min = 5 if use_tight_smt_peak else 12
        swing_slice = 4

        pullback_bars = ctx.history_df.iloc[-lookback_bars:]
        if direction == "LONG":
            anchor_price = float(pullback_bars["low"].min())
            anchor_time = pullback_bars["low"].idxmin()
            if ctx.confirmed_swings:
                recent_lows = [
                    s for s in ctx.confirmed_swings[-swing_slice:]
                    if s["type"] == "SwingLow"
                    and (ctx.timestamp - s["time"]) <= pd.Timedelta(minutes=swing_age_min)
                    and s["price"] < price
                ]
                if recent_lows:
                    cand_s = min(recent_lows, key=lambda x: x["price"])
                    if float(cand_s["price"]) < anchor_price:
                        anchor_price = float(cand_s["price"])
                        anchor_time = cand_s["time"]

            eff_buf = 0.0 if use_exact_cluster_low else buffer_pts
            dist = (price - anchor_price) + eff_buf
            anchor_str = f"below {anchor_time.strftime('%H:%M')} low @ {anchor_price:.2f}"
            if dist > max_valid_sl and len(ctx.history_df) >= 25:
                return None, float(dist), anchor_str
            return float(max(min_sl, dist)), float(dist), anchor_str
        else:
            anchor_price = float(pullback_bars["high"].max())
            anchor_time = pullback_bars["high"].idxmax()
            if ctx.confirmed_swings:
                recent_highs = [
                    s for s in ctx.confirmed_swings[-swing_slice:]
                    if s["type"] == "SwingHigh"
                    and (ctx.timestamp - s["time"]) <= pd.Timedelta(minutes=swing_age_min)
                    and s["price"] > price
                ]
                if recent_highs:
                    cand_s = max(recent_highs, key=lambda x: x["price"])
                    if float(cand_s["price"]) > anchor_price:
                        anchor_price = float(cand_s["price"])
                        anchor_time = cand_s["time"]

            dist = (anchor_price - price) + buffer_pts
            anchor_str = f"above {anchor_time.strftime('%H:%M')} high @ {anchor_price:.2f}"
            if dist > max_valid_sl and len(ctx.history_df) >= 25:
                return None, float(dist), anchor_str
            return float(max(min_sl, dist)), float(dist), anchor_str

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
        has_cross_index_htf_smt_alignment: bool = False,
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
            pros.append(f"Intraday / 15m Equal H/L Target ({target_info['target_type']}, RR={target_info['rr']:.2f})")
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
        elif aligned_count == 1 and has_cross_index_htf_smt_alignment:
            score += 2
            pros.append(f"HTF Structure Aligned via Confirmed 1H/4H SMT & Companion HTF MSS (4H={t_4h}, 1H={t_1h}, 15m={t_15m})")
        elif aligned_count == 1:
            score += 1
            cons.append(f"Mixed HTF Structure (4H={t_4h}, 1H={t_1h}, 15m={t_15m})")
        elif has_smt or "WEEKEND_GAP" in target_info["target_type"]:
            score += 1
            cons.append(
                f"Counter-HTF on Primary Index (4H={t_4h}, 1H={t_1h}, 15m={t_15m} — Overridden by Confirmed Cross-Index SMT)"
            )
        else:
            score -= 1
            cons.append(f"Counter-HTF Structure (4H={t_4h}, 1H={t_1h}, 15m={t_15m})")

        if liquidity_taken_level == "HIGH":
            score -= 1 if has_smt else 3
            cons.append("High Cross-Index Session/HTF Liquidity Already Taken Prior to Entry")
        elif liquidity_taken_level == "MEDIUM":
            pros.append("Medium Pre-Entry Liquidity Taken (Primary Targets Intact)")
        else:
            pros.append("Low Pre-Entry Liquidity Exhaustion (Clean Draw Ahead)")

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

    def _calculate_contracts(self, portfolio: Portfolio, symbol: str, actual_risk_pts: float) -> Tuple[ float, str ]:
        """
        Targets ~$900-$1,000 risk and NEVER exceeds $1,100 risk.
        1. Tries whole standard contracts (ES / NQ) first:
           - If `low_lots` risk is > $800 (and <= $1,000), uses `low_lots` (or `high_lots` if <= $1,100 and closer to $1,000).
           - If `low_lots` risk is too low (<= $800), raises to `high_lots` ONLY if `high_lots` risk <= $1,100.
        2. If standard contracts cannot land in ($800, $1,100] (e.g., 1 NQ > $1,100 on Oct 1, or 2 ES = $750 / 3 ES = $1,125 on Oct 6),
           automatically switches to Micro contracts (MNQ / MES, where 1 Micro = 0.1 Standard contract) to size right around ~$1,000!
        """
        clean_sym = symbol.upper()
        if not self.use_dynamic_sizing or actual_risk_pts <= 0:
            return float(self.quantity), f"{self.quantity} {clean_sym}"

        target_risk = self.initial_target_risk if hasattr(self, "initial_target_risk") else (100_000.0 * self.risk_per_trade_pct)
        min_acceptable_risk = target_risk * 0.80  # $800.00
        max_acceptable_risk = target_risk * 1.10  # $1,100.00

        std_point_val = portfolio.get_point_value(clean_sym)
        std_unit_risk = actual_risk_pts * std_point_val
        if std_unit_risk <= 0:
            return float(self.quantity), f"{self.quantity} {clean_sym}"

        low_std = int(target_risk // std_unit_risk)
        high_std = low_std + 1

        low_std_risk = low_std * std_unit_risk
        high_std_risk = high_std * std_unit_risk

        chosen_std: Optional[int] = None
        if low_std >= 1 and low_std_risk > min_acceptable_risk:
            if high_std_risk <= max_acceptable_risk and abs(high_std_risk - target_risk) < abs(low_std_risk - target_risk):
                chosen_std = high_std
            else:
                chosen_std = low_std
        elif high_std >= 1 and high_std_risk <= max_acceptable_risk and high_std_risk > min_acceptable_risk:
            chosen_std = high_std

        if chosen_std is not None:
            return float(chosen_std), f"{chosen_std} {clean_sym}"

        # Fallback to Micro contracts (MES = 0.1 ES, MNQ = 0.1 NQ)
        micro_sym = f"M{clean_sym}" if not clean_sym.startswith("M") else clean_sym
        micro_unit_risk = std_unit_risk * 0.10

        low_micro = int(target_risk // micro_unit_risk)
        high_micro = low_micro + 1

        low_micro_risk = low_micro * micro_unit_risk
        high_micro_risk = high_micro * micro_unit_risk

        if low_micro >= 1 and low_micro_risk > min_acceptable_risk:
            if high_micro_risk <= max_acceptable_risk and abs(high_micro_risk - target_risk) < abs(low_micro_risk - target_risk):
                chosen_micro = high_micro
            else:
                chosen_micro = low_micro
        elif high_micro_risk <= max_acceptable_risk:
            chosen_micro = max(1, high_micro)
        else:
            chosen_micro = max(1, low_micro)

        # 1 Micro contract = 0.1 Standard contract equivalent for PnL math
        std_equiv_qty = round(chosen_micro * 0.10, 2)
        return std_equiv_qty, f"{chosen_micro} {micro_sym}"

    def on_bar(
        self,
        ctx: BarContext,
        broker: SimulatedBroker,
        portfolio: Portfolio,
    ) -> None:
        if portfolio.has_open_position(ctx.symbol):
            self._manage_open_position(ctx, broker, portfolio)
            return

        session = SessionManager.get_session(ctx.timestamp)
        if self.require_killzone and self.allowed_sessions and session not in self.allowed_sessions:
            return

        if self.require_ny_open_window:
            bar_time = ctx.timestamp.time()
            if not (self.entry_start_time <= bar_time <= self.entry_end_time):
                return

        trade_date = ctx.timestamp.date()
        date_str = trade_date.strftime("%Y-%m-%d")
        t_hm = ctx.timestamp.strftime("%H:%M")

        companion_info = (
            self.shared_daily_trade_dates.get(trade_date)
            if self.shared_daily_trade_dates is not None
            else None
        )
        companion_entry_ts = companion_info[1] if isinstance(companion_info, tuple) else None
        context_lock_cutoff = (
            companion_entry_ts.time() if companion_entry_ts is not None else time(9, 55)
        )

        summary = self.daily_window_summary.setdefault(
            date_str,
            {
                "htf_trends": dict(ctx.htf_trends),
                "context_time": t_hm,
                "liquidity_already_taken": ctx.liquidity_taken_level,
                "session_taken_pools": list(ctx.session_taken_pools),
                "smt_descriptions": [],
            },
        )
        if ctx.timestamp.time() <= context_lock_cutoff:
            summary["htf_trends"] = dict(ctx.htf_trends)
            summary["context_time"] = t_hm

        summary["liquidity_already_taken"] = ctx.liquidity_taken_level
        summary["session_taken_pools"] = list(ctx.session_taken_pools)
        for ev in ctx.recent_smt_events:
            d_str = ev.get("description", "")
            if d_str and d_str not in summary["smt_descriptions"]:
                summary["smt_descriptions"].append(d_str)

        if self._trades_taken_by_date.get(trade_date, 0) >= self.max_trades_per_day:
            return

        if companion_info is not None:
            other_sym = companion_info[0] if isinstance(companion_info, tuple) else str(companion_info)
            other_t_str = companion_entry_ts.strftime("%H:%M") if companion_entry_ts is not None else date_str
            self._log_diagnostic(
                date_str,
                f"Cross-Index Exclusivity: Companion index ({other_sym}) already executed the trade of the day at {other_t_str}.",
            )
            return

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

        intraday_bull_smt = any(bool(ev.get("is_intraday_post_open", False)) for ev in bull_smts)
        intraday_bear_smt = any(bool(ev.get("is_intraday_post_open", False)) for ev in bear_smts)

        has_htf_bull_smt = any(str(ev.get("timeframe", "")) in ("1h", "4h") for ev in bull_smts)
        has_htf_bear_smt = any(str(ev.get("timeframe", "")) in ("1h", "4h") for ev in bear_smts)

        has_5m_and_1m_bull_smt = (
            any(str(ev.get("timeframe", "")).startswith("5m") for ev in bull_smts)
            and any(str(ev.get("timeframe", "")).startswith("1m") for ev in bull_smts)
        )
        has_5m_and_1m_bear_smt = (
            any(str(ev.get("timeframe", "")).startswith("5m") for ev in bear_smts)
            and any(str(ev.get("timeframe", "")).startswith("1m") for ev in bear_smts)
        )

        trend_5m = ctx.htf_trends.get("5min", "NEUTRAL")
        t_4h = ctx.htf_trends.get("4h", "NEUTRAL")
        t_1h = ctx.htf_trends.get("1h", "NEUTRAL")
        t_15m = ctx.htf_trends.get("15min", "NEUTRAL")

        sec_4h = ctx.sec_htf_trends.get("4h", t_4h)
        sec_1h = ctx.sec_htf_trends.get("1h", t_1h)
        sec_15m = ctx.sec_htf_trends.get("15min", t_15m)

        is_ath_bullish_regime = (
            ctx.htf_trends.get("ath_regime") == "ACTIVE"
            and t_4h == "BULLISH"
            and t_1h == "BULLISH"
            and t_15m == "BULLISH"
            and sec_4h == "BULLISH"
            and sec_1h == "BULLISH"
            and sec_15m == "BULLISH"
        )

        all_active_smt_descriptions = [ev["description"] for ev in ctx.recent_smt_events]

        ny_open_ts = ctx.timestamp.replace(hour=9, minute=30, second=0, microsecond=0)
        rth_df = ctx.history_df[ctx.history_df.index >= ny_open_ts]
        rth_high = float(rth_df["high"].max()) if not rth_df.empty else price
        rth_low = float(rth_df["low"].min()) if not rth_df.empty else price
        rth_range = rth_high - rth_low
        max_exhaustion_leg_pts = 18.0 if ctx.symbol == "ES" else 150.0

        for direction in ("LONG", "SHORT"):
            wanted_bias = "BULLISH" if direction == "LONG" else "BEARISH"
            htf_aligned_count = sum(1 for b in (t_4h, t_1h, t_15m) if b == wanted_bias)
            sec_htf_aligned_count = sum(1 for b in (sec_4h, sec_1h, sec_15m) if b == wanted_bias)
            has_dir_htf_smt = has_htf_bull_smt if direction == "LONG" else has_htf_bear_smt
            has_cross_index_htf_smt_alignment = (
                htf_aligned_count >= 1
                and sec_htf_aligned_count >= 2
                and has_dir_htf_smt
            )

            is_synthetic_short_df = len(ctx.history_df) < 25
            dir_exhaustion = ctx.liquidity_taken_by_dir.get(direction, ctx.liquidity_taken_level)
            dir_taken_pools = ctx.session_taken_by_dir.get(direction, ctx.session_taken_pools)
            has_dir_intraday_smt = intraday_bull_smt if direction == "LONG" else intraday_bear_smt
            allow_or_target = (
                (has_5m_and_1m_bull_smt if direction == "LONG" else has_5m_and_1m_bear_smt)
                and htf_aligned_count >= 2
            )

            if direction == "LONG":
                has_catalyst = ssl_swept or bullish_smt or htf_aligned_count >= 2
                is_5m_trend_aligned = (trend_5m == "BULLISH" and (htf_aligned_count >= 1 or bullish_smt or is_synthetic_short_df))
                dir_sweep_details = [d["label"] for d in ssl_details]
            else:
                has_catalyst = bsl_swept or bearish_smt or htf_aligned_count >= 2
                is_5m_trend_aligned = (trend_5m == "BEARISH" and (htf_aligned_count >= 1 or bearish_smt or is_synthetic_short_df))
                dir_sweep_details = [d["label"] for d in bsl_details]

            triggered, retrace_type, retrace_detail, retrace_pts, matching_fvgs, trigger_desc, fail_reason = (
                self._evaluate_retracement_and_trigger(
                    ctx=ctx,
                    direction=direction,
                    is_5m_trend_aligned=is_5m_trend_aligned,
                    has_intraday_post_open_smt=has_dir_intraday_smt,
                    is_ath_bullish_regime=is_ath_bullish_regime,
                )
            )

            if not triggered:
                if trigger_desc and fail_reason:
                    self._log_diagnostic(date_str, f"[{t_hm}] {direction} ({trigger_desc}) -> Rejected: {fail_reason}")
                continue

            if not has_catalyst:
                self._log_diagnostic(
                    date_str,
                    f"[{t_hm}] {direction} ({trigger_desc}) -> Rejected: No opposite-side manipulation sweep, SMT, or HTF alignment.",
                )
                continue

            # Overextended Exhaustion-Leg Filter (e.g., Oct 16 10:26 NQ Short):
            if dir_exhaustion == "HIGH" and not is_synthetic_short_df:
                if direction == "SHORT":
                    dist_below_high = rth_high - price
                    if rth_range >= max_exhaustion_leg_pts and dist_below_high >= (max_exhaustion_leg_pts * 0.75):
                        self._log_diagnostic(
                            date_str,
                            f"[{t_hm}] SHORT ({trigger_desc}, {retrace_type}) -> Rejected: HIGH liquidity exhaustion "
                            f"+ large downside expansion leg already completed ({rth_range:.2f} pts drop from RTH high {rth_high:.2f}; "
                            f"currently {dist_below_high:.2f} pts below high).",
                        )
                        continue
                else:
                    dist_above_low = price - rth_low
                    if rth_range >= max_exhaustion_leg_pts and dist_above_low >= (max_exhaustion_leg_pts * 0.75):
                        self._log_diagnostic(
                            date_str,
                            f"[{t_hm}] LONG ({trigger_desc}, {retrace_type}) -> Rejected: HIGH liquidity exhaustion "
                            f"+ large upside expansion leg already completed ({rth_range:.2f} pts rally from RTH low {rth_low:.2f}; "
                            f"currently {dist_above_low:.2f} pts above low).",
                        )
                        continue

            raw_sl_dist, true_struct_dist, sl_anchor_str = self._compute_stop_loss(
                ctx,
                direction,
                matching_fvgs,
                tick_size,
                retrace_type=retrace_type,
                retrace_pts=retrace_pts,
            )
            if raw_sl_dist is None:
                eff_max_sl = max(self.stop_loss_points * 1.6, 102.0) if (ctx.symbol == "NQ" and retrace_pts >= 3) else (self.stop_loss_points * 1.6)
                self._log_diagnostic(
                    date_str,
                    f"[{t_hm}] {direction} ({trigger_desc}, {retrace_type}) -> Rejected: True structural Stop-Loss "
                    f"({true_struct_dist:.2f} pts, {sl_anchor_str}) exceeds max valid SL ({eff_max_sl:.2f} pts).",
                )
                continue

            sl_price = price - raw_sl_dist if direction == "LONG" else price + raw_sl_dist
            fill_price = price + slip_pts if direction == "LONG" else price - slip_pts
            actual_risk_pts = abs(fill_price - sl_price)

            allow_exception_tp1 = (
                retrace_pts >= 3
                and dir_exhaustion in ("LOW", "MEDIUM")
                and (
                    htf_aligned_count >= 2
                    or has_cross_index_htf_smt_alignment
                    or (bullish_smt if direction == "LONG" else bearish_smt)
                )
            )

            target_info, target_fail_reason = self._select_best_target(
                ctx,
                direction,
                fill_price,
                actual_risk_pts,
                tick_size,
                allow_opening_range_target=allow_or_target,
                allow_exception_tp1=allow_exception_tp1,
            )
            if target_info is None:
                self._log_diagnostic(
                    date_str,
                    f"[{t_hm}] {direction} ({trigger_desc}, {retrace_type}, SL={sl_price:.2f} [{sl_anchor_str}]) -> Rejected: {target_fail_reason}",
                )
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
                has_cross_index_htf_smt_alignment=has_cross_index_htf_smt_alignment,
            )

            partial_targets = target_info.get("partial_targets", [])

            setup_metadata = {
                "session": session,
                "grade": grade,
                "score": score,
                "target_type": target_info["target_type"],
                "target_tier": target_info["tier"],
                "planned_rr": round(target_info["rr"], 2),
                "partial_targets": partial_targets,
                "retracement_type": retrace_type,
                "retracement_detail": retrace_detail,
                "entry_trigger_detail": trigger_desc,
                "sl_mode": self.sl_mode,
                "sl_anchor_detail": sl_anchor_str,
                "liquidity_already_taken": dir_exhaustion,
                "session_taken_pools": list(dir_taken_pools),
                "htf_trends": dict(ctx.htf_trends),
                "ssl_swept": ssl_swept,
                "bsl_swept": bsl_swept,
                "bullish_smt": bullish_smt,
                "bearish_smt": bearish_smt,
                "swept_pools_detail": dir_sweep_details,
                "smt_detail": all_active_smt_descriptions,
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
                self._log_diagnostic(
                    date_str,
                    f"[{t_hm}] {direction} @ {fill_price:.2f} ({trigger_desc}, {retrace_type}, SL={sl_price:.2f} [{sl_anchor_str}], Target={target_info['target_type']}) "
                    f"-> Rejected: Graded {grade} (Score {score}/10) < min_grade '{self.min_grade}' | Cons: {'; '.join(cons)}",
                )
                continue

            tp_price = target_info["price"]
            contracts, display_qty_str = self._calculate_contracts(portfolio, ctx.symbol, actual_risk_pts)
            setup_metadata["display_qty"] = display_qty_str

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
            self._trades_taken_by_date[trade_date] = self._trades_taken_by_date.get(trade_date, 0) + 1
            if self.shared_daily_trade_dates is not None:
                self.shared_daily_trade_dates[trade_date] = (ctx.symbol, ctx.timestamp)
            return