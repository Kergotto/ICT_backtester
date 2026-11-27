# Copyright (C) 2026 Kyrylo Homberh
# SPDX-License-Identifier: AGPL-3.0-only

from datetime import time
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

from domain.fvg import FairValueGap
from domain.liquidity import SessionManager
from engine.event_bus import BarContext
from engine.broker import SimulatedBroker
from engine.portfolio import Portfolio
from strategies.base import BaseStrategy
from strategies.experience import (
    GRADE_RANK,
    ExperienceConfig,
    ICTExperienceLayer,
    grade_setup,
)
from strategies.trailing_stops import (
    BaseTrailingStopManager,
    ICTStructuralTrailingStop,
)


_UNSET_TRAILING_MANAGER = object()


class ICTMultiTimeframeStrategy(BaseStrategy):
    """
    3-Layer Modular ICT Multi-Timeframe Strategy:
    - Layer 1 (this file): Objective mechanical Entry Model (Killzone window,
      Model 1 5m Trend Continuation, Model 2 Intraday SMT Reversal, base structural SL,
      base DOL Target, base risk sizing).
    - Layer 2 (`strategies/experience.py`): Toggleable discretionary trader edge
      via `ExperienceConfig` (`ICTExperienceLayer`).
    - Layer 3 (`strategies/trailing_stops.py`): Standalone, strictly optional
      pluggable trailing stop manager (`ICTStructuralTrailingStop` or `None`).
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
        experience_config: Optional[ExperienceConfig] = None,
        trailing_stop_manager: Any = _UNSET_TRAILING_MANAGER,
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

        # Layer 2: Discretionary Experience Layer
        self.experience_config: ExperienceConfig = (
            experience_config if experience_config is not None else ExperienceConfig.full()
        )
        self.experience: ICTExperienceLayer = ICTExperienceLayer(self.experience_config)

        # Layer 3: Pluggable Trailing Stop Manager
        if trailing_stop_manager is _UNSET_TRAILING_MANAGER:
            self.trailing_stop_manager: Optional[BaseTrailingStopManager] = (
                ICTStructuralTrailingStop() if self.enable_trailing_sl else None
            )
        elif trailing_stop_manager is None:
            self.trailing_stop_manager = None
            self.enable_trailing_sl = False
        else:
            self.trailing_stop_manager = (
                trailing_stop_manager if self.enable_trailing_sl else None
            )

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
        return ICTExperienceLayer.target_priority_rank(target_type)

    def _collect_directional_targets(
        self,
        ctx: BarContext,
        direction: str,
        fill_price: float,
        actual_risk_pts: float,
        tick_size: float,
    ) -> List[Dict[str, Any]]:
        front_run = self.front_run_ticks * tick_size
        all_directional_targets: List[Dict[str, Any]] = []

        # 1. Unfilled Weekend Gaps (Priority 1)
        for gap in ctx.active_weekend_gaps:
            if direction == "LONG" and gap.lower_bound > fill_price:
                for bound_price, label in (
                        (gap.lower_bound, "WEEKEND_GAP_START"),
                        (gap.upper_bound, "WEEKEND_GAP_FILL"),
                ):
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
                for bound_price, label in (
                        (gap.upper_bound, "WEEKEND_GAP_START"),
                        (gap.lower_bound, "WEEKEND_GAP_FILL"),
                ):
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

        return all_directional_targets

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
        all_directional_targets = self._collect_directional_targets(
            ctx, direction, fill_price, actual_risk_pts, tick_size
        )

        # Layer 2: Discretionary target & partial TP heuristics
        if self.experience_config.enable_discretionary_targets:
            return self.experience.select_discretionary_target(
                ctx=ctx,
                direction=direction,
                fill_price=fill_price,
                actual_risk_pts=actual_risk_pts,
                all_directional_targets=all_directional_targets,
                min_rr=self.min_rr,
                min_tp1_rr=self.min_tp1_rr,
                max_rr=self.max_rr,
                partial_cut_pct=self.partial_cut_pct,
                enable_partials=self.enable_partials,
                reward_risk_ratio=self.reward_risk_ratio,
                allow_opening_range_target=allow_opening_range_target,
                allow_exception_tp1=allow_exception_tp1,
            )

        # Layer 1 (Pure Strategy): Base mechanical DOL target inside [min_rr, max_rr]
        valid_candidates = [
            c for c in all_directional_targets
            if c["priority"] <= 2 and self.min_rr <= c["rr"] <= self.max_rr
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
                f"No unbroken target inside [{self.min_rr:.2f}R, {self.max_rr:.2f}R] "
                f"(closest candidate was {best_rr['target_type']} at {best_rr['rr']:.2f}R).",
            )
        return None, "No unbroken Session/HTF target pools remaining ahead in trade direction."

    def _manage_open_position(
        self,
        ctx: BarContext,
        broker: SimulatedBroker,
        portfolio: Portfolio,
    ) -> None:
        if not self.enable_trailing_sl or self.trailing_stop_manager is None:
            return
        pos = portfolio.positions[ctx.symbol]
        self.trailing_stop_manager.on_bar_manage(ctx, pos, broker, portfolio)

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

        # Layer 2: Block counter-trend Bearish SMT shorts when in ATH + Dual-Index HTF-Bullish regime
        ath_block_reason = self.experience.check_ath_short_block(direction, is_ath_bullish_regime)
        if ath_block_reason is not None:
            return False, "NONE", "", 0, [], trigger_desc, ath_block_reason

        # MODEL 2: Counter-5m Intraday Post-Open SMT Reversal
        if not is_5m_trend_aligned:
            if has_intraday_post_open_smt:
                passes_chase, chase_fail = self.experience.check_smt_anti_chasing(
                    ctx, direction, confirm_event, self.stop_loss_points
                )
                if passes_chase:
                    return (
                        True,
                        "SMT_REVERSAL_MODEL",
                        f"Confirmed Intraday {wanted_struct_dir.title()} SMT Reversal + {trigger_desc}",
                        3,
                        [],
                        trigger_desc,
                        "",
                    )
                return False, "NONE", "", 0, [], trigger_desc, chase_fail
            if ctx.htf_trends.get("5min") == wanted_struct_dir:
                misalign_reason = (
                    f"5m is {wanted_struct_dir}, but all HTFs (4H/1H/15m) are opposing "
                    f"and no confirmed intraday post-09:30 {wanted_struct_dir.title()} SMT is active."
                )
            else:
                misalign_reason = (
                    f"5m trend is opposite ({ctx.htf_trends.get('5min')}) "
                    f"and no confirmed intraday post-09:30 {wanted_struct_dir.title()} SMT is active."
                )
            return False, "NONE", "", 0, [], trigger_desc, misalign_reason

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

        dr_ref = ctx.active_dealing_range_5m if not is_short_synthetic else (
                    ctx.active_dealing_range_5m or ctx.active_dealing_range)
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

        fib_ref = ctx.active_fib_range_5m if not is_short_synthetic else (
                    ctx.active_fib_range_5m or ctx.active_fib_range)
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
                    if ev_name == "INVERTED" and (confirm_event["close_time"] - pd.Timedelta(minutes=6)) <= ev_ts < \
                            confirm_event["close_time"]:
                        inverted_1m_fvg_desc = (
                            f"1m {rf.fvg_type} [{rf.bottom:.2f}-{rf.top:.2f}] inverted at {ev_ts.strftime('%H:%M')}"
                        )
                        break

        # Layer 2: ATH regime override when London BSL already swept
        ath_swept_res = self.experience.evaluate_ath_swept_continuation(
            ctx=ctx,
            direction=direction,
            is_ath_bullish_regime=is_ath_bullish_regime,
            has_intraday_post_open_smt=has_intraday_post_open_smt,
            counter_events=counter_events,
            inverted_1m_fvg_desc=inverted_1m_fvg_desc,
            trigger_desc=trigger_desc,
        )
        if ath_swept_res is not None:
            return ath_swept_res

        if touched_5m_fvgs and in_favorable_eq:
            detail = f"{fvg_5m_strs[0]} + {eq_detail_str}"
            return True, "5M_FVG_AND_EQ", detail, 3, touched_5m_fvgs, trigger_desc, ""
        if touched_5m_fvgs:
            return True, "5M_FVG_RETRACE", fvg_5m_strs[0], 3, touched_5m_fvgs, trigger_desc, ""
        if in_favorable_eq:
            return True, "5M_EQ_RETRACE", eq_detail_str, 3, [], trigger_desc, ""
        if in_ote:
            return True, "5M_OTE_RETRACE", ote_detail_str, 2, [], trigger_desc, ""

        # Layer 2: ATH regime fallback continuation override
        ath_fallback_res = self.experience.evaluate_ath_fallback_continuation(
            direction=direction,
            is_ath_bullish_regime=is_ath_bullish_regime,
            has_intraday_post_open_smt=has_intraday_post_open_smt,
            counter_events=counter_events,
            inverted_1m_fvg_desc=inverted_1m_fvg_desc,
            trigger_desc=trigger_desc,
        )
        if ath_fallback_res is not None:
            return ath_fallback_res

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

        sl_params = self.experience.get_stop_loss_parameters(
            symbol=ctx.symbol,
            direction=direction,
            retrace_type=retrace_type,
            retrace_pts=retrace_pts,
            stop_loss_points=self.stop_loss_points,
            buffer_pts=buffer_pts,
        )
        max_valid_sl: Optional[float] = sl_params["max_valid_sl"]
        lookback_bars: int = sl_params["lookback_bars"]
        swing_age_min: int = sl_params["swing_age_min"]
        eff_buf: float = sl_params["eff_buf"]

        if self.sl_mode == "wide_5m" and ctx.confirmed_swings_5m:
            if direction == "LONG":
                lows_5m = [s for s in ctx.confirmed_swings_5m[-3:] if s["type"] == "SwingLow" and s["price"] < price]
                if lows_5m:
                    best_s = min(lows_5m, key=lambda x: x["price"])
                    dist = (price - float(best_s["price"])) + buffer_pts
                    anchor_str = f"below 5m SwingLow @ {best_s['price']:.2f} (formed {best_s['time'].strftime('%H:%M')})"
                    return (
                        (float(max(min_sl, dist)), float(dist), anchor_str)
                        if (max_valid_sl is None or dist <= max_valid_sl * 1.25)
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
                        if (max_valid_sl is None or dist <= max_valid_sl * 1.25)
                        else (None, float(dist), anchor_str)
                    )

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

            dist = (price - anchor_price) + eff_buf
            anchor_str = f"below {anchor_time.strftime('%H:%M')} low @ {anchor_price:.2f}"
            if max_valid_sl is not None and dist > max_valid_sl and len(ctx.history_df) >= 25:
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
            if max_valid_sl is not None and dist > max_valid_sl and len(ctx.history_df) >= 25:
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
        return ICTExperienceLayer.grade_setup(
            direction=direction,
            target_info=target_info,
            retracement_type=retracement_type,
            retracement_pts=retracement_pts,
            has_smt=has_smt,
            has_macro_sweep=has_macro_sweep,
            has_structural_sweep=has_structural_sweep,
            htf_trends=htf_trends,
            liquidity_taken_level=liquidity_taken_level,
            has_cross_index_htf_smt_alignment=has_cross_index_htf_smt_alignment,
        )

    def _calculate_contracts(
        self,
        portfolio: Portfolio,
        symbol: str,
        actual_risk_pts: float,
        num_partials: int = 0,
    ) -> Tuple[float, str, int, str]:
        return self.experience.calculate_contracts(
            portfolio=portfolio,
            symbol=symbol,
            actual_risk_pts=actual_risk_pts,
            use_dynamic_sizing=self.use_dynamic_sizing,
            quantity=self.quantity,
            risk_per_trade_pct=self.risk_per_trade_pct,
            partial_cut_pct=self.partial_cut_pct,
            num_partials=num_partials,
        )

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

        companion_info, companion_entry_ts = self.experience.get_companion_trade_info(
            self.shared_daily_trade_dates, trade_date
        )
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
            self._log_diagnostic(
                date_str,
                self.experience.format_cross_index_rejection(companion_info, companion_entry_ts, date_str),
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

        bull_tfs = {
            tf for ev in bull_smts
            for tf in ev.get("all_timeframes", {str(ev.get("timeframe", ""))})
        }
        bear_tfs = {
            tf for ev in bear_smts
            for tf in ev.get("all_timeframes", {str(ev.get("timeframe", ""))})
        }
        has_5m_and_1m_bull_smt = any(tf.startswith("5m") for tf in bull_tfs) and any(
            tf.startswith("1m") for tf in bull_tfs)
        has_5m_and_1m_bear_smt = any(tf.startswith("5m") for tf in bear_tfs) and any(
            tf.startswith("1m") for tf in bear_tfs)

        trend_5m = ctx.htf_trends.get("5min", "NEUTRAL")
        t_4h = ctx.htf_trends.get("4h", "NEUTRAL")
        t_1h = ctx.htf_trends.get("1h", "NEUTRAL")
        t_15m = ctx.htf_trends.get("15min", "NEUTRAL")

        sec_4h = ctx.sec_htf_trends.get("4h", t_4h)
        sec_1h = ctx.sec_htf_trends.get("1h", t_1h)
        sec_15m = ctx.sec_htf_trends.get("15min", t_15m)

        is_ath_bullish_regime = self.experience.detect_ath_bullish_regime(
            ctx, price, t_4h, t_1h, t_15m, sec_4h, sec_1h, sec_15m
        )

        sec_5m = ctx.sec_htf_trends.get("5min", trend_5m)

        all_active_smt_descriptions = [ev["description"] for ev in ctx.recent_smt_events]

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
                is_5m_trend_aligned = (
                            trend_5m == "BULLISH" and (htf_aligned_count >= 1 or bullish_smt or is_synthetic_short_df))
                dir_sweep_details = [d["label"] for d in ssl_details]
            else:
                has_catalyst = bsl_swept or bearish_smt or htf_aligned_count >= 2
                is_5m_trend_aligned = (
                            trend_5m == "BEARISH" and (htf_aligned_count >= 1 or bearish_smt or is_synthetic_short_df))
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

            # Layer 2 Filters: Leader-vs-Laggard Gate & Exhaustion-Leg Filter
            laggard_msg = self.experience.check_laggard_index_smt_filter(
                direction=direction,
                retrace_type=retrace_type,
                trend_5m=trend_5m,
                sec_trend_5m=sec_5m,
            )
            if laggard_msg is not None:
                self._log_diagnostic(
                    date_str,
                    f"[{t_hm}] {direction} ({trigger_desc}, {retrace_type}) -> Rejected: {laggard_msg}",
                )
                continue

            exhaustion_msg = self.experience.check_exhaustion_leg_filter(
                ctx=ctx,
                direction=direction,
                price=price,
                dir_exhaustion=dir_exhaustion,
                is_synthetic_short_df=is_synthetic_short_df,
                t_hm=t_hm,
                trigger_desc=trigger_desc,
                retrace_type=retrace_type,
            )
            if exhaustion_msg is not None:
                self._log_diagnostic(date_str, exhaustion_msg)
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
                eff_max_sl = max(self.stop_loss_points * 1.6, 102.0) if (ctx.symbol == "NQ" and retrace_pts >= 3) else (
                            self.stop_loss_points * 1.6)
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

            distant_fade_reason = self.experience.check_counter_trend_distant_target_filter(
                direction=direction,
                retrace_type=retrace_type,
                t_15m=t_15m,
                htf_aligned_count=htf_aligned_count,
                target_info=target_info,
            )
            if distant_fade_reason is not None:
                self._log_diagnostic(
                    date_str,
                    f"[{t_hm}] {direction} ({trigger_desc}, {retrace_type}, SL={sl_price:.2f} [{sl_anchor_str}]) -> Rejected: {distant_fade_reason}",
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

            if not self.experience.passes_grade_filter(grade, self.min_grade):
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
            partial_targets_list = target_info.get("partial_targets", [])
            contracts, display_qty_str, total_int_lots, unit_sym = self._calculate_contracts(
                portfolio, ctx.symbol, actual_risk_pts, num_partials=len(partial_targets_list)
            )

            self.experience.quantize_partial_lots(
                partial_targets_list=partial_targets_list,
                total_int_lots=total_int_lots,
                unit_sym=unit_sym,
                partial_cut_pct=self.partial_cut_pct,
            )

            setup_metadata["display_qty"] = display_qty_str
            setup_metadata["total_int_lots"] = total_int_lots
            setup_metadata["unit_sym"] = unit_sym

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
            self.experience.record_cross_index_trade(
                self.shared_daily_trade_dates, trade_date, ctx.symbol, ctx.timestamp
            )
            return