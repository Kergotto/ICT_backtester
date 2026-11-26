# Copyright (C) 2026 Kyrylo Homberh
# SPDX-License-Identifier: AGPL-3.0-only

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd

from engine.event_bus import BarContext
from engine.broker import SimulatedBroker
from engine.portfolio import Portfolio, Position


class BaseTrailingStopManager(ABC):
    """
    Pluggable interface for post-entry active Stop-Loss management.
    Pass `enable_trailing_sl=False` or `trailing_stop_manager=None` to run
    Fixed Set-and-Forget SL with zero impact on entry or experience logic.
    """

    @abstractmethod
    def on_bar_manage(
        self,
        ctx: BarContext,
        position: Position,
        broker: SimulatedBroker,
        portfolio: Portfolio,
    ) -> None:
        """Called on every 1m bar while `position` is open."""
        pass

    def reset(self) -> None:
        """Optional hook to clear per-run state if a manager instance is reused."""
        pass


class ICTStructuralTrailingStop(BaseTrailingStopManager):
    """
    Natural 1m Structural Pullback, FVG Inversion (`IFVG_RECLAIM`),
    Liquidity Cluster Sweep, and Single-TP Break-Even Trailing Stop Manager.
    """

    def on_bar_manage(
            self,
            ctx: BarContext,
            position: Position,
            broker: SimulatedBroker,
            portfolio: Portfolio,
    ) -> None:
        pos: Position = position
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
        tp2_hit_prior = any(
            pe.get("label") == "TP2" and pe["time"] < ctx.timestamp
            for pe in pos.partial_exits
        )
        already_trailed_once = abs(pos.stop_loss - pos.initial_stop_loss) > 1e-9

        wanted_struct_dir = "BULLISH" if pos.direction == "LONG" else "BEARISH"
        opposite_struct_dir = "BEARISH" if pos.direction == "LONG" else "BULLISH"

        # 1. Straight-expansion Break-Even protection ONLY on single-TP trades (when >= 80% toward FullTP)
        has_planned_partials = len(pos.partial_targets) > 0
        if not has_planned_partials and progress_to_full_tp >= 0.80 and not already_trailed_once:
            if pos.direction == "LONG":
                be_swings = [
                    s for s in ctx.confirmed_swings
                    if s["type"] == "SwingLow"
                       and s["time"] >= pos.entry_time
                       and (pos.entry_price - 0.15 * risk_pts) <= float(s["price"]) <= (
                               pos.entry_price + 0.35 * risk_pts)
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
                       and (pos.entry_price - 0.35 * risk_pts) <= float(s["price"]) <= (
                               pos.entry_price + 0.15 * risk_pts)
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

        # 2. Check if the current 1m bar completed a structure break back in the trade direction
        recent_1m_events = ctx.structure_events_1m[-10:] if ctx.structure_events_1m else []
        cont_event = None
        for ev in reversed(recent_1m_events):
            if ev["direction"] == wanted_struct_dir and ev["close_time"] == ctx.timestamp:
                cont_event = ev
                break

        last_trail_ts = pos.metadata.get("last_trail_event_time", pos.entry_time)
        already_structurally_trailed = "last_trail_event_time" in pos.metadata
        last_trailed_pivot_ts = pos.metadata.get("last_trailed_pivot_time", pos.entry_time)

        # Advance `window_anchor_ts` to the most recent partial exit (TP1 or TP2) that occurred
        # strictly BEFORE the current bar (`pe["time"] < ctx.timestamp`).
        window_anchor_ts = last_trail_ts
        prior_partial_exits = [
            pe for pe in pos.partial_exits
            if pe["time"] < ctx.timestamp
        ]
        if prior_partial_exits:
            last_pe_ts = prior_partial_exits[-1]["time"]
            if last_pe_ts > window_anchor_ts:
                window_anchor_ts = last_pe_ts

        fvg_type_needed = "BISI" if pos.direction == "LONG" else "SIBI"
        reclaimed_fvg_origin_ts: Optional[pd.Timestamp] = None

        # Detect when a 1m BISI/SIBI FVG that was inverted during the pullback (e.g. Oct 8 10:14)
        # is reclaimed by a candle close back across the FVG boundary (e.g. Oct 8 10:18 closing @ 7838.25)
        if cont_event is None:
            for rf in getattr(ctx, "recent_fvgs_all", []):
                if rf.fvg_type != fvg_type_needed or rf.creation_time < (pos.entry_time - pd.Timedelta(minutes=2)):
                    continue
                inv_times = [
                    ev_ts for ev_name, ev_ts, _ in rf.events
                    if ev_name == "INVERTED" and window_anchor_ts < ev_ts < (ctx.timestamp - pd.Timedelta(minutes=1))
                ]
                if not inv_times:
                    continue
                inv_ts = inv_times[-1]
                post_inv_df = ctx.history_df[(ctx.history_df.index > inv_ts) & (ctx.history_df.index < ctx.timestamp)]
                if len(post_inv_df) < 2:
                    continue
                if pos.direction == "LONG":
                    if float(post_inv_df["close"].max()) < rf.top and ctx.close >= rf.top and ctx.close > ctx.open:
                        cont_event = {
                            "time": ctx.timestamp,
                            "close_time": ctx.timestamp,
                            "kind": "IFVG_RECLAIM",
                            "direction": "BULLISH",
                            "broken_level": float(rf.top),
                        }
                        reclaimed_fvg_origin_ts = rf.creation_time - pd.Timedelta(minutes=2)
                        break
                else:
                    if float(
                            post_inv_df["close"].min()) > rf.bottom and ctx.close <= rf.bottom and ctx.close < ctx.open:
                        cont_event = {
                            "time": ctx.timestamp,
                            "close_time": ctx.timestamp,
                            "kind": "IFVG_RECLAIM",
                            "direction": "BEARISH",
                            "broken_level": float(rf.bottom),
                        }
                        reclaimed_fvg_origin_ts = rf.creation_time - pd.Timedelta(minutes=2)
                        break

        if cont_event is None:
            return

        if cont_event["close_time"] <= last_trail_ts:
            return

        # Identify all FVGs of `fvg_type_needed` in the active leg (since `last_trailed_pivot_ts`)
        # that were still uninverted entering `window_anchor_ts`. Only inverting the LAST (most extreme)
        # FVG of the leg counts as an FVG trend break to the opposite side.
        leg_fvgs = [
            rf for rf in getattr(ctx, "recent_fvgs_all", [])
            if rf.fvg_type == fvg_type_needed
               and last_trailed_pivot_ts <= rf.creation_time < cont_event["close_time"]
               and not any(
                ev_name == "INVERTED" and ev_ts <= window_anchor_ts
                for ev_name, ev_ts, _ in rf.events
            )
        ]
        if pos.direction == "LONG":
            last_fvg_of_leg = min(leg_fvgs, key=lambda f: f.bottom) if leg_fvgs else None
        else:
            last_fvg_of_leg = max(leg_fvgs, key=lambda f: f.top) if leg_fvgs else None

        had_fvg_inversion = cont_event.get("kind") == "IFVG_RECLAIM"
        if not had_fvg_inversion and last_fvg_of_leg is not None:
            for ev_name, ev_ts, _ in last_fvg_of_leg.events:
                if ev_name == "INVERTED" and window_anchor_ts < ev_ts < cont_event["close_time"]:
                    had_fvg_inversion = True
                    if reclaimed_fvg_origin_ts is None:
                        reclaimed_fvg_origin_ts = last_fvg_of_leg.creation_time - pd.Timedelta(minutes=2)
                    break

        had_counter_bos = False
        for e in recent_1m_events:
            if e["direction"] == opposite_struct_dir and window_anchor_ts < e["close_time"] < cont_event["close_time"]:
                if e.get("kind") == "IFVG_BOS" and not had_fvg_inversion:
                    continue
                had_counter_bos = True
                break

        eff_window_start = window_anchor_ts
        if (
                reclaimed_fvg_origin_ts is not None
                and reclaimed_fvg_origin_ts > last_trailed_pivot_ts
                and reclaimed_fvg_origin_ts < eff_window_start
        ):
            eff_window_start = reclaimed_fvg_origin_ts - pd.Timedelta(minutes=1)

        already_trailed_after_tp1 = (
                tp1_hit
                and any(pe.get("label") == "TP1" and pe["time"] < last_trail_ts for pe in pos.partial_exits)
        )

        window_df = ctx.history_df[ctx.history_df.index > eff_window_start]
        if len(window_df) < 2:
            return

        # If we already trailed (post-TP1, or single-TP after a counter-BOS) and price rallied to a new
        # interim peak before pulling back and breaking that peak, slice from that interim peak forward
        # so we trail under the new pullback pivot rather than the prior breakout wick.
        allow_peak_slice = already_trailed_after_tp1 or (
                (not has_planned_partials) and already_structurally_trailed and had_counter_bos
        )
        if allow_peak_slice and cont_event.get("kind") != "IFVG_RECLAIM" and len(window_df) >= 5:
            prior_sub = window_df.iloc[:-1]
            if pos.direction == "LONG":
                peak_ts = prior_sub["high"].idxmax()
                if peak_ts > window_df.index[1] and peak_ts < prior_sub.index[-1]:
                    window_df = ctx.history_df[ctx.history_df.index >= peak_ts]
            else:
                trough_ts = prior_sub["low"].idxmin()
                if trough_ts > window_df.index[1] and trough_ts < prior_sub.index[-1]:
                    window_df = ctx.history_df[ctx.history_df.index >= trough_ts]

        if len(window_df) < 2:
            return

        w_opens = window_df["open"].to_numpy(dtype=float)
        w_highs = window_df["high"].to_numpy(dtype=float)
        w_lows = window_df["low"].to_numpy(dtype=float)
        w_closes = window_df["close"].to_numpy(dtype=float)

        if pos.direction == "LONG":
            pullback_low = float(window_df["low"].min())
            low_matches = window_df.index[window_df["low"] <= (pullback_low + 1e-9)]
            if len(low_matches) >= 2 and (low_matches[-1] - low_matches[0]) <= pd.Timedelta(minutes=2):
                pullback_time = low_matches[-1]
            else:
                pullback_time = window_df["low"].idxmin()

            prior_swings = [
                s for s in ctx.confirmed_swings
                if s["type"] == "SwingLow" and eff_window_start <= s["time"] < pullback_time
            ]
            swept_cluster = sum(1 for s in prior_swings if pullback_low < float(s["price"])) >= 1
            cluster_touches = int((window_df["low"] <= (pullback_low + buffer_pts)).sum())

            had_internal_counter_break = any(
                w_closes[idx - 1] > w_opens[idx - 1] and w_closes[idx] < w_lows[idx - 1]
                for idx in range(1, len(window_df) - 1)
            )
            valid_second_pre_tp1_trail = (
                    already_trailed_once and had_internal_counter_break and cluster_touches >= 2
            )
            tp1_price = float(pos.partial_exits[0]["price"]) if tp1_hit else pos.entry_price
            post_tp1_subsequent_trail = (
                    already_trailed_after_tp1
                    and (had_fvg_inversion or had_internal_counter_break or swept_cluster or cluster_touches >= 2)
            )

            if not (
                    had_counter_bos
                    or had_fvg_inversion
                    or swept_cluster
                    or (tp2_hit_prior and cluster_touches >= 2)
                    or post_tp1_subsequent_trail
                    or valid_second_pre_tp1_trail
            ):
                return

            raw_new_sl = pullback_low if (tp2_hit and cluster_touches >= 2) else (pullback_low - buffer_pts)
            if tp2_hit:
                cand_sl = min(raw_new_sl, tp1_price + 0.15 * risk_pts)
            elif post_tp1_subsequent_trail:
                cand_sl = raw_new_sl
            elif tp1_hit or progress_to_full_tp >= 0.75:
                cand_sl = min(raw_new_sl, pos.entry_price + 0.35 * risk_pts)
            elif valid_second_pre_tp1_trail and raw_new_sl <= (pos.entry_price + 0.15 * risk_pts):
                cand_sl = raw_new_sl
            elif (not has_planned_partials) and already_structurally_trailed and had_counter_bos:
                cand_sl = raw_new_sl
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
                    pos.metadata["last_trailed_pivot_time"] = pullback_time

        else:
            pullback_high = float(window_df["high"].max())
            high_matches = window_df.index[window_df["high"] >= (pullback_high - 1e-9)]
            if len(high_matches) >= 2 and (high_matches[-1] - high_matches[0]) <= pd.Timedelta(minutes=2):
                pullback_time = high_matches[-1]
            else:
                pullback_time = window_df["high"].idxmax()

            prior_swings = [
                s for s in ctx.confirmed_swings
                if s["type"] == "SwingHigh" and pos.entry_time <= s["time"] < pullback_time
            ]
            swept_cluster = sum(1 for s in prior_swings if pullback_high > float(s["price"])) >= 1
            cluster_touches = int((window_df["high"] >= (pullback_high - buffer_pts)).sum())

            had_internal_counter_break = any(
                w_closes[idx - 1] < w_opens[idx - 1] and w_closes[idx] > w_highs[idx - 1]
                for idx in range(1, len(window_df) - 1)
            )
            valid_second_pre_tp1_trail = (
                    already_trailed_once and had_internal_counter_break and cluster_touches >= 2
            )
            tp1_price = float(pos.partial_exits[0]["price"]) if tp1_hit else pos.entry_price
            post_tp1_subsequent_trail = (
                    already_trailed_after_tp1
                    and (had_fvg_inversion or had_internal_counter_break or swept_cluster or cluster_touches >= 2)
            )

            if not (
                    had_counter_bos
                    or had_fvg_inversion
                    or swept_cluster
                    or (tp2_hit_prior and cluster_touches >= 2)
                    or post_tp1_subsequent_trail
                    or valid_second_pre_tp1_trail
            ):
                return

            raw_new_sl = pullback_high if (tp2_hit and cluster_touches >= 2) else (pullback_high + buffer_pts)
            if tp2_hit:
                cand_sl = max(raw_new_sl, tp1_price - 0.15 * risk_pts)
            elif post_tp1_subsequent_trail:
                cand_sl = raw_new_sl
            elif tp1_hit or progress_to_full_tp >= 0.75:
                cand_sl = max(raw_new_sl, pos.entry_price - 0.35 * risk_pts)
            elif valid_second_pre_tp1_trail and raw_new_sl >= (pos.entry_price - 0.15 * risk_pts):
                cand_sl = raw_new_sl
            elif (not has_planned_partials) and already_trailed_once and had_counter_bos:
                cand_sl = raw_new_sl
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
                    pos.metadata["last_trailed_pivot_time"] = pullback_time