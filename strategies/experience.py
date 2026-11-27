# Copyright (C) 2026 Kyrylo Homberh
# SPDX-License-Identifier: AGPL-3.0-only

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from domain.fvg import FairValueGap
from engine.event_bus import BarContext
from engine.portfolio import Portfolio


GRADE_RANK: Dict[str, int] = {
    "B": 1,
    "A": 2,
    "A/A+": 3,
    "A+": 4,
}


@dataclass
class ExperienceConfig:
    """
    Granular toggles for every discretionary trader-experience component.
    - `ExperienceConfig.full()` enables all discretionary rules (Golden Baseline).
    - `ExperienceConfig.pure_strategy()` disables all discretionary overrides/filters
      so Layer 1 (Pure Entry Model) can be backtested in isolation.
    """
    enable_grading: bool = True
    enable_ath_regime: bool = True
    enable_exhaustion_filter: bool = True
    enable_anti_chasing: bool = True
    enable_symbol_sl_heuristics: bool = True
    enable_cross_index_exclusivity: bool = True
    enable_discretionary_targets: bool = True
    enable_partial_lot_quantization: bool = True

    @classmethod
    def full(cls) -> "ExperienceConfig":
        return cls()

    @classmethod
    def pure_strategy(cls) -> "ExperienceConfig":
        return cls(
            enable_grading=False,
            enable_ath_regime=False,
            enable_exhaustion_filter=False,
            enable_anti_chasing=False,
            enable_symbol_sl_heuristics=False,
            enable_cross_index_exclusivity=False,
            enable_discretionary_targets=False,
            enable_partial_lot_quantization=False,
        )


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


class ICTExperienceLayer:
    """
    Houses all discretionary ICT trader judgment:
    - Setup Grading & Quality Filtering
    - ATH + Dual-Index HTF-Bullish Regime Overrides
    - Exhaustion-Leg & Manipulation Anti-Chasing Filters
    - Symbol-Specific Structural SL Heuristics
    - Cross-Index Portfolio Exclusivity
    - Discretionary Multi-Target & Partial Take-Profit Heuristics
    - Whole-Contract Partial Lot Quantization & 3-TP Majority Sizing
    """

    def __init__(self, config: Optional[ExperienceConfig] = None):
        self.config: ExperienceConfig = config if config is not None else ExperienceConfig.full()

    @staticmethod
    def target_priority_rank(target_type: str) -> int:
        t_up = target_type.upper()
        if any(k in t_up for k in ("SESSION_", "WEEKEND_GAP", "4H_POOL", "EQUAL_HL_4H", "1H_POOL", "EQUAL_HL_1H")):
            return 1
        if "EQUAL_HL_15MIN" in t_up or "NY_OPENING_RANGE" in t_up:
            return 2
        return 3

    # ------------------------------------------------------------------
    # 1. Cross-Index Portfolio Exclusivity
    # ------------------------------------------------------------------
    def get_companion_trade_info(
        self,
        shared_daily_trade_dates: Optional[Dict[Any, Any]],
        trade_date: Any,
    ) -> Tuple[Optional[Any], Optional[pd.Timestamp]]:
        if not self.config.enable_cross_index_exclusivity or shared_daily_trade_dates is None:
            return None, None
        companion_info = shared_daily_trade_dates.get(trade_date)
        companion_entry_ts = companion_info[1] if isinstance(companion_info, tuple) else None
        return companion_info, companion_entry_ts

    def format_cross_index_rejection(
        self,
        companion_info: Any,
        companion_entry_ts: Optional[pd.Timestamp],
        date_str: str,
    ) -> str:
        other_sym = companion_info[0] if isinstance(companion_info, tuple) else str(companion_info)
        other_t_str = companion_entry_ts.strftime("%H:%M") if companion_entry_ts is not None else date_str
        return (
            f"Cross-Index Exclusivity: Companion index ({other_sym}) "
            f"already executed the trade of the day at {other_t_str}."
        )

    def record_cross_index_trade(
        self,
        shared_daily_trade_dates: Optional[Dict[Any, Any]],
        trade_date: Any,
        symbol: str,
        timestamp: pd.Timestamp,
    ) -> None:
        if self.config.enable_cross_index_exclusivity and shared_daily_trade_dates is not None:
            shared_daily_trade_dates[trade_date] = (symbol, timestamp)

    @staticmethod
    def compute_leader_score(
            trade: Any,
            df_1m: pd.DataFrame,
            er_bars: int = 15,
            range_bars: int = 20,
    ) -> Tuple[float, str]:
        """
        Scores whether an index is actively pushing in the trade direction (Leader)
        versus chopping sideways (Consolidator), strictly using bars up to entry_time.
        """
        entry_ts = trade.entry_time
        hist = df_1m.loc[:entry_ts]
        if len(hist) < 3:
            return 0.0, "insufficient_bars"

        sign = 1.0 if trade.direction == "LONG" else -1.0

        # 1. Signed Kaufman Efficiency Ratio (directional push vs. chop)
        er_slice = hist["close"].iloc[-(er_bars + 1):].to_numpy(dtype=float)
        if len(er_slice) >= 2:
            net_move = (er_slice[-1] - er_slice[0]) * sign
            path_len = float(abs(er_slice[1:] - er_slice[:-1]).sum())
            dir_er = (net_move / path_len) if path_len > 1e-9 else 0.0
        else:
            dir_er = 0.0

        # 2. Directional progress inside recent 20-bar range [0.0 .. 1.0]
        rng_slice = hist.iloc[-range_bars:]
        r_high = float(rng_slice["high"].max())
        r_low = float(rng_slice["low"].min())
        c_now = float(rng_slice["close"].iloc[-1])
        span = r_high - r_low
        if span > 1e-9:
            range_pos = ((c_now - r_low) / span) if trade.direction == "LONG" else ((r_high - c_now) / span)
        else:
            range_pos = 0.5

        # 3. Active intraday SMT displacement confirmer (which index actually broke 1m MSS)
        meta = trade.metadata or {}
        sym = trade.symbol.upper()
        dir_prefix = "Bullish SMT" if trade.direction == "LONG" else "Bearish SMT"
        conf_markers = (f"--> 1m MSS on {sym} ", f"--> 1m IFVG_MSS on {sym} ")
        is_smt_confirmer = any(
            (dir_prefix in line) and any(m in line for m in conf_markers)
            for line in meta.get("smt_detail", [])
        )
        smt_bonus = 0.25 if is_smt_confirmer else 0.0

        # Tiny tie-breaker from setup score (0..10)
        setup_score_bonus = 0.01 * float(meta.get("score", 0))

        total = (0.50 * dir_er) + (0.35 * range_pos) + smt_bonus + setup_score_bonus
        detail = f"leader={total:.2f} (dir_ER={dir_er:+.2f}, rng_pos={range_pos:.2f}, smt_confirmer={is_smt_confirmer})"
        return total, detail

    @classmethod
    def resolve_cross_index_winners(
            cls,
            res_es: Any,
            res_nq: Any,
            df_es_1m: pd.DataFrame,
            df_nq_1m: pd.DataFrame,
            tie_window_minutes: int = 0,
            verbose: bool = False,
    ) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
        """
        Resolves daily cross-index exclusivity with zero lookahead bias:
        - Strictly earlier valid entry wins in real time (`tie_window_minutes=0`).
        - When both indices fire on the exact same 1-minute bar, the actively
          pushing Leader index beats the Consolidator index.
        """
        es_by_date = {t.entry_time.date(): t for t in res_es.trades}
        nq_by_date = {t.entry_time.date(): t for t in res_nq.trades}

        blocked_for_es: Dict[Any, Any] = {}
        blocked_for_nq: Dict[Any, Any] = {}

        for d in sorted(set(es_by_date.keys()) | set(nq_by_date.keys())):
            t_es = es_by_date.get(d)
            t_nq = nq_by_date.get(d)

            if t_es is not None and t_nq is None:
                blocked_for_nq[d] = ("ES", t_es.entry_time)
                continue
            if t_nq is not None and t_es is None:
                blocked_for_es[d] = ("NQ", t_nq.entry_time)
                continue

            delta_min = (t_nq.entry_time - t_es.entry_time).total_seconds() / 60.0
            es_score, es_det = cls.compute_leader_score(t_es, df_es_1m)
            nq_score, nq_det = cls.compute_leader_score(t_nq, df_nq_1m)

            if abs(delta_min) > tie_window_minutes:
                winner = "ES" if t_es.entry_time < t_nq.entry_time else "NQ"
                reason = f"earlier ({abs(delta_min):.0f}m lead)"
            else:
                winner = "NQ" if nq_score > es_score else "ES"
                reason = "same-bar leader_score"

            if verbose:
                print(
                    f"   ⚖️  [Cross-Index {d}] ES {t_es.direction}@{t_es.entry_time.strftime('%H:%M')} [{es_det}] "
                    f"vs NQ {t_nq.direction}@{t_nq.entry_time.strftime('%H:%M')} [{nq_det}] "
                    f"--> WINNER: {winner} ({reason})"
                )

            if winner == "ES":
                blocked_for_nq[d] = ("ES", t_es.entry_time)
            else:
                blocked_for_es[d] = ("NQ", t_nq.entry_time)

        return blocked_for_es, blocked_for_nq

    # ------------------------------------------------------------------
    # 2. Market Regime Overrides (ATH Regime)
    # ------------------------------------------------------------------
    def detect_ath_bullish_regime(
        self,
        ctx: BarContext,
        price: float,
        t_4h: str,
        t_1h: str,
        t_15m: str,
        sec_4h: str,
        sec_1h: str,
        sec_15m: str,
    ) -> bool:
        if not self.config.enable_ath_regime:
            return False

        ny_open_ts = ctx.timestamp.replace(hour=9, minute=30, second=0, microsecond=0)
        overhead_htf_bsl_pools = [
            tp for tp in ctx.target_pools
            if tp["pool_type"] == "BSL"
               and float(tp["price"]) > price
               and any(
                k in str(tp["target_type"]).upper() for k in ("4H_POOL", "EQUAL_HL_4H", "1H_POOL", "EQUAL_HL_1H"))
               and tp.get("creation_time", ny_open_ts) < ny_open_ts
        ]
        has_recent_overhead_htf_bsl = any(
            (ctx.timestamp - tp["creation_time"]) <= pd.Timedelta(days=30)
            for tp in overhead_htf_bsl_pools
            if "creation_time" in tp
        )
        has_close_overhead_htf_bsl = any(
            (float(tp["price"]) - price) <= (15.0 if ctx.symbol == "ES" else 120.0)
            for tp in overhead_htf_bsl_pools
        )
        is_true_ath_overhead = (not has_recent_overhead_htf_bsl) and (not has_close_overhead_htf_bsl)

        return (
            ctx.htf_trends.get("ath_regime") == "ACTIVE"
            and is_true_ath_overhead
            and t_4h == "BULLISH"
            and t_1h == "BULLISH"
            and t_15m == "BULLISH"
            and sec_4h == "BULLISH"
            and sec_1h == "BULLISH"
            and sec_15m == "BULLISH"
        )

    def check_ath_short_block(
        self,
        direction: str,
        is_ath_bullish_regime: bool,
    ) -> Optional[str]:
        if self.config.enable_ath_regime and direction == "SHORT" and is_ath_bullish_regime:
            return (
                "ATH + Dual-Index HTF-Bullish Regime active (ES & NQ 4H/1H/15m BULLISH): "
                "Counter-trend Bearish SMT shorts are forbidden at ATH."
            )
        return None

    def evaluate_ath_swept_continuation(
        self,
        ctx: BarContext,
        direction: str,
        is_ath_bullish_regime: bool,
        has_intraday_post_open_smt: bool,
        counter_events: List[Dict[str, Any]],
        inverted_1m_fvg_desc: str,
        trigger_desc: str,
    ) -> Optional[Tuple[bool, str, str, int, List[FairValueGap], str, str]]:
        if not (
            self.config.enable_ath_regime
            and direction == "LONG"
            and is_ath_bullish_regime
            and ctx.liquidity_taken_level in ("MEDIUM", "HIGH")
        ):
            return None

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

    def evaluate_ath_fallback_continuation(
        self,
        direction: str,
        is_ath_bullish_regime: bool,
        has_intraday_post_open_smt: bool,
        counter_events: List[Dict[str, Any]],
        inverted_1m_fvg_desc: str,
        trigger_desc: str,
    ) -> Optional[Tuple[bool, str, str, int, List[FairValueGap], str, str]]:
        if not (
            self.config.enable_ath_regime
            and direction == "LONG"
            and is_ath_bullish_regime
            and has_intraday_post_open_smt
            and (counter_events or inverted_1m_fvg_desc)
        ):
            return None

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

    # ------------------------------------------------------------------
    # 3. Exhaustion & Anti-Chasing Filters
    # ------------------------------------------------------------------
    def check_smt_anti_chasing(
        self,
        ctx: BarContext,
        direction: str,
        confirm_event: Dict[str, Any],
        stop_loss_points: float,
    ) -> Tuple[bool, str]:
        if not self.config.enable_anti_chasing:
            return True, ""

        recent_6 = ctx.history_df.iloc[-6:]
        if direction == "LONG":
            dist_from_manipulation = ctx.close - float(recent_6["low"].min())
        else:
            dist_from_manipulation = float(recent_6["high"].max()) - ctx.close

        # Immediate 1m IFVG_BOS on NQ (e.g., Oct 26 09:40) is an impulse reversal entry, not a late chase;
        # ES keeps the 1.5x cap so ES rejects 09:40 at 12.50 pts and hands the trade to NQ.
        is_immediate_nq_ifvg = (ctx.symbol == "NQ" and confirm_event.get("kind") == "IFVG_BOS")
        if is_immediate_nq_ifvg or dist_from_manipulation <= (stop_loss_points * 1.5):
            return True, ""

        return (
            False,
            f"Intraday SMT present, but price already moved {dist_from_manipulation:.2f} pts from manipulation extreme (> {stop_loss_points * 1.5:.2f} pts).",
        )

    def check_exhaustion_leg_filter(
        self,
        ctx: BarContext,
        direction: str,
        price: float,
        dir_exhaustion: str,
        is_synthetic_short_df: bool,
        t_hm: str,
        trigger_desc: str,
        retrace_type: str,
    ) -> Optional[str]:
        if not self.config.enable_exhaustion_filter:
            return None

        if dir_exhaustion == "HIGH" and not is_synthetic_short_df:
            ny_open_ts = ctx.timestamp.replace(hour=9, minute=30, second=0, microsecond=0)
            rth_df = ctx.history_df[ctx.history_df.index >= ny_open_ts]
            rth_high = float(rth_df["high"].max()) if not rth_df.empty else price
            rth_low = float(rth_df["low"].min()) if not rth_df.empty else price
            rth_range = rth_high - rth_low
            max_exhaustion_leg_pts = 18.0 if ctx.symbol == "ES" else 150.0

            if direction == "SHORT":
                dist_below_high = rth_high - price
                if rth_range >= max_exhaustion_leg_pts and dist_below_high >= (max_exhaustion_leg_pts * 0.75):
                    return (
                        f"[{t_hm}] SHORT ({trigger_desc}, {retrace_type}) -> Rejected: HIGH liquidity exhaustion "
                        f"+ large downside expansion leg already completed ({rth_range:.2f} pts drop from RTH high {rth_high:.2f}; "
                        f"currently {dist_below_high:.2f} pts below high)."
                    )
            else:
                dist_above_low = price - rth_low
                if rth_range >= max_exhaustion_leg_pts and dist_above_low >= (max_exhaustion_leg_pts * 0.75):
                    return (
                        f"[{t_hm}] LONG ({trigger_desc}, {retrace_type}) -> Rejected: HIGH liquidity exhaustion "
                        f"+ large upside expansion leg already completed ({rth_range:.2f} pts rally from RTH low {rth_low:.2f}; "
                        f"currently {dist_above_low:.2f} pts above low)."
                    )
        return None

    def check_counter_trend_distant_target_filter(
            self,
            direction: str,
            retrace_type: str,
            t_15m: str,
            htf_aligned_count: int,
            target_info: Dict[str, Any],
    ) -> Optional[str]:
        """
        Rejects pure 1m counter-5m SMT reversals (`SMT_REVERSAL_MODEL`) when 15m is
        also against the trade, HTF is not aligned (< 2 aligned), FullTP is >= 3.0R,
        and even the FIRST available target (TP1 or FullTP) is >= 2.0R away.
        """
        if not (self.config.enable_discretionary_targets and self.config.enable_exhaustion_filter):
            return None

        wanted_bias = "BULLISH" if direction == "LONG" else "BEARISH"
        is_15m_opposite = t_15m not in (wanted_bias, "NEUTRAL")
        planned_rr = float(target_info.get("rr", 0.0))
        partials = target_info.get("partial_targets", [])
        first_target_rr = float(partials[0]["rr"]) if partials else planned_rr

        if (
                retrace_type == "SMT_REVERSAL_MODEL"
                and is_15m_opposite
                and htf_aligned_count < 2
                and planned_rr >= 3.00
                and first_target_rr >= 2.00
        ):
            return (
                f"Counter-trend 1m SMT reversal against {t_15m} 15m/5m structure with mixed/opposing HTF "
                f"({htf_aligned_count}/3 aligned) and no nearby partial (< 2.00R; first target at "
                f"{first_target_rr:.2f}R, FullTP at {planned_rr:.2f}R -> {target_info['target_type']})."
            )
        return None

    def check_laggard_index_smt_filter(
            self,
            direction: str,
            retrace_type: str,
            trend_5m: str,
            sec_trend_5m: str,
    ) -> Optional[str]:
        """
        Zero-lookahead leader-vs-laggard gate:
        Rejects a counter-5m `SMT_REVERSAL_MODEL` entry on the lagging index
        when the companion index's 5m structure is already aligned in the trade direction.
        """
        if not self.config.enable_cross_index_exclusivity:
            return None

        wanted_bias = "BULLISH" if direction == "LONG" else "BEARISH"
        if (
                retrace_type == "SMT_REVERSAL_MODEL"
                and trend_5m != wanted_bias
                and sec_trend_5m == wanted_bias
        ):
            return (
                f"Laggard index block: primary 5m is still {trend_5m} while companion index "
                f"5m is already {sec_trend_5m} (defer counter-5m SMT entry to leading index)."
            )
        return None

    # ------------------------------------------------------------------
    # 4. Symbol-Specific Structural SL Heuristics
    # ------------------------------------------------------------------
    def get_stop_loss_parameters(
        self,
        symbol: str,
        direction: str,
        retrace_type: str,
        retrace_pts: int,
        stop_loss_points: float,
        buffer_pts: float,
    ) -> Dict[str, Any]:
        if self.config.enable_symbol_sl_heuristics:
            # Allow structural Stop-Losses up to 102.0 pts on NQ whenever backed by a 3-point
            # 5m FVG/EQ or SMT entry model (regardless of whether liquidity exhaustion is LOW, MEDIUM, or HIGH).
            if symbol == "NQ" and retrace_pts >= 3:
                max_valid_sl: Optional[float] = max(stop_loss_points * 1.6, 102.0)
            else:
                max_valid_sl = stop_loss_points * 1.6

            use_tight_smt_peak = (retrace_type == "SMT_REVERSAL_MODEL" and symbol == "ES")
            lookback_bars = 5 if use_tight_smt_peak else 8
            swing_age_min = 5 if use_tight_smt_peak else 12
        else:
            max_valid_sl = None
            lookback_bars = 8
            swing_age_min = 12

        use_exact_cluster_low = (
            self.config.enable_ath_regime
            and symbol == "ES"
            and direction == "LONG"
            and retrace_type == "ATH_SMT_CONTINUATION"
        )
        eff_buf = 0.0 if use_exact_cluster_low else buffer_pts

        return {
            "max_valid_sl": max_valid_sl,
            "lookback_bars": lookback_bars,
            "swing_age_min": swing_age_min,
            "eff_buf": eff_buf,
        }

    # ------------------------------------------------------------------
    # 5. Discretionary Target & Partial Take-Profit Heuristics
    # ------------------------------------------------------------------
    def select_discretionary_target(
            self,
            ctx: BarContext,
            direction: str,
            fill_price: float,
            actual_risk_pts: float,
            all_directional_targets: List[Dict[str, Any]],
            min_rr: float,
            min_tp1_rr: float,
            max_rr: float,
            partial_cut_pct: float,
            enable_partials: bool,
            reward_risk_ratio: float,
            allow_opening_range_target: bool = False,
            allow_exception_tp1: bool = False,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
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

        eff_tp1_min_rr = 0.55 if allow_exception_tp1 else min_tp1_rr
        full_tp_min_rr = 1.60 if allow_exception_tp1 else (
            max(1.65, min_rr) if enable_partials else min_rr
        )

        prio1_targets = [c for c in all_directional_targets if c["priority"] == 1 and c["rr"] >= 0.25]
        if prio1_targets:
            prio1_targets.sort(key=lambda x: x["rr"])
            closest_prio1 = prio1_targets[0]

            if closest_prio1["rr"] < (eff_tp1_min_rr if enable_partials else min_rr):
                return (
                    None,
                    f"Closest Priority-1 target {closest_prio1['target_type']} only offered "
                    f"{closest_prio1['rr']:.2f}R (< min TP1 {eff_tp1_min_rr:.2f}R); skipping past a major target is forbidden.",
                )

            # MULTI-TP PATH A: Closest Priority-1 pool is in [eff_tp1_min_rr, full_tp_min_rr) -> use as TP1 (30%) and look for FullTP >= 1.60R
            if enable_partials and closest_prio1["rr"] < full_tp_min_rr:
                tp1_cand = closest_prio1
                further_cands = [
                    c for c in all_directional_targets
                    if c["priority"] <= 2
                       and (tp1_cand["rr"] + 0.35) <= c["rr"] <= max_rr
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
                if not mid_cands and tp1_cand["rr"] >= min_tp1_rr and full_tp["rr"] <= 2.00:
                    higher_prio1 = [
                        c for c in full_tp_cands
                        if c["priority"] == 1 and (full_tp["rr"] + 0.50) <= c["rr"] <= max_rr
                    ]
                    if higher_prio1:
                        mid_cands = [full_tp]
                        full_tp = dict(higher_prio1[0])

                # If there is a wide gap (>= 1.80R) between TP1 and FullTP (e.g. Oct 8 ES Long: TP1 @ 1.23R -> FullTP @ 3.62R),
                # look for an intermediate 15m liquidity pool (such as 7845.00 @ 2.46R) to serve as TP2.
                if not mid_cands and (full_tp["rr"] - tp1_cand["rr"]) >= 1.80:
                    mid_target_rr = 0.5 * (tp1_cand["rr"] + full_tp["rr"])
                    mid_15m_cands = [
                        c for c in all_directional_targets
                        if "15MIN" in c["target_type"].upper()
                           and (tp1_cand["rr"] + 0.80) <= c["rr"] <= (full_tp["rr"] - 0.60)
                    ]
                    mid_15m_cands.sort(key=lambda x: abs(x["rr"] - mid_target_rr))
                    if mid_15m_cands:
                        mid_cands = [mid_15m_cands[0]]

                # If TP1 sacrificed below 0.80R (e.g. 0.57R on Oct 1 NQ Short),
                # strictly require BOTH a valid TP2 (>= 0.95R) and FullTP (>= 1.60R)!
                if tp1_cand["rr"] < min_tp1_rr and not mid_cands:
                    return (
                        None,
                        f"TP1 {tp1_cand['target_type']} is < {min_tp1_rr:.2f}R ({tp1_cand['rr']:.2f}R) "
                        f"and no intermediate Priority-1/2 TP2 exists between TP1 and FullTP.",
                    )

                partial_list: List[Dict[str, Any]] = [
                    {
                        "label": "TP1",
                        "price": float(tp1_cand["price"]),
                        "rr": round(float(tp1_cand["rr"]), 2),
                        "weight": partial_cut_pct,
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
                        "weight": partial_cut_pct,
                        "target_type": tp2_cand["target_type"],
                        "tier": tp2_cand["tier"],
                        "hit": False,
                    })

                if len(partial_list) >= 2 and all(
                        self.target_priority_rank(pt["target_type"]) == 1 for pt in partial_list
                ):
                    full_tp["priority"] = 1
                    full_tp["tier"] = 1

                full_tp["partial_targets"] = partial_list
                return full_tp, ""

            # MULTI-TP PATH B (HOMERUN TRADE >= 3.0R TO CLOSEST PRIORITY-1 POOL, e.g., Oct 6 ES Long to 7892.00 @ 3.23R):
            if enable_partials and 3.00 <= closest_prio1["rr"] <= max_rr:
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
                        "weight": partial_cut_pct,
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
                            "weight": partial_cut_pct,
                            "target_type": tp2_cand["target_type"],
                            "tier": tp2_cand["tier"],
                            "hit": False,
                        })

                full_tp["partial_targets"] = partial_list
                return full_tp, ""

        # SINGLE-TP PATH: Closest major target already offers >= 1.60R (e.g., Oct 13 ES @ 2.32R, Oct 14 NQ @ 2.04R, Oct 21 ES @ 1.67R, Oct 26 NQ @ 2.35R)
        valid_candidates = [
            c for c in all_directional_targets
            if c["priority"] <= 2 and full_tp_min_rr <= c["rr"] <= max_rr
        ]
        if valid_candidates:
            valid_candidates.sort(key=lambda x: (x["priority"], x["rr"]))
            best = dict(valid_candidates[0])
            best["partial_targets"] = []

            # When no intermediate Priority-1 pool exists before FullTP (e.g. Oct 26 NQ Short @ 09:40 to 29072.75 @ 2.35R,
            # or Oct 21 ES Short @ 09:42 to 7757.25 @ 1.67R), check if an unbroken NY Opening Range Low/High sits ahead
            # with >= 0.80R room to FullTP. Use body closure (min/max of open, close) when >= 0.70R, or wick if body < 0.70R.
            if enable_partials and len(ctx.history_df) >= 25:
                ny_open_ts = ctx.timestamp.replace(hour=9, minute=30, second=0, microsecond=0)
                or_cutoff_ts = ctx.timestamp - pd.Timedelta(minutes=5)
                or_df = ctx.history_df[
                    (ctx.history_df.index >= ny_open_ts) & (ctx.history_df.index <= or_cutoff_ts)
                    ]
                post_open_df = ctx.history_df[
                    (ctx.history_df.index >= ny_open_ts) & (ctx.history_df.index < ctx.timestamp)
                    ]
                if not or_df.empty:
                    if direction == "SHORT":
                        or_ts = or_df["low"].idxmin()
                        wick_p = float(or_df["low"].min())
                        bar = ctx.history_df.loc[or_ts] if or_ts in ctx.history_df.index else None
                        body_p = float(min(bar["open"], bar["close"])) if bar is not None else wick_p
                        after_or_df = post_open_df[post_open_df.index > or_ts]
                        unbroken_or = after_or_df.empty or float(after_or_df["low"].min()) >= wick_p

                        body_rr = (fill_price - body_p) / actual_risk_pts if actual_risk_pts > 0 else 0.0
                        wick_rr = (fill_price - wick_p) / actual_risk_pts if actual_risk_pts > 0 else 0.0
                        or_p, or_rr = (body_p, body_rr) if body_rr >= 0.70 else (wick_p, wick_rr)

                        if unbroken_or and 0.70 <= or_rr <= 1.25 and (best["rr"] - or_rr) >= 0.80:
                            best["partial_targets"] = [{
                                "label": "TP1",
                                "price": or_p,
                                "rr": round(float(or_rr), 2),
                                "weight": partial_cut_pct,
                                "target_type": f"NY_OPENING_RANGE_LOW (SSL @ {or_p:.2f}, formed {or_ts.strftime('%m-%d %H:%M')})",
                                "tier": 2,
                                "hit": False,
                            }]
                    else:
                        or_ts = or_df["high"].idxmax()
                        wick_p = float(or_df["high"].max())
                        bar = ctx.history_df.loc[or_ts] if or_ts in ctx.history_df.index else None
                        body_p = float(max(bar["open"], bar["close"])) if bar is not None else wick_p
                        after_or_df = post_open_df[post_open_df.index > or_ts]
                        unbroken_or = after_or_df.empty or float(after_or_df["high"].max()) <= wick_p

                        body_rr = (body_p - fill_price) / actual_risk_pts if actual_risk_pts > 0 else 0.0
                        wick_rr = (wick_p - fill_price) / actual_risk_pts if actual_risk_pts > 0 else 0.0
                        or_p, or_rr = (body_p, body_rr) if body_rr >= 0.70 else (wick_p, wick_rr)

                        if unbroken_or and 0.70 <= or_rr <= 1.25 and (best["rr"] - or_rr) >= 0.80:
                            best["partial_targets"] = [{
                                "label": "TP1",
                                "price": or_p,
                                "rr": round(float(or_rr), 2),
                                "weight": partial_cut_pct,
                                "target_type": f"NY_OPENING_RANGE_HIGH (BSL @ {or_p:.2f}, formed {or_ts.strftime('%m-%d %H:%M')})",
                                "tier": 2,
                                "hit": False,
                            }]

            return best, ""

        # Fallback: Tier-2 lower-TF targets (regular 15MIN_POOL, priority == 3).
        # Passed to grade_setup() strictly as Grade B so Grade A/A+ logs the full
        # Grade B rejection (15m target + exhaustion/HTF cons) while Allow Grade B tracks them.
        if self.config.enable_grading:
            tier2_candidates = [
                c for c in all_directional_targets
                if c["priority"] == 3 and full_tp_min_rr <= c["rr"] <= max_rr
            ]
            if tier2_candidates:
                tier2_candidates.sort(key=lambda x: x["rr"])
                best = dict(tier2_candidates[0])
                best["partial_targets"] = []
                return best, ""

        if len(ctx.history_df) < 25:
            default_tp = (
                fill_price + (actual_risk_pts * reward_risk_ratio)
                if direction == "LONG"
                else fill_price - (actual_risk_pts * reward_risk_ratio)
            )
            return {
                "price": float(default_tp),
                "rr": float(reward_risk_ratio),
                "target_type": f"SYNTHETIC_DEFAULT_RR @ {default_tp:.2f}",
                "tier": 1,
                "priority": 1,
                "partial_targets": [],
            }, ""

        if all_directional_targets:
            best_rr = min(all_directional_targets, key=lambda x: abs(x["rr"] - 2.0))
            return (
                None,
                f"No unbroken target inside [{full_tp_min_rr:.2f}R, {max_rr:.2f}R] "
                f"(closest candidate was {best_rr['target_type']} at {best_rr['rr']:.2f}R).",
            )
        return None, "No unbroken Session/HTF target pools remaining ahead in trade direction."

    # ------------------------------------------------------------------
    # 6. Whole-Contract Sizing & Partial Lot Quantization
    # ------------------------------------------------------------------
    def calculate_contracts(
        self,
        portfolio: Portfolio,
        symbol: str,
        actual_risk_pts: float,
        use_dynamic_sizing: bool,
        quantity: int,
        risk_per_trade_pct: float,
        partial_cut_pct: float,
        num_partials: int = 0,
    ) -> Tuple[float, str, int, str]:
        """
        Sizes positions to match TradingView whole-contract execution:
        1. Tries whole standard contracts (ES / NQ) within ($800, $1,100] as long as there are enough
           whole contracts to split across all partial targets (`chosen_std > num_partials`).
        2. Otherwise falls back to Micro contracts (MES / MNQ).
        3. On 3-target setups (`num_partials == 2`, 30%/30%/40%), if `chosen_micro` would force an equal
           lot split where FullTP has no majority over TP1 (e.g., 6 MNQ -> 2 / 2 / 2), bumps by +1 Micro
           (to 7 MNQ -> 2 / 2 / 3) if risk stays <= $1,200.
        """
        clean_sym = symbol.upper()
        if not use_dynamic_sizing or actual_risk_pts <= 0:
            lots = max(1, int(quantity))
            return float(lots), f"{lots} {clean_sym}", lots, clean_sym

        target_risk = 100_000.0 * risk_per_trade_pct  # $1,000.00
        min_acceptable_risk = target_risk * 0.80  # $800.00
        max_acceptable_risk = target_risk * 1.10  # $1,100.00

        std_point_val = portfolio.get_point_value(clean_sym)
        std_unit_risk = actual_risk_pts * std_point_val
        if std_unit_risk <= 0:
            lots = max(1, int(quantity))
            return float(lots), f"{lots} {clean_sym}", lots, clean_sym

        low_std = int(target_risk // std_unit_risk)
        high_std = low_std + 1

        low_std_risk = low_std * std_unit_risk
        high_std_risk = high_std * std_unit_risk

        chosen_std: Optional[int] = None
        if low_std >= 1 and low_std_risk > min_acceptable_risk:
            if high_std_risk <= max_acceptable_risk and abs(high_std_risk - target_risk) < abs(
                low_std_risk - target_risk
            ):
                chosen_std = high_std
            else:
                chosen_std = low_std
        elif high_std >= 1 and high_std_risk <= max_acceptable_risk and high_std_risk > min_acceptable_risk:
            chosen_std = high_std

        if self.config.enable_partial_lot_quantization:
            # Ensure standard contract count can support whole-contract partial exits on TradingView
            if chosen_std is not None and chosen_std > num_partials:
                if num_partials == 2:
                    tp1_l = max(1, int(round(chosen_std * partial_cut_pct)))
                    tp2_l = max(1, int(round(chosen_std * partial_cut_pct)))
                    full_l = chosen_std - tp1_l - tp2_l
                    if full_l >= tp1_l:
                        return float(chosen_std), f"{chosen_std} {clean_sym}", chosen_std, clean_sym
                else:
                    return float(chosen_std), f"{chosen_std} {clean_sym}", chosen_std, clean_sym
        else:
            if chosen_std is not None:
                return float(chosen_std), f"{chosen_std} {clean_sym}", chosen_std, clean_sym

        # Fallback to Micro contracts (MES = 0.1 ES, MNQ = 0.1 NQ)
        micro_sym = f"M{clean_sym}" if not clean_sym.startswith("M") else clean_sym
        micro_unit_risk = std_unit_risk * 0.10

        low_micro = int(target_risk // micro_unit_risk)
        high_micro = low_micro + 1

        low_micro_risk = low_micro * micro_unit_risk
        high_micro_risk = high_micro * micro_unit_risk

        if low_micro >= 1 and low_micro_risk > min_acceptable_risk:
            if high_micro_risk <= max_acceptable_risk and abs(high_micro_risk - target_risk) < abs(
                low_micro_risk - target_risk
            ):
                chosen_micro = high_micro
            else:
                chosen_micro = low_micro
        elif high_micro_risk <= max_acceptable_risk:
            chosen_micro = max(1, high_micro)
        else:
            chosen_micro = max(1, low_micro)

        # TradingView 3-target whole-contract rule: avoid 2/2/2 equal split on 6 MNQ when 7 MNQ (2/2/3) keeps FullTP majority
        if self.config.enable_partial_lot_quantization and num_partials == 2 and chosen_micro >= 3:
            tp1_m = max(1, int(round(chosen_micro * partial_cut_pct)))
            tp2_m = max(1, int(round(chosen_micro * partial_cut_pct)))
            full_m = chosen_micro - tp1_m - tp2_m
            if full_m <= tp1_m and (chosen_micro + 1) * micro_unit_risk <= target_risk * 1.20:
                chosen_micro += 1

        std_equiv_qty = round(chosen_micro * 0.10, 2)
        return std_equiv_qty, f"{chosen_micro} {micro_sym}", chosen_micro, micro_sym

    def quantize_partial_lots(
        self,
        partial_targets_list: List[Dict[str, Any]],
        total_int_lots: int,
        unit_sym: str,
        partial_cut_pct: float,
    ) -> None:
        if not self.config.enable_partial_lot_quantization:
            return

        # Quantize partial exits to exact whole TradingView contracts (e.g., 7 MNQ -> 2 / 2 / 3 MNQ)
        if len(partial_targets_list) == 1 and total_int_lots >= 2:
            tp1_lots = max(1, min(total_int_lots - 1, int(round(total_int_lots * partial_cut_pct))))
            partial_targets_list[0]["lots"] = tp1_lots
            partial_targets_list[0]["unit_sym"] = unit_sym
            partial_targets_list[0]["weight"] = tp1_lots / float(total_int_lots)
        elif len(partial_targets_list) == 2 and total_int_lots >= 3:
            tp1_lots = max(1, int(round(total_int_lots * partial_cut_pct)))
            tp2_lots = max(1, int(round(total_int_lots * partial_cut_pct)))
            if tp1_lots + tp2_lots >= total_int_lots:
                tp1_lots = 1
                tp2_lots = 1
            partial_targets_list[0]["lots"] = tp1_lots
            partial_targets_list[0]["unit_sym"] = unit_sym
            partial_targets_list[0]["weight"] = tp1_lots / float(total_int_lots)
            partial_targets_list[1]["lots"] = tp2_lots
            partial_targets_list[1]["unit_sym"] = unit_sym
            partial_targets_list[1]["weight"] = tp2_lots / float(total_int_lots)

    # ------------------------------------------------------------------
    # 7. Setup Grading & Quality Gate
    # ------------------------------------------------------------------
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
            pros.append(
                f"HTF Structure Aligned via Confirmed 1H/4H SMT & Companion HTF MSS (4H={t_4h}, 1H={t_1h}, 15m={t_15m})")
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

        if prio >= 3:
            grade = "B"
        elif liquidity_taken_level == "HIGH" and not has_smt:
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

    def passes_grade_filter(self, grade: str, min_grade: str) -> bool:
        if not self.config.enable_grading:
            return True
        return GRADE_RANK.get(grade, 0) >= GRADE_RANK.get(min_grade, 2)