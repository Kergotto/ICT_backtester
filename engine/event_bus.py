from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, Tuple
import bisect
import numpy as np
import pandas as pd

from core.data_loader import resample_timeframe
from domain.swings import detect_swings
from domain.structure import detect_structure
from domain.liquidity import track_liquidity, LiquidityPool, GapEngine, WeekendGap
from domain.fvg import track_fvgs, FairValueGap
from domain.equilibrium import track_dealing_ranges, DealingRange
from domain.fibonacci import track_fibonacci_ranges, FibonacciRange
from domain.smt import detect_smt_divergence, is_smt_active_at
from engine.session_engine import extract_session_liquidity
from engine.portfolio import Portfolio, TradeRecord
from engine.broker import SimulatedBroker


@dataclass
class Event:
    event_type: str
    timestamp: pd.Timestamp
    payload: Dict[str, Any] = field(default_factory=dict)


class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Event], None]]] = {}
        self.event_log: List[Event] = []

    def subscribe(self, event_type: str, handler: Callable[[Event], None]) -> None:
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    def publish(self, event: Event) -> None:
        self.event_log.append(event)
        for handler in self._subscribers.get(event.event_type, []):
            handler(event)


@dataclass
class BarContext:
    """
    Zero-Lookahead multi-timeframe market snapshot at closed 1-minute candle `timestamp`.
    """
    symbol: str
    timestamp: pd.Timestamp
    bar_index: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    history_df: pd.DataFrame
    # 1-Minute Primitives
    confirmed_swings: List[Dict[str, Any]] = field(default_factory=list)
    active_fvgs: List[FairValueGap] = field(default_factory=list)
    recent_sweeps: List[LiquidityPool] = field(default_factory=list)
    recent_sweep_details: List[Dict[str, Any]] = field(default_factory=list)
    active_dealing_range: Optional[DealingRange] = None
    active_fib_range: Optional[FibonacciRange] = None
    recent_smt_events: List[Dict[str, Any]] = field(default_factory=list)
    structure_events_1m: List[Dict[str, Any]] = field(default_factory=list)
    # 5-Minute & Higher-Timeframe Primitives
    confirmed_swings_5m: List[Dict[str, Any]] = field(default_factory=list)
    structure_events_5m: List[Dict[str, Any]] = field(default_factory=list)
    active_fvgs_5m: List[FairValueGap] = field(default_factory=list)
    active_dealing_range_5m: Optional[DealingRange] = None
    active_fib_range_5m: Optional[FibonacciRange] = None
    htf_trends: Dict[str, str] = field(default_factory=dict)
    # Targets & Pre-Entry Liquidity Exhaustion
    target_pools: List[Dict[str, Any]] = field(default_factory=list)
    active_weekend_gaps: List[WeekendGap] = field(default_factory=list)
    liquidity_taken_level: str = "LOW"
    session_taken_pools: List[str] = field(default_factory=list)


@dataclass
class BacktestResult:
    strategy_name: str
    symbol: str
    initial_capital: float
    final_equity: float
    net_profit: float
    total_return_pct: float
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown_pct: float
    max_drawdown_cash: float
    avg_r_multiple: float
    trades: List[TradeRecord]
    equity_df: pd.DataFrame


def calculate_metrics(portfolio: Portfolio, strategy_name: str, symbol: str) -> BacktestResult:
    trades = portfolio.closed_trades
    total_trades = len(trades)

    if portfolio.equity_curve:
        equity_df = pd.DataFrame(portfolio.equity_curve).set_index("timestamp")
        final_equity = float(equity_df["equity"].iloc[-1])
    else:
        equity_df = pd.DataFrame(columns=["cash", "unrealized_pnl", "equity"])
        final_equity = float(portfolio.cash)

    net_profit = final_equity - portfolio.initial_capital
    total_return_pct = (net_profit / portfolio.initial_capital) * 100.0 if portfolio.initial_capital > 0 else 0.0

    if total_trades > 0:
        wins = [t.net_pnl for t in trades if t.net_pnl > 0]
        losses = [abs(t.net_pnl) for t in trades if t.net_pnl <= 0]
        win_rate = (len(wins) / total_trades) * 100.0
        gross_win = sum(wins)
        gross_loss = sum(losses)
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
        avg_r = float(np.mean([t.r_multiple for t in trades]))
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_r = 0.0

    if len(equity_df) > 1:
        eq_series = equity_df["equity"].astype(float)
        running_max = eq_series.cummax()
        dd_cash = running_max - eq_series
        dd_pct = (dd_cash / running_max.replace(0, np.nan)) * 100.0

        max_drawdown_cash = float(dd_cash.max())
        max_drawdown_pct = float(dd_pct.max()) if not np.isnan(dd_pct.max()) else 0.0

        returns = eq_series.pct_change().dropna()
        std_ret = float(returns.std())
        if std_ret > 1e-12:
            sharpe_ratio = float((returns.mean() / std_ret) * np.sqrt(252 * 390))
        else:
            sharpe_ratio = 0.0
    else:
        max_drawdown_cash = 0.0
        max_drawdown_pct = 0.0
        sharpe_ratio = 0.0

    return BacktestResult(
        strategy_name=strategy_name,
        symbol=symbol,
        initial_capital=portfolio.initial_capital,
        final_equity=final_equity,
        net_profit=net_profit,
        total_return_pct=total_return_pct,
        total_trades=total_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        sharpe_ratio=sharpe_ratio,
        max_drawdown_pct=max_drawdown_pct,
        max_drawdown_cash=max_drawdown_cash,
        avg_r_multiple=avg_r,
        trades=trades,
        equity_df=equity_df,
    )


class BacktestEngine:
    """
    Step-by-step chronological simulation engine with strict Zero Look-Ahead enforcement
    across 1m, 5m, 15m, 1h, and 4h timeframes.
    """

    def __init__(
        self,
        symbol: str = "ES",
        initial_capital: float = 100_000.0,
        commission_per_contract: float = 2.00,
        slippage_ticks: int = 1,
        swing_confirmation_bars: int = 2,
    ):
        self.symbol = symbol.upper()
        self.initial_capital = float(initial_capital)
        self.commission_per_contract = float(commission_per_contract)
        self.slippage_ticks = int(slippage_ticks)
        self.swing_confirmation_bars = int(swing_confirmation_bars)

    @staticmethod
    def _get_pool_price(pool: LiquidityPool) -> float:
        """
        Extracts the exact anchor price of a LiquidityPool regardless of internal field naming.
        """
        if hasattr(pool, "price"):
            return float(getattr(pool, "price"))
        low_b = float(getattr(pool, "lower_bound", getattr(pool, "bottom", 0.0)))
        high_b = float(getattr(pool, "upper_bound", getattr(pool, "top", 0.0)))
        return low_b if pool.pool_type == "BSL" else high_b

    @staticmethod
    def _group_macros(macro_pools: List[Dict[str, Any]]) -> Dict[pd.Timestamp, List[Dict[str, Any]]]:
        grouped: Dict[pd.Timestamp, List[Dict[str, Any]]] = {}
        for pool in macro_pools:
            t = pool["time"]
            if t not in grouped:
                grouped[t] = []
            grouped[t].append(pool)
        return grouped

    @staticmethod
    def _extract_structure_events(df_struct: pd.DataFrame, tf_minutes: int) -> List[Dict[str, Any]]:
        """
        Reads boolean columns from `domain/structure.py` (`choch_bullish`,
        `choch_bearish`, `bos_bullish`, `bos_bearish`) and tags each event with
        its exact 1-minute candle closing timestamp (`close_time`).
        """
        events: List[Dict[str, Any]] = []
        if df_struct.empty or "trend" not in df_struct.columns:
            return events

        bar_close_offset = pd.Timedelta(minutes=max(0, tf_minutes - 1))
        choch_bull = df_struct["choch_bullish"].to_numpy(dtype=bool)
        choch_bear = df_struct["choch_bearish"].to_numpy(dtype=bool)
        bos_bull = df_struct["bos_bullish"].to_numpy(dtype=bool)
        bos_bear = df_struct["bos_bearish"].to_numpy(dtype=bool)
        closes = df_struct["close"].to_numpy(dtype=float)
        sh_arr = df_struct["current_swing_high"].to_numpy(dtype=float)
        sl_arr = df_struct["current_swing_low"].to_numpy(dtype=float)
        times = df_struct.index

        for idx in range(len(df_struct)):
            close_time = times[idx] + bar_close_offset
            if choch_bull[idx]:
                lvl = float(sh_arr[idx]) if not np.isnan(sh_arr[idx]) else float(closes[idx])
                events.append({
                    "time": times[idx],
                    "close_time": close_time,
                    "kind": "CHoCH",
                    "label": f"Bullish CHoCH (> {lvl:.2f})",
                    "direction": "BULLISH",
                    "broken_level": lvl,
                    "close_price": float(closes[idx]),
                })
            elif bos_bull[idx]:
                lvl = float(sh_arr[idx]) if not np.isnan(sh_arr[idx]) else float(closes[idx])
                events.append({
                    "time": times[idx],
                    "close_time": close_time,
                    "kind": "BOS",
                    "label": f"Bullish BOS (> {lvl:.2f})",
                    "direction": "BULLISH",
                    "broken_level": lvl,
                    "close_price": float(closes[idx]),
                })

            if choch_bear[idx]:
                lvl = float(sl_arr[idx]) if not np.isnan(sl_arr[idx]) else float(closes[idx])
                events.append({
                    "time": times[idx],
                    "close_time": close_time,
                    "kind": "CHoCH",
                    "label": f"Bearish CHoCH (< {lvl:.2f})",
                    "direction": "BEARISH",
                    "broken_level": lvl,
                    "close_price": float(closes[idx]),
                })
            elif bos_bear[idx]:
                lvl = float(sl_arr[idx]) if not np.isnan(sl_arr[idx]) else float(closes[idx])
                events.append({
                    "time": times[idx],
                    "close_time": close_time,
                    "kind": "BOS",
                    "label": f"Bearish BOS (< {lvl:.2f})",
                    "direction": "BEARISH",
                    "broken_level": lvl,
                    "close_price": float(closes[idx]),
                })
        return events

    @staticmethod
    def _latest_trend_at(
        df_struct: pd.DataFrame,
        close_times: pd.DatetimeIndex,
        t: pd.Timestamp,
    ) -> str:
        """
        Returns the strictly confirmed market structure orderflow ('BULLISH' or 'BEARISH')
        at timestamp `t` using `df_struct['trend']`, with swing/displacement fallback
        if the dataset starts mid-trend before the first HTF fractal break.
        """
        if df_struct.empty or len(close_times) == 0:
            return "NEUTRAL"

        idx = close_times.searchsorted(t, side="right") - 1
        if idx < 0:
            idx = 0

        trend_val = int(df_struct["trend"].iloc[idx])
        if trend_val == 1:
            return "BULLISH"
        if trend_val == -1:
            return "BEARISH"

        sh = df_struct["current_swing_high"].iloc[: idx + 1].dropna()
        sl = df_struct["current_swing_low"].iloc[: idx + 1].dropna()
        close_now = float(df_struct["close"].iloc[idx])

        if len(sh) >= 1 and len(sl) >= 1:
            mid = (float(sh.iloc[-1]) + float(sl.iloc[-1])) * 0.5
            return "BULLISH" if close_now >= mid else "BEARISH"

        ref_idx = max(0, idx - 3)
        return "BULLISH" if close_now >= float(df_struct["open"].iloc[ref_idx]) else "BEARISH"

    @staticmethod
    def _is_pool_unbroken_at(pool: LiquidityPool, t: pd.Timestamp) -> bool:
        if pool.creation_time > t:
            return False
        for ev_type, ev_ts in pool.events:
            if ev_type in ("BREAK", "SWEEP") and ev_ts <= t:
                return False
        return True

    def run(
        self,
        df_primary: pd.DataFrame,
        strategy: Any,
        df_secondary: Optional[pd.DataFrame] = None,
        precompute_domain: bool = True,
    ) -> BacktestResult:
        if df_primary.empty:
            raise ValueError("df_primary cannot be empty.")

        portfolio = Portfolio(initial_capital=self.initial_capital)
        broker = SimulatedBroker(
            portfolio=portfolio,
            commission_per_contract=self.commission_per_contract,
            slippage_ticks=self.slippage_ticks,
        )
        event_bus = EventBus()

        times = df_primary.index
        opens = df_primary["open"].to_numpy(dtype=float)
        highs = df_primary["high"].to_numpy(dtype=float)
        lows = df_primary["low"].to_numpy(dtype=float)
        closes = df_primary["close"].to_numpy(dtype=float)
        volumes = (
            df_primary["volume"].to_numpy(dtype=float)
            if "volume" in df_primary.columns
            else np.zeros(len(df_primary), dtype=float)
        )

        swings_1m, all_pools_1m, fvgs_1m, dealing_ranges_1m, fib_ranges_1m = [], [], [], [], []
        structure_1m = pd.DataFrame()
        struct_events_1m, struct_1m_times = [], []
        swings_5m, fvgs_5m, dealing_ranges_5m, fib_ranges_5m = [], [], [], []
        struct_events_5m, struct_5m_times = [], []
        struct_df_by_tf: Dict[str, Tuple[pd.DataFrame, pd.DatetimeIndex]] = {}
        htf_target_catalog: List[Dict[str, Any]] = []
        smt_events: List[Dict[str, Any]] = []

        if precompute_domain and len(df_primary) >= 5:
            # 1. 1-Minute Primitives
            swings_1m = detect_swings(df_primary)
            structure_1m = detect_structure(df_primary, swings_1m)
            struct_events_1m = self._extract_structure_events(structure_1m, tf_minutes=1)
            struct_1m_times = [e["close_time"] for e in struct_events_1m]
            struct_df_by_tf["1min"] = (structure_1m, structure_1m.index)

            macro_list = extract_session_liquidity(df_primary)
            macro_by_time = self._group_macros(macro_list)
            _, liq_engine_1m = track_liquidity(structure_1m, swings_1m, macro_by_time, tolerance_ticks=2)
            all_pools_1m = liq_engine_1m.get_all_pools()
            fvgs_1m = track_fvgs(df_primary)
            dealing_ranges_1m = track_dealing_ranges(df_primary, swings_1m)
            fib_ranges_1m = track_fibonacci_ranges(df_primary, swings_1m)

            for p in all_pools_1m:
                if getattr(p, "source", "") == "Macro_Session":
                    htf_target_catalog.append({
                        "pool": p,
                        "timeframe": "SESSION",
                        "target_type": f"SESSION_{p.session_origin.upper()}",
                        "tier": 1,
                        "confirm_time": p.creation_time,
                    })

            # 2. Multi-Timeframe Primitives (5m, 15m, 1h, 4h)
            if len(df_primary) >= 25:
                df_5m = resample_timeframe(df_primary, "5min")
                if len(df_5m) >= 3:
                    swings_5m = detect_swings(df_5m)
                    structure_5m = detect_structure(df_5m, swings_5m)
                    struct_events_5m = self._extract_structure_events(structure_5m, tf_minutes=5)
                    struct_5m_times = [e["close_time"] for e in struct_events_5m]
                    struct_df_by_tf["5min"] = (
                        structure_5m,
                        structure_5m.index + pd.Timedelta(minutes=4),
                    )
                    fvgs_5m = track_fvgs(df_5m)
                    dealing_ranges_5m = track_dealing_ranges(df_5m, swings_5m)
                    fib_ranges_5m = track_fibonacci_ranges(df_5m, swings_5m)

                for tf_rule, tf_min, tier_num in (("15min", 15, 2), ("1h", 60, 1), ("4h", 240, 1)):
                    df_htf = resample_timeframe(df_primary, tf_rule)
                    if len(df_htf) >= 2:
                        sw_htf = detect_swings(df_htf)
                        st_htf = detect_structure(df_htf, sw_htf)
                        struct_df_by_tf[tf_rule] = (
                            st_htf,
                            st_htf.index + pd.Timedelta(minutes=tf_min - 1),
                        )

                        _, liq_htf = track_liquidity(st_htf, sw_htf, {}, tolerance_ticks=2)
                        confirm_lag = pd.Timedelta(minutes=tf_min * 3 - 1)
                        for p in liq_htf.get_all_pools():
                            is_eq_hl = p.touches >= 2
                            effective_tier = 1 if (tier_num == 1 or is_eq_hl) else 2
                            t_type = (
                                f"EQUAL_HL_{tf_rule.upper()}"
                                if is_eq_hl
                                else f"{tf_rule.upper()}_POOL"
                            )
                            htf_target_catalog.append({
                                "pool": p,
                                "timeframe": tf_rule,
                                "target_type": t_type,
                                "tier": effective_tier,
                                "confirm_time": p.creation_time + confirm_lag,
                            })

            # 3. Multi-Timeframe SMT Divergence (1m, 5m, 15m, 1h, 4h)
            if df_secondary is not None and not df_secondary.empty:
                sec_swings_1m = detect_swings(df_secondary)
                smt_events.extend(
                    detect_smt_divergence(
                        swings_1m, sec_swings_1m, df_primary, df_secondary,
                        time_window_minutes=15, timeframe="1min"
                    )
                )
                if len(df_primary) >= 30 and len(df_secondary) >= 30:
                    for smt_tf, smt_win in (("5min", 20), ("15min", 60), ("1h", 180), ("4h", 480)):
                        df_p_tf = resample_timeframe(df_primary, smt_tf)
                        df_s_tf = resample_timeframe(df_secondary, smt_tf)
                        if len(df_p_tf) >= 2 and len(df_s_tf) >= 2:
                            smt_events.extend(
                                detect_smt_divergence(
                                    detect_swings(df_p_tf),
                                    detect_swings(df_s_tf),
                                    df_p_tf,
                                    df_s_tf,
                                    time_window_minutes=smt_win,
                                    timeframe=smt_tf,
                                    df_es_1m=df_primary,
                                    df_nq_1m=df_secondary,
                                )
                            )

        confirm_delta_1m = pd.Timedelta(minutes=self.swing_confirmation_bars)
        confirm_delta_5m = pd.Timedelta(minutes=5 * 3 - 1)
        fvg_close_delta_5m = pd.Timedelta(minutes=4)
        recent_window = pd.Timedelta(minutes=20)

        sweeps_by_time: Dict[pd.Timestamp, List[Tuple[LiquidityPool, str]]] = {}
        for pool in all_pools_1m:
            lbl = (
                f"{pool.session_origin} Session {pool.pool_type}"
                if getattr(pool, "source", "") == "Macro_Session"
                else f"1m Structural {pool.pool_type}"
            )
            for ev_type, ev_ts in pool.events:
                if ev_type == "SWEEP":
                    if ev_ts not in sweeps_by_time:
                        sweeps_by_time[ev_ts] = []
                    sweeps_by_time[ev_ts].append((pool, lbl))

        for item in htf_target_catalog:
            if item["timeframe"] == "SESSION":
                continue
            pool_htf: LiquidityPool = item["pool"]
            tf_min_map = {"15min": 15, "1h": 60, "4h": 240}
            tf_m = tf_min_map.get(item["timeframe"], 15)
            for ev_type, ev_ts in pool_htf.events:
                if ev_type == "SWEEP":
                    close_ev_ts = ev_ts + pd.Timedelta(minutes=tf_m - 1)
                    lbl = f"{item['timeframe']} {pool_htf.pool_type}"
                    if close_ev_ts not in sweeps_by_time:
                        sweeps_by_time[close_ev_ts] = []
                    sweeps_by_time[close_ev_ts].append((pool_htf, lbl))

        recent_sweeps_buffer: List[Tuple[pd.Timestamp, LiquidityPool, str]] = []
        gap_engine = GapEngine()

        for i in range(len(df_primary)):
            t = times[i]
            o, h, l, c, v = opens[i], highs[i], lows[i], closes[i], volumes[i]

            gap_engine.process_candle(t, o, c, h, l)

            closed_trades = broker.process_candle(self.symbol, t, o, h, l, c)
            for tr in closed_trades:
                event_bus.publish(
                    Event("TRADE_CLOSED", t, {"symbol": self.symbol, "trade": tr})
                )

            confirmed_cutoff_1m = t - confirm_delta_1m
            confirmed_cutoff_5m = t - confirm_delta_5m

            if precompute_domain:
                if t in sweeps_by_time:
                    for p, lbl in sweeps_by_time[t]:
                        recent_sweeps_buffer.append((t, p, lbl))
                recent_sweeps_buffer = [
                    (ts, p, lbl) for ts, p, lbl in recent_sweeps_buffer if (t - ts) <= recent_window
                ]
                vis_sweeps = [p for _, p, _ in recent_sweeps_buffer]
                vis_sweep_details = [
                    {
                        "sweep_time": ts,
                        "pool_type": p.pool_type,
                        "price": self._get_pool_price(p),
                        "source": getattr(p, "source", "Structural"),
                        "session_origin": getattr(p, "session_origin", "Unknown"),
                        "creation_time": p.creation_time,
                        "label": (
                            f"{lbl} @ {self._get_pool_price(p):.2f} "
                            f"(formed {p.creation_time.strftime('%m-%d %H:%M')}, swept {ts.strftime('%H:%M')})"
                        ),
                    }
                    for ts, p, lbl in recent_sweeps_buffer
                ]

                vis_fvgs_1m = [
                    f for f in fvgs_1m
                    if f.creation_time <= t
                    and not any(
                        ev in ("FILLED_BY_CLOSE", "INVERTED") and ev_ts <= t
                        for ev, ev_ts, _ in f.events
                    )
                ]

                active_dr_1m = None
                for dr in dealing_ranges_1m:
                    if dr.creation_time <= confirmed_cutoff_1m:
                        if dr.break_time is None or dr.break_time > t:
                            active_dr_1m = dr
                    else:
                        break

                active_fib_1m = None
                for fib in fib_ranges_1m:
                    if fib.extreme_time <= confirmed_cutoff_1m:
                        if fib.break_time is None or fib.break_time > t:
                            active_fib_1m = fib

                vis_smt = [smt for smt in smt_events if is_smt_active_at(smt, t)]
                vis_swings_1m = [s for s in swings_1m if s["time"] <= confirmed_cutoff_1m]

                idx_1m_st = bisect.bisect_right(struct_1m_times, t)
                vis_struct_1m = struct_events_1m[max(0, idx_1m_st - 12):idx_1m_st]

                vis_swings_5m = [s for s in swings_5m if s["time"] <= confirmed_cutoff_5m]
                idx_5m_st = bisect.bisect_right(struct_5m_times, t)
                vis_struct_5m = struct_events_5m[max(0, idx_5m_st - 12):idx_5m_st]

                vis_fvgs_5m = [
                    f for f in fvgs_5m
                    if (f.creation_time + fvg_close_delta_5m) <= t
                    and not any(
                        ev in ("FILLED_BY_CLOSE", "INVERTED") and (ev_ts + fvg_close_delta_5m) <= t
                        for ev, ev_ts, _ in f.events
                    )
                ]

                active_dr_5m = None
                for dr in dealing_ranges_5m:
                    if dr.creation_time <= confirmed_cutoff_5m:
                        if dr.break_time is None or (dr.break_time + fvg_close_delta_5m) > t:
                            active_dr_5m = dr
                    else:
                        break

                active_fib_5m = None
                for fib in fib_ranges_5m:
                    if fib.extreme_time <= confirmed_cutoff_5m:
                        if fib.break_time is None or (fib.break_time + fvg_close_delta_5m) > t:
                            active_fib_5m = fib

                htf_trends: Dict[str, str] = {}
                for tf_key in ("1min", "5min", "15min", "1h", "4h"):
                    if tf_key in struct_df_by_tf:
                        st_df, st_close_times = struct_df_by_tf[tf_key]
                        htf_trends[tf_key] = self._latest_trend_at(st_df, st_close_times, t)
                    else:
                        htf_trends[tf_key] = htf_trends.get("1min", "BULLISH")

                vis_targets: List[Dict[str, Any]] = []
                session_start = t.replace(hour=9, minute=30, second=0, microsecond=0)
                taken_since_open: List[str] = []

                for item in htf_target_catalog:
                    if item["confirm_time"] > t:
                        continue
                    pool_obj: LiquidityPool = item["pool"]
                    p_price = self._get_pool_price(pool_obj)
                    if self._is_pool_unbroken_at(pool_obj, t):
                        vis_targets.append({
                            "pool_type": pool_obj.pool_type,
                            "price": p_price,
                            "target_type": item["target_type"],
                            "timeframe": item["timeframe"],
                            "tier": item["tier"],
                            "touches": pool_obj.touches,
                            "creation_time": pool_obj.creation_time,
                        })
                    elif t >= session_start:
                        tf_m = {"SESSION": 1, "15min": 15, "1h": 60, "4h": 240}.get(item["timeframe"], 1)
                        for ev_name, ev_ts in pool_obj.events:
                            real_ev_ts = ev_ts + pd.Timedelta(minutes=tf_m - 1)
                            if session_start <= real_ev_ts <= t:
                                taken_since_open.append(
                                    f"{item['target_type']} ({pool_obj.pool_type} @ {p_price:.2f} {ev_name} at {real_ev_ts.strftime('%H:%M')})"
                                )
                                break

                has_tier1_remaining = any(vt["tier"] == 1 for vt in vis_targets) or any(
                    not g.is_filled for g in gap_engine.active_gaps
                )

                if len(taken_since_open) == 0:
                    liq_taken_lvl = "LOW"
                elif len(taken_since_open) == 1 and has_tier1_remaining:
                    liq_taken_lvl = "MEDIUM"
                else:
                    liq_taken_lvl = "HIGH"

                vis_gaps = [g for g in gap_engine.active_gaps if not g.is_filled]
            else:
                vis_sweeps, vis_sweep_details, vis_fvgs_1m = [], [], []
                active_dr_1m, active_fib_1m = None, None
                vis_smt, vis_swings_1m, vis_struct_1m = [], [], []
                vis_swings_5m, vis_struct_5m, vis_fvgs_5m = [], [], []
                active_dr_5m, active_fib_5m = None, None
                htf_trends = {}
                vis_targets, vis_gaps, taken_since_open = [], [], []
                liq_taken_lvl = "LOW"

            ctx = BarContext(
                symbol=self.symbol,
                timestamp=t,
                bar_index=i,
                open=o,
                high=h,
                low=l,
                close=c,
                volume=v,
                history_df=df_primary.iloc[: i + 1],
                confirmed_swings=vis_swings_1m,
                active_fvgs=vis_fvgs_1m,
                recent_sweeps=vis_sweeps,
                recent_sweep_details=vis_sweep_details,
                active_dealing_range=active_dr_1m,
                active_fib_range=active_fib_1m,
                recent_smt_events=vis_smt,
                structure_events_1m=vis_struct_1m,
                confirmed_swings_5m=vis_swings_5m,
                structure_events_5m=vis_struct_5m,
                active_fvgs_5m=vis_fvgs_5m,
                active_dealing_range_5m=active_dr_5m,
                active_fib_range_5m=active_fib_5m,
                htf_trends=htf_trends,
                target_pools=vis_targets,
                active_weekend_gaps=vis_gaps,
                liquidity_taken_level=liq_taken_lvl,
                session_taken_pools=taken_since_open,
            )

            strategy.on_bar(ctx, broker, portfolio)
            portfolio.update_mark_to_market(t, {self.symbol: c})

        if portfolio.has_open_position(self.symbol):
            last_time = times[-1]
            last_close = closes[-1]
            portfolio.close_position(
                symbol=self.symbol,
                exit_price=last_close,
                timestamp=last_time,
                exit_reason="END_OF_DATA",
                commission_per_side=self.commission_per_contract,
            )
            if portfolio.equity_curve:
                portfolio.equity_curve[-1]["cash"] = portfolio.cash
                portfolio.equity_curve[-1]["unrealized_pnl"] = 0.0
                portfolio.equity_curve[-1]["equity"] = portfolio.cash

        strategy_name = getattr(strategy, "name", strategy.__class__.__name__)
        return calculate_metrics(portfolio, strategy_name=strategy_name, symbol=self.symbol)