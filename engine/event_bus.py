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
from domain.smt import (
    detect_smt_divergence,
    is_smt_active_at,
    resample_tv_timeframe,
    detect_htf_swings,
)
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
    # Targets & Pre-Entry Liquidity Exhaustion (by direction)
    target_pools: List[Dict[str, Any]] = field(default_factory=list)
    active_weekend_gaps: List[WeekendGap] = field(default_factory=list)
    liquidity_taken_level: str = "LOW"
    liquidity_taken_by_dir: Dict[str, str] = field(default_factory=lambda: {"LONG": "LOW", "SHORT": "LOW"})
    session_taken_pools: List[str] = field(default_factory=list)
    session_taken_by_dir: Dict[str, List[str]] = field(default_factory=lambda: {"LONG": [], "SHORT": []})


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
        if hasattr(pool, "price"):
            return float(getattr(pool, "price"))
        low_b = float(getattr(pool, "lower_bound", getattr(pool, "bottom", 0.0)))
        high_b = float(getattr(pool, "upper_bound", getattr(pool, "top", 0.0)))
        return low_b if pool.pool_type == "BSL" else high_b

    @staticmethod
    def _shift_swings_for_structure(df_tf: pd.DataFrame, swings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Without modifying `domain/structure.py`, shifts each swing's placement index by +1 bar
        when feeding `detect_structure` so that a breakout candle `i` is always evaluated against
        the PREVIOUS active swing before registering its own high/low as the new swing.
        """
        if df_tf.empty or not swings:
            return swings
        idx_list = df_tf.index
        n = len(idx_list)
        shifted: List[Dict[str, Any]] = []
        for s in swings:
            pos = idx_list.searchsorted(s["time"])
            if pos + 1 < n:
                s_copy = dict(s)
                s_copy["time"] = idx_list[pos + 1]
                shifted.append(s_copy)
        return shifted

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
    def _extract_structure_events(
        df_struct: pd.DataFrame,
        tf_minutes: int,
        swings: Optional[List[Dict[str, Any]]] = None,
        fvgs: Optional[List[FairValueGap]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extracts standard BOS/CHoCH events from `df_struct` AND implements Option A
        (Inverting the Last Remaining FVG of the Leg = IFVG_BOS):
        - In an upward leg from the latest SwingLow, if >= 1 un-inverted BISI FVG exists
          and candle `idx` closes strictly below the LOWEST remaining BISI FVG of that leg
          (so 0 un-inverted BISI FVGs remain in the leg), emits `Bearish IFVG_BOS` and updates `trend` to -1.
        - In a downward leg from the latest SwingHigh, if >= 1 un-inverted SIBI FVG exists
          and candle `idx` closes strictly above the HIGHEST remaining SIBI FVG of that leg
          (so 0 un-inverted SIBI FVGs remain in the leg), emits `Bullish IFVG_BOS` and updates `trend` to +1.
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
        trends = df_struct["trend"].to_numpy(dtype=int).copy()
        times = df_struct.index

        swing_lows = [s for s in (swings or []) if s["type"] == "SwingLow"]
        swing_highs = [s for s in (swings or []) if s["type"] == "SwingHigh"]
        swing_low_times = [s["time"] for s in swing_lows]
        swing_high_times = [s["time"] for s in swing_highs]

        bisi_fvgs = [f for f in (fvgs or []) if f.fvg_type == "BISI"]
        sibi_fvgs = [f for f in (fvgs or []) if f.fvg_type == "SIBI"]

        # Precompute the exact bar index where each FVG is first inverted by a candle close
        bisi_inv_idx: Dict[int, int] = {}
        for f_i, f in enumerate(bisi_fvgs):
            start_k = times.searchsorted(f.creation_time, side="right")
            for k in range(start_k, len(times)):
                if closes[k] < f.bottom:
                    bisi_inv_idx[f_i] = k
                    break

        sibi_inv_idx: Dict[int, int] = {}
        for f_i, f in enumerate(sibi_fvgs):
            start_k = times.searchsorted(f.creation_time, side="right")
            for k in range(start_k, len(times)):
                if closes[k] > f.top:
                    sibi_inv_idx[f_i] = k
                    break

        consumed_bull_legs = set()
        consumed_bear_legs = set()
        override_trend: Optional[int] = None
        last_std_trend = 0

        for idx in range(len(df_struct)):
            t_bar = times[idx]
            close_time = t_bar + bar_close_offset
            std_t = int(trends[idx])

            if std_t != last_std_trend and std_t != 0:
                override_trend = None
                last_std_trend = std_t

            if override_trend is not None:
                trends[idx] = override_trend

            has_bull_ev = False
            has_bear_ev = False

            if choch_bull[idx]:
                lvl = float(sh_arr[idx]) if not np.isnan(sh_arr[idx]) else float(closes[idx])
                events.append({
                    "time": t_bar,
                    "close_time": close_time,
                    "kind": "CHoCH",
                    "label": f"Bullish CHoCH (> {lvl:.2f})",
                    "direction": "BULLISH",
                    "broken_level": lvl,
                    "close_price": float(closes[idx]),
                })
                has_bull_ev = True
                override_trend = 1
                trends[idx] = 1
            elif bos_bull[idx]:
                lvl = float(sh_arr[idx]) if not np.isnan(sh_arr[idx]) else float(closes[idx])
                events.append({
                    "time": t_bar,
                    "close_time": close_time,
                    "kind": "BOS",
                    "label": f"Bullish BOS (> {lvl:.2f})",
                    "direction": "BULLISH",
                    "broken_level": lvl,
                    "close_price": float(closes[idx]),
                })
                has_bull_ev = True
                override_trend = 1
                trends[idx] = 1

            if choch_bear[idx]:
                lvl = float(sl_arr[idx]) if not np.isnan(sl_arr[idx]) else float(closes[idx])
                events.append({
                    "time": t_bar,
                    "close_time": close_time,
                    "kind": "CHoCH",
                    "label": f"Bearish CHoCH (< {lvl:.2f})",
                    "direction": "BEARISH",
                    "broken_level": lvl,
                    "close_price": float(closes[idx]),
                })
                has_bear_ev = True
                override_trend = -1
                trends[idx] = -1
            elif bos_bear[idx]:
                lvl = float(sl_arr[idx]) if not np.isnan(sl_arr[idx]) else float(closes[idx])
                events.append({
                    "time": t_bar,
                    "close_time": close_time,
                    "kind": "BOS",
                    "label": f"Bearish BOS (< {lvl:.2f})",
                    "direction": "BEARISH",
                    "broken_level": lvl,
                    "close_price": float(closes[idx]),
                })
                has_bear_ev = True
                override_trend = -1
                trends[idx] = -1

            # Option A: Check if the LAST remaining FVG of the active leg was inverted on bar `idx`
            if bisi_fvgs and not has_bear_ev:
                sl_pos = bisect.bisect_left(swing_low_times, t_bar) - 1
                if sl_pos >= 0:
                    leg_low_time = swing_lows[sl_pos]["time"]
                    leg_low_price = float(swing_lows[sl_pos]["price"])
                    if leg_low_time not in consumed_bull_legs:
                        # Find all BISI FVGs in this upward leg that were still UN-INVERTED entering bar `idx`
                        uninv_entering = [
                            (f_i, f) for f_i, f in enumerate(bisi_fvgs)
                            if leg_low_time <= f.creation_time < t_bar
                            and f.bottom >= leg_low_price
                            and bisi_inv_idx.get(f_i, 10**9) >= idx
                        ]
                        if uninv_entering:
                            min_bisi_bottom = min(f.bottom for _, f in uninv_entering)
                            # If candle `idx` closes strictly below the LOWEST remaining BISI of the leg,
                            # 0 BISI FVGs remain in the leg -> Bearish IFVG_BOS!
                            if closes[idx] < min_bisi_bottom:
                                events.append({
                                    "time": t_bar,
                                    "close_time": close_time,
                                    "kind": "IFVG_BOS",
                                    "label": f"Bearish IFVG_BOS (< {min_bisi_bottom:.2f})",
                                    "direction": "BEARISH",
                                    "broken_level": float(min_bisi_bottom),
                                    "close_price": float(closes[idx]),
                                })
                                consumed_bull_legs.add(leg_low_time)
                                override_trend = -1
                                trends[idx] = -1

            if sibi_fvgs and not has_bull_ev:
                sh_pos = bisect.bisect_left(swing_high_times, t_bar) - 1
                if sh_pos >= 0:
                    leg_high_time = swing_highs[sh_pos]["time"]
                    leg_high_price = float(swing_highs[sh_pos]["price"])
                    if leg_high_time not in consumed_bear_legs:
                        # Find all SIBI FVGs in this downward leg that were still UN-INVERTED entering bar `idx`
                        uninv_entering = [
                            (f_i, f) for f_i, f in enumerate(sibi_fvgs)
                            if leg_high_time <= f.creation_time < t_bar
                            and f.top <= leg_high_price
                            and sibi_inv_idx.get(f_i, 10**9) >= idx
                        ]
                        if uninv_entering:
                            max_sibi_top = max(f.top for _, f in uninv_entering)
                            # If candle `idx` closes strictly above the HIGHEST remaining SIBI of the leg,
                            # 0 SIBI FVGs remain in the leg -> Bullish IFVG_BOS!
                            if closes[idx] > max_sibi_top:
                                events.append({
                                    "time": t_bar,
                                    "close_time": close_time,
                                    "kind": "IFVG_BOS",
                                    "label": f"Bullish IFVG_BOS (> {max_sibi_top:.2f})",
                                    "direction": "BULLISH",
                                    "broken_level": float(max_sibi_top),
                                    "close_price": float(closes[idx]),
                                })
                                consumed_bear_legs.add(leg_high_time)
                                override_trend = 1
                                trends[idx] = 1

        df_struct["trend"] = trends
        return events

    @staticmethod
    def _latest_trend_at(
        df_struct: pd.DataFrame,
        close_times: pd.DatetimeIndex,
        t: pd.Timestamp,
    ) -> str:
        """
        Returns the strictly confirmed market structure orderflow ('BULLISH' or 'BEARISH')
        at timestamp `t` using ONLY fully closed candles (`close_time <= t`).
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

        sh_val = float(df_struct["current_swing_high"].iloc[idx])
        sl_val = float(df_struct["current_swing_low"].iloc[idx])
        closed_bar_price = float(df_struct["close"].iloc[idx])

        if not np.isnan(sh_val) and not np.isnan(sl_val):
            mid = (sh_val + sl_val) * 0.5
            return "BULLISH" if closed_bar_price >= mid else "BEARISH"

        ref_idx = max(0, idx - 3)
        return "BULLISH" if closed_bar_price >= float(df_struct["open"].iloc[ref_idx]) else "BEARISH"

    @staticmethod
    def _compute_1m_pool_resolution(
        pool_type: str,
        pool_price: float,
        confirm_time: pd.Timestamp,
        times_1m: pd.DatetimeIndex,
        highs_1m: np.ndarray,
        lows_1m: np.ndarray,
        closes_1m: np.ndarray,
    ) -> Tuple[Optional[str], Optional[pd.Timestamp]]:
        start_idx = times_1m.searchsorted(confirm_time, side="right")
        for idx in range(start_idx, len(times_1m)):
            if pool_type == "BSL":
                if closes_1m[idx] > pool_price:
                    return "BREAK", times_1m[idx]
                if highs_1m[idx] > pool_price:
                    return "SWEEP", times_1m[idx]
            else:
                if closes_1m[idx] < pool_price:
                    return "BREAK", times_1m[idx]
                if lows_1m[idx] < pool_price:
                    return "SWEEP", times_1m[idx]
        return None, None

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

        dedup_price_tol = 1.50 if self.symbol == "ES" else 6.00

        if precompute_domain and len(df_primary) >= 5:
            # 1. 1-Minute Primitives (with Option A IFVG_BOS enabled)
            swings_1m = detect_swings(df_primary)
            fvgs_1m = track_fvgs(df_primary)
            structure_1m = detect_structure(
                df_primary, self._shift_swings_for_structure(df_primary, swings_1m)
            )
            struct_events_1m = self._extract_structure_events(
                structure_1m, tf_minutes=1, swings=swings_1m, fvgs=fvgs_1m
            )
            struct_1m_times = [e["close_time"] for e in struct_events_1m]
            struct_df_by_tf["1min"] = (structure_1m, structure_1m.index)

            macro_list = extract_session_liquidity(df_primary)
            macro_by_time = self._group_macros(macro_list)
            _, liq_engine_1m = track_liquidity(structure_1m, swings_1m, macro_by_time, tolerance_ticks=2)
            all_pools_1m = liq_engine_1m.get_all_pools()
            dealing_ranges_1m = track_dealing_ranges(df_primary, swings_1m)
            fib_ranges_1m = track_fibonacci_ranges(df_primary, swings_1m)

            # Register Macro Session Pools FIRST (Highest Priority)
            for p in all_pools_1m:
                if getattr(p, "source", "") == "Macro_Session":
                    p_price = self._get_pool_price(p)
                    res_type, res_ts = self._compute_1m_pool_resolution(
                        p.pool_type, p_price, p.creation_time, times, highs, lows, closes
                    )
                    htf_target_catalog.append({
                        "pool": p,
                        "pool_type": p.pool_type,
                        "price": p_price,
                        "timeframe": "SESSION",
                        "target_type": f"SESSION_{p.session_origin.upper()}",
                        "tier": 1,
                        "is_major_exhaustion_pool": True,
                        "confirm_time": p.creation_time,
                        "first_touch_type": res_type,
                        "first_touch_time": res_ts,
                    })

            # 2. Multi-Timeframe Primitives (5m with Option A IFVG_BOS, then 4h -> 1h -> 15m)
            if len(df_primary) >= 25:
                df_5m = resample_tv_timeframe(df_primary, "5min")
                if len(df_5m) >= 3:
                    swings_5m = detect_htf_swings(df_5m)
                    fvgs_5m = track_fvgs(df_5m)
                    structure_5m = detect_structure(
                        df_5m, self._shift_swings_for_structure(df_5m, swings_5m)
                    )
                    struct_events_5m = self._extract_structure_events(
                        structure_5m, tf_minutes=5, swings=swings_5m, fvgs=fvgs_5m
                    )
                    struct_5m_times = [e["close_time"] for e in struct_events_5m]
                    struct_df_by_tf["5min"] = (
                        structure_5m,
                        structure_5m.index + pd.Timedelta(minutes=4),
                    )
                    dealing_ranges_5m = track_dealing_ranges(df_5m, swings_5m)
                    fib_ranges_5m = track_fibonacci_ranges(df_5m, swings_5m)

                for tf_rule, tf_min, tier_num, is_major in (
                    ("4h", 240, 1, True),
                    ("1h", 60, 1, True),
                    ("15min", 15, 2, False),
                ):
                    df_htf = resample_tv_timeframe(df_primary, tf_rule)
                    if len(df_htf) >= 3:
                        sw_htf = detect_htf_swings(df_htf)
                        st_htf = detect_structure(
                            df_htf, self._shift_swings_for_structure(df_htf, sw_htf)
                        )
                        struct_df_by_tf[tf_rule] = (
                            st_htf,
                            st_htf.index + pd.Timedelta(minutes=tf_min - 1),
                        )

                        _, liq_htf = track_liquidity(st_htf, sw_htf, {}, tolerance_ticks=2)
                        confirm_lag = pd.Timedelta(minutes=tf_min * 2 - 1)
                        for p in liq_htf.get_all_pools():
                            p_price = self._get_pool_price(p)
                            conf_ts = p.creation_time + confirm_lag

                            # Deduplicate if this level already exists as a Macro Session or higher-TF pool
                            is_duplicate = False
                            for existing in htf_target_catalog:
                                if existing["pool_type"] == p.pool_type:
                                    if abs(existing["price"] - p_price) <= dedup_price_tol and abs(
                                        (existing["pool"].creation_time - p.creation_time).total_seconds()
                                    ) <= 12 * 3600:
                                        is_duplicate = True
                                        break
                            if is_duplicate:
                                continue

                            is_eq_hl = p.touches >= 2
                            effective_tier = 1 if (tier_num == 1 or is_eq_hl) else 2
                            t_type = (
                                f"EQUAL_HL_{tf_rule.upper()}"
                                if is_eq_hl
                                else f"{tf_rule.upper()}_POOL"
                            )
                            res_type, res_ts = self._compute_1m_pool_resolution(
                                p.pool_type, p_price, conf_ts, times, highs, lows, closes
                            )
                            htf_target_catalog.append({
                                "pool": p,
                                "pool_type": p.pool_type,
                                "price": p_price,
                                "timeframe": tf_rule,
                                "target_type": t_type,
                                "tier": effective_tier,
                                "is_major_exhaustion_pool": is_major,
                                "confirm_time": conf_ts,
                                "first_touch_type": res_type,
                                "first_touch_time": res_ts,
                            })

            # 3. Multi-Timeframe SMT Divergence (Always pass ES as 1st arg and NQ as 2nd arg)
            if df_secondary is not None and not df_secondary.empty:
                if self.symbol == "NQ":
                    df_es_base, df_nq_base = df_secondary, df_primary
                    es_sw_1m, nq_sw_1m = detect_swings(df_secondary), swings_1m
                else:
                    df_es_base, df_nq_base = df_primary, df_secondary
                    es_sw_1m, nq_sw_1m = swings_1m, detect_swings(df_secondary)

                smt_events.extend(
                    detect_smt_divergence(
                        es_sw_1m, nq_sw_1m, df_es_base, df_nq_base,
                        time_window_minutes=15, timeframe="1min"
                    )
                )
                if len(df_es_base) >= 30 and len(df_nq_base) >= 30:
                    for smt_tf, smt_win in (("5min", 20), ("15min", 60), ("1h", 180), ("4h", 480)):
                        df_es_tf = resample_tv_timeframe(df_es_base, smt_tf)
                        df_nq_tf = resample_tv_timeframe(df_nq_base, smt_tf)
                        if len(df_es_tf) >= 3 and len(df_nq_tf) >= 3:
                            smt_events.extend(
                                detect_smt_divergence(
                                    detect_htf_swings(df_es_tf),
                                    detect_htf_swings(df_nq_tf),
                                    df_es_tf,
                                    df_nq_tf,
                                    time_window_minutes=smt_win,
                                    timeframe=smt_tf,
                                    df_es_1m=df_es_base,
                                    df_nq_1m=df_nq_base,
                                )
                            )

        confirm_delta_1m = pd.Timedelta(minutes=self.swing_confirmation_bars)
        confirm_delta_5m = pd.Timedelta(minutes=5 * 2 - 1)
        fvg_close_delta_5m = pd.Timedelta(minutes=4)
        recent_window = pd.Timedelta(minutes=25)

        sweeps_by_time: Dict[pd.Timestamp, List[Tuple[LiquidityPool, str]]] = {}
        include_1m_structural = len(df_primary) < 25

        if include_1m_structural:
            for pool in all_pools_1m:
                lbl = f"1m Structural {pool.pool_type}"
                for ev_type, ev_ts in pool.events:
                    if ev_type == "SWEEP":
                        sweeps_by_time.setdefault(ev_ts, []).append((pool, lbl))

        for item in htf_target_catalog:
            res_ts = item.get("first_touch_time")
            res_type = item.get("first_touch_type")
            if res_ts is not None and res_type in ("SWEEP", "BREAK"):
                pool_obj = item["pool"]
                lbl = (
                    f"{pool_obj.session_origin} Session {pool_obj.pool_type}"
                    if item["timeframe"] == "SESSION"
                    else f"{item['timeframe']} {pool_obj.pool_type}"
                )
                sweeps_by_time.setdefault(res_ts, []).append((pool_obj, lbl))

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
                            f"(formed {p.creation_time.strftime('%m-%d %H:%M')}, swept/taken {ts.strftime('%H:%M')})"
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
                taken_bsl_since_open: List[str] = []
                taken_ssl_since_open: List[str] = []
                major_bsl_taken = 0
                major_ssl_taken = 0

                for item in htf_target_catalog:
                    if item["confirm_time"] > t:
                        continue
                    pool_obj: LiquidityPool = item["pool"]
                    if item["timeframe"] == "SESSION" and (t - pool_obj.creation_time) > pd.Timedelta(hours=30):
                        continue

                    p_price = float(item["price"])
                    res_ts = item.get("first_touch_time")
                    res_type = item.get("first_touch_type")

                    if res_ts is None or res_ts > t:
                        vis_targets.append({
                            "pool_type": pool_obj.pool_type,
                            "price": p_price,
                            "target_type": item["target_type"],
                            "timeframe": item["timeframe"],
                            "tier": item["tier"],
                            "touches": pool_obj.touches,
                            "creation_time": pool_obj.creation_time,
                        })
                    elif session_start <= res_ts <= t:
                        entry_str = f"{item['target_type']} ({pool_obj.pool_type} @ {p_price:.2f} {res_type} at {res_ts.strftime('%H:%M')})"
                        if pool_obj.pool_type == "BSL":
                            taken_bsl_since_open.append(entry_str)
                            if item.get("is_major_exhaustion_pool", False):
                                major_bsl_taken += 1
                        else:
                            taken_ssl_since_open.append(entry_str)
                            if item.get("is_major_exhaustion_pool", False):
                                major_ssl_taken += 1

                has_major_bsl_left = any(
                    vt["pool_type"] == "BSL" and vt["timeframe"] in ("SESSION", "4h", "1h") for vt in vis_targets
                )
                has_major_ssl_left = any(
                    vt["pool_type"] == "SSL" and vt["timeframe"] in ("SESSION", "4h", "1h") for vt in vis_targets
                )

                if major_bsl_taken == 0:
                    long_exh = "LOW"
                elif major_bsl_taken == 1 and has_major_bsl_left:
                    long_exh = "MEDIUM"
                else:
                    long_exh = "HIGH"

                if major_ssl_taken == 0:
                    short_exh = "LOW"
                elif major_ssl_taken == 1 and has_major_ssl_left:
                    short_exh = "MEDIUM"
                else:
                    short_exh = "HIGH"

                liq_taken_by_dir = {"LONG": long_exh, "SHORT": short_exh}
                session_taken_by_dir = {
                    "LONG": taken_bsl_since_open,
                    "SHORT": taken_ssl_since_open,
                }
                liq_taken_lvl = long_exh if long_exh == short_exh else "MEDIUM"
                taken_since_open = taken_bsl_since_open + taken_ssl_since_open
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
                liq_taken_by_dir = {"LONG": "LOW", "SHORT": "LOW"}
                session_taken_by_dir = {"LONG": [], "SHORT": []}

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
                liquidity_taken_by_dir=liq_taken_by_dir,
                session_taken_pools=taken_since_open,
                session_taken_by_dir=session_taken_by_dir,
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