from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, Tuple
from datetime import time as dtime
import bisect
import time as pytime
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
    detect_session_smt_divergence,
    extract_macro_dict_fields,
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
    symbol: str
    timestamp: pd.Timestamp
    bar_index: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    history_df: pd.DataFrame
    confirmed_swings: List[Dict[str, Any]] = field(default_factory=list)
    active_fvgs: List[FairValueGap] = field(default_factory=list)
    recent_fvgs_all: List[FairValueGap] = field(default_factory=list)
    recent_sweeps: List[LiquidityPool] = field(default_factory=list)
    recent_sweep_details: List[Dict[str, Any]] = field(default_factory=list)
    active_dealing_range: Optional[DealingRange] = None
    active_fib_range: Optional[FibonacciRange] = None
    recent_smt_events: List[Dict[str, Any]] = field(default_factory=list)
    structure_events_1m: List[Dict[str, Any]] = field(default_factory=list)
    confirmed_swings_5m: List[Dict[str, Any]] = field(default_factory=list)
    structure_events_5m: List[Dict[str, Any]] = field(default_factory=list)
    active_fvgs_5m: List[FairValueGap] = field(default_factory=list)
    active_dealing_range_5m: Optional[DealingRange] = None
    active_fib_range_5m: Optional[FibonacciRange] = None
    htf_trends: Dict[str, str] = field(default_factory=dict)
    sec_htf_trends: Dict[str, str] = field(default_factory=dict)
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
    win_count: int = 0
    be_count: int = 0
    loss_count: int = 0


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
        be_trades = [t for t in trades if t.exit_reason == "BE_TRAILED_SL"]
        win_trades = [t for t in trades if t.exit_reason != "BE_TRAILED_SL" and t.net_pnl > 0]
        loss_trades = [t for t in trades if t.exit_reason != "BE_TRAILED_SL" and t.net_pnl <= 0]

        win_count = len(win_trades)
        be_count = len(be_trades)
        loss_count = len(loss_trades)

        win_rate = (win_count / total_trades) * 100.0

        positive_pnls = [t.net_pnl for t in trades if t.net_pnl > 0]
        negative_pnls = [abs(t.net_pnl) for t in trades if t.net_pnl <= 0]
        gross_win = sum(positive_pnls)
        gross_loss = sum(negative_pnls)
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
        avg_r = float(np.mean([t.r_multiple for t in trades]))
    else:
        win_count = 0
        be_count = 0
        loss_count = 0
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
        sharpe_ratio = float((returns.mean() / std_ret) * np.sqrt(252 * 390)) if std_ret > 1e-12 else 0.0
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
        win_count=win_count,
        be_count=be_count,
        loss_count=loss_count,
    )


class BacktestEngine:
    _GLOBAL_SYMBOL_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    _GLOBAL_SMT_CACHE: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}

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
        self._domain_cache_key: Optional[Tuple[Any, ...]] = None
        self._domain_cache: Optional[Dict[str, Any]] = None

    @staticmethod
    def _get_pool_price(pool: LiquidityPool) -> float:
        if hasattr(pool, "price"):
            return float(getattr(pool, "price"))
        low_b = float(getattr(pool, "lower_bound", getattr(pool, "bottom", 0.0)))
        high_b = float(getattr(pool, "upper_bound", getattr(pool, "top", 0.0)))
        return float((low_b + high_b) * 0.5)

    @staticmethod
    def _shift_swings_for_structure(df_tf: pd.DataFrame, swings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
    def _filter_1h_structural_swings(df_1h: pd.DataFrame, swings_1h: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(swings_1h) < 4:
            return swings_1h

        filtered: List[Dict[str, Any]] = []
        last_major_high = None
        last_major_low = None

        for i, s in enumerate(swings_1h):
            p = float(s["price"])
            if s["type"] == "SwingHigh":
                if last_major_high is None or p >= last_major_high:
                    last_major_high = p
                filtered.append(s)
            else:
                if last_major_low is None or last_major_high is None or p <= last_major_low:
                    last_major_low = p
                    filtered.append(s)
                    continue

                leg_range = last_major_high - last_major_low
                made_higher_high = False
                for nxt in swings_1h[i + 1: i + 5]:
                    if nxt["type"] == "SwingHigh" and float(nxt["price"]) > last_major_high:
                        made_higher_high = True
                        break
                if made_higher_high or leg_range <= 0 or (last_major_high - p) >= 0.35 * leg_range:
                    last_major_low = p
                    filtered.append(s)

        return filtered

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
        events: List[Dict[str, Any]] = []
        if df_struct.empty or "trend" not in df_struct.columns:
            return events

        bar_close_offset = pd.Timedelta(minutes=max(0, tf_minutes - 1))
        choch_bull = df_struct["choch_bullish"].to_numpy(dtype=bool)
        choch_bear = df_struct["choch_bearish"].to_numpy(dtype=bool)
        bos_bull = df_struct["bos_bullish"].to_numpy(dtype=bool)
        bos_bear = df_struct["bos_bearish"].to_numpy(dtype=bool)
        closes = df_struct["close"].to_numpy(dtype=float)
        highs = df_struct["high"].to_numpy(dtype=float)
        lows = df_struct["low"].to_numpy(dtype=float)
        sh_arr = df_struct["current_swing_high"].to_numpy(dtype=float)
        sl_arr = df_struct["current_swing_low"].to_numpy(dtype=float)
        trends = df_struct["trend"].to_numpy(dtype=int).copy()
        times = df_struct.index
        n_bars = len(df_struct)

        swing_lows = [s for s in (swings or []) if s["type"] == "SwingLow"]
        swing_highs = [s for s in (swings or []) if s["type"] == "SwingHigh"]
        swing_low_times = [s["time"] for s in swing_lows]
        swing_high_times = [s["time"] for s in swing_highs]

        bisi_fvgs = [f for f in (fvgs or []) if f.fvg_type == "BISI"]
        sibi_fvgs = [f for f in (fvgs or []) if f.fvg_type == "SIBI"]
        bisi_creation_times = [f.creation_time for f in bisi_fvgs]
        sibi_creation_times = [f.creation_time for f in sibi_fvgs]

        bisi_inv_idx: Dict[int, int] = {}
        bisi_start_idx: List[int] = []
        for f_i, f in enumerate(bisi_fvgs):
            start_k = int(times.searchsorted(f.creation_time, side="right"))
            bisi_start_idx.append(start_k)
            if start_k < n_bars:
                inv_mask = closes[start_k:] < f.bottom
                if inv_mask.any():
                    bisi_inv_idx[f_i] = start_k + int(np.argmax(inv_mask))

        sibi_inv_idx: Dict[int, int] = {}
        sibi_start_idx: List[int] = []
        for f_i, f in enumerate(sibi_fvgs):
            start_k = int(times.searchsorted(f.creation_time, side="right"))
            sibi_start_idx.append(start_k)
            if start_k < n_bars:
                inv_mask = closes[start_k:] > f.top
                if inv_mask.any():
                    sibi_inv_idx[f_i] = start_k + int(np.argmax(inv_mask))

        consumed_bull_legs = set()
        consumed_bear_legs = set()
        override_trend: Optional[int] = None
        last_std_trend = 0

        for idx in range(n_bars):
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

            if bisi_fvgs and not has_bear_ev:
                sl_pos = bisect.bisect_left(swing_low_times, t_bar) - 1
                if sl_pos >= 0:
                    leg_low_time = swing_lows[sl_pos]["time"]
                    leg_low_price = float(swing_lows[sl_pos]["price"])
                    if leg_low_time not in consumed_bull_legs:
                        lo_f = bisect.bisect_left(bisi_creation_times, leg_low_time)
                        hi_f = bisect.bisect_left(bisi_creation_times, t_bar)
                        uninv_entering = []
                        for f_i in range(lo_f, hi_f):
                            f = bisi_fvgs[f_i]
                            if f.bottom < leg_low_price:
                                continue
                            if bisi_inv_idx.get(f_i, 10**9) < idx:
                                continue
                            # For 1m/5m/15m, ignore FVGs whose bottom was already breached on an earlier candle;
                            # for 1H (`tf_minutes == 60`), an FVG that was merely wicked (like 03:00 on Oct 1 NQ)
                            # remains the active last FVG of the range until a candle CLOSES below `f.bottom`!
                            if tf_minutes < 60:
                                f_start = bisi_start_idx[f_i]
                                if f_start < idx and float(lows[f_start:idx].min()) < f.bottom:
                                    continue
                            uninv_entering.append((f_i, f))

                        if uninv_entering:
                            min_bisi_bottom = min(f.bottom for _, f in uninv_entering)
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
                        lo_f = bisect.bisect_left(sibi_creation_times, leg_high_time)
                        hi_f = bisect.bisect_left(sibi_creation_times, t_bar)
                        uninv_entering = []
                        for f_i in range(lo_f, hi_f):
                            f = sibi_fvgs[f_i]
                            if f.top > leg_high_price:
                                continue
                            if sibi_inv_idx.get(f_i, 10**9) < idx:
                                continue
                            if tf_minutes < 60:
                                f_start = sibi_start_idx[f_i]
                                if f_start < idx and float(highs[f_start:idx].max()) > f.top:
                                    continue
                            uninv_entering.append((f_i, f))

                        if uninv_entering:
                            max_sibi_top = max(f.top for _, f in uninv_entering)
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
    def _session_resolution_start(creation_time: pd.Timestamp, session_origin: str) -> pd.Timestamp:
        s_up = session_origin.upper()
        if "ASIA" in s_up:
            if creation_time.hour >= 18:
                return (creation_time + pd.Timedelta(days=1)).replace(hour=3, minute=0, second=0, microsecond=0)
            return creation_time.replace(hour=3, minute=0, second=0, microsecond=0)
        if "LONDON" in s_up:
            return creation_time.replace(hour=9, minute=30, second=0, microsecond=0)
        return creation_time + pd.Timedelta(minutes=5)

    @staticmethod
    def _compute_1m_pool_resolution(
        pool_type: str,
        pool_price: float,
        scan_start_time: pd.Timestamp,
        times_1m: pd.DatetimeIndex,
        highs_1m: np.ndarray,
        lows_1m: np.ndarray,
        closes_1m: np.ndarray,
    ) -> Tuple[Optional[str], Optional[pd.Timestamp]]:
        start_idx = int(times_1m.searchsorted(scan_start_time, side="right"))
        if start_idx >= len(times_1m):
            return None, None

        if pool_type == "BSL":
            mask = (highs_1m[start_idx:] > pool_price) | (closes_1m[start_idx:] > pool_price)
            if not mask.any():
                return None, None
            first_idx = start_idx + int(np.argmax(mask))
            if closes_1m[first_idx] > pool_price:
                return "BREAK", times_1m[first_idx]
            return "SWEEP", times_1m[first_idx]
        else:
            mask = (lows_1m[start_idx:] < pool_price) | (closes_1m[start_idx:] < pool_price)
            if not mask.any():
                return None, None
            first_idx = start_idx + int(np.argmax(mask))
            if closes_1m[first_idx] < pool_price:
                return "BREAK", times_1m[first_idx]
            return "SWEEP", times_1m[first_idx]

    @staticmethod
    def _classify_session_pool_for_bar(pool_creation_time: pd.Timestamp, session_origin: str, current_bar_time: pd.Timestamp) -> Tuple[str, str]:
        cycle_start = (
            (current_bar_time - pd.Timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
            if current_bar_time.hour < 18
            else current_bar_time.replace(hour=18, minute=0, second=0, microsecond=0)
        )
        s_up = session_origin.upper()
        if pool_creation_time >= cycle_start and ("ASIA" in s_up or "LONDON" in s_up):
            return "SESSION", f"SESSION_{s_up}"
        return "1h", "1H_POOL"

    def _build_single_symbol_primitives(
        self,
        symbol: str,
        df_active: pd.DataFrame,
    ) -> Dict[str, Any]:
        df_htf_src = getattr(df_active, "_htf_lookback_df", None)
        if df_htf_src is None:
            df_htf_src = df_active.attrs.get("htf_lookback_df", df_active)

        sym_key = (
            symbol,
            len(df_active),
            df_active.index[0] if len(df_active) > 0 else None,
            df_active.index[-1] if len(df_active) > 0 else None,
            len(df_htf_src),
        )
        if sym_key in BacktestEngine._GLOBAL_SYMBOL_CACHE:
            return BacktestEngine._GLOBAL_SYMBOL_CACHE[sym_key]

        times = df_active.index
        highs = df_active["high"].to_numpy(dtype=float)
        lows = df_active["low"].to_numpy(dtype=float)
        closes = df_active["close"].to_numpy(dtype=float)

        htf_times = df_htf_src.index
        htf_highs = df_htf_src["high"].to_numpy(dtype=float)
        htf_lows = df_htf_src["low"].to_numpy(dtype=float)
        htf_closes = df_htf_src["close"].to_numpy(dtype=float)

        htf_run_max = np.maximum.accumulate(htf_highs)
        near_ath_mask = htf_highs >= (htf_run_max * 0.9955)
        near_ath_dates = {ts.date() for ts in htf_times[near_ath_mask]}

        swings_1m, all_pools_1m, fvgs_1m, dealing_ranges_1m, fib_ranges_1m = [], [], [], [], []
        structure_1m = pd.DataFrame()
        struct_events_1m, struct_1m_times = [], []
        swings_5m, fvgs_5m, dealing_ranges_5m, fib_ranges_5m = [], [], [], []
        struct_events_5m, struct_5m_times = [], []
        struct_df_by_tf: Dict[str, Tuple[pd.DataFrame, pd.DatetimeIndex]] = {}
        htf_target_catalog: List[Dict[str, Any]] = []
        major_session_catalog: List[Dict[str, Any]] = []
        macro_list: List[Dict[str, Any]] = []

        dedup_price_tol = 1.50 if symbol == "ES" else 6.00
        is_short_synthetic = len(df_active) < 25

        if len(df_active) >= 5:
            swings_1m = detect_swings(df_active)
            fvgs_1m = track_fvgs(df_active)
            structure_1m = detect_structure(
                df_active, self._shift_swings_for_structure(df_active, swings_1m)
            )
            struct_events_1m = self._extract_structure_events(
                structure_1m, tf_minutes=1, swings=swings_1m, fvgs=fvgs_1m
            )
            struct_1m_times = [e["close_time"] for e in struct_events_1m]
            struct_df_by_tf["1min"] = (structure_1m, structure_1m.index)

            macro_list = extract_session_liquidity(df_active)
            macro_by_time = self._group_macros(macro_list)
            swings_for_1m_liq = swings_1m if is_short_synthetic else []
            _, liq_engine_1m = track_liquidity(structure_1m, swings_for_1m_liq, macro_by_time, tolerance_ticks=2)
            all_pools_1m = liq_engine_1m.get_all_pools()

            if is_short_synthetic:
                dealing_ranges_1m = track_dealing_ranges(df_active, swings_1m)
                fib_ranges_1m = track_fibonacci_ranges(df_active, swings_1m)

            for p in all_pools_1m:
                if getattr(p, "source", "") == "Macro_Session":
                    p_price = self._get_pool_price(p)
                    s_orig = str(getattr(p, "session_origin", ""))
                    res_start = self._session_resolution_start(p.creation_time, s_orig)
                    res_type, res_ts = self._compute_1m_pool_resolution(
                        p.pool_type, p_price, res_start, times, highs, lows, closes
                    )
                    htf_target_catalog.append({
                        "pool": p,
                        "pool_type": p.pool_type,
                        "price": p_price,
                        "timeframe": "SESSION",
                        "session_origin": s_orig,
                        "target_type": f"SESSION_{s_orig.upper()}",
                        "tier": 1,
                        "is_major_exhaustion_pool": True,
                        "confirm_time": p.creation_time,
                        "first_touch_type": res_type,
                        "first_touch_time": res_ts,
                    })
                    if s_orig.upper() in ("ASIA", "LONDON"):
                        major_session_catalog.append({
                            "symbol": symbol,
                            "pool_type": p.pool_type,
                            "price": p_price,
                            "creation_time": p.creation_time,
                            "session_origin": s_orig,
                            "target_type": f"SESSION_{s_orig.upper()}",
                            "first_touch_type": res_type,
                            "first_touch_time": res_ts,
                        })

            if not is_short_synthetic:
                df_5m = resample_tv_timeframe(df_active, "5min")
                if len(df_5m) >= 3:
                    swings_5m = detect_swings(df_5m)
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
                    df_htf = resample_tv_timeframe(df_htf_src, tf_rule)
                    if len(df_htf) >= 3:
                        sw_for_liq = detect_htf_swings(df_htf)
                        sw_htf = sw_for_liq if tf_rule == "4h" else detect_swings(df_htf)
                        sw_for_struct = (
                            self._filter_1h_structural_swings(df_htf, sw_htf)
                            if tf_rule == "1h"
                            else sw_htf
                        )
                        fvgs_htf = track_fvgs(df_htf)
                        st_htf = detect_structure(
                            df_htf, self._shift_swings_for_structure(df_htf, sw_for_struct)
                        )
                        if tf_rule != "4h":
                            # On 1H, pass 3-bar pivots (`sw_for_liq`) filtered for major legs to `_extract_structure_events`
                            # so 3-bar range lows (like 09-30 20:00 @ 30724.75 on NQ) anchor the last FVG of the range!
                            sw_for_ifvg = (
                                self._filter_1h_structural_swings(df_htf, sw_for_liq)
                                if tf_rule == "1h"
                                else sw_for_struct
                            )
                            self._extract_structure_events(
                                st_htf, tf_minutes=tf_min, swings=sw_for_ifvg, fvgs=fvgs_htf
                            )
                        struct_df_by_tf[tf_rule] = (
                            st_htf,
                            st_htf.index + pd.Timedelta(minutes=tf_min - 1),
                        )

                        _, liq_htf = track_liquidity(st_htf, sw_for_liq, {}, tolerance_ticks=2)
                        confirm_lag = pd.Timedelta(minutes=tf_min * 2 - 1)
                        candle_close_lag = pd.Timedelta(minutes=tf_min - 1)

                        raw_htf_candidates: List[Tuple[LiquidityPool, float, pd.Timestamp]] = []
                        for p in liq_htf.get_all_pools():
                            p_price = self._get_pool_price(p)
                            raw_htf_candidates.append((p, p_price, p.creation_time))

                            scan_start_ts = p.creation_time + candle_close_lag
                            _, res_ts = self._compute_1m_pool_resolution(
                                p.pool_type, p_price, scan_start_ts, htf_times, htf_highs, htf_lows, htf_closes
                            )
                            if res_ts is not None:
                                matching_later_swings = [
                                    sw for sw in sw_for_liq
                                    if (
                                        (p.pool_type == "SSL" and sw["type"] == "SwingLow")
                                        or (p.pool_type == "BSL" and sw["type"] == "SwingHigh")
                                    )
                                    and p.creation_time < sw["time"] <= res_ts
                                    and abs(float(sw["price"]) - p_price) <= dedup_price_tol
                                ]
                                if matching_later_swings:
                                    latest_sw = matching_later_swings[-1]
                                    sw_p = float(latest_sw["price"])
                                    new_pool = LiquidityPool(
                                        pool_type=p.pool_type,
                                        lower_bound=sw_p - 0.25,
                                        upper_bound=sw_p + 0.25,
                                        creation_time=latest_sw["time"],
                                        touches=p.touches,
                                        source="Structural",
                                    )
                                    raw_htf_candidates.append((new_pool, sw_p, latest_sw["time"]))

                        for p, p_price, orig_creation_ts in raw_htf_candidates:
                            conf_ts = orig_creation_ts + confirm_lag
                            scan_start_ts = orig_creation_ts + candle_close_lag

                            res_type, res_ts = self._compute_1m_pool_resolution(
                                p.pool_type, p_price, scan_start_ts, htf_times, htf_highs, htf_lows, htf_closes
                            )
                            if res_ts is not None and res_ts < times[0]:
                                continue

                            is_duplicate = False
                            for existing in htf_target_catalog:
                                if existing["pool_type"] == p.pool_type:
                                    if (
                                        abs(existing["price"] - p_price) <= dedup_price_tol
                                        and abs((existing["pool"].creation_time - orig_creation_ts).total_seconds()) <= 12 * 3600
                                    ):
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
                            htf_target_catalog.append({
                                "pool": p,
                                "pool_type": p.pool_type,
                                "price": p_price,
                                "timeframe": tf_rule,
                                "session_origin": "",
                                "target_type": t_type,
                                "tier": effective_tier,
                                "is_major_exhaustion_pool": is_major,
                                "confirm_time": conf_ts,
                                "first_touch_type": res_type,
                                "first_touch_time": res_ts,
                            })

        far_future = pd.Timestamp("2200-01-01", tz="America/New_York")
        fvg_close_delta_5m = pd.Timedelta(minutes=4)

        swings_1m_times = [s["time"] for s in swings_1m]
        swings_5m_times = [s["time"] for s in swings_5m]

        fvg_1m_creation_times = [f.creation_time for f in fvgs_1m]
        fvg_1m_death_times = [
            min(
                (ev_ts for ev, ev_ts, _ in f.events if ev in ("FILLED_BY_CLOSE", "INVERTED")),
                default=far_future,
            )
            for f in fvgs_1m
        ]

        fvg_5m_ready_times = [f.creation_time + fvg_close_delta_5m for f in fvgs_5m]
        fvg_5m_death_times = [
            min(
                (ev_ts + fvg_close_delta_5m for ev, ev_ts, _ in f.events if ev in ("FILLED_BY_CLOSE", "INVERTED")),
                default=far_future,
            )
            for f in fvgs_5m
        ]

        dr_1m_creation_times = [dr.creation_time for dr in dealing_ranges_1m]
        fib_1m_extreme_times = [fib.extreme_time for fib in fib_ranges_1m]
        dr_5m_creation_times = [dr.creation_time for dr in dealing_ranges_5m]
        fib_5m_extreme_times = [fib.extreme_time for fib in fib_ranges_5m]

        res = {
            "swings_1m": swings_1m,
            "swings_1m_times": swings_1m_times,
            "all_pools_1m": all_pools_1m,
            "fvgs_1m": fvgs_1m,
            "fvg_1m_creation_times": fvg_1m_creation_times,
            "fvg_1m_death_times": fvg_1m_death_times,
            "dealing_ranges_1m": dealing_ranges_1m,
            "dr_1m_creation_times": dr_1m_creation_times,
            "fib_ranges_1m": fib_ranges_1m,
            "fib_1m_extreme_times": fib_1m_extreme_times,
            "struct_events_1m": struct_events_1m,
            "struct_1m_times": struct_1m_times,
            "swings_5m": swings_5m,
            "swings_5m_times": swings_5m_times,
            "fvgs_5m": fvgs_5m,
            "fvg_5m_ready_times": fvg_5m_ready_times,
            "fvg_5m_death_times": fvg_5m_death_times,
            "dealing_ranges_5m": dealing_ranges_5m,
            "dr_5m_creation_times": dr_5m_creation_times,
            "fib_ranges_5m": fib_ranges_5m,
            "fib_5m_extreme_times": fib_5m_extreme_times,
            "struct_events_5m": struct_events_5m,
            "struct_5m_times": struct_5m_times,
            "struct_df_by_tf": struct_df_by_tf,
            "htf_target_catalog": htf_target_catalog,
            "major_session_catalog": major_session_catalog,
            "macro_list": macro_list,
            "near_ath_dates": near_ath_dates,
            "htf_highs": htf_highs,
            "htf_lows": htf_lows,
            "htf_times": htf_times,
        }
        BacktestEngine._GLOBAL_SYMBOL_CACHE[sym_key] = res
        return res

    def _get_or_build_domain(
        self,
        df_primary: pd.DataFrame,
        df_secondary: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        cache_key = (
            self.symbol,
            len(df_primary),
            df_primary.index[0] if len(df_primary) > 0 else None,
            df_primary.index[-1] if len(df_primary) > 0 else None,
            len(df_secondary) if df_secondary is not None else 0,
        )
        if self._domain_cache_key == cache_key and self._domain_cache is not None:
            print(f"   ⚡ Reusing cached multi-TF ICT domain for {self.symbol} (0.00s)...")
            return self._domain_cache

        t0 = pytime.perf_counter()
        prim_data = self._build_single_symbol_primitives(self.symbol, df_primary)

        sec_symbol = "NQ" if self.symbol == "ES" else "ES"
        sec_struct_df_by_tf: Dict[str, Tuple[pd.DataFrame, pd.DatetimeIndex]] = {}
        secondary_major_catalog: List[Dict[str, Any]] = []
        smt_events: List[Dict[str, Any]] = []
        sec_near_ath_dates = set()

        if df_secondary is not None and not df_secondary.empty:
            sec_data = self._build_single_symbol_primitives(sec_symbol, df_secondary)
            sec_struct_df_by_tf = sec_data["struct_df_by_tf"]
            secondary_major_catalog = sec_data["major_session_catalog"]
            sec_near_ath_dates = sec_data["near_ath_dates"]

            smt_key = (
                len(df_primary),
                df_primary.index[0],
                df_primary.index[-1],
                len(df_secondary),
            )
            if smt_key in BacktestEngine._GLOBAL_SMT_CACHE:
                smt_events = BacktestEngine._GLOBAL_SMT_CACHE[smt_key]
            else:
                if self.symbol == "NQ":
                    df_es_base, df_nq_base = df_secondary, df_primary
                    es_data, nq_data = sec_data, prim_data
                else:
                    df_es_base, df_nq_base = df_primary, df_secondary
                    es_data, nq_data = prim_data, sec_data

                es_sw_1m = es_data["swings_1m"]
                nq_sw_1m = nq_data["swings_1m"]
                es_mac = es_data["macro_list"]
                nq_mac = nq_data["macro_list"]

                if len(df_es_base) >= 250:
                    t_start_1m, t_end_1m = dtime(8, 0), dtime(10, 45)
                    es_sw_1m_morning = [s for s in es_sw_1m if t_start_1m <= s["time"].time() <= t_end_1m]
                    nq_sw_1m_morning = [s for s in nq_sw_1m if t_start_1m <= s["time"].time() <= t_end_1m]
                else:
                    es_sw_1m_morning = es_sw_1m
                    nq_sw_1m_morning = nq_sw_1m

                smt_events.extend(
                    detect_smt_divergence(
                        es_sw_1m_morning, nq_sw_1m_morning, df_es_base, df_nq_base,
                        time_window_minutes=15, timeframe="1min"
                    )
                )
                smt_events.extend(
                    detect_session_smt_divergence(
                        es_mac, nq_mac, es_sw_1m, nq_sw_1m, df_es_base, df_nq_base
                    )
                )
                if len(df_es_base) >= 30 and len(df_nq_base) >= 30:
                    for smt_tf, smt_win in (("5min", 20), ("15min", 60), ("1h", 180), ("4h", 480)):
                        df_es_tf = resample_tv_timeframe(df_es_base, smt_tf)
                        df_nq_tf = resample_tv_timeframe(df_nq_base, smt_tf)
                        if len(df_es_tf) >= 3 and len(df_nq_tf) >= 3:
                            sw_es_tf = detect_htf_swings(df_es_tf)
                            sw_nq_tf = detect_htf_swings(df_nq_tf)
                            if smt_tf == "5min" and len(df_es_base) >= 250:
                                t_start_5m, t_end_5m = dtime(6, 0), dtime(10, 45)
                                sw_es_tf = [s for s in sw_es_tf if t_start_5m <= s["time"].time() <= t_end_5m]
                                sw_nq_tf = [s for s in sw_nq_tf if t_start_5m <= s["time"].time() <= t_end_5m]
                            smt_events.extend(
                                detect_smt_divergence(
                                    sw_es_tf,
                                    sw_nq_tf,
                                    df_es_tf,
                                    df_nq_tf,
                                    time_window_minutes=smt_win,
                                    timeframe=smt_tf,
                                    df_es_1m=df_es_base,
                                    df_nq_1m=df_nq_base,
                                )
                            )
                BacktestEngine._GLOBAL_SMT_CACHE[smt_key] = smt_events

        elapsed = pytime.perf_counter() - t0
        print(f"   ⚡ Precomputed multi-TF ICT domain for {self.symbol} in {elapsed:.2f}s (cached)...")

        bundle = {
            **prim_data,
            "sec_struct_df_by_tf": sec_struct_df_by_tf,
            "secondary_major_catalog": secondary_major_catalog,
            "sec_near_ath_dates": sec_near_ath_dates,
            "smt_events": smt_events,
        }
        self._domain_cache_key = cache_key
        self._domain_cache = bundle
        return bundle

    def run(
        self,
        df_primary: pd.DataFrame,
        strategy: Any,
        df_secondary: Optional[pd.DataFrame] = None,
        precompute_domain: bool = True,
        trade_start_date: Optional[str] = None,
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

        start_bar_idx = 0
        if trade_start_date:
            ts_start = pd.Timestamp(trade_start_date)
            ts_start = ts_start.tz_localize("America/New_York") if ts_start.tzinfo is None else ts_start.tz_convert("America/New_York")
            start_bar_idx = int(times.searchsorted(ts_start, side="left"))

        if precompute_domain and len(df_primary) >= 5:
            dom = self._get_or_build_domain(df_primary, df_secondary)
        else:
            dom = {}

        swings_1m = dom.get("swings_1m", [])
        swings_1m_times = dom.get("swings_1m_times", [])
        all_pools_1m = dom.get("all_pools_1m", [])
        fvgs_1m = dom.get("fvgs_1m", [])
        fvg_1m_creation_times = dom.get("fvg_1m_creation_times", [])
        fvg_1m_death_times = dom.get("fvg_1m_death_times", [])
        dealing_ranges_1m = dom.get("dealing_ranges_1m", [])
        dr_1m_creation_times = dom.get("dr_1m_creation_times", [])
        fib_ranges_1m = dom.get("fib_ranges_1m", [])
        fib_1m_extreme_times = dom.get("fib_1m_extreme_times", [])
        struct_events_1m = dom.get("struct_events_1m", [])
        struct_1m_times = dom.get("struct_1m_times", [])
        swings_5m = dom.get("swings_5m", [])
        swings_5m_times = dom.get("swings_5m_times", [])
        fvgs_5m = dom.get("fvgs_5m", [])
        fvg_5m_ready_times = dom.get("fvg_5m_ready_times", [])
        fvg_5m_death_times = dom.get("fvg_5m_death_times", [])
        dealing_ranges_5m = dom.get("dealing_ranges_5m", [])
        dr_5m_creation_times = dom.get("dr_5m_creation_times", [])
        fib_ranges_5m = dom.get("fib_ranges_5m", [])
        fib_5m_extreme_times = dom.get("fib_5m_extreme_times", [])
        struct_events_5m = dom.get("struct_events_5m", [])
        struct_5m_times = dom.get("struct_5m_times", [])
        struct_df_by_tf = dom.get("struct_df_by_tf", {})
        sec_struct_df_by_tf = dom.get("sec_struct_df_by_tf", {})
        htf_target_catalog = dom.get("htf_target_catalog", [])
        secondary_major_catalog = dom.get("secondary_major_catalog", [])
        smt_events = dom.get("smt_events", [])
        prim_near_ath_dates = dom.get("near_ath_dates", set())
        sec_near_ath_dates = dom.get("sec_near_ath_dates", set())

        confirm_delta_1m = pd.Timedelta(minutes=self.swing_confirmation_bars)
        confirm_delta_5m = pd.Timedelta(minutes=5 * 3 - 1)
        fvg_close_delta_5m = pd.Timedelta(minutes=4)
        recent_window = pd.Timedelta(minutes=25)
        max_pool_age = pd.Timedelta(hours=48)

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

        req_ny_win = bool(getattr(strategy, "require_ny_open_window", False))
        entry_start = getattr(strategy, "entry_start_time", None)
        entry_end = getattr(strategy, "entry_end_time", None)

        for w_i in range(start_bar_idx):
            gap_engine.process_candle(times[w_i], opens[w_i], closes[w_i], highs[w_i], lows[w_i])

        for i in range(start_bar_idx, len(df_primary)):
            t = times[i]
            o, h, l, c, v = opens[i], highs[i], lows[i], closes[i], volumes[i]

            gap_engine.process_candle(t, o, c, h, l)

            closed_trades = broker.process_candle(self.symbol, t, o, h, l, c)
            for tr in closed_trades:
                event_bus.publish(
                    Event("TRADE_CLOSED", t, {"symbol": self.symbol, "trade": tr})
                )

            if precompute_domain and t in sweeps_by_time:
                for p, lbl in sweeps_by_time[t]:
                    recent_sweeps_buffer.append((t, p, lbl))

            has_open_pos = portfolio.has_open_position(self.symbol)

            if precompute_domain and not has_open_pos and req_ny_win and entry_start is not None and entry_end is not None:
                bt = t.time()
                if not (entry_start <= bt <= entry_end):
                    portfolio.update_mark_to_market(t, {self.symbol: c})
                    continue

            confirmed_cutoff_1m = t - confirm_delta_1m
            confirmed_cutoff_5m = t - confirm_delta_5m

            if precompute_domain and has_open_pos and req_ny_win and entry_start is not None and entry_end is not None:
                bt = t.time()
                if not (entry_start <= bt <= entry_end):
                    end_sw1 = bisect.bisect_right(swings_1m_times, confirmed_cutoff_1m)
                    vis_swings_1m = swings_1m[:end_sw1]
                    idx_1m_st = bisect.bisect_right(struct_1m_times, t)
                    vis_struct_1m = struct_events_1m[max(0, idx_1m_st - 12):idx_1m_st]
                    hi_fvg1 = bisect.bisect_right(fvg_1m_creation_times, t)
                    lo_fvg1_recent = bisect.bisect_left(fvg_1m_creation_times, t - pd.Timedelta(minutes=15))
                    recent_fvgs_1m_all = fvgs_1m[lo_fvg1_recent:hi_fvg1]
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
                        recent_fvgs_all=recent_fvgs_1m_all,
                        structure_events_1m=vis_struct_1m,
                    )
                    strategy.on_bar(ctx, broker, portfolio)
                    portfolio.update_mark_to_market(t, {self.symbol: c})
                    continue

            if precompute_domain:
                if recent_sweeps_buffer:
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

                hi_fvg1 = bisect.bisect_right(fvg_1m_creation_times, t)
                lo_fvg1_recent = bisect.bisect_left(fvg_1m_creation_times, t - pd.Timedelta(minutes=15))
                vis_fvgs_1m = [
                    fvgs_1m[k] for k in range(hi_fvg1) if fvg_1m_death_times[k] > t
                ]
                recent_fvgs_1m_all = fvgs_1m[lo_fvg1_recent:hi_fvg1]

                active_dr_1m = None
                hi_dr1 = bisect.bisect_right(dr_1m_creation_times, confirmed_cutoff_1m)
                for k in range(hi_dr1 - 1, -1, -1):
                    dr = dealing_ranges_1m[k]
                    if dr.break_time is None or dr.break_time > t:
                        active_dr_1m = dr
                        break

                active_fib_1m = None
                hi_fib1 = bisect.bisect_right(fib_1m_extreme_times, confirmed_cutoff_1m)
                for k in range(hi_fib1 - 1, -1, -1):
                    fib = fib_ranges_1m[k]
                    if fib.break_time is None or fib.break_time > t:
                        active_fib_1m = fib
                        break

                vis_smt = [smt for smt in smt_events if is_smt_active_at(smt, t)]

                end_sw1 = bisect.bisect_right(swings_1m_times, confirmed_cutoff_1m)
                vis_swings_1m = swings_1m[:end_sw1]

                idx_1m_st = bisect.bisect_right(struct_1m_times, t)
                vis_struct_1m = struct_events_1m[max(0, idx_1m_st - 12):idx_1m_st]

                end_sw5 = bisect.bisect_right(swings_5m_times, confirmed_cutoff_5m)
                vis_swings_5m = swings_5m[:end_sw5]

                idx_5m_st = bisect.bisect_right(struct_5m_times, t)
                vis_struct_5m = struct_events_5m[max(0, idx_5m_st - 12):idx_5m_st]

                hi_fvg5 = bisect.bisect_right(fvg_5m_ready_times, t)
                vis_fvgs_5m = [
                    fvgs_5m[k] for k in range(hi_fvg5) if fvg_5m_death_times[k] > t
                ]

                active_dr_5m = None
                hi_dr5 = bisect.bisect_right(dr_5m_creation_times, confirmed_cutoff_5m)
                for k in range(hi_dr5 - 1, -1, -1):
                    dr = dealing_ranges_5m[k]
                    if dr.break_time is None or (dr.break_time + fvg_close_delta_5m) > t:
                        active_dr_5m = dr
                        break

                active_fib_5m = None
                hi_fib5 = bisect.bisect_right(fib_5m_extreme_times, confirmed_cutoff_5m)
                for k in range(hi_fib5 - 1, -1, -1):
                    fib = fib_ranges_5m[k]
                    if fib.break_time is None or (fib.break_time + fvg_close_delta_5m) > t:
                        active_fib_5m = fib
                        break

                htf_trends: Dict[str, str] = {}
                for tf_key in ("1min", "5min", "15min", "1h", "4h"):
                    if tf_key in struct_df_by_tf:
                        st_df, st_close_times = struct_df_by_tf[tf_key]
                        htf_trends[tf_key] = self._latest_trend_at(st_df, st_close_times, t)
                    else:
                        htf_trends[tf_key] = htf_trends.get("1min", "BULLISH")

                sec_htf_trends: Dict[str, str] = {}
                for tf_key in ("5min", "15min", "1h", "4h"):
                    if tf_key in sec_struct_df_by_tf:
                        s_df, s_close_times = sec_struct_df_by_tf[tf_key]
                        sec_htf_trends[tf_key] = self._latest_trend_at(s_df, s_close_times, t)
                    else:
                        sec_htf_trends[tf_key] = htf_trends.get(tf_key, "BULLISH")

                cur_date = t.date()
                if (cur_date in prim_near_ath_dates) or (cur_date in sec_near_ath_dates):
                    htf_trends["ath_regime"] = "ACTIVE"
                else:
                    htf_trends["ath_regime"] = "NONE"

                vis_targets: List[Dict[str, Any]] = []
                session_start = t.replace(hour=9, minute=30, second=0, microsecond=0)
                cycle_start = (
                    (t - pd.Timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
                    if t.hour < 18
                    else t.replace(hour=18, minute=0, second=0, microsecond=0)
                )

                taken_major_all: List[str] = []
                unique_major_pools_taken = set()
                htf_pools_taken_candidates: List[Tuple[str, str, float, pd.Timestamp]] = []
                min_creation_cutoff = t - max_pool_age

                lookback_48h_idx = max(0, i - 2880)
                high_48h = float(highs[lookback_48h_idx : i + 1].max())

                for item in htf_target_catalog:
                    if item["confirm_time"] > t:
                        continue
                    pool_obj: LiquidityPool = item["pool"]
                    p_price = float(item["price"])
                    res_ts = item.get("first_touch_time")
                    res_type = item.get("first_touch_type")

                    is_unbroken = (res_ts is None or res_ts > t)
                    is_unmitigated_historical = (
                        is_unbroken
                        and htf_trends.get("ath_regime") == "ACTIVE"
                        and pool_obj.pool_type == "BSL"
                        and item["timeframe"] in ("15min", "1h", "4h")
                        and p_price > high_48h
                    )

                    if pool_obj.creation_time < min_creation_cutoff and not is_unmitigated_historical:
                        continue

                    eff_tf = item["timeframe"]
                    eff_target_type = item["target_type"]
                    if eff_tf == "SESSION":
                        eff_tf, eff_target_type = self._classify_session_pool_for_bar(
                            pool_obj.creation_time, item.get("session_origin", ""), t
                        )

                    if is_unbroken:
                        vis_targets.append({
                            "pool_type": pool_obj.pool_type,
                            "price": p_price,
                            "target_type": eff_target_type,
                            "timeframe": eff_tf,
                            "tier": item["tier"],
                            "touches": pool_obj.touches,
                            "creation_time": pool_obj.creation_time,
                        })
                    elif session_start <= res_ts <= t and item.get("is_major_exhaustion_pool", False):
                        if item["timeframe"] == "SESSION" and pool_obj.creation_time < cycle_start:
                            continue
                        entry_str = f"[{self.symbol}] {eff_target_type} ({pool_obj.pool_type} @ {p_price:.2f} {res_type} at {res_ts.strftime('%H:%M')})"
                        taken_major_all.append(entry_str)
                        if eff_tf == "SESSION":
                            unique_major_pools_taken.add(("SESSION", eff_target_type, pool_obj.pool_type))
                        else:
                            htf_pools_taken_candidates.append(
                                (eff_target_type, pool_obj.pool_type, p_price, pool_obj.creation_time)
                            )

                for sec_item in secondary_major_catalog:
                    if sec_item["creation_time"] < cycle_start:
                        continue
                    res_ts = sec_item.get("first_touch_time")
                    res_type = sec_item.get("first_touch_type")
                    if res_ts is not None and session_start <= res_ts <= t:
                        entry_str = (
                            f"[{sec_item['symbol']}] {sec_item['target_type']} "
                            f"({sec_item['pool_type']} @ {sec_item['price']:.2f} {res_type} at {res_ts.strftime('%H:%M')})"
                        )
                        taken_major_all.append(entry_str)
                        unique_major_pools_taken.add(("SESSION", sec_item["target_type"], sec_item["pool_type"]))

                swept_session_sides = {side for kind, _, side in unique_major_pools_taken if kind == "SESSION"}
                for eff_t_type, p_side, p_pr, c_ts in htf_pools_taken_candidates:
                    if c_ts >= cycle_start and p_side in swept_session_sides:
                        continue
                    unique_major_pools_taken.add(("HTF", eff_t_type, p_side, round(p_pr, 1)))

                total_unique_major_taken = len(unique_major_pools_taken)
                if total_unique_major_taken == 0:
                    combined_exh = "LOW"
                elif total_unique_major_taken == 1:
                    combined_exh = "MEDIUM"
                else:
                    combined_exh = "HIGH"

                liq_taken_by_dir = {"LONG": combined_exh, "SHORT": combined_exh}
                session_taken_by_dir = {
                    "LONG": taken_major_all,
                    "SHORT": taken_major_all,
                }
                liq_taken_lvl = combined_exh
                taken_since_open = taken_major_all
                vis_gaps = [g for g in gap_engine.active_gaps if not g.is_filled]
            else:
                vis_sweeps, vis_sweep_details, vis_fvgs_1m, recent_fvgs_1m_all = [], [], [], []
                active_dr_1m, active_fib_1m = None, None
                vis_smt, vis_swings_1m, vis_struct_1m = [], [], []
                vis_swings_5m, vis_struct_5m, vis_fvgs_5m = [], [], []
                active_dr_5m, active_fib_5m = None, None
                htf_trends, sec_htf_trends = {}, {}
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
                recent_fvgs_all=recent_fvgs_1m_all,
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
                sec_htf_trends=sec_htf_trends,
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