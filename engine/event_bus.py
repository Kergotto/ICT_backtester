from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, Tuple
from datetime import time as dtime
import bisect
import hashlib
import pickle
import time as pytime
from pathlib import Path
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
    _augment_4h_leg_origin_swings,
    TIMEFRAME_HIERARCHY,
    SMT_CONFIRMED,
)
from engine.session_engine import extract_session_liquidity
from engine.portfolio import Portfolio, TradeRecord
from engine.broker import SimulatedBroker


_UNIT_TO_NS_MULT: Dict[str, int] = {
    "ns": 1,
    "us": 1_000,
    "ms": 1_000_000,
    "s": 1_000_000_000,
}
_FAR_FUTURE_NS: int = 1 << 62


def _ts_to_ns(ts: pd.Timestamp) -> int:
    return int(ts.as_unit("ns").value) if hasattr(ts, "as_unit") else int(ts.value)


def _index_to_ns(idx: pd.DatetimeIndex) -> np.ndarray:
    if len(idx) == 0:
        return np.empty(0, dtype=np.int64)
    return idx.as_unit("ns").asi8 if hasattr(idx, "as_unit") else idx.asi8


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
    _SOURCE_CODE_HASH_CACHE: Optional[str] = None

    def __init__(
        self,
        symbol: str = "ES",
        initial_capital: float = 100_000.0,
        commission_per_contract: float = 2.00,
        slippage_ticks: int = 1,
        swing_confirmation_bars: int = 2,
        enable_stage_profiling: bool = True,
        bypass_cache: bool = False,
        cache_dir: str | Path = "data/cache",
    ):
        self.symbol = symbol.upper()
        self.initial_capital = float(initial_capital)
        self.commission_per_contract = float(commission_per_contract)
        self.slippage_ticks = int(slippage_ticks)
        self.swing_confirmation_bars = int(swing_confirmation_bars)
        self.enable_stage_profiling = bool(enable_stage_profiling)
        self.bypass_cache = bool(bypass_cache)
        self.cache_dir = Path(cache_dir)
        self._domain_cache_key: Optional[Tuple[Any, ...]] = None
        self._domain_cache: Optional[Dict[str, Any]] = None

    @classmethod
    def _compute_source_code_hash(cls) -> str:
        if cls._SOURCE_CODE_HASH_CACHE is not None:
            return cls._SOURCE_CODE_HASH_CACHE

        project_root = Path(__file__).resolve().parent.parent
        files_to_hash: List[Path] = []
        for sub in ("core", "domain"):
            sub_dir = project_root / sub
            if sub_dir.exists():
                files_to_hash.extend(sorted(sub_dir.glob("*.py")))
        for rel in ("engine/session_engine.py", "engine/event_bus.py"):
            p = project_root / rel
            if p.exists():
                files_to_hash.append(p)

        h = hashlib.sha256()
        for p in sorted(set(files_to_hash), key=lambda x: x.as_posix()):
            h.update(p.relative_to(project_root).as_posix().encode("utf-8"))
            h.update(p.read_bytes())

        cls._SOURCE_CODE_HASH_CACHE = h.hexdigest()
        return cls._SOURCE_CODE_HASH_CACHE

    @staticmethod
    def _update_df_fingerprint(h: "hashlib._Hash", tag: str, df: Optional[pd.DataFrame]) -> None:
        if df is None:
            h.update(f"{tag}:NONE".encode("utf-8"))
            return
        n = len(df)
        t0 = _ts_to_ns(df.index[0]) if n > 0 else 0
        t1 = _ts_to_ns(df.index[-1]) if n > 0 else 0
        tz_str = str(getattr(df.index, "tz", None))
        h.update(f"{tag}:{n}:{t0}:{t1}:{tz_str}".encode("utf-8"))
        if n > 0 and "close" in df.columns:
            h.update(df["close"].to_numpy(dtype=np.float64, copy=False).tobytes())

    def _compute_disk_cache_hash(
        self,
        df_primary: pd.DataFrame,
        df_secondary: Optional[pd.DataFrame],
    ) -> str:
        project_root = Path(__file__).resolve().parent.parent
        h = hashlib.sha256()
        h.update(b"domain_cache_v1")
        h.update(f"|sym={self.symbol}|sw_conf={self.swing_confirmation_bars}".encode("utf-8"))
        h.update(self._compute_source_code_hash().encode("utf-8"))

        for pq_name in ("es_1m_databento_badj.parquet", "nq_1m_databento_badj.parquet"):
            pq_path = project_root / "data" / pq_name
            if pq_path.exists():
                st = pq_path.stat()
                h.update(f"|pq:{pq_name}:{st.st_size}:{st.st_mtime_ns}".encode("utf-8"))

        self._update_df_fingerprint(h, "prim", df_primary)
        prim_lb = getattr(df_primary, "_htf_lookback_df", None)
        if prim_lb is None:
            prim_lb = df_primary.attrs.get("htf_lookback_df")
        self._update_df_fingerprint(h, "prim_lb", prim_lb if prim_lb is not df_primary else None)

        self._update_df_fingerprint(h, "sec", df_secondary)
        if df_secondary is not None:
            sec_lb = getattr(df_secondary, "_htf_lookback_df", None)
            if sec_lb is None:
                sec_lb = df_secondary.attrs.get("htf_lookback_df")
            self._update_df_fingerprint(h, "sec_lb", sec_lb if sec_lb is not df_secondary else None)

        return h.hexdigest()[:16]

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
        idx_ns = _index_to_ns(df_tf.index)
        idx_list = df_tf.index.tolist()
        n = len(idx_list)
        shifted: List[Dict[str, Any]] = []
        for s in swings:
            pos = int(np.searchsorted(idx_ns, _ts_to_ns(s["time"])))
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
    def _apply_4h_pullback_structure_breaks(df_4h: pd.DataFrame) -> None:
        if df_4h.empty or "trend" not in df_4h.columns or len(df_4h) < 4:
            return

        opens = df_4h["open"].to_numpy(dtype=float)
        highs = df_4h["high"].to_numpy(dtype=float)
        lows = df_4h["low"].to_numpy(dtype=float)
        closes = df_4h["close"].to_numpy(dtype=float)
        trends = df_4h["trend"].to_numpy(dtype=int).copy()
        choch_bull = df_4h["choch_bullish"].to_numpy(dtype=bool)
        bos_bull = df_4h["bos_bullish"].to_numpy(dtype=bool)

        override_bearish = False
        anchor_high = 0.0

        for k in range(3, len(df_4h)):
            if override_bearish:
                if closes[k] > anchor_high or choch_bull[k] or bos_bull[k]:
                    override_bearish = False
                else:
                    trends[k] = -1

            if not override_bearish and trends[k] == 1:
                if (
                        highs[k - 2] > highs[k - 3]
                        and lows[k - 2] > lows[k - 3]
                        and lows[k - 3] < lows[k - 1] < lows[k - 2]
                        and closes[k - 1] > opens[k - 1]
                        and highs[k - 1] <= highs[k - 2]
                        and highs[k] <= highs[k - 2]
                        and closes[k] < lows[k - 1]
                ):
                    override_bearish = True
                    anchor_high = float(max(highs[k - 2], highs[k - 1]))
                    trends[k] = -1

        df_4h["trend"] = trends

    @staticmethod
    def _filter_4h_structural_swings(
            df_4h: pd.DataFrame, swings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if df_4h.empty or len(df_4h) < 4 or not swings:
            return swings

        opens = df_4h["open"].to_numpy(dtype=float)
        highs = df_4h["high"].to_numpy(dtype=float)
        lows = df_4h["low"].to_numpy(dtype=float)
        closes = df_4h["close"].to_numpy(dtype=float)
        times_ns = _index_to_ns(df_4h.index)

        filtered: List[Dict[str, Any]] = []
        for s in swings:
            s_ns = _ts_to_ns(s["time"])
            idx = int(np.searchsorted(times_ns, s_ns))
            if 1 <= idx < len(df_4h) - 2 and times_ns[idx] == s_ns:
                if s["type"] == "SwingHigh":
                    if (
                            closes[idx - 1] > opens[idx - 1]
                            and closes[idx] > opens[idx]
                            and closes[idx + 1] > opens[idx + 1]
                            and closes[idx + 2] > opens[idx + 2]
                            and closes[idx - 1] < closes[idx] < closes[idx + 1] < closes[idx + 2]
                            and lows[idx - 1] < lows[idx] < lows[idx + 1] < lows[idx + 2]
                            and highs[idx + 2] > highs[idx]
                    ):
                        continue
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
        times_ns = _index_to_ns(times)
        times_ns_list = times_ns.tolist()
        times_list = times.tolist()
        n_bars = len(df_struct)

        swing_lows = [s for s in (swings or []) if s["type"] == "SwingLow"]
        swing_highs = [s for s in (swings or []) if s["type"] == "SwingHigh"]
        swing_low_times_ns = [_ts_to_ns(s["time"]) for s in swing_lows]
        swing_high_times_ns = [_ts_to_ns(s["time"]) for s in swing_highs]
        swing_low_prices = [float(s["price"]) for s in swing_lows]
        swing_high_prices = [float(s["price"]) for s in swing_highs]

        bisi_fvgs = [f for f in (fvgs or []) if f.fvg_type == "BISI"]
        sibi_fvgs = [f for f in (fvgs or []) if f.fvg_type == "SIBI"]
        bisi_creation_times_ns = [_ts_to_ns(f.creation_time) for f in bisi_fvgs]
        sibi_creation_times_ns = [_ts_to_ns(f.creation_time) for f in sibi_fvgs]
        bisi_bottoms = [f.bottom for f in bisi_fvgs]
        sibi_tops = [f.top for f in sibi_fvgs]

        bisi_inv_idx: List[int] = []
        bisi_start_idx: List[int] = []
        for f_i, f in enumerate(bisi_fvgs):
            start_k = bisect.bisect_right(times_ns_list, bisi_creation_times_ns[f_i])
            bisi_start_idx.append(start_k)
            inv_k = 10**9
            if start_k < n_bars:
                inv_mask = closes[start_k:] < f.bottom
                if inv_mask.any():
                    inv_k = start_k + int(np.argmax(inv_mask))
            bisi_inv_idx.append(inv_k)

        sibi_inv_idx: List[int] = []
        sibi_start_idx: List[int] = []
        for f_i, f in enumerate(sibi_fvgs):
            start_k = bisect.bisect_right(times_ns_list, sibi_creation_times_ns[f_i])
            sibi_start_idx.append(start_k)
            inv_k = 10**9
            if start_k < n_bars:
                inv_mask = closes[start_k:] > f.top
                if inv_mask.any():
                    inv_k = start_k + int(np.argmax(inv_mask))
            sibi_inv_idx.append(inv_k)

        consumed_bull_legs = set()
        consumed_bear_legs = set()
        override_trend: Optional[int] = None
        last_std_trend = 0

        bisect_left = bisect.bisect_left
        has_bisi = len(bisi_fvgs) > 0
        has_sibi = len(sibi_fvgs) > 0
        lt_60 = tf_minutes < 60

        for idx in range(n_bars):
            t_bar_ns = times_ns_list[idx]
            std_t = int(trends[idx])

            if std_t != last_std_trend and std_t != 0:
                override_trend = None
                last_std_trend = std_t

            if override_trend is not None:
                trends[idx] = override_trend

            has_bull_ev = False
            has_bear_ev = False

            if choch_bull[idx]:
                t_bar = times_list[idx]
                lvl = float(sh_arr[idx]) if not np.isnan(sh_arr[idx]) else float(closes[idx])
                events.append({
                    "time": t_bar,
                    "close_time": t_bar + bar_close_offset,
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
                t_bar = times_list[idx]
                lvl = float(sh_arr[idx]) if not np.isnan(sh_arr[idx]) else float(closes[idx])
                events.append({
                    "time": t_bar,
                    "close_time": t_bar + bar_close_offset,
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
                t_bar = times_list[idx]
                lvl = float(sl_arr[idx]) if not np.isnan(sl_arr[idx]) else float(closes[idx])
                events.append({
                    "time": t_bar,
                    "close_time": t_bar + bar_close_offset,
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
                t_bar = times_list[idx]
                lvl = float(sl_arr[idx]) if not np.isnan(sl_arr[idx]) else float(closes[idx])
                events.append({
                    "time": t_bar,
                    "close_time": t_bar + bar_close_offset,
                    "kind": "BOS",
                    "label": f"Bearish BOS (< {lvl:.2f})",
                    "direction": "BEARISH",
                    "broken_level": lvl,
                    "close_price": float(closes[idx]),
                })
                has_bear_ev = True
                override_trend = -1
                trends[idx] = -1

            if has_bisi and not has_bear_ev:
                sl_pos = bisect_left(swing_low_times_ns, t_bar_ns) - 1
                if sl_pos >= 0:
                    leg_low_ns = swing_low_times_ns[sl_pos]
                    if leg_low_ns not in consumed_bull_legs:
                        leg_low_price = swing_low_prices[sl_pos]
                        lo_f = bisect_left(bisi_creation_times_ns, leg_low_ns)
                        hi_f = bisect_left(bisi_creation_times_ns, t_bar_ns)
                        if lo_f < hi_f:
                            min_bisi_bottom = float("inf")
                            found_uninv = False
                            for f_i in range(lo_f, hi_f):
                                f_bot = bisi_bottoms[f_i]
                                if f_bot < leg_low_price:
                                    continue
                                if bisi_inv_idx[f_i] < idx:
                                    continue
                                if lt_60:
                                    f_start = bisi_start_idx[f_i]
                                    if f_start < idx and float(lows[f_start:idx].min()) < f_bot:
                                        continue
                                found_uninv = True
                                if f_bot < min_bisi_bottom:
                                    min_bisi_bottom = f_bot

                            if found_uninv and closes[idx] < min_bisi_bottom:
                                t_bar = times_list[idx]
                                events.append({
                                    "time": t_bar,
                                    "close_time": t_bar + bar_close_offset,
                                    "kind": "IFVG_BOS",
                                    "label": f"Bearish IFVG_BOS (< {min_bisi_bottom:.2f})",
                                    "direction": "BEARISH",
                                    "broken_level": float(min_bisi_bottom),
                                    "close_price": float(closes[idx]),
                                })
                                consumed_bull_legs.add(leg_low_ns)
                                override_trend = -1
                                trends[idx] = -1

            if has_sibi and not has_bull_ev:
                sh_pos = bisect_left(swing_high_times_ns, t_bar_ns) - 1
                if sh_pos >= 0:
                    leg_high_ns = swing_high_times_ns[sh_pos]
                    if leg_high_ns not in consumed_bear_legs:
                        leg_high_price = swing_high_prices[sh_pos]
                        lo_f = bisect_left(sibi_creation_times_ns, leg_high_ns)
                        hi_f = bisect_left(sibi_creation_times_ns, t_bar_ns)
                        if lo_f < hi_f:
                            max_sibi_top = float("-inf")
                            found_uninv = False
                            for f_i in range(lo_f, hi_f):
                                f_top = sibi_tops[f_i]
                                if f_top > leg_high_price:
                                    continue
                                if sibi_inv_idx[f_i] < idx:
                                    continue
                                if lt_60:
                                    f_start = sibi_start_idx[f_i]
                                    if f_start < idx and float(highs[f_start:idx].max()) > f_top:
                                        continue
                                found_uninv = True
                                if f_top > max_sibi_top:
                                    max_sibi_top = f_top

                            if found_uninv and closes[idx] > max_sibi_top:
                                t_bar = times_list[idx]
                                events.append({
                                    "time": t_bar,
                                    "close_time": t_bar + bar_close_offset,
                                    "kind": "IFVG_BOS",
                                    "label": f"Bullish IFVG_BOS (> {max_sibi_top:.2f})",
                                    "direction": "BULLISH",
                                    "broken_level": float(max_sibi_top),
                                    "close_price": float(closes[idx]),
                                })
                                consumed_bear_legs.add(leg_high_ns)
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
    def _prepare_fast_trend_lookup(
        struct_df_by_tf: Dict[str, Tuple[pd.DataFrame, pd.DatetimeIndex]],
    ) -> Dict[str, Tuple[List[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]:
        prepared: Dict[str, Tuple[List[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = {}
        for tf_key, (st_df, st_close_times) in struct_df_by_tf.items():
            if st_df.empty or len(st_close_times) == 0:
                prepared[tf_key] = ([], np.empty(0, dtype=int), np.empty(0), np.empty(0), np.empty(0), np.empty(0), 0)
            else:
                prepared[tf_key] = (
                    _index_to_ns(st_close_times).tolist(),
                    st_df["trend"].to_numpy(dtype=int, copy=False),
                    st_df["current_swing_high"].to_numpy(dtype=float, copy=False),
                    st_df["current_swing_low"].to_numpy(dtype=float, copy=False),
                    st_df["close"].to_numpy(dtype=float, copy=False),
                    st_df["open"].to_numpy(dtype=float, copy=False),
                    len(st_close_times),
                )
        return prepared

    @staticmethod
    def _latest_trend_at_fast(
        entry: Tuple[List[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int],
        t_ns: int,
    ) -> str:
        close_ns_list, trends_arr, sh_arr, sl_arr, closes_arr, opens_arr, n_rows = entry
        if n_rows == 0:
            return "NEUTRAL"

        idx = bisect.bisect_right(close_ns_list, t_ns) - 1
        if idx < 0:
            idx = 0

        trend_val = int(trends_arr[idx])
        if trend_val == 1:
            return "BULLISH"
        if trend_val == -1:
            return "BEARISH"

        sh_val = float(sh_arr[idx])
        sl_val = float(sl_arr[idx])
        closed_bar_price = float(closes_arr[idx])

        if not np.isnan(sh_val) and not np.isnan(sl_val):
            mid = (sh_val + sl_val) * 0.5
            return "BULLISH" if closed_bar_price >= mid else "BEARISH"

        ref_idx = max(0, idx - 3)
        return "BULLISH" if closed_bar_price >= float(opens_arr[ref_idx]) else "BEARISH"

    @staticmethod
    def _session_resolution_start(creation_time: pd.Timestamp, session_origin: str) -> pd.Timestamp:
        s_up = session_origin.upper()
        if "ASIA" in s_up:
            if creation_time.hour >= 18:
                return (creation_time + pd.Timedelta(days=1)).replace(hour=3, minute=0, second=0, microsecond=0)
            return creation_time.replace(hour=3, minute=0, second=0, microsecond=0)
        if "LONDON" in s_up:
            return creation_time.replace(hour=9, minute=0, second=0, microsecond=0)
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
        times_1m_ns: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[str], Optional[pd.Timestamp]]:
        if times_1m_ns is not None:
            start_idx = int(np.searchsorted(times_1m_ns, _ts_to_ns(scan_start_time), side="right"))
        else:
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

        t_prim_start = pytime.perf_counter()
        st = {
            "resample": 0.0,
            "swings": 0.0,
            "fvg": 0.0,
            "structure": 0.0,
            "liquidity": 0.0,
            "eq_fib": 0.0,
            "index_prep": 0.0,
        }

        times = df_active.index
        times_ns = _index_to_ns(times)
        highs = df_active["high"].to_numpy(dtype=float)
        lows = df_active["low"].to_numpy(dtype=float)
        closes = df_active["close"].to_numpy(dtype=float)

        htf_times = df_htf_src.index
        htf_times_ns = _index_to_ns(htf_times)
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
            _t = pytime.perf_counter()
            swings_1m = detect_swings(df_active)
            st["swings"] += pytime.perf_counter() - _t

            _t = pytime.perf_counter()
            fvgs_1m = track_fvgs(df_active)
            st["fvg"] += pytime.perf_counter() - _t

            _t = pytime.perf_counter()
            structure_1m = detect_structure(
                df_active, self._shift_swings_for_structure(df_active, swings_1m)
            )
            struct_events_1m = self._extract_structure_events(
                structure_1m, tf_minutes=1, swings=swings_1m, fvgs=fvgs_1m
            )
            struct_1m_times = [e["close_time"] for e in struct_events_1m]
            struct_df_by_tf["1min"] = (structure_1m, structure_1m.index)
            st["structure"] += pytime.perf_counter() - _t

            _t = pytime.perf_counter()
            macro_list = extract_session_liquidity(df_active)
            macro_by_time = self._group_macros(macro_list)
            swings_for_1m_liq = swings_1m if is_short_synthetic else []
            _, liq_engine_1m = track_liquidity(structure_1m, swings_for_1m_liq, macro_by_time, tolerance_ticks=2)
            all_pools_1m = liq_engine_1m.get_all_pools()
            st["liquidity"] += pytime.perf_counter() - _t

            if is_short_synthetic:
                _t = pytime.perf_counter()
                dealing_ranges_1m = track_dealing_ranges(df_active, swings_1m)
                fib_ranges_1m = track_fibonacci_ranges(df_active, swings_1m)
                st["eq_fib"] += pytime.perf_counter() - _t

            _t = pytime.perf_counter()
            for p in all_pools_1m:
                if getattr(p, "source", "") == "Macro_Session":
                    p_price = self._get_pool_price(p)
                    s_orig = str(getattr(p, "session_origin", ""))
                    res_start = self._session_resolution_start(p.creation_time, s_orig)
                    res_type, res_ts = self._compute_1m_pool_resolution(
                        p.pool_type, p_price, res_start, times, highs, lows, closes, times_1m_ns=times_ns
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
            st["liquidity"] += pytime.perf_counter() - _t

            if not is_short_synthetic:
                _t = pytime.perf_counter()
                df_5m = resample_tv_timeframe(df_active, "5min")
                st["resample"] += pytime.perf_counter() - _t

                if len(df_5m) >= 3:
                    _t = pytime.perf_counter()
                    swings_5m = detect_swings(df_5m)
                    st["swings"] += pytime.perf_counter() - _t

                    _t = pytime.perf_counter()
                    fvgs_5m = track_fvgs(df_5m)
                    st["fvg"] += pytime.perf_counter() - _t

                    _t = pytime.perf_counter()
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
                    st["structure"] += pytime.perf_counter() - _t

                    _t = pytime.perf_counter()
                    dealing_ranges_5m = track_dealing_ranges(df_5m, swings_5m)
                    fib_ranges_5m = track_fibonacci_ranges(df_5m, swings_5m)
                    st["eq_fib"] += pytime.perf_counter() - _t

                for tf_rule, tf_min, tier_num, is_major in (
                        ("4h", 240, 1, True),
                        ("1h", 60, 1, True),
                        ("15min", 15, 2, False),
                ):
                    _t = pytime.perf_counter()
                    df_htf = resample_tv_timeframe(df_htf_src, tf_rule)
                    st["resample"] += pytime.perf_counter() - _t

                    if len(df_htf) >= 3:
                        _t = pytime.perf_counter()
                        base_htf_sw = detect_htf_swings(df_htf)
                        if tf_rule == "4h":
                            aug_4h = _augment_4h_leg_origin_swings(df_htf, base_htf_sw)
                            sw_for_liq = [
                                s for s in aug_4h
                                if s in base_htf_sw or s["time"] < times[0]
                            ]
                        else:
                            sw_for_liq = base_htf_sw
                        sw_htf = base_htf_sw if tf_rule == "4h" else detect_swings(df_htf)
                        if tf_rule == "1h":
                            sw_for_struct = self._filter_1h_structural_swings(df_htf, sw_htf)
                        elif tf_rule == "4h":
                            sw_for_struct = self._filter_4h_structural_swings(df_htf, sw_htf)
                        else:
                            sw_for_struct = sw_htf
                        st["swings"] += pytime.perf_counter() - _t

                        _t = pytime.perf_counter()
                        fvgs_htf = track_fvgs(df_htf)
                        st["fvg"] += pytime.perf_counter() - _t

                        _t = pytime.perf_counter()
                        st_htf = detect_structure(
                            df_htf, self._shift_swings_for_structure(df_htf, sw_for_struct)
                        )
                        if tf_rule != "4h":
                            sw_for_ifvg = (
                                self._filter_1h_structural_swings(df_htf, sw_for_liq)
                                if tf_rule == "1h"
                                else sw_for_struct
                            )
                            self._extract_structure_events(
                                st_htf, tf_minutes=tf_min, swings=sw_for_ifvg, fvgs=fvgs_htf
                            )
                        else:
                            self._apply_4h_pullback_structure_breaks(st_htf)
                        struct_df_by_tf[tf_rule] = (
                            st_htf,
                            st_htf.index + pd.Timedelta(minutes=tf_min - 1),
                        )
                        st["structure"] += pytime.perf_counter() - _t

                        _t = pytime.perf_counter()
                        _, liq_htf = track_liquidity(st_htf, sw_for_liq, {}, tolerance_ticks=2)
                        confirm_lag = pd.Timedelta(minutes=tf_min * 2 - 1)
                        candle_close_lag = pd.Timedelta(minutes=tf_min - 1)

                        # Pre-split swings by type for fast later-swing matching
                        sw_liq_lows = [sw for sw in sw_for_liq if sw["type"] == "SwingLow"]
                        sw_liq_highs = [sw for sw in sw_for_liq if sw["type"] == "SwingHigh"]
                        sw_liq_low_ns = [_ts_to_ns(sw["time"]) for sw in sw_liq_lows]
                        sw_liq_high_ns = [_ts_to_ns(sw["time"]) for sw in sw_liq_highs]

                        raw_htf_candidates: List[
                            Tuple[LiquidityPool, float, pd.Timestamp, bool, Optional[Tuple[Optional[str], Optional[pd.Timestamp]]]]
                        ] = []
                        for p in liq_htf.get_all_pools():
                            p_price = self._get_pool_price(p)
                            scan_start_ts = p.creation_time + candle_close_lag
                            res_type_base, res_ts_base = self._compute_1m_pool_resolution(
                                p.pool_type, p_price, scan_start_ts, htf_times, htf_highs, htf_lows, htf_closes,
                                times_1m_ns=htf_times_ns,
                            )
                            raw_htf_candidates.append((p, p_price, p.creation_time, False, (res_type_base, res_ts_base)))

                            if res_ts_base is not None:
                                p_cre_ns = _ts_to_ns(p.creation_time)
                                res_base_ns = _ts_to_ns(res_ts_base)
                                if p.pool_type == "SSL":
                                    lo_sw = bisect.bisect_right(sw_liq_low_ns, p_cre_ns)
                                    hi_sw = bisect.bisect_right(sw_liq_low_ns, res_base_ns)
                                    sw_slice = sw_liq_lows[lo_sw:hi_sw]
                                else:
                                    lo_sw = bisect.bisect_right(sw_liq_high_ns, p_cre_ns)
                                    hi_sw = bisect.bisect_right(sw_liq_high_ns, res_base_ns)
                                    sw_slice = sw_liq_highs[lo_sw:hi_sw]

                                latest_sw = None
                                for sw in reversed(sw_slice):
                                    if abs(float(sw["price"]) - p_price) <= dedup_price_tol:
                                        latest_sw = sw
                                        break
                                if latest_sw is not None:
                                    sw_p = float(latest_sw["price"])
                                    new_pool = LiquidityPool(
                                        pool_type=p.pool_type,
                                        lower_bound=sw_p - 0.25,
                                        upper_bound=sw_p + 0.25,
                                        creation_time=latest_sw["time"],
                                        touches=p.touches,
                                        source="Structural",
                                    )
                                    raw_htf_candidates.append((new_pool, sw_p, latest_sw["time"], False, None))

                        # Include 15m pullback-bounce highs (e.g. Oct 8 02:00 @ 7845.25 -> 7845.00 pool)
                        if tf_rule == "15min" and len(df_htf) >= 5:
                            h_15 = df_htf["high"].to_numpy(dtype=float)
                            l_15 = df_htf["low"].to_numpy(dtype=float)
                            o_15 = df_htf["open"].to_numpy(dtype=float)
                            c_15 = df_htf["close"].to_numpy(dtype=float)
                            t_15 = df_htf.index
                            for k_15 in range(1, len(df_htf) - 2):
                                if t_15[k_15] < times[0]:
                                    continue
                                if (
                                        c_15[k_15 - 1] < o_15[k_15 - 1]
                                        and h_15[k_15] > c_15[k_15 - 1]
                                        and h_15[k_15] > h_15[k_15 + 1]
                                        and h_15[k_15] > h_15[k_15 + 2]
                                ):
                                    sw_p = float(h_15[k_15])
                                    new_pool = LiquidityPool(
                                        pool_type="BSL",
                                        lower_bound=sw_p - 0.25,
                                        upper_bound=sw_p + 0.25,
                                        creation_time=t_15[k_15],
                                        touches=1,
                                        source="Structural",
                                    )
                                    raw_htf_candidates.append((new_pool, sw_p, t_15[k_15], True, None))
                                if (
                                        c_15[k_15 - 1] > o_15[k_15 - 1]
                                        and l_15[k_15] < c_15[k_15 - 1]
                                        and l_15[k_15] < l_15[k_15 + 1]
                                        and l_15[k_15] < l_15[k_15 + 2]
                                ):
                                    sw_p = float(l_15[k_15]) + 0.25
                                    new_pool = LiquidityPool(
                                        pool_type="SSL",
                                        lower_bound=sw_p - 0.25,
                                        upper_bound=sw_p + 0.25,
                                        creation_time=t_15[k_15],
                                        touches=1,
                                        source="Structural",
                                    )
                                    raw_htf_candidates.append((new_pool, sw_p, t_15[k_15], True, None))

                        for p, p_price, orig_creation_ts, is_tp_only, cached_res in raw_htf_candidates:
                            conf_ts = orig_creation_ts + confirm_lag
                            if cached_res is not None:
                                res_type, res_ts = cached_res
                            else:
                                scan_start_ts = orig_creation_ts + candle_close_lag
                                res_type, res_ts = self._compute_1m_pool_resolution(
                                    p.pool_type, p_price, scan_start_ts, htf_times, htf_highs, htf_lows, htf_closes,
                                    times_1m_ns=htf_times_ns,
                                )
                            if res_ts is not None and res_ts < times[0]:
                                continue

                            is_duplicate = False
                            for existing in htf_target_catalog:
                                if existing["pool_type"] == p.pool_type:
                                    if (
                                            abs(existing["price"] - p_price) <= dedup_price_tol
                                            and abs((existing[
                                                         "pool"].creation_time - orig_creation_ts).total_seconds()) <= 12 * 3600
                                    ):
                                        if not is_tp_only or existing.get("first_touch_time") is None:
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
                                "is_target_only_pool": is_tp_only,
                                "confirm_time": conf_ts,
                                "first_touch_type": res_type,
                                "first_touch_time": res_ts,
                            })
                        st["liquidity"] += pytime.perf_counter() - _t

        _t = pytime.perf_counter()
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
        st["index_prep"] += pytime.perf_counter() - _t

        if self.enable_stage_profiling and len(df_active) >= 100:
            prim_total = pytime.perf_counter() - t_prim_start
            print(
                f"     ⏱️  [Primitives {symbol}] total={prim_total:.2f}s | "
                f"resample={st['resample']:.2f}s | swings={st['swings']:.2f}s | fvg={st['fvg']:.2f}s | "
                f"structure={st['structure']:.2f}s | liquidity={st['liquidity']:.2f}s | "
                f"eq_fib={st['eq_fib']:.2f}s | index_prep={st['index_prep']:.2f}s"
            )

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

        disk_cache_file: Optional[Path] = None
        if not self.bypass_cache and len(df_primary) >= 100:
            disk_hash = self._compute_disk_cache_hash(df_primary, df_secondary)
            disk_cache_file = self.cache_dir / f"domain_{self.symbol}_{disk_hash}.pkl"
            if disk_cache_file.exists():
                try:
                    t_load = pytime.perf_counter()
                    with open(disk_cache_file, "rb") as f:
                        bundle = pickle.load(f)
                    load_s = pytime.perf_counter() - t_load
                    print(f"   ⚡ Loaded cached multi-TF ICT domain for {self.symbol} from disk ({load_s:.2f}s)...")
                    self._domain_cache_key = cache_key
                    self._domain_cache = bundle
                    return bundle
                except Exception as e:
                    print(f"   ⚠️ Disk cache load failed for {self.symbol}: {e}. Rebuilding...")

        t0 = pytime.perf_counter()
        _tp = pytime.perf_counter()
        prim_data = self._build_single_symbol_primitives(self.symbol, df_primary)
        t_prim_s = pytime.perf_counter() - _tp

        sec_symbol = "NQ" if self.symbol == "ES" else "ES"
        sec_struct_df_by_tf: Dict[str, Tuple[pd.DataFrame, pd.DatetimeIndex]] = {}
        secondary_major_catalog: List[Dict[str, Any]] = []
        smt_events: List[Dict[str, Any]] = []
        sec_near_ath_dates = set()
        t_sec_s = 0.0
        smt_prof: Dict[str, float] = {}

        if df_secondary is not None and not df_secondary.empty:
            _ts = pytime.perf_counter()
            sec_data = self._build_single_symbol_primitives(sec_symbol, df_secondary)
            t_sec_s = pytime.perf_counter() - _ts

            sec_struct_df_by_tf = sec_data["struct_df_by_tf"]
            secondary_major_catalog = sec_data["major_session_catalog"]
            sec_near_ath_dates = sec_data["near_ath_dates"]

            if self.symbol == "NQ":
                df_es_base, df_nq_base = df_secondary, df_primary
                es_data, nq_data = sec_data, prim_data
            else:
                df_es_base, df_nq_base = df_primary, df_secondary
                es_data, nq_data = prim_data, sec_data

            # Canonical (ES, NQ) key so ES and NQ share the identical SMT event set without recomputing
            smt_key = (
                len(df_es_base),
                df_es_base.index[0],
                df_es_base.index[-1],
                len(df_nq_base),
                df_nq_base.index[0],
                df_nq_base.index[-1],
            )
            if smt_key in BacktestEngine._GLOBAL_SMT_CACHE:
                smt_events = BacktestEngine._GLOBAL_SMT_CACHE[smt_key]
                smt_prof["cache_hit"] = 1.0
            else:
                smt_prof = {
                    "1min": 0.0,
                    "session": 0.0,
                    "resample_swings": 0.0,
                    "5min": 0.0,
                    "15min": 0.0,
                    "1h": 0.0,
                    "4h": 0.0,
                    "dedup": 0.0,
                }

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

                _t = pytime.perf_counter()
                smt_events.extend(
                    detect_smt_divergence(
                        es_sw_1m_morning, nq_sw_1m_morning, df_es_base, df_nq_base,
                        time_window_minutes=15, timeframe="1min"
                    )
                )
                smt_prof["1min"] += pytime.perf_counter() - _t

                _t = pytime.perf_counter()
                smt_events.extend(
                    detect_session_smt_divergence(
                        es_mac, nq_mac, es_sw_1m, nq_sw_1m, df_es_base, df_nq_base
                    )
                )
                smt_prof["session"] += pytime.perf_counter() - _t

                if len(df_es_base) >= 30 and len(df_nq_base) >= 30:
                    for smt_tf, smt_win in (("5min", 20), ("15min", 60), ("1h", 180), ("4h", 480)):
                        _tr = pytime.perf_counter()
                        df_es_tf = resample_tv_timeframe(df_es_base, smt_tf)
                        df_nq_tf = resample_tv_timeframe(df_nq_base, smt_tf)
                        if len(df_es_tf) >= 3 and len(df_nq_tf) >= 3:
                            sw_es_tf = detect_htf_swings(df_es_tf)
                            sw_nq_tf = detect_htf_swings(df_nq_tf)
                            if smt_tf == "5min" and len(df_es_base) >= 250:
                                t_start_5m, t_end_5m = dtime(6, 0), dtime(10, 45)
                                sw_es_tf = [s for s in sw_es_tf if t_start_5m <= s["time"].time() <= t_end_5m]
                                sw_nq_tf = [s for s in sw_nq_tf if t_start_5m <= s["time"].time() <= t_end_5m]
                            smt_prof["resample_swings"] += pytime.perf_counter() - _tr

                            _td = pytime.perf_counter()
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
                            smt_prof[smt_tf] += pytime.perf_counter() - _td
                        else:
                            smt_prof["resample_swings"] += pytime.perf_counter() - _tr

                _t = pytime.perf_counter()
                deduped_smt: List[Dict[str, Any]] = []
                by_sweep_sig: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
                for ev in sorted(smt_events, key=lambda x: x.get("confirmation_close_time", x["time"])):
                    sig = (
                        ev.get("type"),
                        ev["time"].date(),
                        round(float(ev.get("swept_level", 0.0)), 2),
                        round(float(ev.get("sweep_extreme", 0.0)), 2),
                    )
                    if sig[2] != 0.0 and sig in by_sweep_sig:
                        existing_ev = by_sweep_sig[sig]
                        tfs = existing_ev.setdefault("all_timeframes", {str(existing_ev.get("timeframe", ""))})
                        tfs.add(str(ev.get("timeframe", "")))
                        continue
                    ev["all_timeframes"] = {str(ev.get("timeframe", ""))}
                    by_sweep_sig[sig] = ev
                    deduped_smt.append(ev)
                smt_events = sorted(deduped_smt, key=lambda x: x["time"])
                smt_prof["dedup"] += pytime.perf_counter() - _t
                BacktestEngine._GLOBAL_SMT_CACHE[smt_key] = smt_events

        elapsed = pytime.perf_counter() - t0
        print(f"   ⚡ Precomputed multi-TF ICT domain for {self.symbol} in {elapsed:.2f}s (cached)...")
        if self.enable_stage_profiling and len(df_primary) >= 100:
            if smt_prof.get("cache_hit"):
                smt_str = "SMT=CACHE_HIT"
            elif smt_prof:
                smt_total = sum(smt_prof.values())
                smt_str = (
                    f"SMT_total={smt_total:.2f}s "
                    f"(1m={smt_prof['1min']:.2f}s, sess={smt_prof['session']:.2f}s, "
                    f"5m={smt_prof['5min']:.2f}s, 15m={smt_prof['15min']:.2f}s, "
                    f"1h={smt_prof['1h']:.2f}s, 4h={smt_prof['4h']:.2f}s, "
                    f"resamp_sw={smt_prof['resample_swings']:.2f}s, dedup={smt_prof['dedup']:.2f}s)"
                )
            else:
                smt_str = "SMT=0.00s"
            print(
                f"     ⏱️  [Domain {self.symbol}] prim={t_prim_s:.2f}s | sec={t_sec_s:.2f}s | {smt_str}"
            )

        bundle = {
            **prim_data,
            "sec_struct_df_by_tf": sec_struct_df_by_tf,
            "secondary_major_catalog": secondary_major_catalog,
            "sec_near_ath_dates": sec_near_ath_dates,
            "smt_events": smt_events,
        }

        if disk_cache_file is not None:
            try:
                disk_cache_file.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = disk_cache_file.with_suffix(".pkl.tmp")
                with open(tmp_path, "wb") as f:
                    pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
                tmp_path.replace(disk_cache_file)
            except Exception as e:
                print(f"   ⚠️ Disk cache save failed for {self.symbol}: {e}")

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

        t_run_start = pytime.perf_counter()
        t_precompute_s = 0.0
        t_broker_gap_s = 0.0
        t_ctx_build_s = 0.0
        t_strategy_s = 0.0
        t_mtm_s = 0.0
        bars_skipped = 0
        bars_open_fast = 0
        bars_full_ctx = 0

        portfolio = Portfolio(initial_capital=self.initial_capital)
        broker = SimulatedBroker(
            portfolio=portfolio,
            commission_per_contract=self.commission_per_contract,
            slippage_ticks=self.slippage_ticks,
        )
        event_bus = EventBus()

        times = df_primary.index
        times_ns = _index_to_ns(times)
        times_ns_list = times_ns.tolist()
        times_list = times.tolist()
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
            start_bar_idx = int(np.searchsorted(times_ns, _ts_to_ns(ts_start), side="left"))

        if precompute_domain and len(df_primary) >= 5:
            _tp = pytime.perf_counter()
            dom = self._get_or_build_domain(df_primary, df_secondary)
            t_precompute_s = pytime.perf_counter() - _tp
        else:
            dom = {}

        t_setup_start = pytime.perf_counter()
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

        confirm_delta_1m_ns = self.swing_confirmation_bars * 60_000_000_000
        confirm_delta_5m_ns = (5 * 3 - 1) * 60_000_000_000
        fvg_close_delta_5m_ns = 4 * 60_000_000_000
        recent_15m_ns = 15 * 60_000_000_000
        recent_window_ns = 25 * 60_000_000_000
        max_pool_age_ns = 48 * 3600 * 1_000_000_000

        # Pre-extract nanosecond index lists & monotonic active-set schedules
        swings_1m_ns = [_ts_to_ns(ts) for ts in swings_1m_times]
        swings_5m_ns = [_ts_to_ns(ts) for ts in swings_5m_times]
        struct_1m_ns = [_ts_to_ns(ts) for ts in struct_1m_times]
        struct_5m_ns = [_ts_to_ns(ts) for ts in struct_5m_times]

        fvg_1m_creation_ns = [_ts_to_ns(ts) for ts in fvg_1m_creation_times]
        fvg_1m_death_ns = [_ts_to_ns(ts) for ts in fvg_1m_death_times]
        fvg1_ptr = 0
        n_fvg1 = len(fvgs_1m)
        active_fvg1: List[Tuple[int, FairValueGap]] = []

        fvg_5m_ready_ns = [_ts_to_ns(ts) for ts in fvg_5m_ready_times]
        fvg_5m_death_ns = [_ts_to_ns(ts) for ts in fvg_5m_death_times]
        fvg5_ptr = 0
        n_fvg5 = len(fvgs_5m)
        active_fvg5: List[Tuple[int, FairValueGap]] = []

        dr_1m_creation_ns = [_ts_to_ns(ts) for ts in dr_1m_creation_times]
        dr_1m_break_ns = [_ts_to_ns(dr.break_time) if dr.break_time is not None else _FAR_FUTURE_NS for dr in dealing_ranges_1m]
        fib_1m_extreme_ns = [_ts_to_ns(ts) for ts in fib_1m_extreme_times]
        fib_1m_break_ns = [_ts_to_ns(fib.break_time) if fib.break_time is not None else _FAR_FUTURE_NS for fib in fib_ranges_1m]

        dr_5m_creation_ns = [_ts_to_ns(ts) for ts in dr_5m_creation_times]
        dr_5m_break_ns = [
            (_ts_to_ns(dr.break_time) + fvg_close_delta_5m_ns) if dr.break_time is not None else _FAR_FUTURE_NS
            for dr in dealing_ranges_5m
        ]
        fib_5m_extreme_ns = [_ts_to_ns(ts) for ts in fib_5m_extreme_times]
        fib_5m_break_ns = [
            (_ts_to_ns(fib.break_time) + fvg_close_delta_5m_ns) if fib.break_time is not None else _FAR_FUTURE_NS
            for fib in fib_ranges_5m
        ]

        # Fast array-backed HTF trend lookup tables
        prim_trend_fast = self._prepare_fast_trend_lookup(struct_df_by_tf)
        sec_trend_fast = self._prepare_fast_trend_lookup(sec_struct_df_by_tf)

        # Monotonic schedule for active SMT events (100% equivalent to is_smt_active_at)
        smt_schedule: List[Tuple[int, int, int, Dict[str, Any]]] = []
        for orig_idx, smt in enumerate(smt_events):
            conf_time = smt.get("confirmation_close_time", smt.get("confirmation_time"))
            if conf_time is None:
                continue
            conf_ns = _ts_to_ns(conf_time)
            tf_key = str(smt.get("timeframe", "1min")).lower().strip()
            max_age_min = int(TIMEFRAME_HIERARCHY.get(tf_key, TIMEFRAME_HIERARCHY["1min"]).get("max_age_min", 180))
            end_excl_ns = conf_ns + max_age_min * 60_000_000_000 + 1
            res_time = smt.get("resolved_close_time", smt.get("resolved_time"))
            if res_time is not None and str(smt.get("status", SMT_CONFIRMED)) != SMT_CONFIRMED:
                res_ns = _ts_to_ns(res_time)
                if res_ns < end_excl_ns:
                    end_excl_ns = res_ns
            if conf_ns < end_excl_ns:
                smt_schedule.append((conf_ns, orig_idx, end_excl_ns, smt))
        smt_schedule.sort(key=lambda x: (x[0], x[1]))
        smt_ptr = 0
        n_smt_sched = len(smt_schedule)
        active_smts: List[Tuple[int, int, Dict[str, Any]]] = []

        # Monotonic schedule for htf_target_catalog (preserves exact catalog order & rules)
        htf_schedule: List[Tuple[Any, ...]] = []
        for orig_idx, item in enumerate(htf_target_catalog):
            pool_obj: LiquidityPool = item["pool"]
            p_type = pool_obj.pool_type
            p_price = float(item["price"])
            tf = item["timeframe"]
            s_orig_up = str(item.get("session_origin", "")).upper()
            is_asia_or_london = ("ASIA" in s_orig_up) or ("LONDON" in s_orig_up)
            sess_t_type = f"SESSION_{s_orig_up}"
            t_type = item["target_type"]
            tier = item["tier"]
            touches = pool_obj.touches
            cre_ts = pool_obj.creation_time
            cre_ns = _ts_to_ns(cre_ts)
            conf_ns = _ts_to_ns(item["confirm_time"])
            is_major = bool(item.get("is_major_exhaustion_pool", False))
            res_ts = item.get("first_touch_time")
            res_type = item.get("first_touch_type")
            res_ns = _ts_to_ns(res_ts) if res_ts is not None else -1
            res_hm = res_ts.strftime("%H:%M") if res_ts is not None else ""
            can_be_hist = (p_type == "BSL") and (tf in ("15min", "1h", "4h"))

            if res_ns != -1:
                dead_after_ns = (res_ns + 36 * 3600 * 1_000_000_000) if is_major else res_ns
            else:
                dead_after_ns = _FAR_FUTURE_NS if can_be_hist else (cre_ns + max_pool_age_ns + 1)

            htf_schedule.append((
                conf_ns, orig_idx, dead_after_ns, p_type, p_price, tf,
                is_asia_or_london, sess_t_type, t_type, tier, touches,
                cre_ts, cre_ns, is_major, res_ns, res_type, res_hm, can_be_hist,
            ))
        htf_schedule.sort(key=lambda x: (x[0], x[1]))
        htf_ptr = 0
        n_htf_sched = len(htf_schedule)
        active_htf: List[Tuple[Any, ...]] = []

        # Pre-extract secondary_major_catalog entries that have a resolution touch
        sec_major_fast: List[Tuple[int, int, str, str, str, float, str, str]] = []
        for sec_item in secondary_major_catalog:
            res_ts = sec_item.get("first_touch_time")
            if res_ts is not None:
                sec_major_fast.append((
                    _ts_to_ns(sec_item["creation_time"]),
                    _ts_to_ns(res_ts),
                    sec_item["symbol"],
                    sec_item["target_type"],
                    sec_item["pool_type"],
                    float(sec_item["price"]),
                    str(sec_item.get("first_touch_type", "")),
                    res_ts.strftime("%H:%M"),
                ))

        sweeps_by_time_ns: Dict[int, List[Tuple[LiquidityPool, str]]] = {}
        include_1m_structural = len(df_primary) < 25

        if include_1m_structural:
            for pool in all_pools_1m:
                lbl = f"1m Structural {pool.pool_type}"
                for ev_type, ev_ts in pool.events:
                    if ev_type == "SWEEP":
                        sweeps_by_time_ns.setdefault(_ts_to_ns(ev_ts), []).append((pool, lbl))

        for item in htf_target_catalog:
            if item.get("is_target_only_pool", False):
                continue
            res_ts = item.get("first_touch_time")
            res_type = item.get("first_touch_type")
            if res_ts is not None and res_type in ("SWEEP", "BREAK"):
                pool_obj = item["pool"]
                lbl = (
                    f"{pool_obj.session_origin} Session {pool_obj.pool_type}"
                    if item["timeframe"] == "SESSION"
                    else f"{item['timeframe']} {pool_obj.pool_type}"
                )
                sweeps_by_time_ns.setdefault(_ts_to_ns(res_ts), []).append((pool_obj, lbl))

        recent_sweeps_buffer: List[Tuple[int, pd.Timestamp, LiquidityPool, str]] = []
        gap_engine = GapEngine()

        req_ny_win = bool(getattr(strategy, "require_ny_open_window", False))
        entry_start = getattr(strategy, "entry_start_time", None)
        entry_end = getattr(strategy, "entry_end_time", None)

        for w_i in range(start_bar_idx):
            gap_engine.process_candle(times_list[w_i], opens[w_i], closes[w_i], highs[w_i], lows[w_i])

        cached_day_half_key: Optional[Tuple[Any, bool]] = None
        cached_session_start_ns: int = 0
        cached_cycle_start_ns: int = 0

        bisect_right = bisect.bisect_right
        bisect_left = bisect.bisect_left
        sym = self.symbol
        trend_fn = self._latest_trend_at_fast

        t_setup_s = pytime.perf_counter() - t_setup_start

        for i in range(start_bar_idx, len(df_primary)):
            _tb = pytime.perf_counter()
            t = times_list[i]
            t_ns = times_ns_list[i]
            o, h, l, c, v = opens[i], highs[i], lows[i], closes[i], volumes[i]

            gap_engine.process_candle(t, o, c, h, l)

            closed_trades = broker.process_candle(sym, t, o, h, l, c)
            for tr in closed_trades:
                event_bus.publish(
                    Event("TRADE_CLOSED", t, {"symbol": sym, "trade": tr})
                )

            if precompute_domain and t_ns in sweeps_by_time_ns:
                for p, lbl in sweeps_by_time_ns[t_ns]:
                    recent_sweeps_buffer.append((t_ns, t, p, lbl))

            has_open_pos = portfolio.has_open_position(sym)
            t_broker_gap_s += pytime.perf_counter() - _tb

            if precompute_domain and not has_open_pos and req_ny_win and entry_start is not None and entry_end is not None:
                bt = t.time()
                if not (entry_start <= bt <= entry_end):
                    _tm = pytime.perf_counter()
                    portfolio.update_mark_to_market(t, {sym: c})
                    t_mtm_s += pytime.perf_counter() - _tm
                    bars_skipped += 1
                    continue

            confirmed_cutoff_1m_ns = t_ns - confirm_delta_1m_ns
            confirmed_cutoff_5m_ns = t_ns - confirm_delta_5m_ns

            if precompute_domain and has_open_pos and req_ny_win and entry_start is not None and entry_end is not None:
                bt = t.time()
                if not (entry_start <= bt <= entry_end):
                    _tc = pytime.perf_counter()
                    end_sw1 = bisect_right(swings_1m_ns, confirmed_cutoff_1m_ns)
                    vis_swings_1m = swings_1m[:end_sw1]
                    idx_1m_st = bisect_right(struct_1m_ns, t_ns)
                    vis_struct_1m = struct_events_1m[max(0, idx_1m_st - 12):idx_1m_st]
                    hi_fvg1 = bisect_right(fvg_1m_creation_ns, t_ns)
                    lo_fvg1_recent = bisect_left(fvg_1m_creation_ns, t_ns - recent_15m_ns)
                    recent_fvgs_1m_all = fvgs_1m[lo_fvg1_recent:hi_fvg1]
                    ctx = BarContext(
                        symbol=sym,
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
                    t_ctx_build_s += pytime.perf_counter() - _tc

                    _ts = pytime.perf_counter()
                    strategy.on_bar(ctx, broker, portfolio)
                    t_strategy_s += pytime.perf_counter() - _ts

                    _tm = pytime.perf_counter()
                    portfolio.update_mark_to_market(t, {sym: c})
                    t_mtm_s += pytime.perf_counter() - _tm
                    bars_open_fast += 1
                    continue

            _tc = pytime.perf_counter()
            if precompute_domain:
                if recent_sweeps_buffer:
                    recent_sweeps_buffer = [
                        item for item in recent_sweeps_buffer if (t_ns - item[0]) <= recent_window_ns
                    ]
                vis_sweeps = [p for _, _, p, _ in recent_sweeps_buffer]
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
                    for _, ts, p, lbl in recent_sweeps_buffer
                ]

                while fvg1_ptr < n_fvg1 and fvg_1m_creation_ns[fvg1_ptr] <= t_ns:
                    d_ns = fvg_1m_death_ns[fvg1_ptr]
                    if d_ns > t_ns:
                        active_fvg1.append((d_ns, fvgs_1m[fvg1_ptr]))
                    fvg1_ptr += 1
                if active_fvg1:
                    active_fvg1 = [pair for pair in active_fvg1 if pair[0] > t_ns]
                vis_fvgs_1m = [pair[1] for pair in active_fvg1]

                hi_fvg1 = bisect_right(fvg_1m_creation_ns, t_ns)
                lo_fvg1_recent = bisect_left(fvg_1m_creation_ns, t_ns - recent_15m_ns)
                recent_fvgs_1m_all = fvgs_1m[lo_fvg1_recent:hi_fvg1]

                active_dr_1m = None
                hi_dr1 = bisect_right(dr_1m_creation_ns, confirmed_cutoff_1m_ns)
                for k in range(hi_dr1 - 1, -1, -1):
                    if dr_1m_break_ns[k] > t_ns:
                        active_dr_1m = dealing_ranges_1m[k]
                        break

                active_fib_1m = None
                hi_fib1 = bisect_right(fib_1m_extreme_ns, confirmed_cutoff_1m_ns)
                for k in range(hi_fib1 - 1, -1, -1):
                    if fib_1m_break_ns[k] > t_ns:
                        active_fib_1m = fib_ranges_1m[k]
                        break

                added_smt = False
                while smt_ptr < n_smt_sched and smt_schedule[smt_ptr][0] <= t_ns:
                    _, o_idx, end_excl_ns, smt_obj = smt_schedule[smt_ptr]
                    if t_ns < end_excl_ns:
                        active_smts.append((o_idx, end_excl_ns, smt_obj))
                        added_smt = True
                    smt_ptr += 1
                if active_smts:
                    active_smts = [item for item in active_smts if t_ns < item[1]]
                    if added_smt and len(active_smts) > 1:
                        active_smts.sort(key=lambda x: x[0])
                vis_smt = [item[2] for item in active_smts]

                end_sw1 = bisect_right(swings_1m_ns, confirmed_cutoff_1m_ns)
                vis_swings_1m = swings_1m[:end_sw1]

                idx_1m_st = bisect_right(struct_1m_ns, t_ns)
                vis_struct_1m = struct_events_1m[max(0, idx_1m_st - 12):idx_1m_st]

                end_sw5 = bisect_right(swings_5m_ns, confirmed_cutoff_5m_ns)
                vis_swings_5m = swings_5m[:end_sw5]

                idx_5m_st = bisect_right(struct_5m_ns, t_ns)
                vis_struct_5m = struct_events_5m[max(0, idx_5m_st - 12):idx_5m_st]

                while fvg5_ptr < n_fvg5 and fvg_5m_ready_ns[fvg5_ptr] <= t_ns:
                    d_ns = fvg_5m_death_ns[fvg5_ptr]
                    if d_ns > t_ns:
                        active_fvg5.append((d_ns, fvgs_5m[fvg5_ptr]))
                    fvg5_ptr += 1
                if active_fvg5:
                    active_fvg5 = [pair for pair in active_fvg5 if pair[0] > t_ns]
                vis_fvgs_5m = [pair[1] for pair in active_fvg5]

                active_dr_5m = None
                hi_dr5 = bisect_right(dr_5m_creation_ns, confirmed_cutoff_5m_ns)
                for k in range(hi_dr5 - 1, -1, -1):
                    if dr_5m_break_ns[k] > t_ns:
                        active_dr_5m = dealing_ranges_5m[k]
                        break

                active_fib_5m = None
                hi_fib5 = bisect_right(fib_5m_extreme_ns, confirmed_cutoff_5m_ns)
                for k in range(hi_fib5 - 1, -1, -1):
                    if fib_5m_break_ns[k] > t_ns:
                        active_fib_5m = fib_ranges_5m[k]
                        break

                htf_trends: Dict[str, str] = {}
                for tf_key in ("1min", "5min", "15min", "1h", "4h"):
                    entry = prim_trend_fast.get(tf_key)
                    if entry is not None:
                        htf_trends[tf_key] = trend_fn(entry, t_ns)
                    else:
                        htf_trends[tf_key] = htf_trends.get("1min", "BULLISH")

                sec_htf_trends: Dict[str, str] = {}
                for tf_key in ("5min", "15min", "1h", "4h"):
                    entry = sec_trend_fast.get(tf_key)
                    if entry is not None:
                        sec_htf_trends[tf_key] = trend_fn(entry, t_ns)
                    else:
                        sec_htf_trends[tf_key] = htf_trends.get(tf_key, "BULLISH")

                cur_date = t.date()
                ath_active = (cur_date in prim_near_ath_dates) or (cur_date in sec_near_ath_dates)
                htf_trends["ath_regime"] = "ACTIVE" if ath_active else "NONE"

                day_half_key = (cur_date, t.hour < 18)
                if day_half_key != cached_day_half_key:
                    session_start = t.replace(hour=9, minute=30, second=0, microsecond=0)
                    cycle_start = (
                        (t - pd.Timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
                        if t.hour < 18
                        else t.replace(hour=18, minute=0, second=0, microsecond=0)
                    )
                    cached_session_start_ns = _ts_to_ns(session_start)
                    cached_cycle_start_ns = _ts_to_ns(cycle_start)
                    cached_day_half_key = day_half_key
                session_start_ns = cached_session_start_ns
                cycle_start_ns = cached_cycle_start_ns

                vis_targets: List[Dict[str, Any]] = []
                taken_major_all: List[str] = []
                unique_major_pools_taken = set()
                htf_pools_taken_candidates: List[Tuple[str, str, float, int]] = []
                min_creation_cutoff_ns = t_ns - max_pool_age_ns

                if ath_active:
                    lookback_48h_idx = max(0, i - 2880)
                    high_48h = float(highs[lookback_48h_idx : i + 1].max())
                else:
                    high_48h = 0.0

                added_htf = False
                while htf_ptr < n_htf_sched and htf_schedule[htf_ptr][0] <= t_ns:
                    rec = htf_schedule[htf_ptr]
                    if t_ns < rec[2]:
                        active_htf.append(rec)
                        added_htf = True
                    htf_ptr += 1
                if active_htf:
                    active_htf = [rec for rec in active_htf if t_ns < rec[2]]
                    if added_htf and len(active_htf) > 1:
                        active_htf.sort(key=lambda x: x[1])

                for rec in active_htf:
                    (
                        _, _, _, p_type, p_price, tf,
                        is_asia_or_london, sess_t_type, t_type, tier, touches,
                        cre_ts, cre_ns, is_major, res_ns, res_type, res_hm, can_be_hist,
                    ) = rec

                    is_unbroken = (res_ns == -1) or (res_ns > t_ns)
                    is_unmitigated_historical = (
                        is_unbroken and ath_active and can_be_hist and (p_price > high_48h)
                    )

                    if cre_ns < min_creation_cutoff_ns and not is_unmitigated_historical:
                        continue

                    if tf == "SESSION":
                        if cre_ns >= cycle_start_ns and is_asia_or_london:
                            eff_tf = "SESSION"
                            eff_target_type = sess_t_type
                        else:
                            eff_tf = "1h"
                            eff_target_type = "1H_POOL"
                    else:
                        eff_tf = tf
                        eff_target_type = t_type

                    if is_unbroken:
                        if tf == "SESSION" and cre_ns >= session_start_ns:
                            continue
                        vis_targets.append({
                            "pool_type": p_type,
                            "price": p_price,
                            "target_type": eff_target_type,
                            "timeframe": eff_tf,
                            "tier": tier,
                            "touches": touches,
                            "creation_time": cre_ts,
                        })
                    elif is_major and session_start_ns <= res_ns <= t_ns:
                        if tf == "SESSION" and cre_ns < cycle_start_ns:
                            continue
                        entry_str = f"[{sym}] {eff_target_type} ({p_type} @ {p_price:.2f} {res_type} at {res_hm})"
                        taken_major_all.append(entry_str)
                        if eff_tf == "SESSION":
                            unique_major_pools_taken.add(("SESSION", eff_target_type, p_type))
                        else:
                            htf_pools_taken_candidates.append(
                                (eff_target_type, p_type, p_price, cre_ns)
                            )

                for sec_cre_ns, sec_res_ns, sec_sym, sec_t_type, sec_p_type, sec_price, sec_res_type, sec_res_hm in sec_major_fast:
                    if sec_cre_ns < cycle_start_ns:
                        continue
                    if session_start_ns <= sec_res_ns <= t_ns:
                        entry_str = (
                            f"[{sec_sym}] {sec_t_type} "
                            f"({sec_p_type} @ {sec_price:.2f} {sec_res_type} at {sec_res_hm})"
                        )
                        taken_major_all.append(entry_str)
                        unique_major_pools_taken.add(("SESSION", sec_t_type, sec_p_type))

                swept_session_sides = {side for kind, _, side in unique_major_pools_taken if kind == "SESSION"}
                for eff_t_type, p_side, p_pr, c_ns in htf_pools_taken_candidates:
                    if c_ns >= cycle_start_ns and p_side in swept_session_sides:
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
                symbol=sym,
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
            t_ctx_build_s += pytime.perf_counter() - _tc
            bars_full_ctx += 1

            _ts = pytime.perf_counter()
            strategy.on_bar(ctx, broker, portfolio)
            t_strategy_s += pytime.perf_counter() - _ts

            _tm = pytime.perf_counter()
            portfolio.update_mark_to_market(t, {sym: c})
            t_mtm_s += pytime.perf_counter() - _tm

        if portfolio.has_open_position(sym):
            last_time = times_list[-1]
            last_close = closes[-1]
            portfolio.close_position(
                symbol=sym,
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
        _tmet = pytime.perf_counter()
        res = calculate_metrics(portfolio, strategy_name=strategy_name, symbol=sym)
        t_metrics_s = pytime.perf_counter() - _tmet

        if self.enable_stage_profiling and precompute_domain and len(df_primary) >= 100:
            run_total = pytime.perf_counter() - t_run_start
            print(
                f"     ⏱️  [Run {sym} | {strategy_name}] total={run_total:.2f}s | "
                f"precomp={t_precompute_s:.2f}s | setup={t_setup_s:.2f}s | broker_gap={t_broker_gap_s:.2f}s | "
                f"ctx={t_ctx_build_s:.2f}s | on_bar={t_strategy_s:.2f}s | mtm={t_mtm_s:.2f}s | "
                f"metrics={t_metrics_s:.2f}s (bars: full={bars_full_ctx}, open_fast={bars_open_fast}, skip={bars_skipped})"
            )
        return res