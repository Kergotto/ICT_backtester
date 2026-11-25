# Copyright (C) 2026 Kyrylo Homberh
# SPDX-License-Identifier: AGPL-3.0-only

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


_TWO_MIN_NS: int = 120_000_000_000
_TWELVE_HR_NS: int = 43_200_000_000_000


def _ts_to_ns(ts: Any) -> int:
    if hasattr(ts, "as_unit"):
        return int(ts.as_unit("ns").value)
    if hasattr(ts, "value"):
        return int(ts.value)
    return int(pd.Timestamp(ts).as_unit("ns").value)


def _index_to_ns(idx: pd.DatetimeIndex) -> np.ndarray:
    if len(idx) == 0:
        return np.empty(0, dtype=np.int64)
    return idx.as_unit("ns").asi8 if hasattr(idx, "as_unit") else idx.asi8


@dataclass(slots=True)
class LiquidityPool:
    pool_type: str
    upper_bound: float
    lower_bound: float
    creation_time: pd.Timestamp
    touches: int = 1
    last_touch_time: pd.Timestamp = field(default=None)
    events: List[Tuple[str, pd.Timestamp]] = field(default_factory=list)
    session_origin: str = "Unknown"
    source: str = "Standard"
    initial_price: float = field(default=0.0)
    touch_history: List[Tuple[pd.Timestamp, int, float, float]] = field(default_factory=list)
    _creation_ns: int = field(init=False, repr=False, compare=False)
    _last_touch_ns: int = field(init=False, repr=False, compare=False)
    _broken: bool = field(init=False, repr=False, compare=False)
    _swept: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.last_touch_time is None:
            self.last_touch_time = self.creation_time
        if self.initial_price == 0.0:
            self.initial_price = self.upper_bound if self.pool_type == 'BSL' else self.lower_bound
        if not self.touch_history:
            self.touch_history.append((self.creation_time, self.touches, self.lower_bound, self.upper_bound))
        self._creation_ns = _ts_to_ns(self.creation_time)
        self._last_touch_ns = _ts_to_ns(self.last_touch_time)
        broken = False
        swept = False
        for ev_type, _ in self.events:
            if ev_type == "BREAK":
                broken = True
            elif ev_type == "SWEEP":
                swept = True
        self._broken = broken
        self._swept = swept

    @property
    def is_active(self) -> bool:
        if not self.events:
            return True
        for event_type, _ in self.events:
            if event_type == "BREAK":
                return False
        return True

    @property
    def has_been_swept(self) -> bool:
        if not self.events:
            return False
        for event_type, _ in self.events:
            if event_type == "SWEEP":
                return True
        return False

    def evaluate_candle(self, high: float, low: float, close: float, timestamp: pd.Timestamp) -> None:
        if not self.is_active or timestamp <= self.creation_time:
            return
        self._evaluate_candle_fast(high, low, close, timestamp, _ts_to_ns(timestamp))

    def _evaluate_candle_fast(
        self, high: float, low: float, close: float, timestamp: pd.Timestamp, ts_ns: int
    ) -> None:
        if self._broken or ts_ns <= self._creation_ns:
            return

        if self.pool_type == 'BSL':
            ub = self.upper_bound
            if close > ub:
                self._broken = True
                self.events.append(("BREAK", timestamp))
                return

            if high > ub:
                if not self._swept:
                    if self.source != "Macro_Session" and (high - ub) <= 2.0:
                        self.upper_bound = float(high)
                    self._swept = True
                self.events.append(("SWEEP", timestamp))
                self.touch_history.append((timestamp, self.touches, self.lower_bound, self.upper_bound))
                return

            if not self._swept and (ts_ns - self._creation_ns) > _TWO_MIN_NS and high > self.lower_bound:
                new_lb = high if high < ub else ub
                if new_lb > self.lower_bound:
                    self.lower_bound = float(new_lb)
                self.touch_history.append((timestamp, self.touches, self.lower_bound, ub))
        else:
            lb = self.lower_bound
            if close < lb:
                self._broken = True
                self.events.append(("BREAK", timestamp))
                return

            if low < lb:
                if not self._swept:
                    if self.source != "Macro_Session" and (lb - low) <= 2.0:
                        self.lower_bound = float(low)
                    self._swept = True
                self.events.append(("SWEEP", timestamp))
                self.touch_history.append((timestamp, self.touches, self.lower_bound, self.upper_bound))
                return

            if not self._swept and (ts_ns - self._creation_ns) > _TWO_MIN_NS and low < self.upper_bound:
                new_ub = low if low > lb else lb
                if new_ub < self.upper_bound:
                    self.upper_bound = float(new_ub)
                self.touch_history.append((timestamp, self.touches, lb, self.upper_bound))

    def _is_body_break(self, close: float) -> bool:
        return (self.pool_type == 'BSL' and close > self.upper_bound) or \
            (self.pool_type == 'SSL' and close < self.lower_bound)

    def _is_wick_sweep(self, high: float, low: float, close: float) -> bool:
        return (self.pool_type == 'BSL' and high > self.upper_bound and close <= self.upper_bound) or \
            (self.pool_type == 'SSL' and low < self.lower_bound and close >= self.lower_bound)

    def _is_partial_mitigation(self, high: float, low: float) -> bool:
        return (self.pool_type == 'BSL' and high > self.lower_bound) or \
            (self.pool_type == 'SSL' and low < self.upper_bound)

    def distance_to(self, current_price: float) -> float:
        """Calculates the exact point distance from the current price to the pool's closest boundary."""
        if self.pool_type == 'BSL':
            return self.lower_bound - current_price
        elif self.pool_type == 'SSL':
            return current_price - self.upper_bound
        return 0.0


@dataclass(slots=True)
class WeekendGap:
    friday_close_time: pd.Timestamp
    friday_close_price: float
    sunday_open_time: pd.Timestamp
    sunday_open_price: float
    is_filled: bool = False

    @property
    def upper_bound(self) -> float:
        """Returns the top of the gap."""
        return max(self.friday_close_price, self.sunday_open_price)

    @property
    def lower_bound(self) -> float:
        """Returns the bottom of the gap."""
        return min(self.friday_close_price, self.sunday_open_price)

    def distance_to(self, current_price: float) -> float:
        """Calculates distance to the closest gap boundary."""
        if current_price > self.upper_bound:
            return current_price - self.upper_bound
        elif current_price < self.lower_bound:
            return self.lower_bound - current_price
        return 0.0


class GapEngine:
    """Independent engine to track and manage weekend gaps as liquidity draws."""

    def __init__(self):
        self.active_gaps: List[WeekendGap] = []
        self._last_candle_time: Optional[pd.Timestamp] = None
        self._last_close_price: float = 0.0

    def process_candle(self, timestamp: pd.Timestamp, open_price: float, close_price: float, high: float, low: float):
        """Processes each new candle to detect weekend gaps and check for fills."""
        if self._last_candle_time is not None:
            was_friday = self._last_candle_time.dayofweek == 4
            is_sunday = timestamp.dayofweek == 6

            if was_friday and is_sunday:
                new_gap = WeekendGap(
                    friday_close_time=self._last_candle_time,
                    friday_close_price=self._last_close_price,
                    sunday_open_time=timestamp,
                    sunday_open_price=open_price
                )
                self.active_gaps.append(new_gap)

        if self.active_gaps:
            self._check_fills(high, low)

        self._last_candle_time = timestamp
        self._last_close_price = close_price

    def _check_fills(self, current_high: float, current_low: float):
        """Marks gaps as filled when price reaches the origin of the gap."""
        for gap in self.active_gaps:
            if not gap.is_filled:
                is_gap_up = gap.sunday_open_price > gap.friday_close_price
                if is_gap_up:
                    if current_low <= gap.lower_bound:
                        gap.is_filled = True
                else:
                    if current_high >= gap.upper_bound:
                        gap.is_filled = True


class TickTolerance:
    def __init__(self, ticks: int = 4, tick_size: float = 0.25):
        self.max_distance = ticks * tick_size


class SessionManager:
    """Evaluates pd.Timestamp to define the current trading session."""

    @staticmethod
    def get_session(timestamp: pd.Timestamp) -> str:
        hour = timestamp.hour
        minute = timestamp.minute

        if hour >= 18 or hour < 3:
            return "Asia"
        elif hour >= 3 and (hour < 9 or (hour == 9 and minute < 30)):
            return "London"
        elif (hour == 9 and minute >= 30) or (hour >= 10 and hour < 17):
            return "New York"
        elif hour == 17:
            return "CME Gap"
        return "Unknown"


class LiquidityEngine:
    """
    Spatial-bucketed liquidity index that maintains only active pools in its
    hash buckets (self.vault) by pruning broken pools into self.archived_pools.
    """

    def __init__(self, tolerance_ticks: int = 4, bucket_size: float = 100.0):
        self.tolerance = TickTolerance(ticks=tolerance_ticks)
        self.bucket_size = bucket_size
        self._step = int(bucket_size)
        self.vault: Dict[int, List[LiquidityPool]] = {}
        self.archived_pools: List[LiquidityPool] = []

    def _get_bucket_key(self, price: float) -> int:
        return int(price // self.bucket_size) * self._step

    def add_new_swing(
        self,
        pool_type: str,
        price: float,
        timestamp: pd.Timestamp,
        ts_ns: Optional[int] = None,
    ):
        if ts_ns is None:
            ts_ns = _ts_to_ns(timestamp)
        bucket_key = int(price // self.bucket_size) * self._step
        step = self._step
        max_dist = self.tolerance.max_distance
        vault = self.vault

        for b_key in (bucket_key - step, bucket_key, bucket_key + step):
            bucket = vault.get(b_key)
            if bucket is None:
                continue
            if pool_type == 'BSL':
                for pool in bucket:
                    if pool._broken or pool._swept or pool.pool_type != 'BSL':
                        continue
                    ub = pool.upper_bound
                    if ub < price <= ub + max_dist:
                        pool.touches = 2
                        pool.upper_bound = float(price)
                        pool.last_touch_time = timestamp
                        pool._last_touch_ns = ts_ns
                        pool.touch_history.append((timestamp, pool.touches, pool.lower_bound, pool.upper_bound))
                        return
                    elif pool.lower_bound <= price <= ub:
                        if (ts_ns - pool._last_touch_ns) >= _TWO_MIN_NS:
                            pool.touches += 1
                            new_lb = price if price > pool.lower_bound else pool.lower_bound
                            pool.lower_bound = float(ub if ub < new_lb else new_lb)
                            pool.last_touch_time = timestamp
                            pool._last_touch_ns = ts_ns
                            pool.touch_history.append((timestamp, pool.touches, pool.lower_bound, ub))
                        return
            else:
                for pool in bucket:
                    if pool._broken or pool._swept or pool.pool_type != 'SSL':
                        continue
                    lb = pool.lower_bound
                    if lb - max_dist <= price < lb:
                        pool.touches = 2
                        pool.lower_bound = float(price)
                        pool.last_touch_time = timestamp
                        pool._last_touch_ns = ts_ns
                        pool.touch_history.append((timestamp, pool.touches, pool.lower_bound, pool.upper_bound))
                        return
                    elif lb <= price <= pool.upper_bound:
                        if (ts_ns - pool._last_touch_ns) >= _TWO_MIN_NS:
                            pool.touches += 1
                            new_ub = price if price < pool.upper_bound else pool.upper_bound
                            pool.upper_bound = float(lb if lb > new_ub else new_ub)
                            pool.last_touch_time = timestamp
                            pool._last_touch_ns = ts_ns
                            pool.touch_history.append((timestamp, pool.touches, lb, pool.upper_bound))
                        return

        new_pool = LiquidityPool(
            pool_type=pool_type,
            upper_bound=float(price) if pool_type == 'BSL' else float(price + max_dist),
            lower_bound=float(price - max_dist) if pool_type == 'BSL' else float(price),
            creation_time=timestamp,
            session_origin=SessionManager.get_session(timestamp)
        )
        bucket = vault.get(bucket_key)
        if bucket is None:
            vault[bucket_key] = [new_pool]
        else:
            bucket.append(new_pool)

    def clean_duplicate_pools(self):
        all_pools = self.get_all_pools()
        global_macros = [p for p in all_pools if p.source == "Macro_Session"]

        if not global_macros:
            return

        macro_sigs = [
            (mp.pool_type, mp._creation_ns, mp.upper_bound if mp.pool_type == 'BSL' else mp.lower_bound)
            for mp in global_macros
        ]

        def _is_duplicate(p: LiquidityPool) -> bool:
            if p.source == "Macro_Session":
                return False
            p_type = p.pool_type
            p_ns = p._creation_ns
            p_init = p.initial_price
            for mp_type, mp_ns, mp_price in macro_sigs:
                if p_type == mp_type:
                    diff = p_ns - mp_ns if p_ns >= mp_ns else mp_ns - p_ns
                    if diff <= _TWELVE_HR_NS and abs(p_init - mp_price) <= 1.0:
                        return True
            return False

        for bucket_key in list(self.vault.keys()):
            cleaned = [p for p in self.vault[bucket_key] if not _is_duplicate(p)]
            if cleaned:
                self.vault[bucket_key] = cleaned
            else:
                del self.vault[bucket_key]

        self.archived_pools = [p for p in self.archived_pools if not _is_duplicate(p)]

    def inject_macro_session_pool(self, pool_type: str, price: float, timestamp: pd.Timestamp, session_name: str):
        bucket_key = int(price // self.bucket_size) * self._step
        max_dist = self.tolerance.max_distance

        new_pool = LiquidityPool(
            pool_type=pool_type,
            upper_bound=float(price) if pool_type == 'BSL' else float(price + max_dist),
            lower_bound=float(price - max_dist) if pool_type == 'BSL' else float(price),
            creation_time=timestamp,
            session_origin=session_name,
            source="Macro_Session",
            touches=1
        )

        bucket = self.vault.get(bucket_key)
        if bucket is None:
            self.vault[bucket_key] = [new_pool]
        else:
            bucket.append(new_pool)

    def check_for_sweeps_and_breaks(
        self,
        high: float,
        low: float,
        close: float,
        timestamp: pd.Timestamp,
        ts_ns: Optional[int] = None,
    ):
        vault = self.vault
        if not vault:
            return

        step = self._step
        start_bucket = int(low // self.bucket_size) * step
        end_bucket = int(high // self.bucket_size) * step

        if start_bucket == end_bucket:
            bucket = vault.get(start_bucket)
            if bucket is None:
                return
            self._process_bucket(start_bucket, bucket, high, low, close, timestamp, ts_ns)
            return

        key = start_bucket
        while key <= end_bucket:
            bucket = vault.get(key)
            if bucket is not None:
                self._process_bucket(key, bucket, high, low, close, timestamp, ts_ns)
            key += step

    def _process_bucket(
        self,
        key: int,
        bucket: List[LiquidityPool],
        high: float,
        low: float,
        close: float,
        timestamp: pd.Timestamp,
        ts_ns: Optional[int],
    ) -> None:
        any_broken = False
        for pool in bucket:
            if pool._broken:
                any_broken = True
                continue
            if pool.pool_type == 'BSL':
                if close <= pool.upper_bound and high <= pool.upper_bound and high <= pool.lower_bound:
                    continue
            else:
                if close >= pool.lower_bound and low >= pool.lower_bound and low >= pool.upper_bound:
                    continue

            if ts_ns is None:
                ts_ns = _ts_to_ns(timestamp)
            pool._evaluate_candle_fast(high, low, close, timestamp, ts_ns)
            if pool._broken:
                any_broken = True

        if any_broken:
            active_in_bucket: List[LiquidityPool] = []
            archived = self.archived_pools
            for pool in bucket:
                if pool._broken:
                    archived.append(pool)
                else:
                    active_in_bucket.append(pool)
            if active_in_bucket:
                self.vault[key] = active_in_bucket
            else:
                del self.vault[key]

    def get_active_pools(self) -> List[LiquidityPool]:
        """Returns a flat list of all currently active pools across the spatial index."""
        return [pool for bucket in self.vault.values() for pool in bucket]

    def get_all_pools(self) -> List[LiquidityPool]:
        """Returns both active and archived (broken) pools for visualization and reporting."""
        return self.get_active_pools() + self.archived_pools


def track_liquidity(
    df_structure: pd.DataFrame,
    swings: List[Dict[str, Any]],
    macro_pools_by_time: Dict[pd.Timestamp, List[Dict]],
    tolerance_ticks: int = 4,
) -> tuple[pd.DataFrame, LiquidityEngine]:
    df = df_structure.copy()
    times = df.index
    n = len(df)
    engine = LiquidityEngine(tolerance_ticks=tolerance_ticks)
    if n == 0:
        return df, engine

    times_ns = _index_to_ns(times).tolist()
    times_list = times.tolist()

    # Sort swings chronologically
    sorted_swings = sorted(swings, key=lambda x: x["time"])
    swing_ns = [_ts_to_ns(s["time"]) for s in sorted_swings]
    swing_is_high = [s["type"] == "SwingHigh" for s in sorted_swings]
    swing_is_low = [s["type"] == "SwingLow" for s in sorted_swings]
    swing_prices = [float(s["price"]) for s in sorted_swings]
    swing_idx = 0
    num_swings = len(sorted_swings)

    # Flatten and sort all macro session pools chronologically so they inject
    # cleanly on ANY resampled timeframe (1m, 5m, 15m, 1h, 4h)
    flat_macros: List[Dict[str, Any]] = []
    for ts in sorted(macro_pools_by_time.keys()):
        for m_pool in macro_pools_by_time[ts]:
            flat_macros.append(m_pool)
    macro_ns = [_ts_to_ns(m["time"]) for m in flat_macros]
    macro_idx = 0
    num_macros = len(flat_macros)

    highs = df["high"].to_numpy(dtype=float, copy=False).tolist()
    lows = df["low"].to_numpy(dtype=float, copy=False).tolist()
    closes = df["close"].to_numpy(dtype=float, copy=False).tolist()

    add_swing = engine.add_new_swing
    inject_macro = engine.inject_macro_session_pool
    check_sweeps = engine.check_for_sweeps_and_breaks

    for i in range(n):
        c_ns = times_ns[i]
        current_time = times_list[i]

        # 1. Inject any swings that occurred up to `current_time`
        while swing_idx < num_swings and swing_ns[swing_idx] <= c_ns:
            if swing_is_high[swing_idx]:
                add_swing("BSL", swing_prices[swing_idx], current_time, ts_ns=c_ns)
            elif swing_is_low[swing_idx]:
                add_swing("SSL", swing_prices[swing_idx], current_time, ts_ns=c_ns)
            swing_idx += 1

        # 2. Inject any Macro Session pools that occurred up to `current_time`
        while macro_idx < num_macros and macro_ns[macro_idx] <= c_ns:
            m_pool = flat_macros[macro_idx]
            inject_macro(
                pool_type=m_pool["type"],
                price=m_pool["price"],
                timestamp=current_time,
                session_name=m_pool["session"],
            )
            macro_idx += 1

        # 3. Evaluate sweeps and breaks on the current candle
        if i >= 1:
            check_sweeps(highs[i], lows[i], closes[i], current_time, ts_ns=c_ns)

    engine.clean_duplicate_pools()

    return df, engine