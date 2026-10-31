from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd


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

    def __post_init__(self):
        if self.last_touch_time is None:
            self.last_touch_time = self.creation_time
        if self.initial_price == 0.0:
            self.initial_price = self.upper_bound if self.pool_type == 'BSL' else self.lower_bound
        if not self.touch_history:
            self.touch_history.append((self.creation_time, self.touches, self.lower_bound, self.upper_bound))

    @property
    def is_active(self) -> bool:
        return not any(event_type == "BREAK" for event_type, _ in self.events)

    @property
    def has_been_swept(self) -> bool:
        return any(event_type == "SWEEP" for event_type, _ in self.events)

    def evaluate_candle(self, high: float, low: float, close: float, timestamp: pd.Timestamp) -> None:
        if not self.is_active or timestamp <= self.creation_time:
            return

        if self._is_body_break(close):
            self.events.append(("BREAK", timestamp))
            return

        if self._is_wick_sweep(high, low, close):
            if not self.has_been_swept:
                if self.source != "Macro_Session":
                    if self.pool_type == 'BSL' and (high - self.upper_bound) <= 2.0:
                        self.upper_bound = float(max(self.upper_bound, high))
                    elif self.pool_type == 'SSL' and (self.lower_bound - low) <= 2.0:
                        self.lower_bound = float(min(self.lower_bound, low))
            self.events.append(("SWEEP", timestamp))
            self.touch_history.append((timestamp, self.touches, self.lower_bound, self.upper_bound))
            return

        if (timestamp - self.creation_time).total_seconds() > 120:
            if not self.has_been_swept and self._is_partial_mitigation(high, low):
                if self.pool_type == 'BSL':
                    self.lower_bound = float(max(self.lower_bound, min(high, self.upper_bound)))
                elif self.pool_type == 'SSL':
                    self.upper_bound = float(min(self.upper_bound, max(low, self.lower_bound)))
                self.touch_history.append((timestamp, self.touches, self.lower_bound, self.upper_bound))

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
        self.vault: Dict[int, List[LiquidityPool]] = {}
        self.archived_pools: List[LiquidityPool] = []

    def _get_bucket_key(self, price: float) -> int:
        return int(price // self.bucket_size) * int(self.bucket_size)

    def add_new_swing(self, pool_type: str, price: float, timestamp: pd.Timestamp):
        bucket_key = self._get_bucket_key(price)
        step = int(self.bucket_size)
        buckets_to_check = [bucket_key - step, bucket_key, bucket_key + step]

        for b_key in buckets_to_check:
            if b_key not in self.vault:
                continue
            for pool in self.vault[b_key]:
                # Skip already-swept pools so subsequent swings can form their own pools
                if not pool.is_active or pool.has_been_swept or pool.pool_type != pool_type:
                    continue

                if pool_type == 'BSL':
                    if pool.upper_bound < price <= pool.upper_bound + self.tolerance.max_distance:
                        pool.touches = 2
                        pool.upper_bound = float(price)
                        pool.last_touch_time = timestamp
                        pool.touch_history.append((timestamp, pool.touches, pool.lower_bound, pool.upper_bound))
                        return
                    elif pool.lower_bound <= price <= pool.upper_bound:
                        if (timestamp - pool.last_touch_time).total_seconds() >= 120:
                            pool.touches += 1
                            pool.lower_bound = float(min(pool.upper_bound, max(pool.lower_bound, price)))
                            pool.last_touch_time = timestamp
                            pool.touch_history.append((timestamp, pool.touches, pool.lower_bound, pool.upper_bound))
                        return

                elif pool_type == 'SSL':
                    if pool.lower_bound - self.tolerance.max_distance <= price < pool.lower_bound:
                        pool.touches = 2
                        pool.lower_bound = float(price)
                        pool.last_touch_time = timestamp
                        pool.touch_history.append((timestamp, pool.touches, pool.lower_bound, pool.upper_bound))
                        return
                    elif pool.lower_bound <= price <= pool.upper_bound:
                        if (timestamp - pool.last_touch_time).total_seconds() >= 120:
                            pool.touches += 1
                            pool.upper_bound = float(max(pool.lower_bound, min(pool.upper_bound, price)))
                            pool.last_touch_time = timestamp
                            pool.touch_history.append((timestamp, pool.touches, pool.lower_bound, pool.upper_bound))
                        return

        new_pool = LiquidityPool(
            pool_type=pool_type,
            upper_bound=float(price) if pool_type == 'BSL' else float(price + self.tolerance.max_distance),
            lower_bound=float(price - self.tolerance.max_distance) if pool_type == 'BSL' else float(price),
            creation_time=timestamp,
            session_origin=SessionManager.get_session(timestamp)
        )
        if bucket_key not in self.vault:
            self.vault[bucket_key] = []
        self.vault[bucket_key].append(new_pool)

    def clean_duplicate_pools(self):
        all_pools = self.get_all_pools()
        global_macros = [p for p in all_pools if getattr(p, 'source', '') == "Macro_Session"]

        if not global_macros:
            return

        def _is_duplicate(p: LiquidityPool) -> bool:
            if getattr(p, 'source', '') == "Macro_Session":
                return False
            for mp in global_macros:
                if p.pool_type == mp.pool_type and abs((p.creation_time - mp.creation_time).total_seconds()) <= 43200:
                    mp_price = mp.upper_bound if mp.pool_type == 'BSL' else mp.lower_bound
                    if abs(p.initial_price - mp_price) <= 1.0:
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
        bucket_key = self._get_bucket_key(price)

        new_pool = LiquidityPool(
            pool_type=pool_type,
            upper_bound=float(price) if pool_type == 'BSL' else float(price + self.tolerance.max_distance),
            lower_bound=float(price - self.tolerance.max_distance) if pool_type == 'BSL' else float(price),
            creation_time=timestamp,
            session_origin=session_name,
            source="Macro_Session",
            touches=1
        )

        if bucket_key not in self.vault:
            self.vault[bucket_key] = []
        self.vault[bucket_key].append(new_pool)

    def check_for_sweeps_and_breaks(self, high: float, low: float, close: float, timestamp: pd.Timestamp):
        start_bucket = self._get_bucket_key(low)
        end_bucket = self._get_bucket_key(high)
        step = int(self.bucket_size)
        keys_to_check = range(start_bucket, end_bucket + step, step)

        for key in keys_to_check:
            if key not in self.vault:
                continue

            active_in_bucket: List[LiquidityPool] = []
            for pool in self.vault[key]:
                if pool.is_active:
                    pool.evaluate_candle(high, low, close, timestamp)

                # Keep only unbroken pools in the active spatial bucket;
                # prune broken pools into archived_pools immediately.
                if pool.is_active:
                    active_in_bucket.append(pool)
                else:
                    self.archived_pools.append(pool)

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
    engine = LiquidityEngine(tolerance_ticks=tolerance_ticks)

    # Sort swings chronologically
    sorted_swings = sorted(swings, key=lambda x: x["time"])
    swing_idx = 0
    num_swings = len(sorted_swings)

    # Flatten and sort all macro session pools chronologically so they inject
    # cleanly on ANY resampled timeframe (1m, 5m, 15m, 1h, 4h)
    flat_macros: List[Dict[str, Any]] = []
    for ts in sorted(macro_pools_by_time.keys()):
        for m_pool in macro_pools_by_time[ts]:
            flat_macros.append(m_pool)
    macro_idx = 0
    num_macros = len(flat_macros)

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    for i in range(len(df)):
        current_time = times[i]

        # 1. Inject any swings that occurred up to `current_time`
        while swing_idx < num_swings and sorted_swings[swing_idx]["time"] <= current_time:
            swing = sorted_swings[swing_idx]
            if swing["type"] == "SwingHigh":
                engine.add_new_swing("BSL", swing["price"], current_time)
            elif swing["type"] == "SwingLow":
                engine.add_new_swing("SSL", swing["price"], current_time)
            swing_idx += 1

        # 2. Inject any Macro Session pools that occurred up to `current_time`
        while macro_idx < num_macros and flat_macros[macro_idx]["time"] <= current_time:
            m_pool = flat_macros[macro_idx]
            engine.inject_macro_session_pool(
                pool_type=m_pool["type"],
                price=m_pool["price"],
                timestamp=current_time,
                session_name=m_pool["session"],
            )
            macro_idx += 1

        # 3. Evaluate sweeps and breaks on the current candle
        if i >= 1:
            engine.check_for_sweeps_and_breaks(highs[i], lows[i], closes[i], current_time)

    engine.clean_duplicate_pools()

    return df, engine