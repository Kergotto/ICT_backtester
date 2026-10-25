from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class DealingRange:
    """
    Represents an ICT Dealing Range with 50% Equilibrium (EQ),
    Premium/Discount zones, and Optimal Trade Entry (OTE) bounds.
    """
    direction: str  # 'BULLISH' (Low -> High) or 'BEARISH' (High -> Low)
    low: float
    high: float
    start_time: pd.Timestamp
    creation_time: pd.Timestamp
    status: str = "ACTIVE"  # 'ACTIVE', 'EQ_MITIGATED', or 'BROKEN'
    eq_mitigated: bool = False
    eq_mitigation_time: Optional[pd.Timestamp] = None
    break_time: Optional[pd.Timestamp] = None
    events: List[Tuple[str, pd.Timestamp]] = field(default_factory=list)

    @property
    def eq(self) -> float:
        """Exact 50% Equilibrium price level."""
        return (self.high + self.low) / 2.0

    @property
    def range_size(self) -> float:
        """Total point size of the dealing range."""
        return self.high - self.low

    @property
    def ote_bounds(self) -> Tuple[float, float]:
        """
        Returns the (lower_bound, upper_bound) of the 62% - 79% retracement
        Optimal Trade Entry (OTE) zone based on range direction.
        """
        if self.direction == "BULLISH":
            ote_top = self.high - 0.62 * self.range_size
            ote_bottom = self.high - 0.79 * self.range_size
            return (ote_bottom, ote_top)
        else:
            ote_bottom = self.low + 0.62 * self.range_size
            ote_top = self.low + 0.79 * self.range_size
            return (ote_bottom, ote_top)

    def get_zone(self, price: float) -> str:
        """
        Classifies a given price as 'PREMIUM', 'DISCOUNT', or 'EQUILIBRIUM'.
        """
        if np.isclose(price, self.eq, atol=1e-5):
            return "EQUILIBRIUM"
        return "PREMIUM" if price > self.eq else "DISCOUNT"

    def is_in_ote(self, price: float) -> bool:
        """Checks if a price sits inside the 62% - 79% OTE zone."""
        lower, upper = self.ote_bounds
        return lower <= price <= upper

    def get_position_pct(self, price: float) -> float:
        """
        Returns the normalized position within the range:
        0.0 = Range Low, 0.5 = Equilibrium, 1.0 = Range High.
        """
        if self.range_size == 0:
            return 0.5
        return (price - self.low) / self.range_size


def track_dealing_ranges(df: pd.DataFrame, swings: List[Dict]) -> List[DealingRange]:
    """
    Constructs and tracks External Dealing Ranges from swing points and OHLC data.
    Internal swings inside an active range are ignored until the range is broken.
    """
    ranges: List[DealingRange] = []
    if len(swings) < 2:
        return ranges

    has_ohlc = (
        not df.empty
        and all(col in df.columns for col in ("high", "low", "close"))
    )
    if has_ohlc:
        times = df.index
        high_arr = df["high"].to_numpy()
        low_arr = df["low"].to_numpy()
        close_arr = df["close"].to_numpy()

    # 1. Establish the initial dealing range from the first opposing swing pair
    first_s = swings[0]
    init_idx = None
    for idx in range(1, len(swings)):
        if swings[idx]["type"] != first_s["type"]:
            init_idx = idx
            break

    if init_idx is None:
        return ranges

    second_s = swings[init_idx]
    if first_s["type"] == "SwingLow" and second_s["type"] == "SwingHigh":
        if second_s["price"] > first_s["price"]:
            current_range = DealingRange(
                direction="BULLISH",
                low=first_s["price"],
                high=second_s["price"],
                start_time=first_s["time"],
                creation_time=second_s["time"],
            )
            ranges.append(current_range)
        else:
            current_range = None
    else:
        if first_s["price"] > second_s["price"]:
            current_range = DealingRange(
                direction="BEARISH",
                low=second_s["price"],
                high=first_s["price"],
                start_time=first_s["time"],
                creation_time=second_s["time"],
            )
            ranges.append(current_range)
        else:
            current_range = None

    # 2. Iterate through subsequent swings to detect external range expansions
    for i in range(init_idx + 1, len(swings)):
        s = swings[i]

        if current_range is None:
            prev_s = swings[i - 1]
            if prev_s["type"] == "SwingLow" and s["type"] == "SwingHigh" and s["price"] > prev_s["price"]:
                current_range = DealingRange("BULLISH", prev_s["price"], s["price"], prev_s["time"], s["time"])
                ranges.append(current_range)
            elif prev_s["type"] == "SwingHigh" and s["type"] == "SwingLow" and prev_s["price"] > s["price"]:
                current_range = DealingRange("BEARISH", s["price"], prev_s["price"], prev_s["time"], s["time"])
                ranges.append(current_range)
            continue

        # Bullish expansion above current range high
        if s["type"] == "SwingHigh" and s["price"] > current_range.high:
            # If continuing Bullish, search lows after the previous high (creation_time);
            # If reversing from Bearish, include the Bearish low at creation_time.
            search_start = current_range.creation_time
            intervening_lows = [
                sw for sw in swings[:i]
                if sw["type"] == "SwingLow" and sw["time"] >= search_start
            ]
            anchor_low = min(intervening_lows, key=lambda x: x["price"]) if intervening_lows else {
                "price": current_range.low,
                "time": current_range.start_time,
            }

            current_range = DealingRange(
                direction="BULLISH",
                low=anchor_low["price"],
                high=s["price"],
                start_time=anchor_low["time"],
                creation_time=s["time"],
            )
            ranges.append(current_range)

        # Bearish expansion below current range low
        elif s["type"] == "SwingLow" and s["price"] < current_range.low:
            # If continuing Bearish, search highs after the previous low (creation_time);
            # If reversing from Bullish, include the Bullish high at creation_time.
            search_start = current_range.creation_time
            intervening_highs = [
                sw for sw in swings[:i]
                if sw["type"] == "SwingHigh" and sw["time"] >= search_start
            ]
            anchor_high = max(intervening_highs, key=lambda x: x["price"]) if intervening_highs else {
                "price": current_range.high,
                "time": current_range.start_time,
            }

            current_range = DealingRange(
                direction="BEARISH",
                low=s["price"],
                high=anchor_high["price"],
                start_time=anchor_high["time"],
                creation_time=s["time"],
            )
            ranges.append(current_range)

    # 3. Track EQ mitigation and range breaks using OHLC candle data
    if has_ohlc:
        for idx, dr in enumerate(ranges):
            start_pos = times.searchsorted(dr.creation_time, side="right")
            next_creation = ranges[idx + 1].creation_time if idx + 1 < len(ranges) else None
            end_pos = times.searchsorted(next_creation, side="right") if next_creation is not None else len(times)

            for pos in range(start_pos, end_pos):
                t = times[pos]
                c_high = high_arr[pos]
                c_low = low_arr[pos]
                c_close = close_arr[pos]

                # Check Equilibrium (50%) mitigation first
                if not dr.eq_mitigated:
                    if dr.direction == "BULLISH" and c_low <= dr.eq:
                        dr.eq_mitigated = True
                        dr.eq_mitigation_time = t
                        dr.status = "EQ_MITIGATED"
                        dr.events.append(("EQ_MITIGATED", t))
                    elif dr.direction == "BEARISH" and c_high >= dr.eq:
                        dr.eq_mitigated = True
                        dr.eq_mitigation_time = t
                        dr.status = "EQ_MITIGATED"
                        dr.events.append(("EQ_MITIGATED", t))

                # Check if candle body closes outside the Dealing Range bounds
                if c_close > dr.high or c_close < dr.low:
                    dr.status = "BROKEN"
                    dr.break_time = t
                    dr.events.append(("BREAK", t))
                    break

            # If superseded by a new range via wick expansion before a body close break
            if dr.break_time is None and next_creation is not None:
                dr.status = "BROKEN"
                dr.break_time = next_creation
                dr.events.append(("BREAK", next_creation))

    return ranges


def get_active_dealing_range(
    ranges: List[DealingRange], timestamp: pd.Timestamp
) -> Optional[DealingRange]:
    """
    Returns the most recent DealingRange that was active at the given timestamp.
    """
    active = None
    for dr in ranges:
        if dr.creation_time <= timestamp:
            if dr.break_time is None or timestamp <= dr.break_time:
                active = dr
        else:
            break
    return active