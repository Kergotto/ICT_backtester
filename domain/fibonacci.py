from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import pandas as pd


@dataclass
class FibonacciRange:
    """
    Represents a multi-swing ICT Fibonacci Retracement range.
    - Spans from the trend's lowest low (1.0) to the highest point/candle wick (0.0)
      in a Bullish trend, or highest high (1.0) to lowest point/candle wick (0.0)
      in a Bearish trend.
    - Tracks 0.382, 0.5, and Optimal Trade Entry (OTE: 0.62 - 0.705 - 0.79) levels.
    """
    direction: str  # 'BULLISH' or 'BEARISH'
    anchor_price: float  # 1.0 level (Lowest Low for Bullish, Highest High for Bearish)
    extreme_price: float  # 0.0 level (Highest Point for Bullish, Lowest Point for Bearish)
    start_time: pd.Timestamp
    extreme_time: pd.Timestamp
    swing_count: int = 1
    status: str = "ACTIVE"  # 'ACTIVE', 'OTE_MITIGATED', or 'BROKEN'
    ote_mitigated: bool = False
    ote_mitigation_time: Optional[pd.Timestamp] = None
    break_time: Optional[pd.Timestamp] = None
    events: List[Tuple[str, pd.Timestamp]] = field(default_factory=list)

    @property
    def low(self) -> float:
        return min(self.anchor_price, self.extreme_price)

    @property
    def high(self) -> float:
        return max(self.anchor_price, self.extreme_price)

    @property
    def range_size(self) -> float:
        return self.high - self.low

    def get_level(self, fib_ratio: float) -> float:
        """
        Calculates the price at a given Fibonacci retracement ratio:
        0.0 = Extreme point (end of trend move)
        0.382 = 38.2% retracement
        0.5 = 50.0% retracement
        0.62 = 62.0% retracement (OTE start)
        0.705 = 70.5% retracement (OTE sweet spot)
        0.79 = 79.0% retracement (OTE end)
        1.0 = Anchor point (start of trend move)
        """
        if self.direction == "BULLISH":
            return self.extreme_price - fib_ratio * self.range_size
        else:
            return self.extreme_price + fib_ratio * self.range_size

    @property
    def levels(self) -> Dict[float, float]:
        """Returns the standard ICT Fibonacci retracement levels dictionary."""
        return {
            0.0: self.get_level(0.0),
            0.382: self.get_level(0.382),
            0.5: self.get_level(0.5),
            0.62: self.get_level(0.62),
            0.705: self.get_level(0.705),
            0.79: self.get_level(0.79),
            1.0: self.get_level(1.0),
        }

    @property
    def ote_bounds(self) -> Tuple[float, float]:
        """Returns (lower_bound, upper_bound) of the 0.62 - 0.79 OTE zone."""
        lvl_62 = self.get_level(0.62)
        lvl_79 = self.get_level(0.79)
        return (min(lvl_62, lvl_79), max(lvl_62, lvl_79))

    @property
    def ote_sweet_spot(self) -> float:
        """Returns the 0.705 OTE midpoint level."""
        return self.get_level(0.705)

    def is_in_ote(self, price: float) -> bool:
        """Checks if a given price is inside the 0.62 - 0.79 OTE retracement zone."""
        lower, upper = self.ote_bounds
        return lower <= price <= upper


def track_fibonacci_ranges(df: pd.DataFrame, swings: List[Dict]) -> List[FibonacciRange]:
    """
    Tracks multi-swing trend legs for Fibonacci Retracement & OTE analysis.
    Spans across multiple higher highs and higher lows (uptrend) or lower lows and
    lower highs (downtrend), anchoring 0.0 at the highest/lowest candle point in the range.
    """
    fib_ranges: List[FibonacciRange] = []
    if len(swings) < 2 or df.empty:
        return fib_ranges

    times = df.index
    high_arr = df["high"].to_numpy()
    low_arr = df["low"].to_numpy()
    close_arr = df["close"].to_numpy()

    # Build alternating swing sequence
    alt_swings: List[Dict] = [swings[0]]
    for s in swings[1:]:
        if s["type"] != alt_swings[-1]["type"]:
            alt_swings.append(s)
        else:
            if s["type"] == "SwingHigh" and s["price"] >= alt_swings[-1]["price"]:
                alt_swings[-1] = s
            elif s["type"] == "SwingLow" and s["price"] <= alt_swings[-1]["price"]:
                alt_swings[-1] = s

    if len(alt_swings) < 2:
        return fib_ranges

    current_fib: Optional[FibonacciRange] = None
    last_internal_low: Optional[float] = None
    last_internal_high: Optional[float] = None

    for i in range(1, len(alt_swings)):
        s = alt_swings[i]
        prev_s = alt_swings[i - 1]

        if current_fib is None:
            if prev_s["type"] == "SwingLow" and s["type"] == "SwingHigh" and s["price"] > prev_s["price"]:
                current_fib = FibonacciRange(
                    direction="BULLISH",
                    anchor_price=prev_s["price"],
                    extreme_price=s["price"],
                    start_time=prev_s["time"],
                    extreme_time=s["time"],
                    swing_count=1,
                )
                last_internal_low = prev_s["price"]
                fib_ranges.append(current_fib)
            elif prev_s["type"] == "SwingHigh" and s["type"] == "SwingLow" and prev_s["price"] > s["price"]:
                current_fib = FibonacciRange(
                    direction="BEARISH",
                    anchor_price=prev_s["price"],
                    extreme_price=s["price"],
                    start_time=prev_s["time"],
                    extreme_time=s["time"],
                    swing_count=1,
                )
                last_internal_high = prev_s["price"]
                fib_ranges.append(current_fib)
            continue

        if current_fib.direction == "BULLISH":
            if s["type"] == "SwingLow":
                # Reverses to BEARISH if price breaks below anchor low or breaks the prior higher low
                if s["price"] < current_fib.anchor_price or (
                    last_internal_low is not None and s["price"] < last_internal_low
                ):
                    current_fib.status = "BROKEN"
                    current_fib.break_time = s["time"]
                    current_fib = FibonacciRange(
                        direction="BEARISH",
                        anchor_price=current_fib.extreme_price,
                        extreme_price=s["price"],
                        start_time=current_fib.extreme_time,
                        extreme_time=s["time"],
                        swing_count=1,
                    )
                    last_internal_high = current_fib.anchor_price
                    fib_ranges.append(current_fib)
                else:
                    last_internal_low = s["price"]
            elif s["type"] == "SwingHigh":
                if s["price"] > current_fib.extreme_price:
                    current_fib.extreme_price = s["price"]
                    current_fib.extreme_time = s["time"]
                    current_fib.swing_count += 1
                else:
                    # Failed to make a higher high; if a prior pullback low exists,
                    # start tracking this new bearish leg if next low breaks down.
                    last_internal_high = s["price"]

        elif current_fib.direction == "BEARISH":
            if s["type"] == "SwingHigh":
                # Reverses to BULLISH if price breaks above anchor high or breaks the prior lower high
                if s["price"] > current_fib.anchor_price or (
                    last_internal_high is not None and s["price"] > last_internal_high
                ):
                    current_fib.status = "BROKEN"
                    current_fib.break_time = s["time"]
                    current_fib = FibonacciRange(
                        direction="BULLISH",
                        anchor_price=current_fib.extreme_price,
                        extreme_price=s["price"],
                        start_time=current_fib.extreme_time,
                        extreme_time=s["time"],
                        swing_count=1,
                    )
                    last_internal_low = current_fib.anchor_price
                    fib_ranges.append(current_fib)
                else:
                    last_internal_high = s["price"]
            elif s["type"] == "SwingLow":
                if s["price"] < current_fib.extreme_price:
                    current_fib.extreme_price = s["price"]
                    current_fib.extreme_time = s["time"]
                    current_fib.swing_count += 1
                else:
                    last_internal_low = s["price"]

    # Refine 0.0 extreme to the highest/lowest candle point in the range
    # and track OTE (0.62 - 0.79) mitigation & breaks.
    for fib in fib_ranges:
        start_pos = times.searchsorted(fib.start_time, side="left")
        end_time = fib.break_time if fib.break_time is not None else times[-1]
        end_pos = times.searchsorted(end_time, side="right")

        extreme_search_end = times.searchsorted(fib.extreme_time, side="right")
        window_end = min(len(times), max(extreme_search_end + 3, start_pos + 1))
        if fib.direction == "BULLISH":
            for pos in range(start_pos, window_end):
                if high_arr[pos] >= fib.extreme_price:
                    fib.extreme_price = float(high_arr[pos])
                    fib.extreme_time = times[pos]
        else:
            for pos in range(start_pos, window_end):
                if low_arr[pos] <= fib.extreme_price:
                    fib.extreme_price = float(low_arr[pos])
                    fib.extreme_time = times[pos]

        scan_start = times.searchsorted(fib.extreme_time, side="right")
        ote_low, ote_high = fib.ote_bounds

        for pos in range(scan_start, end_pos):
            t = times[pos]
            c_high = high_arr[pos]
            c_low = low_arr[pos]
            c_close = close_arr[pos]

            if not fib.ote_mitigated:
                if fib.direction == "BULLISH" and c_low <= ote_high:
                    fib.ote_mitigated = True
                    fib.ote_mitigation_time = t
                    if fib.status == "ACTIVE":
                        fib.status = "OTE_MITIGATED"
                    fib.events.append(("OTE_MITIGATED", t))
                elif fib.direction == "BEARISH" and c_high >= ote_low:
                    fib.ote_mitigated = True
                    fib.ote_mitigation_time = t
                    if fib.status == "ACTIVE":
                        fib.status = "OTE_MITIGATED"
                    fib.events.append(("OTE_MITIGATED", t))

            if fib.direction == "BULLISH" and c_close < fib.anchor_price:
                fib.status = "BROKEN"
                fib.break_time = t
                fib.events.append(("BREAK", t))
                break
            elif fib.direction == "BEARISH" and c_close > fib.anchor_price:
                fib.status = "BROKEN"
                fib.break_time = t
                fib.events.append(("BREAK", t))
                break

    return fib_ranges