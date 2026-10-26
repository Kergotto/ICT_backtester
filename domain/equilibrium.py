from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class DealingRange:
    """
    Represents a single-leg ICT Equilibrium range (0, 0.5, 1) strictly from
    one Swing Low to the immediate next Swing High (or Swing High to Swing Low).
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
        """Exact 0.5 (50%) Equilibrium price level."""
        return (self.high + self.low) / 2.0

    @property
    def range_size(self) -> float:
        """Total point size of the single-leg dealing range."""
        return self.high - self.low

    def get_zone(self, price: float) -> str:
        """
        Classifies a given price as 'PREMIUM' (> 0.5), 'DISCOUNT' (< 0.5), or 'EQUILIBRIUM'.
        """
        if np.isclose(price, self.eq, atol=1e-5):
            return "EQUILIBRIUM"
        return "PREMIUM" if price > self.eq else "DISCOUNT"

    def get_position_pct(self, price: float) -> float:
        """
        Returns the normalized position within the range:
        0.0 = Range Low, 0.5 = Equilibrium, 1.0 = Range High.
        """
        if self.range_size == 0:
            return 0.5
        return (price - self.low) / self.range_size


def _build_alternating_swings(swings: List[Dict]) -> List[Dict]:
    """
    Collapses consecutive same-type swings to their most extreme pivot
    so every adjacent pair forms a valid (SwingLow <-> SwingHigh) leg.
    """
    if not swings:
        return []

    alt: List[Dict] = [swings[0]]
    for s in swings[1:]:
        prev = alt[-1]
        if s["type"] != prev["type"]:
            alt.append(s)
        else:
            if s["type"] == "SwingHigh" and s["price"] >= prev["price"]:
                alt[-1] = s
            elif s["type"] == "SwingLow" and s["price"] <= prev["price"]:
                alt[-1] = s
    return alt


def track_dealing_ranges(df: pd.DataFrame, swings: List[Dict]) -> List[DealingRange]:
    """
    Constructs and tracks single-leg Dealing Ranges (Equilibrium 0 - 0.5 - 1)
    on EVERY SwingLow -> SwingHigh and SwingHigh -> SwingLow leg, including
    internal legs that form inside larger ranges.
    """
    ranges: List[DealingRange] = []
    alt_swings = _build_alternating_swings(swings)
    if len(alt_swings) < 2:
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

    # 1. Spawn an Equilibrium range on every adjacent High/Low or Low/High leg
    for i in range(1, len(alt_swings)):
        prev_s = alt_swings[i - 1]
        s = alt_swings[i]

        if prev_s["type"] == "SwingLow" and s["type"] == "SwingHigh":
            if s["price"] > prev_s["price"]:
                ranges.append(
                    DealingRange(
                        direction="BULLISH",
                        low=prev_s["price"],
                        high=s["price"],
                        start_time=prev_s["time"],
                        creation_time=s["time"],
                    )
                )
        elif prev_s["type"] == "SwingHigh" and s["type"] == "SwingLow":
            if prev_s["price"] > s["price"]:
                ranges.append(
                    DealingRange(
                        direction="BEARISH",
                        low=s["price"],
                        high=prev_s["price"],
                        start_time=prev_s["time"],
                        creation_time=s["time"],
                    )
                )

    # 2. Track 50% Equilibrium mitigation and range breaks on OHLC candles
    if has_ohlc:
        for dr in ranges:
            start_pos = times.searchsorted(dr.creation_time, side="right")
            end_pos = len(times)

            for pos in range(start_pos, end_pos):
                t = times[pos]
                c_high = high_arr[pos]
                c_low = low_arr[pos]
                c_close = close_arr[pos]

                # Check Equilibrium (0.5) mitigation first
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