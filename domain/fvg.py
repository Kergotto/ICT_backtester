# Copyright (C) 2026 Kyrylo Homberh
# SPDX-License-Identifier: AGPL-3.0-only

from dataclasses import dataclass, field
import heapq
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass(slots=True)
class FairValueGap:
    fvg_type: str  # 'BISI' (Bullish) or 'SIBI' (Bearish)
    top: float
    bottom: float
    creation_time: pd.Timestamp
    ce: float = field(init=False)  # Consequent Encroachment (50% level)

    # Dynamic tracking for the future AI mentor metadata
    unmitigated_top: float = field(init=False)
    unmitigated_bottom: float = field(init=False)
    status: str = "Active"  # Possible statuses: Active, Partially_Mitigated, Filled, Inverted
    events: List[Tuple[str, pd.Timestamp, float]] = field(default_factory=list)

    def __post_init__(self):
        self.ce = (self.top + self.bottom) / 2.0
        self.unmitigated_top = self.top
        self.unmitigated_bottom = self.bottom

    @property
    def is_active(self) -> bool:
        # Include "Filled" so the gap stays alive to listen for an "Inverted" breakout
        return self.status in ("Active", "Partially_Mitigated", "Filled")

    def evaluate_candle(self, high: float, low: float, close: float, timestamp: pd.Timestamp) -> None:
        if not self.is_active or timestamp <= self.creation_time:
            return

        if self.fvg_type == 'BISI':  # Bullish FVG (Support)
            # 1. Inverted: Body closed BELOW the gap
            if close < self.bottom:
                self.status = "Inverted"
                self.events.append(("INVERTED", timestamp, close))
                return

            # If the gap is already Filled, it only waits for inversion. Skip other checks.
            if self.status == "Filled":
                return

            # 2. Filled: Body closed INSIDE the gap
            if self.bottom <= close <= self.top:
                self.status = "Filled"
                self.events.append(("FILLED_BY_CLOSE", timestamp, close))
                self.unmitigated_top = min(self.unmitigated_top, low)  # Record the depth of penetration for metadata
                return

            # 3. Partially Mitigated: Wick entered the gap
            if low < self.unmitigated_top:
                self.unmitigated_top = low
                if self.status == "Active":
                    self.status = "Partially_Mitigated"
                self.events.append(("MITIGATED", timestamp, low))

                # If the wick pierced the entire gap but closed above it, consider it Filled
                if low <= self.bottom:
                    self.status = "Filled"
                    self.events.append(("FILLED_BY_WICK", timestamp, low))

        elif self.fvg_type == 'SIBI':  # Bearish FVG (Resistance)
            # 1. Inverted: Body closed ABOVE the gap
            if close > self.top:
                self.status = "Inverted"
                self.events.append(("INVERTED", timestamp, close))
                return

            if self.status == "Filled":
                return

            # 2. Filled: Body closed INSIDE the gap
            if self.bottom <= close <= self.top:
                self.status = "Filled"
                self.events.append(("FILLED_BY_CLOSE", timestamp, close))
                self.unmitigated_bottom = max(self.unmitigated_bottom, high)
                return

            # 3. Partially Mitigated: Wick entered the gap
            if high > self.unmitigated_bottom:
                self.unmitigated_bottom = high
                if self.status == "Active":
                    self.status = "Partially_Mitigated"
                self.events.append(("MITIGATED", timestamp, high))

                if high >= self.top:
                    self.status = "Filled"
                    self.events.append(("FILLED_BY_WICK", timestamp, high))


def _detect_fvgs_indexed(df: pd.DataFrame) -> Tuple[List[FairValueGap], List[int]]:
    n = len(df)
    if n < 3:
        return [], []

    highs = df['high'].to_numpy(dtype=float)
    lows = df['low'].to_numpy(dtype=float)
    times = df.index

    c1_high = highs[:-2]
    c1_low = lows[:-2]
    c3_high = highs[2:]
    c3_low = lows[2:]

    bisi_mask = c3_low > c1_high
    sibi_mask = (~bisi_mask) & (c1_low > c3_high)
    any_indices = np.flatnonzero(bisi_mask | sibi_mask)

    fvgs: List[FairValueGap] = []
    bar_indices: List[int] = []

    for k in any_indices.tolist():
        bar_i = k + 2
        if bisi_mask[k]:
            fvgs.append(FairValueGap(
                fvg_type='BISI',
                top=float(c3_low[k]),
                bottom=float(c1_high[k]),
                creation_time=times[bar_i],
            ))
        else:
            fvgs.append(FairValueGap(
                fvg_type='SIBI',
                top=float(c1_low[k]),
                bottom=float(c3_high[k]),
                creation_time=times[bar_i],
            ))
        bar_indices.append(bar_i)

    return fvgs, bar_indices


def detect_fvgs(df: pd.DataFrame) -> List[FairValueGap]:
    """
    Scans historical data and detects all classic 3-candle FVG patterns.
    """
    fvgs, _ = _detect_fvgs_indexed(df)
    return fvgs


def track_fvgs(df: pd.DataFrame) -> List[FairValueGap]:
    """
    Detects FVGs and runs them through chronological price action
    to track their lifecycle (Mitigated, Filled, Inverted) without lookahead bias.
    """
    fvgs, bar_indices = _detect_fvgs_indexed(df)
    n = len(df)
    if not fvgs or n == 0:
        return fvgs

    highs = df['high'].to_numpy(dtype=float).tolist()
    lows = df['low'].to_numpy(dtype=float).tolist()
    closes = df['close'].to_numpy(dtype=float).tolist()
    times = df.index

    bisi_unfilled: List[Tuple[float, int, FairValueGap]] = []
    bisi_filled: List[Tuple[float, int, FairValueGap]] = []
    sibi_unfilled: List[Tuple[float, int, FairValueGap]] = []
    sibi_filled: List[Tuple[float, int, FairValueGap]] = []

    heappush = heapq.heappush
    heappop = heapq.heappop

    num_fvgs = len(fvgs)
    fvg_idx = 0
    next_fvg_bar = bar_indices[0]

    for i in range(n):
        h = highs[i]
        l = lows[i]
        c = closes[i]
        min_lc = l if l < c else c
        max_hc = h if h > c else c

        current_time: Optional[pd.Timestamp] = None

        # 1. Filled BISI: only transitions to Inverted when close < bottom
        if bisi_filled and -bisi_filled[0][0] > c:
            current_time = times[i]
            repush_bf = []
            while bisi_filled and -bisi_filled[0][0] > c:
                entry = heappop(bisi_filled)
                fvg = entry[2]
                fvg.evaluate_candle(h, l, c, current_time)
                if fvg.status != "Inverted":
                    repush_bf.append(entry)
            for entry in repush_bf:
                heappush(bisi_filled, entry)

        # 2. Unfilled BISI: can only transition when min(low, close) <= top
        if bisi_unfilled and -bisi_unfilled[0][0] >= min_lc:
            if current_time is None:
                current_time = times[i]
            repush_bu = []
            while bisi_unfilled and -bisi_unfilled[0][0] >= min_lc:
                entry = heappop(bisi_unfilled)
                fvg = entry[2]
                fvg.evaluate_candle(h, l, c, current_time)
                st = fvg.status
                if st == "Inverted":
                    continue
                elif st == "Filled":
                    heappush(bisi_filled, (-fvg.bottom, entry[1], fvg))
                else:
                    repush_bu.append(entry)
            for entry in repush_bu:
                heappush(bisi_unfilled, entry)

        # 3. Filled SIBI: only transitions to Inverted when close > top
        if sibi_filled and sibi_filled[0][0] < c:
            if current_time is None:
                current_time = times[i]
            repush_sf = []
            while sibi_filled and sibi_filled[0][0] < c:
                entry = heappop(sibi_filled)
                fvg = entry[2]
                fvg.evaluate_candle(h, l, c, current_time)
                if fvg.status != "Inverted":
                    repush_sf.append(entry)
            for entry in repush_sf:
                heappush(sibi_filled, entry)

        # 4. Unfilled SIBI: can only transition when max(high, close) >= bottom
        if sibi_unfilled and sibi_unfilled[0][0] <= max_hc:
            if current_time is None:
                current_time = times[i]
            repush_su = []
            while sibi_unfilled and sibi_unfilled[0][0] <= max_hc:
                entry = heappop(sibi_unfilled)
                fvg = entry[2]
                fvg.evaluate_candle(h, l, c, current_time)
                st = fvg.status
                if st == "Inverted":
                    continue
                elif st == "Filled":
                    heappush(sibi_filled, (fvg.top, entry[1], fvg))
                else:
                    repush_su.append(entry)
            for entry in repush_su:
                heappush(sibi_unfilled, entry)

        # 5. Introduce newly formed FVGs at bar i (evaluate_candle skips timestamp <= creation_time)
        if i == next_fvg_bar:
            while fvg_idx < num_fvgs and bar_indices[fvg_idx] == i:
                new_fvg = fvgs[fvg_idx]
                if new_fvg.fvg_type == 'BISI':
                    heappush(bisi_unfilled, (-new_fvg.top, fvg_idx, new_fvg))
                else:
                    heappush(sibi_unfilled, (new_fvg.bottom, fvg_idx, new_fvg))
                fvg_idx += 1
            next_fvg_bar = bar_indices[fvg_idx] if fvg_idx < num_fvgs else -1

    return fvgs