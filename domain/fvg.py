from dataclasses import dataclass, field
from typing import List, Tuple
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
        return self.status in ["Active", "Partially_Mitigated", "Filled"]

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


def detect_fvgs(df: pd.DataFrame) -> List[FairValueGap]:
    """
    Scans historical data and detects all classic 3-candle FVG patterns.
    """
    fvgs = []
    highs = df['high'].values
    lows = df['low'].values
    times = df.index

    for i in range(2, len(df)):
        c1_high, c1_low = highs[i - 2], lows[i - 2]
        c3_high, c3_low = highs[i], lows[i]

        # Bullish FVG (BISI): 3rd candle Low is higher than 1st candle High
        if c3_low > c1_high:
            fvgs.append(FairValueGap(
                fvg_type='BISI',
                top=float(c3_low),
                bottom=float(c1_high),
                creation_time=times[i]
            ))

        # Bearish FVG (SIBI): 3rd candle High is lower than 1st candle Low
        elif c1_low > c3_high:
            fvgs.append(FairValueGap(
                fvg_type='SIBI',
                top=float(c1_low),
                bottom=float(c3_high),
                creation_time=times[i]
            ))

    return fvgs


def track_fvgs(df: pd.DataFrame) -> List[FairValueGap]:
    """
    Detects FVGs and runs them through chronological price action
    to track their lifecycle (Mitigated, Filled, Inverted) without lookahead bias.
    """
    fvgs = detect_fvgs(df)
    active_fvgs = []
    fvg_idx = 0

    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    times = df.index

    for i in range(len(df)):
        current_time = times[i]

        # Introduce newly formed FVGs at their precise creation time
        while fvg_idx < len(fvgs) and fvgs[fvg_idx].creation_time == current_time:
            active_fvgs.append(fvgs[fvg_idx])
            fvg_idx += 1

        # Evaluate active FVGs against the current candle
        for fvg in active_fvgs:
            if fvg.is_active:
                fvg.evaluate_candle(highs[i], lows[i], closes[i], current_time)

    return fvgs