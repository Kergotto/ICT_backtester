import pandas as pd
import numpy as np
from typing import List, Dict, Any


def detect_structure(df_price: pd.DataFrame, swings: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Detects Change of Character (CHoCH) for reversals and Break of Structure (BOS)
    for trend continuations. Requires a strict 1-tick close beyond the structural
    level, or a valid lower-timeframe gap-through.
    """
    df = df_price.copy()

    df['current_swing_high'] = np.nan
    df['current_swing_low'] = np.nan

    for s in swings:
        if s['type'] == 'SwingHigh':
            df.at[s['time'], 'current_swing_high'] = s['price']
        elif s['type'] == 'SwingLow':
            df.at[s['time'], 'current_swing_low'] = s['price']

    df['current_swing_high'] = df['current_swing_high'].ffill()
    df['current_swing_low'] = df['current_swing_low'].ffill()

    opens = df['open'].values
    closes = df['close'].values
    swing_highs = df['current_swing_high'].values
    swing_lows = df['current_swing_low'].values

    trends = np.zeros(len(df), dtype=int)
    choch_bull_flags = np.zeros(len(df), dtype=bool)
    choch_bear_flags = np.zeros(len(df), dtype=bool)
    bos_bull_flags = np.zeros(len(df), dtype=bool)
    bos_bear_flags = np.zeros(len(df), dtype=bool)

    current_trend = 0
    last_broken_high = np.nan
    last_broken_low = np.nan

    for i in range(len(df)):
        trends[i] = current_trend
        current_high = swing_highs[i]
        current_low = swing_lows[i]

        if not np.isnan(current_high):
            is_bull_break = (closes[i] > current_high) or (opens[i] > current_high and closes[i] == current_high)

            if is_bull_break and current_high != last_broken_high:
                if current_trend in [-1, 0]:
                    choch_bull_flags[i] = True
                    current_trend = 1
                elif current_trend == 1:
                    bos_bull_flags[i] = True

                trends[i] = current_trend
                last_broken_high = current_high

        if not np.isnan(current_low):
            is_bear_break = (closes[i] < current_low) or (opens[i] < current_low and closes[i] == current_low)

            if is_bear_break and current_low != last_broken_low:
                if current_trend in [1, 0]:
                    choch_bear_flags[i] = True
                    current_trend = -1
                elif current_trend == -1:
                    bos_bear_flags[i] = True

                trends[i] = current_trend
                last_broken_low = current_low

    df['trend'] = trends
    df['choch_bullish'] = choch_bull_flags
    df['choch_bearish'] = choch_bear_flags
    df['bos_bullish'] = bos_bull_flags
    df['bos_bearish'] = bos_bear_flags

    return df