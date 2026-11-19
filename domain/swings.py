import numpy as np
import pandas as pd


def detect_swings(df: pd.DataFrame):
    """
    Scans the dataframe for strict 2-candle structural swings.
    Swing Low = 1 Bearish Close -> 1 Bullish Close (minimum wick of the two).
    Swing High = 1 Bullish Close -> 1 Bearish Close (maximum wick of the two).
    Handles dojis by evaluating their close relative to the previous candle's close.
    """
    n = len(df)
    if n < 2:
        return []

    opens = df['open'].to_numpy(dtype=np.float64, copy=False)
    highs = df['high'].to_numpy(copy=False)
    lows = df['low'].to_numpy(copy=False)
    closes = df['close'].to_numpy(dtype=np.float64, copy=False)

    # 1. Evaluate synthetic colors for the entire DataFrame upfront
    prev_closes = np.empty(n, dtype=np.float64)
    prev_closes[0] = np.nan
    prev_closes[1:] = closes[:-1]

    is_green = closes > opens
    is_red = closes < opens
    is_doji = closes == opens

    # Resolve dojis: Bullish if closed >= prev close, Bearish if closed < prev close
    bullish_mask = is_green | (is_doji & (closes >= prev_closes))
    bearish_mask = is_red | (is_doji & (closes < prev_closes))

    # 2. Vectorized 2-candle transition masks (positions 0..n-2 correspond to bar indices 1..n-1)
    sl_mask = bearish_mask[:-1] & bullish_mask[1:]
    sh_mask = (~sl_mask) & bullish_mask[:-1] & bearish_mask[1:]
    any_pos = np.flatnonzero(sl_mask | sh_mask)

    if len(any_pos) == 0:
        return []

    bar_indices = any_pos + 1
    swing_times = df.index[bar_indices].tolist()
    min_lows = np.minimum(lows[any_pos], lows[bar_indices]).tolist()
    max_highs = np.maximum(highs[any_pos], highs[bar_indices]).tolist()
    is_sl = sl_mask[any_pos].tolist()

    return [
        {'type': 'SwingLow', 'price': min_lows[k], 'time': swing_times[k]}
        if is_sl[k]
        else {'type': 'SwingHigh', 'price': max_highs[k], 'time': swing_times[k]}
        for k in range(len(any_pos))
    ]