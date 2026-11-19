import bisect
from typing import List, Dict, Any
import numpy as np
import pandas as pd


def _ffill_1d(arr: np.ndarray) -> np.ndarray:
    n = len(arr)
    if n == 0:
        return arr
    mask = ~np.isnan(arr)
    if not mask.any():
        return arr
    idx = np.where(mask, np.arange(n, dtype=np.intp), 0)
    np.maximum.accumulate(idx, out=idx)
    return arr[idx]


def detect_structure(df_price: pd.DataFrame, swings: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Detects Change of Character (CHoCH) for reversals and Break of Structure (BOS)
    for trend continuations. Requires a strict 1-tick close beyond the structural
    level, or a valid lower-timeframe gap-through.
    """
    df = df_price.copy()
    n = len(df)
    if n == 0:
        df['current_swing_high'] = np.empty(0, dtype=np.float64)
        df['current_swing_low'] = np.empty(0, dtype=np.float64)
        df['trend'] = np.empty(0, dtype=int)
        df['choch_bullish'] = np.empty(0, dtype=bool)
        df['choch_bearish'] = np.empty(0, dtype=bool)
        df['bos_bullish'] = np.empty(0, dtype=bool)
        df['bos_bearish'] = np.empty(0, dtype=bool)
        return df

    sh_raw = np.full(n, np.nan, dtype=np.float64)
    sl_raw = np.full(n, np.nan, dtype=np.float64)
    fallback_swings: List[Dict[str, Any]] = []

    if swings:
        idx = df.index
        if isinstance(idx, pd.DatetimeIndex) and idx.is_monotonic_increasing and idx.is_unique:
            idx_ns = idx.as_unit("ns").asi8 if hasattr(idx, "as_unit") else idx.asi8
            idx_ns_list = idx_ns.tolist()
            bisect_left = bisect.bisect_left
            for s in swings:
                st = s['type']
                if st != 'SwingHigh' and st != 'SwingLow':
                    continue
                t = s['time']
                t_ns = (
                    int(t.as_unit("ns").value)
                    if hasattr(t, "as_unit")
                    else int(pd.Timestamp(t).as_unit("ns").value)
                )
                pos = bisect_left(idx_ns_list, t_ns)
                if pos < n and idx_ns_list[pos] == t_ns:
                    if st == 'SwingHigh':
                        sh_raw[pos] = s['price']
                    else:
                        sl_raw[pos] = s['price']
                else:
                    fallback_swings.append(s)
        else:
            fallback_swings = list(swings)

    if fallback_swings:
        df['current_swing_high'] = sh_raw
        df['current_swing_low'] = sl_raw
        for s in fallback_swings:
            if s['type'] == 'SwingHigh':
                df.at[s['time'], 'current_swing_high'] = s['price']
            elif s['type'] == 'SwingLow':
                df.at[s['time'], 'current_swing_low'] = s['price']
        df['current_swing_high'] = df['current_swing_high'].ffill()
        df['current_swing_low'] = df['current_swing_low'].ffill()
        swing_highs = df['current_swing_high'].to_numpy(dtype=np.float64, copy=False)
        swing_lows = df['current_swing_low'].to_numpy(dtype=np.float64, copy=False)
        n = len(df)
    else:
        swing_highs = _ffill_1d(sh_raw)
        swing_lows = _ffill_1d(sl_raw)
        df['current_swing_high'] = swing_highs
        df['current_swing_low'] = swing_lows

    opens = df['open'].to_numpy(dtype=np.float64, copy=False)
    closes = df['close'].to_numpy(dtype=np.float64, copy=False)

    trends = np.zeros(n, dtype=int)
    choch_bull_flags = np.zeros(n, dtype=bool)
    choch_bear_flags = np.zeros(n, dtype=bool)
    bos_bull_flags = np.zeros(n, dtype=bool)
    bos_bear_flags = np.zeros(n, dtype=bool)

    # IEEE-754 comparisons with NaN evaluate to False, matching `not np.isnan(...)`
    bull_cand = (closes > swing_highs) | ((opens > swing_highs) & (closes == swing_highs))
    bear_cand = (closes < swing_lows) | ((opens < swing_lows) & (closes == swing_lows))
    cand_indices = np.flatnonzero(bull_cand | bear_cand)

    if len(cand_indices) > 0:
        current_trend = 0
        last_broken_high = np.nan
        last_broken_low = np.nan
        prev_pos = 0

        for i in cand_indices.tolist():
            if current_trend != 0 and prev_pos < i:
                trends[prev_pos:i] = current_trend

            if bull_cand[i]:
                current_high = swing_highs[i]
                if current_high != last_broken_high:
                    if current_trend in (-1, 0):
                        choch_bull_flags[i] = True
                        current_trend = 1
                    elif current_trend == 1:
                        bos_bull_flags[i] = True
                    last_broken_high = current_high

            if bear_cand[i]:
                current_low = swing_lows[i]
                if current_low != last_broken_low:
                    if current_trend in (1, 0):
                        choch_bear_flags[i] = True
                        current_trend = -1
                    elif current_trend == -1:
                        bos_bear_flags[i] = True
                    last_broken_low = current_low

            trends[i] = current_trend
            prev_pos = i + 1

        if current_trend != 0 and prev_pos < n:
            trends[prev_pos:n] = current_trend

    df['trend'] = trends
    df['choch_bullish'] = choch_bull_flags
    df['choch_bearish'] = choch_bear_flags
    df['bos_bullish'] = bos_bull_flags
    df['bos_bearish'] = bos_bear_flags

    return df