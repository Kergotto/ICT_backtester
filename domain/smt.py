from typing import List, Dict, Optional, Any
import bisect
import numpy as np
import pandas as pd

from core.data_loader import resample_timeframe
from domain.swings import detect_swings


# Official SMT Lifecycle Status Constants
SMT_PENDING = "PENDING"
SMT_CONFIRMED = "CONFIRMED"      # LTF MSS confirmed; setup is live and tradable
SMT_RESPECTED = "RESPECTED"      # Opposing target low/high swept after confirmation (Done)
SMT_INVALIDATED = "INVALIDATED"  # Protective SMT high/low breached after confirmation (Eliminated)


# Multi-Timeframe SMT -> Lower Timeframe MSS Confirmation Hierarchy
TIMEFRAME_HIERARCHY: Dict[str, Dict[str, object]] = {
    "1min":  {"htf_min": 1,   "ltf_rule": "1min",  "ltf_min": 1,  "ltf_label": "1m",  "sync_bars": 3},
    "1m":    {"htf_min": 1,   "ltf_rule": "1min",  "ltf_min": 1,  "ltf_label": "1m",  "sync_bars": 3},
    "5min":  {"htf_min": 5,   "ltf_rule": "1min",  "ltf_min": 1,  "ltf_label": "1m",  "sync_bars": 3},
    "5m":    {"htf_min": 5,   "ltf_rule": "1min",  "ltf_min": 1,  "ltf_label": "1m",  "sync_bars": 3},
    "15min": {"htf_min": 15,  "ltf_rule": "5min",  "ltf_min": 5,  "ltf_label": "5m",  "sync_bars": 3},
    "15m":   {"htf_min": 15,  "ltf_rule": "5min",  "ltf_min": 5,  "ltf_label": "5m",  "sync_bars": 3},
    "30min": {"htf_min": 30,  "ltf_rule": "15min", "ltf_min": 15, "ltf_label": "15m", "sync_bars": 3},
    "30m":   {"htf_min": 30,  "ltf_rule": "15min", "ltf_min": 15, "ltf_label": "15m", "sync_bars": 3},
    "1h":    {"htf_min": 60,  "ltf_rule": "15min", "ltf_min": 15, "ltf_label": "15m", "sync_bars": 3},
    "1hr":   {"htf_min": 60,  "ltf_rule": "15min", "ltf_min": 15, "ltf_label": "15m", "sync_bars": 3},
    "4h":    {"htf_min": 240, "ltf_rule": "1h",    "ltf_min": 60, "ltf_label": "1h",  "sync_bars": 3},
    "4hr":   {"htf_min": 240, "ltf_rule": "1h",    "ltf_min": 60, "ltf_label": "1h",  "sync_bars": 3},
}


def get_smt_status_at(smt: Dict[str, Any], timestamp: pd.Timestamp) -> str:
    """
    Returns the exact Zero-Lookahead status of an SMT event at `timestamp`.
    Accounts for the LTF confirmation candle's closing offset (`confirmation_close_time`).
    """
    conf_time = smt.get("confirmation_close_time", smt.get("confirmation_time"))
    if conf_time is None or timestamp < conf_time:
        return SMT_PENDING

    res_time = smt.get("resolved_close_time", smt.get("resolved_time"))
    if res_time is not None and timestamp >= res_time:
        return str(smt.get("status", SMT_CONFIRMED))

    return SMT_CONFIRMED


def is_smt_active_at(smt: Dict[str, Any], timestamp: pd.Timestamp) -> bool:
    """Returns True if the SMT is confirmed and has not yet been respected or invalidated at `timestamp`."""
    return get_smt_status_at(smt, timestamp) == SMT_CONFIRMED


def get_closest_swing_fast(target_time, target_swings, target_times, max_diff_minutes=3):
    if not target_times:
        return None
    idx = bisect.bisect_left(target_times, target_time)
    best_match = None
    min_diff = pd.Timedelta(minutes=max_diff_minutes)

    for i in (idx - 1, idx, idx + 1):
        if 0 <= i < len(target_times):
            diff = abs(target_times[i] - target_time)
            if diff <= min_diff:
                min_diff = diff
                best_match = target_swings[i]
    return best_match


def _find_ltf_sweep_start(
    htf_sweep_time: pd.Timestamp,
    htf_min: int,
    ltf_times: pd.DatetimeIndex,
    ltf_extreme_arr: np.ndarray,
    prev_level: float,
    is_bearish: bool,
) -> pd.Timestamp:
    if htf_min <= 1 or len(ltf_times) == 0 or len(ltf_extreme_arr) == 0:
        return htf_sweep_time

    win_end = htf_sweep_time + pd.Timedelta(minutes=htf_min)
    start_idx = ltf_times.searchsorted(htf_sweep_time, side="left")
    end_idx = ltf_times.searchsorted(win_end, side="left")

    for idx in range(start_idx, min(end_idx, len(ltf_times))):
        if is_bearish and ltf_extreme_arr[idx] > prev_level:
            return ltf_times[idx]
        if not is_bearish and ltf_extreme_arr[idx] < prev_level:
            return ltf_times[idx]

    return htf_sweep_time


def _find_intervening_extreme(
    swings: List[Dict],
    swing_times: List[pd.Timestamp],
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    find_min: bool,
) -> Optional[float]:
    s_idx = bisect.bisect_right(swing_times, start_time)
    e_idx = bisect.bisect_left(swing_times, end_time)
    if s_idx >= e_idx:
        return None
    prices = [swings[k]["price"] for k in range(s_idx, e_idx)]
    return float(min(prices)) if find_min else float(max(prices))


def _find_prior_target_beyond_confirm(
    swings: List[Dict],
    swing_times: List[pd.Timestamp],
    before_time: pd.Timestamp,
    reference_price: float,
    is_bearish: bool,
    lookback_count: int = 15,
) -> Optional[float]:
    end_idx = bisect.bisect_left(swing_times, before_time)
    start_idx = max(0, end_idx - lookback_count)
    candidates = []
    for k in range(start_idx, end_idx):
        p = float(swings[k]["price"])
        if is_bearish and p < reference_price:
            candidates.append(p)
        elif not is_bearish and p > reference_price:
            candidates.append(p)
    if not candidates:
        return None
    return max(candidates) if is_bearish else min(candidates)


def _evaluate_post_confirmation_lifecycle(
    is_bearish: bool,
    confirm_time: pd.Timestamp,
    es_inv_level: float,
    nq_inv_level: float,
    es_target_level: Optional[float],
    nq_target_level: Optional[float],
    ltf_es_times: pd.DatetimeIndex,
    ltf_es_high_arr: np.ndarray,
    ltf_es_low_arr: np.ndarray,
    has_ltf_es_ohlc: bool,
    ltf_nq_times: pd.DatetimeIndex,
    ltf_nq_high_arr: np.ndarray,
    ltf_nq_low_arr: np.ndarray,
    has_ltf_nq_ohlc: bool,
) -> tuple[str, Optional[pd.Timestamp]]:
    es_start = ltf_es_times.searchsorted(confirm_time, side="right") if has_ltf_es_ohlc else len(ltf_es_times)
    nq_start = ltf_nq_times.searchsorted(confirm_time, side="right") if has_ltf_nq_ohlc else len(ltf_nq_times)

    es_ptr = es_start
    nq_ptr = nq_start
    n_es = len(ltf_es_times) if has_ltf_es_ohlc else 0
    n_nq = len(ltf_nq_times) if has_ltf_nq_ohlc else 0

    while es_ptr < n_es or nq_ptr < n_nq:
        t_es = ltf_es_times[es_ptr] if es_ptr < n_es else None
        t_nq = ltf_nq_times[nq_ptr] if nq_ptr < n_nq else None

        if t_es is not None and (t_nq is None or t_es <= t_nq):
            t = t_es
        else:
            t = t_nq

        if t_es == t:
            if is_bearish and ltf_es_high_arr[es_ptr] >= es_inv_level:
                return SMT_INVALIDATED, t
            if not is_bearish and ltf_es_low_arr[es_ptr] <= es_inv_level:
                return SMT_INVALIDATED, t
        if t_nq == t:
            if is_bearish and ltf_nq_high_arr[nq_ptr] >= nq_inv_level:
                return SMT_INVALIDATED, t
            if not is_bearish and ltf_nq_low_arr[nq_ptr] <= nq_inv_level:
                return SMT_INVALIDATED, t

        if t_es == t and es_target_level is not None:
            if is_bearish and ltf_es_low_arr[es_ptr] < es_target_level:
                return SMT_RESPECTED, t
            if not is_bearish and ltf_es_high_arr[es_ptr] > es_target_level:
                return SMT_RESPECTED, t
        if t_nq == t and nq_target_level is not None:
            if is_bearish and ltf_nq_low_arr[nq_ptr] < nq_target_level:
                return SMT_RESPECTED, t
            if not is_bearish and ltf_nq_high_arr[nq_ptr] > nq_target_level:
                return SMT_RESPECTED, t

        if t_es == t:
            es_ptr += 1
        if t_nq == t:
            nq_ptr += 1

    return SMT_CONFIRMED, None


def detect_smt_divergence(
    es_swings: List[Dict],
    nq_swings: List[Dict],
    df_es: pd.DataFrame,
    df_nq: pd.DataFrame,
    time_window_minutes: int = 3,
    timeframe: str = "1min",
    df_es_1m: Optional[pd.DataFrame] = None,
    df_nq_1m: Optional[pd.DataFrame] = None,
) -> List[Dict]:
    """
    Detects Multi-Timeframe SMT Divergence between ES and NQ and tracks its full lifecycle:
    - 4h SMT  -> confirmed by 1h MSS
    - 1h SMT  -> confirmed by 15m MSS
    - 15m SMT -> confirmed by 5m MSS
    - 5m SMT  -> confirmed by 1m MSS
    - 1m SMT  -> confirmed by 1m MSS
    """
    smt_events = []

    if df_es.empty and df_nq.empty:
        return smt_events

    tf_key = timeframe.lower().strip()
    cfg = TIMEFRAME_HIERARCHY.get(tf_key, TIMEFRAME_HIERARCHY["1min"])
    htf_min = int(cfg["htf_min"])
    ltf_rule = str(cfg["ltf_rule"])
    ltf_min = int(cfg["ltf_min"])
    ltf_label = str(cfg["ltf_label"])
    sync_bars = int(cfg["sync_bars"])

    if htf_min > 1 and df_es_1m is not None and df_nq_1m is not None and not df_es_1m.empty and not df_nq_1m.empty:
        if ltf_rule == "1min":
            df_es_ltf = df_es_1m
            df_nq_ltf = df_nq_1m
        else:
            df_es_ltf = resample_timeframe(df_es_1m, ltf_rule)
            df_nq_ltf = resample_timeframe(df_nq_1m, ltf_rule)
        ltf_es_swings = detect_swings(df_es_ltf)
        ltf_nq_swings = detect_swings(df_nq_ltf)
    else:
        df_es_ltf = df_es
        df_nq_ltf = df_nq
        ltf_es_swings = es_swings
        ltf_nq_swings = nq_swings
        if htf_min > 1:
            ltf_min = htf_min

    es_highs = [s for s in es_swings if s['type'] == 'SwingHigh']
    es_lows = [s for s in es_swings if s['type'] == 'SwingLow']
    nq_highs = [s for s in nq_swings if s['type'] == 'SwingHigh']
    nq_lows = [s for s in nq_swings if s['type'] == 'SwingLow']

    es_high_times = [s['time'] for s in es_highs]
    es_low_times = [s['time'] for s in es_lows]
    nq_high_times = [s['time'] for s in nq_highs]
    nq_low_times = [s['time'] for s in nq_lows]

    ltf_es_highs = [s for s in ltf_es_swings if s['type'] == 'SwingHigh']
    ltf_es_lows = [s for s in ltf_es_swings if s['type'] == 'SwingLow']
    ltf_nq_highs = [s for s in ltf_nq_swings if s['type'] == 'SwingHigh']
    ltf_nq_lows = [s for s in ltf_nq_swings if s['type'] == 'SwingLow']

    ltf_es_high_times = [s['time'] for s in ltf_es_highs]
    ltf_es_low_times = [s['time'] for s in ltf_es_lows]
    ltf_nq_high_times = [s['time'] for s in ltf_nq_highs]
    ltf_nq_low_times = [s['time'] for s in ltf_nq_lows]

    has_es_ohlc = not df_es.empty and all(c in df_es.columns for c in ('close', 'high', 'low')) and len(df_es) > 1
    has_nq_ohlc = not df_nq.empty and all(c in df_nq.columns for c in ('close', 'high', 'low')) and len(df_nq) > 1

    es_times = df_es.index if not df_es.empty else pd.DatetimeIndex([])
    nq_times = df_nq.index if not df_nq.empty else pd.DatetimeIndex([])

    es_high_arr = df_es['high'].to_numpy() if has_es_ohlc else np.array([])
    es_low_arr = df_es['low'].to_numpy() if has_es_ohlc else np.array([])
    nq_high_arr = df_nq['high'].to_numpy() if has_nq_ohlc else np.array([])
    nq_low_arr = df_nq['low'].to_numpy() if has_nq_ohlc else np.array([])

    has_ltf_es_close = not df_es_ltf.empty and 'close' in df_es_ltf.columns
    has_ltf_nq_close = not df_nq_ltf.empty and 'close' in df_nq_ltf.columns
    has_ltf_es_ohlc = has_ltf_es_close and 'high' in df_es_ltf.columns and 'low' in df_es_ltf.columns and len(df_es_ltf) > 1
    has_ltf_nq_ohlc = has_ltf_nq_close and 'high' in df_nq_ltf.columns and 'low' in df_nq_ltf.columns and len(df_nq_ltf) > 1

    ltf_es_times = df_es_ltf.index if has_ltf_es_close else pd.DatetimeIndex([])
    ltf_nq_times = df_nq_ltf.index if has_ltf_nq_close else pd.DatetimeIndex([])

    ltf_es_close_arr = df_es_ltf['close'].to_numpy() if has_ltf_es_close else np.array([])
    ltf_nq_close_arr = df_nq_ltf['close'].to_numpy() if has_ltf_nq_close else np.array([])
    ltf_es_high_arr = df_es_ltf['high'].to_numpy() if has_ltf_es_ohlc else np.array([])
    ltf_es_low_arr = df_es_ltf['low'].to_numpy() if has_ltf_es_ohlc else np.array([])
    ltf_nq_high_arr = df_nq_ltf['high'].to_numpy() if has_ltf_nq_ohlc else np.array([])
    ltf_nq_low_arr = df_nq_ltf['low'].to_numpy() if has_ltf_nq_ohlc else np.array([])

    if htf_min == 1:
        sync_minutes = min(time_window_minutes, 3)
        min_sep_seconds = 180
    else:
        sync_minutes = htf_min * sync_bars
        min_sep_seconds = htf_min * 2 * 60

    htf_bar_delta = pd.Timedelta(minutes=htf_min)
    ltf_close_offset = pd.Timedelta(minutes=max(0, ltf_min - 1))
    seen_confirmations = set()

    # Build candidate sweep points:
    # Includes both confirmed swings AND (for HTF > 1m) live HTF candle extremes
    # so 1H/4H sweeps are detected immediately on LTF MSS without waiting hours for a 3-bar HTF fractal.
    bear_candidates: List[Tuple[pd.Timestamp, float, pd.Timestamp, float]] = []
    for curr_es in es_highs:
        curr_nq = get_closest_swing_fast(curr_es['time'], nq_highs, nq_high_times, max_diff_minutes=sync_minutes)
        if curr_nq:
            bear_candidates.append((curr_es['time'], float(curr_es['price']), curr_nq['time'], float(curr_nq['price'])))

    if htf_min > 1 and has_es_ohlc and has_nq_ohlc:
        for k in range(len(es_times)):
            t_bar = es_times[k]
            nq_k = nq_times.searchsorted(t_bar)
            if nq_k < len(nq_times) and nq_times[nq_k] == t_bar:
                bear_candidates.append((t_bar, float(es_high_arr[k]), t_bar, float(nq_high_arr[nq_k])))

    bull_candidates: List[Tuple[pd.Timestamp, float, pd.Timestamp, float]] = []
    for curr_es in es_lows:
        curr_nq = get_closest_swing_fast(curr_es['time'], nq_lows, nq_low_times, max_diff_minutes=sync_minutes)
        if curr_nq:
            bull_candidates.append((curr_es['time'], float(curr_es['price']), curr_nq['time'], float(curr_nq['price'])))

    if htf_min > 1 and has_es_ohlc and has_nq_ohlc:
        for k in range(len(es_times)):
            t_bar = es_times[k]
            nq_k = nq_times.searchsorted(t_bar)
            if nq_k < len(nq_times) and nq_times[nq_k] == t_bar:
                bull_candidates.append((t_bar, float(es_low_arr[k]), t_bar, float(nq_low_arr[nq_k])))

    # =========================================================================
    # BEARISH SMT (HTF BSL Sweep + LTF Bearish MSS)
    # =========================================================================
    for c_es_time, c_es_price, c_nq_time, c_nq_price in bear_candidates:
        creation_time = max(c_es_time, c_nq_time)
        sweep_start_time = min(c_es_time, c_nq_time)

        prior_idx = bisect.bisect_left(es_high_times, sweep_start_time)
        past_es_highs = es_highs[max(0, prior_idx - 15):prior_idx]

        for prev_es in reversed(past_es_highs):
            if (sweep_start_time - prev_es['time']).total_seconds() < min_sep_seconds:
                continue

            prev_nq = get_closest_swing_fast(prev_es['time'], nq_highs, nq_high_times, max_diff_minutes=sync_minutes)
            if not prev_nq or prev_nq['time'] >= c_nq_time:
                continue

            es_max_in_win = c_es_price
            nq_max_in_win = c_nq_price

            if has_es_ohlc:
                es_p_idx = es_times.searchsorted(prev_es['time'])
                es_win_start = es_times.searchsorted(sweep_start_time - htf_bar_delta)
                es_win_end = min(len(es_times), es_times.searchsorted(creation_time, side='right'))
                if es_win_start > es_p_idx + 1 and es_high_arr[es_p_idx + 1:es_win_start].max() >= prev_es['price']:
                    continue
                if es_win_end > es_win_start:
                    es_max_in_win = max(c_es_price, float(es_high_arr[es_win_start:es_win_end].max()))

            if has_nq_ohlc:
                nq_p_idx = nq_times.searchsorted(prev_nq['time'])
                nq_win_start = nq_times.searchsorted(sweep_start_time - htf_bar_delta)
                nq_win_end = min(len(nq_times), nq_times.searchsorted(creation_time, side='right'))
                if nq_win_start > nq_p_idx + 1 and nq_high_arr[nq_p_idx + 1:nq_win_start].max() >= prev_nq['price']:
                    continue
                if nq_win_end > nq_win_start:
                    nq_max_in_win = max(c_nq_price, float(nq_high_arr[nq_win_start:nq_win_end].max()))

            es_swept_bsl = (c_es_price > prev_es['price']) and (nq_max_in_win < prev_nq['price'])
            nq_swept_bsl = (c_nq_price > prev_nq['price']) and (es_max_in_win < prev_es['price'])

            if not (es_swept_bsl or nq_swept_bsl):
                continue

            es_intervening_low = _find_intervening_extreme(
                es_lows, es_low_times, prev_es['time'], c_es_time, find_min=True
            )
            nq_intervening_low = _find_intervening_extreme(
                nq_lows, nq_low_times, prev_nq['time'], c_nq_time, find_min=True
            )

            if htf_min == 1:
                ltf_anchor = creation_time
            elif es_swept_bsl:
                ltf_anchor = _find_ltf_sweep_start(
                    c_es_time, htf_min, ltf_es_times, ltf_es_high_arr, prev_es['price'], is_bearish=True
                )
            else:
                ltf_anchor = _find_ltf_sweep_start(
                    c_nq_time, htf_min, ltf_nq_times, ltf_nq_high_arr, prev_nq['price'], is_bearish=True
                )

            confirm_time = None
            confirm_asset = None
            confirm_level = None
            disrespected = False
            es_sweep_peak = max(prev_es['price'], c_es_price)
            nq_sweep_peak = max(prev_nq['price'], c_nq_price)

            for step in range(1, 21):
                t = ltf_anchor + pd.Timedelta(minutes=step * ltf_min)

                es_idx = ltf_es_times.searchsorted(t) if has_ltf_es_close else -1
                has_es_bar = has_ltf_es_close and es_idx < len(ltf_es_times) and ltf_es_times[es_idx] == t

                nq_idx = ltf_nq_times.searchsorted(t) if has_ltf_nq_close else -1
                has_nq_bar = has_ltf_nq_close and nq_idx < len(ltf_nq_times) and ltf_nq_times[nq_idx] == t

                if has_ltf_es_ohlc and has_es_bar:
                    es_sweep_peak = max(es_sweep_peak, float(ltf_es_high_arr[es_idx]))
                if has_ltf_nq_ohlc and has_nq_bar:
                    nq_sweep_peak = max(nq_sweep_peak, float(ltf_nq_high_arr[nq_idx]))

                if es_swept_bsl and has_ltf_nq_ohlc and has_nq_bar:
                    if ltf_nq_high_arr[nq_idx] >= prev_nq['price']:
                        disrespected = True
                        break
                if nq_swept_bsl and has_ltf_es_ohlc and has_es_bar:
                    if ltf_es_high_arr[es_idx] >= prev_es['price']:
                        disrespected = True
                        break

                es_l_idx = bisect.bisect_left(ltf_es_low_times, t) - 1
                es_mss_level = (
                    ltf_es_lows[es_l_idx]['price']
                    if (es_l_idx >= 0 and ltf_es_low_times[es_l_idx] > prev_es['time'])
                    else None
                )

                nq_l_idx = bisect.bisect_left(ltf_nq_low_times, t) - 1
                nq_mss_level = (
                    ltf_nq_lows[nq_l_idx]['price']
                    if (nq_l_idx >= 0 and ltf_nq_low_times[nq_l_idx] > prev_nq['time'])
                    else None
                )

                if htf_min > 1:
                    if has_ltf_es_ohlc and has_es_bar and es_intervening_low is not None:
                        if es_mss_level is None or es_intervening_low < es_mss_level:
                            if ltf_es_low_arr[es_idx] < es_intervening_low and not (
                                es_mss_level is not None and ltf_es_close_arr[es_idx] < es_mss_level
                            ):
                                disrespected = True
                                break
                    if has_ltf_nq_ohlc and has_nq_bar and nq_intervening_low is not None:
                        if nq_mss_level is None or nq_intervening_low < nq_mss_level:
                            if ltf_nq_low_arr[nq_idx] < nq_intervening_low and not (
                                nq_mss_level is not None and ltf_nq_close_arr[nq_idx] < nq_mss_level
                            ):
                                disrespected = True
                                break

                if has_es_bar and es_mss_level is not None and ltf_es_close_arr[es_idx] < es_mss_level:
                    confirm_time = t
                    confirm_asset = "ES"
                    confirm_level = es_mss_level
                    break

                if has_nq_bar and nq_mss_level is not None and ltf_nq_close_arr[nq_idx] < nq_mss_level:
                    confirm_time = t
                    confirm_asset = "NQ"
                    confirm_level = nq_mss_level
                    break

            if not disrespected and confirm_time is not None:
                dedup_key = ('Bearish_SMT', prev_es['time'], prev_nq['time'])
                if dedup_key in seen_confirmations:
                    break
                seen_confirmations.add(dedup_key)

                es_inv_high = es_sweep_peak if es_swept_bsl else float(prev_es['price'])
                nq_inv_high = float(prev_nq['price']) if es_swept_bsl else nq_sweep_peak

                es_conf_idx = ltf_es_times.searchsorted(confirm_time) if has_ltf_es_close else -1
                es_ref_low = (
                    float(ltf_es_low_arr[es_conf_idx])
                    if (has_ltf_es_ohlc and es_conf_idx < len(ltf_es_times) and ltf_es_times[es_conf_idx] == confirm_time)
                    else float(confirm_level)
                )
                nq_conf_idx = ltf_nq_times.searchsorted(confirm_time) if has_ltf_nq_close else -1
                nq_ref_low = (
                    float(ltf_nq_low_arr[nq_conf_idx])
                    if (has_ltf_nq_ohlc and nq_conf_idx < len(ltf_nq_times) and ltf_nq_times[nq_conf_idx] == confirm_time)
                    else float(confirm_level)
                )

                es_target = (
                    es_intervening_low
                    if (es_intervening_low is not None and es_intervening_low < es_ref_low)
                    else _find_prior_target_beyond_confirm(es_lows, es_low_times, c_es_time, es_ref_low, is_bearish=True)
                )
                nq_target = (
                    nq_intervening_low
                    if (nq_intervening_low is not None and nq_intervening_low < nq_ref_low)
                    else _find_prior_target_beyond_confirm(nq_lows, nq_low_times, c_nq_time, nq_ref_low, is_bearish=True)
                )

                status, resolved_time = _evaluate_post_confirmation_lifecycle(
                    is_bearish=True,
                    confirm_time=confirm_time,
                    es_inv_level=es_inv_high,
                    nq_inv_level=nq_inv_high,
                    es_target_level=es_target,
                    nq_target_level=nq_target,
                    ltf_es_times=ltf_es_times,
                    ltf_es_high_arr=ltf_es_high_arr,
                    ltf_es_low_arr=ltf_es_low_arr,
                    has_ltf_es_ohlc=has_ltf_es_ohlc,
                    ltf_nq_times=ltf_nq_times,
                    ltf_nq_high_arr=ltf_nq_high_arr,
                    ltf_nq_low_arr=ltf_nq_low_arr,
                    has_ltf_nq_ohlc=has_ltf_nq_ohlc,
                )

                status_tag = (
                    f"[Status: {status} at {resolved_time.strftime('%H:%M')}]"
                    if resolved_time is not None
                    else f"[Status: {status}]"
                )

                if es_swept_bsl:
                    desc = (
                        f"[{creation_time.strftime('%H:%M')}] Bearish SMT ({timeframe}): "
                        f"ES swept BSL {prev_es['price']:.2f} (from {prev_es['time'].strftime('%H:%M')}) "
                        f"to {c_es_price:.2f}, while NQ held below {prev_nq['price']:.2f} (from {prev_nq['time'].strftime('%H:%M')}) "
                        f"--> {ltf_label} MSS on {confirm_asset} at [{confirm_time.strftime('%H:%M')}] (< {confirm_level:.2f}) {status_tag}"
                    )
                else:
                    desc = (
                        f"[{creation_time.strftime('%H:%M')}] Bearish SMT ({timeframe}): "
                        f"NQ swept BSL {prev_nq['price']:.2f} (from {prev_nq['time'].strftime('%H:%M')}) "
                        f"to {c_nq_price:.2f}, while ES held below {prev_es['price']:.2f} (from {prev_es['time'].strftime('%H:%M')}) "
                        f"--> {ltf_label} MSS on {confirm_asset} at [{confirm_time.strftime('%H:%M')}] (< {confirm_level:.2f}) {status_tag}"
                    )

                smt_events.append({
                    'type': 'Bearish_SMT',
                    'time': creation_time,
                    'confirmation_time': confirm_time,
                    'confirmation_close_time': confirm_time + ltf_close_offset,
                    'status': status,
                    'resolved_time': resolved_time,
                    'resolved_close_time': (resolved_time + ltf_close_offset) if resolved_time is not None else None,
                    'es_target_level': es_target,
                    'nq_target_level': nq_target,
                    'es_invalidation_level': es_inv_high,
                    'nq_invalidation_level': nq_inv_high,
                    'timeframe': timeframe,
                    'confirmation_timeframe': ltf_label,
                    'description': desc
                })
                break

    # =========================================================================
    # BULLISH SMT (HTF SSL Sweep + LTF Bullish MSS)
    # =========================================================================
    for c_es_time, c_es_price, c_nq_time, c_nq_price in bull_candidates:
        creation_time = max(c_es_time, c_nq_time)
        sweep_start_time = min(c_es_time, c_nq_time)

        prior_idx = bisect.bisect_left(es_low_times, sweep_start_time)
        past_es_lows = es_lows[max(0, prior_idx - 15):prior_idx]

        for prev_es in reversed(past_es_lows):
            if (sweep_start_time - prev_es['time']).total_seconds() < min_sep_seconds:
                continue

            prev_nq = get_closest_swing_fast(prev_es['time'], nq_lows, nq_low_times, max_diff_minutes=sync_minutes)
            if not prev_nq or prev_nq['time'] >= c_nq_time:
                continue

            es_min_in_win = c_es_price
            nq_min_in_win = c_nq_price

            if has_es_ohlc:
                es_p_idx = es_times.searchsorted(prev_es['time'])
                es_win_start = es_times.searchsorted(sweep_start_time - htf_bar_delta)
                es_win_end = min(len(es_times), es_times.searchsorted(creation_time, side='right'))
                if es_win_start > es_p_idx + 1 and es_low_arr[es_p_idx + 1:es_win_start].min() <= prev_es['price']:
                    continue
                if es_win_end > es_win_start:
                    es_min_in_win = min(c_es_price, float(es_low_arr[es_win_start:es_win_end].min()))

            if has_nq_ohlc:
                nq_p_idx = nq_times.searchsorted(prev_nq['time'])
                nq_win_start = nq_times.searchsorted(sweep_start_time - htf_bar_delta)
                nq_win_end = min(len(nq_times), nq_times.searchsorted(creation_time, side='right'))
                if nq_win_start > nq_p_idx + 1 and nq_low_arr[nq_p_idx + 1:nq_win_start].min() <= prev_nq['price']:
                    continue
                if nq_win_end > nq_win_start:
                    nq_min_in_win = min(c_nq_price, float(nq_low_arr[nq_win_start:nq_win_end].min()))

            es_swept_ssl = (c_es_price < prev_es['price']) and (nq_min_in_win > prev_nq['price'])
            nq_swept_ssl = (c_nq_price < prev_nq['price']) and (es_min_in_win > prev_es['price'])

            if not (es_swept_ssl or nq_swept_ssl):
                continue

            es_intervening_high = _find_intervening_extreme(
                es_highs, es_high_times, prev_es['time'], c_es_time, find_min=False
            )
            nq_intervening_high = _find_intervening_extreme(
                nq_highs, nq_high_times, prev_nq['time'], c_nq_time, find_min=False
            )

            if htf_min == 1:
                ltf_anchor = creation_time
            elif es_swept_ssl:
                ltf_anchor = _find_ltf_sweep_start(
                    c_es_time, htf_min, ltf_es_times, ltf_es_low_arr, prev_es['price'], is_bearish=False
                )
            else:
                ltf_anchor = _find_ltf_sweep_start(
                    c_nq_time, htf_min, ltf_nq_times, ltf_nq_low_arr, prev_nq['price'], is_bearish=False
                )

            confirm_time = None
            confirm_asset = None
            confirm_level = None
            disrespected = False
            es_sweep_trough = min(prev_es['price'], c_es_price)
            nq_sweep_trough = min(prev_nq['price'], c_nq_price)

            for step in range(1, 21):
                t = ltf_anchor + pd.Timedelta(minutes=step * ltf_min)

                es_idx = ltf_es_times.searchsorted(t) if has_ltf_es_close else -1
                has_es_bar = has_ltf_es_close and es_idx < len(ltf_es_times) and ltf_es_times[es_idx] == t

                nq_idx = ltf_nq_times.searchsorted(t) if has_ltf_nq_close else -1
                has_nq_bar = has_ltf_nq_close and nq_idx < len(ltf_nq_times) and ltf_nq_times[nq_idx] == t

                if has_ltf_es_ohlc and has_es_bar:
                    es_sweep_trough = min(es_sweep_trough, float(ltf_es_low_arr[es_idx]))
                if has_ltf_nq_ohlc and has_nq_bar:
                    nq_sweep_trough = min(nq_sweep_trough, float(ltf_nq_low_arr[nq_idx]))

                if es_swept_ssl and has_ltf_nq_ohlc and has_nq_bar:
                    if ltf_nq_low_arr[nq_idx] <= prev_nq['price']:
                        disrespected = True
                        break
                if nq_swept_ssl and has_ltf_es_ohlc and has_es_bar:
                    if ltf_es_low_arr[es_idx] <= prev_es['price']:
                        disrespected = True
                        break

                es_h_idx = bisect.bisect_left(ltf_es_high_times, t) - 1
                es_mss_level = (
                    ltf_es_highs[es_h_idx]['price']
                    if (es_h_idx >= 0 and ltf_es_high_times[es_h_idx] > prev_es['time'])
                    else None
                )

                nq_h_idx = bisect.bisect_left(ltf_nq_high_times, t) - 1
                nq_mss_level = (
                    ltf_nq_highs[nq_h_idx]['price']
                    if (nq_h_idx >= 0 and ltf_nq_high_times[nq_h_idx] > prev_nq['time'])
                    else None
                )

                if htf_min > 1:
                    if has_ltf_es_ohlc and has_es_bar and es_intervening_high is not None:
                        if es_mss_level is None or es_intervening_high > es_mss_level:
                            if ltf_es_high_arr[es_idx] > es_intervening_high and not (
                                es_mss_level is not None and ltf_es_close_arr[es_idx] > es_mss_level
                            ):
                                disrespected = True
                                break
                    if has_ltf_nq_ohlc and has_nq_bar and nq_intervening_high is not None:
                        if nq_mss_level is None or nq_intervening_high > nq_mss_level:
                            if ltf_nq_high_arr[nq_idx] > nq_intervening_high and not (
                                nq_mss_level is not None and ltf_nq_close_arr[nq_idx] > nq_mss_level
                            ):
                                disrespected = True
                                break

                if has_es_bar and es_mss_level is not None and ltf_es_close_arr[es_idx] > es_mss_level:
                    confirm_time = t
                    confirm_asset = "ES"
                    confirm_level = es_mss_level
                    break

                if has_nq_bar and nq_mss_level is not None and ltf_nq_close_arr[nq_idx] > nq_mss_level:
                    confirm_time = t
                    confirm_asset = "NQ"
                    confirm_level = nq_mss_level
                    break

            if not disrespected and confirm_time is not None:
                dedup_key = ('Bullish_SMT', prev_es['time'], prev_nq['time'])
                if dedup_key in seen_confirmations:
                    break
                seen_confirmations.add(dedup_key)

                es_inv_low = es_sweep_trough if es_swept_ssl else float(prev_es['price'])
                nq_inv_low = float(prev_nq['price']) if es_swept_ssl else nq_sweep_trough

                es_conf_idx = ltf_es_times.searchsorted(confirm_time) if has_ltf_es_close else -1
                es_ref_high = (
                    float(ltf_es_high_arr[es_conf_idx])
                    if (has_ltf_es_ohlc and es_conf_idx < len(ltf_es_times) and ltf_es_times[es_conf_idx] == confirm_time)
                    else float(confirm_level)
                )
                nq_conf_idx = ltf_nq_times.searchsorted(confirm_time) if has_ltf_nq_close else -1
                nq_ref_high = (
                    float(ltf_nq_high_arr[nq_conf_idx])
                    if (has_ltf_nq_ohlc and nq_conf_idx < len(ltf_nq_times) and ltf_nq_times[nq_conf_idx] == confirm_time)
                    else float(confirm_level)
                )

                es_target = (
                    es_intervening_high
                    if (es_intervening_high is not None and es_intervening_high > es_ref_high)
                    else _find_prior_target_beyond_confirm(es_highs, es_high_times, c_es_time, es_ref_high, is_bearish=False)
                )
                nq_target = (
                    nq_intervening_high
                    if (nq_intervening_high is not None and nq_intervening_high > nq_ref_high)
                    else _find_prior_target_beyond_confirm(nq_highs, nq_high_times, c_nq_time, nq_ref_high, is_bearish=False)
                )

                status, resolved_time = _evaluate_post_confirmation_lifecycle(
                    is_bearish=False,
                    confirm_time=confirm_time,
                    es_inv_level=es_inv_low,
                    nq_inv_level=nq_inv_low,
                    es_target_level=es_target,
                    nq_target_level=nq_target,
                    ltf_es_times=ltf_es_times,
                    ltf_es_high_arr=ltf_es_high_arr,
                    ltf_es_low_arr=ltf_es_low_arr,
                    has_ltf_es_ohlc=has_ltf_es_ohlc,
                    ltf_nq_times=ltf_nq_times,
                    ltf_nq_high_arr=ltf_nq_high_arr,
                    ltf_nq_low_arr=ltf_nq_low_arr,
                    has_ltf_nq_ohlc=has_ltf_nq_ohlc,
                )

                status_tag = (
                    f"[Status: {status} at {resolved_time.strftime('%H:%M')}]"
                    if resolved_time is not None
                    else f"[Status: {status}]"
                )

                if es_swept_ssl:
                    desc = (
                        f"[{creation_time.strftime('%H:%M')}] Bullish SMT ({timeframe}): "
                        f"ES swept SSL {prev_es['price']:.2f} (from {prev_es['time'].strftime('%H:%M')}) "
                        f"to {c_es_price:.2f}, while NQ held above {prev_nq['price']:.2f} (from {prev_nq['time'].strftime('%H:%M')}) "
                        f"--> {ltf_label} MSS on {confirm_asset} at [{confirm_time.strftime('%H:%M')}] (> {confirm_level:.2f}) {status_tag}"
                    )
                else:
                    desc = (
                        f"[{creation_time.strftime('%H:%M')}] Bullish SMT ({timeframe}): "
                        f"NQ swept SSL {prev_nq['price']:.2f} (from {prev_nq['time'].strftime('%H:%M')}) "
                        f"to {c_nq_price:.2f}, while ES held above {prev_es['price']:.2f} (from {prev_es['time'].strftime('%H:%M')}) "
                        f"--> {ltf_label} MSS on {confirm_asset} at [{confirm_time.strftime('%H:%M')}] (> {confirm_level:.2f}) {status_tag}"
                    )

                smt_events.append({
                    'type': 'Bullish_SMT',
                    'time': creation_time,
                    'confirmation_time': confirm_time,
                    'confirmation_close_time': confirm_time + ltf_close_offset,
                    'status': status,
                    'resolved_time': resolved_time,
                    'resolved_close_time': (resolved_time + ltf_close_offset) if resolved_time is not None else None,
                    'es_target_level': es_target,
                    'nq_target_level': nq_target,
                    'es_invalidation_level': es_inv_low,
                    'nq_invalidation_level': nq_inv_low,
                    'timeframe': timeframe,
                    'confirmation_timeframe': ltf_label,
                    'description': desc
                })
                break

    smt_events.sort(key=lambda x: x['time'])
    return smt_events