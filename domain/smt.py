from datetime import time
from typing import List, Dict, Optional, Any, Tuple
import bisect
import numpy as np
import pandas as pd

from core.data_loader import resample_timeframe
from domain.swings import detect_swings


# Official SMT Lifecycle Status Constants
SMT_PENDING = "PENDING"
SMT_CONFIRMED = "CONFIRMED"      # LTF MSS confirmed; setup is live and tradable
SMT_RESPECTED = "RESPECTED"      # Opposing target low/high swept after confirmation (Done)
SMT_INVALIDATED = "INVALIDATED"  # Holding ("outsider") index breached its held swing high/low


TIMEFRAME_HIERARCHY: Dict[str, Dict[str, object]] = {
    "1min":  {"htf_min": 1,   "ltf_rule": "1min",  "ltf_min": 1,  "ltf_label": "1m",  "sync_bars": 3, "max_age_min": 90},
    "1m":    {"htf_min": 1,   "ltf_rule": "1min",  "ltf_min": 1,  "ltf_label": "1m",  "sync_bars": 3, "max_age_min": 90},
    "5min":  {"htf_min": 5,   "ltf_rule": "1min",  "ltf_min": 1,  "ltf_label": "1m",  "sync_bars": 2, "max_age_min": 120},
    "5m":    {"htf_min": 5,   "ltf_rule": "1min",  "ltf_min": 1,  "ltf_label": "1m",  "sync_bars": 2, "max_age_min": 120},
    "15min": {"htf_min": 15,  "ltf_rule": "5min",  "ltf_min": 5,  "ltf_label": "5m",  "sync_bars": 2, "max_age_min": 240},
    "15m":   {"htf_min": 15,  "ltf_rule": "5min",  "ltf_min": 5,  "ltf_label": "5m",  "sync_bars": 2, "max_age_min": 240},
    "30min": {"htf_min": 30,  "ltf_rule": "15min", "ltf_min": 15, "ltf_label": "15m", "sync_bars": 1, "max_age_min": 360},
    "30m":   {"htf_min": 30,  "ltf_rule": "15min", "ltf_min": 15, "ltf_label": "15m", "sync_bars": 1, "max_age_min": 360},
    "1h":    {"htf_min": 60,  "ltf_rule": "15min", "ltf_min": 15, "ltf_label": "15m", "sync_bars": 1, "max_age_min": 720},
    "1hr":   {"htf_min": 60,  "ltf_rule": "15min", "ltf_min": 15, "ltf_label": "15m", "sync_bars": 1, "max_age_min": 720},
    "4h":    {"htf_min": 240, "ltf_rule": "1h",    "ltf_min": 60, "ltf_label": "1h",  "sync_bars": 1, "max_age_min": 2160},
    "4hr":   {"htf_min": 240, "ltf_rule": "1h",    "ltf_min": 60, "ltf_label": "1h",  "sync_bars": 1, "max_age_min": 2160},
}


def extract_macro_dict_fields(mp: Dict[str, Any]) -> Tuple[str, str, float, pd.Timestamp]:
    raw_type = str(
        mp.get("pool_type")
        or mp.get("type")
        or mp.get("kind")
        or mp.get("direction")
        or mp.get("label")
        or mp.get("name")
        or ""
    ).upper()
    p_type = "BSL" if ("BSL" in raw_type or "HIGH" in raw_type or "BUY" in raw_type) else "SSL"

    raw_sess = str(
        mp.get("session")
        or mp.get("session_origin")
        or mp.get("origin")
        or mp.get("label")
        or mp.get("name")
        or raw_type
    ).upper()
    if "ASIA" in raw_sess:
        sess = "Asia"
    elif "LONDON" in raw_sess:
        sess = "London"
    elif "NEW YORK" in raw_sess or "NY" in raw_sess:
        sess = "New York"
    else:
        sess = "Session"

    price = float(
        mp.get("price", mp.get("level", mp.get("value", mp.get("high", mp.get("low", 0.0)))))
    )
    return p_type, sess, price, mp["time"]


def resample_tv_timeframe(df_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if df_1m.empty:
        return df_1m
    tf_clean = timeframe.lower().strip()
    if tf_clean in ("4h", "4hr"):
        agg_dict = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if "volume" in df_1m.columns:
            agg_dict["volume"] = "sum"
        return df_1m.resample("4h", offset="2h", label="left", closed="left").agg(agg_dict).dropna()
    return resample_timeframe(df_1m, timeframe)


def detect_htf_swings(df_htf: pd.DataFrame) -> List[Dict[str, Any]]:
    if len(df_htf) < 3:
        return detect_swings(df_htf)
    highs = df_htf["high"].to_numpy(dtype=float)
    lows = df_htf["low"].to_numpy(dtype=float)
    times = df_htf.index
    swings: List[Dict[str, Any]] = []
    for i in range(1, len(df_htf) - 1):
        if highs[i] > highs[i - 1] and highs[i] >= highs[i + 1]:
            swings.append({"type": "SwingHigh", "time": times[i], "price": float(highs[i])})
        if lows[i] < lows[i - 1] and lows[i] <= lows[i + 1]:
            swings.append({"type": "SwingLow", "time": times[i], "price": float(lows[i])})
    return swings


def get_smt_status_at(smt: Dict[str, Any], timestamp: pd.Timestamp) -> str:
    conf_time = smt.get("confirmation_close_time", smt.get("confirmation_time"))
    if conf_time is None or timestamp < conf_time:
        return SMT_PENDING
    res_time = smt.get("resolved_close_time", smt.get("resolved_time"))
    if res_time is not None and timestamp >= res_time:
        return str(smt.get("status", SMT_CONFIRMED))
    return SMT_CONFIRMED


def is_smt_active_at(smt: Dict[str, Any], timestamp: pd.Timestamp) -> bool:
    if get_smt_status_at(smt, timestamp) != SMT_CONFIRMED:
        return False
    conf_time = smt.get("confirmation_close_time", smt.get("confirmation_time"))
    tf_key = str(smt.get("timeframe", "1min")).lower().strip()
    max_age_min = int(TIMEFRAME_HIERARCHY.get(tf_key, TIMEFRAME_HIERARCHY["1min"]).get("max_age_min", 180))
    if conf_time is not None and (timestamp - conf_time) > pd.Timedelta(minutes=max_age_min):
        return False
    return True


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


def _get_htf_companion_swing(
    target_time: pd.Timestamp,
    target_swings: List[Dict],
    target_times: List[pd.Timestamp],
    other_times: pd.DatetimeIndex,
    other_extreme_arr: np.ndarray,
    max_diff_minutes: int,
    htf_min: int,
    is_high: bool,
) -> Optional[Dict[str, Any]]:
    if htf_min >= 60 and len(other_times) > 1:
        k = other_times.searchsorted(target_time)
        if k < len(other_times) - 1 and other_times[k] == target_time:
            if is_high and other_extreme_arr[k] > other_extreme_arr[k + 1]:
                return {"type": "SwingHigh", "time": target_time, "price": float(other_extreme_arr[k])}
            if not is_high and other_extreme_arr[k] < other_extreme_arr[k + 1]:
                return {"type": "SwingLow", "time": target_time, "price": float(other_extreme_arr[k])}
    return get_closest_swing_fast(target_time, target_swings, target_times, max_diff_minutes=max_diff_minutes)


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
    es_was_holder: bool,
    nq_was_holder: bool,
    es_held_level: float,
    nq_held_level: float,
    es_target_level: Optional[float],
    nq_target_level: Optional[float],
    check_es_target: bool,
    check_nq_target: bool,
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
        t = t_es if (t_es is not None and (t_nq is None or t_es <= t_nq)) else t_nq

        if t_es == t and es_was_holder:
            if is_bearish and ltf_es_high_arr[es_ptr] >= es_held_level:
                return SMT_INVALIDATED, t
            if not is_bearish and ltf_es_low_arr[es_ptr] <= es_held_level:
                return SMT_INVALIDATED, t
        if t_nq == t and nq_was_holder:
            if is_bearish and ltf_nq_high_arr[nq_ptr] >= nq_held_level:
                return SMT_INVALIDATED, t
            if not is_bearish and ltf_nq_low_arr[nq_ptr] <= nq_held_level:
                return SMT_INVALIDATED, t

        if t_es == t and check_es_target and es_target_level is not None:
            if is_bearish and ltf_es_low_arr[es_ptr] < es_target_level:
                return SMT_RESPECTED, t
            if not is_bearish and ltf_es_high_arr[es_ptr] > es_target_level:
                return SMT_RESPECTED, t
        if t_nq == t and check_nq_target and nq_target_level is not None:
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
            ltf_es_swings = detect_swings(df_es_ltf)
            ltf_nq_swings = detect_swings(df_nq_ltf)
        else:
            df_es_ltf = resample_tv_timeframe(df_es_1m, ltf_rule)
            df_nq_ltf = resample_tv_timeframe(df_nq_1m, ltf_rule)
            ltf_es_swings = detect_htf_swings(df_es_ltf)
            ltf_nq_swings = detect_htf_swings(df_nq_ltf)
    else:
        df_es_ltf = df_es
        df_nq_ltf = df_nq
        ltf_es_swings = es_swings
        ltf_nq_swings = nq_swings
        if htf_min > 1:
            ltf_min = htf_min

    df_es_res = df_es_1m if (df_es_1m is not None and not df_es_1m.empty) else df_es_ltf
    df_nq_res = df_nq_1m if (df_nq_1m is not None and not df_nq_1m.empty) else df_nq_ltf
    res_es_ohlc = not df_es_res.empty and all(c in df_es_res.columns for c in ('high', 'low')) and len(df_es_res) > 1
    res_nq_ohlc = not df_nq_res.empty and all(c in df_nq_res.columns for c in ('high', 'low')) and len(df_nq_res) > 1
    res_es_times = df_es_res.index if res_es_ohlc else pd.DatetimeIndex([])
    res_nq_times = df_nq_res.index if res_nq_ohlc else pd.DatetimeIndex([])
    res_es_high = df_es_res['high'].to_numpy() if res_es_ohlc else np.array([])
    res_es_low = df_es_res['low'].to_numpy() if res_es_ohlc else np.array([])
    res_nq_high = df_nq_res['high'].to_numpy() if res_nq_ohlc else np.array([])
    res_nq_low = df_nq_res['low'].to_numpy() if res_nq_ohlc else np.array([])

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
        min_sep_seconds = htf_min * 60

    htf_bar_delta = pd.Timedelta(minutes=htf_min)
    ltf_close_offset = pd.Timedelta(minutes=max(0, ltf_min - 1))
    seen_confirmations = set()

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

    # BEARISH SMT
    for c_es_time, c_es_price, c_nq_time, c_nq_price in bear_candidates:
        creation_time = max(c_es_time, c_nq_time)
        sweep_start_time = min(c_es_time, c_nq_time)

        prior_idx = bisect.bisect_left(es_high_times, sweep_start_time)
        past_es_highs = es_highs[max(0, prior_idx - 15):prior_idx]

        for prev_es in reversed(past_es_highs):
            if (sweep_start_time - prev_es['time']).total_seconds() < min_sep_seconds:
                continue

            prev_nq = _get_htf_companion_swing(
                prev_es['time'], nq_highs, nq_high_times, nq_times, nq_high_arr,
                max_diff_minutes=sync_minutes, htf_min=htf_min, is_high=True
            )
            if not prev_nq or prev_nq['time'] >= c_nq_time:
                continue

            es_max_in_win = c_es_price
            nq_max_in_win = c_nq_price
            pre_win_bars = 2 if htf_min >= 60 else 1

            if has_es_ohlc:
                es_p_idx = es_times.searchsorted(prev_es['time'])
                es_win_start = es_times.searchsorted(sweep_start_time - htf_bar_delta * pre_win_bars)
                es_win_end = min(len(es_times), es_times.searchsorted(creation_time, side='right'))
                if es_win_start > es_p_idx + 1 and es_high_arr[es_p_idx + 1:es_win_start].max() >= prev_es['price']:
                    continue
                if es_win_end > es_win_start:
                    es_max_in_win = max(c_es_price, float(es_high_arr[es_win_start:es_win_end].max()))

            if has_nq_ohlc:
                nq_p_idx = nq_times.searchsorted(prev_nq['time'])
                nq_win_start = nq_times.searchsorted(sweep_start_time - htf_bar_delta * pre_win_bars)
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

            for step in range(1, 21):
                t = ltf_anchor + pd.Timedelta(minutes=step * ltf_min)
                es_idx = ltf_es_times.searchsorted(t) if has_ltf_es_close else -1
                has_es_bar = has_ltf_es_close and es_idx < len(ltf_es_times) and ltf_es_times[es_idx] == t
                nq_idx = ltf_nq_times.searchsorted(t) if has_ltf_nq_close else -1
                has_nq_bar = has_ltf_nq_close and nq_idx < len(ltf_nq_times) and ltf_nq_times[nq_idx] == t

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
                    confirm_time, confirm_asset, confirm_level = t, "ES", es_mss_level
                    break
                if has_nq_bar and nq_mss_level is not None and ltf_nq_close_arr[nq_idx] < nq_mss_level:
                    confirm_time, confirm_asset, confirm_level = t, "NQ", nq_mss_level
                    break

            if not disrespected and confirm_time is not None:
                dedup_key = ('Bearish_SMT', prev_es['time'], prev_nq['time'])
                if dedup_key in seen_confirmations:
                    break
                seen_confirmations.add(dedup_key)

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

                conf_close_time = confirm_time + ltf_close_offset
                es_is_holder = bool(nq_swept_bsl) or not res_nq_ohlc
                nq_is_holder = bool(es_swept_bsl) or not res_es_ohlc
                es_held_lvl = float(prev_es['price']) if nq_swept_bsl else float(c_es_price)
                nq_held_lvl = float(prev_nq['price']) if es_swept_bsl else float(c_nq_price)

                check_es_t = True if htf_min < 60 else bool(es_swept_bsl or not res_nq_ohlc)
                check_nq_t = True if htf_min < 60 else bool(nq_swept_bsl or not res_es_ohlc)

                status, resolved_time = _evaluate_post_confirmation_lifecycle(
                    is_bearish=True,
                    confirm_time=conf_close_time,
                    es_was_holder=es_is_holder,
                    nq_was_holder=nq_is_holder,
                    es_held_level=es_held_lvl,
                    nq_held_level=nq_held_lvl,
                    es_target_level=es_target,
                    nq_target_level=nq_target,
                    check_es_target=check_es_t,
                    check_nq_target=check_nq_t,
                    ltf_es_times=res_es_times,
                    ltf_es_high_arr=res_es_high,
                    ltf_es_low_arr=res_es_low,
                    has_ltf_es_ohlc=res_es_ohlc,
                    ltf_nq_times=res_nq_times,
                    ltf_nq_high_arr=res_nq_high,
                    ltf_nq_low_arr=res_nq_low,
                    has_ltf_nq_ohlc=res_nq_ohlc,
                )

                status_tag = (
                    f"[Status: {status} at {resolved_time.strftime('%m-%d %H:%M')}]"
                    if resolved_time is not None
                    else f"[Status: {status}]"
                )

                if es_swept_bsl:
                    desc = (
                        f"[{creation_time.strftime('%m-%d %H:%M')}] Bearish SMT ({timeframe}): "
                        f"ES swept BSL {prev_es['price']:.2f} (from {prev_es['time'].strftime('%m-%d %H:%M')}) "
                        f"to {c_es_price:.2f}, while NQ held below {prev_nq['price']:.2f} (from {prev_nq['time'].strftime('%m-%d %H:%M')}) "
                        f"--> {ltf_label} MSS on {confirm_asset} at [{conf_close_time.strftime('%m-%d %H:%M')}] (< {confirm_level:.2f}) {status_tag}"
                    )
                else:
                    desc = (
                        f"[{creation_time.strftime('%m-%d %H:%M')}] Bearish SMT ({timeframe}): "
                        f"NQ swept BSL {prev_nq['price']:.2f} (from {prev_nq['time'].strftime('%m-%d %H:%M')}) "
                        f"to {c_nq_price:.2f}, while ES held below {prev_es['price']:.2f} (from {prev_es['time'].strftime('%m-%d %H:%M')}) "
                        f"--> {ltf_label} MSS on {confirm_asset} at [{conf_close_time.strftime('%m-%d %H:%M')}] (< {confirm_level:.2f}) {status_tag}"
                    )

                is_intraday_post_open = bool(
                    htf_min <= 15
                    and creation_time.date() == conf_close_time.date()
                    and creation_time.time() >= time(9, 30)
                )

                smt_events.append({
                    'type': 'Bearish_SMT',
                    'time': creation_time,
                    'confirmation_time': confirm_time,
                    'confirmation_close_time': conf_close_time,
                    'status': status,
                    'resolved_time': resolved_time,
                    'resolved_close_time': resolved_time,
                    'es_target_level': es_target,
                    'nq_target_level': nq_target,
                    'es_invalidation_level': es_held_lvl,
                    'nq_invalidation_level': nq_held_lvl,
                    'timeframe': timeframe,
                    'confirmation_timeframe': ltf_label,
                    'is_intraday_post_open': is_intraday_post_open,
                    'description': desc
                })
                break

    # BULLISH SMT
    for c_es_time, c_es_price, c_nq_time, c_nq_price in bull_candidates:
        creation_time = max(c_es_time, c_nq_time)
        sweep_start_time = min(c_es_time, c_nq_time)

        prior_idx = bisect.bisect_left(es_low_times, sweep_start_time)
        past_es_lows = es_lows[max(0, prior_idx - 15):prior_idx]

        for prev_es in reversed(past_es_lows):
            if (sweep_start_time - prev_es['time']).total_seconds() < min_sep_seconds:
                continue

            prev_nq = _get_htf_companion_swing(
                prev_es['time'], nq_lows, nq_low_times, nq_times, nq_low_arr,
                max_diff_minutes=sync_minutes, htf_min=htf_min, is_high=False
            )
            if not prev_nq or prev_nq['time'] >= c_nq_time:
                continue

            es_min_in_win = c_es_price
            nq_min_in_win = c_nq_price
            pre_win_bars = 2 if htf_min >= 60 else 1

            if has_es_ohlc:
                es_p_idx = es_times.searchsorted(prev_es['time'])
                es_win_start = es_times.searchsorted(sweep_start_time - htf_bar_delta * pre_win_bars)
                es_win_end = min(len(es_times), es_times.searchsorted(creation_time, side='right'))
                if es_win_start > es_p_idx + 1 and es_low_arr[es_p_idx + 1:es_win_start].min() <= prev_es['price']:
                    continue
                if es_win_end > es_win_start:
                    es_min_in_win = min(c_es_price, float(es_low_arr[es_win_start:es_win_end].min()))

            if has_nq_ohlc:
                nq_p_idx = nq_times.searchsorted(prev_nq['time'])
                nq_win_start = nq_times.searchsorted(sweep_start_time - htf_bar_delta * pre_win_bars)
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

            for step in range(1, 21):
                t = ltf_anchor + pd.Timedelta(minutes=step * ltf_min)
                es_idx = ltf_es_times.searchsorted(t) if has_ltf_es_close else -1
                has_es_bar = has_ltf_es_close and es_idx < len(ltf_es_times) and ltf_es_times[es_idx] == t
                nq_idx = ltf_nq_times.searchsorted(t) if has_ltf_nq_close else -1
                has_nq_bar = has_ltf_nq_close and nq_idx < len(ltf_nq_times) and ltf_nq_times[nq_idx] == t

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
                    confirm_time, confirm_asset, confirm_level = t, "ES", es_mss_level
                    break
                if has_nq_bar and nq_mss_level is not None and ltf_nq_close_arr[nq_idx] > nq_mss_level:
                    confirm_time, confirm_asset, confirm_level = t, "NQ", nq_mss_level
                    break

            if not disrespected and confirm_time is not None:
                dedup_key = ('Bullish_SMT', prev_es['time'], prev_nq['time'])
                if dedup_key in seen_confirmations:
                    break
                seen_confirmations.add(dedup_key)

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

                conf_close_time = confirm_time + ltf_close_offset
                es_is_holder = bool(nq_swept_ssl) or not res_nq_ohlc
                nq_is_holder = bool(es_swept_ssl) or not res_es_ohlc
                es_held_lvl = float(prev_es['price']) if nq_swept_ssl else float(c_es_price)
                nq_held_lvl = float(prev_nq['price']) if es_swept_ssl else float(c_nq_price)

                check_es_t = True if htf_min < 60 else bool(es_swept_ssl or not res_nq_ohlc)
                check_nq_t = True if htf_min < 60 else bool(nq_swept_ssl or not res_es_ohlc)

                status, resolved_time = _evaluate_post_confirmation_lifecycle(
                    is_bearish=False,
                    confirm_time=conf_close_time,
                    es_was_holder=es_is_holder,
                    nq_was_holder=nq_is_holder,
                    es_held_level=es_held_lvl,
                    nq_held_level=nq_held_lvl,
                    es_target_level=es_target,
                    nq_target_level=nq_target,
                    check_es_target=check_es_t,
                    check_nq_target=check_nq_t,
                    ltf_es_times=res_es_times,
                    ltf_es_high_arr=res_es_high,
                    ltf_es_low_arr=res_es_low,
                    has_ltf_es_ohlc=res_es_ohlc,
                    ltf_nq_times=res_nq_times,
                    ltf_nq_high_arr=res_nq_high,
                    ltf_nq_low_arr=res_nq_low,
                    has_ltf_nq_ohlc=res_nq_ohlc,
                )

                status_tag = (
                    f"[Status: {status} at {resolved_time.strftime('%m-%d %H:%M')}]"
                    if resolved_time is not None
                    else f"[Status: {status}]"
                )

                if es_swept_ssl:
                    desc = (
                        f"[{creation_time.strftime('%m-%d %H:%M')}] Bullish SMT ({timeframe}): "
                        f"ES swept SSL {prev_es['price']:.2f} (from {prev_es['time'].strftime('%m-%d %H:%M')}) "
                        f"to {c_es_price:.2f}, while NQ held above {prev_nq['price']:.2f} (from {prev_nq['time'].strftime('%m-%d %H:%M')}) "
                        f"--> {ltf_label} MSS on {confirm_asset} at [{conf_close_time.strftime('%m-%d %H:%M')}] (> {confirm_level:.2f}) {status_tag}"
                    )
                else:
                    desc = (
                        f"[{creation_time.strftime('%m-%d %H:%M')}] Bullish SMT ({timeframe}): "
                        f"NQ swept SSL {prev_nq['price']:.2f} (from {prev_nq['time'].strftime('%m-%d %H:%M')}) "
                        f"to {c_nq_price:.2f}, while ES held above {prev_es['price']:.2f} (from {prev_es['time'].strftime('%m-%d %H:%M')}) "
                        f"--> {ltf_label} MSS on {confirm_asset} at [{conf_close_time.strftime('%m-%d %H:%M')}] (> {confirm_level:.2f}) {status_tag}"
                    )

                is_intraday_post_open = bool(
                    htf_min <= 15
                    and creation_time.date() == conf_close_time.date()
                    and creation_time.time() >= time(9, 30)
                )

                smt_events.append({
                    'type': 'Bullish_SMT',
                    'time': creation_time,
                    'confirmation_time': confirm_time,
                    'confirmation_close_time': conf_close_time,
                    'status': status,
                    'resolved_time': resolved_time,
                    'resolved_close_time': resolved_time,
                    'es_target_level': es_target,
                    'nq_target_level': nq_target,
                    'es_invalidation_level': es_held_lvl,
                    'nq_invalidation_level': nq_held_lvl,
                    'timeframe': timeframe,
                    'confirmation_timeframe': ltf_label,
                    'is_intraday_post_open': is_intraday_post_open,
                    'description': desc
                })
                break

    smt_events.sort(key=lambda x: x['time'])
    return smt_events


def detect_session_smt_divergence(
    es_macro_pools: List[Dict[str, Any]],
    nq_macro_pools: List[Dict[str, Any]],
    es_swings_1m: List[Dict[str, Any]],
    nq_swings_1m: List[Dict[str, Any]],
    df_es_1m: pd.DataFrame,
    df_nq_1m: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Detects intraday Session SMT Divergences on unbroken Asia/London pools and attaches
    the opposing session high/low as the target so the SMT is properly marked RESPECTED
    as soon as that opposing session pool is taken (e.g., 2026-10-13 09:32 Asia SSL SMT
    -> RESPECTED at 09:41 when London BSL is swept).
    """
    events: List[Dict[str, Any]] = []
    if df_es_1m.empty or df_nq_1m.empty:
        return events

    es_times = df_es_1m.index
    nq_times = df_nq_1m.index
    es_high_arr = df_es_1m["high"].to_numpy(dtype=float)
    es_low_arr = df_es_1m["low"].to_numpy(dtype=float)
    es_close_arr = df_es_1m["close"].to_numpy(dtype=float)
    nq_high_arr = df_nq_1m["high"].to_numpy(dtype=float)
    nq_low_arr = df_nq_1m["low"].to_numpy(dtype=float)
    nq_close_arr = df_nq_1m["close"].to_numpy(dtype=float)

    es_highs = [s for s in es_swings_1m if s["type"] == "SwingHigh"]
    es_lows = [s for s in es_swings_1m if s["type"] == "SwingLow"]
    nq_highs = [s for s in nq_swings_1m if s["type"] == "SwingHigh"]
    nq_lows = [s for s in nq_swings_1m if s["type"] == "SwingLow"]
    es_high_times = [s["time"] for s in es_highs]
    es_low_times = [s["time"] for s in es_lows]
    nq_high_times = [s["time"] for s in nq_highs]
    nq_low_times = [s["time"] for s in nq_lows]

    def _ny_trade_date(ts: pd.Timestamp):
        return (ts + pd.Timedelta(hours=6)).date()

    es_by_key: Dict[Tuple[Any, str, str], Tuple[float, pd.Timestamp]] = {}
    for mp in es_macro_pools:
        p_type, sess, price, ts = extract_macro_dict_fields(mp)
        if sess in ("Asia", "London"):
            es_by_key[(_ny_trade_date(ts), sess.upper(), p_type)] = (price, ts)

    nq_by_key: Dict[Tuple[Any, str, str], Tuple[float, pd.Timestamp]] = {}
    for mp in nq_macro_pools:
        p_type, sess, price, ts = extract_macro_dict_fields(mp)
        if sess in ("Asia", "London"):
            nq_by_key[(_ny_trade_date(ts), sess.upper(), p_type)] = (price, ts)

    for key, (es_lvl, es_form_ts) in es_by_key.items():
        if key not in nq_by_key:
            continue
        nq_lvl, nq_form_ts = nq_by_key[key]
        trade_date, sess_label, p_type = key

        ny_open = pd.Timestamp(trade_date).tz_localize(es_times.tz) + pd.Timedelta(hours=9, minutes=30)
        ny_cutoff = ny_open + pd.Timedelta(minutes=60)

        es_start = es_times.searchsorted(ny_open)
        es_end = es_times.searchsorted(ny_cutoff)
        nq_start = nq_times.searchsorted(ny_open)
        if es_start >= es_end or nq_start >= len(nq_times):
            continue

        es_pre_idx = es_times.searchsorted(es_form_ts, side="right")
        nq_pre_idx = nq_times.searchsorted(nq_form_ts, side="right")
        if p_type == "BSL":
            if es_pre_idx < es_start and float(es_high_arr[es_pre_idx:es_start].max()) > es_lvl:
                continue
            if nq_pre_idx < nq_start and float(nq_high_arr[nq_pre_idx:nq_start].max()) > nq_lvl:
                continue
        else:
            if es_pre_idx < es_start and float(es_low_arr[es_pre_idx:es_start].min()) < es_lvl:
                continue
            if nq_pre_idx < nq_start and float(nq_low_arr[nq_pre_idx:nq_start].min()) < nq_lvl:
                continue

        # Find opposing session target level (London/Asia High for Bullish SMT; London/Asia Low for Bearish SMT)
        opp_type = "BSL" if p_type == "SSL" else "SSL"
        es_opp_candidates = [
            es_by_key[(trade_date, s_name, opp_type)][0]
            for s_name in ("LONDON", "ASIA")
            if (trade_date, s_name, opp_type) in es_by_key
        ]
        nq_opp_candidates = [
            nq_by_key[(trade_date, s_name, opp_type)][0]
            for s_name in ("LONDON", "ASIA")
            if (trade_date, s_name, opp_type) in nq_by_key
        ]
        es_opp_target = (min(es_opp_candidates) if p_type == "SSL" else max(es_opp_candidates)) if es_opp_candidates else None
        nq_opp_target = (min(nq_opp_candidates) if p_type == "SSL" else max(nq_opp_candidates)) if nq_opp_candidates else None

        for k in range(es_start, es_end):
            t = es_times[k]
            nq_k = nq_times.searchsorted(t)
            if nq_k >= len(nq_times) or nq_times[nq_k] != t:
                continue

            if p_type == "SSL":
                es_min_so_far = float(es_low_arr[es_start:k + 1].min())
                nq_min_so_far = float(nq_low_arr[nq_start:nq_k + 1].min())
                es_swept = es_min_so_far < es_lvl and nq_min_so_far > nq_lvl
                nq_swept = nq_min_so_far < nq_lvl and es_min_so_far > es_lvl
                if not (es_swept or nq_swept):
                    continue

                confirm_time, confirm_asset, confirm_level, disrespected = None, None, None, False
                for m in range(1, 16):
                    tm = t + pd.Timedelta(minutes=m)
                    e_i = es_times.searchsorted(tm)
                    n_i = nq_times.searchsorted(tm)
                    if e_i >= len(es_times) or es_times[e_i] != tm or n_i >= len(nq_times) or nq_times[n_i] != tm:
                        continue
                    if es_swept and nq_low_arr[n_i] <= nq_lvl:
                        disrespected = True
                        break
                    if nq_swept and es_low_arr[e_i] <= es_lvl:
                        disrespected = True
                        break

                    eh_idx = bisect.bisect_left(es_high_times, tm) - 1
                    nh_idx = bisect.bisect_left(nq_high_times, tm) - 1
                    es_mss = es_highs[eh_idx]["price"] if eh_idx >= 0 and es_high_times[eh_idx] >= ny_open else None
                    nq_mss = nq_highs[nh_idx]["price"] if nh_idx >= 0 and nq_high_times[nh_idx] >= ny_open else None

                    if es_swept and es_mss is not None and es_close_arr[e_i] > es_mss:
                        confirm_time, confirm_asset, confirm_level = tm, "ES", float(es_mss)
                        break
                    if nq_swept and nq_mss is not None and nq_close_arr[n_i] > nq_mss:
                        confirm_time, confirm_asset, confirm_level = tm, "NQ", float(nq_mss)
                        break

                if not disrespected and confirm_time is not None:
                    status, resolved_time = _evaluate_post_confirmation_lifecycle(
                        is_bearish=False,
                        confirm_time=confirm_time,
                        es_was_holder=bool(nq_swept),
                        nq_was_holder=bool(es_swept),
                        es_held_level=es_lvl,
                        nq_held_level=nq_lvl,
                        es_target_level=es_opp_target,
                        nq_target_level=nq_opp_target,
                        check_es_target=True,
                        check_nq_target=True,
                        ltf_es_times=es_times,
                        ltf_es_high_arr=es_high_arr,
                        ltf_es_low_arr=es_low_arr,
                        has_ltf_es_ohlc=True,
                        ltf_nq_times=nq_times,
                        ltf_nq_high_arr=nq_high_arr,
                        ltf_nq_low_arr=nq_low_arr,
                        has_ltf_nq_ohlc=True,
                    )
                    status_tag = (
                        f"[Status: {status} at {resolved_time.strftime('%m-%d %H:%M')}]"
                        if resolved_time is not None
                        else f"[Status: {status}]"
                    )
                    sweeper_str = (
                        f"ES swept {sess_label} SSL {es_lvl:.2f} to {es_min_so_far:.2f}, while NQ held above {sess_label} SSL {nq_lvl:.2f}"
                        if es_swept
                        else f"NQ swept {sess_label} SSL {nq_lvl:.2f} to {nq_min_so_far:.2f}, while ES held above {sess_label} SSL {es_lvl:.2f}"
                    )
                    desc = (
                        f"[{t.strftime('%m-%d %H:%M')}] Bullish SMT (Session_{sess_label}): "
                        f"{sweeper_str} --> 1m MSS on {confirm_asset} at [{confirm_time.strftime('%m-%d %H:%M')}] (> {confirm_level:.2f}) {status_tag}"
                    )
                    events.append({
                        "type": "Bullish_SMT",
                        "time": t,
                        "confirmation_time": confirm_time,
                        "confirmation_close_time": confirm_time,
                        "status": status,
                        "resolved_time": resolved_time,
                        "resolved_close_time": resolved_time,
                        "timeframe": "1min",
                        "confirmation_timeframe": "1m",
                        "is_intraday_post_open": True,
                        "description": desc,
                    })
                    break

            elif p_type == "BSL":
                es_max_so_far = float(es_high_arr[es_start:k + 1].max())
                nq_max_so_far = float(nq_high_arr[nq_start:nq_k + 1].max())
                es_swept = es_max_so_far > es_lvl and nq_max_so_far < nq_lvl
                nq_swept = nq_max_so_far > nq_lvl and es_max_so_far < es_lvl
                if not (es_swept or nq_swept):
                    continue

                confirm_time, confirm_asset, confirm_level, disrespected = None, None, None, False
                for m in range(1, 16):
                    tm = t + pd.Timedelta(minutes=m)
                    e_i = es_times.searchsorted(tm)
                    n_i = nq_times.searchsorted(tm)
                    if e_i >= len(es_times) or es_times[e_i] != tm or n_i >= len(nq_times) or nq_times[n_i] != tm:
                        continue
                    if es_swept and nq_high_arr[n_i] >= nq_lvl:
                        disrespected = True
                        break
                    if nq_swept and es_high_arr[e_i] >= es_lvl:
                        disrespected = True
                        break

                    el_idx = bisect.bisect_left(es_low_times, tm) - 1
                    nl_idx = bisect.bisect_left(nq_low_times, tm) - 1
                    es_mss = es_lows[el_idx]["price"] if el_idx >= 0 and es_low_times[el_idx] >= ny_open else None
                    nq_mss = nq_lows[nl_idx]["price"] if nl_idx >= 0 and nq_low_times[nl_idx] >= ny_open else None

                    if es_swept and es_mss is not None and es_close_arr[e_i] < es_mss:
                        confirm_time, confirm_asset, confirm_level = tm, "ES", float(es_mss)
                        break
                    if nq_swept and nq_mss is not None and nq_close_arr[n_i] < nq_mss:
                        confirm_time, confirm_asset, confirm_level = tm, "NQ", float(nq_mss)
                        break

                if not disrespected and confirm_time is not None:
                    status, resolved_time = _evaluate_post_confirmation_lifecycle(
                        is_bearish=True,
                        confirm_time=confirm_time,
                        es_was_holder=bool(nq_swept),
                        nq_was_holder=bool(es_swept),
                        es_held_level=es_lvl,
                        nq_held_level=nq_lvl,
                        es_target_level=es_opp_target,
                        nq_target_level=nq_opp_target,
                        check_es_target=True,
                        check_nq_target=True,
                        ltf_es_times=es_times,
                        ltf_es_high_arr=es_high_arr,
                        ltf_es_low_arr=es_low_arr,
                        has_ltf_es_ohlc=True,
                        ltf_nq_times=nq_times,
                        ltf_nq_high_arr=nq_high_arr,
                        ltf_nq_low_arr=nq_low_arr,
                        has_ltf_nq_ohlc=True,
                    )
                    status_tag = (
                        f"[Status: {status} at {resolved_time.strftime('%m-%d %H:%M')}]"
                        if resolved_time is not None
                        else f"[Status: {status}]"
                    )
                    sweeper_str = (
                        f"ES swept {sess_label} BSL {es_lvl:.2f} to {es_max_so_far:.2f}, while NQ held below {sess_label} BSL {nq_lvl:.2f}"
                        if es_swept
                        else f"NQ swept {sess_label} BSL {nq_lvl:.2f} to {nq_max_so_far:.2f}, while ES held below {sess_label} BSL {es_lvl:.2f}"
                    )
                    desc = (
                        f"[{t.strftime('%m-%d %H:%M')}] Bearish SMT (Session_{sess_label}): "
                        f"{sweeper_str} --> 1m MSS on {confirm_asset} at [{confirm_time.strftime('%m-%d %H:%M')}] (< {confirm_level:.2f}) {status_tag}"
                    )
                    events.append({
                        "type": "Bearish_SMT",
                        "time": t,
                        "confirmation_time": confirm_time,
                        "confirmation_close_time": confirm_time,
                        "status": status,
                        "resolved_time": resolved_time,
                        "resolved_close_time": resolved_time,
                        "timeframe": "1min",
                        "confirmation_timeframe": "1m",
                        "is_intraday_post_open": True,
                        "description": desc,
                    })
                    break

    return events