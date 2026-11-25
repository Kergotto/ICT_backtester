# Copyright (C) 2026 Kyrylo Homberh
# SPDX-License-Identifier: AGPL-3.0-only

from datetime import time
from typing import List, Dict, Optional, Any, Tuple
import bisect
import numpy as np
import pandas as pd

from core.data_loader import resample_timeframe
from domain.swings import detect_swings
from domain.fvg import track_fvgs, FairValueGap


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


_MIN_NS: int = 60_000_000_000


def _ts_to_ns(ts: pd.Timestamp) -> int:
    return int(ts.as_unit("ns").value) if hasattr(ts, "as_unit") else int(ts.value)


def _index_to_ns(idx: pd.DatetimeIndex) -> np.ndarray:
    if len(idx) == 0:
        return np.empty(0, dtype=np.int64)
    return idx.as_unit("ns").asi8 if hasattr(idx, "as_unit") else idx.asi8


def _first_idx_ge(arr: np.ndarray, start: int, threshold: float) -> int:
    n = len(arr)
    if start >= n:
        return -1
    chunk_end = start + 512
    if chunk_end < n:
        m = arr[start:chunk_end] >= threshold
        if m.any():
            return start + int(np.argmax(m))
        m2 = arr[chunk_end:] >= threshold
        if m2.any():
            return chunk_end + int(np.argmax(m2))
        return -1
    m = arr[start:] >= threshold
    return (start + int(np.argmax(m))) if m.any() else -1


def _first_idx_le(arr: np.ndarray, start: int, threshold: float) -> int:
    n = len(arr)
    if start >= n:
        return -1
    chunk_end = start + 512
    if chunk_end < n:
        m = arr[start:chunk_end] <= threshold
        if m.any():
            return start + int(np.argmax(m))
        m2 = arr[chunk_end:] <= threshold
        if m2.any():
            return chunk_end + int(np.argmax(m2))
        return -1
    m = arr[start:] <= threshold
    return (start + int(np.argmax(m))) if m.any() else -1


def _first_idx_gt(arr: np.ndarray, start: int, threshold: float) -> int:
    n = len(arr)
    if start >= n:
        return -1
    chunk_end = start + 512
    if chunk_end < n:
        m = arr[start:chunk_end] > threshold
        if m.any():
            return start + int(np.argmax(m))
        m2 = arr[chunk_end:] > threshold
        if m2.any():
            return chunk_end + int(np.argmax(m2))
        return -1
    m = arr[start:] > threshold
    return (start + int(np.argmax(m))) if m.any() else -1


def _first_idx_lt(arr: np.ndarray, start: int, threshold: float) -> int:
    n = len(arr)
    if start >= n:
        return -1
    chunk_end = start + 512
    if chunk_end < n:
        m = arr[start:chunk_end] < threshold
        if m.any():
            return start + int(np.argmax(m))
        m2 = arr[chunk_end:] < threshold
        if m2.any():
            return chunk_end + int(np.argmax(m2))
        return -1
    m = arr[start:] < threshold
    return (start + int(np.argmax(m))) if m.any() else -1


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
    if len(df_htf) < 3 or not all(c in df_htf.columns for c in ("high", "low")):
        return detect_swings(df_htf) if all(c in df_htf.columns for c in ("high", "low")) else []
    highs = df_htf["high"].to_numpy(dtype=float, copy=False)
    lows = df_htf["low"].to_numpy(dtype=float, copy=False)

    sh_mask = (highs[1:-1] > highs[:-2]) & (highs[1:-1] >= highs[2:])
    sl_mask = (lows[1:-1] < lows[:-2]) & (lows[1:-1] <= lows[2:])
    any_pos = np.flatnonzero(sh_mask | sl_mask)
    if len(any_pos) == 0:
        return []

    bar_indices = any_pos + 1
    swing_times = df_htf.index[bar_indices].tolist()
    sh_flags = sh_mask[any_pos].tolist()
    sl_flags = sl_mask[any_pos].tolist()
    h_vals = highs[bar_indices].tolist()
    l_vals = lows[bar_indices].tolist()

    swings: List[Dict[str, Any]] = []
    for k in range(len(any_pos)):
        t = swing_times[k]
        if sh_flags[k]:
            swings.append({"type": "SwingHigh", "time": t, "price": h_vals[k]})
        if sl_flags[k]:
            swings.append({"type": "SwingLow", "time": t, "price": l_vals[k]})
    return swings


def _augment_4h_leg_origin_swings(df_4h: pd.DataFrame, base_swings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(df_4h) < 3:
        return base_swings

    highs = df_4h["high"].to_numpy(dtype=float)
    lows = df_4h["low"].to_numpy(dtype=float)
    times = df_4h.index

    existing_high_times = {s["time"] for s in base_swings if s["type"] == "SwingHigh"}
    existing_low_times = {s["time"] for s in base_swings if s["type"] == "SwingLow"}
    augmented = list(base_swings)

    for i in range(len(df_4h) - 1):
        if highs[i] > highs[i + 1] and lows[i] > lows[i + 1]:
            if times[i] not in existing_high_times:
                augmented.append({"type": "SwingHigh", "time": times[i], "price": float(highs[i])})
                existing_high_times.add(times[i])
        if lows[i] < lows[i + 1] and highs[i] < highs[i + 1]:
            if times[i] not in existing_low_times:
                augmented.append({"type": "SwingLow", "time": times[i], "price": float(lows[i])})
                existing_low_times.add(times[i])

    augmented.sort(key=lambda s: s["time"])
    return augmented


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


def _get_closest_swing_ns(
    target_ns: int,
    target_swings: List[Dict[str, Any]],
    target_times_ns: List[int],
    max_diff_ns: int,
) -> Optional[Tuple[int, Dict[str, Any]]]:
    n = len(target_times_ns)
    if n == 0:
        return None
    idx = bisect.bisect_left(target_times_ns, target_ns)
    best_match = None
    best_ns = 0
    min_diff = max_diff_ns
    for i in (idx - 1, idx, idx + 1):
        if 0 <= i < n:
            ti = target_times_ns[i]
            diff = ti - target_ns if ti >= target_ns else target_ns - ti
            if diff <= min_diff:
                min_diff = diff
                best_match = target_swings[i]
                best_ns = ti
    if best_match is None:
        return None
    return best_ns, best_match


def _get_htf_companion_swing(
    target_time: pd.Timestamp,
    target_swings: List[Dict],
    target_times: List[pd.Timestamp],
    other_times: pd.DatetimeIndex,
    other_extreme_arr: np.ndarray,
    max_diff_minutes: int,
    htf_min: int,
    is_high: bool,
    primary_swing_times_set: Optional[set] = None,
) -> Optional[Dict[str, Any]]:
    if htf_min >= 60 and len(other_times) > 1:
        k = other_times.searchsorted(target_time)
        if k < len(other_times) - 1 and other_times[k] == target_time:
            if is_high and other_extreme_arr[k] > other_extreme_arr[k + 1]:
                return {"type": "SwingHigh", "time": target_time, "price": float(other_extreme_arr[k])}
            if not is_high and other_extreme_arr[k] < other_extreme_arr[k + 1]:
                return {"type": "SwingLow", "time": target_time, "price": float(other_extreme_arr[k])}
    match = get_closest_swing_fast(target_time, target_swings, target_times, max_diff_minutes=max_diff_minutes)
    if (
        match is not None
        and htf_min >= 5
        and match["time"] != target_time
        and primary_swing_times_set is not None
        and match["time"] in primary_swing_times_set
    ):
        return None
    return match


def _get_htf_companion_swing_ns(
    target_time: pd.Timestamp,
    target_ns: int,
    target_swings: List[Dict[str, Any]],
    target_times_ns: List[int],
    other_times_ns: List[int],
    other_extreme_arr: np.ndarray,
    max_diff_ns: int,
    htf_min: int,
    is_high: bool,
    primary_swing_times_ns_set: Optional[set] = None,
) -> Optional[Tuple[int, Dict[str, Any]]]:
    n_other = len(other_times_ns)
    if htf_min >= 60 and n_other > 1:
        k = bisect.bisect_left(other_times_ns, target_ns)
        if k < n_other - 1 and other_times_ns[k] == target_ns:
            if is_high and other_extreme_arr[k] > other_extreme_arr[k + 1]:
                return target_ns, {"type": "SwingHigh", "time": target_time, "price": float(other_extreme_arr[k])}
            if not is_high and other_extreme_arr[k] < other_extreme_arr[k + 1]:
                return target_ns, {"type": "SwingLow", "time": target_time, "price": float(other_extreme_arr[k])}
    res = _get_closest_swing_ns(target_ns, target_swings, target_times_ns, max_diff_ns)
    if res is None:
        return None
    match_ns, match = res
    if (
        htf_min >= 5
        and match_ns != target_ns
        and primary_swing_times_ns_set is not None
        and match_ns in primary_swing_times_ns_set
    ):
        return None
    return match_ns, match


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


def _find_ltf_sweep_start_ns(
    htf_sweep_time: pd.Timestamp,
    htf_sweep_ns: int,
    htf_min: int,
    ltf_times_ns: List[int],
    ltf_times_list: List[pd.Timestamp],
    ltf_extreme_arr: np.ndarray,
    prev_level: float,
    is_bearish: bool,
) -> Tuple[pd.Timestamp, int]:
    n_ltf = len(ltf_times_ns)
    if htf_min <= 1 or n_ltf == 0 or len(ltf_extreme_arr) == 0:
        return htf_sweep_time, htf_sweep_ns
    win_end_ns = htf_sweep_ns + htf_min * _MIN_NS
    start_idx = bisect.bisect_left(ltf_times_ns, htf_sweep_ns)
    end_idx = min(bisect.bisect_left(ltf_times_ns, win_end_ns), n_ltf)
    if is_bearish:
        for idx in range(start_idx, end_idx):
            if ltf_extreme_arr[idx] > prev_level:
                return ltf_times_list[idx], ltf_times_ns[idx]
    else:
        for idx in range(start_idx, end_idx):
            if ltf_extreme_arr[idx] < prev_level:
                return ltf_times_list[idx], ltf_times_ns[idx]
    return htf_sweep_time, htf_sweep_ns


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


def _find_intervening_extreme_ns(
    swing_prices: List[float],
    swing_times_ns: List[int],
    start_ns: int,
    end_ns: int,
    find_min: bool,
) -> Optional[float]:
    s_idx = bisect.bisect_right(swing_times_ns, start_ns)
    e_idx = bisect.bisect_left(swing_times_ns, end_ns)
    if s_idx >= e_idx:
        return None
    sub = swing_prices[s_idx:e_idx]
    return float(min(sub)) if find_min else float(max(sub))


def _find_prior_target_beyond_confirm(
    swings: List[Dict],
    swing_times: List[pd.Timestamp],
    before_time: pd.Timestamp,
    reference_price: float,
    is_bearish: bool,
    htf_min: int = 1,
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
    if htf_min == 1:
        return float(min(candidates)) if is_bearish else float(max(candidates))
    return float(max(candidates)) if is_bearish else float(min(candidates))


def _find_prior_target_beyond_confirm_ns(
    swing_prices: List[float],
    swing_times_ns: List[int],
    before_ns: int,
    reference_price: float,
    is_bearish: bool,
    htf_min: int = 1,
    lookback_count: int = 15,
) -> Optional[float]:
    end_idx = bisect.bisect_left(swing_times_ns, before_ns)
    start_idx = max(0, end_idx - lookback_count)
    candidates = []
    if is_bearish:
        for k in range(start_idx, end_idx):
            p = swing_prices[k]
            if p < reference_price:
                candidates.append(p)
    else:
        for k in range(start_idx, end_idx):
            p = swing_prices[k]
            if p > reference_price:
                candidates.append(p)
    if not candidates:
        return None
    if htf_min == 1:
        return float(min(candidates)) if is_bearish else float(max(candidates))
    return float(max(candidates)) if is_bearish else float(min(candidates))


def _prepare_fvg_index(
    fvgs: List[FairValueGap],
    ltf_times_ns: List[int],
) -> Tuple[Tuple[List[int], List[float], List[int]], Tuple[List[int], List[float], List[int]]]:
    bisi_cre_ns: List[int] = []
    bisi_bottoms: List[float] = []
    bisi_start_idx: List[int] = []
    sibi_cre_ns: List[int] = []
    sibi_tops: List[float] = []
    sibi_start_idx: List[int] = []
    bisect_right = bisect.bisect_right

    for f in fvgs:
        cre_ns = _ts_to_ns(f.creation_time)
        start_i = bisect_right(ltf_times_ns, cre_ns)
        if f.fvg_type == "BISI":
            bisi_cre_ns.append(cre_ns)
            bisi_bottoms.append(float(f.bottom))
            bisi_start_idx.append(start_i)
        elif f.fvg_type == "SIBI":
            sibi_cre_ns.append(cre_ns)
            sibi_tops.append(float(f.top))
            sibi_start_idx.append(start_i)

    return (bisi_cre_ns, bisi_bottoms, bisi_start_idx), (sibi_cre_ns, sibi_tops, sibi_start_idx)


def _check_ltf_ifvg_mss_level(
    t: pd.Timestamp,
    bar_idx: int,
    anchor_idx: int,
    prev_anchor_time: pd.Timestamp,
    ltf_swings: List[Dict[str, Any]],
    ltf_swing_times: List[pd.Timestamp],
    ltf_fvgs: List[FairValueGap],
    ltf_times: pd.DatetimeIndex,
    ltf_closes: np.ndarray,
    ltf_highs: np.ndarray,
    ltf_lows: np.ndarray,
    is_bearish: bool,
    htf_min: int = 240,
) -> Optional[float]:
    if not ltf_fvgs or bar_idx < 0 or bar_idx >= len(ltf_times) or len(ltf_highs) == 0 or len(ltf_lows) == 0:
        return None

    anchor_time = ltf_times[anchor_idx] if 0 <= anchor_idx < len(ltf_times) else t
    search_end_time = anchor_time if htf_min < 240 else t
    pos = bisect.bisect_left(ltf_swing_times, search_end_time) - 1
    if pos >= 0 and ltf_swings[pos]["time"] >= prev_anchor_time:
        leg_start_time = ltf_swings[pos]["time"]
        leg_start_price = float(ltf_swings[pos]["price"])
    else:
        if htf_min < 240:
            return None
        leg_start_time = prev_anchor_time
        leg_start_price = -1e9 if is_bearish else 1e9

    pre_sweep_cutoff = min(bar_idx, anchor_idx)
    fvg_end_time = anchor_time if htf_min < 240 else t

    if is_bearish:
        uninv = []
        for f in ltf_fvgs:
            if f.fvg_type != "BISI":
                continue
            if not (leg_start_time <= f.creation_time <= fvg_end_time and f.bottom >= leg_start_price):
                continue
            f_start = int(ltf_times.searchsorted(f.creation_time, side="right"))
            if htf_min < 240:
                if f_start < bar_idx and float(ltf_lows[f_start:bar_idx].min()) < f.bottom:
                    continue
            else:
                if f_start < pre_sweep_cutoff and float(ltf_closes[f_start:pre_sweep_cutoff].min()) < f.bottom:
                    continue
            uninv.append(f)
        if uninv:
            min_bottom = float(min(f.bottom for f in uninv))
            if ltf_closes[bar_idx] < min_bottom:
                return min_bottom
    else:
        uninv = []
        for f in ltf_fvgs:
            if f.fvg_type != "SIBI":
                continue
            if not (leg_start_time <= f.creation_time <= fvg_end_time and f.top <= leg_start_price):
                continue
            f_start = int(ltf_times.searchsorted(f.creation_time, side="right"))
            if htf_min < 240:
                if f_start < bar_idx and float(ltf_highs[f_start:bar_idx].max()) > f.top:
                    continue
            else:
                if f_start < pre_sweep_cutoff and float(ltf_closes[f_start:pre_sweep_cutoff].max()) > f.top:
                    continue
            uninv.append(f)
        if uninv:
            max_top = float(max(f.top for f in uninv))
            if ltf_closes[bar_idx] > max_top:
                return max_top

    return None


def _check_ltf_ifvg_mss_level_fast(
    t_ns: int,
    bar_idx: int,
    anchor_idx: int,
    prev_anchor_ns: int,
    ltf_swing_prices: List[float],
    ltf_swing_times_ns: List[int],
    fvg_cre_ns: List[int],
    fvg_levels: List[float],
    fvg_start_idx: List[int],
    ltf_times_ns: List[int],
    ltf_closes: np.ndarray,
    ltf_highs: np.ndarray,
    ltf_lows: np.ndarray,
    is_bearish: bool,
    htf_min: int = 240,
) -> Optional[float]:
    n_ltf = len(ltf_times_ns)
    if not fvg_cre_ns or bar_idx < 0 or bar_idx >= n_ltf or len(ltf_highs) == 0 or len(ltf_lows) == 0:
        return None

    anchor_ns = ltf_times_ns[anchor_idx] if 0 <= anchor_idx < n_ltf else t_ns
    search_end_ns = anchor_ns if htf_min < 240 else t_ns
    pos = bisect.bisect_left(ltf_swing_times_ns, search_end_ns) - 1
    if pos >= 0 and ltf_swing_times_ns[pos] >= prev_anchor_ns:
        leg_start_ns = ltf_swing_times_ns[pos]
        leg_start_price = ltf_swing_prices[pos]
    else:
        if htf_min < 240:
            return None
        leg_start_ns = prev_anchor_ns
        leg_start_price = -1e9 if is_bearish else 1e9

    pre_sweep_cutoff = bar_idx if bar_idx < anchor_idx else anchor_idx
    fvg_end_ns = anchor_ns if htf_min < 240 else t_ns

    lo = bisect.bisect_left(fvg_cre_ns, leg_start_ns)
    hi = bisect.bisect_right(fvg_cre_ns, fvg_end_ns)
    if lo >= hi:
        return None

    if is_bearish:
        min_bottom = float("inf")
        found = False
        if htf_min < 240:
            for f_i in range(lo, hi):
                f_bot = fvg_levels[f_i]
                if f_bot < leg_start_price:
                    continue
                f_start = fvg_start_idx[f_i]
                if f_start < bar_idx and float(ltf_lows[f_start:bar_idx].min()) < f_bot:
                    continue
                found = True
                if f_bot < min_bottom:
                    min_bottom = f_bot
        else:
            for f_i in range(lo, hi):
                f_bot = fvg_levels[f_i]
                if f_bot < leg_start_price:
                    continue
                f_start = fvg_start_idx[f_i]
                if f_start < pre_sweep_cutoff and float(ltf_closes[f_start:pre_sweep_cutoff].min()) < f_bot:
                    continue
                found = True
                if f_bot < min_bottom:
                    min_bottom = f_bot
        if found and ltf_closes[bar_idx] < min_bottom:
            return float(min_bottom)
    else:
        max_top = float("-inf")
        found = False
        if htf_min < 240:
            for f_i in range(lo, hi):
                f_top = fvg_levels[f_i]
                if f_top > leg_start_price:
                    continue
                f_start = fvg_start_idx[f_i]
                if f_start < bar_idx and float(ltf_highs[f_start:bar_idx].max()) > f_top:
                    continue
                found = True
                if f_top > max_top:
                    max_top = f_top
        else:
            for f_i in range(lo, hi):
                f_top = fvg_levels[f_i]
                if f_top > leg_start_price:
                    continue
                f_start = fvg_start_idx[f_i]
                if f_start < pre_sweep_cutoff and float(ltf_closes[f_start:pre_sweep_cutoff].max()) > f_top:
                    continue
                found = True
                if f_top > max_top:
                    max_top = f_top
        if found and ltf_closes[bar_idx] > max_top:
            return float(max_top)

    return None


def _evaluate_post_confirmation_lifecycle_ns(
    is_bearish: bool,
    confirm_ns: int,
    es_was_holder: bool,
    nq_was_holder: bool,
    es_held_level: float,
    nq_held_level: float,
    es_target_level: Optional[float],
    nq_target_level: Optional[float],
    check_es_target: bool,
    check_nq_target: bool,
    ltf_es_times_ns: List[int],
    ltf_es_times_list: List[pd.Timestamp],
    ltf_es_high_arr: np.ndarray,
    ltf_es_low_arr: np.ndarray,
    has_ltf_es_ohlc: bool,
    ltf_nq_times_ns: List[int],
    ltf_nq_times_list: List[pd.Timestamp],
    ltf_nq_high_arr: np.ndarray,
    ltf_nq_low_arr: np.ndarray,
    has_ltf_nq_ohlc: bool,
) -> tuple[str, Optional[pd.Timestamp]]:
    es_start = bisect.bisect_right(ltf_es_times_ns, confirm_ns) if has_ltf_es_ohlc else len(ltf_es_times_ns)
    nq_start = bisect.bisect_right(ltf_nq_times_ns, confirm_ns) if has_ltf_nq_ohlc else len(ltf_nq_times_ns)

    inv_ns: Optional[int] = None
    inv_ts: Optional[pd.Timestamp] = None
    res_ns: Optional[int] = None
    res_ts: Optional[pd.Timestamp] = None

    if is_bearish:
        if has_ltf_es_ohlc and es_was_holder:
            idx = _first_idx_ge(ltf_es_high_arr, es_start, es_held_level)
            if idx != -1:
                inv_ns = ltf_es_times_ns[idx]
                inv_ts = ltf_es_times_list[idx]
        if has_ltf_nq_ohlc and nq_was_holder:
            idx = _first_idx_ge(ltf_nq_high_arr, nq_start, nq_held_level)
            if idx != -1:
                t_cand = ltf_nq_times_ns[idx]
                if inv_ns is None or t_cand < inv_ns:
                    inv_ns = t_cand
                    inv_ts = ltf_nq_times_list[idx]

        if has_ltf_es_ohlc and check_es_target and es_target_level is not None:
            idx = _first_idx_lt(ltf_es_low_arr, es_start, es_target_level)
            if idx != -1:
                res_ns = ltf_es_times_ns[idx]
                res_ts = ltf_es_times_list[idx]
        if has_ltf_nq_ohlc and check_nq_target and nq_target_level is not None:
            idx = _first_idx_lt(ltf_nq_low_arr, nq_start, nq_target_level)
            if idx != -1:
                t_cand = ltf_nq_times_ns[idx]
                if res_ns is None or t_cand < res_ns:
                    res_ns = t_cand
                    res_ts = ltf_nq_times_list[idx]
    else:
        if has_ltf_es_ohlc and es_was_holder:
            idx = _first_idx_le(ltf_es_low_arr, es_start, es_held_level)
            if idx != -1:
                inv_ns = ltf_es_times_ns[idx]
                inv_ts = ltf_es_times_list[idx]
        if has_ltf_nq_ohlc and nq_was_holder:
            idx = _first_idx_le(ltf_nq_low_arr, nq_start, nq_held_level)
            if idx != -1:
                t_cand = ltf_nq_times_ns[idx]
                if inv_ns is None or t_cand < inv_ns:
                    inv_ns = t_cand
                    inv_ts = ltf_nq_times_list[idx]

        if has_ltf_es_ohlc and check_es_target and es_target_level is not None:
            idx = _first_idx_gt(ltf_es_high_arr, es_start, es_target_level)
            if idx != -1:
                res_ns = ltf_es_times_ns[idx]
                res_ts = ltf_es_times_list[idx]
        if has_ltf_nq_ohlc and check_nq_target and nq_target_level is not None:
            idx = _first_idx_gt(ltf_nq_high_arr, nq_start, nq_target_level)
            if idx != -1:
                t_cand = ltf_nq_times_ns[idx]
                if res_ns is None or t_cand < res_ns:
                    res_ns = t_cand
                    res_ts = ltf_nq_times_list[idx]

    if inv_ns is not None and (res_ns is None or inv_ns <= res_ns):
        return SMT_INVALIDATED, inv_ts
    if res_ns is not None:
        return SMT_RESPECTED, res_ts
    return SMT_CONFIRMED, None


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
    return _evaluate_post_confirmation_lifecycle_ns(
        is_bearish=is_bearish,
        confirm_ns=_ts_to_ns(confirm_time),
        es_was_holder=es_was_holder,
        nq_was_holder=nq_was_holder,
        es_held_level=es_held_level,
        nq_held_level=nq_held_level,
        es_target_level=es_target_level,
        nq_target_level=nq_target_level,
        check_es_target=check_es_target,
        check_nq_target=check_nq_target,
        ltf_es_times_ns=_index_to_ns(ltf_es_times).tolist() if has_ltf_es_ohlc else [],
        ltf_es_times_list=ltf_es_times.tolist() if has_ltf_es_ohlc else [],
        ltf_es_high_arr=ltf_es_high_arr,
        ltf_es_low_arr=ltf_es_low_arr,
        has_ltf_es_ohlc=has_ltf_es_ohlc,
        ltf_nq_times_ns=_index_to_ns(ltf_nq_times).tolist() if has_ltf_nq_ohlc else [],
        ltf_nq_times_list=ltf_nq_times.tolist() if has_ltf_nq_ohlc else [],
        ltf_nq_high_arr=ltf_nq_high_arr,
        ltf_nq_low_arr=ltf_nq_low_arr,
        has_ltf_nq_ohlc=has_ltf_nq_ohlc,
    )


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

    if htf_min == 240:
        es_swings = _augment_4h_leg_origin_swings(df_es, es_swings)
        nq_swings = _augment_4h_leg_origin_swings(df_nq, nq_swings)

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

    can_track_es_fvg = (
        not df_es_ltf.empty
        and all(c in df_es_ltf.columns for c in ("high", "low", "close"))
        and len(df_es_ltf) >= 3
    )
    can_track_nq_fvg = (
        not df_nq_ltf.empty
        and all(c in df_nq_ltf.columns for c in ("high", "low", "close"))
        and len(df_nq_ltf) >= 3
    )
    need_ifvg = htf_min in (1, 5, 240)
    ltf_es_fvgs = track_fvgs(df_es_ltf) if (can_track_es_fvg and need_ifvg) else []
    ltf_nq_fvgs = track_fvgs(df_nq_ltf) if (can_track_nq_fvg and need_ifvg) else []

    if htf_min <= 15:
        piv3_es_swings = detect_htf_swings(df_es_ltf) if can_track_es_fvg else ltf_es_swings
        piv3_nq_swings = detect_htf_swings(df_nq_ltf) if can_track_nq_fvg else ltf_nq_swings
        piv3_es_highs = [s for s in piv3_es_swings if s["type"] == "SwingHigh"]
        piv3_es_lows = [s for s in piv3_es_swings if s["type"] == "SwingLow"]
        piv3_nq_highs = [s for s in piv3_nq_swings if s["type"] == "SwingHigh"]
        piv3_nq_lows = [s for s in piv3_nq_swings if s["type"] == "SwingLow"]
        piv3_es_high_ns = [_ts_to_ns(s["time"]) for s in piv3_es_highs]
        piv3_es_high_prices = [float(s["price"]) for s in piv3_es_highs]
        piv3_es_low_ns = [_ts_to_ns(s["time"]) for s in piv3_es_lows]
        piv3_es_low_prices = [float(s["price"]) for s in piv3_es_lows]
        piv3_nq_high_ns = [_ts_to_ns(s["time"]) for s in piv3_nq_highs]
        piv3_nq_high_prices = [float(s["price"]) for s in piv3_nq_highs]
        piv3_nq_low_ns = [_ts_to_ns(s["time"]) for s in piv3_nq_lows]
        piv3_nq_low_prices = [float(s["price"]) for s in piv3_nq_lows]
    else:
        piv3_es_high_ns, piv3_es_high_prices = [], []
        piv3_es_low_ns, piv3_es_low_prices = [], []
        piv3_nq_high_ns, piv3_nq_high_prices = [], []
        piv3_nq_low_ns, piv3_nq_low_prices = [], []

    df_es_res = df_es_1m if (df_es_1m is not None and not df_es_1m.empty) else df_es_ltf
    df_nq_res = df_nq_1m if (df_nq_1m is not None and not df_nq_1m.empty) else df_nq_ltf
    res_es_ohlc = not df_es_res.empty and all(c in df_es_res.columns for c in ('high', 'low')) and len(df_es_res) > 1
    res_nq_ohlc = not df_nq_res.empty and all(c in df_nq_res.columns for c in ('high', 'low')) and len(df_nq_res) > 1
    res_es_times_ns = _index_to_ns(df_es_res.index).tolist() if res_es_ohlc else []
    res_es_times_list = df_es_res.index.tolist() if res_es_ohlc else []
    res_nq_times_ns = _index_to_ns(df_nq_res.index).tolist() if res_nq_ohlc else []
    res_nq_times_list = df_nq_res.index.tolist() if res_nq_ohlc else []
    res_es_high = df_es_res['high'].to_numpy(dtype=float, copy=False) if res_es_ohlc else np.empty(0, dtype=float)
    res_es_low = df_es_res['low'].to_numpy(dtype=float, copy=False) if res_es_ohlc else np.empty(0, dtype=float)
    res_nq_high = df_nq_res['high'].to_numpy(dtype=float, copy=False) if res_nq_ohlc else np.empty(0, dtype=float)
    res_nq_low = df_nq_res['low'].to_numpy(dtype=float, copy=False) if res_nq_ohlc else np.empty(0, dtype=float)

    es_highs = [s for s in es_swings if s['type'] == 'SwingHigh']
    es_lows = [s for s in es_swings if s['type'] == 'SwingLow']
    nq_highs = [s for s in nq_swings if s['type'] == 'SwingHigh']
    nq_lows = [s for s in nq_swings if s['type'] == 'SwingLow']

    es_high_ns = [_ts_to_ns(s['time']) for s in es_highs]
    es_high_prices = [float(s['price']) for s in es_highs]
    es_low_ns = [_ts_to_ns(s['time']) for s in es_lows]
    es_low_prices = [float(s['price']) for s in es_lows]
    nq_high_ns = [_ts_to_ns(s['time']) for s in nq_highs]
    nq_high_prices = [float(s['price']) for s in nq_highs]
    nq_low_ns = [_ts_to_ns(s['time']) for s in nq_lows]
    nq_low_prices = [float(s['price']) for s in nq_lows]
    es_high_ns_set = set(es_high_ns)
    es_low_ns_set = set(es_low_ns)

    ltf_es_highs = [s for s in ltf_es_swings if s['type'] == 'SwingHigh']
    ltf_es_lows = [s for s in ltf_es_swings if s['type'] == 'SwingLow']
    ltf_nq_highs = [s for s in ltf_nq_swings if s['type'] == 'SwingHigh']
    ltf_nq_lows = [s for s in ltf_nq_swings if s['type'] == 'SwingLow']

    ltf_es_high_ns = [_ts_to_ns(s['time']) for s in ltf_es_highs]
    ltf_es_high_prices = [float(s['price']) for s in ltf_es_highs]
    ltf_es_low_ns = [_ts_to_ns(s['time']) for s in ltf_es_lows]
    ltf_es_low_prices = [float(s['price']) for s in ltf_es_lows]
    ltf_nq_high_ns = [_ts_to_ns(s['time']) for s in ltf_nq_highs]
    ltf_nq_high_prices = [float(s['price']) for s in ltf_nq_highs]
    ltf_nq_low_ns = [_ts_to_ns(s['time']) for s in ltf_nq_lows]
    ltf_nq_low_prices = [float(s['price']) for s in ltf_nq_lows]

    has_es_ohlc = not df_es.empty and all(c in df_es.columns for c in ('close', 'high', 'low')) and len(df_es) > 1
    has_nq_ohlc = not df_nq.empty and all(c in df_nq.columns for c in ('close', 'high', 'low')) and len(df_nq) > 1

    es_times_ns = _index_to_ns(df_es.index).tolist() if not df_es.empty else []
    es_times_list = df_es.index.tolist() if not df_es.empty else []
    nq_times_ns = _index_to_ns(df_nq.index).tolist() if not df_nq.empty else []
    n_es_bars = len(es_times_ns)
    n_nq_bars = len(nq_times_ns)

    es_high_arr = df_es['high'].to_numpy(dtype=float, copy=False) if has_es_ohlc else np.empty(0, dtype=float)
    es_low_arr = df_es['low'].to_numpy(dtype=float, copy=False) if has_es_ohlc else np.empty(0, dtype=float)
    nq_high_arr = df_nq['high'].to_numpy(dtype=float, copy=False) if has_nq_ohlc else np.empty(0, dtype=float)
    nq_low_arr = df_nq['low'].to_numpy(dtype=float, copy=False) if has_nq_ohlc else np.empty(0, dtype=float)

    has_ltf_es_close = not df_es_ltf.empty and 'close' in df_es_ltf.columns
    has_ltf_nq_close = not df_nq_ltf.empty and 'close' in df_nq_ltf.columns
    has_ltf_es_ohlc = has_ltf_es_close and 'high' in df_es_ltf.columns and 'low' in df_es_ltf.columns and len(df_es_ltf) > 1
    has_ltf_nq_ohlc = has_ltf_nq_close and 'high' in df_nq_ltf.columns and 'low' in df_nq_ltf.columns and len(df_nq_ltf) > 1

    ltf_es_times_ns = _index_to_ns(df_es_ltf.index).tolist() if has_ltf_es_close else []
    ltf_es_times_list = df_es_ltf.index.tolist() if has_ltf_es_close else []
    ltf_nq_times_ns = _index_to_ns(df_nq_ltf.index).tolist() if has_ltf_nq_close else []
    ltf_nq_times_list = df_nq_ltf.index.tolist() if has_ltf_nq_close else []
    n_ltf_es = len(ltf_es_times_ns)
    n_ltf_nq = len(ltf_nq_times_ns)

    ltf_es_close_arr = df_es_ltf['close'].to_numpy(dtype=float, copy=False) if has_ltf_es_close else np.empty(0, dtype=float)
    ltf_nq_close_arr = df_nq_ltf['close'].to_numpy(dtype=float, copy=False) if has_ltf_nq_close else np.empty(0, dtype=float)
    ltf_es_high_arr = df_es_ltf['high'].to_numpy(dtype=float, copy=False) if has_ltf_es_ohlc else np.empty(0, dtype=float)
    ltf_es_low_arr = df_es_ltf['low'].to_numpy(dtype=float, copy=False) if has_ltf_es_ohlc else np.empty(0, dtype=float)
    ltf_nq_high_arr = df_nq_ltf['high'].to_numpy(dtype=float, copy=False) if has_ltf_nq_ohlc else np.empty(0, dtype=float)
    ltf_nq_low_arr = df_nq_ltf['low'].to_numpy(dtype=float, copy=False) if has_ltf_nq_ohlc else np.empty(0, dtype=float)

    (es_bisi_cre_ns, es_bisi_bottoms, es_bisi_start_idx), (es_sibi_cre_ns, es_sibi_tops, es_sibi_start_idx) = (
        _prepare_fvg_index(ltf_es_fvgs, ltf_es_times_ns) if ltf_es_fvgs else (([], [], []), ([], [], []))
    )
    (nq_bisi_cre_ns, nq_bisi_bottoms, nq_bisi_start_idx), (nq_sibi_cre_ns, nq_sibi_tops, nq_sibi_start_idx) = (
        _prepare_fvg_index(ltf_nq_fvgs, ltf_nq_times_ns) if ltf_nq_fvgs else (([], [], []), ([], [], []))
    )

    if htf_min == 1:
        sync_minutes = min(time_window_minutes, 3)
        min_sep_ns = 180 * 1_000_000_000
    else:
        sync_minutes = htf_min * sync_bars
        min_sep_ns = htf_min * _MIN_NS

    sync_ns = sync_minutes * _MIN_NS
    pre_win_bars = 2 if htf_min >= 60 else 1
    pre_win_ns = htf_min * pre_win_bars * _MIN_NS
    ltf_step_ns = ltf_min * _MIN_NS
    ltf_close_offset_min = max(0, ltf_min - 1)
    ltf_close_offset = pd.Timedelta(minutes=ltf_close_offset_min)
    ltf_close_offset_ns = ltf_close_offset_min * _MIN_NS
    seen_confirmations = set()

    bisect_left = bisect.bisect_left
    bisect_right = bisect.bisect_right

    bear_candidates: List[Tuple[pd.Timestamp, int, float, pd.Timestamp, int, float]] = []
    if htf_min < 240:
        for i_h, curr_es in enumerate(es_highs):
            es_t_ns = es_high_ns[i_h]
            res = _get_closest_swing_ns(es_t_ns, nq_highs, nq_high_ns, sync_ns)
            if res is not None:
                nq_t_ns, curr_nq = res
                bear_candidates.append((curr_es['time'], es_t_ns, es_high_prices[i_h], curr_nq['time'], nq_t_ns, float(curr_nq['price'])))

    if htf_min > 1 and has_es_ohlc and has_nq_ohlc:
        for k in range(n_es_bars):
            t_ns = es_times_ns[k]
            nq_k = bisect_left(nq_times_ns, t_ns)
            if nq_k < n_nq_bars and nq_times_ns[nq_k] == t_ns:
                t_bar = es_times_list[k]
                bear_candidates.append((t_bar, t_ns, float(es_high_arr[k]), t_bar, t_ns, float(nq_high_arr[nq_k])))

    bull_candidates: List[Tuple[pd.Timestamp, int, float, pd.Timestamp, int, float]] = []
    if htf_min < 240:
        for i_l, curr_es in enumerate(es_lows):
            es_t_ns = es_low_ns[i_l]
            res = _get_closest_swing_ns(es_t_ns, nq_lows, nq_low_ns, sync_ns)
            if res is not None:
                nq_t_ns, curr_nq = res
                bull_candidates.append((curr_es['time'], es_t_ns, es_low_prices[i_l], curr_nq['time'], nq_t_ns, float(curr_nq['price'])))

    if htf_min > 1 and has_es_ohlc and has_nq_ohlc:
        for k in range(n_es_bars):
            t_ns = es_times_ns[k]
            nq_k = bisect_left(nq_times_ns, t_ns)
            if nq_k < n_nq_bars and nq_times_ns[nq_k] == t_ns:
                t_bar = es_times_list[k]
                bull_candidates.append((t_bar, t_ns, float(es_low_arr[k]), t_bar, t_ns, float(nq_low_arr[nq_k])))

    start_step = 0 if htf_min == 240 else 1
    end_step = 5 if htf_min == 240 else 21

    # BEARISH SMT
    for c_es_time, c_es_ns, c_es_price, c_nq_time, c_nq_ns, c_nq_price in bear_candidates:
        if c_es_ns >= c_nq_ns:
            max_cand_ns = c_es_ns
            max_cand_time = c_es_time
            sweep_start_ns = c_nq_ns
        else:
            max_cand_ns = c_nq_ns
            max_cand_time = c_nq_time
            sweep_start_ns = c_es_ns

        prior_idx = bisect_left(es_high_ns, sweep_start_ns)
        lo_prior = prior_idx - 15 if prior_idx > 15 else 0

        for p_idx in range(prior_idx - 1, lo_prior - 1, -1):
            prev_es_ns = es_high_ns[p_idx]
            if (sweep_start_ns - prev_es_ns) < min_sep_ns:
                continue

            prev_es = es_highs[p_idx]
            prev_es_price = es_high_prices[p_idx]

            comp = _get_htf_companion_swing_ns(
                prev_es['time'], prev_es_ns, nq_highs, nq_high_ns, nq_times_ns, nq_high_arr,
                max_diff_ns=sync_ns, htf_min=htf_min, is_high=True,
                primary_swing_times_ns_set=es_high_ns_set,
            )
            if comp is None:
                continue
            prev_nq_ns, prev_nq = comp
            if prev_nq_ns >= c_nq_ns:
                continue
            prev_nq_price = float(prev_nq['price'])

            # Exact algebraic pre-filter: es_max_in_win >= c_es_price and nq_max_in_win >= c_nq_price always hold
            if not (
                (c_es_price > prev_es_price and c_nq_price < prev_nq_price)
                or (c_nq_price > prev_nq_price and c_es_price < prev_es_price)
            ):
                continue

            es_max_in_win = c_es_price
            nq_max_in_win = c_nq_price

            if has_es_ohlc:
                es_p_idx = bisect_left(es_times_ns, prev_es_ns)
                es_win_start = bisect_left(es_times_ns, sweep_start_ns - pre_win_ns)
                es_win_end = bisect_right(es_times_ns, max_cand_ns)
                if es_win_end > n_es_bars:
                    es_win_end = n_es_bars
                if es_win_start > es_p_idx + 2 and es_high_arr[es_p_idx + 2:es_win_start].max() > prev_es_price:
                    continue
                if es_win_end > es_win_start:
                    w_max = float(es_high_arr[es_win_start:es_win_end].max())
                    if w_max > es_max_in_win:
                        es_max_in_win = w_max

            if has_nq_ohlc:
                nq_p_idx = bisect_left(nq_times_ns, prev_nq_ns)
                nq_win_start = bisect_left(nq_times_ns, sweep_start_ns - pre_win_ns)
                nq_win_end = bisect_right(nq_times_ns, max_cand_ns)
                if nq_win_end > n_nq_bars:
                    nq_win_end = n_nq_bars
                if nq_win_start > nq_p_idx + 2 and nq_high_arr[nq_p_idx + 2:nq_win_start].max() > prev_nq_price:
                    continue
                if nq_win_end > nq_win_start:
                    w_max = float(nq_high_arr[nq_win_start:nq_win_end].max())
                    if w_max > nq_max_in_win:
                        nq_max_in_win = w_max

            es_swept_bsl = (c_es_price > prev_es_price) and (nq_max_in_win < prev_nq_price)
            nq_swept_bsl = (c_nq_price > prev_nq_price) and (es_max_in_win < prev_es_price)

            if not (es_swept_bsl or nq_swept_bsl):
                continue

            if has_es_ohlc and has_nq_ohlc:
                if es_swept_bsl:
                    creation_time, creation_ns = c_es_time, c_es_ns
                else:
                    creation_time, creation_ns = c_nq_time, c_nq_ns
            else:
                creation_time, creation_ns = max_cand_time, max_cand_ns

            if htf_min == 1:
                ltf_anchor_time, ltf_anchor_ns = creation_time, creation_ns
            elif es_swept_bsl:
                ltf_anchor_time, ltf_anchor_ns = _find_ltf_sweep_start_ns(
                    c_es_time, c_es_ns, htf_min, ltf_es_times_ns, ltf_es_times_list, ltf_es_high_arr, prev_es_price, is_bearish=True
                )
            else:
                ltf_anchor_time, ltf_anchor_ns = _find_ltf_sweep_start_ns(
                    c_nq_time, c_nq_ns, htf_min, ltf_nq_times_ns, ltf_nq_times_list, ltf_nq_high_arr, prev_nq_price, is_bearish=True
                )

            es_end_bound_ns = c_es_ns if c_es_ns >= ltf_anchor_ns else ltf_anchor_ns
            nq_end_bound_ns = c_nq_ns if c_nq_ns >= ltf_anchor_ns else ltf_anchor_ns

            es_intervening_low = _find_intervening_extreme_ns(
                es_low_prices, es_low_ns, prev_es_ns, es_end_bound_ns, find_min=True
            )
            if htf_min <= 15 and piv3_es_low_ns:
                piv_es_low = _find_intervening_extreme_ns(
                    piv3_es_low_prices, piv3_es_low_ns, prev_es_ns, es_end_bound_ns, find_min=True
                )
                if piv_es_low is not None:
                    es_intervening_low = min(es_intervening_low, piv_es_low) if es_intervening_low is not None else piv_es_low

            nq_intervening_low = _find_intervening_extreme_ns(
                nq_low_prices, nq_low_ns, prev_nq_ns, nq_end_bound_ns, find_min=True
            )
            if htf_min <= 15 and piv3_nq_low_ns:
                piv_nq_low = _find_intervening_extreme_ns(
                    piv3_nq_low_prices, piv3_nq_low_ns, prev_nq_ns, nq_end_bound_ns, find_min=True
                )
                if piv_nq_low is not None:
                    nq_intervening_low = min(nq_intervening_low, piv_nq_low) if nq_intervening_low is not None else piv_nq_low

            es_anchor_idx = bisect_left(ltf_es_times_ns, ltf_anchor_ns) if has_ltf_es_close else 0
            nq_anchor_idx = bisect_left(ltf_nq_times_ns, ltf_anchor_ns) if has_ltf_nq_close else 0

            confirm_time = None
            confirm_ns = 0
            confirm_asset = None
            confirm_level = None
            confirm_kind = "MSS"
            disrespected = False

            if htf_min < 240:
                if has_ltf_es_ohlc and es_anchor_idx < n_ltf_es and ltf_es_times_ns[es_anchor_idx] == ltf_anchor_ns:
                    if es_intervening_low is not None and ltf_es_low_arr[es_anchor_idx] < es_intervening_low:
                        disrespected = True
                if has_ltf_nq_ohlc and nq_anchor_idx < n_ltf_nq and ltf_nq_times_ns[nq_anchor_idx] == ltf_anchor_ns:
                    if nq_intervening_low is not None and ltf_nq_low_arr[nq_anchor_idx] < nq_intervening_low:
                        disrespected = True

            for step in range(start_step, end_step):
                if disrespected:
                    break
                t_ns = ltf_anchor_ns + step * ltf_step_ns
                es_idx = bisect_left(ltf_es_times_ns, t_ns) if has_ltf_es_close else -1
                has_es_bar = has_ltf_es_close and es_idx < n_ltf_es and ltf_es_times_ns[es_idx] == t_ns
                nq_idx = bisect_left(ltf_nq_times_ns, t_ns) if has_ltf_nq_close else -1
                has_nq_bar = has_ltf_nq_close and nq_idx < n_ltf_nq and ltf_nq_times_ns[nq_idx] == t_ns

                if es_swept_bsl and has_ltf_nq_ohlc and has_nq_bar:
                    if ltf_nq_high_arr[nq_idx] >= prev_nq_price:
                        disrespected = True
                        break
                if nq_swept_bsl and has_ltf_es_ohlc and has_es_bar:
                    if ltf_es_high_arr[es_idx] >= prev_es_price:
                        disrespected = True
                        break

                es_l_idx = bisect_left(ltf_es_low_ns, t_ns) - 1
                es_mss_level = (
                    ltf_es_low_prices[es_l_idx]
                    if (es_l_idx >= 0 and ltf_es_low_ns[es_l_idx] > prev_es_ns)
                    else None
                )
                nq_l_idx = bisect_left(ltf_nq_low_ns, t_ns) - 1
                nq_mss_level = (
                    ltf_nq_low_prices[nq_l_idx]
                    if (nq_l_idx >= 0 and ltf_nq_low_ns[nq_l_idx] > prev_nq_ns)
                    else None
                )

                es_ifvg_level = (
                    _check_ltf_ifvg_mss_level_fast(
                        t_ns, es_idx, es_anchor_idx, prev_es_ns, ltf_es_low_prices, ltf_es_low_ns,
                        es_bisi_cre_ns, es_bisi_bottoms, es_bisi_start_idx,
                        ltf_es_times_ns, ltf_es_close_arr, ltf_es_high_arr, ltf_es_low_arr,
                        is_bearish=True, htf_min=htf_min,
                    )
                    if (need_ifvg and has_es_bar and has_ltf_es_ohlc)
                    else None
                )
                nq_ifvg_level = (
                    _check_ltf_ifvg_mss_level_fast(
                        t_ns, nq_idx, nq_anchor_idx, prev_nq_ns, ltf_nq_low_prices, ltf_nq_low_ns,
                        nq_bisi_cre_ns, nq_bisi_bottoms, nq_bisi_start_idx,
                        ltf_nq_times_ns, ltf_nq_close_arr, ltf_nq_high_arr, ltf_nq_low_arr,
                        is_bearish=True, htf_min=htf_min,
                    )
                    if (need_ifvg and has_nq_bar and has_ltf_nq_ohlc)
                    else None
                )

                if has_ltf_es_ohlc and has_es_bar and es_intervening_low is not None:
                    if es_mss_level is None or es_intervening_low <= es_mss_level:
                        if ltf_es_low_arr[es_idx] < es_intervening_low and not (
                            (es_mss_level is not None and ltf_es_close_arr[es_idx] < es_mss_level)
                            or es_ifvg_level is not None
                        ):
                            disrespected = True
                            break
                if has_ltf_nq_ohlc and has_nq_bar and nq_intervening_low is not None:
                    if nq_mss_level is None or nq_intervening_low <= nq_mss_level:
                        if ltf_nq_low_arr[nq_idx] < nq_intervening_low and not (
                            (nq_mss_level is not None and ltf_nq_close_arr[nq_idx] < nq_mss_level)
                            or nq_ifvg_level is not None
                        ):
                            disrespected = True
                            break

                if has_es_bar and es_mss_level is not None and ltf_es_close_arr[es_idx] < es_mss_level:
                    confirm_time, confirm_ns, confirm_asset, confirm_level, confirm_kind = (
                        ltf_es_times_list[es_idx], t_ns, "ES", es_mss_level, "MSS"
                    )
                    break
                if has_nq_bar and nq_mss_level is not None and ltf_nq_close_arr[nq_idx] < nq_mss_level:
                    confirm_time, confirm_ns, confirm_asset, confirm_level, confirm_kind = (
                        ltf_nq_times_list[nq_idx], t_ns, "NQ", nq_mss_level, "MSS"
                    )
                    break
                if es_ifvg_level is not None:
                    confirm_time, confirm_ns, confirm_asset, confirm_level, confirm_kind = (
                        ltf_es_times_list[es_idx], t_ns, "ES", es_ifvg_level, "IFVG_MSS"
                    )
                    break
                if nq_ifvg_level is not None:
                    confirm_time, confirm_ns, confirm_asset, confirm_level, confirm_kind = (
                        ltf_nq_times_list[nq_idx], t_ns, "NQ", nq_ifvg_level, "IFVG_MSS"
                    )
                    break

            if not disrespected and confirm_time is not None:
                dedup_key = ('Bearish_SMT', prev_es_ns, prev_nq_ns)
                if dedup_key in seen_confirmations:
                    break
                seen_confirmations.add(dedup_key)

                es_conf_idx = bisect_left(ltf_es_times_ns, confirm_ns) if has_ltf_es_close else -1
                es_ref_low = (
                    float(confirm_level)
                    if confirm_asset == "ES"
                    else (
                        float(ltf_es_low_arr[es_conf_idx])
                        if (has_ltf_es_ohlc and es_conf_idx < n_ltf_es and ltf_es_times_ns[es_conf_idx] == confirm_ns)
                        else float(confirm_level)
                    )
                )
                nq_conf_idx = bisect_left(ltf_nq_times_ns, confirm_ns) if has_ltf_nq_close else -1
                nq_ref_low = (
                    float(confirm_level)
                    if confirm_asset == "NQ"
                    else (
                        float(ltf_nq_low_arr[nq_conf_idx])
                        if (has_ltf_nq_ohlc and nq_conf_idx < n_ltf_nq and ltf_nq_times_ns[nq_conf_idx] == confirm_ns)
                        else float(confirm_level)
                    )
                )

                es_intervening_valid = es_intervening_low is not None and es_intervening_low < es_ref_low
                nq_intervening_valid = nq_intervening_low is not None and nq_intervening_low < nq_ref_low

                es_target = (
                    es_intervening_low
                    if es_intervening_valid
                    else _find_prior_target_beyond_confirm_ns(
                        es_low_prices, es_low_ns, c_es_ns, es_ref_low, is_bearish=True, htf_min=htf_min
                    )
                )
                nq_target = (
                    nq_intervening_low
                    if nq_intervening_valid
                    else _find_prior_target_beyond_confirm_ns(
                        nq_low_prices, nq_low_ns, c_nq_ns, nq_ref_low, is_bearish=True, htf_min=htf_min
                    )
                )

                conf_close_time = confirm_time + ltf_close_offset
                conf_close_ns = confirm_ns + ltf_close_offset_ns
                es_is_holder = bool(nq_swept_bsl) or not res_nq_ohlc
                nq_is_holder = bool(es_swept_bsl) or not res_es_ohlc
                es_held_lvl = prev_es_price if nq_swept_bsl else float(c_es_price)
                nq_held_lvl = prev_nq_price if es_swept_bsl else float(c_nq_price)

                if es_intervening_valid or nq_intervening_valid:
                    check_es_t = True
                    check_nq_t = True
                elif htf_min >= 60:
                    check_es_t = bool(es_swept_bsl or not res_nq_ohlc)
                    check_nq_t = bool(nq_swept_bsl or not res_es_ohlc)
                elif htf_min == 1:
                    check_es_t = (confirm_asset == "ES")
                    check_nq_t = (confirm_asset == "NQ")
                else:
                    check_es_t = True
                    check_nq_t = True

                status, resolved_time = _evaluate_post_confirmation_lifecycle_ns(
                    is_bearish=True,
                    confirm_ns=conf_close_ns,
                    es_was_holder=es_is_holder,
                    nq_was_holder=nq_is_holder,
                    es_held_level=es_held_lvl,
                    nq_held_level=nq_held_lvl,
                    es_target_level=es_target,
                    nq_target_level=nq_target,
                    check_es_target=check_es_t,
                    check_nq_target=check_nq_t,
                    ltf_es_times_ns=res_es_times_ns,
                    ltf_es_times_list=res_es_times_list,
                    ltf_es_high_arr=res_es_high,
                    ltf_es_low_arr=res_es_low,
                    has_ltf_es_ohlc=res_es_ohlc,
                    ltf_nq_times_ns=res_nq_times_ns,
                    ltf_nq_times_list=res_nq_times_list,
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
                        f"ES swept BSL {prev_es_price:.2f} (from {prev_es['time'].strftime('%m-%d %H:%M')}) "
                        f"to {c_es_price:.2f}, while NQ held below {prev_nq_price:.2f} (from {prev_nq['time'].strftime('%m-%d %H:%M')}) "
                        f"--> {ltf_label} {confirm_kind} on {confirm_asset} at [{conf_close_time.strftime('%m-%d %H:%M')}] (< {confirm_level:.2f}) {status_tag}"
                    )
                else:
                    desc = (
                        f"[{creation_time.strftime('%m-%d %H:%M')}] Bearish SMT ({timeframe}): "
                        f"NQ swept BSL {prev_nq_price:.2f} (from {prev_nq['time'].strftime('%m-%d %H:%M')}) "
                        f"to {c_nq_price:.2f}, while ES held below {prev_es_price:.2f} (from {prev_es['time'].strftime('%m-%d %H:%M')}) "
                        f"--> {ltf_label} {confirm_kind} on {confirm_asset} at [{conf_close_time.strftime('%m-%d %H:%M')}] (< {confirm_level:.2f}) {status_tag}"
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
                    'swept_level': prev_es_price if es_swept_bsl else prev_nq_price,
                    'sweep_extreme': float(c_es_price) if es_swept_bsl else float(c_nq_price),
                    'timeframe': timeframe,
                    'confirmation_timeframe': ltf_label,
                    'is_intraday_post_open': is_intraday_post_open,
                    'description': desc
                })
                break

    # BULLISH SMT
    for c_es_time, c_es_ns, c_es_price, c_nq_time, c_nq_ns, c_nq_price in bull_candidates:
        if c_es_ns >= c_nq_ns:
            max_cand_ns = c_es_ns
            max_cand_time = c_es_time
            sweep_start_ns = c_nq_ns
        else:
            max_cand_ns = c_nq_ns
            max_cand_time = c_nq_time
            sweep_start_ns = c_es_ns

        prior_idx = bisect_left(es_low_ns, sweep_start_ns)
        lo_prior = prior_idx - 15 if prior_idx > 15 else 0

        for p_idx in range(prior_idx - 1, lo_prior - 1, -1):
            prev_es_ns = es_low_ns[p_idx]
            if (sweep_start_ns - prev_es_ns) < min_sep_ns:
                continue

            prev_es = es_lows[p_idx]
            prev_es_price = es_low_prices[p_idx]

            comp = _get_htf_companion_swing_ns(
                prev_es['time'], prev_es_ns, nq_lows, nq_low_ns, nq_times_ns, nq_low_arr,
                max_diff_ns=sync_ns, htf_min=htf_min, is_high=False,
                primary_swing_times_ns_set=es_low_ns_set,
            )
            if comp is None:
                continue
            prev_nq_ns, prev_nq = comp
            if prev_nq_ns >= c_nq_ns:
                continue
            prev_nq_price = float(prev_nq['price'])

            # Exact algebraic pre-filter: es_min_in_win <= c_es_price and nq_min_in_win <= c_nq_price always hold
            if not (
                (c_es_price < prev_es_price and c_nq_price > prev_nq_price)
                or (c_nq_price < prev_nq_price and c_es_price > prev_es_price)
            ):
                continue

            es_min_in_win = c_es_price
            nq_min_in_win = c_nq_price

            if has_es_ohlc:
                es_p_idx = bisect_left(es_times_ns, prev_es_ns)
                es_win_start = bisect_left(es_times_ns, sweep_start_ns - pre_win_ns)
                es_win_end = bisect_right(es_times_ns, max_cand_ns)
                if es_win_end > n_es_bars:
                    es_win_end = n_es_bars
                if es_win_start > es_p_idx + 2 and es_low_arr[es_p_idx + 2:es_win_start].min() < prev_es_price:
                    continue
                if es_win_end > es_win_start:
                    w_min = float(es_low_arr[es_win_start:es_win_end].min())
                    if w_min < es_min_in_win:
                        es_min_in_win = w_min

            if has_nq_ohlc:
                nq_p_idx = bisect_left(nq_times_ns, prev_nq_ns)
                nq_win_start = bisect_left(nq_times_ns, sweep_start_ns - pre_win_ns)
                nq_win_end = bisect_right(nq_times_ns, max_cand_ns)
                if nq_win_end > n_nq_bars:
                    nq_win_end = n_nq_bars
                if nq_win_start > nq_p_idx + 2 and nq_low_arr[nq_p_idx + 2:nq_win_start].min() < prev_nq_price:
                    continue
                if nq_win_end > nq_win_start:
                    w_min = float(nq_low_arr[nq_win_start:nq_win_end].min())
                    if w_min < nq_min_in_win:
                        nq_min_in_win = w_min

            es_swept_ssl = (c_es_price < prev_es_price) and (nq_min_in_win > prev_nq_price)
            nq_swept_ssl = (c_nq_price < prev_nq_price) and (es_min_in_win > prev_es_price)

            if not (es_swept_ssl or nq_swept_ssl):
                continue

            if has_es_ohlc and has_nq_ohlc:
                if es_swept_ssl:
                    creation_time, creation_ns = c_es_time, c_es_ns
                else:
                    creation_time, creation_ns = c_nq_time, c_nq_ns
            else:
                creation_time, creation_ns = max_cand_time, max_cand_ns

            if htf_min == 1:
                ltf_anchor_time, ltf_anchor_ns = creation_time, creation_ns
            elif es_swept_ssl:
                ltf_anchor_time, ltf_anchor_ns = _find_ltf_sweep_start_ns(
                    c_es_time, c_es_ns, htf_min, ltf_es_times_ns, ltf_es_times_list, ltf_es_low_arr, prev_es_price, is_bearish=False
                )
            else:
                ltf_anchor_time, ltf_anchor_ns = _find_ltf_sweep_start_ns(
                    c_nq_time, c_nq_ns, htf_min, ltf_nq_times_ns, ltf_nq_times_list, ltf_nq_low_arr, prev_nq_price, is_bearish=False
                )

            es_end_bound_ns = c_es_ns if c_es_ns >= ltf_anchor_ns else ltf_anchor_ns
            nq_end_bound_ns = c_nq_ns if c_nq_ns >= ltf_anchor_ns else ltf_anchor_ns

            es_intervening_high = _find_intervening_extreme_ns(
                es_high_prices, es_high_ns, prev_es_ns, es_end_bound_ns, find_min=False
            )
            if htf_min <= 15 and piv3_es_high_ns:
                piv_es_high = _find_intervening_extreme_ns(
                    piv3_es_high_prices, piv3_es_high_ns, prev_es_ns, es_end_bound_ns, find_min=False
                )
                if piv_es_high is not None:
                    es_intervening_high = max(es_intervening_high, piv_es_high) if es_intervening_high is not None else piv_es_high

            nq_intervening_high = _find_intervening_extreme_ns(
                nq_high_prices, nq_high_ns, prev_nq_ns, nq_end_bound_ns, find_min=False
            )
            if htf_min <= 15 and piv3_nq_high_ns:
                piv_nq_high = _find_intervening_extreme_ns(
                    piv3_nq_high_prices, piv3_nq_high_ns, prev_nq_ns, nq_end_bound_ns, find_min=False
                )
                if piv_nq_high is not None:
                    nq_intervening_high = max(nq_intervening_high, piv_nq_high) if nq_intervening_high is not None else piv_nq_high

            es_anchor_idx = bisect_left(ltf_es_times_ns, ltf_anchor_ns) if has_ltf_es_close else 0
            nq_anchor_idx = bisect_left(ltf_nq_times_ns, ltf_anchor_ns) if has_ltf_nq_close else 0

            confirm_time = None
            confirm_ns = 0
            confirm_asset = None
            confirm_level = None
            confirm_kind = "MSS"
            disrespected = False

            if htf_min < 240:
                if has_ltf_es_ohlc and es_anchor_idx < n_ltf_es and ltf_es_times_ns[es_anchor_idx] == ltf_anchor_ns:
                    if es_intervening_high is not None and ltf_es_high_arr[es_anchor_idx] > es_intervening_high:
                        disrespected = True
                if has_ltf_nq_ohlc and nq_anchor_idx < n_ltf_nq and ltf_nq_times_ns[nq_anchor_idx] == ltf_anchor_ns:
                    if nq_intervening_high is not None and ltf_nq_high_arr[nq_anchor_idx] > nq_intervening_high:
                        disrespected = True

            for step in range(start_step, end_step):
                if disrespected:
                    break
                t_ns = ltf_anchor_ns + step * ltf_step_ns
                es_idx = bisect_left(ltf_es_times_ns, t_ns) if has_ltf_es_close else -1
                has_es_bar = has_ltf_es_close and es_idx < n_ltf_es and ltf_es_times_ns[es_idx] == t_ns
                nq_idx = bisect_left(ltf_nq_times_ns, t_ns) if has_ltf_nq_close else -1
                has_nq_bar = has_ltf_nq_close and nq_idx < n_ltf_nq and ltf_nq_times_ns[nq_idx] == t_ns

                if es_swept_ssl and has_ltf_nq_ohlc and has_nq_bar:
                    if ltf_nq_low_arr[nq_idx] <= prev_nq_price:
                        disrespected = True
                        break
                if nq_swept_ssl and has_ltf_es_ohlc and has_es_bar:
                    if ltf_es_low_arr[es_idx] <= prev_es_price:
                        disrespected = True
                        break

                es_h_idx = bisect_left(ltf_es_high_ns, t_ns) - 1
                es_mss_level = (
                    ltf_es_high_prices[es_h_idx]
                    if (es_h_idx >= 0 and ltf_es_high_ns[es_h_idx] > prev_es_ns)
                    else None
                )
                nq_h_idx = bisect_left(ltf_nq_high_ns, t_ns) - 1
                nq_mss_level = (
                    ltf_nq_high_prices[nq_h_idx]
                    if (nq_h_idx >= 0 and ltf_nq_high_ns[nq_h_idx] > prev_nq_ns)
                    else None
                )

                es_ifvg_level = (
                    _check_ltf_ifvg_mss_level_fast(
                        t_ns, es_idx, es_anchor_idx, prev_es_ns, ltf_es_high_prices, ltf_es_high_ns,
                        es_sibi_cre_ns, es_sibi_tops, es_sibi_start_idx,
                        ltf_es_times_ns, ltf_es_close_arr, ltf_es_high_arr, ltf_es_low_arr,
                        is_bearish=False, htf_min=htf_min,
                    )
                    if (need_ifvg and has_es_bar and has_ltf_es_ohlc)
                    else None
                )
                nq_ifvg_level = (
                    _check_ltf_ifvg_mss_level_fast(
                        t_ns, nq_idx, nq_anchor_idx, prev_nq_ns, ltf_nq_high_prices, ltf_nq_high_ns,
                        nq_sibi_cre_ns, nq_sibi_tops, nq_sibi_start_idx,
                        ltf_nq_times_ns, ltf_nq_close_arr, ltf_nq_high_arr, ltf_nq_low_arr,
                        is_bearish=False, htf_min=htf_min,
                    )
                    if (need_ifvg and has_nq_bar and has_ltf_nq_ohlc)
                    else None
                )

                if has_ltf_es_ohlc and has_es_bar and es_intervening_high is not None:
                    if es_mss_level is None or es_intervening_high >= es_mss_level:
                        if ltf_es_high_arr[es_idx] > es_intervening_high and not (
                            (es_mss_level is not None and ltf_es_close_arr[es_idx] > es_mss_level)
                            or es_ifvg_level is not None
                        ):
                            disrespected = True
                            break
                if has_ltf_nq_ohlc and has_nq_bar and nq_intervening_high is not None:
                    if nq_mss_level is None or nq_intervening_high >= nq_mss_level:
                        if ltf_nq_high_arr[nq_idx] > nq_intervening_high and not (
                            (nq_mss_level is not None and ltf_nq_close_arr[nq_idx] > nq_mss_level)
                            or nq_ifvg_level is not None
                        ):
                            disrespected = True
                            break

                if has_es_bar and es_mss_level is not None and ltf_es_close_arr[es_idx] > es_mss_level:
                    confirm_time, confirm_ns, confirm_asset, confirm_level, confirm_kind = (
                        ltf_es_times_list[es_idx], t_ns, "ES", es_mss_level, "MSS"
                    )
                    break
                if has_nq_bar and nq_mss_level is not None and ltf_nq_close_arr[nq_idx] > nq_mss_level:
                    confirm_time, confirm_ns, confirm_asset, confirm_level, confirm_kind = (
                        ltf_nq_times_list[nq_idx], t_ns, "NQ", nq_mss_level, "MSS"
                    )
                    break
                if es_ifvg_level is not None:
                    confirm_time, confirm_ns, confirm_asset, confirm_level, confirm_kind = (
                        ltf_es_times_list[es_idx], t_ns, "ES", es_ifvg_level, "IFVG_MSS"
                    )
                    break
                if nq_ifvg_level is not None:
                    confirm_time, confirm_ns, confirm_asset, confirm_level, confirm_kind = (
                        ltf_nq_times_list[nq_idx], t_ns, "NQ", nq_ifvg_level, "IFVG_MSS"
                    )
                    break

            if not disrespected and confirm_time is not None:
                dedup_key = ('Bullish_SMT', prev_es_ns, prev_nq_ns)
                if dedup_key in seen_confirmations:
                    break
                seen_confirmations.add(dedup_key)

                es_conf_idx = bisect_left(ltf_es_times_ns, confirm_ns) if has_ltf_es_close else -1
                es_ref_high = (
                    float(confirm_level)
                    if confirm_asset == "ES"
                    else (
                        float(ltf_es_high_arr[es_conf_idx])
                        if (has_ltf_es_ohlc and es_conf_idx < n_ltf_es and ltf_es_times_ns[es_conf_idx] == confirm_ns)
                        else float(confirm_level)
                    )
                )
                nq_conf_idx = bisect_left(ltf_nq_times_ns, confirm_ns) if has_ltf_nq_close else -1
                nq_ref_high = (
                    float(confirm_level)
                    if confirm_asset == "NQ"
                    else (
                        float(ltf_nq_high_arr[nq_conf_idx])
                        if (has_ltf_nq_ohlc and nq_conf_idx < n_ltf_nq and ltf_nq_times_ns[nq_conf_idx] == confirm_ns)
                        else float(confirm_level)
                    )
                )

                es_intervening_valid = es_intervening_high is not None and es_intervening_high > es_ref_high
                nq_intervening_valid = nq_intervening_high is not None and nq_intervening_high > nq_ref_high

                es_target = (
                    es_intervening_high
                    if es_intervening_valid
                    else _find_prior_target_beyond_confirm_ns(
                        es_high_prices, es_high_ns, c_es_ns, es_ref_high, is_bearish=False, htf_min=htf_min
                    )
                )
                nq_target = (
                    nq_intervening_high
                    if nq_intervening_valid
                    else _find_prior_target_beyond_confirm_ns(
                        nq_high_prices, nq_high_ns, c_nq_ns, nq_ref_high, is_bearish=False, htf_min=htf_min
                    )
                )

                conf_close_time = confirm_time + ltf_close_offset
                conf_close_ns = confirm_ns + ltf_close_offset_ns
                es_is_holder = bool(nq_swept_ssl) or not res_nq_ohlc
                nq_is_holder = bool(es_swept_ssl) or not res_es_ohlc
                es_held_lvl = prev_es_price if nq_swept_ssl else float(c_es_price)
                nq_held_lvl = prev_nq_price if es_swept_ssl else float(c_nq_price)

                if es_intervening_valid or nq_intervening_valid:
                    check_es_t = True
                    check_nq_t = True
                elif htf_min >= 60:
                    check_es_t = bool(es_swept_ssl or not res_nq_ohlc)
                    check_nq_t = bool(nq_swept_ssl or not res_es_ohlc)
                elif htf_min == 1:
                    check_es_t = (confirm_asset == "ES")
                    check_nq_t = (confirm_asset == "NQ")
                else:
                    check_es_t = True
                    check_nq_t = True

                status, resolved_time = _evaluate_post_confirmation_lifecycle_ns(
                    is_bearish=False,
                    confirm_ns=conf_close_ns,
                    es_was_holder=es_is_holder,
                    nq_was_holder=nq_is_holder,
                    es_held_level=es_held_lvl,
                    nq_held_level=nq_held_lvl,
                    es_target_level=es_target,
                    nq_target_level=nq_target,
                    check_es_target=check_es_t,
                    check_nq_target=check_nq_t,
                    ltf_es_times_ns=res_es_times_ns,
                    ltf_es_times_list=res_es_times_list,
                    ltf_es_high_arr=res_es_high,
                    ltf_es_low_arr=res_es_low,
                    has_ltf_es_ohlc=res_es_ohlc,
                    ltf_nq_times_ns=res_nq_times_ns,
                    ltf_nq_times_list=res_nq_times_list,
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
                        f"ES swept SSL {prev_es_price:.2f} (from {prev_es['time'].strftime('%m-%d %H:%M')}) "
                        f"to {c_es_price:.2f}, while NQ held above {prev_nq_price:.2f} (from {prev_nq['time'].strftime('%m-%d %H:%M')}) "
                        f"--> {ltf_label} {confirm_kind} on {confirm_asset} at [{conf_close_time.strftime('%m-%d %H:%M')}] (> {confirm_level:.2f}) {status_tag}"
                    )
                else:
                    desc = (
                        f"[{creation_time.strftime('%m-%d %H:%M')}] Bullish SMT ({timeframe}): "
                        f"NQ swept SSL {prev_nq_price:.2f} (from {prev_nq['time'].strftime('%m-%d %H:%M')}) "
                        f"to {c_nq_price:.2f}, while ES held above {prev_es_price:.2f} (from {prev_es['time'].strftime('%m-%d %H:%M')}) "
                        f"--> {ltf_label} {confirm_kind} on {confirm_asset} at [{conf_close_time.strftime('%m-%d %H:%M')}] (> {confirm_level:.2f}) {status_tag}"
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
                    'swept_level': prev_es_price if es_swept_ssl else prev_nq_price,
                    'sweep_extreme': float(c_es_price) if es_swept_ssl else float(c_nq_price),
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
    events: List[Dict[str, Any]] = []
    if df_es_1m.empty or df_nq_1m.empty:
        return events

    es_times = df_es_1m.index
    es_times_ns = _index_to_ns(es_times).tolist()
    es_times_list = es_times.tolist()
    nq_times_ns = _index_to_ns(df_nq_1m.index).tolist()
    nq_times_list = df_nq_1m.index.tolist()
    n_es = len(es_times_ns)
    n_nq = len(nq_times_ns)

    es_high_arr = df_es_1m["high"].to_numpy(dtype=float, copy=False)
    es_low_arr = df_es_1m["low"].to_numpy(dtype=float, copy=False)
    es_close_arr = df_es_1m["close"].to_numpy(dtype=float, copy=False)
    nq_high_arr = df_nq_1m["high"].to_numpy(dtype=float, copy=False)
    nq_low_arr = df_nq_1m["low"].to_numpy(dtype=float, copy=False)
    nq_close_arr = df_nq_1m["close"].to_numpy(dtype=float, copy=False)

    es_highs = [s for s in es_swings_1m if s["type"] == "SwingHigh"]
    es_lows = [s for s in es_swings_1m if s["type"] == "SwingLow"]
    nq_highs = [s for s in nq_swings_1m if s["type"] == "SwingHigh"]
    nq_lows = [s for s in nq_swings_1m if s["type"] == "SwingLow"]

    es_high_ns = [_ts_to_ns(s["time"]) for s in es_highs]
    es_high_prices = [float(s["price"]) for s in es_highs]
    es_low_ns = [_ts_to_ns(s["time"]) for s in es_lows]
    es_low_prices = [float(s["price"]) for s in es_lows]
    nq_high_ns = [_ts_to_ns(s["time"]) for s in nq_highs]
    nq_high_prices = [float(s["price"]) for s in nq_highs]
    nq_low_ns = [_ts_to_ns(s["time"]) for s in nq_lows]
    nq_low_prices = [float(s["price"]) for s in nq_lows]

    es_fvgs_1m = track_fvgs(df_es_1m)
    nq_fvgs_1m = track_fvgs(df_nq_1m)

    es_bisi_cre_ns = [_ts_to_ns(f.creation_time) for f in es_fvgs_1m if f.fvg_type == "BISI"]
    es_bisi_bottoms = [float(f.bottom) for f in es_fvgs_1m if f.fvg_type == "BISI"]
    nq_bisi_cre_ns = [_ts_to_ns(f.creation_time) for f in nq_fvgs_1m if f.fvg_type == "BISI"]
    nq_bisi_bottoms = [float(f.bottom) for f in nq_fvgs_1m if f.fvg_type == "BISI"]

    def _ny_trade_date(ts: pd.Timestamp):
        return (ts + pd.Timedelta(hours=6)).date()

    es_by_key: Dict[Tuple[Any, str, str], Tuple[float, pd.Timestamp, int]] = {}
    for mp in es_macro_pools:
        p_type, sess, price, ts = extract_macro_dict_fields(mp)
        if sess in ("Asia", "London"):
            es_by_key[(_ny_trade_date(ts), sess.upper(), p_type)] = (price, ts, _ts_to_ns(ts))

    nq_by_key: Dict[Tuple[Any, str, str], Tuple[float, pd.Timestamp, int]] = {}
    for mp in nq_macro_pools:
        p_type, sess, price, ts = extract_macro_dict_fields(mp)
        if sess in ("Asia", "London"):
            nq_by_key[(_ny_trade_date(ts), sess.upper(), p_type)] = (price, ts, _ts_to_ns(ts))

    bisect_left = bisect.bisect_left
    bisect_right = bisect.bisect_right

    for key, (es_lvl, es_form_ts, es_form_ns) in es_by_key.items():
        if key not in nq_by_key:
            continue
        nq_lvl, nq_form_ts, nq_form_ns = nq_by_key[key]
        trade_date, sess_label, p_type = key

        ny_open = pd.Timestamp(trade_date).tz_localize(es_times.tz) + pd.Timedelta(hours=9, minutes=30)
        ny_open_ns = _ts_to_ns(ny_open)
        ny_cutoff_ns = ny_open_ns + 60 * _MIN_NS

        es_start = bisect_left(es_times_ns, ny_open_ns)
        es_end = bisect_left(es_times_ns, ny_cutoff_ns)
        nq_start = bisect_left(nq_times_ns, ny_open_ns)
        if es_start >= es_end or nq_start >= n_nq:
            continue

        es_pre_idx = bisect_right(es_times_ns, es_form_ns)
        nq_pre_idx = bisect_right(nq_times_ns, nq_form_ns)
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

        ny_open_plus_4m_ns = ny_open_ns + 4 * _MIN_NS
        ny_open_plus_15m_ns = ny_open_ns + 15 * _MIN_NS

        for k in range(es_start, es_end):
            t_ns = es_times_ns[k]
            nq_k = bisect_left(nq_times_ns, t_ns)
            if nq_k >= n_nq or nq_times_ns[nq_k] != t_ns:
                continue
            t = es_times_list[k]

            if p_type == "SSL":
                es_min_so_far = float(es_low_arr[es_start:k + 1].min())
                nq_min_so_far = float(nq_low_arr[nq_start:nq_k + 1].min())
                es_swept = es_min_so_far < es_lvl and nq_min_so_far > nq_lvl
                nq_swept = nq_min_so_far < nq_lvl and es_min_so_far > es_lvl
                if not (es_swept or nq_swept):
                    continue

                confirm_time, confirm_ns, confirm_asset, confirm_level, disrespected = None, 0, None, None, False
                for m in range(1, 16):
                    tm_ns = t_ns + m * _MIN_NS
                    e_i = bisect_left(es_times_ns, tm_ns)
                    n_i = bisect_left(nq_times_ns, tm_ns)
                    if e_i >= n_es or es_times_ns[e_i] != tm_ns or n_i >= n_nq or nq_times_ns[n_i] != tm_ns:
                        continue
                    if es_swept and nq_low_arr[n_i] <= nq_lvl:
                        disrespected = True
                        break
                    if nq_swept and es_low_arr[e_i] <= es_lvl:
                        disrespected = True
                        break

                    eh_idx = bisect_left(es_high_ns, tm_ns) - 1
                    nh_idx = bisect_left(nq_high_ns, tm_ns) - 1
                    es_mss = es_high_prices[eh_idx] if eh_idx >= 0 and es_high_ns[eh_idx] >= ny_open_ns else None
                    nq_mss = nq_high_prices[nh_idx] if nh_idx >= 0 and nq_high_ns[nh_idx] >= ny_open_ns else None

                    if es_swept and es_mss is not None and es_close_arr[e_i] > es_mss:
                        confirm_time, confirm_ns, confirm_asset, confirm_level = es_times_list[e_i], tm_ns, "ES", es_mss
                        break
                    if nq_swept and nq_mss is not None and nq_close_arr[n_i] > nq_mss:
                        confirm_time, confirm_ns, confirm_asset, confirm_level = nq_times_list[n_i], tm_ns, "NQ", nq_mss
                        break

                if not disrespected and confirm_time is not None:
                    es_interv_high = _find_intervening_extreme_ns(es_high_prices, es_high_ns, es_form_ns, t_ns, find_min=False)
                    nq_interv_high = _find_intervening_extreme_ns(nq_high_prices, nq_high_ns, nq_form_ns, t_ns, find_min=False)
                    es_res_target = es_interv_high if (es_interv_high is not None and (confirm_asset != "ES" or es_interv_high > confirm_level)) else es_opp_target
                    nq_res_target = nq_interv_high if (nq_interv_high is not None and (confirm_asset != "NQ" or nq_interv_high > confirm_level)) else nq_opp_target

                    status, resolved_time = _evaluate_post_confirmation_lifecycle_ns(
                        is_bearish=False,
                        confirm_ns=confirm_ns,
                        es_was_holder=bool(nq_swept),
                        nq_was_holder=bool(es_swept),
                        es_held_level=es_lvl,
                        nq_held_level=nq_lvl,
                        es_target_level=es_res_target,
                        nq_target_level=nq_res_target,
                        check_es_target=True,
                        check_nq_target=True,
                        ltf_es_times_ns=es_times_ns,
                        ltf_es_times_list=es_times_list,
                        ltf_es_high_arr=es_high_arr,
                        ltf_es_low_arr=es_low_arr,
                        has_ltf_es_ohlc=True,
                        ltf_nq_times_ns=nq_times_ns,
                        ltf_nq_times_list=nq_times_list,
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
                        "swept_level": float(es_lvl) if es_swept else float(nq_lvl),
                        "sweep_extreme": float(es_min_so_far) if es_swept else float(nq_min_so_far),
                        "es_invalidation_level": float(es_lvl),
                        "nq_invalidation_level": float(nq_lvl),
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

                confirm_time, confirm_ns, confirm_asset, confirm_level, confirm_kind, disrespected = None, 0, None, None, "MSS", False
                for m in range(1, 16):
                    tm_ns = t_ns + m * _MIN_NS
                    e_i = bisect_left(es_times_ns, tm_ns)
                    n_i = bisect_left(nq_times_ns, tm_ns)
                    if e_i >= n_es or es_times_ns[e_i] != tm_ns or n_i >= n_nq or nq_times_ns[n_i] != tm_ns:
                        continue
                    if es_swept and nq_high_arr[n_i] >= nq_lvl:
                        disrespected = True
                        break
                    if nq_swept and es_high_arr[e_i] >= es_lvl:
                        disrespected = True
                        break

                    el_idx = bisect_left(es_low_ns, tm_ns) - 1
                    nl_idx = bisect_left(nq_low_ns, tm_ns) - 1
                    es_mss = es_low_prices[el_idx] if el_idx >= 0 and es_low_ns[el_idx] >= ny_open_ns else None
                    nq_mss = nq_low_prices[nl_idx] if nl_idx >= 0 and nq_low_ns[nl_idx] >= ny_open_ns else None

                    if es_swept and es_mss is not None and es_close_arr[e_i] < es_mss:
                        confirm_time, confirm_ns, confirm_asset, confirm_level, confirm_kind = es_times_list[e_i], tm_ns, "ES", es_mss, "MSS"
                        break
                    if nq_swept and nq_mss is not None and nq_close_arr[n_i] < nq_mss:
                        confirm_time, confirm_ns, confirm_asset, confirm_level, confirm_kind = nq_times_list[n_i], tm_ns, "NQ", nq_mss, "MSS"
                        break

                    if t_ns <= ny_open_plus_15m_ns:
                        hi_fvg_ns = tm_ns - _MIN_NS
                        if es_swept and es_bisi_cre_ns:
                            lo_f = bisect_left(es_bisi_cre_ns, ny_open_plus_4m_ns)
                            hi_f = bisect_right(es_bisi_cre_ns, hi_fvg_ns)
                            if lo_f < hi_f:
                                c_es = es_close_arr[e_i]
                                inv_lvl = float("inf")
                                found_inv = False
                                for f_i in range(lo_f, hi_f):
                                    b = es_bisi_bottoms[f_i]
                                    if c_es < b:
                                        found_inv = True
                                        if b < inv_lvl:
                                            inv_lvl = b
                                if found_inv:
                                    confirm_time, confirm_ns, confirm_asset, confirm_level, confirm_kind = (
                                        es_times_list[e_i], tm_ns, "ES", inv_lvl, "IFVG_MSS"
                                    )
                                    break
                        if nq_swept and nq_bisi_cre_ns:
                            lo_f = bisect_left(nq_bisi_cre_ns, ny_open_plus_4m_ns)
                            hi_f = bisect_right(nq_bisi_cre_ns, hi_fvg_ns)
                            if lo_f < hi_f:
                                c_nq = nq_close_arr[n_i]
                                inv_lvl = float("inf")
                                found_inv = False
                                for f_i in range(lo_f, hi_f):
                                    b = nq_bisi_bottoms[f_i]
                                    if c_nq < b:
                                        found_inv = True
                                        if b < inv_lvl:
                                            inv_lvl = b
                                if found_inv:
                                    confirm_time, confirm_ns, confirm_asset, confirm_level, confirm_kind = (
                                        nq_times_list[n_i], tm_ns, "NQ", inv_lvl, "IFVG_MSS"
                                    )
                                    break

                if not disrespected and confirm_time is not None:
                    es_interv_low = _find_intervening_extreme_ns(es_low_prices, es_low_ns, es_form_ns, t_ns, find_min=True)
                    nq_interv_low = _find_intervening_extreme_ns(nq_low_prices, nq_low_ns, nq_form_ns, t_ns, find_min=True)
                    es_res_target = es_interv_low if (es_interv_low is not None and (confirm_asset != "ES" or es_interv_low < confirm_level)) else es_opp_target
                    nq_res_target = nq_interv_low if (nq_interv_low is not None and (confirm_asset != "NQ" or nq_interv_low < confirm_level)) else nq_opp_target

                    status, resolved_time = _evaluate_post_confirmation_lifecycle_ns(
                        is_bearish=True,
                        confirm_ns=confirm_ns,
                        es_was_holder=bool(nq_swept),
                        nq_was_holder=bool(es_swept),
                        es_held_level=es_lvl,
                        nq_held_level=nq_lvl,
                        es_target_level=es_res_target,
                        nq_target_level=nq_res_target,
                        check_es_target=True,
                        check_nq_target=True,
                        ltf_es_times_ns=es_times_ns,
                        ltf_es_times_list=es_times_list,
                        ltf_es_high_arr=es_high_arr,
                        ltf_es_low_arr=es_low_arr,
                        has_ltf_es_ohlc=True,
                        ltf_nq_times_ns=nq_times_ns,
                        ltf_nq_times_list=nq_times_list,
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
                        f"{sweeper_str} --> 1m {confirm_kind} on {confirm_asset} at [{confirm_time.strftime('%m-%d %H:%M')}] (< {confirm_level:.2f}) {status_tag}"
                    )
                    events.append({
                        "type": "Bearish_SMT",
                        "time": t,
                        "confirmation_time": confirm_time,
                        "confirmation_close_time": confirm_time,
                        "status": status,
                        "resolved_time": resolved_time,
                        "resolved_close_time": resolved_time,
                        "swept_level": float(es_lvl) if es_swept else float(nq_lvl),
                        "sweep_extreme": float(es_max_so_far) if es_swept else float(nq_max_so_far),
                        "es_invalidation_level": float(es_lvl),
                        "nq_invalidation_level": float(nq_lvl),
                        "timeframe": "1min",
                        "confirmation_timeframe": "1m",
                        "is_intraday_post_open": True,
                        "description": desc,
                    })
                    break

    return events