from typing import List, Dict
import numpy as np
import pandas as pd
import bisect


def get_closest_swing_fast(target_time, target_swings, target_times, max_diff_minutes=3):
    """
    Uses binary search to find the corresponding swing on the other index
    within a tight time window (default 3 minutes).
    """
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


def detect_smt_divergence(
    es_swings: List[Dict],
    nq_swings: List[Dict],
    df_es: pd.DataFrame,
    df_nq: pd.DataFrame,
    time_window_minutes: int = 3
) -> List[Dict]:
    """
    Detects 1-minute SMT Divergence between ES and NQ:
    1. Identifies synchronized prior structural swings (BSL/SSL) on both indices.
    2. Strictly invalidates the setup if the holding index sweeps OR makes an Equal High/Low
       (>= for BSL, <= for SSL) at any point prior to or during MSS confirmation.
    3. Dynamically tracks the most recent 1-minute opposing swing and confirms MSS on the
       exact candle whose body closes beyond that swing level on either index.
    """
    smt_events = []

    if df_es.empty and df_nq.empty:
        return smt_events

    es_highs = [s for s in es_swings if s['type'] == 'SwingHigh']
    es_lows = [s for s in es_swings if s['type'] == 'SwingLow']
    nq_highs = [s for s in nq_swings if s['type'] == 'SwingHigh']
    nq_lows = [s for s in nq_swings if s['type'] == 'SwingLow']

    has_es_close = not df_es.empty and 'close' in df_es.columns
    has_nq_close = not df_nq.empty and 'close' in df_nq.columns
    has_es_ohlc = has_es_close and 'high' in df_es.columns and 'low' in df_es.columns and len(df_es) > 1
    has_nq_ohlc = has_nq_close and 'high' in df_nq.columns and 'low' in df_nq.columns and len(df_nq) > 1

    es_times = df_es.index if has_es_close else pd.DatetimeIndex([])
    nq_times = df_nq.index if has_nq_close else pd.DatetimeIndex([])

    es_close_arr = df_es['close'].to_numpy() if has_es_close else np.array([])
    nq_close_arr = df_nq['close'].to_numpy() if has_nq_close else np.array([])

    es_high_arr = df_es['high'].to_numpy() if has_es_ohlc else np.array([])
    es_low_arr = df_es['low'].to_numpy() if has_es_ohlc else np.array([])
    nq_high_arr = df_nq['high'].to_numpy() if has_nq_ohlc else np.array([])
    nq_low_arr = df_nq['low'].to_numpy() if has_nq_ohlc else np.array([])

    nq_high_times = [s['time'] for s in nq_highs]
    nq_low_times = [s['time'] for s in nq_lows]
    es_high_times = [s['time'] for s in es_highs]
    es_low_times = [s['time'] for s in es_lows]

    sync_minutes = min(time_window_minutes, 3)
    seen_confirmations = set()

    # BEARISH SMT (BSL Sweep + 1m Bearish MSS)
    for i, curr_es in enumerate(es_highs):
        curr_nq = get_closest_swing_fast(curr_es['time'], nq_highs, nq_high_times, max_diff_minutes=sync_minutes)
        if not curr_nq:
            continue

        creation_time = max(curr_es['time'], curr_nq['time'])
        sweep_start_time = min(curr_es['time'], curr_nq['time'])

        past_es_highs = es_highs[max(0, i - 15):i]

        for prev_es in reversed(past_es_highs):
            if (sweep_start_time - prev_es['time']).total_seconds() < 180:
                continue

            prev_nq = get_closest_swing_fast(prev_es['time'], nq_highs, nq_high_times, max_diff_minutes=sync_minutes)
            if not prev_nq or prev_nq['time'] >= curr_nq['time']:
                continue

            es_max_in_win = curr_es['price']
            nq_max_in_win = curr_nq['price']

            # Pre-sweep & sweep window check: Equal High (>=) or Sweep invalidates the pool
            if has_es_ohlc:
                es_p_idx = es_times.searchsorted(prev_es['time'])
                es_win_start = es_times.searchsorted(sweep_start_time - pd.Timedelta(minutes=1))
                es_win_end = min(len(es_times), es_times.searchsorted(creation_time, side='right'))
                if es_win_start > es_p_idx + 1 and es_high_arr[es_p_idx + 1:es_win_start].max() >= prev_es['price']:
                    continue
                if es_win_end > es_win_start:
                    es_max_in_win = es_high_arr[es_win_start:es_win_end].max()

            if has_nq_ohlc:
                nq_p_idx = nq_times.searchsorted(prev_nq['time'])
                nq_win_start = nq_times.searchsorted(sweep_start_time - pd.Timedelta(minutes=1))
                nq_win_end = min(len(nq_times), nq_times.searchsorted(creation_time, side='right'))
                if nq_win_start > nq_p_idx + 1 and nq_high_arr[nq_p_idx + 1:nq_win_start].max() >= prev_nq['price']:
                    continue
                if nq_win_end > nq_win_start:
                    nq_max_in_win = nq_high_arr[nq_win_start:nq_win_end].max()

            # Holding index must stay STRICTLY below (<) its previous high (no Equal Highs)
            es_swept_bsl = (curr_es['price'] > prev_es['price']) and (nq_max_in_win < prev_nq['price'])
            nq_swept_bsl = (curr_nq['price'] > prev_nq['price']) and (es_max_in_win < prev_es['price'])

            if not (es_swept_bsl or nq_swept_bsl):
                continue

            # Step forward minute-by-minute (up to 20 mins) to check disrespect and dynamic 1m MSS
            confirm_time = None
            confirm_asset = None
            confirm_level = None
            disrespected = False

            for m in range(1, 21):
                t = creation_time + pd.Timedelta(minutes=m)

                es_idx = es_times.searchsorted(t) if has_es_close else -1
                has_es_bar = has_es_close and es_idx < len(es_times) and es_times[es_idx] == t

                nq_idx = nq_times.searchsorted(t) if has_nq_close else -1
                has_nq_bar = has_nq_close and nq_idx < len(nq_times) and nq_times[nq_idx] == t

                # 1. Check if the holding index disrespected its high (Equal High or Sweep)
                if es_swept_bsl and has_nq_ohlc and has_nq_bar:
                    if nq_high_arr[nq_idx] >= prev_nq['price']:
                        disrespected = True
                        break
                if nq_swept_bsl and has_es_ohlc and has_es_bar:
                    if es_high_arr[es_idx] >= prev_es['price']:
                        disrespected = True
                        break

                # Dynamically get the most recent 1m SwingLow formed before candle `t`
                es_l_idx = bisect.bisect_left(es_low_times, t) - 1
                es_mss_level = (
                    es_lows[es_l_idx]['price']
                    if (es_l_idx >= 0 and es_low_times[es_l_idx] > prev_es['time'])
                    else None
                )

                nq_l_idx = bisect.bisect_left(nq_low_times, t) - 1
                nq_mss_level = (
                    nq_lows[nq_l_idx]['price']
                    if (nq_l_idx >= 0 and nq_low_times[nq_l_idx] > prev_nq['time'])
                    else None
                )

                # Check for 1m candle body close strictly below the MSS swing low
                if has_es_bar and es_mss_level is not None and es_close_arr[es_idx] < es_mss_level:
                    confirm_time = t
                    confirm_asset = "ES"
                    confirm_level = es_mss_level
                    break

                if has_nq_bar and nq_mss_level is not None and nq_close_arr[nq_idx] < nq_mss_level:
                    confirm_time = t
                    confirm_asset = "NQ"
                    confirm_level = nq_mss_level
                    break

            if not disrespected and confirm_time is not None:
                dedup_key = ('Bearish_SMT', prev_es['time'], prev_nq['time'], confirm_time)
                if dedup_key in seen_confirmations:
                    break
                seen_confirmations.add(dedup_key)

                if es_swept_bsl:
                    desc = (
                        f"[{creation_time.strftime('%H:%M')}] Bearish SMT: "
                        f"ES swept BSL {prev_es['price']:.2f} (from {prev_es['time'].strftime('%H:%M')}) "
                        f"to {curr_es['price']:.2f}, while NQ held below {prev_nq['price']:.2f} (from {prev_nq['time'].strftime('%H:%M')}) "
                        f"--> 1m MSS on {confirm_asset} at [{confirm_time.strftime('%H:%M')}] (< {confirm_level:.2f})"
                    )
                else:
                    desc = (
                        f"[{creation_time.strftime('%H:%M')}] Bearish SMT: "
                        f"NQ swept BSL {prev_nq['price']:.2f} (from {prev_nq['time'].strftime('%H:%M')}) "
                        f"to {curr_nq['price']:.2f}, while ES held below {prev_es['price']:.2f} (from {prev_es['time'].strftime('%H:%M')}) "
                        f"--> 1m MSS on {confirm_asset} at [{confirm_time.strftime('%H:%M')}] (< {confirm_level:.2f})"
                    )

                smt_events.append({
                    'type': 'Bearish_SMT',
                    'time': creation_time,
                    'confirmation_time': confirm_time,
                    'description': desc
                })
                break

    # BULLISH SMT (SSL Sweep + 1m Bullish MSS)
    for i, curr_es in enumerate(es_lows):
        curr_nq = get_closest_swing_fast(curr_es['time'], nq_lows, nq_low_times, max_diff_minutes=sync_minutes)
        if not curr_nq:
            continue

        creation_time = max(curr_es['time'], curr_nq['time'])
        sweep_start_time = min(curr_es['time'], curr_nq['time'])

        past_es_lows = es_lows[max(0, i - 15):i]

        for prev_es in reversed(past_es_lows):
            if (sweep_start_time - prev_es['time']).total_seconds() < 180:
                continue

            prev_nq = get_closest_swing_fast(prev_es['time'], nq_lows, nq_low_times, max_diff_minutes=sync_minutes)
            if not prev_nq or prev_nq['time'] >= curr_nq['time']:
                continue

            es_min_in_win = curr_es['price']
            nq_min_in_win = curr_nq['price']

            # Pre-sweep & sweep window check: Equal Low (<=) or Sweep invalidates the pool
            if has_es_ohlc:
                es_p_idx = es_times.searchsorted(prev_es['time'])
                es_win_start = es_times.searchsorted(sweep_start_time - pd.Timedelta(minutes=1))
                es_win_end = min(len(es_times), es_times.searchsorted(creation_time, side='right'))
                if es_win_start > es_p_idx + 1 and es_low_arr[es_p_idx + 1:es_win_start].min() <= prev_es['price']:
                    continue
                if es_win_end > es_win_start:
                    es_min_in_win = es_low_arr[es_win_start:es_win_end].min()

            if has_nq_ohlc:
                nq_p_idx = nq_times.searchsorted(prev_nq['time'])
                nq_win_start = nq_times.searchsorted(sweep_start_time - pd.Timedelta(minutes=1))
                nq_win_end = min(len(nq_times), nq_times.searchsorted(creation_time, side='right'))
                if nq_win_start > nq_p_idx + 1 and nq_low_arr[nq_p_idx + 1:nq_win_start].min() <= prev_nq['price']:
                    continue
                if nq_win_end > nq_win_start:
                    nq_min_in_win = nq_low_arr[nq_win_start:nq_win_end].min()

            # Holding index must stay STRICTLY above (>) its previous low (no Equal Lows)
            es_swept_ssl = (curr_es['price'] < prev_es['price']) and (nq_min_in_win > prev_nq['price'])
            nq_swept_ssl = (curr_nq['price'] < prev_nq['price']) and (es_min_in_win > prev_es['price'])

            if not (es_swept_ssl or nq_swept_ssl):
                continue

            confirm_time = None
            confirm_asset = None
            confirm_level = None
            disrespected = False

            for m in range(1, 21):
                t = creation_time + pd.Timedelta(minutes=m)

                es_idx = es_times.searchsorted(t) if has_es_close else -1
                has_es_bar = has_es_close and es_idx < len(es_times) and es_times[es_idx] == t

                nq_idx = nq_times.searchsorted(t) if has_nq_close else -1
                has_nq_bar = has_nq_close and nq_idx < len(nq_times) and nq_times[nq_idx] == t

                # Check if the holding index disrespected its low (Equal Low or Sweep)
                if es_swept_ssl and has_nq_ohlc and has_nq_bar:
                    if nq_low_arr[nq_idx] <= prev_nq['price']:
                        disrespected = True
                        break
                if nq_swept_ssl and has_es_ohlc and has_es_bar:
                    if es_low_arr[es_idx] <= prev_es['price']:
                        disrespected = True
                        break

                # Dynamically get the most recent 1m SwingHigh formed before candle `t`
                es_h_idx = bisect.bisect_left(es_high_times, t) - 1
                es_mss_level = (
                    es_highs[es_h_idx]['price']
                    if (es_h_idx >= 0 and es_high_times[es_h_idx] > prev_es['time'])
                    else None
                )

                nq_h_idx = bisect.bisect_left(nq_high_times, t) - 1
                nq_mss_level = (
                    nq_highs[nq_h_idx]['price']
                    if (nq_h_idx >= 0 and nq_high_times[nq_h_idx] > prev_nq['time'])
                    else None
                )

                # Check for 1m candle body close strictly above the MSS swing high
                if has_es_bar and es_mss_level is not None and es_close_arr[es_idx] > es_mss_level:
                    confirm_time = t
                    confirm_asset = "ES"
                    confirm_level = es_mss_level
                    break

                if has_nq_bar and nq_mss_level is not None and nq_close_arr[nq_idx] > nq_mss_level:
                    confirm_time = t
                    confirm_asset = "NQ"
                    confirm_level = nq_mss_level
                    break

            if not disrespected and confirm_time is not None:
                dedup_key = ('Bullish_SMT', prev_es['time'], prev_nq['time'], confirm_time)
                if dedup_key in seen_confirmations:
                    break
                seen_confirmations.add(dedup_key)

                if es_swept_ssl:
                    desc = (
                        f"[{creation_time.strftime('%H:%M')}] Bullish SMT: "
                        f"ES swept SSL {prev_es['price']:.2f} (from {prev_es['time'].strftime('%H:%M')}) "
                        f"to {curr_es['price']:.2f}, while NQ held above {prev_nq['price']:.2f} (from {prev_nq['time'].strftime('%H:%M')}) "
                        f"--> 1m MSS on {confirm_asset} at [{confirm_time.strftime('%H:%M')}] (> {confirm_level:.2f})"
                    )
                else:
                    desc = (
                        f"[{creation_time.strftime('%H:%M')}] Bullish SMT: "
                        f"NQ swept SSL {prev_nq['price']:.2f} (from {prev_nq['time'].strftime('%H:%M')}) "
                        f"to {curr_nq['price']:.2f}, while ES held above {prev_es['price']:.2f} (from {prev_es['time'].strftime('%H:%M')}) "
                        f"--> 1m MSS on {confirm_asset} at [{confirm_time.strftime('%H:%M')}] (> {confirm_level:.2f})"
                    )

                smt_events.append({
                    'type': 'Bullish_SMT',
                    'time': creation_time,
                    'confirmation_time': confirm_time,
                    'description': desc
                })
                break

    smt_events.sort(key=lambda x: x['time'])
    return smt_events