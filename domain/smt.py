from typing import List, Dict
import pandas as pd
import bisect


def get_closest_swing_fast(target_time, target_swings, target_times, max_diff_minutes=15):
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


def filter_significant_swings(swings, df, is_high, window=10):
    """
    Filters out micro-wiggles. A swing must be the highest/lowest point
    within a 20-minute window (10 before, 10 after) to be considered an ICT structural pool.
    """
    # Safety fallback for unit tests that pass mock DataFrames
    if df.empty or 'high' not in df.columns or 'low' not in df.columns:
        return swings

    sig_swings = []
    if is_high:
        # Calculate the rolling local maximums
        rolling_series = df['high'].rolling(window=window * 2 + 1, center=True, min_periods=1).max()
        for s in swings:
            if s['time'] in rolling_series.index and s['price'] >= rolling_series.loc[s['time']]:
                sig_swings.append(s)
    else:
        # Calculate the rolling local minimums
        rolling_series = df['low'].rolling(window=window * 2 + 1, center=True, min_periods=1).min()
        for s in swings:
            if s['time'] in rolling_series.index and s['price'] <= rolling_series.loc[s['time']]:
                sig_swings.append(s)

    return sig_swings


def detect_smt_divergence(
        es_swings: List[Dict],
        nq_swings: List[Dict],
        df_es: pd.DataFrame,
        df_nq: pd.DataFrame,
        time_window_minutes: int = 15
) -> List[Dict]:
    smt_events = []

    # 1. Separate all raw swings (Used later for immediate MSS confirmation)
    raw_es_highs = [s for s in es_swings if s['type'] == 'SwingHigh']
    raw_es_lows = [s for s in es_swings if s['type'] == 'SwingLow']

    # 2. FILTER OUT NOISE! Only keep true Macro Swings for the SMT sweep comparisons
    es_highs = filter_significant_swings(raw_es_highs, df_es, is_high=True, window=10)
    es_lows = filter_significant_swings(raw_es_lows, df_es, is_high=False, window=10)

    nq_highs = [s for s in nq_swings if s['type'] == 'SwingHigh']
    nq_lows = [s for s in nq_swings if s['type'] == 'SwingLow']

    nq_high_times = [s['time'] for s in nq_highs]
    nq_low_times = [s['time'] for s in nq_lows]

    # --- 1. BEARISH SMT (Sweeping old BSL Highs) ---
    for i, curr_es in enumerate(es_highs):
        curr_nq = get_closest_swing_fast(curr_es['time'], nq_highs, nq_high_times, max_diff_minutes=5)
        if not curr_nq: continue

        start_t = curr_es['time'] + pd.Timedelta(minutes=1)
        end_t = curr_es['time'] + pd.Timedelta(minutes=60)
        post_sweep = df_es.loc[start_t:end_t]
        if post_sweep.empty: continue

        # Check against the last 5 MAJOR structural pools
        past_es_highs = es_highs[max(0, i - 5):i]

        for prev_es in reversed(past_es_highs):
            # Enforce distance: Pool must be at least 5 minutes old to prevent chop
            if (curr_es['time'] - prev_es['time']).total_seconds() < 300:
                continue

            prev_nq = get_closest_swing_fast(prev_es['time'], nq_highs, nq_high_times, max_diff_minutes=15)
            if not prev_nq: continue

            es_swept_bsl = curr_es['price'] > prev_es['price']
            nq_swept_bsl = curr_nq['price'] > prev_nq['price']

            if es_swept_bsl != nq_swept_bsl:
                # Use raw lows here because MSS might occur on a minor intervening low
                intervening_lows = [s for s in raw_es_lows if prev_es['time'] < s['time'] < curr_es['time']]
                if not intervening_lows: continue

                mss_level = min(intervening_lows, key=lambda x: x['price'])['price']
                mss_candles = post_sweep[post_sweep['close'] < mss_level]

                if not mss_candles.empty:
                    creation_time = max(curr_es['time'], curr_nq['time'])
                    confirmation_time = mss_candles.index[0]

                    leader = "ES" if es_swept_bsl else "NQ"
                    failed = "NQ" if es_swept_bsl else "ES"
                    swept_price = prev_es['price'] if es_swept_bsl else prev_nq['price']
                    failed_price = prev_nq['price'] if es_swept_bsl else prev_es['price']

                    smt_events.append({
                        'type': 'Bearish_SMT',
                        'time': creation_time,
                        'confirmation_time': confirmation_time,
                        'description': (f"[{creation_time.strftime('%H:%M')}] Bearish SMT: "
                                        f"{leader} swept BSL at {swept_price:.2f} (from {prev_es['time'].strftime('%m-%d %H:%M')}), "
                                        f"but {failed} failed to break {failed_price:.2f}  -->  "
                                        f"MSS Confirmed at [{confirmation_time.strftime('%H:%M')}]")
                    })
                    break

                    # --- 2. BULLISH SMT (Sweeping old SSL Lows) ---
    for i, curr_es in enumerate(es_lows):
        curr_nq = get_closest_swing_fast(curr_es['time'], nq_lows, nq_low_times, max_diff_minutes=5)
        if not curr_nq: continue

        start_t = curr_es['time'] + pd.Timedelta(minutes=1)
        end_t = curr_es['time'] + pd.Timedelta(minutes=60)
        post_sweep = df_es.loc[start_t:end_t]
        if post_sweep.empty: continue

        past_es_lows = es_lows[max(0, i - 5):i]

        for prev_es in reversed(past_es_lows):
            if (curr_es['time'] - prev_es['time']).total_seconds() < 300:
                continue

            prev_nq = get_closest_swing_fast(prev_es['time'], nq_lows, nq_low_times, max_diff_minutes=15)
            if not prev_nq: continue

            es_swept_ssl = curr_es['price'] < prev_es['price']
            nq_swept_ssl = curr_nq['price'] < prev_nq['price']

            if es_swept_ssl != nq_swept_ssl:
                intervening_highs = [s for s in raw_es_highs if prev_es['time'] < s['time'] < curr_es['time']]
                if not intervening_highs: continue

                mss_level = max(intervening_highs, key=lambda x: x['price'])['price']
                mss_candles = post_sweep[post_sweep['close'] > mss_level]

                if not mss_candles.empty:
                    creation_time = max(curr_es['time'], curr_nq['time'])
                    confirmation_time = mss_candles.index[0]

                    leader = "ES" if es_swept_ssl else "NQ"
                    failed = "NQ" if es_swept_ssl else "ES"
                    swept_price = prev_es['price'] if es_swept_ssl else prev_nq['price']
                    failed_price = prev_nq['price'] if es_swept_ssl else prev_es['price']

                    smt_events.append({
                        'type': 'Bullish_SMT',
                        'time': creation_time,
                        'confirmation_time': confirmation_time,
                        'description': (f"[{creation_time.strftime('%H:%M')}] Bullish SMT: "
                                        f"{leader} swept SSL at {swept_price:.2f} (from {prev_es['time'].strftime('%m-%d %H:%M')}), "
                                        f"but {failed} failed to break {failed_price:.2f}  -->  "
                                        f"MSS Confirmed at [{confirmation_time.strftime('%H:%M')}]")
                    })
                    break

    smt_events.sort(key=lambda x: x['time'])
    return smt_events