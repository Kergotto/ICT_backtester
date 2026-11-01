import pandas as pd
from datetime import timedelta
from domain.smt import (
    detect_smt_divergence,
    get_smt_status_at,
    is_smt_active_at,
    TIMEFRAME_HIERARCHY,
    SMT_PENDING,
    SMT_CONFIRMED,
    SMT_RESPECTED,
    SMT_INVALIDATED,
)


def test_bearish_smt():
    base_time = pd.Timestamp("2026-09-28 10:00:00")

    es_swings = [
        {'type': 'SwingHigh', 'time': base_time, 'price': 4000.0},
        {'type': 'SwingLow', 'time': base_time + timedelta(minutes=5), 'price': 3980.0},
        {'type': 'SwingHigh', 'time': base_time + timedelta(minutes=10), 'price': 4010.0}
    ]

    nq_swings = [
        {'type': 'SwingHigh', 'time': base_time, 'price': 15000.0},
        {'type': 'SwingHigh', 'time': base_time + timedelta(minutes=11), 'price': 14950.0}
    ]

    df_es = pd.DataFrame({'close': [3970.0]}, index=[base_time + timedelta(minutes=15)])
    df_nq = pd.DataFrame()

    smts = detect_smt_divergence(es_swings, nq_swings, df_es, df_nq, time_window_minutes=15)

    assert len(smts) == 1
    assert smts[0]['type'] == 'Bearish_SMT'
    assert smts[0]['status'] == SMT_CONFIRMED


def test_bullish_smt():
    base_time = pd.Timestamp("2026-09-28 10:00:00")

    es_swings = [
        {'type': 'SwingLow', 'time': base_time, 'price': 4000.0},
        {'type': 'SwingHigh', 'time': base_time + timedelta(minutes=5), 'price': 4020.0},
        {'type': 'SwingLow', 'time': base_time + timedelta(minutes=10), 'price': 3990.0}
    ]

    nq_swings = [
        {'type': 'SwingLow', 'time': base_time, 'price': 15000.0},
        {'type': 'SwingLow', 'time': base_time + timedelta(minutes=11), 'price': 15050.0}
    ]

    df_es = pd.DataFrame({'close': [4030.0]}, index=[base_time + timedelta(minutes=15)])
    df_nq = pd.DataFrame()

    smts = detect_smt_divergence(es_swings, nq_swings, df_es, df_nq, time_window_minutes=15)

    assert len(smts) == 1
    assert smts[0]['type'] == 'Bullish_SMT'
    assert smts[0]['status'] == SMT_CONFIRMED


def test_mtf_smt_hierarchy_and_15m_alignment():
    assert TIMEFRAME_HIERARCHY["4h"]["ltf_label"] == "1h"
    assert TIMEFRAME_HIERARCHY["1h"]["ltf_label"] == "15m"
    assert TIMEFRAME_HIERARCHY["15min"]["ltf_label"] == "5m"
    assert TIMEFRAME_HIERARCHY["5min"]["ltf_label"] == "1m"
    assert TIMEFRAME_HIERARCHY["1min"]["ltf_label"] == "1m"

    base_time = pd.Timestamp("2026-09-28 09:00:00")

    es_15m_swings = [
        {'type': 'SwingLow', 'time': base_time, 'price': 5000.0},
        {'type': 'SwingHigh', 'time': base_time + timedelta(minutes=30), 'price': 5020.0},
        {'type': 'SwingLow', 'time': base_time + timedelta(minutes=60), 'price': 4990.0},
    ]

    nq_15m_swings = [
        {'type': 'SwingLow', 'time': base_time, 'price': 20000.0},
        {'type': 'SwingLow', 'time': base_time + timedelta(minutes=75), 'price': 20025.0},
    ]

    df_es_15m = pd.DataFrame(
        {'close': [5025.0]},
        index=[base_time + timedelta(minutes=75)]
    )
    df_nq_15m = pd.DataFrame()

    smts = detect_smt_divergence(
        es_15m_swings, nq_15m_swings, df_es_15m, df_nq_15m,
        time_window_minutes=60, timeframe="15min"
    )

    assert len(smts) == 1
    assert smts[0]['type'] == 'Bullish_SMT'
    assert smts[0]['confirmation_timeframe'] == '5m'


def test_smt_lifecycle_confirmed_to_respected_and_invalidated():
    """
    Verifies that a Bearish SMT transitions:
    PENDING -> CONFIRMED (at MSS) -> RESPECTED (when target low is swept)
    or INVALIDATED (when protective SMT high is breached).
    """
    base_time = pd.Timestamp("2026-09-28 10:00:00")

    # Prior target low at 09:55 (3960.0), High 1 at 10:00 (4000.0),
    # MSS Low at 10:05 (3980.0), High 2 (Sweep) at 10:10 (4010.0)
    es_swings = [
        {'type': 'SwingLow', 'time': base_time - timedelta(minutes=5), 'price': 3960.0},
        {'type': 'SwingHigh', 'time': base_time, 'price': 4000.0},
        {'type': 'SwingLow', 'time': base_time + timedelta(minutes=5), 'price': 3980.0},
        {'type': 'SwingHigh', 'time': base_time + timedelta(minutes=10), 'price': 4010.0},
    ]
    nq_swings = [
        {'type': 'SwingHigh', 'time': base_time, 'price': 15000.0},
        {'type': 'SwingHigh', 'time': base_time + timedelta(minutes=10), 'price': 14950.0},
    ]

    # Scenario 1: Confirmed at 10:15 (close=3975 < 3980), still active at 10:16, sweeps target 3960 at 10:17
    times = [
        base_time + timedelta(minutes=15),
        base_time + timedelta(minutes=16),
        base_time + timedelta(minutes=17),
    ]
    df_es_respected = pd.DataFrame(
        {
            'open':  [3982.0, 3975.0, 3968.0],
            'high':  [3985.0, 3978.0, 3970.0],
            'low':   [3972.0, 3965.0, 3955.0],  # 3955.0 sweeps target 3960.0 at 10:17
            'close': [3975.0, 3968.0, 3958.0],
        },
        index=times,
    )

    smts_resp = detect_smt_divergence(es_swings, nq_swings, df_es_respected, pd.DataFrame(), time_window_minutes=15)
    assert len(smts_resp) == 1
    smt = smts_resp[0]
    assert smt['status'] == SMT_RESPECTED
    assert smt['resolved_time'] == base_time + timedelta(minutes=17)

    # Check point-in-time Zero-Lookahead status helper
    assert get_smt_status_at(smt, base_time + timedelta(minutes=12)) == SMT_PENDING
    assert get_smt_status_at(smt, base_time + timedelta(minutes=15)) == SMT_CONFIRMED
    assert is_smt_active_at(smt, base_time + timedelta(minutes=16)) is True
    assert get_smt_status_at(smt, base_time + timedelta(minutes=17)) == SMT_RESPECTED
    assert is_smt_active_at(smt, base_time + timedelta(minutes=17)) is False

    # Scenario 2: Confirmed at 10:15, but reverses above protective high (4010.0) at 10:16 -> INVALIDATED
    df_es_invalid = pd.DataFrame(
        {
            'open':  [3982.0, 3975.0],
            'high':  [3985.0, 4012.0],  # Breaches 4010.0 protective high at 10:16
            'low':   [3972.0, 3974.0],
            'close': [3975.0, 4011.0],
        },
        index=times[:2],
    )
    smts_inv = detect_smt_divergence(es_swings, nq_swings, df_es_invalid, pd.DataFrame(), time_window_minutes=15)
    assert len(smts_inv) == 1
    assert smts_inv[0]['status'] == SMT_INVALIDATED
    assert smts_inv[0]['resolved_time'] == base_time + timedelta(minutes=16)