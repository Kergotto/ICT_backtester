import pandas as pd
from datetime import timedelta
from domain.smt import detect_smt_divergence


def test_bearish_smt():
    base_time = pd.Timestamp("2026-09-28 10:00:00")

    # ES makes a Higher High. We must include an intervening Swing Low to track MSS.
    es_swings = [
        {'type': 'SwingHigh', 'time': base_time, 'price': 4000.0},
        {'type': 'SwingLow', 'time': base_time + timedelta(minutes=5), 'price': 3980.0},  # Intervening Low
        {'type': 'SwingHigh', 'time': base_time + timedelta(minutes=10), 'price': 4010.0}
    ]

    # NQ makes a Lower High
    nq_swings = [
        {'type': 'SwingHigh', 'time': base_time, 'price': 15000.0},
        {'type': 'SwingHigh', 'time': base_time + timedelta(minutes=11), 'price': 14950.0}
    ]

    # Mock df_es: Simulates a candle 5 minutes later breaking the intervening low (3980.0)
    df_es = pd.DataFrame({'close': [3970.0]}, index=[base_time + timedelta(minutes=15)])
    df_nq = pd.DataFrame()  # Empty dummy DataFrame

    smts = detect_smt_divergence(es_swings, nq_swings, df_es, df_nq, time_window_minutes=15)

    assert len(smts) == 1
    assert smts[0]['type'] == 'Bearish_SMT'


def test_bullish_smt():
    base_time = pd.Timestamp("2026-09-28 10:00:00")

    # ES makes a Lower Low. We must include an intervening Swing High to track MSS.
    es_swings = [
        {'type': 'SwingLow', 'time': base_time, 'price': 4000.0},
        {'type': 'SwingHigh', 'time': base_time + timedelta(minutes=5), 'price': 4020.0},  # Intervening High
        {'type': 'SwingLow', 'time': base_time + timedelta(minutes=10), 'price': 3990.0}
    ]

    # NQ makes a Higher Low
    nq_swings = [
        {'type': 'SwingLow', 'time': base_time, 'price': 15000.0},
        {'type': 'SwingLow', 'time': base_time + timedelta(minutes=11), 'price': 15050.0}
    ]

    # Mock df_es: Simulates a candle 5 minutes later breaking the intervening high (4020.0)
    df_es = pd.DataFrame({'close': [4030.0]}, index=[base_time + timedelta(minutes=15)])
    df_nq = pd.DataFrame()  # Empty dummy DataFrame

    smts = detect_smt_divergence(es_swings, nq_swings, df_es, df_nq, time_window_minutes=15)

    assert len(smts) == 1
    assert smts[0]['type'] == 'Bullish_SMT'