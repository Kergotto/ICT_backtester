import pandas as pd
from datetime import timedelta
from domain.equilibrium import DealingRange, track_dealing_ranges, get_active_dealing_range


def test_bullish_dealing_range_and_zones():
    base_time = pd.Timestamp("2026-09-28 10:00:00")

    swings = [
        {"type": "SwingLow", "time": base_time, "price": 4000.0},
        {"type": "SwingHigh", "time": base_time + timedelta(minutes=10), "price": 4100.0},
    ]

    # Candles after 10:10:
    # 10:11 stays in Premium (4080)
    # 10:12 dips to 4045 (crosses 4050 EQ into Discount)
    df = pd.DataFrame(
        {
            "open": [4090.0, 4080.0],
            "high": [4095.0, 4085.0],
            "low": [4080.0, 4045.0],
            "close": [4080.0, 4048.0],
        },
        index=[
            base_time + timedelta(minutes=11),
            base_time + timedelta(minutes=12),
        ],
    )

    ranges = track_dealing_ranges(df, swings)

    assert len(ranges) == 1
    dr = ranges[0]

    assert dr.direction == "BULLISH"
    assert dr.low == 4000.0
    assert dr.high == 4100.0
    assert dr.eq == 4050.0

    # Verify zone classification
    assert dr.get_zone(4075.0) == "PREMIUM"
    assert dr.get_zone(4025.0) == "DISCOUNT"
    assert dr.get_zone(4050.0) == "EQUILIBRIUM"

    # Verify OTE (62% - 79% retracement from 4100 down toward 4000 -> 4021.0 to 4038.0)
    assert dr.is_in_ote(4030.0) is True
    assert dr.is_in_ote(4060.0) is False

    # Verify EQ mitigation was recorded at 10:12
    assert dr.eq_mitigated is True
    assert dr.status == "EQ_MITIGATED"
    assert dr.eq_mitigation_time == base_time + timedelta(minutes=12)


def test_internal_swings_ignored_until_range_break():
    base_time = pd.Timestamp("2026-09-28 10:00:00")

    swings = [
        # Initial Bearish Dealing Range: 5000 -> 4900 (EQ = 4950)
        {"type": "SwingHigh", "time": base_time, "price": 5000.0},
        {"type": "SwingLow", "time": base_time + timedelta(minutes=10), "price": 4900.0},
        # Internal retracement swings (inside 4900 - 5000): should NOT create a new range
        {"type": "SwingHigh", "time": base_time + timedelta(minutes=15), "price": 4960.0},
        {"type": "SwingLow", "time": base_time + timedelta(minutes=20), "price": 4920.0},
        # External expansion below 4900: SHOULD create a new Bearish Dealing Range
        {"type": "SwingLow", "time": base_time + timedelta(minutes=25), "price": 4850.0},
    ]

    df = pd.DataFrame(
        {
            "open": [4910.0, 4955.0, 4890.0],
            "high": [4960.0, 4960.0, 4895.0],
            "low": [4905.0, 4920.0, 4850.0],
            "close": [4955.0, 4925.0, 4860.0],
        },
        index=[
            base_time + timedelta(minutes=15),
            base_time + timedelta(minutes=20),
            base_time + timedelta(minutes=25),
        ],
    )

    ranges = track_dealing_ranges(df, swings)

    # Only 2 external ranges should exist (the internal 4960/4920 swings are ignored)
    assert len(ranges) == 2

    first_dr = ranges[0]
    assert first_dr.direction == "BEARISH"
    assert first_dr.high == 5000.0
    assert first_dr.low == 4900.0
    assert first_dr.eq == 4950.0
    assert first_dr.eq_mitigated is True
    assert first_dr.status == "BROKEN"
    assert first_dr.break_time == base_time + timedelta(minutes=25)

    second_dr = ranges[1]
    assert second_dr.direction == "BEARISH"
    assert second_dr.low == 4850.0

    # Active range lookup at 10:18 should return the first range
    active_at_18 = get_active_dealing_range(ranges, base_time + timedelta(minutes=18))
    assert active_at_18 == first_dr