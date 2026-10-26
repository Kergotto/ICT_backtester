import pandas as pd
from datetime import timedelta
from domain.equilibrium import track_dealing_ranges


def test_single_leg_equilibrium_and_zones():
    base_time = pd.Timestamp("2026-09-28 10:00:00")

    swings = [
        {"type": "SwingLow", "time": base_time, "price": 4000.0},
        {"type": "SwingHigh", "time": base_time + timedelta(minutes=10), "price": 4100.0},
    ]

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

    assert dr.get_zone(4075.0) == "PREMIUM"
    assert dr.get_zone(4025.0) == "DISCOUNT"
    assert dr.get_zone(4050.0) == "EQUILIBRIUM"

    assert dr.eq_mitigated is True
    assert dr.status == "EQ_MITIGATED"
    assert dr.eq_mitigation_time == base_time + timedelta(minutes=12)


def test_equilibrium_spawns_on_every_high_low_leg():
    base_time = pd.Timestamp("2026-09-28 15:46:00")

    swings = [
        {"type": "SwingLow", "time": base_time, "price": 29309.75},
        {"type": "SwingHigh", "time": base_time + timedelta(minutes=2), "price": 29359.00},
        {"type": "SwingLow", "time": base_time + timedelta(minutes=4), "price": 29343.00},
        {"type": "SwingHigh", "time": base_time + timedelta(minutes=7), "price": 29372.25},
    ]

    df = pd.DataFrame(
        {
            "open": [29350.0, 29345.0, 29365.0],
            "high": [29359.0, 29350.0, 29372.25],
            "low": [29345.0, 29343.0, 29360.0],
            "close": [29348.0, 29349.0, 29370.0],
        },
        index=[
            base_time + timedelta(minutes=2),
            base_time + timedelta(minutes=4),
            base_time + timedelta(minutes=7),
        ],
    )

    ranges = track_dealing_ranges(df, swings)

    # All 3 legs (Bullish 15:46->15:48, Bearish 15:48->15:50, Bullish 15:50->15:53) spawn an Equilibrium
    assert len(ranges) == 3

    assert ranges[0].direction == "BULLISH"
    assert ranges[0].low == 29309.75
    assert ranges[0].high == 29359.00

    assert ranges[1].direction == "BEARISH"
    assert ranges[1].high == 29359.00
    assert ranges[1].low == 29343.00

    assert ranges[2].direction == "BULLISH"
    assert ranges[2].low == 29343.00
    assert ranges[2].high == 29372.25