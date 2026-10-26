import pandas as pd
from datetime import timedelta
from domain.fibonacci import track_fibonacci_ranges


def test_multi_swing_fibonacci_and_highest_candle_point():
    base_time = pd.Timestamp("2026-09-28 10:00:00")

    # Uptrend consisting of 2 higher highs and higher lows
    swings = [
        {"type": "SwingLow", "time": base_time, "price": 74913.0},
        {"type": "SwingHigh", "time": base_time + timedelta(minutes=15), "price": 82000.0},
        {"type": "SwingLow", "time": base_time + timedelta(minutes=30), "price": 76000.0},
        {"type": "SwingHigh", "time": base_time + timedelta(minutes=45), "price": 87000.0},
    ]

    # At 10:46, a candle wick reaches 87392.0 (higher than the 87000 SwingHigh),
    # and at 10:50 price retraces into OTE (79000.0)
    df = pd.DataFrame(
        {
            "open": [75000.0, 81500.0, 76500.0, 86500.0, 86900.0, 81000.0],
            "high": [75500.0, 82000.0, 77000.0, 87000.0, 87392.0, 81500.0],
            "low": [74913.0, 80000.0, 76000.0, 85000.0, 86000.0, 79000.0],
            "close": [75200.0, 80500.0, 76800.0, 86800.0, 86200.0, 79500.0],
        },
        index=[
            base_time,
            base_time + timedelta(minutes=15),
            base_time + timedelta(minutes=30),
            base_time + timedelta(minutes=45),
            base_time + timedelta(minutes=46),
            base_time + timedelta(minutes=50),
        ],
    )

    fibs = track_fibonacci_ranges(df, swings)

    assert len(fibs) == 1
    fib = fibs[0]

    assert fib.direction == "BULLISH"
    assert fib.anchor_price == 74913.0  # 1.0 level (lowest low)
    assert fib.extreme_price == 87392.0  # 0.0 level (highest candle point)
    assert fib.swing_count == 2

    # Verify Fibonacci levels
    lvls = fib.levels
    assert round(lvls[0.0]) == 87392
    assert round(lvls[0.382]) == 82625
    assert round(lvls[0.5]) == 81152
    assert round(lvls[0.62]) == 79655
    assert round(lvls[0.705]) == 78594
    assert round(lvls[0.79]) == 77534
    assert round(lvls[1.0]) == 74913

    assert fib.is_in_ote(79000.0) is True
    assert fib.ote_mitigated is True