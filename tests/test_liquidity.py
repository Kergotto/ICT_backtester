import pytest
import pandas as pd
from domain.swings import detect_swings
from domain.structure import detect_structure
from domain.liquidity import track_liquidity


def test_bullish_liquidity_sweep():
    """
    Simulates a controlled 10-candle environment to guarantee
    the spatial hash vault detects an SSL sweep correctly.
    """

    times = pd.date_range(
        start='2026-07-20 09:30:00',
        periods=10,
        freq='1min',
        tz='America/New_York'
    )

    # Candle 1 (Index 1): Forms a Swing Low at exactly 4000.
    # Candle 7 (Index 7): Price wicks down to 3998, sweeping the low, then closes at 4005.
    data = {
        'open': [4010, 4005, 4008, 4015, 4020, 4025, 4020, 4010, 4015, 4020],
        'high': [4015, 4010, 4012, 4020, 4025, 4030, 4025, 4015, 4020, 4025],
        'low': [4005, 4000, 4002, 4008, 4015, 4020, 4015, 3998, 4010, 4015],
        'close': [4008, 4002, 4010, 4018, 4022, 4028, 4018, 4005, 4018, 4022],
        'volume': [100] * 10
    }

    df_mock = pd.DataFrame(data, index=times)

    df_swings = detect_swings(df_mock)
    df_struct = detect_structure(df_mock, df_swings)
    df_final, engine = track_liquidity(df_struct, df_swings, {}, tolerance_ticks=4)

    all_pools = []
    for bucket in engine.vault.values():
        all_pools.extend(bucket)

    # Isolate the Sell-Side Liquidity pools
    ssl_pools = [p for p in all_pools if p.pool_type == "SSL"]

    assert len(ssl_pools) >= 1, "Engine completely failed to detect the SSL pool."

    target_pool = ssl_pools[0]

    # Verify the pool grabbed the correct price level
    # Verify the pool stretched its boundary to the sweeping wick tip (3998.0)
    assert target_pool.lower_bound == 3998.0, f"Expected pool at 4000.0, got {target_pool.lower_bound}"

    # Verify the boolean state flipped
    assert target_pool.has_been_swept == True, "Engine failed to mark the pool state as swept."

    # Verify the tripwire event log caught the exact action
    events = [e[0] for e in target_pool.events]
    assert "SWEEP" in events, "The 'SWEEP' event is missing from the pool's lifecycle log."