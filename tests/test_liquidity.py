import pytest
import pandas as pd
from domain.swings import detect_swings
from domain.structure import detect_structure
from domain.liquidity import track_liquidity


def test_bullish_liquidity_sweep():
    """
    Simulates a controlled 10-candle environment to guarantee
    the spatial-indexed liquidity engine detects an SSL sweep correctly.
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

    all_pools = engine.get_all_pools()
    ssl_pools = [p for p in all_pools if p.pool_type == "SSL"]

    assert len(ssl_pools) >= 1, "Engine completely failed to detect the SSL pool."

    target_pool = ssl_pools[0]

    assert target_pool.lower_bound == 3998.0, f"Expected pool at 3998.0, got {target_pool.lower_bound}"
    assert target_pool.has_been_swept is True, "Engine failed to mark the pool state as swept."

    events = [e[0] for e in target_pool.events]
    assert "SWEEP" in events, "The 'SWEEP' event is missing from the pool's lifecycle log."


def test_broken_pool_pruned_from_active_buckets():
    """
    Verifies that when a candle body closes beyond a liquidity pool (BREAK),
    the pool is immediately pruned from the active spatial buckets (engine.vault)
    and moved to engine.archived_pools.
    """
    times = pd.date_range(
        start='2026-07-20 09:30:00',
        periods=10,
        freq='1min',
        tz='America/New_York'
    )

    # Candle 1 forms a Swing Low at 4000.
    # Candle 7 closes at 3995 (body break below 4000 SSL pool).
    data = {
        'open': [4010, 4005, 4008, 4015, 4020, 4025, 4020, 4010, 3996, 3998],
        'high': [4015, 4010, 4012, 4020, 4025, 4030, 4025, 4012, 4000, 4002],
        'low': [4005, 4000, 4002, 4008, 4015, 4020, 4015, 3994, 3993, 3995],
        'close': [4008, 4002, 4010, 4018, 4022, 4028, 4018, 3995, 3998, 4000],
        'volume': [100] * 10
    }

    df_mock = pd.DataFrame(data, index=times)
    df_swings = detect_swings(df_mock)
    df_struct = detect_structure(df_mock, df_swings)
    _, engine = track_liquidity(df_struct, df_swings, {}, tolerance_ticks=4)

    active_ssl = [p for p in engine.get_active_pools() if p.pool_type == "SSL"]
    archived_ssl = [p for p in engine.archived_pools if p.pool_type == "SSL"]

    # The broken SSL pool at 4000 must be evicted from active buckets and stored in archived_pools
    assert all(p.initial_price != 4000.0 for p in active_ssl)
    assert any(p.initial_price == 4000.0 and not p.is_active for p in archived_ssl)