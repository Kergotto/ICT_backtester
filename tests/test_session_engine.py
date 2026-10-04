import pandas as pd
from engine.session_engine import extract_session_liquidity


def test_session_liquidity_extraction():
    """
    Simulates a 24-hour period to ensure Asia and London
    session extremes are correctly identified and isolated.
    """
    times = pd.date_range(
        start='2026-08-10 17:00:00',
        end='2026-08-11 16:59:00',
        freq='1min'
    )

    df_mock = pd.DataFrame({
        'open': [100.0] * len(times),
        'high': [101.0] * len(times),
        'low': [99.0] * len(times),
        'close': [100.0] * len(times)
    }, index=times)

    asia_high_time = pd.Timestamp('2026-08-10 20:15:00')
    df_mock.loc[asia_high_time, 'high'] = 150.0

    asia_low_time = pd.Timestamp('2026-08-11 01:45:00')
    df_mock.loc[asia_low_time, 'low'] = 50.0

    london_high_time = pd.Timestamp('2026-08-11 04:30:00')
    df_mock.loc[london_high_time, 'high'] = 180.0

    london_low_time = pd.Timestamp('2026-08-11 07:12:00')
    df_mock.loc[london_low_time, 'low'] = 30.0

    macro_pools = extract_session_liquidity(df_mock)

    # Sort highs/lows by session and type for easy assertion
    asia_bsl = next(p for p in macro_pools if p['session'] == 'Asia' and p['type'] == 'BSL')
    asia_ssl = next(p for p in macro_pools if p['session'] == 'Asia' and p['type'] == 'SSL')
    london_bsl = next(p for p in macro_pools if p['session'] == 'London' and p['type'] == 'BSL')
    london_ssl = next(p for p in macro_pools if p['session'] == 'London' and p['type'] == 'SSL')

    assert asia_bsl['price'] == 150.0, "Failed to find Asia High"
    assert asia_ssl['price'] == 50.0, "Failed to find Asia Low"

    assert london_bsl['price'] == 180.0, "Failed to find London High"
    assert london_ssl['price'] == 30.0, "Failed to find London Low"

    print("Session Engine passed! All Killzone extremes identified perfectly.")