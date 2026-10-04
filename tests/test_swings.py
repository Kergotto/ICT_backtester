import pytest
import pandas as pd
from domain.swings import detect_swings

@pytest.fixture
def sample_swing_data() -> pd.DataFrame:
    """
    Creates a mock sequence to test strict 2-candle color transitions.
    Columns are strictly lowercase to match the historical Parquet data schema.
    """
    data = {
        "high": [100.0, 110.0, 120.0, 105.0],
        "low": [90.0, 95.0, 85.0, 70.0],
        "open": [100.0, 100.0, 90.0, 110.0],
        "close": [100.0, 90.0, 105.0, 95.0],
    }
    times = pd.date_range(start="2026-07-15 09:00:00", periods=4, freq="1min", tz="America/New_York")
    df = pd.DataFrame(data, index=times)
    return df

def test_detect_swings_logic(sample_swing_data):
    """Verifies that strict color-sequence swings are correctly isolated."""
    df_swings = detect_swings(sample_swing_data)

    assert len(df_swings) == 2

    # Check the Swing Low (Bearish -> Bullish transition)
    assert df_swings[0]['type'] == 'SwingLow'
    assert df_swings[0]['price'] == 85.0

    # Check the Swing High (Bullish -> Bearish transition)
    assert df_swings[1]['type'] == 'SwingHigh'
    assert df_swings[1]['price'] == 120.0