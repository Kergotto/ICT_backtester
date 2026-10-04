import pytest
import pandas as pd
from core.timeframes import resample_data


@pytest.fixture
def sample_1m_data() -> pd.DataFrame:
    """Creates a mock 5-minute window of 1-minute bars to test aggregation."""
    times = pd.date_range(start="2026-07-15 09:00:00", periods=5, freq="1min", tz="America/New_York")

    data = {
        "open": [100.0, 102.0, 101.0, 99.0, 103.0],
        "high": [103.0, 104.0, 102.0, 101.0, 105.0],
        "low": [99.0, 101.0, 98.0, 97.0, 101.0],
        "close": [102.0, 101.0, 99.0, 103.0, 102.5],
        "volume": [100.0, 150.0, 200.0, 50.0, 300.0]
    }

    df = pd.DataFrame(data, index=times)
    df.index.name = "datetime"
    return df


def test_5m_resampling(sample_1m_data):
    """Checks if OHLCV values are compressed mathematically correctly."""
    df_5m = resample_data(sample_1m_data, "5m")

    assert len(df_5m) == 1

    # Open of 5m bar must be open of the first 1m bar (09:00)
    assert df_5m.iloc[0]['open'] == 100.0

    # Close of 5m bar must be close of the last 1m bar (09:04)
    assert df_5m.iloc[0]['close'] == 102.5

    # High of 5m bar must be maximum high in that 5m span (105.0)
    assert df_5m.iloc[0]['high'] == 105.0

    # Low of 5m bar must be minimum low in that 5m span (97.0)
    assert df_5m.iloc[0]['low'] == 97.0

    # Volume of 5m bar must be the sum of all volumes (100 + 150 + 200 + 50 + 300 = 800)
    assert df_5m.iloc[0]['volume'] == 800.0


def test_invalid_timeframe_raises_error(sample_1m_data):
    """Ensures the function rejects nonsense timeframes gracefully."""
    with pytest.raises(ValueError):
        resample_data(sample_1m_data, "3H")  # Not in our standard list