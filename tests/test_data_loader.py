import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from core.data_loader import load_historical_data, save_to_parquet


@pytest.fixture
def sample_csv_data(tmp_path) -> Path:
    """
    Fixture that creates a temporary valid 1-minute CSV dataset for unit tests.
    We represent a 5-minute slice of UTC market time.
    """
    file_path = tmp_path / "test_data.csv"

    # Create 5 rows of 1-minute data
    # Raw datetime strings look like UTC naive timestamps
    data = {
        "Time": [
            "2026-07-15 13:00:00",
            "2026-07-15 13:01:00",
            "2026-07-15 13:02:00",
            "2026-07-15 13:03:00",
            "2026-07-15 13:04:00"
        ],
        "Open": [15000.00, 15005.25, 15003.50, 15010.00, 15008.75],
        "High": [15012.50, 15008.00, 15015.00, 15011.50, 15020.00],
        "Low": [14998.00, 15002.00, 15001.25, 15007.50, 15005.00],
        "Close": [15005.00, 15003.50, 15010.00, 15008.75, 15018.50],
        "Volume": [120, 95, 210, 150, 340]
    }

    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    return file_path


def test_load_historical_data_success(sample_csv_data):
    """
    Verifies that a valid CSV file is loaded successfully, and the column
    schemas, row shapes, and data structures are correct.
    """
    df = load_historical_data(sample_csv_data)

    # Test that the data was read and matches correct row counts
    assert len(df) == 5

    # Test schema standardization (columns must be strictly lowercase)
    expected_cols = ['open', 'high', 'low', 'close', 'volume']
    assert list(df.columns) == expected_cols

    # Check that our numeric data types are correctly converted to float64
    assert df['open'].dtype == np.float64
    assert df['volume'].dtype == np.float64


def test_timezone_conversion(sample_csv_data):
    """
    ICT TRADING CRITICAL TEST:
    Verifies that naive input data (interpreted as UTC) is correctly transformed into
    the US Eastern timezone (America/New_York), maintaining DST awareness.
    """
    df = load_historical_data(sample_csv_data)

    # Verify index is timezone-aware and set to New York
    assert df.index.tz is not None
    assert str(df.index.tz) == 'America/New_York'

    # UTC 13:00 (1:00 PM) in July 2026 (DST Active) is EDT 09:00 (9:00 AM)
    # This is exactly the pre-market session open window for US equity indices.
    first_row_time = df.index[0]
    assert first_row_time.hour == 9
    assert first_row_time.minute == 0


def test_missing_file_raises_error():
    """
    Verifies the system throws a descriptive error if the database file does not exist.
    """
    with pytest.raises(FileNotFoundError):
        load_historical_data("non_existent_file_path_1234.csv")


def test_invalid_columns_raise_error(tmp_path):
    """
    Verifies loader raises KeyError if vital OHLC values are missing.
    """
    bad_file = tmp_path / "bad_data.csv"
    data = {
        "Time": ["2026-07-15 13:00:00"],
        "Open": [15000.0],
        # Missing high, low, close
    }
    pd.DataFrame(data).to_csv(bad_file, index=False)

    with pytest.raises(KeyError):
        load_historical_data(bad_file)


def test_parquet_saving_and_loading(sample_csv_data, tmp_path):
    """
    Verifies that saving the formatted data to Parquet works, and that loading
    the Parquet file yields the exact same structure as loading raw CSV.
    """
    df_csv = load_historical_data(sample_csv_data)
    parquet_path = tmp_path / "test_data.parquet"

    # Save the processed data
    save_to_parquet(df_csv, parquet_path)

    # Load back the saved parquet file
    df_parquet = load_historical_data(parquet_path)

    # They should be perfectly identical
    pd.testing.assert_frame_equal(df_csv, df_parquet)