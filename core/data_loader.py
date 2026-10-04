from pathlib import Path
import pandas as pd
import numpy as np


def save_to_parquet(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Saves a cleaned DataFrame to Parquet format.
    Using Parquet drastically speeds up loading speeds for 10-year datasets.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression='snappy')


def load_historical_data(filepath: str | Path) -> pd.DataFrame:
    """
    Loads raw historical 1-minute futures data, standardizes column names,
    enforces a strict Datetime Index aligned to New York Time, and validates integrity.
    Automatically converts CSVs to Parquet for instant future loading.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path.absolute()}")

    if path.suffix == '.csv':
        parquet_path = path.with_suffix('.parquet')
        if parquet_path.exists():
            print(f"⚡ Found optimized Parquet file. Loading {parquet_path.name} instantly...")
            # If the parquet exists, it's already cleaned and timezone-adjusted. Return immediately.
            return pd.read_parquet(parquet_path)
        else:
            print(f"⏳ Loading raw CSV {path.name} (this might take a moment)...")
            df = pd.read_csv(path)
    elif path.suffix == '.parquet':
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .csv or .parquet.")

    df.columns = df.columns.str.lower().str.strip()

    if not isinstance(df.index, pd.DatetimeIndex):
        time_col = None
        for possible_name in ['datetime', 'time', 'timestamp', 'date']:
            if possible_name in df.columns:
                time_col = possible_name
                break

        if not time_col:
            raise KeyError("Could not find a valid timestamp column (datetime, time, timestamp, or date).")

        df = df.rename(columns={time_col: 'datetime'})
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')

    df.index.name = 'datetime'

    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required price column '{col}' is missing from the dataset.")
        df[col] = df[col].astype(np.float64)

    if 'volume' in df.columns:
        df['volume'] = df['volume'].astype(np.float64)
    else:
        df['volume'] = 0.0

    df = df[required_cols + ['volume']]

    if df.index.tz is None:
        df = df.tz_localize('UTC').tz_convert('America/New_York')
    else:
        df = df.tz_convert('America/New_York')

    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()

    if (df[required_cols] <= 0).any().any():
        raise ValueError("Dataset contains invalid prices (less than or equal to zero).")

    if path.suffix == '.csv':
        print(f"💾 Saving cleaned data to {parquet_path.name} for instant loading next time...")
        save_to_parquet(df, parquet_path)

    return df


def resample_timeframe(df_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Aggregates 1-minute data into a higher timeframe (e.g., '15min', '1H', '4H').
    Ensures accurate OHLCV mapping without lookahead bias.
    """
    aggregation_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }

    df_resampled = df_1m.resample(
        timeframe,
        closed='left',
        label='left'
    ).agg(aggregation_rules)

    # Drop empty rows (weekends/market close)
    df_resampled.dropna(inplace=True)

    return df_resampled