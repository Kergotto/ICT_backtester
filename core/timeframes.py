import pandas as pd


def resample_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resamples a 1-minute DataFrame with a DatetimeIndex into a higher timeframe.

    Args:
        df (pd.DataFrame): Pristine 1-minute OHLCV DataFrame.
        timeframe (str): Target timeframe. Must be one of:
                         '5m', '15m', '1H', '4H', '1D'

    Returns:
        pd.DataFrame: The resampled higher-timeframe DataFrame.
    """
    tf_mapping = {
        '5m': '5min',
        '15m': '15min',
        '1H': '1h',
        '4H': '4h',
        '1D': '1D'
    }

    if timeframe not in tf_mapping:
        raise ValueError(f"Unsupported timeframe: '{timeframe}'. Use one of {list(tf_mapping.keys())}")

    freq = tf_mapping[timeframe]

    aggregation_rules = {
        'open': 'first',
        'close': 'last',
        'high': 'max',
        'low': 'min',
        'volume': 'sum'
    }

    resampled = df.resample(freq, closed='left', label='left').agg(aggregation_rules)

    # Drop any periods where the market was closed (where no 1-minute bars existed)
    resampled = resampled.dropna(subset=['open'])

    return resampled