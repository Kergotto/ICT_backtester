from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np


def save_to_parquet(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Saves a cleaned DataFrame to Parquet format.
    Using Parquet drastically speeds up loading speeds for multi-year datasets.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="snappy")


def _extract_continuous_front_month(df_raw: pd.DataFrame, root_symbol: str) -> Optional[pd.DataFrame]:
    """
    Filters a raw Databento OHLCV DataFrame down to outright futures contracts for `root_symbol`
    (e.g. 'ES' or 'NQ'), removes calendar spreads and options, aligns timestamps to 'America/New_York',
    and selects the highest-volume front-month contract per CME Globex session (like ES1! / NQ1!).
    Returns None if the root symbol is not present in `df_raw`.
    """
    sym_series = df_raw["symbol"].astype(str).str.strip()
    outright_pattern = rf"^{root_symbol.upper()}[FGHJKMNQUVXZ]\d{{1,2}}$"
    continuous_pattern = rf"^{root_symbol.upper()}\.[vcns]\.\d+$"

    mask = sym_series.str.match(outright_pattern, na=False) | sym_series.str.match(continuous_pattern, na=False)
    sub = df_raw.loc[mask].copy()
    if sub.empty:
        prefix_matches = sym_series[sym_series.str.startswith(root_symbol.upper(), na=False)].unique()[:8]
        print(
            f"   ⚠️ No outright '{root_symbol}' futures found in this CSV. "
            f"(Sample '{root_symbol}*' symbols in file: {list(prefix_matches)})"
        )
        return None

    matched_contracts = sub["symbol"].unique()[:8]
    print(f"   ✅ Found {len(sub):,} rows for '{root_symbol}' futures (sample contracts: {list(matched_contracts)})")

    time_col = None
    for possible_name in ("ts_event", "datetime", "time", "timestamp", "date"):
        if possible_name in sub.columns:
            time_col = possible_name
            break
    if not time_col:
        raise KeyError("Could not find a valid timestamp column (ts_event, datetime, time, timestamp, or date).")

    try:
        dt_series = pd.to_datetime(sub[time_col], utc=True, format="ISO8601")
    except Exception:
        dt_series = pd.to_datetime(sub[time_col], utc=True)

    sub["datetime"] = dt_series.dt.tz_convert("America/New_York")

    required_cols = ["open", "high", "low", "close"]
    for col in required_cols:
        sub[col] = sub[col].astype(np.float64)
    sub["volume"] = sub["volume"].astype(np.float64) if "volume" in sub.columns else 0.0

    # CME Globex session date: 18:00 NY starts the next day's trading session.
    sub["session_date"] = (sub["datetime"] + pd.Timedelta(hours=6)).dt.date
    daily_vol = sub.groupby(["session_date", "symbol"], as_index=False)["volume"].sum()
    front_contracts = (
        daily_vol.sort_values(["session_date", "volume"], ascending=[True, False])
        .drop_duplicates(subset=["session_date"], keep="first")[["session_date", "symbol"]]
    )

    sub = sub.merge(front_contracts, on=["session_date", "symbol"], how="inner")
    sub = sub.set_index("datetime").sort_index()
    sub = sub[~sub.index.duplicated(keep="first")]
    sub = sub[required_cols + ["volume"]]
    sub = sub[(sub[required_cols] > 0).all(axis=1)]
    return sub


def _slice_date_range(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    if start_date:
        start_ts = pd.Timestamp(start_date)
        start_ts = start_ts.tz_localize("America/New_York") if start_ts.tzinfo is None else start_ts.tz_convert("America/New_York")
        df = df[df.index >= start_ts]
    if end_date:
        end_ts = pd.Timestamp(end_date)
        end_ts = end_ts.tz_localize("America/New_York") if end_ts.tzinfo is None else end_ts.tz_convert("America/New_York")
        df = df[df.index <= end_ts]
    return df


def load_databento_pair(
    data_dir: str | Path = "data",
    databento_csv_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads 1-minute continuous front-month ES and NQ DataFrames from the Databento dataset.
    Caches `es_1m_databento.parquet` and `nq_1m_databento.parquet` independently so that
    if one symbol is already extracted, it never needs to be re-parsed.
    """
    data_path = Path(data_dir)
    es_parquet = data_path / "es_1m_databento.parquet"
    nq_parquet = data_path / "nq_1m_databento.parquet"

    df_es: Optional[pd.DataFrame] = None
    df_nq: Optional[pd.DataFrame] = None

    if es_parquet.exists():
        print(f"⚡ Found optimized {es_parquet.name}. Loading ES instantly...")
        df_es = pd.read_parquet(es_parquet)

    if nq_parquet.exists():
        print(f"⚡ Found optimized {nq_parquet.name}. Loading NQ instantly...")
        df_nq = pd.read_parquet(nq_parquet)

    if df_es is None or df_nq is None:
        if databento_csv_name:
            csv_files: List[Path] = [data_path / databento_csv_name]
        else:
            csv_files = sorted(data_path.glob("glbx-mdp3-*.csv"))

        for csv_path in csv_files:
            if df_es is not None and df_nq is not None:
                break
            print(f"⏳ Scanning Databento CSV ({csv_path.name})...")
            df_raw = pd.read_csv(csv_path)
            df_raw.columns = df_raw.columns.str.lower().str.strip()

            if df_es is None:
                df_es_extracted = _extract_continuous_front_month(df_raw, "ES")
                if df_es_extracted is not None:
                    print(f"💾 Caching cleaned 5-year ES Parquet ({es_parquet.name})...")
                    save_to_parquet(df_es_extracted, es_parquet)
                    df_es = df_es_extracted

            if df_nq is None:
                df_nq_extracted = _extract_continuous_front_month(df_raw, "NQ")
                if df_nq_extracted is not None:
                    print(f"💾 Caching cleaned 5-year NQ Parquet ({nq_parquet.name})...")
                    save_to_parquet(df_nq_extracted, nq_parquet)
                    df_nq = df_nq_extracted

    if df_es is None:
        fallback_es = data_path / "es_1m_sample.csv"
        print(
            "⚠️ 'ES' outright futures were not found in the Databento CSV (looks like 'ES1' options were downloaded instead of 'ES' futures).\n"
            "   --> Temporarily falling back to `es_1m_sample.parquet` for ES until you drop the 'ES' Databento CSV into `data/`."
        )
        df_es = load_historical_data(fallback_es)

    if df_nq is None:
        fallback_nq = data_path / "nq_1m_sample.csv"
        print("⚠️ 'NQ' outright futures were not found in the Databento CSV. Falling back to `nq_1m_sample.parquet`.")
        df_nq = load_historical_data(fallback_nq)

    return _slice_date_range(df_es, start_date, end_date), _slice_date_range(df_nq, start_date, end_date)


def load_historical_data(
    filepath: str | Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Loads raw historical 1-minute futures data, standardizes column names,
    enforces a strict Datetime Index aligned to New York Time, and validates integrity.
    Automatically converts CSVs to Parquet for instant future loading.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path.absolute()}")

    if path.suffix == ".csv":
        parquet_path = path.with_suffix(".parquet")
        if parquet_path.exists():
            print(f"⚡ Found optimized Parquet file. Loading {parquet_path.name} instantly...")
            df = pd.read_parquet(parquet_path)
            return _slice_date_range(df, start_date, end_date)
        else:
            print(f"⏳ Loading raw CSV {path.name} (this might take a moment)...")
            df = pd.read_csv(path)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .csv or .parquet.")

    df.columns = df.columns.str.lower().str.strip()

    if not isinstance(df.index, pd.DatetimeIndex):
        time_col = None
        for possible_name in ["ts_event", "datetime", "time", "timestamp", "date"]:
            if possible_name in df.columns:
                time_col = possible_name
                break

        if not time_col:
            raise KeyError("Could not find a valid timestamp column (ts_event, datetime, time, timestamp, or date).")

        df = df.rename(columns={time_col: "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")

    df.index.name = "datetime"

    required_cols = ["open", "high", "low", "close"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required price column '{col}' is missing from the dataset.")
        df[col] = df[col].astype(np.float64)

    if "volume" in df.columns:
        df["volume"] = df["volume"].astype(np.float64)
    else:
        df["volume"] = 0.0

    df = df[required_cols + ["volume"]]

    if df.index.tz is None:
        df = df.tz_localize("UTC").tz_convert("America/New_York")
    else:
        df = df.tz_convert("America/New_York")

    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    if (df[required_cols] <= 0).any().any():
        raise ValueError("Dataset contains invalid prices (less than or equal to zero).")

    if path.suffix == ".csv":
        print(f"💾 Saving cleaned data to {parquet_path.name} for instant loading next time...")
        save_to_parquet(df, parquet_path)

    return _slice_date_range(df, start_date, end_date)


def resample_timeframe(df_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Aggregates 1-minute data into a higher timeframe (e.g., '15min', '1H', '4H').
    Ensures accurate OHLCV mapping without lookahead bias.
    """
    aggregation_rules = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    df_resampled = df_1m.resample(
        timeframe,
        closed="left",
        label="left",
    ).agg(aggregation_rules)

    df_resampled.dropna(inplace=True)
    return df_resampled