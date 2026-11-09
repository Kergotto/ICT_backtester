from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np


def save_to_parquet(df: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, compression="snappy")
    tmp_path.replace(path)


def _safe_read_parquet(parquet_path: Path) -> Optional[pd.DataFrame]:
    if not parquet_path.exists():
        return None
    try:
        if parquet_path.stat().st_size < 128:
            parquet_path.unlink(missing_ok=True)
            return None
        return pd.read_parquet(parquet_path)
    except Exception:
        parquet_path.unlink(missing_ok=True)
        return None


def _extract_continuous_front_month(df_raw: pd.DataFrame, root_symbol: str) -> Optional[pd.DataFrame]:
    sym_series = df_raw["symbol"].astype(str).str.strip()
    outright_pattern = rf"^{root_symbol.upper()}[FGHJKMNQUVXZ]\d{{1,2}}$"
    continuous_pattern = rf"^{root_symbol.upper()}\.[vcns]\.\d+$"

    mask = sym_series.str.match(outright_pattern, na=False) | sym_series.str.match(continuous_pattern, na=False)
    sub = df_raw.loc[mask].copy()
    if sub.empty:
        return None

    time_col = None
    for possible_name in ("ts_event", "datetime", "time", "timestamp", "date"):
        if possible_name in sub.columns:
            time_col = possible_name
            break
    if not time_col:
        raise KeyError("Could not find a valid timestamp column.")

    try:
        dt_series = pd.to_datetime(sub[time_col], utc=True, format="ISO8601")
    except Exception:
        dt_series = pd.to_datetime(sub[time_col], utc=True)

    sub["datetime"] = dt_series.dt.tz_convert("America/New_York")

    required_cols = ["open", "high", "low", "close"]
    for col in required_cols:
        sub[col] = sub[col].astype(np.float64)
    sub["volume"] = sub["volume"].astype(np.float64) if "volume" in sub.columns else 0.0

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
    htf_lookback_days: int = 60,
) -> pd.DataFrame:
    full_df = df
    start_ts: Optional[pd.Timestamp] = None
    end_ts: Optional[pd.Timestamp] = None

    if start_date:
        start_ts = pd.Timestamp(start_date)
        start_ts = start_ts.tz_localize("America/New_York") if start_ts.tzinfo is None else start_ts.tz_convert("America/New_York")
        df = df[df.index >= start_ts]
    if end_date:
        end_ts = pd.Timestamp(end_date)
        end_ts = end_ts.tz_localize("America/New_York") if end_ts.tzinfo is None else end_ts.tz_convert("America/New_York")
        df = df[df.index <= end_ts]

    if start_ts is not None:
        lookback_start = start_ts - pd.Timedelta(days=htf_lookback_days)
        lb_df = full_df[full_df.index >= lookback_start]
        if end_ts is not None:
            lb_df = lb_df[lb_df.index <= end_ts]
        # Attach lookback DataFrame safely as a custom attribute without triggering Pandas aggregate ambiguity
        object.__setattr__(df, "_htf_lookback_df", lb_df)
    return df


def _find_databento_csv_files(data_path: Path, databento_csv_name: Optional[str] = None) -> List[Path]:
    if databento_csv_name:
        explicit = data_path / databento_csv_name
        if explicit.exists():
            return [explicit]

    csv_files = sorted(data_path.glob("glbx-mdp3-*.csv"))
    downloads_dir = Path.home() / "Downloads"
    if downloads_dir.exists():
        for dl_csv in sorted(downloads_dir.glob("glbx-mdp3-*.csv")):
            if dl_csv not in csv_files:
                csv_files.append(dl_csv)
        for dl_sub_csv in sorted(downloads_dir.glob("GLBX-*/glbx-mdp3-*.csv")):
            if dl_sub_csv not in csv_files:
                csv_files.append(dl_sub_csv)
    return csv_files


def load_databento_pair(
    data_dir: str | Path = "data",
    databento_csv_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_path = Path(data_dir)
    es_parquet = data_path / "es_1m_databento.parquet"
    nq_parquet = data_path / "nq_1m_databento.parquet"

    df_es = _safe_read_parquet(es_parquet)
    if df_es is not None:
        print(f"⚡ Found optimized {es_parquet.name}. Loading ES instantly...")

    df_nq = _safe_read_parquet(nq_parquet)
    if df_nq is not None:
        print(f"⚡ Found optimized {nq_parquet.name}. Loading NQ instantly...")

    if df_es is None or df_nq is None:
        csv_files = _find_databento_csv_files(data_path, databento_csv_name)

        for csv_path in csv_files:
            if df_es is not None and df_nq is not None:
                break
            if not csv_path.exists() or csv_path.stat().st_size == 0:
                continue
            print(f"⏳ Scanning Databento CSV ({csv_path})...")
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
        fallback_es_pq = data_path / "es_1m_sample.parquet"
        fallback_es_csv = data_path / "es_1m_sample.csv"
        if fallback_es_pq.exists():
            df_es = load_historical_data(fallback_es_pq)
        elif fallback_es_csv.exists():
            df_es = load_historical_data(fallback_es_csv)
        else:
            raise FileNotFoundError("Could not find valid ES data.")

    if df_nq is None:
        fallback_nq_pq = data_path / "nq_1m_sample.parquet"
        fallback_nq_csv = data_path / "nq_1m_sample.csv"
        if fallback_nq_pq.exists():
            df_nq = load_historical_data(fallback_nq_pq)
        elif fallback_nq_csv.exists():
            df_nq = load_historical_data(fallback_nq_csv)
        else:
            raise FileNotFoundError("Could not find valid NQ data.")

    return _slice_date_range(df_es, start_date, end_date), _slice_date_range(df_nq, start_date, end_date)


def load_historical_data(
    filepath: str | Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path.absolute()}")

    if path.suffix == ".csv":
        parquet_path = path.with_suffix(".parquet")
        df_cached = _safe_read_parquet(parquet_path)
        if df_cached is not None:
            return _slice_date_range(df_cached, start_date, end_date)
        df = pd.read_csv(path)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}.")

    df.columns = df.columns.str.lower().str.strip()

    if not isinstance(df.index, pd.DatetimeIndex):
        time_col = None
        for possible_name in ["ts_event", "datetime", "time", "timestamp", "date"]:
            if possible_name in df.columns:
                time_col = possible_name
                break

        if not time_col:
            raise KeyError("Could not find a valid timestamp column.")

        df = df.rename(columns={time_col: "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")

    df.index.name = "datetime"

    required_cols = ["open", "high", "low", "close"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required price column '{col}' is missing.")
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

    if path.suffix == ".csv":
        save_to_parquet(df, parquet_path)

    return _slice_date_range(df, start_date, end_date)


def resample_timeframe(df_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
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