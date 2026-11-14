from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np


def save_to_parquet(df: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".parquet.tmp")
    clean_df = df.copy()
    clean_df.attrs = {}
    clean_df.to_parquet(tmp_path, compression="snappy")
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


def _apply_rollover_back_adjustment_from_raw(
    sub: pd.DataFrame,
    front_contracts: pd.DataFrame,
    tick_size: float = 0.25,
) -> pd.DataFrame:
    """
    Applies TradingView-style continuous back-adjustment (B-ADJ) across contract rollovers
    using the exact simultaneous 1-minute close price difference between the new and old
    front-month contracts at each rollover boundary. The most recent contract has 0.0 adjustment.
    """
    fc = front_contracts.sort_values("session_date").reset_index(drop=True)
    if len(fc) <= 1:
        return sub

    roll_events = []
    for k in range(1, len(fc)):
        old_sym = fc.loc[k - 1, "symbol"]
        new_sym = fc.loc[k, "symbol"]
        if old_sym != new_sym:
            roll_date = fc.loc[k, "session_date"]
            prev_date = fc.loc[k - 1, "session_date"]
            window = sub[sub["session_date"].isin([prev_date, roll_date])]
            old_bars = window[window["symbol"] == old_sym].set_index("datetime")["close"]
            new_bars = window[window["symbol"] == new_sym].set_index("datetime")["close"]
            common_ts = old_bars.index.intersection(new_bars.index)

            if len(common_ts) > 0:
                ref_ts = common_ts[-1]
                raw_diff = float(new_bars.loc[ref_ts] - old_bars.loc[ref_ts])
            elif len(old_bars) > 0 and len(new_bars) > 0:
                raw_diff = float(new_bars.iloc[0] - old_bars.iloc[-1])
            else:
                raw_diff = 0.0

            adj_diff = round(raw_diff / tick_size) * tick_size
            roll_events.append((roll_date, adj_diff))

    merged = sub.merge(fc, on=["session_date", "symbol"], how="inner").copy()
    if not roll_events:
        return merged

    total_adj = pd.Series(0.0, index=merged.index, dtype=np.float64)
    for roll_date, adj_diff in roll_events:
        total_adj += np.where(merged["session_date"] < roll_date, adj_diff, 0.0)

    for col in ("open", "high", "low", "close"):
        merged[col] = merged[col] + total_adj
    return merged


def _apply_fallback_rollover_badj_on_stitched(df: pd.DataFrame, root_symbol: str) -> pd.DataFrame:
    """
    If only an unadjusted stitched Parquet exists (and raw CSVs were deleted),
    detects quarterly CME rollover jumps in Mar/Jun/Sep/Dec (days 8-19 at 18:00 NY)
    and back-adjusts prior bars to match TradingView ES1!/NQ1! (B-ADJ).
    """
    out = df.copy()
    idx = out.index
    if len(out) < 1000:
        return out

    closes = out["close"].to_numpy(dtype=float)
    opens = out["open"].to_numpy(dtype=float)
    jumps = opens[1:] - closes[:-1]
    min_roll_jump = 35.0 if root_symbol.upper() == "ES" else 140.0

    # Check if August 17, 2026 is already back-adjusted (~7892.25 on ES)
    aug17 = out[(idx >= pd.Timestamp("2026-08-17 00:00", tz="America/New_York")) & (idx <= pd.Timestamp("2026-08-17 16:00", tz="America/New_York"))]
    if not aug17.empty and root_symbol.upper() == "ES" and float(aug17["high"].max()) >= 7885.0:
        return out

    roll_indices = []
    for i in range(len(jumps)):
        ts_next = idx[i + 1]
        ts_prev = idx[i]
        gap_hours = (ts_next - ts_prev).total_seconds() / 3600.0
        if ts_next.month in (3, 6, 9, 12) and 8 <= ts_next.day <= 21 and abs(jumps[i]) >= min_roll_jump and gap_hours <= 72.0:
            roll_indices.append((i + 1, round(float(jumps[i]) / 0.25) * 0.25))

    if not roll_indices:
        return out

    adj_arr = np.zeros(len(out), dtype=np.float64)
    for r_idx, jump_val in roll_indices:
        adj_arr[:r_idx] += jump_val

    for col in ("open", "high", "low", "close"):
        out[col] = out[col].to_numpy(dtype=np.float64) + adj_arr
    return out


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

    sub = _apply_rollover_back_adjustment_from_raw(sub, front_contracts, tick_size=0.25)
    sub = sub.set_index("datetime").sort_index()
    sub = sub[~sub.index.duplicated(keep="first")]
    sub = sub[required_cols + ["volume"]]
    sub = sub[(sub[required_cols] > 0).all(axis=1)]
    return sub


def _slice_date_range(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    htf_lookback_days: int = 125,
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
    data_dir: Path,
    databento_csv_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    htf_lookback_days: int = 125,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_path = Path(data_dir)
    es_badj_pq = data_path / "es_1m_databento_badj.parquet"
    nq_badj_pq = data_path / "nq_1m_databento_badj.parquet"
    es_raw_pq = data_path / "es_1m_databento.parquet"
    nq_raw_pq = data_path / "nq_1m_databento.parquet"

    df_es = _safe_read_parquet(es_badj_pq)
    if df_es is not None:
        print(f"⚡ Found optimized back-adjusted {es_badj_pq.name}. Loading ES instantly...")

    df_nq = _safe_read_parquet(nq_badj_pq)
    if df_nq is not None:
        print(f"⚡ Found optimized back-adjusted {nq_badj_pq.name}. Loading NQ instantly...")

    if df_es is None or df_nq is None:
        csv_files = _find_databento_csv_files(data_path, databento_csv_name)
        for csv_path in csv_files:
            if df_es is not None and df_nq is not None:
                break
            if not csv_path.exists() or csv_path.stat().st_size == 0:
                continue
            print(f"⏳ Building TradingView B-ADJ continuous series from {csv_path.name}...")
            df_raw = pd.read_csv(csv_path)
            df_raw.columns = df_raw.columns.str.lower().str.strip()

            if df_es is None:
                df_es_ext = _extract_continuous_front_month(df_raw, "ES")
                if df_es_ext is not None:
                    save_to_parquet(df_es_ext, es_badj_pq)
                    df_es = df_es_ext

            if df_nq is None:
                df_nq_ext = _extract_continuous_front_month(df_raw, "NQ")
                if df_nq_ext is not None:
                    save_to_parquet(df_nq_ext, nq_badj_pq)
                    df_nq = df_nq_ext

    if df_es is None:
        df_es_unadj = _safe_read_parquet(es_raw_pq)
        if df_es_unadj is not None:
            print("🔧 Applying TradingView B-ADJ rollover alignment to existing es_1m_databento.parquet...")
            df_es = _apply_fallback_rollover_badj_on_stitched(df_es_unadj, "ES")
            save_to_parquet(df_es, es_badj_pq)
        else:
            raise FileNotFoundError("Could not find valid ES data.")

    if df_nq is None:
        df_nq_unadj = _safe_read_parquet(nq_raw_pq)
        if df_nq_unadj is not None:
            print("🔧 Applying TradingView B-ADJ rollover alignment to existing nq_1m_databento.parquet...")
            df_nq = _apply_fallback_rollover_badj_on_stitched(df_nq_unadj, "NQ")
            save_to_parquet(df_nq, nq_badj_pq)
        else:
            raise FileNotFoundError("Could not find valid NQ data.")

    return (
        _slice_date_range(df_es, start_date, end_date, htf_lookback_days=htf_lookback_days),
        _slice_date_range(df_nq, start_date, end_date, htf_lookback_days=htf_lookback_days),
    )


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

    if (df[required_cols] <= 0).any().any():
        raise ValueError("Dataset contains invalid prices (less than or equal to zero).")

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