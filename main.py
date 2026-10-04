from pathlib import Path
from tests.analytics import print_structure_events
from core.data_loader import load_historical_data, resample_timeframe
from domain.swings import detect_swings
from domain.structure import detect_structure
from domain.liquidity import track_liquidity, GapEngine
from engine.session_engine import extract_session_liquidity
from visualization.plotter import plot_liquidity_chart

ENABLE_PLOTTING = True  # Set to False when running batch backtests for max speed


def run_pipeline(data_path: Path):
    print("1. Loading 1-minute historical data...")
    df_1m = load_historical_data(data_path)
    df_1m = df_1m[(df_1m['high'] - df_1m['low']) < 40.0]

    print("2. Resampling higher timeframes (15m context)...")
    df_15m = resample_timeframe(df_1m, '15min')

    print("3. Detecting Swings & Market Structure...")
    df_swings = detect_swings(df_1m)
    df_structure = detect_structure(df_1m, df_swings)

    print("4. Extracting Macro Session Liquidity (Asia/London)...")
    macro_pools = extract_session_liquidity(df_1m)

    # Group macro pools by their injection time to avoid lookahead bias
    macro_pools_by_time = {}
    for pool in macro_pools:
        injection_time = pool['time']

        if injection_time not in macro_pools_by_time:
            macro_pools_by_time[injection_time] = []
        macro_pools_by_time[injection_time].append(pool)

    print("5. Running Spatial Hash Liquidity Vault & Gap Engine...")
    df_final, liq_engine = track_liquidity(df_structure, df_swings, macro_pools_by_time, tolerance_ticks=4)

    # Run Gap Engine over the dataset
    gap_engine = GapEngine()
    for row in df_1m.itertuples():
        gap_engine.process_candle(
            timestamp=row.Index,
            open_price=row.open,
            close_price=row.close,
            high=row.high,
            low=row.low
        )

    # Collect Active Pools & Unfilled Gaps for strategy processing
    all_pools = [pool for bucket in liq_engine.vault.values() for pool in bucket]
    active_pools = liq_engine.get_active_pools()
    unfilled_gaps = [g for g in gap_engine.active_gaps if not g.is_filled]

    print("\n" + "=" * 50)
    print("          ENGINE EXECUTION SUMMARY          ")
    print("=" * 50)
    print(f"Total Pools Detected : {len(all_pools)}")
    print(f"Active Pools Stored  : {len(active_pools)}")
    print(f"Weekend Gaps Tracked : {len(gap_engine.active_gaps)} (Unfilled: {len(unfilled_gaps)})")
    print("=" * 50 + "\n")

    print("--- GAP DETAILS ---")
    for i, gap in enumerate(gap_engine.active_gaps):
        status = "🟢 FILLED" if gap.is_filled else "🔴 UNFILLED"
        print(f"Gap {i + 1}: Fri {gap.friday_close_time.strftime('%m-%d %H:%M')} -> "
              f"Sun {gap.sunday_open_time.strftime('%m-%d %H:%M')} | Status: {status}")

    return df_1m, df_swings, all_pools, gap_engine


def main():
    es_data_path = Path("data/es_1m_sample.csv")

    df_1m, df_swings, all_pools, gap_engine = run_pipeline(es_data_path)

    if ENABLE_PLOTTING:
        print("Rendering chart overlay...")

        macro_pools = [p for p in all_pools if p.source == "Macro_Session"]

        if macro_pools:
            target_time = macro_pools[0].creation_time
            print(f"\nCentering plot on Macro Session Pool at: {target_time}")

            target_idx = df_1m.index.get_indexer([target_time], method='nearest')[0]

            # Slice 300 candles before and 700 candles after to see the whole day
            start_idx = max(0, target_idx - 300)
            end_idx = min(len(df_1m), target_idx + 700)
            plot_df = df_1m.iloc[start_idx:end_idx]
        else:
            print("\nNo Macro Session pools found, defaulting to end of dataset.")
            plot_df = df_1m.iloc[-1000:]

        start_time = plot_df.index[0]
        end_time = plot_df.index[-1]

        # Toggle this to False to hide all the structural noise and ONLY see Macro Highs/Lows
        SHOW_STRUCTURAL_POOLS = True

        visible_pools = []
        for p in all_pools:
            # Always draw the true Macro Session extremes (Gold, Blue, Purple)
            if getattr(p, 'source', '') == "Macro_Session":
                if p.creation_time >= start_time and p.creation_time <= end_time:
                    visible_pools.append(p)

            # Only draw the standard red/green structural pools if the toggle is True
            elif SHOW_STRUCTURAL_POOLS:
                if (p.creation_time >= start_time and p.creation_time <= end_time) or \
                        any(start_time <= t <= end_time for _, t in p.events):
                    visible_pools.append(p)

        # Hide the swing markers if structural pools are disabled
        if SHOW_STRUCTURAL_POOLS:
            visible_swings = [s for s in df_swings if start_time <= s['time'] <= end_time]
        else:
            visible_swings = []

        print(f"Drawing {len(visible_pools)} pools and {len(visible_swings)} swings...")
        plot_liquidity_chart(plot_df, visible_pools, visible_swings)



if __name__ == "__main__":
    main()