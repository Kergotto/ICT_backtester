from pathlib import Path
from domain.fvg import track_fvgs
from core.data_loader import load_historical_data, resample_timeframe
from domain.equilibrium import track_dealing_ranges
from domain.fibonacci import track_fibonacci_ranges
from domain.swings import detect_swings
from domain.structure import detect_structure
from domain.liquidity import track_liquidity, GapEngine
from engine.session_engine import extract_session_liquidity
from visualization.plotter import plot_liquidity_chart
from domain.smt import detect_smt_divergence

ENABLE_PLOTTING = True


def group_macros(macro_pools):
    grouped = {}
    for pool in macro_pools:
        t = pool['time']
        if t not in grouped:
            grouped[t] = []
        grouped[t].append(pool)
    return grouped


def run_pipeline(primary_path: Path, secondary_path: Path):
    print("1. Loading 1-minute historical data (ES & NQ)...")
    df_es = load_historical_data(primary_path)
    df_es = df_es[(df_es['high'] - df_es['low']) < 40.0]

    df_nq = load_historical_data(secondary_path)
    df_nq = df_nq[(df_nq['high'] - df_nq['low']) < 150.0]

    print("2. Detecting Swings, Market Structure, Equilibrium & Fibonacci Ranges...")
    es_swings = detect_swings(df_es)
    es_structure = detect_structure(df_es, es_swings)
    es_ranges = track_dealing_ranges(df_es, es_swings)
    es_fibs = track_fibonacci_ranges(df_es, es_swings)

    nq_swings = detect_swings(df_nq)
    nq_structure = detect_structure(df_nq, nq_swings)
    nq_ranges = track_dealing_ranges(df_nq, nq_swings)
    nq_fibs = track_fibonacci_ranges(df_nq, nq_swings)

    print("3. Extracting Macro Session Liquidity...")
    es_macro_by_time = group_macros(extract_session_liquidity(df_es))
    nq_macro_by_time = group_macros(extract_session_liquidity(df_nq))

    print("4. Running Spatial Hash Liquidity Vaults (Tolerance=2)...")
    _, es_liq_engine = track_liquidity(es_structure, es_swings, es_macro_by_time, tolerance_ticks=2)
    _, nq_liq_engine = track_liquidity(nq_structure, nq_swings, nq_macro_by_time, tolerance_ticks=2)

    # Weekend Gap Engine (ES only to save processing speed, usually sufficient)
    gap_engine = GapEngine()
    for row in df_es.itertuples():
        gap_engine.process_candle(row.Index, row.open, row.close, row.high, row.low)

    print("5. Detecting and tracking Fair Value Gaps (FVG)...")
    es_fvgs = track_fvgs(df_es)
    nq_fvgs = track_fvgs(df_nq)

    print("6. Calculating SMT Divergence...")
    smt_events = detect_smt_divergence(es_swings, nq_swings, df_es, df_nq, time_window_minutes=15)
    print(f"--> Found {len(smt_events)} SMT Events.")
    for smt in smt_events[-5:]:
        print(f"    {smt['description']}")

    es_pools = [pool for bucket in es_liq_engine.vault.values() for pool in bucket]
    nq_pools = [pool for bucket in nq_liq_engine.vault.values() for pool in bucket]

    return (
        df_es, es_swings, es_pools, es_fvgs, es_ranges, es_fibs,
        df_nq, nq_swings, nq_pools, nq_fvgs, nq_ranges, nq_fibs,
        gap_engine, smt_events
    )


def main():
    es_data_path = Path("data/es_1m_sample.csv")
    nq_data_path = Path("data/nq_1m_sample.csv")

    (
        df_es, es_swings, es_pools, es_fvgs, es_ranges, es_fibs,
        df_nq, nq_swings, nq_pools, nq_fvgs, nq_ranges, nq_fibs,
        gap_engine, smt_events
    ) = run_pipeline(es_data_path, nq_data_path)

    if ENABLE_PLOTTING:
        print("Rendering chart overlays...")

        common_index = df_es.index.intersection(df_nq.index)
        if len(common_index) == 0:
            print("CRITICAL: ES and NQ datasets have absolutely no overlapping timeframes.")
            return

        if smt_events:
            target_time = smt_events[-1]['time']
            print(f"Centering 2-hour window on SMT Event at: {target_time.strftime('%H:%M')}")
            target_idx = common_index.get_indexer([target_time], method='nearest')[0]

            start_idx = max(0, target_idx - 60)
            end_idx = min(len(common_index) - 1, target_idx + 60)
        else:
            end_idx = len(common_index) - 1
            start_idx = max(0, end_idx - 120)

        start_time = common_index[start_idx]
        end_time = common_index[end_idx]

        plot_df_es = df_es[(df_es.index >= start_time) & (df_es.index <= end_time)]
        plot_df_nq = df_nq[(df_nq.index >= start_time) & (df_nq.index <= end_time)]

        SHOW_STRUCTURAL_POOLS = True

        def filter_elements(pools, swings, fvgs, ranges, fibs, t_start, t_end, min_price, max_price):
            vis_pools, vis_swings, vis_fvgs, vis_ranges, vis_fibs = [], [], [], [], []

            for p in pools:
                bottom = getattr(p, 'lower_bound', getattr(p, 'bottom', 0))
                top = getattr(p, 'upper_bound', getattr(p, 'top', 0))

                if top < min_price or bottom > max_price:
                    continue

                if getattr(p, 'source', '') == "Macro_Session" or SHOW_STRUCTURAL_POOLS:
                    created_in_window = (t_start <= p.creation_time <= t_end)
                    active_at_start = (p.creation_time < t_start) and not any(
                        ev == "BREAK" and ts < t_start for ev, ts in p.events)
                    if created_in_window or active_at_start:
                        vis_pools.append(p)

            if SHOW_STRUCTURAL_POOLS:
                vis_swings = [s for s in swings if t_start <= s['time'] <= t_end]

            for f in fvgs:
                bottom = getattr(f, 'bottom', 0)
                top = getattr(f, 'top', 0)

                if top < min_price or bottom > max_price:
                    continue

                created_in_window = (t_start <= f.creation_time <= t_end)
                active_at_start = (f.creation_time < t_start) and not any(
                    ev in ["FILLED_BY_CLOSE", "INVERTED"] and ts < t_start for ev, ts, _ in f.events)
                if created_in_window or active_at_start:
                    vis_fvgs.append(f)

            for dr in ranges:
                if dr.high > max_price or dr.low < min_price:
                    continue
                if dr.creation_time >= t_start and dr.start_time <= t_end:
                    vis_ranges.append(dr)

            for fib in fibs:
                if fib.high > max_price or fib.low < min_price:
                    continue
                # Show multi-swing Fib ranges (swing_count >= 2) or significant trend legs in window
                in_window = (fib.extreme_time >= t_start and fib.start_time <= t_end)
                if in_window and fib.swing_count >= 2:
                    vis_fibs.append(fib)

            return vis_pools, vis_swings, vis_fvgs, vis_ranges, vis_fibs

        es_vis_pools, es_vis_swings, es_vis_fvgs, es_vis_ranges, es_vis_fibs = filter_elements(
            es_pools, es_swings, es_fvgs, es_ranges, es_fibs, start_time, end_time,
            plot_df_es['low'].min() - 50, plot_df_es['high'].max() + 50
        )

        nq_vis_pools, nq_vis_swings, nq_vis_fvgs, nq_vis_ranges, nq_vis_fibs = filter_elements(
            nq_pools, nq_swings, nq_fvgs, nq_ranges, nq_fibs, start_time, end_time,
            plot_df_nq['low'].min() - 100, plot_df_nq['high'].max() + 100
        )

        print(f"Plotting ES Chart ({len(es_vis_ranges)} Equilibriums, {len(es_vis_fibs)} Fib Ranges)...")
        plot_liquidity_chart(
            plot_df_es, es_vis_pools, es_vis_swings,
            fvgs=es_vis_fvgs, ranges=es_vis_ranges, fibs=es_vis_fibs, title="E-Mini S&P 500 (ES)"
        )

        print(f"Plotting NQ Chart ({len(nq_vis_ranges)} Equilibriums, {len(nq_vis_fibs)} Fib Ranges)...")
        plot_liquidity_chart(
            plot_df_nq, nq_vis_pools, nq_vis_swings,
            fvgs=nq_vis_fvgs, ranges=nq_vis_ranges, fibs=nq_vis_fibs, title="E-Mini Nasdaq 100 (NQ)"
        )


if __name__ == "__main__":
    main()