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

CHART_TIMEFRAME = "15min"

# If True (and CHART_TIMEFRAME == "1min"), overlays HTF_OVERLAY_TF confluences onto the 1m chart
OVERLAY_HTF_ON_1M = False
HTF_OVERLAY_TF = "15min"


def group_macros(macro_pools):
    grouped = {}
    for pool in macro_pools:
        t = pool['time']
        if t not in grouped:
            grouped[t] = []
        grouped[t].append(pool)
    return grouped


def run_pipeline(primary_path: Path, secondary_path: Path, timeframe: str = "1min"):
    print("1. Loading 1-minute historical data (ES & NQ)...")
    df_es_1m = load_historical_data(primary_path)
    df_es_1m = df_es_1m[(df_es_1m['high'] - df_es_1m['low']) < 40.0]

    df_nq_1m = load_historical_data(secondary_path)
    df_nq_1m = df_nq_1m[(df_nq_1m['high'] - df_nq_1m['low']) < 150.0]

    # Macro session liquidity is always extracted from 1m data for exact session boundaries
    print("2. Extracting Macro Session Liquidity...")
    es_macro_by_time = group_macros(extract_session_liquidity(df_es_1m))
    nq_macro_by_time = group_macros(extract_session_liquidity(df_nq_1m))

    # Resample if Higher Timeframe is selected
    if timeframe != "1min":
        print(f"   Resampling 1m data to Higher Timeframe ({timeframe})...")
        df_es = resample_timeframe(df_es_1m, timeframe)
        df_nq = resample_timeframe(df_nq_1m, timeframe)
        smt_window = 60  # Wider SMT alignment window for HTF swings
    else:
        df_es = df_es_1m
        df_nq = df_nq_1m
        smt_window = 15

    print(f"3. Detecting {timeframe} Swings, Structure, Equilibrium & Fibonacci Ranges...")
    es_swings = detect_swings(df_es)
    es_structure = detect_structure(df_es, es_swings)
    es_ranges = track_dealing_ranges(df_es, es_swings)
    es_fibs = track_fibonacci_ranges(df_es, es_swings)

    nq_swings = detect_swings(df_nq)
    nq_structure = detect_structure(df_nq, nq_swings)
    nq_ranges = track_dealing_ranges(df_nq, nq_swings)
    nq_fibs = track_fibonacci_ranges(df_nq, nq_swings)

    print(f"4. Running Spatial-Indexed Liquidity Engine on {timeframe} (Tolerance=2)...")
    _, es_liq_engine = track_liquidity(es_structure, es_swings, es_macro_by_time, tolerance_ticks=2)
    _, nq_liq_engine = track_liquidity(nq_structure, nq_swings, nq_macro_by_time, tolerance_ticks=2)

    gap_engine = GapEngine()
    for row in df_es.itertuples():
        gap_engine.process_candle(row.Index, row.open, row.close, row.high, row.low)

    print(f"5. Detecting and tracking {timeframe} Fair Value Gaps (FVG)...")
    es_fvgs = track_fvgs(df_es)
    nq_fvgs = track_fvgs(df_nq)

    print(f"6. Calculating {timeframe} SMT Divergence...")
    smt_events = detect_smt_divergence(es_swings, nq_swings, df_es, df_nq, time_window_minutes=smt_window)
    print(f"--> Found {len(smt_events)} SMT Events on {timeframe}.")
    for smt in smt_events[-5:]:
        print(f"    {smt['description']}")

    es_pools = es_liq_engine.get_all_pools()
    nq_pools = nq_liq_engine.get_all_pools()

    # Optional: Compute HTF confluences to overlay onto the 1-minute chart
    if timeframe == "1min" and OVERLAY_HTF_ON_1M:
        print(f"7. Computing {HTF_OVERLAY_TF} HTF Confluences to overlay on 1m chart...")
        df_es_htf = resample_timeframe(df_es_1m, HTF_OVERLAY_TF)
        df_nq_htf = resample_timeframe(df_nq_1m, HTF_OVERLAY_TF)

        es_htf_swings = detect_swings(df_es_htf)
        nq_htf_swings = detect_swings(df_nq_htf)

        es_ranges = track_dealing_ranges(df_es_htf, es_htf_swings)
        nq_ranges = track_dealing_ranges(df_nq_htf, nq_htf_swings)

        es_fibs = track_fibonacci_ranges(df_es_htf, es_htf_swings)
        nq_fibs = track_fibonacci_ranges(df_nq_htf, nq_htf_swings)

        es_fvgs = track_fvgs(df_es_htf)
        nq_fvgs = track_fvgs(df_nq_htf)

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
    ) = run_pipeline(es_data_path, nq_data_path, timeframe=CHART_TIMEFRAME)

    if ENABLE_PLOTTING:
        print(f"Rendering {CHART_TIMEFRAME} chart overlays...")

        common_index = df_es.index.intersection(df_nq.index)
        if len(common_index) == 0:
            print("CRITICAL: ES and NQ datasets have absolutely no overlapping timeframes.")
            return

        # Center a 120-candle window around the most recent SMT event (or last 120 bars)
        if smt_events:
            target_time = smt_events[-1]['time']
            print(f"Centering 120-bar ({CHART_TIMEFRAME}) window on SMT Event at: {target_time.strftime('%Y-%m-%d %H:%M')}")
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
                in_window = (fib.extreme_time >= t_start and fib.start_time <= t_end)
                if in_window and fib.swing_count >= 2:
                    vis_fibs.append(fib)

            return vis_pools, vis_swings, vis_fvgs, vis_ranges, vis_fibs

        # Scale vertical price padding automatically based on the visible price range
        es_span = max(50.0, (plot_df_es['high'].max() - plot_df_es['low'].min()) * 0.5)
        nq_span = max(100.0, (plot_df_nq['high'].max() - plot_df_nq['low'].min()) * 0.5)

        es_vis_pools, es_vis_swings, es_vis_fvgs, es_vis_ranges, es_vis_fibs = filter_elements(
            es_pools, es_swings, es_fvgs, es_ranges, es_fibs, start_time, end_time,
            plot_df_es['low'].min() - es_span, plot_df_es['high'].max() + es_span
        )

        nq_vis_pools, nq_vis_swings, nq_vis_fvgs, nq_vis_ranges, nq_vis_fibs = filter_elements(
            nq_pools, nq_swings, nq_fvgs, nq_ranges, nq_fibs, start_time, end_time,
            plot_df_nq['low'].min() - nq_span, plot_df_nq['high'].max() + nq_span
        )

        print(f"Plotting ES {CHART_TIMEFRAME} Chart ({len(es_vis_ranges)} Equilibriums, {len(es_vis_fibs)} Fib Ranges)...")
        plot_liquidity_chart(
            plot_df_es, es_vis_pools, es_vis_swings,
            fvgs=[], ranges=[], fibs=[],
            title=f"E-Mini S&P 500 (ES) - {CHART_TIMEFRAME}"
        )

        print(f"Plotting NQ {CHART_TIMEFRAME} Chart ({len(nq_vis_ranges)} Equilibriums, {len(nq_vis_fibs)} Fib Ranges)...")
        plot_liquidity_chart(
            plot_df_nq, nq_vis_pools, nq_vis_swings,
            fvgs=[], ranges=[], fibs=[],
            title=f"E-Mini Nasdaq 100 (NQ) - {CHART_TIMEFRAME}"
        )


if __name__ == "__main__":
    main()