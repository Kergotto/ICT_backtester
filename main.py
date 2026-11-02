from pathlib import Path
from typing import List, Optional
from core.data_loader import load_historical_data, resample_timeframe
from domain.equilibrium import track_dealing_ranges
from domain.fibonacci import track_fibonacci_ranges
from domain.fvg import track_fvgs
from domain.liquidity import track_liquidity, GapEngine
from domain.smt import detect_smt_divergence
from domain.structure import detect_structure
from domain.swings import detect_swings
from engine.event_bus import BacktestEngine, BacktestResult
from engine.session_engine import extract_session_liquidity
from strategies.base import BuyAndHoldStrategy, RandomEntryStrategy
from strategies.ict_multi_tf import ICTMultiTimeframeStrategy
from visualization.plotter import plot_liquidity_chart

# --- EXECUTION MODES ---
RUN_BACKTEST_AUDIT = True
ENABLE_PLOTTING = False

# Focus dates for TradingView inspection (matches Trades #74 and #75 in your journal)
# Set to None to print detailed cards for ALL dates in the sample
FOCUS_AUDIT_DATES: Optional[List[str]] = ["2026-10-13", "2026-10-14", "2026-10-16"]

# --- HTF CHART VERIFICATION SETTINGS (Used when ENABLE_PLOTTING = True) ---
CHART_TIMEFRAME = "15min"
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


def print_comparative_table(results: List[BacktestResult]) -> None:
    print("\n" + "=" * 105)
    print(f"{'STRATEGY NAME':<34} | {'SYM':<3} | {'TRADES':>6} | {'WIN %':>6} | {'NET PnL ($)':>12} | {'PF':>5} | {'SHARPE':>6} | {'MAX DD %':>8} | {'AVG R':>6}")
    print("-" * 105)
    for r in results:
        pf_str = f"{r.profit_factor:.2f}" if r.profit_factor != float("inf") else "  INF"
        print(
            f"{r.strategy_name:<34} | {r.symbol:<3} | {r.total_trades:>6} | "
            f"{r.win_rate:>5.1f}% | {r.net_profit:>+12.2f} | {pf_str:>5} | "
            f"{r.sharpe_ratio:>6.2f} | {r.max_drawdown_pct:>7.2f}% | {r.avg_r_multiple:>+5.2f}R"
        )
    print("=" * 105)


def print_tradingview_audit(
    symbol: str,
    result: BacktestResult,
    strategy: ICTMultiTimeframeStrategy,
    focus_dates: Optional[List[str]] = None,
) -> None:
    print(f"\n{'#' * 105}")
    print(f"📋 TRADINGVIEW TRADE-BY-TRADE AUDIT LOG — {symbol} ({result.strategy_name})")
    print(f"{'#' * 105}")

    trades = result.trades
    if focus_dates:
        focused_trades = [t for t in trades if t.entry_time.strftime("%Y-%m-%d") in focus_dates]
        print(f"Total Trades in Sample: {len(trades)} | Trades on Focus Dates {focus_dates}: {len(focused_trades)}")
        display_trades = focused_trades if focused_trades else trades
    else:
        display_trades = trades

    if not display_trades:
        print("   No trades executed in this window.")
    else:
        for idx, tr in enumerate(display_trades, start=1):
            meta = tr.metadata or {}
            htf = meta.get("htf_trends", {})
            htf_str = (
                f"4H={htf.get('4h', '?')}, 1H={htf.get('1h', '?')}, "
                f"15m={htf.get('15min', '?')}, 5m={htf.get('5min', '?')}, 1m={htf.get('1min', '?')}"
            )
            print(
                f"\n  [Trade #{idx}] {symbol} {tr.direction} | Grade: {meta.get('grade', '?')} (Score: {meta.get('score', '?')}/10)"
            )
            print(
                f"    • Entry Time (NY) : {tr.entry_time.strftime('%Y-%m-%d %H:%M')}  --> Check this 1m bar on TradingView"
            )
            print(
                f"    • Exit Time (NY)  : {tr.exit_time.strftime('%Y-%m-%d %H:%M')}  ({tr.exit_reason})"
            )
            print(
                f"    • Prices          : Entry = {tr.entry_price:.2f} | SL = {tr.stop_loss:.2f} | TP = {tr.take_profit:.2f} | Exit = {tr.exit_price:.2f}"
            )
            print(
                f"    • Risk / Reward   : Planned RR = {meta.get('planned_rr', 0.0):.2f}R | Realized = {tr.r_multiple:+.2f}R | Net PnL = ${tr.net_pnl:+.2f} (Qty: {tr.quantity})"
            )
            print(
                f"    • Target (DOL)    : {meta.get('target_type', '?')} (Tier {meta.get('target_tier', '?')})"
            )
            print(
                f"    • 1m Entry Trigger: {meta.get('entry_trigger_detail', '?')}"
            )
            print(
                f"    • Retracement     : {meta.get('retracement_type', '?')} -> {meta.get('retracement_detail', '?')}"
            )
            print(
                f"    • Market Context  : {htf_str} (Locked at Entry)"
            )
            taken_list = meta.get("session_taken_pools", [])
            taken_str = ", ".join(taken_list) if taken_list else "None taken since 09:30 NY open"
            print(
                f"    • Liq Exhaustion  : {meta.get('liquidity_already_taken', '?')} ({taken_str})"
            )
            if meta.get("swept_pools_detail"):
                print(f"    • Swept Liquidity : {'; '.join(meta['swept_pools_detail'])}")
            if meta.get("smt_detail"):
                print(f"    • Active SMT      : {'; '.join(meta['smt_detail'])}")
            if meta.get("pros"):
                print(f"    • Pros (+)        : {'; '.join(meta['pros'])}")
            if meta.get("cons"):
                print(f"    • Cons (-)        : {'; '.join(meta['cons'])}")

    if strategy.skipped_setups:
        print(f"\n  🚫 SKIPPED LOWER-GRADE SETUPS ({len(strategy.skipped_setups)} filtered out by min_grade='{strategy.min_grade}'):")
        for sk in strategy.skipped_setups[:10]:
            ts_str = sk["timestamp"].strftime("%Y-%m-%d %H:%M")
            print(
                f"    - [{ts_str}] {symbol} {sk['direction']} @ {sk['price']:.2f} | "
                f"Grade={sk.get('grade')} (Score={sk.get('score')}) | "
                f"Trigger={sk.get('entry_trigger_detail')} | "
                f"Target={sk.get('target_type')} | Cons: {', '.join(sk.get('cons', []))}"
            )


def run_backtest_suite(es_path: Path, nq_path: Path) -> None:
    print("1. Loading 1-minute historical data (ES & NQ)...")
    df_es_1m = load_historical_data(es_path)
    df_es_1m = df_es_1m[(df_es_1m["high"] - df_es_1m["low"]) < 40.0]

    df_nq_1m = load_historical_data(nq_path)
    df_nq_1m = df_nq_1m[(df_nq_1m["high"] - df_nq_1m["low"]) < 150.0]

    print(
        f"   ES Range: {df_es_1m.index[0]} -> {df_es_1m.index[-1]} ({len(df_es_1m)} bars)\n"
        f"   NQ Range: {df_nq_1m.index[0]} -> {df_nq_1m.index[-1]} ({len(df_nq_1m)} bars)"
    )

    all_results: List[BacktestResult] = []

    for sym, df_prim, df_sec, default_sl in (
        ("ES", df_es_1m, df_nq_1m, 8.0),
        ("NQ", df_nq_1m, df_es_1m, 35.0),
    ):
        print(f"\n2. Running Backtest Suite for {sym}...")
        engine = BacktestEngine(
            symbol=sym,
            initial_capital=100_000.0,
            commission_per_contract=2.00,
            slippage_ticks=1,
        )

        # Baseline 1: Buy and Hold
        bh_strat = BuyAndHoldStrategy(quantity=1)
        bh_res = engine.run(df_prim, strategy=bh_strat, precompute_domain=False)
        all_results.append(bh_res)

        # Baseline 2: Random Entry with matching SL/TP risk geometry
        rnd_strat = RandomEntryStrategy(
            stop_loss_points=default_sl,
            take_profit_points=default_sl * 2.0,
            entry_probability=0.005,
            quantity=1,
            seed=42,
        )
        rnd_res = engine.run(df_prim, strategy=rnd_strat, precompute_domain=False)
        all_results.append(rnd_res)

        # Strategy Variant A: ICT Strategy allowing Grade B setups (Unfiltered)
        ict_all_grades = ICTMultiTimeframeStrategy(
            stop_loss_points=default_sl,
            reward_risk_ratio=2.0,
            min_rr=1.15,
            risk_per_trade_pct=0.01,
            use_dynamic_sizing=True,
            sl_mode="tight_1m",
            min_grade="B",
        )
        ict_all_grades.name = "ICT_Strategy (Allow Grade B)"
        res_b = engine.run(df_prim, strategy=ict_all_grades, df_secondary=df_sec, precompute_domain=True)
        all_results.append(res_b)

        # Strategy Variant B: ICT Strategy strictly filtering for Grade A / A+ only
        ict_grade_a = ICTMultiTimeframeStrategy(
            stop_loss_points=default_sl,
            reward_risk_ratio=2.0,
            min_rr=1.15,
            risk_per_trade_pct=0.01,
            use_dynamic_sizing=True,
            sl_mode="tight_1m",
            min_grade="A",
        )
        ict_grade_a.name = "ICT_Strategy (Grade A/A+ Only)"
        res_a = engine.run(df_prim, strategy=ict_grade_a, df_secondary=df_sec, precompute_domain=True)
        all_results.append(res_a)

        print_tradingview_audit(sym, res_a, ict_grade_a, focus_dates=FOCUS_AUDIT_DATES)

    print_comparative_table(all_results)


def run_pipeline(primary_path: Path, secondary_path: Path, timeframe: str = "1min"):
    print("1. Loading 1-minute historical data (ES & NQ)...")
    df_es_1m = load_historical_data(primary_path)
    df_es_1m = df_es_1m[(df_es_1m['high'] - df_es_1m['low']) < 40.0]

    df_nq_1m = load_historical_data(secondary_path)
    df_nq_1m = df_nq_1m[(df_nq_1m['high'] - df_nq_1m['low']) < 150.0]

    print("2. Extracting Macro Session Liquidity...")
    es_macro_by_time = group_macros(extract_session_liquidity(df_es_1m))
    nq_macro_by_time = group_macros(extract_session_liquidity(df_nq_1m))

    if timeframe != "1min":
        print(f"   Resampling 1m data to Higher Timeframe ({timeframe})...")
        df_es = resample_timeframe(df_es_1m, timeframe)
        df_nq = resample_timeframe(df_nq_1m, timeframe)
        smt_window = 60
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
    smt_events = detect_smt_divergence(
        es_swings,
        nq_swings,
        df_es,
        df_nq,
        time_window_minutes=smt_window,
        timeframe=timeframe,
        df_es_1m=df_es_1m,
        df_nq_1m=df_nq_1m,
    )
    print(f"--> Found {len(smt_events)} SMT Events on {timeframe}.")
    for smt in smt_events[-5:]:
        print(f"    {smt['description']}")

    es_pools = es_liq_engine.get_all_pools()
    nq_pools = nq_liq_engine.get_all_pools()

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

    if RUN_BACKTEST_AUDIT:
        run_backtest_suite(es_data_path, nq_data_path)

    if ENABLE_PLOTTING:
        (
            df_es, es_swings, es_pools, es_fvgs, es_ranges, es_fibs,
            df_nq, nq_swings, nq_pools, nq_fvgs, nq_ranges, nq_fibs,
            gap_engine, smt_events
        ) = run_pipeline(es_data_path, nq_data_path, timeframe=CHART_TIMEFRAME)

        print(f"Rendering {CHART_TIMEFRAME} chart overlays...")
        common_index = df_es.index.intersection(df_nq.index)
        if len(common_index) == 0:
            print("CRITICAL: ES and NQ datasets have absolutely no overlapping timeframes.")
            return

        if smt_events:
            target_time = smt_events[-1]['time']
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

        plot_liquidity_chart(
            plot_df_es, es_pools, es_swings,
            fvgs=[], ranges=[], fibs=[],
            title=f"E-Mini S&P 500 (ES) - {CHART_TIMEFRAME}"
        )
        plot_liquidity_chart(
            plot_df_nq, nq_pools, nq_swings,
            fvgs=[], ranges=[], fibs=[],
            title=f"E-Mini Nasdaq 100 (NQ) - {CHART_TIMEFRAME}"
        )


if __name__ == "__main__":
    main()