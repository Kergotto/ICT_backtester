from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
import numpy as np
import pandas as pd

from core.data_loader import load_historical_data, load_databento_pair
from engine.event_bus import BacktestEngine, BacktestResult
from strategies.base import BuyAndHoldStrategy, RandomEntryStrategy
from strategies.ict_multi_tf import ICTMultiTimeframeStrategy

# ==============================================================================
# 📅 BACKTEST DATE RANGE CONFIGURATION
# ==============================================================================
USE_DATABENTO_5Y: bool = True
BACKTEST_START_DATE: Optional[str] = "2026-09-29"
BACKTEST_END_DATE: Optional[str] = "2026-10-27 17:00:00"

# Focus dates from your manual TradingView journal
FOCUS_AUDIT_DATES: Optional[List[str]] = [
    "2026-10-01",
    "2026-10-02",
    "2026-10-06",
    "2026-10-08",
    "2026-10-13",
    "2026-10-14",
    "2026-10-16",
    "2026-10-21",
    "2026-10-26",
]


def generate_synthetic_pair_data(
    es_path: Path, nq_path: Path, num_bars: int = 1500
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(42)
    idx = pd.date_range("2025-01-02 08:00:00", periods=num_bars, freq="1min", tz="America/New_York")

    es_trend = np.sin(np.linspace(0, 12 * np.pi, num_bars)) * 15.0
    es_noise = np.random.normal(0, 0.75, num_bars).cumsum()
    es_close = np.round((5000.0 + es_trend + es_noise) * 4) / 4
    es_open = np.roll(es_close, 1)
    es_open[0] = es_close[0]
    es_high = np.maximum(es_open, es_close) + np.round(np.abs(np.random.normal(0, 1.0, num_bars)) * 4) / 4
    es_low = np.minimum(es_open, es_close) - np.round(np.abs(np.random.normal(0, 1.0, num_bars)) * 4) / 4
    es_vol = np.random.randint(100, 5000, size=num_bars).astype(float)

    df_es = pd.DataFrame(
        {"open": es_open, "high": es_high, "low": es_low, "close": es_close, "volume": es_vol},
        index=idx,
    )
    df_es.index.name = "datetime"

    nq_trend = np.sin(np.linspace(0, 12 * np.pi, num_bars)) * 60.0
    nq_noise = np.random.normal(0, 2.5, num_bars).cumsum()
    nq_close = np.round((18000.0 + nq_trend + nq_noise) * 4) / 4
    nq_open = np.roll(nq_close, 1)
    nq_open[0] = nq_close[0]
    nq_high = np.maximum(nq_open, nq_close) + np.round(np.abs(np.random.normal(0, 3.0, num_bars)) * 4) / 4
    nq_low = np.minimum(nq_open, nq_close) - np.round(np.abs(np.random.normal(0, 3.0, num_bars)) * 4) / 4
    nq_vol = np.random.randint(100, 5000, size=num_bars).astype(float)

    df_nq = pd.DataFrame(
        {"open": nq_open, "high": nq_high, "low": nq_low, "close": nq_close, "volume": nq_vol},
        index=idx,
    )
    df_nq.index.name = "datetime"

    es_path.parent.mkdir(parents=True, exist_ok=True)
    df_es.to_csv(es_path)
    df_nq.to_csv(nq_path)
    return df_es, df_nq


def _print_tradingview_audit_log(
    result: BacktestResult,
    strategy_obj: ICTMultiTimeframeStrategy,
    focus_dates: Optional[List[str]] = None,
) -> None:
    trades_by_date: Dict[str, List[Any]] = {}
    for t in result.trades:
        d_str = t.entry_time.strftime("%Y-%m-%d")
        trades_by_date.setdefault(d_str, []).append(t)

    if focus_dates:
        dates_to_show = focus_dates
        focus_trade_count = sum(len(trades_by_date.get(d, [])) for d in focus_dates)
        header_sub = (
            f"Total Trades in Sample: {len(result.trades)} | "
            f"Executed Trades on Focus Dates {focus_dates}: {focus_trade_count}"
        )
    else:
        dates_to_show = sorted(trades_by_date.keys())
        header_sub = f"Showing All {len(result.trades)} Executed Trades:"

    print("\n" + "#" * 105)
    print(f"📋 TRADINGVIEW TRADE-BY-TRADE AUDIT LOG — {result.symbol} ({result.strategy_name})")
    print("#" * 105)
    print(header_sub)

    trade_counter = 0
    point_val = 50.0 if result.symbol.upper() == "ES" else 20.0

    for d_str in dates_to_show:
        day_trades = trades_by_date.get(d_str, [])
        if day_trades:
            for t in day_trades:
                trade_counter += 1
                meta = t.metadata or {}
                outcome_label = meta.get(
                    "outcome",
                    "BE" if t.exit_reason == "BE_TRAILED_SL" else ("WIN" if t.net_pnl > 0 else "LOSS"),
                )
                if outcome_label == "WIN":
                    icon = "✅"
                elif outcome_label == "BE":
                    icon = "🟡"
                else:
                    icon = "❌"

                grade = meta.get("grade", "N/A")
                score = meta.get("score", "N/A")
                target_type = meta.get("target_type", "FIXED_RR")
                target_tier = meta.get("target_tier", "N/A")
                planned_rr = meta.get("planned_rr", t.r_multiple)
                partial_targets = meta.get("partial_targets", [])
                mgmt_summary = meta.get("management_summary", "")
                ret_type = meta.get("retracement_type", "N/A")
                ret_det = meta.get("retracement_detail", "N/A")
                trigger_det = meta.get("entry_trigger_detail", "N/A")
                sl_anchor_det = meta.get("sl_anchor_detail", "")
                liq_lvl = meta.get("liquidity_already_taken", "LOW")
                taken_pools = meta.get("session_taken_pools", [])
                htf = meta.get("htf_trends", {})
                swept_det = meta.get("swept_pools_detail", [])
                smt_det = meta.get("smt_detail", [])
                pros = meta.get("pros", [])
                cons = meta.get("cons", [])

                taken_pools_str = ", ".join(taken_pools) if taken_pools else "None taken since 09:30 NY open"
                sl_anchor_tag = f" ({sl_anchor_det})" if sl_anchor_det else ""

                risk_pts = abs(t.entry_price - t.stop_loss)
                dollar_risk = risk_pts * point_val * float(t.quantity)
                risk_pct = (dollar_risk / result.initial_capital) * 100.0
                qty_label = meta.get("display_qty", f"{t.quantity} {t.symbol}")

                if partial_targets:
                    pt_parts = []
                    used_w = 0.0
                    for pt in partial_targets:
                        w_pct = int(round(pt["weight"] * 100))
                        used_w += pt["weight"]
                        pt_parts.append(
                            f"{pt['label']} ({w_pct}%): {pt['price']:.2f} ({pt['rr']:.2f}R, {pt['target_type']})"
                        )
                    rem_pct = max(0, int(round((1.0 - used_w) * 100)))
                    pt_parts.append(f"FullTP ({rem_pct}%): {t.take_profit:.2f} ({planned_rr:.2f}R, {target_type})")
                    target_line_str = f"    • Targets (DOL)   : {' | '.join(pt_parts)}\n"
                else:
                    target_line_str = f"    • Target (DOL)    : {target_type} (Tier {target_tier})\n"

                mgmt_line = f"    • Position Mgmt   : {mgmt_summary}\n" if mgmt_summary else ""

                print(
                    f"\n  {icon} [{d_str} — Trade #{trade_counter}] {t.symbol} {t.direction} | "
                    f"Result: {outcome_label} | Grade: {grade} (Score: {score}/10)\n"
                    f"    • Entry Time (NY) : {t.entry_time.strftime('%Y-%m-%d %H:%M')}  --> Check this 1m bar on TradingView\n"
                    f"    • Exit Time (NY)  : {t.exit_time.strftime('%Y-%m-%d %H:%M')}  ({t.exit_reason})\n"
                    f"    • Prices          : Entry = {t.entry_price:.2f} | SL = {t.stop_loss:.2f}{sl_anchor_tag} | "
                    f"TP = {t.take_profit:.2f} | Exit = {t.exit_price:.2f}\n"
                    f"    • Risk / Reward   : Risk = ${dollar_risk:.2f} ({risk_pct:.2f}% | {risk_pts:.2f} pts) | "
                    f"Planned FullTP RR = {planned_rr:.2f}R | Realized = {t.r_multiple:+.2f}R | "
                    f"Net PnL = ${t.net_pnl:+.2f} (Qty: {qty_label})\n"
                    f"{target_line_str}"
                    f"{mgmt_line}"
                    f"    • 1m Entry Trigger: {trigger_det}\n"
                    f"    • Retracement     : {ret_type} -> {ret_det}\n"
                    f"    • Market Context  : 4H={htf.get('4h')}, 1H={htf.get('1h')}, "
                    f"15m={htf.get('15min')}, 5m={htf.get('5min')}, 1m={htf.get('1min')} (Locked at Entry)\n"
                    f"    • Liq Exhaustion  : {liq_lvl} ({taken_pools_str})"
                )
                if swept_det:
                    print(f"    • Swept Liquidity : {'; '.join(swept_det)}")
                if smt_det:
                    print(f"    • Active SMT      : {'; '.join(smt_det)}")
                if pros:
                    print(f"    • Pros (+)        : {'; '.join(pros)}")
                if cons:
                    print(f"    • Cons (-)        : {'; '.join(cons)}")
        else:
            win_sum = strategy_obj.daily_window_summary.get(d_str, {})
            diags = strategy_obj.daily_diagnostics.get(d_str, [])
            htf = win_sum.get("htf_trends", {})
            ctx_time = win_sum.get("context_time", "09:55")
            liq_lvl = win_sum.get("liquidity_already_taken", "N/A")
            taken_pools = win_sum.get("session_taken_pools", [])
            smts = win_sum.get("smt_descriptions", [])
            taken_pools_str = ", ".join(taken_pools) if taken_pools else "None taken since 09:30 NY open"

            print(
                f"\n  ⚪ [{d_str} — NO TRADE EXECUTED ON {result.symbol}] Diagnostic Breakdown (09:40 - 10:30 NY):\n"
                f"    • Window Context  : 4H={htf.get('4h', 'N/A')}, 1H={htf.get('1h', 'N/A')}, "
                f"15m={htf.get('15min', 'N/A')}, 5m={htf.get('5min', 'N/A')} (Locked at {ctx_time})\n"
                f"    • Liq Exhaustion  : {liq_lvl} ({taken_pools_str})\n"
                f"    • Active SMTs     : {'; '.join(smts) if smts else 'None confirmed during window'}"
            )
            if diags:
                print("    • Why Not Taken   :")
                for reason in diags[:20]:
                    print(f"        - {reason}")
            else:
                print("    • Why Not Taken   : No 1m BOS/CHoCH occurred in a valid direction during 09:40 - 10:30.")
    print("#" * 105 + "\n")


def run_backtest_suite(es_path: Path, nq_path: Path) -> None:
    print("1. Loading 1-minute historical data (ES & NQ)...")
    if USE_DATABENTO_5Y:
        df_es_1m, df_nq_1m = load_databento_pair(
            data_dir=es_path.parent,
            start_date=BACKTEST_START_DATE,
            end_date=BACKTEST_END_DATE,
        )
    else:
        df_es_1m = load_historical_data(es_path, start_date=BACKTEST_START_DATE, end_date=BACKTEST_END_DATE)
        df_nq_1m = load_historical_data(nq_path, start_date=BACKTEST_START_DATE, end_date=BACKTEST_END_DATE)

    es_lb = getattr(df_es_1m, "_htf_lookback_df", None)
    nq_lb = getattr(df_nq_1m, "_htf_lookback_df", None)

    df_es_1m = df_es_1m[(df_es_1m["high"] - df_es_1m["low"]) < 40.0]
    df_nq_1m = df_nq_1m[(df_nq_1m["high"] - df_nq_1m["low"]) < 150.0]

    if es_lb is not None:
        object.__setattr__(df_es_1m, "_htf_lookback_df", es_lb[(es_lb["high"] - es_lb["low"]) < 40.0])
    if nq_lb is not None:
        object.__setattr__(df_nq_1m, "_htf_lookback_df", nq_lb[(nq_lb["high"] - nq_lb["low"]) < 150.0])

    print(
        f"   ES Range: {df_es_1m.index[0]} -> {df_es_1m.index[-1]} ({len(df_es_1m)} bars)\n"
        f"   NQ Range: {df_nq_1m.index[0]} -> {df_nq_1m.index[-1]} ({len(df_nq_1m)} bars)"
    )

    shared_dates_grade_b: Dict[Any, Any] = {}
    shared_dates_grade_a: Dict[Any, Any] = {}

    all_results: List[BacktestResult] = []
    for sym, df_prim, df_sec, default_sl in (
        ("ES", df_es_1m, df_nq_1m, 5.5),
        ("NQ", df_nq_1m, df_es_1m, 42.0),
    ):
        print(f"\n2. Running Backtest Suite for {sym}...")
        engine = BacktestEngine(
            symbol=sym,
            initial_capital=100_000.0,
            commission_per_contract=2.00,
            slippage_ticks=1,
            swing_confirmation_bars=2,
        )

        res_bh = engine.run(df_prim, strategy=BuyAndHoldStrategy(quantity=1), precompute_domain=False)
        res_rand = engine.run(
            df_prim,
            strategy=RandomEntryStrategy(
                stop_loss_points=default_sl,
                take_profit_points=default_sl * 2.0,
                quantity=1,
                seed=42,
            ),
            precompute_domain=False,
        )

        ict_all_grades = ICTMultiTimeframeStrategy(
            stop_loss_points=default_sl,
            reward_risk_ratio=2.0,
            min_rr=1.65,
            min_tp1_rr=0.80,
            max_rr=3.85,
            partial_cut_pct=0.30,
            enable_partials=True,
            enable_trailing_sl=True,
            risk_per_trade_pct=0.01,
            use_dynamic_sizing=True,
            sl_mode="tight_1m",
            min_grade="B",
            require_killzone=True,
            max_trades_per_day=1,
            shared_daily_trade_dates=shared_dates_grade_b,
        )
        ict_all_grades.name = "ICT_Strategy (Allow Grade B)"
        res_b = engine.run(df_prim, strategy=ict_all_grades, df_secondary=df_sec, precompute_domain=True)

        ict_a_only = ICTMultiTimeframeStrategy(
            stop_loss_points=default_sl,
            reward_risk_ratio=2.0,
            min_rr=1.65,
            min_tp1_rr=0.80,
            max_rr=3.85,
            partial_cut_pct=0.30,
            enable_partials=True,
            enable_trailing_sl=True,
            risk_per_trade_pct=0.01,
            use_dynamic_sizing=True,
            sl_mode="tight_1m",
            min_grade="A",
            require_killzone=True,
            max_trades_per_day=1,
            shared_daily_trade_dates=shared_dates_grade_a,
        )
        ict_a_only.name = "ICT_Strategy (Grade A/A+ Only)"
        res_a = engine.run(df_prim, strategy=ict_a_only, df_secondary=df_sec, precompute_domain=True)

        _print_tradingview_audit_log(res_a, ict_a_only, focus_dates=FOCUS_AUDIT_DATES)
        all_results.extend([res_bh, res_rand, res_b, res_a])

    print("=" * 114)
    print(
        f"{'STRATEGY NAME':<34} | {'SYM':<3} | {'TRADES':>6} | {'W/BE/L':>7} | {'WIN %':>6} | "
        f"{'NET PnL ($)':>12} | {'PF':>5} | {'SHARPE':>6} | {'MAX DD %':>8} | {'AVG R':>6}"
    )
    print("-" * 114)
    for r in all_results:
        pf_str = f"{r.profit_factor:.2f}" if r.profit_factor != float("inf") else "INF"
        w_be_l = f"{r.win_count}/{r.be_count}/{r.loss_count}"
        print(
            f"{r.strategy_name:<34} | {r.symbol:<3} | {r.total_trades:>6} | {w_be_l:>7} | "
            f"{r.win_rate:>5.1f}% | {r.net_profit:>+12.2f} | {pf_str:>5} | "
            f"{r.sharpe_ratio:>6.2f} | {r.max_drawdown_pct:>7.2f}% | {r.avg_r_multiple:>+5.2f}R"
        )
    print("=" * 114)


def main() -> None:
    data_dir = Path("data")
    es_data_path = data_dir / "es_1m_sample.csv"
    nq_data_path = data_dir / "nq_1m_sample.csv"

    if not USE_DATABENTO_5Y and (not es_data_path.exists() or not nq_data_path.exists()):
        generate_synthetic_pair_data(es_data_path, nq_data_path, num_bars=1500)

    run_backtest_suite(es_data_path, nq_data_path)


if __name__ == "__main__":
    main()