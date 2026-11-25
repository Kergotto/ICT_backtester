# Copyright (C) 2026 Kyrylo Homberh
# SPDX-License-Identifier: AGPL-3.0-only

import argparse
import inspect
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data_loader import load_databento_pair
from engine.event_bus import BacktestEngine, BacktestResult
from strategies.experience import ExperienceConfig
from strategies.ict_multi_tf import ICTMultiTimeframeStrategy
from main import BACKTEST_START_DATE, BACKTEST_END_DATE, FOCUS_AUDIT_DATES


SNAPSHOT_PATH = PROJECT_ROOT / "tests" / "golden_baseline_snapshot.json"


def _normalize(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "INF" if obj > 0 else "-INF"
        return obj
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.floating,)):
        return _normalize(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_normalize(v) for v in obj)
    return str(obj)


def _serialize_result(res: BacktestResult) -> Dict[str, Any]:
    trades_out: List[Dict[str, Any]] = []
    for t in res.trades:
        trades_out.append({
            "symbol": t.symbol,
            "direction": t.direction,
            "entry_time": _normalize(t.entry_time),
            "exit_time": _normalize(t.exit_time),
            "entry_price": _normalize(t.entry_price),
            "exit_price": _normalize(t.exit_price),
            "quantity": _normalize(t.quantity),
            "stop_loss": _normalize(t.stop_loss),
            "take_profit": _normalize(t.take_profit),
            "exit_reason": t.exit_reason,
            "points_pnl": _normalize(t.points_pnl),
            "gross_pnl": _normalize(t.gross_pnl),
            "commission": _normalize(t.commission),
            "net_pnl": _normalize(t.net_pnl),
            "r_multiple": _normalize(t.r_multiple),
            "metadata": _normalize(t.metadata),
        })

    return {
        "strategy_name": res.strategy_name,
        "symbol": res.symbol,
        "final_equity": _normalize(res.final_equity),
        "net_profit": _normalize(res.net_profit),
        "total_return_pct": _normalize(res.total_return_pct),
        "total_trades": res.total_trades,
        "win_count": res.win_count,
        "be_count": res.be_count,
        "loss_count": res.loss_count,
        "win_rate": _normalize(res.win_rate),
        "profit_factor": _normalize(res.profit_factor),
        "sharpe_ratio": _normalize(res.sharpe_ratio),
        "max_drawdown_pct": _normalize(res.max_drawdown_pct),
        "max_drawdown_cash": _normalize(res.max_drawdown_cash),
        "avg_r_multiple": _normalize(res.avg_r_multiple),
        "trades": trades_out,
    }


def _collect_snapshot(bypass_disk_cache: bool = True) -> Dict[str, Any]:
    t_total = time.perf_counter()

    # Always start from clean in-memory caches
    BacktestEngine._GLOBAL_SYMBOL_CACHE.clear()
    BacktestEngine._GLOBAL_SMT_CACHE.clear()

    df_es_1m, df_nq_1m = load_databento_pair(
        data_dir=PROJECT_ROOT / "data",
        start_date=BACKTEST_START_DATE,
        end_date=BACKTEST_END_DATE,
    )

    es_lb = getattr(df_es_1m, "_htf_lookback_df", None)
    nq_lb = getattr(df_nq_1m, "_htf_lookback_df", None)

    df_es_1m = df_es_1m[(df_es_1m["high"] - df_es_1m["low"]) < 40.0]
    df_nq_1m = df_nq_1m[(df_nq_1m["high"] - df_nq_1m["low"]) < 150.0]

    if es_lb is not None:
        object.__setattr__(df_es_1m, "_htf_lookback_df", es_lb[(es_lb["high"] - es_lb["low"]) < 40.0])
    if nq_lb is not None:
        object.__setattr__(df_nq_1m, "_htf_lookback_df", nq_lb[(nq_lb["high"] - nq_lb["low"]) < 150.0])

    shared_dates_exp_fixed: Dict[Any, Any] = {}
    shared_dates_grade_b: Dict[Any, Any] = {}
    shared_dates_grade_a: Dict[Any, Any] = {}

    engine_init_params = inspect.signature(BacktestEngine.__init__).parameters
    extra_engine_kwargs: Dict[str, Any] = {}
    if "bypass_cache" in engine_init_params:
        extra_engine_kwargs["bypass_cache"] = bypass_disk_cache

    snapshot: Dict[str, Any] = {
        "range": {
            "start_date": BACKTEST_START_DATE,
            "end_date": BACKTEST_END_DATE,
            "es_bars": len(df_es_1m),
            "nq_bars": len(df_nq_1m),
        },
        "symbols": {},
    }

    for sym, df_prim, df_sec, default_sl in (
        ("ES", df_es_1m, df_nq_1m, 5.5),
        ("NQ", df_nq_1m, df_es_1m, 42.0),
    ):
        engine = BacktestEngine(
            symbol=sym,
            initial_capital=100_000.0,
            commission_per_contract=2.00,
            slippage_ticks=1,
            swing_confirmation_bars=2,
            **extra_engine_kwargs,
        )

        ict_pure = ICTMultiTimeframeStrategy(
            stop_loss_points=default_sl,
            reward_risk_ratio=2.0,
            min_rr=1.65,
            min_tp1_rr=0.80,
            max_rr=3.85,
            partial_cut_pct=0.30,
            enable_partials=False,
            enable_trailing_sl=False,
            risk_per_trade_pct=0.01,
            use_dynamic_sizing=True,
            sl_mode="tight_1m",
            min_grade="B",
            require_killzone=True,
            max_trades_per_day=1,
            experience_config=ExperienceConfig.pure_strategy(),
        )
        ict_pure.name = "ICT_PureStrategy (Fixed SL)"
        res_pure = engine.run(df_prim, strategy=ict_pure, df_secondary=df_sec, precompute_domain=True)

        ict_exp_fixed = ICTMultiTimeframeStrategy(
            stop_loss_points=default_sl,
            reward_risk_ratio=2.0,
            min_rr=1.65,
            min_tp1_rr=0.80,
            max_rr=3.85,
            partial_cut_pct=0.30,
            enable_partials=True,
            enable_trailing_sl=False,
            risk_per_trade_pct=0.01,
            use_dynamic_sizing=True,
            sl_mode="tight_1m",
            min_grade="A",
            require_killzone=True,
            max_trades_per_day=1,
            shared_daily_trade_dates=shared_dates_exp_fixed,
            experience_config=ExperienceConfig.full(),
        )
        ict_exp_fixed.name = "ICT_Strategy+Exp (Fixed SL)"
        res_exp_fixed = engine.run(df_prim, strategy=ict_exp_fixed, df_secondary=df_sec, precompute_domain=True)

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
            experience_config=ExperienceConfig.full(),
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
            experience_config=ExperienceConfig.full(),
        )
        ict_a_only.name = "ICT_Strategy (Grade A/A+ Only)"
        res_a = engine.run(df_prim, strategy=ict_a_only, df_secondary=df_sec, precompute_domain=True)

        focus_dates = FOCUS_AUDIT_DATES or []
        focus_summaries = {
            d: _normalize(ict_a_only.daily_window_summary.get(d, {}))
            for d in focus_dates
        }
        focus_diagnostics = {
            d: _normalize(ict_a_only.daily_diagnostics.get(d, []))
            for d in focus_dates
        }

        snapshot["symbols"][sym] = {
            "variants": {
                res_pure.strategy_name: _serialize_result(res_pure),
                res_exp_fixed.strategy_name: _serialize_result(res_exp_fixed),
                res_b.strategy_name: _serialize_result(res_b),
                res_a.strategy_name: _serialize_result(res_a),
            },
            "focus_window_summaries": focus_summaries,
            "focus_diagnostics": focus_diagnostics,
        }

    elapsed = time.perf_counter() - t_total
    print(f"\n⏱️  Total verification backtest wall time: {elapsed:.2f}s (bypass_disk_cache={bypass_disk_cache})")
    return snapshot


def _diff_values(expected: Any, actual: Any, path: str, diffs: List[str], tol: float = 1e-9) -> None:
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)) and not isinstance(expected, bool) and not isinstance(actual, bool):
        if abs(float(expected) - float(actual)) > tol:
            diffs.append(f"{path}: expected {expected!r}, got {actual!r} (delta={actual - expected:+.12g})")
        return

    if type(expected) is not type(actual):
        diffs.append(f"{path}: type mismatch expected {type(expected).__name__} ({expected!r}), got {type(actual).__name__} ({actual!r})")
        return

    if isinstance(expected, dict):
        exp_keys = set(expected.keys())
        act_keys = set(actual.keys())
        for k in sorted(exp_keys - act_keys):
            diffs.append(f"{path}.{k}: missing key (expected {expected[k]!r})")
        for k in sorted(act_keys - exp_keys):
            diffs.append(f"{path}.{k}: unexpected extra key ({actual[k]!r})")
        for k in sorted(exp_keys & act_keys):
            _diff_values(expected[k], actual[k], f"{path}.{k}" if path else str(k), diffs, tol=tol)
        return

    if isinstance(expected, list):
        if len(expected) != len(actual):
            diffs.append(f"{path}: list length mismatch expected {len(expected)}, got {len(actual)}")
        for idx in range(min(len(expected), len(actual))):
            _diff_values(expected[idx], actual[idx], f"{path}[{idx}]", diffs, tol=tol)
        return

    if expected != actual:
        diffs.append(f"{path}: expected {expected!r}, got {actual!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Golden Baseline Snapshot Recorder & Zero-Regression Diff Verifier")
    parser.add_argument("--record", action="store_true", help="Record a new Golden Baseline snapshot JSON.")
    parser.add_argument("--use-cache", action="store_true", help="Allow on-disk domain cache instead of cold bypass.")
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT_PATH, help="Path to snapshot JSON file.")
    args = parser.parse_args()

    current = _collect_snapshot(bypass_disk_cache=not args.use_cache)

    if args.record:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        with open(args.snapshot, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
        print(f"✅ Saved Golden Baseline Snapshot -> {args.snapshot}")
        return 0

    if not args.snapshot.exists():
        print(f"❌ Snapshot not found at {args.snapshot}. Run with --record first.")
        return 1

    with open(args.snapshot, "r", encoding="utf-8") as f:
        golden = json.load(f)

    diffs: List[str] = []
    _diff_values(golden, current, path="", diffs=diffs)

    if not diffs:
        print("✅ ZERO-REGRESSION VERIFICATION: PASS (0 diffs across all trades, PnL, trailing stops, and focus-date diagnostics)")
        return 0

    print(f"❌ ZERO-REGRESSION VERIFICATION: FAIL ({len(diffs)} diffs detected)")
    for line in diffs[:50]:
        print(f"   - {line}")
    if len(diffs) > 50:
        print(f"   ... and {len(diffs) - 50} more diffs.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())