# Copyright (C) 2026 Kyrylo Homberh
# SPDX-License-Identifier: AGPL-3.0-only

import numpy as np
import pandas as pd
from engine.event_bus import BacktestEngine
from strategies.base import BuyAndHoldStrategy, RandomEntryStrategy
from strategies.ict_multi_tf import ICTMultiTimeframeStrategy


def _make_trending_df(periods: int = 60, start_price: float = 5000.0) -> pd.DataFrame:
    times = pd.date_range("2026-10-16 09:30:00", periods=periods, freq="1min", tz="America/New_York")
    closes = [start_price + i * 0.5 for i in range(periods)]
    opens = [c - 0.25 for c in closes]
    highs = [c + 1.0 for c in closes]
    lows = [o - 1.0 for o in opens]
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": [1000] * periods},
        index=times,
    )


def test_buy_and_hold_baseline_and_metrics():
    df = _make_trending_df(periods=30, start_price=5000.0)
    engine = BacktestEngine(symbol="ES", initial_capital=100_000.0, commission_per_contract=2.0, slippage_ticks=0)
    strategy = BuyAndHoldStrategy(quantity=1)

    result = engine.run(df, strategy=strategy, precompute_domain=False)

    assert result.total_trades == 1
    assert result.net_profit > 0
    assert result.win_rate == 100.0
    assert len(result.equity_df) == 30


def test_random_entry_baseline_reproducibility():
    df = _make_trending_df(periods=100, start_price=5000.0)
    engine = BacktestEngine(symbol="ES", initial_capital=100_000.0, commission_per_contract=2.0, slippage_ticks=1)

    strat1 = RandomEntryStrategy(stop_loss_points=5.0, take_profit_points=10.0, entry_probability=0.2, seed=123)
    strat2 = RandomEntryStrategy(stop_loss_points=5.0, take_profit_points=10.0, entry_probability=0.2, seed=123)

    res1 = engine.run(df, strategy=strat1, precompute_domain=False)
    res2 = engine.run(df, strategy=strat2, precompute_domain=False)

    assert res1.total_trades >= 1
    assert res1.total_trades == res2.total_trades
    assert np.isclose(res1.net_profit, res2.net_profit)


def test_ict_multi_tf_strategy_executes_on_ssl_sweep_and_fvg():
    """
    Constructs a 13-candle NY session scenario (09:30 - 09:42) where:
    1. A 5-bar SwingLow forms at 09:32 at 5000.0 (SSL pool) and SwingHigh at 09:34 at 5016.0.
    2. At 09:36, price sweeps below 5000.0 down to 4998.5 and displaces up, creating a BISI FVG [5006 - 5008].
    3. At 09:39, price dips into the BISI FVG (low = 5007.0) and at 09:40 closes at 5018.0 (> 5016.0 SwingHigh),
       triggering a 1m Bullish CHoCH inside the 09:40-10:30 NY window.
    """
    times = pd.date_range("2026-10-16 09:30:00", periods=13, freq="1min", tz="America/New_York")
    data = {
        "open":  [5012, 5007, 5004, 5002, 5009, 5015, 5006, 5001, 5011, 5014, 5012, 5018, 5024],
        "high":  [5014, 5009, 5006, 5010, 5016, 5011, 5006, 5012, 5015, 5014, 5020, 5026, 5035],
        "low":   [5006, 5003, 5000, 5004, 5008, 5005, 4998.5, 5001, 5008, 5007, 5011, 5017, 5023],
        "close": [5007, 5004, 5002, 5009, 5015, 5006, 5001, 5011, 5014, 5012, 5018, 5024, 5032],
        "volume": [500] * 13,
    }
    df = pd.DataFrame(data, index=times)

    engine = BacktestEngine(symbol="ES", initial_capital=100_000.0, commission_per_contract=2.0, slippage_ticks=0)
    strategy = ICTMultiTimeframeStrategy(stop_loss_points=12.0, reward_risk_ratio=2.0, min_grade="A")

    result = engine.run(df, strategy=strategy, precompute_domain=True)

    assert result.total_trades >= 1
    first_trade = result.trades[0]
    assert first_trade.direction == "LONG"
    assert first_trade.entry_time == pd.Timestamp("2026-10-16 09:40:00", tz="America/New_York")
    assert first_trade.metadata.get("ssl_swept") is True
    assert first_trade.metadata.get("bisi_fvg_count", 0) >= 1
    assert first_trade.metadata.get("grade") in ("A", "A/A+", "A+")


def test_setup_grader_assigns_a_plus_vs_b_grade():
    """
    Verifies that ICTMultiTimeframeStrategy.grade_setup accurately scores
    an A+ setup vs. a low-confluence Grade B setup.
    """
    # 1. A+ Setup: Tier-1 London High Target + 5m FVG & EQ + Confirmed SMT + Aligned HTFs + Low Liq Taken
    grade_aplus, score_aplus, pros, _ = ICTMultiTimeframeStrategy.grade_setup(
        direction="LONG",
        target_info={"target_type": "SESSION_LONDON", "tier": 1, "rr": 2.4},
        retracement_type="FVG_AND_EQ",
        retracement_pts=3,
        has_smt=True,
        has_macro_sweep=True,
        has_structural_sweep=True,
        htf_trends={"4h": "BULLISH", "1h": "BULLISH", "15min": "BULLISH"},
        liquidity_taken_level="LOW",
    )
    assert grade_aplus == "A+"
    assert score_aplus >= 9
    assert len(pros) >= 4

    # 2. Grade B Setup: Tier-2 15m target + 1m counter-BOS only + Counter-HTF + High Liq already taken
    grade_b, score_b, _, cons = ICTMultiTimeframeStrategy.grade_setup(
        direction="LONG",
        target_info={"target_type": "15MIN_POOL", "tier": 2, "rr": 1.3},
        retracement_type="1M_COUNTER_BOS_ONLY",
        retracement_pts=1,
        has_smt=False,
        has_macro_sweep=False,
        has_structural_sweep=True,
        htf_trends={"4h": "BEARISH", "1h": "BEARISH", "15min": "BEARISH"},
        liquidity_taken_level="HIGH",
    )
    assert grade_b == "B"
    assert score_b <= 4
    assert len(cons) >= 3

def test_option_a_last_fvg_of_range_inversion_triggers_ifvg_bos():
    """
    Verifies Option A: In an upward leg with 2 BISI FVGs (FVG #1 lower at [5002-5004],
    FVG #2 higher at [5008-5010]), closing below FVG #2 at 5006.0 does NOT trigger a
    Bearish IFVG_BOS because FVG #1 is still holding below. Only when price closes
    below FVG #1 (< 5002.0) does Bearish IFVG_BOS fire.
    """
    from domain.fvg import FairValueGap
    from domain.structure import detect_structure

    times = pd.date_range("2026-10-16 09:30:00", periods=8, freq="1min", tz="America/New_York")
    df = pd.DataFrame(
        {
            "open":  [5001, 5000, 5003, 5007, 5011, 5009, 5006, 5002],
            "high":  [5002, 5002, 5005, 5009, 5012, 5010, 5007, 5003],
            "low":   [5000, 4999, 5001, 5004, 5008, 5005, 5003, 5000],
            "close": [5000, 5001, 5004, 5008, 5011, 5006, 5004, 5001],
        },
        index=times,
    )
    swings = [{"type": "SwingLow", "time": times[1], "price": 4999.0}]
    fvgs = [
        FairValueGap(fvg_type="BISI", top=5004.0, bottom=5002.0, creation_time=times[2]),
        FairValueGap(fvg_type="BISI", top=5010.0, bottom=5008.0, creation_time=times[4]),
    ]

    df_struct = detect_structure(df, swings)
    events = BacktestEngine._extract_structure_events(df_struct, tf_minutes=1, swings=swings, fvgs=fvgs)

    ifvg_events = [e for e in events if e["kind"] == "IFVG_BOS"]
    assert len(ifvg_events) == 1
    # Must NOT fire at times[5] (close=5006 < 5008, when FVG #1 at 5002 is still intact);
    # Must fire at times[7] when close=5001.0 < 5002.0 (inverting the last BISI of the leg)
    assert ifvg_events[0]["close_time"] == times[7]
    assert ifvg_events[0]["direction"] == "BEARISH"
    assert int(df_struct["trend"].iloc[5]) != -1
    assert int(df_struct["trend"].iloc[7]) == -1

def test_trailing_stop_toggle_and_pluggable_manager():
    """
    Verifies Layer 3 isolation:
    - enable_trailing_sl=False or trailing_stop_manager=None keeps SL fixed.
    - enable_trailing_sl=True uses ICTStructuralTrailingStop and trails SL.
    - Custom BaseTrailingStopManager subclasses plug in cleanly.
    """
    from engine.event_bus import BarContext
    from engine.portfolio import Portfolio
    from engine.broker import SimulatedBroker
    from strategies.trailing_stops import BaseTrailingStopManager, ICTStructuralTrailingStop

    t0 = pd.Timestamp("2026-10-16 09:40:00", tz="America/New_York")
    t1 = pd.Timestamp("2026-10-16 09:41:00", tz="America/New_York")
    t2 = pd.Timestamp("2026-10-16 09:42:00", tz="America/New_York")
    hist_df = pd.DataFrame(
        {
            "open":  [5000.0, 5002.0, 5006.0],
            "high":  [5003.0, 5005.0, 5017.0],
            "low":   [4999.0, 5001.0, 5005.0],
            "close": [5002.0, 5004.0, 5015.0],
            "volume": [100, 100, 100],
        },
        index=pd.DatetimeIndex([t0, t1, t2]),
    )
    ctx = BarContext(
        symbol="ES",
        timestamp=t2,
        bar_index=2,
        open=5006.0,
        high=5017.0,
        low=5005.0,
        close=5015.0,
        volume=100.0,
        history_df=hist_df,
        confirmed_swings=[{"type": "SwingLow", "time": t1, "price": 5001.0}],
    )

    def _run_with_strategy(strat):
        pf = Portfolio(initial_capital=100_000.0)
        br = SimulatedBroker(pf, commission_per_contract=0.0, slippage_ticks=0)
        br.submit_order(
            symbol="ES",
            direction="LONG",
            order_type="MARKET",
            quantity=1,
            stop_loss=4990.0,
            take_profit=5020.0,
            timestamp=t0,
            current_close=5000.0,
        )
        strat.on_bar(ctx, br, pf)
        return pf.positions["ES"]

    # 1. Trailing OFF via boolean flag -> SL stays fixed at 4990.0
    strat_off = ICTMultiTimeframeStrategy(enable_trailing_sl=False)
    assert strat_off.trailing_stop_manager is None
    assert _run_with_strategy(strat_off).stop_loss == 4990.0

    # 2. Trailing OFF via explicit trailing_stop_manager=None -> SL stays fixed
    strat_none = ICTMultiTimeframeStrategy(enable_trailing_sl=True, trailing_stop_manager=None)
    assert strat_none.enable_trailing_sl is False
    assert _run_with_strategy(strat_none).stop_loss == 4990.0

    # 3. Trailing ON (default ICTStructuralTrailingStop) -> >=80% progress moves SL to BE swing low - 0.50
    strat_on = ICTMultiTimeframeStrategy(enable_trailing_sl=True)
    assert isinstance(strat_on.trailing_stop_manager, ICTStructuralTrailingStop)
    assert _run_with_strategy(strat_on).stop_loss == 5000.5

    # 4. Pluggable custom trailing manager
    class CustomTrailManager(BaseTrailingStopManager):
        def __init__(self):
            self.calls = 0

        def on_bar_manage(self, bar_ctx, position, broker, portfolio):
            self.calls += 1
            position.update_stop_loss(4995.0, bar_ctx.timestamp, "custom")

    custom_mgr = CustomTrailManager()
    strat_custom = ICTMultiTimeframeStrategy(enable_trailing_sl=True, trailing_stop_manager=custom_mgr)
    pos_custom = _run_with_strategy(strat_custom)
    assert custom_mgr.calls == 1
    assert pos_custom.stop_loss == 4995.0


def test_experience_config_pure_strategy_vs_full_toggles():
    """
    Verifies Layer 2 toggles between ExperienceConfig.full() and ExperienceConfig.pure_strategy().
    """
    from strategies.experience import ExperienceConfig, ICTExperienceLayer

    full_layer = ICTExperienceLayer(ExperienceConfig.full())
    pure_layer = ICTExperienceLayer(ExperienceConfig.pure_strategy())

    # Grading gate
    assert full_layer.passes_grade_filter("B", "A") is False
    assert pure_layer.passes_grade_filter("B", "A") is True

    # Cross-index exclusivity
    d = pd.Timestamp("2026-10-06").date()
    shared = {d: ("ES", pd.Timestamp("2026-10-06 09:40:00", tz="America/New_York"))}
    assert full_layer.get_companion_trade_info(shared, d)[0] is not None
    assert pure_layer.get_companion_trade_info(shared, d) == (None, None)

    # ATH short block
    assert full_layer.check_ath_short_block("SHORT", True) is not None
    assert pure_layer.check_ath_short_block("SHORT", True) is None

    # Symbol SL heuristics (NQ cap vs uncapped pure strategy)
    full_sl = full_layer.get_stop_loss_parameters("NQ", "LONG", "5M_FVG_RETRACE", 3, 42.0, 0.5)
    pure_sl = pure_layer.get_stop_loss_parameters("NQ", "LONG", "5M_FVG_RETRACE", 3, 42.0, 0.5)
    assert full_sl["max_valid_sl"] == 102.0
    assert pure_sl["max_valid_sl"] is None