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