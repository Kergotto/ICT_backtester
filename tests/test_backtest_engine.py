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
    # Same seed must produce identical trade count and net profit
    assert res1.total_trades == res2.total_trades
    assert np.isclose(res1.net_profit, res2.net_profit)


def test_ict_multi_tf_strategy_executes_on_ssl_sweep_and_fvg():
    """
    Constructs a 12-candle NY session scenario where:
    1. A SwingLow forms at 5000.0 (SSL pool).
    2. Price sweeps below 5000.0 down to 4998.5 and displaces up, creating a BISI FVG.
    3. ICTMultiTimeframeStrategy detects the confirmed SSL sweep + BISI FVG and enters LONG.
    """
    times = pd.date_range("2026-10-16 09:30:00", periods=12, freq="1min", tz="America/New_York")
    data = {
        "open":  [5010, 5005, 5002, 5008, 5012, 5015, 5010, 4999, 5004, 5008, 5015, 5022],
        "high":  [5012, 5008, 5009, 5014, 5018, 5017, 5012, 5005, 5012, 5016, 5025, 5028],
        "low":   [5004, 5000, 5001, 5006, 5010, 5009, 4998.5, 4999, 5003, 5007, 5014, 5020],
        "close": [5005, 5002, 5008, 5012, 5016, 5010, 5001, 5004, 5010, 5015, 5024, 5026],
        "volume": [500] * 12,
    }
    df = pd.DataFrame(data, index=times)

    engine = BacktestEngine(symbol="ES", initial_capital=100_000.0, commission_per_contract=2.0, slippage_ticks=0)
    strategy = ICTMultiTimeframeStrategy(stop_loss_points=8.0, reward_risk_ratio=2.0)

    result = engine.run(df, strategy=strategy, precompute_domain=True)

    assert result.total_trades >= 1
    first_trade = result.trades[0]
    assert first_trade.direction == "LONG"
    assert first_trade.metadata.get("ssl_swept") is True
    assert first_trade.metadata.get("bisi_fvg_count", 0) >= 1