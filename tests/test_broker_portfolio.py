import pandas as pd
from datetime import timedelta
from engine.portfolio import Portfolio
from engine.broker import SimulatedBroker


def test_es_long_take_profit_and_accounting():
    base_time = pd.Timestamp("2026-10-16 09:30:00")
    portfolio = Portfolio(initial_capital=100_000.0)
    broker = SimulatedBroker(portfolio, commission_per_contract=2.0, slippage_ticks=0)

    # Enter 1 ES contract LONG at 5000.0, SL = 4990.0 (-10 pts), TP = 5020.0 (+20 pts -> 2.0R)
    broker.submit_order(
        symbol="ES",
        direction="LONG",
        order_type="MARKET",
        quantity=1,
        stop_loss=4990.0,
        take_profit=5020.0,
        timestamp=base_time,
        current_close=5000.0,
        metadata={"setup": "FVG_BISI"},
    )

    assert portfolio.has_open_position("ES") is True

    # Next candle (09:31) rallies to 5022.0 and hits Take-Profit (5020.0)
    closed = broker.process_candle(
        symbol="ES",
        timestamp=base_time + timedelta(minutes=1),
        open_p=5001.0,
        high_p=5022.0,
        low_p=4999.0,
        close_p=5018.0,
    )

    assert len(closed) == 1
    trade = closed[0]
    assert trade.exit_reason == "TAKE_PROFIT"
    assert trade.points_pnl == 20.0
    # ES is $50/point -> 20 points = $1,000 gross PnL, minus $4 round-turn commission = $996 net
    assert trade.gross_pnl == 1000.0
    assert trade.commission == 4.0
    assert trade.net_pnl == 996.0
    assert trade.r_multiple == 2.0
    assert portfolio.cash == 100_996.0


def test_conservative_intrabar_sl_priority_over_tp():
    """
    If a single 1-minute candle touches BOTH Stop-Loss and Take-Profit,
    the broker must conservatively trigger STOP_LOSS first.
    """
    base_time = pd.Timestamp("2026-10-16 10:00:00")
    portfolio = Portfolio(initial_capital=100_000.0)
    broker = SimulatedBroker(portfolio, commission_per_contract=2.0, slippage_ticks=0)

    broker.submit_order(
        symbol="NQ",
        direction="SHORT",
        order_type="MARKET",
        quantity=1,
        stop_loss=20020.0,
        take_profit=19960.0,
        timestamp=base_time,
        current_close=20000.0,
    )

    # Candle at 10:01 spans from 19950.0 (below TP) to 20025.0 (above SL)
    closed = broker.process_candle(
        symbol="NQ",
        timestamp=base_time + timedelta(minutes=1),
        open_p=20000.0,
        high_p=20025.0,
        low_p=19950.0,
        close_p=19980.0,
    )

    assert len(closed) == 1
    trade = closed[0]
    assert trade.exit_reason == "STOP_LOSS"
    assert trade.points_pnl == -20.0
    # NQ is $20/point -> -20 points = -$400 gross, -$404 net
    assert trade.gross_pnl == -400.0
    assert trade.net_pnl == -404.0


def test_zero_lookahead_on_entry_candle():
    """
    Verifies that a position opened at candle t's close cannot be stopped out
    or take profit using candle t's earlier high/low.
    """
    base_time = pd.Timestamp("2026-10-16 10:00:00")
    portfolio = Portfolio(initial_capital=100_000.0)
    broker = SimulatedBroker(portfolio, commission_per_contract=0.0, slippage_ticks=0)

    broker.submit_order(
        symbol="ES",
        direction="LONG",
        order_type="MARKET",
        quantity=1,
        stop_loss=4990.0,
        take_profit=5010.0,
        timestamp=base_time,
        current_close=5000.0,
    )

    # Calling process_candle with the SAME timestamp (base_time) must not close the trade
    closed_same_bar = broker.process_candle(
        symbol="ES",
        timestamp=base_time,
        open_p=5000.0,
        high_p=5015.0,
        low_p=4985.0,
        close_p=5000.0,
    )
    assert len(closed_same_bar) == 0
    assert portfolio.has_open_position("ES") is True