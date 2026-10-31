from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any
import numpy as np
import pandas as pd

from domain.swings import detect_swings
from domain.structure import detect_structure
from domain.liquidity import track_liquidity, LiquidityPool
from domain.fvg import track_fvgs, FairValueGap
from domain.equilibrium import track_dealing_ranges, DealingRange
from domain.fibonacci import track_fibonacci_ranges, FibonacciRange
from domain.smt import detect_smt_divergence
from engine.session_engine import extract_session_liquidity
from engine.portfolio import Portfolio, TradeRecord
from engine.broker import SimulatedBroker


@dataclass
class Event:
    event_type: str
    timestamp: pd.Timestamp
    payload: Dict[str, Any] = field(default_factory=dict)


class EventBus:
    """
    Lightweight publish-subscribe event bus for decoupling simulation components
    and recording an audit trail of backtest events.
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Event], None]]] = {}
        self.event_log: List[Event] = []

    def subscribe(self, event_type: str, handler: Callable[[Event], None]) -> None:
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    def publish(self, event: Event) -> None:
        self.event_log.append(event)
        for handler in self._subscribers.get(event.event_type, []):
            handler(event)


@dataclass
class BarContext:
    """
    Zero-Lookahead market snapshot at closed candle `timestamp`.
    Contains ONLY data and domain primitives confirmed on or before `timestamp`.
    """
    symbol: str
    timestamp: pd.Timestamp
    bar_index: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    history_df: pd.DataFrame
    confirmed_swings: List[Dict[str, Any]] = field(default_factory=list)
    active_fvgs: List[FairValueGap] = field(default_factory=list)
    recent_sweeps: List[LiquidityPool] = field(default_factory=list)
    active_dealing_range: Optional[DealingRange] = None
    active_fib_range: Optional[FibonacciRange] = None
    recent_smt_events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class BacktestResult:
    strategy_name: str
    symbol: str
    initial_capital: float
    final_equity: float
    net_profit: float
    total_return_pct: float
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown_pct: float
    max_drawdown_cash: float
    avg_r_multiple: float
    trades: List[TradeRecord]
    equity_df: pd.DataFrame


def calculate_metrics(portfolio: Portfolio, strategy_name: str, symbol: str) -> BacktestResult:
    """
    Computes standard academic and quantitative performance metrics from a completed Portfolio.
    """
    trades = portfolio.closed_trades
    total_trades = len(trades)

    if portfolio.equity_curve:
        equity_df = pd.DataFrame(portfolio.equity_curve).set_index("timestamp")
        final_equity = float(equity_df["equity"].iloc[-1])
    else:
        equity_df = pd.DataFrame(columns=["cash", "unrealized_pnl", "equity"])
        final_equity = float(portfolio.cash)

    net_profit = final_equity - portfolio.initial_capital
    total_return_pct = (net_profit / portfolio.initial_capital) * 100.0 if portfolio.initial_capital > 0 else 0.0

    if total_trades > 0:
        wins = [t.net_pnl for t in trades if t.net_pnl > 0]
        losses = [abs(t.net_pnl) for t in trades if t.net_pnl <= 0]
        win_rate = (len(wins) / total_trades) * 100.0
        gross_win = sum(wins)
        gross_loss = sum(losses)
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
        avg_r = float(np.mean([t.r_multiple for t in trades]))
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_r = 0.0

    # Max Drawdown & Annualized Sharpe Ratio from bar-by-bar equity curve
    if len(equity_df) > 1:
        eq_series = equity_df["equity"].astype(float)
        running_max = eq_series.cummax()
        dd_cash = running_max - eq_series
        dd_pct = (dd_cash / running_max.replace(0, np.nan)) * 100.0

        max_drawdown_cash = float(dd_cash.max())
        max_drawdown_pct = float(dd_pct.max()) if not np.isnan(dd_pct.max()) else 0.0

        returns = eq_series.pct_change().dropna()
        std_ret = float(returns.std())
        if std_ret > 1e-12:
            # Annualize 1-minute returns (approx. 252 trading days * 390 RTH minutes = 98,280 bars/year)
            sharpe_ratio = float((returns.mean() / std_ret) * np.sqrt(252 * 390))
        else:
            sharpe_ratio = 0.0
    else:
        max_drawdown_cash = 0.0
        max_drawdown_pct = 0.0
        sharpe_ratio = 0.0

    return BacktestResult(
        strategy_name=strategy_name,
        symbol=symbol,
        initial_capital=portfolio.initial_capital,
        final_equity=final_equity,
        net_profit=net_profit,
        total_return_pct=total_return_pct,
        total_trades=total_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        sharpe_ratio=sharpe_ratio,
        max_drawdown_pct=max_drawdown_pct,
        max_drawdown_cash=max_drawdown_cash,
        avg_r_multiple=avg_r,
        trades=trades,
        equity_df=equity_df,
    )


class BacktestEngine:
    """
    Step-by-step chronological simulation engine with strict Zero Look-Ahead enforcement.
    """

    def __init__(
        self,
        symbol: str = "ES",
        initial_capital: float = 100_000.0,
        commission_per_contract: float = 2.00,
        slippage_ticks: int = 1,
        swing_confirmation_bars: int = 2,
    ):
        self.symbol = symbol.upper()
        self.initial_capital = float(initial_capital)
        self.commission_per_contract = float(commission_per_contract)
        self.slippage_ticks = int(slippage_ticks)
        self.swing_confirmation_bars = int(swing_confirmation_bars)

    @staticmethod
    def _group_macros(macro_pools: List[Dict[str, Any]]) -> Dict[pd.Timestamp, List[Dict[str, Any]]]:
        grouped: Dict[pd.Timestamp, List[Dict[str, Any]]] = {}
        for pool in macro_pools:
            t = pool["time"]
            if t not in grouped:
                grouped[t] = []
            grouped[t].append(pool)
        return grouped

    def run(
        self,
        df_primary: pd.DataFrame,
        strategy: Any,
        df_secondary: Optional[pd.DataFrame] = None,
        precompute_domain: bool = True,
    ) -> BacktestResult:
        """
        Executes a chronological bar-by-bar backtest.
        At each bar i (timestamp t):
        1. Broker evaluates intrabar Stop-Loss / Take-Profit and pending orders.
        2. Strategy receives a BarContext containing ONLY closed data up to bar i
           and domain objects confirmed by timestamp t (respecting fractal confirmation lag).
        3. Portfolio records mark-to-market equity.
        """
        if df_primary.empty:
            raise ValueError("df_primary cannot be empty.")

        portfolio = Portfolio(initial_capital=self.initial_capital)
        broker = SimulatedBroker(
            portfolio=portfolio,
            commission_per_contract=self.commission_per_contract,
            slippage_ticks=self.slippage_ticks,
        )
        event_bus = EventBus()

        times = df_primary.index
        opens = df_primary["open"].to_numpy(dtype=float)
        highs = df_primary["high"].to_numpy(dtype=float)
        lows = df_primary["low"].to_numpy(dtype=float)
        closes = df_primary["close"].to_numpy(dtype=float)
        volumes = (
            df_primary["volume"].to_numpy(dtype=float)
            if "volume" in df_primary.columns
            else np.zeros(len(df_primary), dtype=float)
        )

        # Pre-index domain primitives and gate their visibility by confirmation timestamp
        if precompute_domain and len(df_primary) >= 5:
            swings = detect_swings(df_primary)
            structure_df = detect_structure(df_primary, swings)
            macro_by_time = self._group_macros(extract_session_liquidity(df_primary))
            _, liq_engine = track_liquidity(structure_df, swings, macro_by_time, tolerance_ticks=2)
            all_pools = liq_engine.get_all_pools()
            fvgs = track_fvgs(df_primary)
            dealing_ranges = track_dealing_ranges(df_primary, swings)
            fib_ranges = track_fibonacci_ranges(df_primary, swings)

            if df_secondary is not None and not df_secondary.empty:
                sec_swings = detect_swings(df_secondary)
                smt_events = detect_smt_divergence(
                    swings, sec_swings, df_primary, df_secondary, time_window_minutes=15
                )
            else:
                smt_events = []
        else:
            swings, all_pools, fvgs, dealing_ranges, fib_ranges, smt_events = [], [], [], [], [], []

        confirm_delta = pd.Timedelta(minutes=self.swing_confirmation_bars)
        recent_window = pd.Timedelta(minutes=15)

        # Index sweep events by timestamp for fast O(1) bar lookup
        sweeps_by_time: Dict[pd.Timestamp, List[LiquidityPool]] = {}
        for pool in all_pools:
            for ev_type, ev_ts in pool.events:
                if ev_type == "SWEEP":
                    if ev_ts not in sweeps_by_time:
                        sweeps_by_time[ev_ts] = []
                    sweeps_by_time[ev_ts].append(pool)

        recent_sweeps_buffer: List[Tuple[pd.Timestamp, LiquidityPool]] = []

        # Step-by-step chronological loop
        for i in range(len(df_primary)):
            t = times[i]
            o, h, l, c, v = opens[i], highs[i], lows[i], closes[i], volumes[i]

            # 1. Process Broker exits (SL/TP) & pending orders on current bar
            closed_trades = broker.process_candle(self.symbol, t, o, h, l, c)
            for tr in closed_trades:
                event_bus.publish(
                    Event("TRADE_CLOSED", t, {"symbol": self.symbol, "trade": tr})
                )

            # 2. Gather strictly confirmed domain state as of timestamp `t` (Zero Look-Ahead)
            confirmed_cutoff = t - confirm_delta

            if precompute_domain:
                if t in sweeps_by_time:
                    for p in sweeps_by_time[t]:
                        recent_sweeps_buffer.append((t, p))
                recent_sweeps_buffer = [
                    (ts, p) for ts, p in recent_sweeps_buffer if (t - ts) <= recent_window
                ]
                vis_sweeps = [p for _, p in recent_sweeps_buffer]

                vis_fvgs = [
                    f for f in fvgs
                    if f.creation_time <= t
                    and not any(
                        ev in ("FILLED_BY_CLOSE", "INVERTED") and ev_ts <= t
                        for ev, ev_ts, _ in f.events
                    )
                ]

                active_dr = None
                for dr in dealing_ranges:
                    if dr.creation_time <= confirmed_cutoff:
                        if dr.break_time is None or dr.break_time > t:
                            active_dr = dr
                    else:
                        break

                active_fib = None
                for fib in fib_ranges:
                    if fib.extreme_time <= confirmed_cutoff:
                        if fib.break_time is None or fib.break_time > t:
                            active_fib = fib

                vis_smt = [
                    smt for smt in smt_events
                    if confirmed_cutoff - recent_window <= smt["time"] <= confirmed_cutoff
                ]
                vis_swings = [s for s in swings if s["time"] <= confirmed_cutoff]
            else:
                vis_sweeps, vis_fvgs, active_dr, active_fib, vis_smt, vis_swings = (
                    [], [], None, None, [], []
                )

            ctx = BarContext(
                symbol=self.symbol,
                timestamp=t,
                bar_index=i,
                open=o,
                high=h,
                low=l,
                close=c,
                volume=v,
                history_df=df_primary.iloc[: i + 1],
                confirmed_swings=vis_swings,
                active_fvgs=vis_fvgs,
                recent_sweeps=vis_sweeps,
                active_dealing_range=active_dr,
                active_fib_range=active_fib,
                recent_smt_events=vis_smt,
            )

            # 3. Invoke strategy logic on closed candle state
            strategy.on_bar(ctx, broker, portfolio)

            # 4. Update Mark-to-Market Equity Curve
            portfolio.update_mark_to_market(t, {self.symbol: c})

        # Force-close any open position at the final bar so all trades are accounted for
        if portfolio.has_open_position(self.symbol):
            last_time = times[-1]
            last_close = closes[-1]
            portfolio.close_position(
                symbol=self.symbol,
                exit_price=last_close,
                timestamp=last_time,
                exit_reason="END_OF_DATA",
                commission_per_side=self.commission_per_contract,
            )
            if portfolio.equity_curve:
                portfolio.equity_curve[-1]["cash"] = portfolio.cash
                portfolio.equity_curve[-1]["unrealized_pnl"] = 0.0
                portfolio.equity_curve[-1]["equity"] = portfolio.cash

        strategy_name = getattr(strategy, "name", strategy.__class__.__name__)
        return calculate_metrics(portfolio, strategy_name=strategy_name, symbol=self.symbol)