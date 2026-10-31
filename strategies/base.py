from abc import ABC, abstractmethod
import random
from engine.event_bus import BarContext
from engine.broker import SimulatedBroker
from engine.portfolio import Portfolio


class BaseStrategy(ABC):
    """
    Abstract base class for all backtesting strategies.
    Enforces a unified step-by-step interface without look-ahead bias.
    """
    name: str = "BaseStrategy"

    @abstractmethod
    def on_bar(
        self,
        ctx: BarContext,
        broker: SimulatedBroker,
        portfolio: Portfolio,
    ) -> None:
        """Called at the close of every 1-minute bar."""
        pass


class BuyAndHoldStrategy(BaseStrategy):
    """
    Baseline Benchmark #1:
    Enters a LONG position on the first available bar and holds until the end of the dataset.
    """
    name: str = "BuyAndHoldBaseline"

    def __init__(self, quantity: int = 1, protective_stop_points: float = 5000.0, target_points: float = 10000.0):
        self.quantity = int(quantity)
        self.protective_stop_points = float(protective_stop_points)
        self.target_points = float(target_points)
        self._entered = False

    def on_bar(
        self,
        ctx: BarContext,
        broker: SimulatedBroker,
        portfolio: Portfolio,
    ) -> None:
        if not self._entered and not portfolio.has_open_position(ctx.symbol):
            broker.submit_order(
                symbol=ctx.symbol,
                direction="LONG",
                order_type="MARKET",
                quantity=self.quantity,
                stop_loss=ctx.close - self.protective_stop_points,
                take_profit=ctx.close + self.target_points,
                timestamp=ctx.timestamp,
                current_close=ctx.close,
                metadata={"baseline": "BuyAndHold"},
            )
            self._entered = True


class RandomEntryStrategy(BaseStrategy):
    """
    Baseline Benchmark #2 (Required by Thesis Advisor):
    Enters random LONG or SHORT trades using identical Stop-Loss and Take-Profit
    distances as the formal strategy to test statistical significance.
    """
    name: str = "RandomEntryBaseline"

    def __init__(
        self,
        stop_loss_points: float = 10.0,
        take_profit_points: float = 20.0,
        entry_probability: float = 0.05,
        quantity: int = 1,
        seed: int = 42,
    ):
        self.stop_loss_points = float(stop_loss_points)
        self.take_profit_points = float(take_profit_points)
        self.entry_probability = float(entry_probability)
        self.quantity = int(quantity)
        self._rng = random.Random(seed)

    def on_bar(
        self,
        ctx: BarContext,
        broker: SimulatedBroker,
        portfolio: Portfolio,
    ) -> None:
        if portfolio.has_open_position(ctx.symbol):
            return

        if self._rng.random() < self.entry_probability:
            direction = self._rng.choice(["LONG", "SHORT"])
            if direction == "LONG":
                sl = ctx.close - self.stop_loss_points
                tp = ctx.close + self.take_profit_points
            else:
                sl = ctx.close + self.stop_loss_points
                tp = ctx.close - self.take_profit_points

            broker.submit_order(
                symbol=ctx.symbol,
                direction=direction,
                order_type="MARKET",
                quantity=self.quantity,
                stop_loss=sl,
                take_profit=tp,
                timestamp=ctx.timestamp,
                current_close=ctx.close,
                metadata={"baseline": "RandomEntry", "direction": direction},
            )