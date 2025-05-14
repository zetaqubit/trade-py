from collections import defaultdict
from dataclasses import dataclass

@dataclass
class Position:
    quantity: float = 0.0
    price: float = 0.0

class Portfolio:
    """Manages positions and cash, generates orders from signals."""
    def __init__(self, initial_cash=100000.0):
        self.cash = initial_cash
        self.positions: dict[str, Position] = defaultdict(Position)  # symbol -> Position
        self.open_trades = {}  # track open trade info

    def in_long_position(self, symbol):
        return self.positions.get(symbol, 0.0) > 0

    def in_short_position(self, symbol):
        return self.positions.get(symbol, 0.0) < 0

    def on_market_event(self, market_event):
        self.positions[market_event.symbol].price = market_event.ohlc.o

    def on_fill(self, fill):
        """Update cash and positions from a FillEvent."""
        symbol = fill.symbol
        qty = fill.quantity
        cost = fill.price * qty + fill.commission
        self.positions[symbol].quantity += qty
        self.positions[symbol].price = fill.price
        self.cash -= cost
        # (Log trade, update PnL, etc.)

    def net_worth(self):
        return self.cash + sum(p.quantity * p.price for p in self.positions.values())


    def equity_position(self):
        return 1 - self.cash / self.net_worth()


    def __str__(self):
        return (f'Portfolio(cash={self.cash:.2f}, net_worth={self.net_worth():.2f}, '
                f'positions={dict(self.positions)})')

    __repr__ = __str__