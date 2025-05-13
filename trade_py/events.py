from abc import ABC
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

class EventType(Enum):
    UNKNOWN = 0
    MARKET = 1
    ORDER = 2
    FILL = 3


@dataclass
class Event:
    type: EventType
    time: datetime


@dataclass
class OHLC:
    o: float
    h: float
    l: float
    c: float
    start: datetime | None = None
    end: datetime | None = None


@dataclass
class Quantity:
    """Amount of equity, specified either as shares or notional dollar value."""
    shares: float | None = None
    value: float | None = None

    def __post_init__(self):
        assert (self.shares is None) ^ (self.value is None)
        assert bool(self.shares) ^ bool(self.value)  # ensure non-zero


# Events from Broker to Strategy
@dataclass
class MarketEvent(Event):
    """New market data tick/bar."""
    def __init__(self, *, time: datetime, symbol: str, ohlc: OHLC):
        self.type = EventType.MARKET
        self.time = time
        self.symbol = symbol
        self.ohlc = ohlc

    def __str__(self):
        return (f"MarketEvent(time={self.time.strftime('%Y-%m-%d %H:%M:%S')}, "
                f"symbol={self.symbol}, "
                f"OHLC(o={self.ohlc.o:.2f}, h={self.ohlc.h:.2f}, "
                f"l={self.ohlc.l:.2f}, c={self.ohlc.c:.2f}))")


@dataclass
class FillEvent(Event):
    """Confirmation that an order was filled."""
    def __init__(self, *, time: datetime, symbol: str, quantity: float, price: float, commission: float = 0.0):
        self.type = EventType.FILL
        self.time = time
        self.symbol = symbol
        self.quantity = quantity
        self.price = price
        self.commission = commission

# Events from Strategy to Broker
@dataclass
class OrderEvent(Event):
    """Order to send to broker.

    Quantity is positive for buy and negative for sell.
    """
    def __init__(self, *, time: datetime, symbol: str, quantity: Quantity):
        self.type = EventType.ORDER
        self.time = time
        self.symbol = symbol
        self.quantity = quantity

    def __str__(self):
        qty_str = (f"shares={self.quantity.shares:.2f}"
                  if self.quantity.shares is not None
                  else f"value=${self.quantity.value:.2f}")
        return (f"OrderEvent(time={self.time.strftime('%Y-%m-%d %H:%M:%S')}, "
                f"symbol={self.symbol}, {qty_str})")
