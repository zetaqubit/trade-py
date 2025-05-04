import random

from . import broker
from . import events
from . import strategy


def create(cfg: dict, broker: broker.Broker) -> strategy.Strategy:
    if cfg.strategy_type == 'PositionStrategy':
        return PositionStrategy(cfg, broker)
    if cfg.strategy_type == 'RandomStrategy':
        return RandomStrategy(cfg, broker)
    if cfg.strategy_type == 'BuyAndHoldStrategy':
        return BuyAndHoldStrategy(cfg, broker)
    else:
        raise NotImplementedError(f'Unknown {cfg.strategy_type=}.')



class PositionStrategy(strategy.Strategy):
    """Strategy that specifies position to maintain (fraction between 0 and 1).

    Child classes should update target_position in process_market_event, and then call the super
    implentation.
    """

    def __init__(self, cfg, broker):
        self.cfg = cfg
        self.broker = broker
        self.position = 0
        self.target_position = 0

    def process_market_event(self, market_event: events.MarketEvent):
        delta = self.target_position - self.position
        if delta == 0:
            return []

        delta_dollars = delta * self.broker.portfolio.net_worth()
        self.position = self.target_position
        return [
            events.OrderEvent(
                time=market_event.time,
                symbol=self.cfg.symbol,
                quantity=events.Quantity(value=delta_dollars),
            )
        ]


class RandomStrategy(PositionStrategy):
    def process_market_event(self, market_event: events.MarketEvent):
        self.target_position = random.random()
        return super().process_market_event(market_event)


class BuyAndHoldStrategy(PositionStrategy):
    def __init__(self, cfg, broker):
        super().__init__(cfg, broker)
        self.target_position = 1
