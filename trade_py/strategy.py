from trade_py import events


class Strategy:
    def process_market_event(self, market_event: events.MarketEvent):
        raise NotImplementedError()

