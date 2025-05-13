"""Broker factory supporting registration and creation."""

import pandas as pd

from . import broker
from . import configs
from . import events
from . import portfolio
from .alpaca_broker import AlpacaBroker


def create(cfg: dict | str) -> broker.Broker:
    if isinstance(cfg, str):
        cfg = configs.load_config(cfg)
    broker_cls = globals().get(cfg.broker_type, None)
    if not broker_cls or not issubclass(broker_cls, broker.Broker):
        raise NotImplementedError(f'Unknown {cfg.broker_type=}.')
    return broker_cls(cfg)


class SimulatedBroker(broker.Broker):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        print(cfg)
        df = pd.read_csv(cfg.df_path)
        df.timestamp = pd.to_datetime(df.timestamp)
        print(f'Read {len(df)} rows.')
        start, end = pd.to_datetime(cfg.start_date), pd.to_datetime(cfg.end_date)
        df = df[(df.timestamp >= start) & (df.timestamp <= end)]
        print(f'Kept {len(df)} rows from {cfg.start_date} to {cfg.end_date}.')
        print(df)
        self.df = df
        self.current_i = -1
        self.data = None

        self.portfolio = portfolio.Portfolio()

    def next(self):
        if self.current_i >= len(self.df) - 1:
            raise StopIteration()
        self.current_i += 1
        row = self.df.iloc[self.current_i]

        def row_to_market_event(row):
            return events.MarketEvent(
                time=row.timestamp,
                symbol=self.cfg.symbol,
                ohlc=events.OHLC(o=row.open, h=row.high, l=row.low, c=row.close)
            )
        self.data = row_to_market_event(row)
        self.portfolio.on_market_event(self.data)
        return self.data


    def process_order_event(self, order_event: events.OrderEvent):
        o = order_event
        price = self.data.ohlc.o
        if o.quantity.shares is not None:
            quantity = o.quantity.shares
        elif o.quantity.value is not None:
            quantity = o.quantity.value / price
        fill_event = events.FillEvent(
            time=o.time,
            symbol=o.symbol,
            quantity=quantity,
            price=price,
        )
        self.portfolio.on_fill(fill_event)


class DataFrameRowIterator:
    def __init__(self, df, map_fn=None):
        self._rows = df.itertuples(index=True)
        self.map_fn = map_fn

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self._rows)
        if self.map_fn:
            item = self.map_fn(item)
        return item





