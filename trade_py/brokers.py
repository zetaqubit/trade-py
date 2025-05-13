"""Broker factory supporting registration and creation."""

import pandas as pd

from . import broker
from . import configs
from .alpaca_broker import AlpacaBroker
from .simulated_broker import SimulatedBroker


def create(cfg: dict | str) -> broker.Broker:
    if isinstance(cfg, str):
        cfg = configs.load_config(cfg)
    broker_cls = globals().get(cfg.broker_type, None)
    if not broker_cls or not issubclass(broker_cls, broker.Broker):
        raise NotImplementedError(f'Unknown {cfg.broker_type=}.')
    return broker_cls(cfg)
