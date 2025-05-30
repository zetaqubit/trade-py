"""Broker factory supporting registration and creation."""

import pandas as pd  # noqa: F401

from trade_py import broker, configs
from trade_py.alpaca_broker import AlpacaBroker  # noqa: F401
from trade_py.simulated_broker import SimulatedBroker  # noqa: F401


def create(cfg: dict | str) -> broker.Broker:
    if isinstance(cfg, str):
        cfg = configs.load_config(cfg)
    broker_cls = globals().get(cfg.broker_type, None)
    if not broker_cls or not issubclass(broker_cls, broker.Broker):
        raise NotImplementedError(f"Unknown {cfg.broker_type=}.")
    return broker_cls(cfg)
