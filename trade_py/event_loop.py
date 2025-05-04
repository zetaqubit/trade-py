"""Main event loop enabling communication between the Strategy and Broker.
"""

from absl import app
from absl import flags
import yaml

from . import brokers
from . import events
from . import strategies

FLAGS = flags.FLAGS

flags.DEFINE_string('broker_config', None, "yaml config of broker to use.")
flags.DEFINE_string('strategy_config', None, "yaml config of strategy to use.")


def to_dotdict(d):
    if isinstance(d, dict):
        return type('DotDict', (dict,), {
            '__getattr__': lambda self, name: to_dotdict(self[name]),
            '__setattr__': lambda self, name, value: self.__setitem__(name, value)
        })(**{k: to_dotdict(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [to_dotdict(i) for i in d]
    return d

def load_config(path):
    with open(path, 'r') as f:
        d = yaml.safe_load(f)
    return to_dotdict(d)


def main(_):
    """Main entry point for the application."""

    broker_cfg = load_config(FLAGS.broker_config)
    broker = brokers.create(broker_cfg)
    strategy_cfg = load_config(FLAGS.strategy_config)
    strategy = strategies.create(strategy_cfg, broker)

    order_events = []
    try:
        while True:
            market_event = broker.next()
            for order_event in order_events:
                print(f"Processing: {order_event}")
                broker.process_order_event(order_event)
            print(f"Processing: {market_event}")
            order_events = strategy.process_market_event(market_event)
    except StopIteration:
        pass

    print(broker.portfolio.net_worth())

if __name__ == '__main__':
    app.run(main)

