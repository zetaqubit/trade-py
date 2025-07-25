"""Main event loop enabling communication between the Strategy and Broker."""

from absl import app, flags

from trade_py import brokers, strategies

FLAGS = flags.FLAGS

flags.DEFINE_string("broker_config", None, "yaml config of broker to use.")
flags.DEFINE_string("strategy_config", None, "yaml config of strategy to use.")


def main(_):
    """Main entry point for the application."""

    broker = brokers.create(FLAGS.broker_config)
    strategy = strategies.create(FLAGS.strategy_config, broker)

    try:
        while True:
            market_event = broker.next()
            print(f"Processing: {market_event}")
            order_events = strategy.process_market_event(market_event)
            for order_event in order_events:
                print(f"Processing: {order_event}")
                broker.process_order_event(order_event)
            print(broker.portfolio)
    except StopIteration:
        pass

    print(broker.portfolio.net_worth())


if __name__ == "__main__":
    app.run(main)
