import asyncio
import threading
from typing import Tuple


from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.stream import TradingStream


from . import broker
from . import credentials
from . import events


# Initialize the TradingStream with your API credentials


class AlpacaBroker(broker.Broker):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        keys = credentials.get_alpaca_keys()
        self.client = TradingClient(keys['api_key'], keys['api_secret'])

        # Set up stream to listen to trade events
        self.trading_stream = TradingStream(keys['api_key'], keys['api_secret'])
        self.trading_stream.subscribe_trade_updates(self.on_trade_update)

        # Set up stream for real-time price updates.
        self.stock_stream = StockDataStream(keys['api_key'], keys['api_secret'])
        self.stock_stream.subscribe_quotes(self.on_quote_data, self.cfg.symbol)

        self._start_stream(self.trading_stream)
        self._start_stream(self.stock_stream)

    def _start_stream(self, stream):
        try:
            # If an event loop is running (e.g. in IPython), schedule task
            loop = asyncio.get_running_loop()
            loop.create_task(stream._run_forever())
        except RuntimeError:
            # No running loop (e.g. in script), create one in a separate thread
            def runner():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(stream._run_forever())

            thread = threading.Thread(target=runner, daemon=True)
            thread.start()


    async def on_quote_data(self, data):
        print(data)


    def next(self):
        pass

    def process_order_event(self, order_event: events.OrderEvent):
        buy_or_sell, quantity = quantity_to_buy_sell(order_event.quantity)
        order_request = MarketOrderRequest(
            symbol=order_event.symbol,
            qty=quantity.shares,
            notional=quantity.value,
            side=buy_or_sell,
            time_in_force=TimeInForce.DAY,
        )
        print(order_request)

        order = self.client.submit_order(order_data=order_request)
        print(order)

    # Define the asynchronous callback function to handle trade updates
    async def on_trade_update(self, data):
        if data.event == "fill":
            print(f"Order filled: {data.order.id} at {data.price} for {data.qty} shares")



def quantity_to_buy_sell(quantity: events.Quantity) -> Tuple[OrderSide, events.Quantity]:
    if quantity.shares and quantity.shares < 0:
        return (OrderSide.SELL, events.Quantity(shares=-quantity.shares))
    if quantity.value and quantity.value < 0:
        return (OrderSide.SELL, events.Quantity(value=-quantity.value))
    return OrderSide.BUY, quantity
