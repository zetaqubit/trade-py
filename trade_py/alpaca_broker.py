import asyncio
import queue
import threading
from typing import List, Tuple


from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, PositionSide, TimeInForce, TradeEvent
from alpaca.trading.models import Position, TradeAccount, TradeUpdate
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.stream import TradingStream


from . import bars
from . import broker
from . import credentials
from . import events
from . import portfolio


class AlpacaBroker(broker.Broker):
    def __init__(self, cfg: dict):
        super().__init__(cfg)

        keys = credentials.get_alpaca_keys()
        self.client = TradingClient(keys['api_key'], keys['api_secret'])

        # Initialize portfolio to mirror live.
        self.portfolio = self._sync_portfolio()

        # Set up stream to listen to trade events
        self.trading_stream = TradingStream(keys['api_key'], keys['api_secret'])
        self.trading_stream.subscribe_trade_updates(self.on_trade_update)

        # Set up stream for real-time price updates.
        self.stock_stream = StockDataStream(keys['api_key'], keys['api_secret'])
        self.stock_stream.subscribe_quotes(self.on_quote_data, self.cfg.symbol)

        self.bars = bars.Bars()
        self.bars.add_callback(bars.Frequency.MINUTE, self.on_bar)
        self.market_events = queue.Queue()

        self._start_stream(self.trading_stream)
        self._start_stream(self.stock_stream)

    def _sync_portfolio(self):
        p = portfolio.Portfolio()
        account: TradeAccount = self.client.get_account()
        p.cash = float(account.cash)
        positions: List[Position] = self.client.get_all_positions()
        for position in positions:
            assert position.side == PositionSide.LONG
            p.positions[position.symbol] = portfolio.Position(
                quantity=float(position.qty), price=float(position.current_price)
            )
        return p

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
        self.bars.on_event(time=data.timestamp, price=data.ask_price)

    def on_bar(self, ohlc: events.OHLC):
        market_event = events.MarketEvent(time=ohlc.end, symbol=self.cfg.symbol, ohlc=ohlc)
        self.market_events.put(market_event)

    def next(self):
        market_event = self.market_events.get()
        self.portfolio.on_market_event(market_event)
        return market_event

    def process_order_event(self, order_event: events.OrderEvent):
        buy_or_sell, quantity = quantity_to_buy_sell(order_event.quantity)
        if quantity.value:
            quantity.value = round(quantity.value, 2)
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
    async def on_trade_update(self, data: TradeUpdate):
        if data.event in (TradeEvent.PARTIAL_FILL, TradeEvent.FILL):
            time = data.order.filled_at
            avg_price = data.price
            qty = data.qty
            if data.order.side == OrderSide.SELL:
                qty *= -1
            print(f"Order filled: {data.order.id} at {avg_price} for {qty} shares")
            fill_event = events.FillEvent(
                time=time,
                symbol=self.cfg.symbol,
                quantity=qty,
                price=avg_price,
            )
            self.portfolio.on_fill(fill_event)


def quantity_to_buy_sell(quantity: events.Quantity) -> Tuple[OrderSide, events.Quantity]:
    if quantity.shares and quantity.shares < 0:
        return (OrderSide.SELL, events.Quantity(shares=-quantity.shares))
    if quantity.value and quantity.value < 0:
        return (OrderSide.SELL, events.Quantity(value=-quantity.value))
    return OrderSide.BUY, quantity
