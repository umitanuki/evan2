import alpaca_trade_api as alpaca
import asyncio
import pandas as pd
import sys

import logging

logger = logging.getLogger()


class Quote():
    """
    We use Quote objects to represent the bid/ask spread. When we encounter a
    'level change', a move of exactly 1 penny, we may attempt to make one
    trade. Whether or not the trade is successfully filled, we do not submit
    another trade until we see another level change.
    Note: Only moves of 1 penny are considered eligible because larger moves
    could potentially indicate some newsworthy event for the stock, which this
    algorithm is not tuned to trade.
    """

    def __init__(self, symbol):
        self.prev_bid = 0
        self.prev_ask = 0
        self.prev_spread = 0
        self.bid = 0
        self.ask = 0
        self.bid_size = 0
        self.ask_size = 0
        self.spread = 0
        self.traded = True
        self.level_ct = 1
        self.time = pd.Timestamp.now(tz='America/New_York')
        self.elapse = pd.Timedelta(0)

        self._symbol = symbol
        self._l = logger.getChild('Q.' + symbol)

    def reset(self):
        # Called when a level change happens
        self.traded = False
        self.level_ct += 1

    def update(self, data):
        # Update bid and ask sizes and timestamp
        self.bid_size = data.bidsize
        self.ask_size = data.asksize

        # Check if there has been a level change
        if (
            self.bid != data.bidprice
            and self.ask != data.askprice
            and round(data.askprice - data.bidprice, 2) == .01
        ):
            prev_time = self.time
            # Update bids and asks and time of level change
            self.prev_bid = self.bid
            self.prev_ask = self.ask
            self.bid = data.bidprice
            self.ask = data.askprice
            self.time = data.timestamp
            self.elapse = data.timestamp - prev_time
            # Update spreads
            self.prev_spread = round(self.prev_ask - self.prev_bid, 3)
            self.spread = round(self.ask - self.bid, 3)
            self._l.info(
                f'Level change: b={self.prev_bid}/a={self.prev_ask}:'
                f'{self.prev_spread} <- b={self.bid}/a={self.ask}:{self.spread} '
                f'in {self.elapse.value:,}ns')
            # If change is from one penny spread level to a different penny
            # spread level, then initialize for new level (reset stale vars)
            if self.prev_spread == 0.01:
                self.reset()

    def copy(self):
        q = Quote(self._symbol)
        q.prev_bid = self.prev_bid
        q.prev_ask = self.prev_ask
        q.prev_spread = self.prev_spread
        q.bid = self.bid
        q.ask = self.ask
        q.bid_size = self.bid_size
        q.ask_size = self.ask_size
        q.time = self.time
        q.elapse = self.elapse
        q.spread = self.spread
        q.traded = self.traded
        return q


class Trade:

    def __init__(self, symbol):
        self._raw = None
        self._symbol = symbol
        self._l = logger.getChild('Q.' + symbol)

    def update(self, data):
        if self._raw is not None and self._raw.price != data.price:
            self._l.info(f'trade {data.price}')
        self._raw = data
        self.price = data.price
        self.timestamp = data.timestamp
        self.size = data.size
        self.exchange = data.exchange


class SucideAlgo:

    def __init__(self, api, symbol, shares):
        self._api = api
        self._symbol = symbol
        self._shars = shares
        self._entry_quote = None
        self._bars = []
        self._l = logger.getChild(self._symbol)

        now = pd.Timestamp.now(tz='America/New_York').floor('1min')
        market_open = now.replace(hour=9, minute=30)
        today = now.strftime('%Y-%m-%d')
        tomorrow = (now + pd.Timedelta('1day')).strftime('%Y-%m-%d')
        data = api.polygon.historic_agg_v2(
            symbol, 1, 'minute', today, tomorrow, unadjusted=False).df
        bars = data[market_open:]
        self._bars = bars
        self._quote = Quote(symbol)
        self._trade = Trade(symbol)
        self._trade.update(api.polygon.last_trade(symbol))

        self._init_state()

    def _init_state(self):
        symbol = self._symbol
        order = [o for o in self._api.list_orders() if o.symbol == symbol]
        position = [p for p in self._api.list_positions()
                    if p.symbol == symbol]
        self._order = order[0] if len(order) > 0 else None
        self._position = position[0] if len(position) > 0 else None
        if self._position is not None:
            if self._order is None:
                self._state = 'TO_EXIT'
            else:
                self._state = 'EXITING'
                if self._order.side != 'sell':
                    self._l.warn(
                        f'state {self._state} mismatch order {self._order}')
        else:
            if self._order is None:
                self._state = 'TO_ENTER'
            else:
                self._state = 'ENTERING'
                if self._order.side != 'buy':
                    self._l.warn(
                        f'state {self._state} mismatch order {self._order}')

        if self._state != 'TO_ENTER':
            raise Exception('state is not clean, delete orders/positions')

    def _now(self):
        return pd.Timestamp.now(tz='America/New_York')

    def _outofmarket(self):
        return self._now().time() >= pd.Timestamp('15:55').time()

    def checkup(self, position):
        self._l.info('periodic task')

        now = self._now()
        order = self._order
        if (order is not None and order.side == 'buy' and now -
                pd.Timestamp(order.submitted_at, tz='America/New_York') > pd.Timedelta('2 min')):
            last_price = self._api.polygon.last_trade(self._symbol).price
            self._l.info(
                f'canceling missed buy order {order.id} at {order.limit_price} '
                f'(current price = {last_price})')
            self._cancel_order()

        if self._position is not None and self._outofmarket():
            self._submit_exit(bailout=True)

    def _cancel_order(self):
        if self._order is not None:
            self._api.cancel_order(self._order.id)

    def _get_signal(self):
        q = self._quote
        if not q.traded and q.spread == 0.01:
            diff = q.ask - q.prev_ask
            if diff > 0:
                self._l.info(f'entry signal: diff={diff}')
                return True

    def on_quote(self, quote):
        self._quote.update(quote)
        if self._state == 'TO_ENTER':
            signal = self._get_signal()
            if signal:
                self._submit_entry()

    def on_trade(self, trade):
        self._trade.update(trade)

    def on_bar(self, bar):
        self._bars = self._bars.append(pd.DataFrame({
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume,
        }, index=[bar.start]))

        self._l.info(
            f'received bar start = {bar.start}, close = {bar.close}, len(bars) = {len(self._bars)}')

    def on_order_update(self, event, order):
        # self._l.info(f'order update: {event} = {order}')
        if event == 'fill':
            self._l(f'filled at {order["filled_avg_price"]}')
            self._order = None
            if self._state == 'ENTERING':
                self._position = self._api.get_position(self._symbol)
                self._transition('TO_EXIT')
                self._submit_exit()
            elif self._state == 'EXITING':
                self._position = None
                self._transition('TO_ENTER')
        elif event == 'partial_fill':
            self._l(f'partially filled at {order["filled_avg_price"]}')
            self._position = self._api.get_position(self._symbol)
            self._order = self._api.get_order(order['id'])
        elif event in ('canceled', 'rejected'):
            self._l(f'cancel/reject at {order["id"]}')
            if event == 'rejected':
                self._l.warn(f'order rejected: current order = {self._order}')
            self._order = None
            if self._state == 'ENTERING':
                if self._position is not None:
                    self._transition('TO_EXIT')
                    self._submit_exit()
                else:
                    self._transition('TO_ENTER')
            elif self._state == 'EXITING':
                self._transition('TO_EXIT')
                self._submit_exit(bailout=True)
            else:
                self._l.warn(f'unexpected state for {event}: {self._state}')

    def _submit_entry(self):
        # trade = self._trade
        quote = self._quote
        amount = self._shares
        # amount = int(self._lot / trade.price)
        try:
            order = self._api.submit_order(
                symbol=self._symbol,
                side='buy',
                type='limit',
                qty=amount,
                time_in_force='ioc',
                limit_price=quote.ask,
            )
        except Exception as e:
            self._l.info(e)
            self._transition('TO_ENTER')
            return

        self._order = order
        self._entry_quote = self._quote.copy()
        self._l.info(f'submitted entry {order}')
        self._transition('ENTERING')

    def _submit_exit(self, bailout=False):
        params = dict(
            symbol=self._symbol,
            side='sell',
            qty=self._position.qty,
            time_in_force='day',
        )
        if bailout:
            params['type'] = 'market'
        else:
            q_entry = self._entry_quote
            q_last = self._quote
            if q_entry.bid > q_last.bid:
                self._l.info(
                    f'entry bid > last bid: {q_entry.bid} > {q_last.bid}')
            limit_price = q_entry.bid
            params.update(dict(
                type='limit',
                limit_price=limit_price,
            ))
        try:
            order = self._api.submit_order(**params)
        except Exception as e:
            self._l.error(e)
            self._transition('TO_EXIT')
            return

        self._order = order
        self._l.info(f'submitted exit {order}')
        self._transition('EXITING')

    def _transition(self, new_state):
        self._l.info(f'transition from {self._state} to {new_state}')
        self._state = new_state


def main(args):
    api = alpaca.REST()
    stream = alpaca.StreamConn()

    fleet = {}
    symbols = args.symbols
    for symbol in symbols:
        algo = SucideAlgo(api, symbol, shares=args.shares)
        fleet[symbol] = algo

    @stream.on(r'^T')
    async def on_trades(conn, channel, data):
        if data.symbol in fleet:
            fleet[data.symbol].on_trade(data)

    @stream.on(r'^Q')
    async def on_quotes(conn, channel, data):
        if data.symbol in fleet:
            fleet[data.symbol].on_quote(data)

    @stream.on(r'trade_updates')
    async def on_trade_updates(conn, channel, data):
        logger.info(f'trade_updates {data}')
        symbol = data.order['symbol']
        if symbol in fleet:
            fleet[symbol].on_order_update(data.event, data.order)

    async def periodic():
        while True:
            if not api.get_clock().is_open:
                logger.info('exit as market is not open')
                sys.exit(0)
            await asyncio.sleep(30)
            positions = api.list_positions()
            for symbol, algo in fleet.items():
                pos = [p for p in positions if p.symbol == symbol]
                algo.checkup(pos[0] if len(pos) > 0 else None)
    channels = ['trade_updates'] + [
        'Q.' + symbol for symbol in symbols
    ] + [
        'T.' + symbol for symbol in symbols
    ]

    loop = stream.loop
    loop.run_until_complete(asyncio.gather(
        stream.subscribe(channels),
        periodic(),
    ))
    loop.close()


if __name__ == '__main__':
    import argparse

    fmt = '%(asctime)s:%(filename)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s'
    logging.basicConfig(level=logging.INFO, format=fmt)
    fh = logging.FileHandler('console.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)

    parser = argparse.ArgumentParser()
    parser.add_argument('symbols', nargs='+')
    parser.add_argument('--shares', type=float, default=100)

    main(parser.parse_args())
