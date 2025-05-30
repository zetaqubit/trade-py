"""Transforms data of higher frequency to lower frequencies."""

from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import Callable

from trade_py import events


class Frequency(Enum):
    SECOND = 0
    MINUTE = 1
    HOUR = 2


class Bars:

    def __init__(self):
        self.callbacks = defaultdict(list)
        self.current_bars = dict()

    def add_callback(self, frequency: Frequency, callback: Callable[[events.OHLC], None]):
        self.callbacks[frequency].append(callback)

    def on_event(self, time: datetime, price: float):
        for freq in Frequency:
            new_bar = events.OHLC(
                o=price,
                h=price,
                l=price,
                c=price,
                start=time,
                end=time,
            )

            if freq not in self.current_bars:
                self.current_bars[freq] = new_bar
                continue
            bar = self.current_bars[freq]
            if not is_current_bar(bar, freq, time):
                for callback in self.callbacks[freq]:
                    callback(bar)
                self.current_bars[freq] = new_bar
            else:
                bar.h = max(bar.h, price)
                bar.l = min(bar.l, price)
                bar.c = price
                bar.end = time


def is_current_bar(bar: events.OHLC, freq: Frequency, time: datetime):
    def same_day(dt1, dt2):
        assert dt1.tzinfo == dt2.tzinfo
        return (
            dt1.year == dt2.year and dt1.month == dt2.month and dt1.day == dt2.day
        )

    def same_hour(dt1, dt2):
        return same_day(dt1, dt2) and dt1.hour == dt2.hour

    def same_minute(dt1, dt2):
        return same_hour(dt1, dt2) and dt1.minute == dt2.minute

    def same_second(dt1, dt2):
        return same_minute(dt1, dt2) and dt1.second == dt2.second

    return {
        Frequency.SECOND: same_second,
        Frequency.MINUTE: same_minute,
        Frequency.HOUR: same_hour,
    }[freq](bar.start, time)
