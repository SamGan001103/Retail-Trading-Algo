from __future__ import annotations

from typing import Literal, Protocol, Sequence

from trading_algo.core import BUY, SELL

Trend = Literal["up", "down", "neutral"]
Bias = Literal["bullish", "bearish", "neutral"]

SwingPoint = tuple[int, float]


class _PriceLevel(Protocol):
    @property
    def price(self) -> float:
        ...


class SwingDetectorLike(Protocol):
    @property
    def current_high(self) -> _PriceLevel | None:
        ...

    @property
    def current_low(self) -> _PriceLevel | None:
        ...

    @property
    def past_highs(self) -> Sequence[_PriceLevel]:
        ...

    @property
    def past_lows(self) -> Sequence[_PriceLevel]:
        ...


def compute_ltf_trend(high_points: Sequence[SwingPoint], low_points: Sequence[SwingPoint]) -> Trend:
    if len(high_points) < 2 or len(low_points) < 2:
        return "neutral"
    last_high = high_points[-1][1]
    prev_high = high_points[-2][1]
    last_low = low_points[-1][1]
    prev_low = low_points[-2][1]
    if last_high > prev_high and last_low > prev_low:
        return "up"
    if last_high < prev_high and last_low < prev_low:
        return "down"
    return "neutral"


def compute_htf_bias(detector: SwingDetectorLike) -> Bias:
    highs = [level.price for level in detector.past_highs]
    lows = [level.price for level in detector.past_lows]
    if detector.current_high is not None:
        highs.append(detector.current_high.price)
    if detector.current_low is not None:
        lows.append(detector.current_low.price)
    if len(highs) < 2 or len(lows) < 2:
        return "neutral"
    if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
        return "bullish"
    if highs[-1] < highs[-2] and lows[-1] < lows[-2]:
        return "bearish"
    return "neutral"


def is_choch_bullish(trend_before: Trend, close: float, ltf_high_points: Sequence[SwingPoint]) -> bool:
    if trend_before != "down" or not ltf_high_points:
        return False
    last_high = ltf_high_points[-1][1]
    return close > last_high


def is_choch_bearish(trend_before: Trend, close: float, ltf_low_points: Sequence[SwingPoint]) -> bool:
    if trend_before != "up" or not ltf_low_points:
        return False
    last_low = ltf_low_points[-1][1]
    return close < last_low


def equal_levels(
    side: int,
    ltf_high_points: Sequence[SwingPoint],
    ltf_low_points: Sequence[SwingPoint],
    tolerance_bps: float,
) -> bool:
    points = ltf_low_points if side == BUY else ltf_high_points
    if len(points) < 2:
        return False
    p1_price = points[-1][1]
    p2_price = points[-2][1]
    ref = max(1e-12, abs(p2_price))
    diff_bps = abs(p1_price - p2_price) / ref * 10_000.0
    return diff_bps <= tolerance_bps


def fib_retracement_confluence(
    side: int,
    close: float,
    ltf_high_points: Sequence[SwingPoint],
    ltf_low_points: Sequence[SwingPoint],
) -> bool:
    if not ltf_high_points or not ltf_low_points:
        return False

    if side == BUY:
        low_idx, low_price = ltf_low_points[-1]
        high_idx, high_price = ltf_high_points[-1]
        if high_idx <= low_idx or high_price <= low_price:
            return False
        retr = (high_price - close) / max(1e-12, high_price - low_price)
        return 0.5 <= retr <= 0.79

    high_idx, high_price = ltf_high_points[-1]
    low_idx, low_price = ltf_low_points[-1]
    if low_idx <= high_idx or high_price <= low_price:
        return False
    retr = (close - low_price) / max(1e-12, high_price - low_price)
    return 0.5 <= retr <= 0.79


def key_area_proximity(side: int, close: float, detector: SwingDetectorLike, tolerance_bps: float) -> bool:
    if side == BUY:
        level = detector.current_low.price if detector.current_low is not None else None
        if level is None and detector.past_lows:
            level = detector.past_lows[-1].price
    else:
        level = detector.current_high.price if detector.current_high is not None else None
        if level is None and detector.past_highs:
            level = detector.past_highs[-1].price
    if level is None:
        return False
    diff_bps = abs(close - level) / max(1e-12, abs(level)) * 10_000.0
    return diff_bps <= tolerance_bps
