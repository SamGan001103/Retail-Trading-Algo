from __future__ import annotations

from typing import Protocol, Sequence

from trading_algo.strategy.base import MarketBar


class _PriceLevel(Protocol):
    price: float


class SwingSnapshotLike(Protocol):
    current_high: _PriceLevel | None
    current_low: _PriceLevel | None
    past_highs: Sequence[_PriceLevel]
    past_lows: Sequence[_PriceLevel]


def aggregate_bars(bars: Sequence[MarketBar]) -> MarketBar:
    if not bars:
        raise ValueError("bars cannot be empty")
    first = bars[0]
    last = bars[-1]
    return MarketBar(
        ts=last.ts,
        open=first.open,
        high=max(b.high for b in bars),
        low=min(b.low for b in bars),
        close=last.close,
        volume=sum(b.volume for b in bars),
    )


def detect_htf_sweep(bar: MarketBar, snapshot: SwingSnapshotLike, wick_sweep_ratio_min: float) -> tuple[bool, bool]:
    range_size = max(1e-12, bar.high - bar.low)
    upper_wick = max(0.0, bar.high - max(bar.open, bar.close))
    lower_wick = max(0.0, min(bar.open, bar.close) - bar.low)

    ref_high = snapshot.current_high.price if snapshot.current_high is not None else None
    ref_low = snapshot.current_low.price if snapshot.current_low is not None else None
    if ref_high is None and snapshot.past_highs:
        ref_high = snapshot.past_highs[-1].price
    if ref_low is None and snapshot.past_lows:
        ref_low = snapshot.past_lows[-1].price

    sweep_high = (
        ref_high is not None
        and bar.high > ref_high
        and bar.close < ref_high
        and (upper_wick / range_size) >= wick_sweep_ratio_min
    )
    sweep_low = (
        ref_low is not None
        and bar.low < ref_low
        and bar.close > ref_low
        and (lower_wick / range_size) >= wick_sweep_ratio_min
    )
    return sweep_high, sweep_low


def has_recent_sweep(
    *,
    last_sweep_side: int | None,
    last_sweep_index: int | None,
    side: int,
    index: int,
    sweep_expiry_bars: int,
) -> bool:
    if last_sweep_side is None or last_sweep_index is None:
        return False
    if last_sweep_side != side:
        return False
    return (index - last_sweep_index) <= sweep_expiry_bars
