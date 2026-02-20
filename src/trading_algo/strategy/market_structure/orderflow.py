from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Protocol, Sequence

from trading_algo.core import BUY, SELL
from trading_algo.strategy.base import MarketBar, StrategyContext


def num(payload: Mapping[str, Any] | None, *keys: str) -> float | None:
    if payload is None:
        return None
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


class OrderFlowFilter(Protocol):
    def allow_entry(self, side: int, bar: MarketBar, context: StrategyContext, flow: "OrderFlowState") -> bool:
        ...


class AllowAllOrderFlowFilter:
    def allow_entry(self, side: int, bar: MarketBar, context: StrategyContext, flow: "OrderFlowState") -> bool:  # noqa: ARG002
        return True


@dataclass(frozen=True)
class OrderFlowState:
    quote: dict[str, Any] | None = None
    trade: dict[str, Any] | None = None
    depth: dict[str, Any] | None = None

    def best_bid(self) -> float | None:
        bid = num(self.quote, "bid", "bidPrice", "bestBid")
        if bid is not None:
            return bid
        levels = self.depth.get("bids") if self.depth else None
        if isinstance(levels, list) and levels:
            first = levels[0]
            if isinstance(first, dict):
                return num(first, "price", "p")
        return num(self.depth, "bestBid", "bidPrice")

    def best_ask(self) -> float | None:
        ask = num(self.quote, "ask", "askPrice", "bestAsk")
        if ask is not None:
            return ask
        levels = self.depth.get("asks") if self.depth else None
        if isinstance(levels, list) and levels:
            first = levels[0]
            if isinstance(first, dict):
                return num(first, "price", "p")
        return num(self.depth, "bestAsk", "askPrice")

    def top_bid_size(self) -> float | None:
        size = num(self.depth, "bestBidSize", "bidSize")
        if size is not None:
            return size
        levels = self.depth.get("bids") if self.depth else None
        if isinstance(levels, list) and levels:
            first = levels[0]
            if isinstance(first, dict):
                return num(first, "size", "qty", "q")
        return None

    def top_ask_size(self) -> float | None:
        size = num(self.depth, "bestAskSize", "askSize")
        if size is not None:
            return size
        levels = self.depth.get("asks") if self.depth else None
        if isinstance(levels, list) and levels:
            first = levels[0]
            if isinstance(first, dict):
                return num(first, "size", "qty", "q")
        return None

    def imbalance(self) -> float | None:
        bid = self.top_bid_size()
        ask = self.top_ask_size()
        if bid is None or ask is None:
            return None
        denom = bid + ask
        if denom <= 0:
            return None
        return (bid - ask) / denom


@dataclass(frozen=True)
class DepthImbalanceOrderFlowFilter:
    min_abs_imbalance: float = 0.12

    def allow_entry(self, side: int, bar: MarketBar, context: StrategyContext, flow: OrderFlowState) -> bool:  # noqa: ARG002
        imbalance = flow.imbalance()
        if imbalance is None:
            return False
        if side == BUY:
            return imbalance >= self.min_abs_imbalance
        if side == SELL:
            return imbalance <= -self.min_abs_imbalance
        return False


@dataclass(frozen=True)
class TickFlowSample:
    ts: datetime
    price: float
    bid: float | None
    ask: float | None
    bid_size: float | None
    ask_size: float | None
    imbalance: float | None
    trade_price: float | None
    trade_size: float

    @property
    def mid(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return (self.bid + self.ask) / 2.0


@dataclass(frozen=True)
class TickExecutionConfig:
    min_imbalance: float
    min_trade_size: float
    spoof_collapse_ratio: float
    absorption_min_trades: int
    iceberg_min_reloads: int
    tick_size: float


def tick_entry_ready(samples: Sequence[TickFlowSample], side: int, config: TickExecutionConfig) -> bool:
    if len(samples) < 8:
        return False
    latest = samples[-1]
    if latest.imbalance is None:
        return False
    if side == BUY:
        imbalance_ok = latest.imbalance >= config.min_imbalance
    else:
        imbalance_ok = latest.imbalance <= -config.min_imbalance
    if not imbalance_ok:
        return False
    if _has_spoofing_collapse(samples, side, config):
        return False
    return _has_absorption_or_iceberg(samples, side, config) or _micro_timing_directional(samples, side)


def tick_exhaustion_exit_signal(samples: Sequence[TickFlowSample], side: int, config: TickExecutionConfig) -> bool:
    window = list(samples)[-12:]
    if len(window) < 6:
        return False
    aggressive_buy = 0.0
    aggressive_sell = 0.0
    eps = 1e-9
    for sample in window:
        if sample.trade_price is None or sample.trade_size <= 0:
            continue
        if sample.ask is not None and sample.trade_price >= (sample.ask - eps):
            aggressive_buy += sample.trade_size
        elif sample.bid is not None and sample.trade_price <= (sample.bid + eps):
            aggressive_sell += sample.trade_size

    mids = [s.mid for s in window if s.mid is not None]
    if len(mids) < 2:
        return False
    progress = mids[-1] - mids[0]
    latest_imbalance = window[-1].imbalance
    if latest_imbalance is None:
        return False

    if side == BUY:
        buying_aggression = aggressive_buy > aggressive_sell * 1.15
        no_advance = progress <= config.tick_size
        imbalance_flip = latest_imbalance < -abs(config.min_imbalance)
        return buying_aggression and (no_advance or imbalance_flip)

    selling_aggression = aggressive_sell > aggressive_buy * 1.15
    no_advance = progress >= -config.tick_size
    imbalance_flip = latest_imbalance > abs(config.min_imbalance)
    return selling_aggression and (no_advance or imbalance_flip)


def dom_support_liquidity_level(flow: OrderFlowState, side: int, min_size: float) -> float | None:
    levels = flow.depth.get("bids") if side == BUY and flow.depth else None
    if side == SELL and flow.depth:
        levels = flow.depth.get("asks")
    return pick_largest_depth_level(levels, min_size)


def dom_opposing_liquidity_level(flow: OrderFlowState, side: int, min_size: float) -> float | None:
    levels = flow.depth.get("asks") if side == BUY and flow.depth else None
    if side == SELL and flow.depth:
        levels = flow.depth.get("bids")
    return pick_largest_depth_level(levels, min_size)


def pick_largest_depth_level(levels: Any, min_size: float) -> float | None:
    if not isinstance(levels, list) or not levels:
        return None
    best_price: float | None = None
    best_size = max(0.0, min_size)
    for level in levels[:10]:
        if not isinstance(level, dict):
            continue
        price = num(level, "price", "p")
        size = num(level, "size", "qty", "q")
        if price is None or size is None:
            continue
        if size < min_size:
            continue
        if best_price is None or size > best_size:
            best_price = float(price)
            best_size = float(size)
    return best_price


def _has_spoofing_collapse(samples: Sequence[TickFlowSample], side: int, config: TickExecutionConfig) -> bool:
    window = list(samples)[-12:]
    if len(window) < 4:
        return False
    opposite_sizes = [s.ask_size for s in window] if side == BUY else [s.bid_size for s in window]
    valid_sizes = [x for x in opposite_sizes if x is not None]
    if len(valid_sizes) < 3:
        return False
    peak_size = max(valid_sizes)
    latest_size = valid_sizes[-1]
    if peak_size <= 0:
        return False
    collapse = latest_size <= peak_size * config.spoof_collapse_ratio
    if not collapse:
        return False

    mids = [s.mid for s in window if s.mid is not None]
    if len(mids) < 2:
        return True
    favorable_move = mids[-1] > mids[0] if side == BUY else mids[-1] < mids[0]
    return not favorable_move


def _has_absorption_or_iceberg(samples: Sequence[TickFlowSample], side: int, config: TickExecutionConfig) -> bool:
    window = list(samples)[-24:]
    if len(window) < 4:
        return False
    aggressive = 0
    reloads = 0
    eps = 1e-9
    for prev, cur in zip(window, window[1:]):
        if cur.trade_price is None or cur.trade_size < config.min_trade_size:
            continue
        if side == BUY:
            if cur.ask is None or cur.trade_price < (cur.ask - eps):
                continue
            aggressive += 1
            if (
                prev.ask is not None
                and cur.ask is not None
                and abs(cur.ask - prev.ask) <= eps
                and prev.ask_size is not None
                and cur.ask_size is not None
                and cur.ask_size >= prev.ask_size * 0.8
            ):
                reloads += 1
        else:
            if cur.bid is None or cur.trade_price > (cur.bid + eps):
                continue
            aggressive += 1
            if (
                prev.bid is not None
                and cur.bid is not None
                and abs(cur.bid - prev.bid) <= eps
                and prev.bid_size is not None
                and cur.bid_size is not None
                and cur.bid_size >= prev.bid_size * 0.8
            ):
                reloads += 1
    return aggressive >= config.absorption_min_trades and reloads >= config.iceberg_min_reloads


def _micro_timing_directional(samples: Sequence[TickFlowSample], side: int) -> bool:
    window = list(samples)[-8:]
    mids = [s.mid for s in window if s.mid is not None]
    if len(mids) < 3:
        return False
    if side == BUY:
        return mids[-1] > mids[-3]
    return mids[-1] < mids[-3]
