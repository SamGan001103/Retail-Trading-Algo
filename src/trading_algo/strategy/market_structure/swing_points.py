from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from trading_algo.strategy.base import MarketBar


@dataclass
class SwingLevel:
    side: str  # "high" or "low"
    price: float
    start_index: int
    end_index: int
    broken: bool = False


@dataclass(frozen=True)
class SwingPointsSnapshot:
    current_high: SwingLevel | None
    current_low: SwingLevel | None
    past_highs: tuple[SwingLevel, ...]
    past_lows: tuple[SwingLevel, ...]
    high_broken: bool
    low_broken: bool
    new_swing_high: float | None
    new_swing_low: float | None


class SwingPointsDetector:
    """
    Stateful swing-point detector inspired by the Pine scaffold:
    - Tracks one current swing high/low.
    - Archives previous/current levels into past lists.
    - Flags breaks when price exceeds current levels.
    - Optionally removes swept past levels.
    """

    def __init__(
        self,
        swing_strength_high: int = 5,
        swing_strength_low: int = 5,
        remove_swept_levels: bool = False,
        max_past_levels: int = 500,
        swing_source_high: Callable[[MarketBar], float] | None = None,
        swing_source_low: Callable[[MarketBar], float] | None = None,
    ):
        if swing_strength_high < 1 or swing_strength_low < 1:
            raise ValueError("swing strengths must be >= 1")
        if max_past_levels < 1:
            raise ValueError("max_past_levels must be >= 1")

        self.swing_strength_high = int(swing_strength_high)
        self.swing_strength_low = int(swing_strength_low)
        self.remove_swept_levels = bool(remove_swept_levels)
        self.max_past_levels = int(max_past_levels)
        self._source_high = swing_source_high or (lambda b: b.high)
        self._source_low = swing_source_low or (lambda b: b.low)

        self._bars: list[MarketBar] = []
        self.current_high: SwingLevel | None = None
        self.current_low: SwingLevel | None = None
        self.past_highs: list[SwingLevel] = []
        self.past_lows: list[SwingLevel] = []

    def reset(self) -> None:
        self._bars.clear()
        self.current_high = None
        self.current_low = None
        self.past_highs.clear()
        self.past_lows.clear()

    def update(self, bar: MarketBar) -> SwingPointsSnapshot:
        self._bars.append(bar)
        current_index = len(self._bars) - 1

        prev_current_high = self.current_high
        prev_current_low = self.current_low
        high_broken = prev_current_high is not None and bar.high > prev_current_high.price
        low_broken = prev_current_low is not None and bar.low < prev_current_low.price

        detected_high = self._detect_swing_high()
        detected_low = self._detect_swing_low()
        new_swing_high: float | None = None
        new_swing_low: float | None = None

        if detected_high is not None:
            pivot_index, pivot_price = detected_high
            new_swing_high = pivot_price
            if self.current_high is not None:
                self._archive_current_high()
            self.current_high = SwingLevel(
                side="high",
                price=pivot_price,
                start_index=pivot_index,
                end_index=current_index,
                broken=False,
            )

        if detected_low is not None:
            pivot_index, pivot_price = detected_low
            new_swing_low = pivot_price
            if self.current_low is not None:
                self._archive_current_low()
            self.current_low = SwingLevel(
                side="low",
                price=pivot_price,
                start_index=pivot_index,
                end_index=current_index,
                broken=False,
            )

        if self.current_high is not None:
            self.current_high.end_index = current_index
        if self.current_low is not None:
            self.current_low.end_index = current_index

        if high_broken:
            if self.current_high is prev_current_high and self.current_high is not None:
                self.current_high.broken = True
                self._archive_current_high()
                self.current_high = None
            elif prev_current_high is not None:
                prev_current_high.broken = True

        if low_broken:
            if self.current_low is prev_current_low and self.current_low is not None:
                self.current_low.broken = True
                self._archive_current_low()
                self.current_low = None
            elif prev_current_low is not None:
                prev_current_low.broken = True

        if self.remove_swept_levels:
            self._remove_swept_levels(bar)

        return SwingPointsSnapshot(
            current_high=self._copy_level(self.current_high),
            current_low=self._copy_level(self.current_low),
            past_highs=tuple(self._copy_level(level) for level in self.past_highs),
            past_lows=tuple(self._copy_level(level) for level in self.past_lows),
            high_broken=high_broken,
            low_broken=low_broken,
            new_swing_high=new_swing_high,
            new_swing_low=new_swing_low,
        )

    def _copy_level(self, level: SwingLevel | None) -> SwingLevel | None:
        if level is None:
            return None
        return SwingLevel(
            side=level.side,
            price=level.price,
            start_index=level.start_index,
            end_index=level.end_index,
            broken=level.broken,
        )

    def _archive_current_high(self) -> None:
        if self.current_high is None:
            return
        self.past_highs.append(self.current_high)
        self._trim_levels(self.past_highs)

    def _archive_current_low(self) -> None:
        if self.current_low is None:
            return
        self.past_lows.append(self.current_low)
        self._trim_levels(self.past_lows)

    def _trim_levels(self, levels: list[SwingLevel]) -> None:
        overflow = len(levels) - self.max_past_levels
        if overflow > 0:
            del levels[:overflow]

    def _remove_swept_levels(self, bar: MarketBar) -> None:
        self.past_highs = [level for level in self.past_highs if not (bar.high > level.price)]
        self.past_lows = [level for level in self.past_lows if not (bar.low < level.price)]

    def _detect_swing_high(self) -> tuple[int, float] | None:
        s = self.swing_strength_high
        if len(self._bars) < (2 * s + 1):
            return None
        center_idx = len(self._bars) - 1 - s
        center = self._source_high(self._bars[center_idx])

        for i in range(1, s + 1):
            if center <= self._source_high(self._bars[center_idx - i]):
                return None
            if center <= self._source_high(self._bars[center_idx + i]):
                return None
        return center_idx, float(center)

    def _detect_swing_low(self) -> tuple[int, float] | None:
        s = self.swing_strength_low
        if len(self._bars) < (2 * s + 1):
            return None
        center_idx = len(self._bars) - 1 - s
        center = self._source_low(self._bars[center_idx])

        for i in range(1, s + 1):
            if center >= self._source_low(self._bars[center_idx - i]):
                return None
            if center >= self._source_low(self._bars[center_idx + i]):
                return None
        return center_idx, float(center)


class MultiTimeframeSwingPoints:
    """
    Keeps one detector per timeframe key (e.g. "1m", "5m", "15m").
    """

    def __init__(self, detectors: dict[str, SwingPointsDetector]):
        if not detectors:
            raise ValueError("detectors cannot be empty")
        self.detectors = detectors

    def update(self, timeframe: str, bar: MarketBar) -> SwingPointsSnapshot:
        detector = self.detectors.get(timeframe)
        if detector is None:
            raise KeyError(f"No detector configured for timeframe={timeframe!r}")
        return detector.update(bar)
