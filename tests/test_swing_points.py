from trading_algo.strategy import MarketBar
from trading_algo.strategy.market_structure import SwingPointsDetector


def _bar(ts: int, high: float, low: float) -> MarketBar:
    close = (high + low) / 2.0
    return MarketBar(ts=str(ts), open=close, high=high, low=low, close=close, volume=1.0)


def test_detects_current_swing_levels():
    detector = SwingPointsDetector(swing_strength_high=2, swing_strength_low=2, remove_swept_levels=False)
    bars = [
        _bar(0, 1, 5),
        _bar(1, 2, 4),
        _bar(2, 5, 1),
        _bar(3, 2, 4),
        _bar(4, 1, 5),
    ]

    snapshot = None
    for bar in bars:
        snapshot = detector.update(bar)

    assert snapshot is not None
    assert snapshot.current_high is not None
    assert snapshot.current_low is not None
    assert snapshot.current_high.price == 5.0
    assert snapshot.current_low.price == 1.0
    assert snapshot.current_high.start_index == 2
    assert snapshot.current_low.start_index == 2


def test_break_moves_current_level_to_past():
    detector = SwingPointsDetector(swing_strength_high=2, swing_strength_low=2, remove_swept_levels=False)
    bars = [
        _bar(0, 1, 5),
        _bar(1, 2, 4),
        _bar(2, 5, 1),
        _bar(3, 2, 4),
        _bar(4, 1, 5),
        _bar(5, 6, 4),
    ]

    snapshot = None
    for bar in bars:
        snapshot = detector.update(bar)

    assert snapshot is not None
    assert snapshot.high_broken is True
    assert snapshot.current_high is None
    assert len(snapshot.past_highs) == 1
    assert snapshot.past_highs[0].price == 5.0
    assert snapshot.past_highs[0].broken is True


def test_remove_swept_levels_deletes_past_levels():
    detector = SwingPointsDetector(swing_strength_high=2, swing_strength_low=2, remove_swept_levels=True)
    bars = [
        _bar(0, 1, 5),
        _bar(1, 2, 4),
        _bar(2, 5, 1),
        _bar(3, 2, 4),
        _bar(4, 1, 5),
        _bar(5, 6, 4),
    ]

    snapshot = None
    for bar in bars:
        snapshot = detector.update(bar)

    assert snapshot is not None
    assert snapshot.high_broken is True
    assert len(snapshot.past_highs) == 0
