from __future__ import annotations

from collections import deque
import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from trading_algo.core import BUY, SELL
from trading_algo.position_management import StopLossPlanner, TakeProfitPlanner
from trading_algo.strategy.base import MarketBar, PositionState, Strategy, StrategyContext, StrategyDecision

from .liquidity_sweep import aggregate_bars, detect_htf_sweep, has_recent_sweep
from .orderflow import (
    AllowAllOrderFlowFilter,
    DepthImbalanceOrderFlowFilter,
    OrderFlowFilter,
    OrderFlowState,
    TickExecutionConfig,
    TickFlowSample,
    dom_opposing_liquidity_level,
    dom_support_liquidity_level,
    num,
    tick_entry_ready,
    tick_exhaustion_exit_signal,
)
from .structure_signals import (
    Bias,
    Trend,
    compute_htf_bias,
    compute_ltf_trend,
    equal_levels,
    fib_retracement_confluence,
    is_choch_bearish,
    is_choch_bullish,
    key_area_proximity,
)
from .swing_points import SwingPointsDetector


@dataclass(frozen=True)
class SetupEnvironment:
    side: int
    index: int
    ts: str
    close: float
    has_recent_sweep: bool
    htf_bias: Bias
    bias_ok: bool
    continuation: bool
    reversal: bool
    equal_levels: bool
    fib_retracement: bool
    key_area_proximity: bool
    confluence_score: int


@dataclass(frozen=True)
class SetupMLDecision:
    approved: bool
    score: float | None
    reason: str


@dataclass(frozen=True)
class PlannedEntry:
    setup: SetupEnvironment
    size: int
    sl_ticks_abs: int
    tp_ticks_abs: int
    stop_level: float
    target_level: float
    rrr: float
    stop_order_type: str
    take_profit_order_type: str
    ml_score: float | None


class SetupApprovalGate(Protocol):
    def evaluate(self, setup: SetupEnvironment) -> tuple[bool, float | None, str]:
        ...


def _parse_ts(ts: str) -> datetime:
    value = ts.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    shift = (weekday - first.weekday()) % 7
    day = 1 + shift + (n - 1) * 7
    return date(year, month, day)


def _new_york_offset_hours_utc(dt_utc: datetime) -> int:
    year = dt_utc.year
    dst_start_day = _nth_weekday_of_month(year, 3, 6, 2)
    dst_end_day = _nth_weekday_of_month(year, 11, 6, 1)
    dst_start_utc = datetime(year, 3, dst_start_day.day, 7, 0, tzinfo=timezone.utc)
    dst_end_utc = datetime(year, 11, dst_end_day.day, 6, 0, tzinfo=timezone.utc)
    in_dst = dst_start_utc <= dt_utc < dst_end_utc
    return -4 if in_dst else -5


def _to_local(dt: datetime, tz_name: str) -> datetime:
    dt_utc = dt.astimezone(timezone.utc)
    try:
        return dt_utc.astimezone(ZoneInfo(tz_name))
    except ZoneInfoNotFoundError:
        if tz_name == "America/New_York":
            offset = _new_york_offset_hours_utc(dt_utc)
            return dt_utc.astimezone(timezone(timedelta(hours=offset)))
        return dt_utc


def _parse_hhmm(value: str) -> time:
    parts = value.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid session time {value!r}. Use HH:MM.")
    hour = int(parts[0])
    minute = int(parts[1])
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError(f"Invalid session time {value!r}. Use HH:MM.")
    return time(hour=hour, minute=minute)


@dataclass(frozen=True)
class SessionWindow:
    start: time
    end: time
    tz_name: str = "America/New_York"

    def contains(self, ts: str) -> bool:
        dt_local = _to_local(_parse_ts(ts), self.tz_name)
        # NY structure strategy should only trade on weekdays.
        if dt_local.weekday() >= 5:
            return False
        now_t = dt_local.time()
        if self.start <= self.end:
            return self.start <= now_t < self.end
        return now_t >= self.start or now_t < self.end


@dataclass
class NYSessionMarketStructureStrategy(Strategy):
    size: int = 1
    session_start: str = "09:30"
    session_end: str = "16:00"
    tz_name: str = "America/New_York"
    htf_aggregation: int = 5
    bias_aggregation: int = 60
    htf_swing_strength_high: int = 5
    htf_swing_strength_low: int = 5
    ltf_swing_strength_high: int = 3
    ltf_swing_strength_low: int = 3
    wick_sweep_ratio_min: float = 0.5
    sweep_expiry_bars: int = 40
    equal_level_tolerance_bps: float = 8.0
    key_area_tolerance_bps: float = 12.0
    min_confluence_score: int = 1
    require_orderflow: bool = False
    entry_mode: str = "bar"
    avoid_news: bool = False
    news_blackouts_utc: list[tuple[datetime, datetime]] = field(default_factory=list, repr=False)
    news_exit_on_event: bool = False
    max_hold_bars: int = 120
    tick_setup_expiry_bars: int = 3
    tick_history_size: int = 120
    tick_min_imbalance: float = 0.12
    tick_min_trade_size: float = 1.0
    tick_spoof_collapse_ratio: float = 0.35
    tick_absorption_min_trades: int = 2
    tick_iceberg_min_reloads: int = 2
    symbol: str = "MNQ"
    tick_size: float = 0.25
    tick_value: float = 0.5
    account_max_drawdown: float = 2_500.0
    max_trade_drawdown_fraction: float = 0.15
    risk_min_rrr: float = 3.0
    risk_max_rrr: float = 10.0
    sl_noise_buffer_ticks: int = 2
    sl_max_ticks: int = 200
    tp_front_run_ticks: int = 2
    dom_liquidity_wall_size: float = 800.0
    ml_min_size_fraction: float = 0.35
    ml_size_floor_score: float = 0.55
    ml_size_ceiling_score: float = 0.90
    enable_exhaustion_market_exit: bool = True
    ml_decision_policy: str = "off"
    orderflow_filter: OrderFlowFilter | None = None

    _session: SessionWindow = field(init=False, repr=False)
    _htf_detector: SwingPointsDetector = field(init=False, repr=False)
    _bias_detector: SwingPointsDetector = field(init=False, repr=False)
    _ltf_detector: SwingPointsDetector = field(init=False, repr=False)
    _htf_buffer: list[MarketBar] = field(default_factory=list, init=False, repr=False)
    _bias_buffer: list[MarketBar] = field(default_factory=list, init=False, repr=False)
    _ltf_high_points: list[tuple[int, float]] = field(default_factory=list, init=False, repr=False)
    _ltf_low_points: list[tuple[int, float]] = field(default_factory=list, init=False, repr=False)
    _last_ltf_trend: Trend = field(default="neutral", init=False, repr=False)
    _last_sweep_side: int | None = field(default=None, init=False, repr=False)
    _last_sweep_index: int | None = field(default=None, init=False, repr=False)
    _last_sweep_price: float | None = field(default=None, init=False, repr=False)
    _orderflow_state: OrderFlowState = field(default_factory=OrderFlowState, init=False, repr=False)
    _pending_setup: PlannedEntry | None = field(default=None, init=False, repr=False)
    _tick_samples: deque[TickFlowSample] = field(init=False, repr=False)
    _candidate_events: deque[dict[str, Any]] = field(init=False, repr=False)
    _ml_gate: SetupApprovalGate | None = field(default=None, init=False, repr=False)
    _stop_planner: StopLossPlanner = field(init=False, repr=False)
    _take_profit_planner: TakeProfitPlanner = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError("size must be >= 1")
        if self.htf_aggregation < 1:
            raise ValueError("htf_aggregation must be >= 1")
        if self.bias_aggregation < 1:
            raise ValueError("bias_aggregation must be >= 1")
        if self.max_hold_bars < 1:
            raise ValueError("max_hold_bars must be >= 1")
        if self.sweep_expiry_bars < 1:
            raise ValueError("sweep_expiry_bars must be >= 1")
        if self.tick_setup_expiry_bars < 1:
            raise ValueError("tick_setup_expiry_bars must be >= 1")
        if self.tick_history_size < 20:
            raise ValueError("tick_history_size must be >= 20")
        if self.tick_size <= 0:
            raise ValueError("tick_size must be > 0")
        if self.tick_value <= 0:
            raise ValueError("tick_value must be > 0")
        if self.account_max_drawdown <= 0:
            raise ValueError("account_max_drawdown must be > 0")
        if not (0 < self.max_trade_drawdown_fraction <= 1):
            raise ValueError("max_trade_drawdown_fraction must be in (0, 1]")
        if self.risk_min_rrr <= 0:
            raise ValueError("risk_min_rrr must be > 0")
        if self.risk_max_rrr < self.risk_min_rrr:
            raise ValueError("risk_max_rrr must be >= risk_min_rrr")
        if self.sl_noise_buffer_ticks < 0:
            raise ValueError("sl_noise_buffer_ticks must be >= 0")
        if self.sl_max_ticks < 1:
            raise ValueError("sl_max_ticks must be >= 1")
        if self.tp_front_run_ticks < 0:
            raise ValueError("tp_front_run_ticks must be >= 0")
        if self.dom_liquidity_wall_size < 0:
            raise ValueError("dom_liquidity_wall_size must be >= 0")
        if not (0 < self.ml_min_size_fraction <= 1):
            raise ValueError("ml_min_size_fraction must be in (0, 1]")
        if not (0 <= self.ml_size_floor_score <= 1 and 0 <= self.ml_size_ceiling_score <= 1):
            raise ValueError("ml_size_floor_score and ml_size_ceiling_score must be in [0, 1]")
        ml_policy = self.ml_decision_policy.strip().lower()
        if ml_policy not in {"off", "shadow", "enforce"}:
            raise ValueError("ml_decision_policy must be one of: off, shadow, enforce")
        self.ml_decision_policy = ml_policy
        mode = self.entry_mode.strip().lower()
        if mode not in {"bar", "tick"}:
            raise ValueError("entry_mode must be 'bar' or 'tick'")
        self.entry_mode = mode
        self.news_blackouts_utc = self._normalize_news_blackouts(self.news_blackouts_utc)

        self._session = SessionWindow(
            start=_parse_hhmm(self.session_start),
            end=_parse_hhmm(self.session_end),
            tz_name=self.tz_name,
        )
        self._htf_detector = SwingPointsDetector(
            swing_strength_high=self.htf_swing_strength_high,
            swing_strength_low=self.htf_swing_strength_low,
            remove_swept_levels=False,
        )
        self._bias_detector = SwingPointsDetector(
            swing_strength_high=self.htf_swing_strength_high,
            swing_strength_low=self.htf_swing_strength_low,
            remove_swept_levels=False,
        )
        self._ltf_detector = SwingPointsDetector(
            swing_strength_high=self.ltf_swing_strength_high,
            swing_strength_low=self.ltf_swing_strength_low,
            remove_swept_levels=False,
        )
        if self.orderflow_filter is None:
            self.orderflow_filter = (
                DepthImbalanceOrderFlowFilter() if self.require_orderflow else AllowAllOrderFlowFilter()
            )
        self._tick_samples = deque(maxlen=self.tick_history_size)
        self._candidate_events = deque(maxlen=4_000)
        self._stop_planner = StopLossPlanner(
            tick_size=self.tick_size,
            noise_buffer_ticks=self.sl_noise_buffer_ticks,
            min_stop_ticks=2,
            max_stop_ticks=self.sl_max_ticks,
        )
        self._take_profit_planner = TakeProfitPlanner(
            tick_size=self.tick_size,
            min_rrr=self.risk_min_rrr,
            max_rrr=self.risk_max_rrr,
            front_run_ticks=self.tp_front_run_ticks,
        )

    def set_orderflow_state(self, flow: OrderFlowState) -> None:
        self._orderflow_state = flow

    def set_ml_gate(self, ml_gate: SetupApprovalGate | None) -> None:
        self._ml_gate = ml_gate

    def pending_setup(self) -> SetupEnvironment | None:
        if self._pending_setup is None:
            return None
        return self._pending_setup.setup

    def drain_candidate_events(self) -> list[dict[str, Any]]:
        events = list(self._candidate_events)
        self._candidate_events.clear()
        return events

    def is_in_session(self, ts: str) -> bool:
        return self._session.contains(ts)

    def on_bar(self, bar: MarketBar, context: StrategyContext, position: PositionState) -> StrategyDecision:
        in_session = self._session.contains(bar.ts)
        in_news_blackout = self._is_news_blackout(bar.ts)
        trend_before = self._last_ltf_trend

        ltf_snapshot = self._ltf_detector.update(bar)
        if ltf_snapshot.new_swing_high is not None:
            self._ltf_high_points.append((context.index - self.ltf_swing_strength_high, ltf_snapshot.new_swing_high))
        if ltf_snapshot.new_swing_low is not None:
            self._ltf_low_points.append((context.index - self.ltf_swing_strength_low, ltf_snapshot.new_swing_low))
        self._trim_points()

        self._update_htf(bar, context.index)
        self._update_bias(bar)
        trend_now = self._compute_ltf_trend()
        choch_bull = self._is_choch_bullish(trend_before, bar.close)
        choch_bear = self._is_choch_bearish(trend_before, bar.close)
        self._last_ltf_trend = trend_now
        self._expire_pending_setup(context.index)

        if position.in_position:
            self._pending_setup = None
            if in_news_blackout and self.news_exit_on_event:
                return StrategyDecision(
                    should_exit=True,
                    should_enter=False,
                    side=position.side if position.side is not None else BUY,
                    size=position.size,
                    reason="exit-news-blackout",
                )
            if not in_session:
                return StrategyDecision(
                    should_exit=True,
                    should_enter=False,
                    side=position.side if position.side is not None else BUY,
                    size=position.size,
                    reason="exit-session-window",
                )
            if position.bars_in_position >= self.max_hold_bars:
                return StrategyDecision(
                    should_exit=True,
                    should_enter=False,
                    side=position.side if position.side is not None else BUY,
                    size=position.size,
                    reason="exit-time-stop",
                )
            return StrategyDecision(
                should_exit=False,
                should_enter=False,
                side=position.side if position.side is not None else BUY,
                size=position.size,
                reason="hold",
            )

        if not in_session:
            self._pending_setup = None
            return StrategyDecision(
                should_exit=False,
                should_enter=False,
                side=BUY,
                size=self.size,
                reason="flat-outside-session",
            )
        if in_news_blackout:
            self._pending_setup = None
            return StrategyDecision(
                should_exit=False,
                should_enter=False,
                side=BUY,
                size=self.size,
                reason="flat-news-blackout",
            )

        long_setup = self._build_setup_environment(
            side=BUY,
            bar=bar,
            context=context,
            trend_now=trend_now,
            choch_bull=choch_bull,
            choch_bear=choch_bear,
        )
        short_setup = self._build_setup_environment(
            side=SELL,
            bar=bar,
            context=context,
            trend_now=trend_now,
            choch_bull=choch_bull,
            choch_bear=choch_bear,
        )
        long_ready = self._setup_ready(long_setup)
        short_ready = self._setup_ready(short_setup)

        if long_ready == short_ready:
            if long_ready:
                self._emit_candidate_event(long_setup, status="rejected", reason="ambiguous-dual-signal")
                self._emit_candidate_event(short_setup, status="rejected", reason="ambiguous-dual-signal")
                self._pending_setup = None
            return StrategyDecision(
                should_exit=False,
                should_enter=False,
                side=BUY,
                size=self.size,
                reason="flat-no-signal",
            )

        chosen = long_setup if long_ready else short_setup
        self._emit_candidate_event(chosen, status="detected", reason="setup-ready")
        ml_decision = self._evaluate_ml(chosen)
        if not ml_decision.approved:
            self._pending_setup = None
            self._emit_candidate_event(chosen, status="rejected", reason=f"ml-reject:{ml_decision.reason}")
            return StrategyDecision(
                should_exit=False,
                should_enter=False,
                side=chosen.side,
                size=self.size,
                reason=f"flat-ml-reject:{ml_decision.reason}",
            )
        entry_plan = self._build_entry_risk_plan(chosen, ml_score=ml_decision.score)
        if entry_plan is None:
            self._pending_setup = None
            self._emit_candidate_event(chosen, status="rejected", reason="risk-invalid")
            return StrategyDecision(
                should_exit=False,
                should_enter=False,
                side=chosen.side,
                size=self.size,
                reason="flat-risk-invalid",
            )

        if self.entry_mode == "tick":
            self._pending_setup = entry_plan
            self._emit_candidate_event(entry_plan.setup, status="armed", reason="tick-entry-mode")
            return StrategyDecision(
                should_exit=False,
                should_enter=False,
                side=entry_plan.setup.side,
                size=entry_plan.size,
                reason="setup-armed-tick",
            )

        if self.require_orderflow and self.orderflow_filter is not None:
            if not self.orderflow_filter.allow_entry(
                side=entry_plan.setup.side,
                bar=bar,
                context=context,
                flow=self._orderflow_state,
            ):
                self._emit_candidate_event(entry_plan.setup, status="rejected", reason="orderflow-gate")
                return StrategyDecision(
                    should_exit=False,
                    should_enter=False,
                    side=entry_plan.setup.side,
                    size=entry_plan.size,
                    reason="flat-orderflow-gate",
                )

        self._emit_candidate_event(entry_plan.setup, status="entered", reason="entry-ny-structure-bar")
        return StrategyDecision(
            should_exit=False,
            should_enter=True,
            side=entry_plan.setup.side,
            size=entry_plan.size,
            reason="entry-ny-structure-bar",
            sl_ticks_abs=entry_plan.sl_ticks_abs,
            tp_ticks_abs=entry_plan.tp_ticks_abs,
        )

    def on_tick(self, ts: str, price: float, context: StrategyContext, position: PositionState) -> StrategyDecision:
        if self.entry_mode != "tick":
            return StrategyDecision(
                should_exit=False,
                should_enter=False,
                side=position.side if position.side is not None else BUY,
                size=position.size if position.size > 0 else self.size,
                reason="tick-disabled",
            )

        if position.in_position:
            self._pending_setup = None
            if self._is_news_blackout(ts) and self.news_exit_on_event:
                side = position.side if position.side is not None else BUY
                return StrategyDecision(
                    should_exit=True,
                    should_enter=False,
                    side=side,
                    size=position.size if position.size > 0 else self.size,
                    reason="tick-news-blackout-exit",
                )
            self._record_tick_sample(ts, price)
            side = position.side if position.side is not None else BUY
            if self.enable_exhaustion_market_exit and self._tick_exhaustion_exit_signal(side):
                return StrategyDecision(
                    should_exit=True,
                    should_enter=False,
                    side=side,
                    size=position.size if position.size > 0 else self.size,
                    reason="tick-exhaustion-market-exit",
                )
            return StrategyDecision(
                should_exit=False,
                should_enter=False,
                side=side,
                size=position.size if position.size > 0 else self.size,
                reason="tick-hold-position",
            )

        self._expire_pending_setup(context.index)
        planned = self._pending_setup
        if planned is None:
            return StrategyDecision(
                should_exit=False,
                should_enter=False,
                side=BUY,
                size=self.size,
                reason="tick-no-setup",
            )
        if self._is_news_blackout(ts):
            self._emit_candidate_event(planned.setup, status="invalidated", reason="news-blackout")
            self._pending_setup = None
            return StrategyDecision(
                should_exit=False,
                should_enter=False,
                side=planned.setup.side,
                size=planned.size,
                reason="tick-news-blackout",
            )
        if not self._session.contains(ts):
            self._emit_candidate_event(planned.setup, status="invalidated", reason="outside-session")
            self._pending_setup = None
            return StrategyDecision(
                should_exit=False,
                should_enter=False,
                side=planned.setup.side,
                size=planned.size,
                reason="tick-outside-session",
            )

        self._record_tick_sample(ts, price)

        tick_bar = MarketBar(
            ts=ts,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=num(self._orderflow_state.trade, "size", "qty", "quantity", "volume", "lastSize") or 0.0,
        )
        if self.require_orderflow and self.orderflow_filter is not None:
            if not self.orderflow_filter.allow_entry(
                side=planned.setup.side,
                bar=tick_bar,
                context=context,
                flow=self._orderflow_state,
            ):
                self._emit_candidate_event(planned.setup, status="rejected", reason="tick-orderflow-gate")
                return StrategyDecision(
                    should_exit=False,
                    should_enter=False,
                    side=planned.setup.side,
                    size=planned.size,
                    reason="tick-orderflow-gate",
                )
        if not self._tick_entry_ready(planned.setup.side):
            return StrategyDecision(
                should_exit=False,
                should_enter=False,
                side=planned.setup.side,
                size=planned.size,
                reason="tick-wait-micro-timing",
            )

        refreshed_plan = self._build_entry_risk_plan(planned.setup, ml_score=planned.ml_score, entry_price=price)
        if refreshed_plan is None:
            self._emit_candidate_event(planned.setup, status="invalidated", reason="tick-risk-invalidated")
            return StrategyDecision(
                should_exit=False,
                should_enter=False,
                side=planned.setup.side,
                size=planned.size,
                reason="tick-risk-invalidated",
            )

        self._pending_setup = None
        self._emit_candidate_event(refreshed_plan.setup, status="entered", reason="entry-orderflow-sniper")
        return StrategyDecision(
            should_exit=False,
            should_enter=True,
            side=refreshed_plan.setup.side,
            size=refreshed_plan.size,
            reason="entry-orderflow-sniper",
            sl_ticks_abs=refreshed_plan.sl_ticks_abs,
            tp_ticks_abs=refreshed_plan.tp_ticks_abs,
        )

    def _build_setup_environment(
        self,
        side: int,
        bar: MarketBar,
        context: StrategyContext,
        trend_now: Trend,
        choch_bull: bool,
        choch_bear: bool,
    ) -> SetupEnvironment:
        has_recent_sweep_ = self._has_recent_sweep(side, context.index)
        bias = self._compute_htf_bias()
        bias_ok = not ((side == BUY and bias == "bearish") or (side == SELL and bias == "bullish"))
        continuation = (side == BUY and trend_now == "up") or (side == SELL and trend_now == "down")
        reversal = (side == BUY and choch_bull) or (side == SELL and choch_bear)
        equal_levels_ = self._equal_levels(side)
        fib_retracement_ = self._fib_retracement_confluence(side, bar.close)
        key_area_proximity_ = self._key_area_proximity(side, bar.close)
        confluence_score = int(equal_levels_) + int(fib_retracement_) + int(key_area_proximity_)
        return SetupEnvironment(
            side=side,
            index=context.index,
            ts=bar.ts,
            close=bar.close,
            has_recent_sweep=has_recent_sweep_,
            htf_bias=bias,
            bias_ok=bias_ok,
            continuation=continuation,
            reversal=reversal,
            equal_levels=equal_levels_,
            fib_retracement=fib_retracement_,
            key_area_proximity=key_area_proximity_,
            confluence_score=confluence_score,
        )

    def _setup_ready(self, setup: SetupEnvironment) -> bool:
        if not setup.has_recent_sweep:
            return False
        if not setup.bias_ok:
            return False
        if not (setup.continuation or setup.reversal):
            return False
        return setup.confluence_score >= self.min_confluence_score

    def _evaluate_ml(self, setup: SetupEnvironment) -> SetupMLDecision:
        if self.ml_decision_policy == "off":
            return SetupMLDecision(approved=True, score=None, reason="ml-bypassed")
        if self._ml_gate is None:
            return SetupMLDecision(approved=True, score=None, reason="ml-disabled")
        approved, score, reason = self._ml_gate.evaluate(setup)
        if self.ml_decision_policy == "shadow":
            return SetupMLDecision(approved=True, score=score, reason=f"ml-shadow:{reason}")
        return SetupMLDecision(approved=bool(approved), score=score, reason=reason)

    def _build_entry_risk_plan(
        self,
        setup: SetupEnvironment,
        ml_score: float | None,
        entry_price: float | None = None,
    ) -> PlannedEntry | None:
        planned_entry_price = setup.close if entry_price is None else float(entry_price)
        invalidation_levels = self._candidate_invalidation_levels(setup.side)
        stop_plan = self._stop_planner.plan(
            side=setup.side,
            entry_price=planned_entry_price,
            invalidation_levels=invalidation_levels,
        )
        if stop_plan is None:
            return None

        target_levels = self._candidate_target_levels(setup.side)
        take_profit_plan = self._take_profit_planner.plan(
            side=setup.side,
            entry_price=planned_entry_price,
            risk_ticks=stop_plan.ticks,
            target_levels=target_levels,
        )
        if take_profit_plan is None:
            return None

        size = self._size_from_risk_and_ml(stop_plan.ticks, ml_score)
        if size < 1:
            return None

        return PlannedEntry(
            setup=setup,
            size=size,
            sl_ticks_abs=stop_plan.ticks,
            tp_ticks_abs=take_profit_plan.ticks,
            stop_level=stop_plan.level,
            target_level=take_profit_plan.level,
            rrr=take_profit_plan.rrr,
            stop_order_type=stop_plan.order_type,
            take_profit_order_type=take_profit_plan.order_type,
            ml_score=ml_score,
        )

    def _candidate_invalidation_levels(self, side: int) -> list[float]:
        levels: list[float] = []
        if side == BUY:
            levels.extend(price for _, price in self._ltf_lows_only()[-6:])
            if self._htf_detector.current_low is not None:
                levels.append(self._htf_detector.current_low.price)
            levels.extend(level.price for level in self._htf_detector.past_lows[-6:])
            if self._last_sweep_side == BUY and self._last_sweep_price is not None:
                levels.append(self._last_sweep_price)
            support_level = self._dom_support_liquidity_level(side)
            if support_level is not None:
                levels.append(support_level)
        else:
            levels.extend(price for _, price in self._ltf_highs_only()[-6:])
            if self._htf_detector.current_high is not None:
                levels.append(self._htf_detector.current_high.price)
            levels.extend(level.price for level in self._htf_detector.past_highs[-6:])
            if self._last_sweep_side == SELL and self._last_sweep_price is not None:
                levels.append(self._last_sweep_price)
            support_level = self._dom_support_liquidity_level(side)
            if support_level is not None:
                levels.append(support_level)
        return levels

    def _candidate_target_levels(self, side: int) -> list[float]:
        levels: list[float] = []
        if side == BUY:
            levels.extend(price for _, price in self._ltf_highs_only()[-8:])
            if self._htf_detector.current_high is not None:
                levels.append(self._htf_detector.current_high.price)
            levels.extend(level.price for level in self._htf_detector.past_highs[-8:])
            opposing = self._dom_opposing_liquidity_level(side)
            if opposing is not None:
                levels.append(opposing)
        else:
            levels.extend(price for _, price in self._ltf_lows_only()[-8:])
            if self._htf_detector.current_low is not None:
                levels.append(self._htf_detector.current_low.price)
            levels.extend(level.price for level in self._htf_detector.past_lows[-8:])
            opposing = self._dom_opposing_liquidity_level(side)
            if opposing is not None:
                levels.append(opposing)
        return levels

    def _dom_support_liquidity_level(self, side: int) -> float | None:
        return dom_support_liquidity_level(self._orderflow_state, side, self.dom_liquidity_wall_size)

    def _dom_opposing_liquidity_level(self, side: int) -> float | None:
        return dom_opposing_liquidity_level(self._orderflow_state, side, self.dom_liquidity_wall_size)

    def _size_from_risk_and_ml(self, stop_ticks: int, ml_score: float | None) -> int:
        if stop_ticks <= 0:
            return 0
        per_contract_risk = stop_ticks * self.tick_value
        if per_contract_risk <= 0:
            return 0
        risk_budget = self.account_max_drawdown * self.max_trade_drawdown_fraction
        if risk_budget <= 0:
            return 0

        max_contracts = int(math.floor(risk_budget / per_contract_risk))
        if max_contracts < 1:
            return 0
        max_contracts = min(max_contracts, self.size)
        if max_contracts < 1:
            return 0

        if ml_score is None:
            return max_contracts

        score = self._clamp01(ml_score)
        floor_score = min(self.ml_size_floor_score, self.ml_size_ceiling_score)
        ceil_score = max(self.ml_size_floor_score, self.ml_size_ceiling_score)
        if ceil_score - floor_score < 1e-9:
            fraction = 1.0 if score >= ceil_score else self.ml_min_size_fraction
        elif score <= floor_score:
            fraction = self.ml_min_size_fraction
        elif score >= ceil_score:
            fraction = 1.0
        else:
            ratio = (score - floor_score) / (ceil_score - floor_score)
            fraction = self.ml_min_size_fraction + ratio * (1.0 - self.ml_min_size_fraction)
        sized = int(math.floor(max_contracts * fraction))
        return max(1, min(max_contracts, sized))

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _expire_pending_setup(self, bar_index: int) -> None:
        if self._pending_setup is None:
            return
        if (bar_index - self._pending_setup.setup.index) > self.tick_setup_expiry_bars:
            self._emit_candidate_event(self._pending_setup.setup, status="expired", reason="tick-setup-expiry")
            self._pending_setup = None

    @staticmethod
    def _side_name(side: int) -> str:
        return "buy" if side == BUY else "sell"

    @staticmethod
    def _candidate_id(setup: SetupEnvironment) -> str:
        side = "buy" if setup.side == BUY else "sell"
        return f"{setup.ts}:{setup.index}:{side}"

    def _emit_candidate_event(self, setup: SetupEnvironment, *, status: str, reason: str) -> None:
        best_bid = self._orderflow_state.best_bid()
        best_ask = self._orderflow_state.best_ask()
        spread = None
        if best_bid is not None and best_ask is not None:
            spread = float(best_ask) - float(best_bid)
        self._candidate_events.append(
            {
                "candidate_id": self._candidate_id(setup),
                "status": status,
                "reason": reason,
                "side": self._side_name(setup.side),
                "setup_index": setup.index,
                "setup_ts": setup.ts,
                "setup_close": setup.close,
                "has_recent_sweep": setup.has_recent_sweep,
                "htf_bias": setup.htf_bias,
                "bias_ok": setup.bias_ok,
                "continuation": setup.continuation,
                "reversal": setup.reversal,
                "equal_levels": setup.equal_levels,
                "fib_retracement": setup.fib_retracement,
                "key_area_proximity": setup.key_area_proximity,
                "confluence_score": setup.confluence_score,
                "of_imbalance": self._orderflow_state.imbalance(),
                "of_top_bid_size": self._orderflow_state.top_bid_size(),
                "of_top_ask_size": self._orderflow_state.top_ask_size(),
                "of_best_bid": best_bid,
                "of_best_ask": best_ask,
                "of_spread": spread,
                "of_trade_size": num(self._orderflow_state.trade, "size", "qty", "quantity", "volume", "lastSize"),
                "of_trade_price": num(self._orderflow_state.trade, "price", "last", "lastPrice", "tradePrice", "close"),
            }
        )

    @staticmethod
    def _normalize_news_blackouts(
        intervals: list[tuple[datetime, datetime]],
    ) -> list[tuple[datetime, datetime]]:
        normalized: list[tuple[datetime, datetime]] = []
        for raw_start, raw_end in intervals:
            start = raw_start if raw_start.tzinfo is not None else raw_start.replace(tzinfo=timezone.utc)
            end = raw_end if raw_end.tzinfo is not None else raw_end.replace(tzinfo=timezone.utc)
            start_utc = start.astimezone(timezone.utc)
            end_utc = end.astimezone(timezone.utc)
            if end_utc < start_utc:
                start_utc, end_utc = end_utc, start_utc
            normalized.append((start_utc, end_utc))
        if not normalized:
            return []
        normalized.sort(key=lambda x: x[0])
        merged: list[tuple[datetime, datetime]] = []
        for start, end in normalized:
            if not merged:
                merged.append((start, end))
                continue
            prev_start, prev_end = merged[-1]
            if start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, end))
                continue
            merged.append((start, end))
        return merged

    def _is_news_blackout(self, ts: str) -> bool:
        if not self.avoid_news or not self.news_blackouts_utc:
            return False
        try:
            now_utc = _parse_ts(ts).astimezone(timezone.utc)
        except ValueError:
            return False
        for start, end in self.news_blackouts_utc:
            if start <= now_utc < end:
                return True
        return False

    def _record_tick_sample(self, ts: str, price: float) -> None:
        try:
            ts_dt = _parse_ts(ts)
        except ValueError:
            ts_dt = datetime.now(timezone.utc)
        sample = TickFlowSample(
            ts=ts_dt,
            price=price,
            bid=self._orderflow_state.best_bid(),
            ask=self._orderflow_state.best_ask(),
            bid_size=self._orderflow_state.top_bid_size(),
            ask_size=self._orderflow_state.top_ask_size(),
            imbalance=self._orderflow_state.imbalance(),
            trade_price=num(self._orderflow_state.trade, "price", "last", "lastPrice", "tradePrice", "close"),
            trade_size=num(self._orderflow_state.trade, "size", "qty", "quantity", "volume", "lastSize") or 0.0,
        )
        self._tick_samples.append(sample)

    def _tick_entry_ready(self, side: int) -> bool:
        return tick_entry_ready(self._tick_samples, side, self._tick_execution_config())

    def _tick_exhaustion_exit_signal(self, side: int) -> bool:
        return tick_exhaustion_exit_signal(self._tick_samples, side, self._tick_execution_config())

    def _tick_execution_config(self) -> TickExecutionConfig:
        return TickExecutionConfig(
            min_imbalance=self.tick_min_imbalance,
            min_trade_size=self.tick_min_trade_size,
            spoof_collapse_ratio=self.tick_spoof_collapse_ratio,
            absorption_min_trades=self.tick_absorption_min_trades,
            iceberg_min_reloads=self.tick_iceberg_min_reloads,
            tick_size=self.tick_size,
        )

    def _update_htf(self, bar: MarketBar, index: int) -> None:
        self._htf_buffer.append(bar)
        if len(self._htf_buffer) < self.htf_aggregation:
            return

        htf_bar = self._aggregate_bars(self._htf_buffer)
        self._htf_buffer.clear()
        snapshot = self._htf_detector.update(htf_bar)
        sweep_high, sweep_low = self._detect_htf_sweep(htf_bar, snapshot)
        if sweep_high:
            self._last_sweep_side = SELL
            self._last_sweep_index = index
            self._last_sweep_price = htf_bar.high
        elif sweep_low:
            self._last_sweep_side = BUY
            self._last_sweep_index = index
            self._last_sweep_price = htf_bar.low

    def _update_bias(self, bar: MarketBar) -> None:
        self._bias_buffer.append(bar)
        if len(self._bias_buffer) < self.bias_aggregation:
            return
        bias_bar = self._aggregate_bars(self._bias_buffer)
        self._bias_buffer.clear()
        self._bias_detector.update(bias_bar)

    def _aggregate_bars(self, bars: list[MarketBar]) -> MarketBar:
        return aggregate_bars(bars)

    def _detect_htf_sweep(self, bar: MarketBar, snapshot) -> tuple[bool, bool]:
        return detect_htf_sweep(bar, snapshot, self.wick_sweep_ratio_min)

    def _compute_ltf_trend(self) -> Trend:
        return compute_ltf_trend(self._ltf_high_points, self._ltf_low_points)

    def _compute_htf_bias(self) -> Bias:
        return compute_htf_bias(self._bias_detector)

    def _is_choch_bullish(self, trend_before: Trend, close: float) -> bool:
        return is_choch_bullish(trend_before, close, self._ltf_high_points)

    def _is_choch_bearish(self, trend_before: Trend, close: float) -> bool:
        return is_choch_bearish(trend_before, close, self._ltf_low_points)

    def _has_recent_sweep(self, side: int, index: int) -> bool:
        return has_recent_sweep(
            last_sweep_side=self._last_sweep_side,
            last_sweep_index=self._last_sweep_index,
            side=side,
            index=index,
            sweep_expiry_bars=self.sweep_expiry_bars,
        )

    def _equal_levels(self, side: int) -> bool:
        return equal_levels(side, self._ltf_high_points, self._ltf_low_points, self.equal_level_tolerance_bps)

    def _fib_retracement_confluence(self, side: int, close: float) -> bool:
        return fib_retracement_confluence(side, close, self._ltf_high_points, self._ltf_low_points)

    def _key_area_proximity(self, side: int, close: float) -> bool:
        return key_area_proximity(side, close, self._htf_detector, self.key_area_tolerance_bps)

    def _trim_points(self) -> None:
        max_points = 300
        if len(self._ltf_high_points) > max_points:
            del self._ltf_high_points[: len(self._ltf_high_points) - max_points]
        if len(self._ltf_low_points) > max_points:
            del self._ltf_low_points[: len(self._ltf_low_points) - max_points]

    def _ltf_highs_only(self) -> list[tuple[int, float]]:
        return self._ltf_high_points

    def _ltf_lows_only(self) -> list[tuple[int, float]]:
        return self._ltf_low_points
