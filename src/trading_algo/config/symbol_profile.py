from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolProfile:
    tick_size: float
    tick_value: float
    dom_liquidity_wall_size: float


_DEFAULT_PROFILE = SymbolProfile(tick_size=0.25, tick_value=0.5, dom_liquidity_wall_size=800.0)

_PROFILES: dict[str, SymbolProfile] = {
    # Micro equity index futures.
    "MNQ": SymbolProfile(tick_size=0.25, tick_value=0.5, dom_liquidity_wall_size=800.0),
    "MES": SymbolProfile(tick_size=0.25, tick_value=1.25, dom_liquidity_wall_size=800.0),
    # Standard equity index futures.
    "NQ": SymbolProfile(tick_size=0.25, tick_value=5.0, dom_liquidity_wall_size=1_000.0),
    "ES": SymbolProfile(tick_size=0.25, tick_value=12.5, dom_liquidity_wall_size=1_000.0),
    # Gold futures.
    "MGC": SymbolProfile(tick_size=0.1, tick_value=1.0, dom_liquidity_wall_size=600.0),
    "GC": SymbolProfile(tick_size=0.1, tick_value=10.0, dom_liquidity_wall_size=1_000.0),
}


def get_symbol_profile(symbol: str) -> SymbolProfile:
    key = (symbol or "").strip().upper()
    return _PROFILES.get(key, _DEFAULT_PROFILE)

