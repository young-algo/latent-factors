from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional


STATE_VERSION = 1


@dataclass
class Watchlists:
    factors: List[str] = field(default_factory=list)
    tickers: List[str] = field(default_factory=list)


@dataclass
class DashboardState:
    version: int = STATE_VERSION
    last_page: str = "Home"
    universe: str = "VTHR"
    as_of: str = field(default_factory=lambda: date.today().isoformat())
    lookback: int = 63
    selected_factor: Optional[str] = None
    watchlists: Watchlists = field(default_factory=Watchlists)
    saved_views: List[Dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> str:
        payload = asdict(self)
        payload["watchlists"] = asdict(self.watchlists)
        return json.dumps(payload, indent=2, sort_keys=True)

    @classmethod
    def from_mapping(cls, payload: Dict[str, Any]) -> "DashboardState":
        raw_watchlists = payload.get("watchlists") or {}
        watchlists = Watchlists(
            factors=list(raw_watchlists.get("factors") or []),
            tickers=list(raw_watchlists.get("tickers") or []),
        )
        return cls(
            version=int(payload.get("version") or STATE_VERSION),
            last_page=str(payload.get("last_page") or "Home"),
            universe=str(payload.get("universe") or "VTHR"),
            as_of=str(payload.get("as_of") or date.today().isoformat()),
            lookback=int(payload.get("lookback") or 63),
            selected_factor=payload.get("selected_factor") or None,
            watchlists=watchlists,
            saved_views=list(payload.get("saved_views") or []),
        )


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_state(path: Path) -> DashboardState:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return DashboardState()

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return DashboardState()

    if not isinstance(payload, dict):
        return DashboardState()

    return DashboardState.from_mapping(payload)


def save_state(state: DashboardState, path: Path) -> None:
    _ensure_parent(path)
    path.write_text(state.to_json() + "\n", encoding="utf-8")


def state_equals(a: DashboardState, b: DashboardState) -> bool:
    return asdict(a) == asdict(b)

