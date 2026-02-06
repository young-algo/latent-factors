from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from src.ui.state import DashboardState


@dataclass(frozen=True)
class AppPaths:
    project_root: Path
    state_path: Path
    db_path: Path


@dataclass
class AppData:
    factor_returns: Optional[pd.DataFrame]
    factor_loadings: Optional[pd.DataFrame]
    factor_names: dict[str, str]

    portfolio_returns: Optional[pd.Series]
    benchmark_returns: Optional[pd.Series]

    latest_trade_basket: Optional[pd.DataFrame]
    factor_qa_report: Optional[dict]
    factor_analysis_config: Optional[dict]


@dataclass
class AppContext:
    paths: AppPaths
    state: DashboardState
    data: AppData

