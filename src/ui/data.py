from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import streamlit as st

from src.database import check_database_health, get_database_path


@st.cache_data(ttl=300)
def load_factor_returns(path: str) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
    except Exception:
        return None

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index().replace([float("inf"), float("-inf")], pd.NA).dropna(how="all")
    return df if not df.empty else None


@st.cache_data(ttl=300)
def load_factor_loadings(path: str) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, index_col=0)
    except Exception:
        return None
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    return df if not df.empty else None


@st.cache_data(ttl=300)
def load_factor_names(project_root: str) -> dict[str, str]:
    root = Path(project_root)
    json_path = root / "factor_names.json"
    csv_path = root / "factor_names.csv"

    if json_path.exists():
        try:
            return json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    if csv_path.exists():
        try:
            # Try header-aware formats first.
            frame = pd.read_csv(csv_path)
            cols = {c.lower(): c for c in frame.columns}
            if "factor" in cols and "name" in cols:
                return dict(
                    zip(
                        frame[cols["factor"]].astype(str),
                        frame[cols["name"]].astype(str),
                    )
                )

            # Fallback: two-column no-header format.
            frame2 = pd.read_csv(csv_path, header=None)
            if frame2.shape[1] >= 2:
                return dict(zip(frame2.iloc[:, 0].astype(str), frame2.iloc[:, 1].astype(str)))
        except Exception:
            return {}

    return {}


def _load_series_from_csv(path: Path, preferred_names: list[str]) -> Optional[pd.Series]:
    try:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return None

    if frame.empty:
        return None

    if frame.shape[1] == 1:
        series = frame.iloc[:, 0]
    else:
        normalized = {c.lower(): c for c in frame.columns}
        selected_col = None
        for name in preferred_names:
            if name.lower() in normalized:
                selected_col = normalized[name.lower()]
                break
        if selected_col is None:
            selected_col = frame.columns[0]
        series = frame[selected_col]

    series = series.sort_index()
    series.index = pd.to_datetime(series.index, errors="coerce")
    series = pd.to_numeric(series, errors="coerce").dropna()
    return series if not series.empty else None


@st.cache_data(ttl=300)
def load_portfolio_inputs(project_root: str) -> tuple[Optional[pd.Series], Optional[pd.Series]]:
    root = Path(project_root)

    portfolio_returns = None
    benchmark_returns = None

    for fname in ("portfolio_returns.csv", "strategy_returns.csv"):
        path = root / fname
        if path.exists():
            portfolio_returns = _load_series_from_csv(path, preferred_names=["portfolio", "strategy", "return"])
            if portfolio_returns is not None:
                break

    for fname in ("benchmark_returns.csv", "spy_returns.csv"):
        path = root / fname
        if path.exists():
            benchmark_returns = _load_series_from_csv(path, preferred_names=["benchmark", "spy", "return"])
            if benchmark_returns is not None:
                break

    return portfolio_returns, benchmark_returns


@st.cache_data(ttl=300)
def load_latest_trade_basket(project_root: str) -> Optional[pd.DataFrame]:
    root = Path(project_root)
    candidates = [
        "vthr_pca_trade_basket_07_net.csv",
        "basket_fixed.csv",
        "vthr_pca_long_positions.csv",
        "vthr_pca_short_positions.csv",
    ]
    for fname in candidates:
        path = root / fname
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if not df.empty:
            df.attrs["source_path"] = str(path)
            return df
    return None


@st.cache_data(ttl=300)
def load_factor_qa_report(project_root: str) -> Optional[dict]:
    root = Path(project_root)
    report_path = root / "factor_qa_report.json"
    if not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_factor_analysis_config(project_root: str) -> Optional[dict]:
    root = Path(project_root)
    path = root / "factor_analysis_config.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


@st.cache_data(ttl=3600)
def load_fundamentals_from_cache(
    tickers: Iterable[str],
    db_path: Optional[str] = None,
    fields: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load cached fundamentals from SQLite without making network calls."""
    tickers = [t.strip().upper() for t in tickers if t and str(t).strip()]
    if not tickers:
        return pd.DataFrame()

    db = Path(db_path) if db_path else get_database_path()
    if not db.exists():
        return pd.DataFrame()

    placeholders = ",".join(["?"] * len(tickers))
    sql = f"SELECT ticker, json FROM fundamentals WHERE ticker IN ({placeholders})"

    records: dict[str, dict] = {}
    try:
        with sqlite3.connect(str(db)) as con:
            for ticker, raw_json in con.execute(sql, tickers).fetchall():
                if not raw_json:
                    continue
                try:
                    rec = json.loads(raw_json)
                except Exception:
                    continue
                if fields:
                    rec = {k: rec.get(k) for k in fields}
                records[str(ticker).upper()] = rec
    except Exception:
        return pd.DataFrame()

    return pd.DataFrame.from_dict(records, orient="index")


def get_db_health() -> dict:
    try:
        return check_database_health()
    except Exception as exc:
        return {"is_healthy": False, "error": str(exc)}
