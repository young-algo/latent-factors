from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from src.ui.plotly_theme import apply_light_theme


def cumulative_return_chart(series: pd.Series, title: str = "Cumulative return") -> go.Figure:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False)
        return apply_light_theme(fig)

    cumulative = (1.0 + series).cumprod() - 1.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cumulative.index, y=cumulative.values, mode="lines", name="Cumulative"))
    fig.update_layout(title=title, yaxis_tickformat=".0%")
    return apply_light_theme(fig)


def drawdown_chart(series: pd.Series, title: str = "Drawdown") -> go.Figure:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False)
        return apply_light_theme(fig)

    cumulative = (1.0 + series).cumprod()
    running_max = cumulative.expanding().max()
    dd = (cumulative - running_max) / running_max

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown", line=dict(color="#DC2626")))
    fig.update_layout(title=title, yaxis_tickformat=".0%")
    return apply_light_theme(fig)


def regime_probability_bars(probabilities: dict[str, float], title: str = "Regime probabilities") -> go.Figure:
    if not probabilities:
        fig = go.Figure()
        fig.add_annotation(text="No regime probabilities available", showarrow=False)
        return apply_light_theme(fig)

    items = sorted(probabilities.items(), key=lambda kv: kv[1])
    labels = [k.replace("_", " ").title() for k, _ in items]
    values = [float(v) for _, v in items]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=values, y=labels, orientation="h", marker_color="#2563EB"))
    fig.update_layout(title=title, xaxis_tickformat=".0%")
    return apply_light_theme(fig)

