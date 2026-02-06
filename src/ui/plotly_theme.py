from __future__ import annotations

from dataclasses import dataclass

import plotly.graph_objects as go


@dataclass(frozen=True)
class PlotTheme:
    template: str = "plotly_white"
    font_family: str = "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"
    font_color: str = "#111827"
    grid_color: str = "#E5E7EB"


DEFAULT_THEME = PlotTheme()


def apply_light_theme(fig: go.Figure, theme: PlotTheme = DEFAULT_THEME) -> go.Figure:
    fig.update_layout(
        template=theme.template,
        font=dict(family=theme.font_family, color=theme.font_color),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    fig.update_xaxes(showgrid=True, gridcolor=theme.grid_color, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=theme.grid_color, zeroline=False)
    return fig

