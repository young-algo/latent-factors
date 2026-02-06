from __future__ import annotations

import streamlit as st


_BADGE = {
    "ok": {"bg": "#ECFDF5", "fg": "#065F46", "border": "#A7F3D0"},
    "watch": {"bg": "#FFFBEB", "fg": "#92400E", "border": "#FDE68A"},
    "action": {"bg": "#FEF2F2", "fg": "#991B1B", "border": "#FECACA"},
    "neutral": {"bg": "#F3F4F6", "fg": "#374151", "border": "#E5E7EB"},
}


def badge(text: str, kind: str = "neutral") -> None:
    style = _BADGE.get(kind, _BADGE["neutral"])
    st.markdown(
        (
            "<span "
            "style="
            f"\"display:inline-block; padding:2px 8px; border-radius:999px;"
            f" background:{style['bg']}; color:{style['fg']};"
            f" border:1px solid {style['border']};"
            " font-size:12px; font-weight:600;\""
            f">{text}</span>"
        ),
        unsafe_allow_html=True,
    )


def status_kind_from_threshold(value: float | None, ok_max_abs: float, watch_max_abs: float) -> str:
    if value is None:
        return "neutral"
    v = abs(float(value))
    if v <= ok_max_abs:
        return "ok"
    if v <= watch_max_abs:
        return "watch"
    return "action"

