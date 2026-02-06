"""
Equity Factors Dashboard (Streamlit)
====================================

Thin Streamlit entrypoint that delegates rendering to `src.ui`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.app import render_app  # noqa: E402


st.set_page_config(
    page_title="Equity Factors Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    render_app()


if __name__ == "__main__":
    main()
