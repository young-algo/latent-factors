from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import streamlit as st


def dataframe(
    df: pd.DataFrame,
    *,
    key: str,
    height: Optional[int] = None,
    selection_mode: str = "single-row",
) -> Any:
    """Wrapper for consistent dataframe behavior across pages."""
    return st.dataframe(
        df,
        key=key,
        hide_index=True,
        use_container_width=True,
        height=height or 360,
        on_select="rerun",
        selection_mode=selection_mode,
    )
