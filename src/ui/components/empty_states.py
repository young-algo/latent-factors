from __future__ import annotations

from typing import Iterable

import streamlit as st


def missing_data(
    title: str,
    message: str,
    steps: Iterable[str] = (),
) -> None:
    st.error(title)
    st.write(message)
    if steps:
        st.markdown("**Next steps**")
        for step in steps:
            st.markdown(f"- {step}")

