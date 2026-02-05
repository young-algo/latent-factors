"""Tests for Decision Synthesizer module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.decision_synthesizer import SignalState, RegimeState, FactorMomentum


class TestSignalState:
    """Tests for SignalState data structure."""

    def test_signal_state_creation(self):
        """SignalState should hold all signal components."""
        regime = RegimeState(
            name="Low-Vol Bull",
            confidence=0.78,
            days_in_regime=12,
            trend="strengthening"
        )

        momentum = [
            FactorMomentum(factor="F1", name="Tech-Momentum", return_7d=0.021, strength="strong"),
            FactorMomentum(factor="F2", name="Quality", return_7d=0.008, strength="moderate"),
        ]

        state = SignalState(
            date=datetime(2026, 2, 4),
            regime=regime,
            factor_momentum=momentum,
            extremes_detected=[],
            cross_sectional_spread=1.2
        )

        assert state.regime.name == "Low-Vol Bull"
        assert state.regime.confidence == 0.78
        assert len(state.factor_momentum) == 2
        assert state.factor_momentum[0].strength == "strong"
