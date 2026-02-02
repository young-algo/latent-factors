"""
Unit tests for regime_detection module.

Tests cover:
- HMM model fitting
- Regime detection
- Regime probability estimation
- Regime-based factor allocation
"""

import pytest
import numpy as np
import pandas as pd
from src import (
    RegimeDetector,
    SimpleRegimeDetector,
    MarketRegime,
    RegimeState,
    RegimeAllocation
)


class TestRegimeDetector:
    """Test suite for RegimeDetector."""

    @pytest.fixture
    def sample_factor_returns(self):
        """Create sample factor returns for testing."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=500, freq='D')

        # Create factors with different regimes
        # Low vol bull: small positive returns, low volatility
        low_vol_bull = np.random.randn(500) * 0.005 + 0.001

        # High vol bear: negative returns, high volatility
        high_vol_bear = np.random.randn(500) * 0.02 - 0.002

        # Normal market
        normal = np.random.randn(500) * 0.01

        returns = pd.DataFrame({
            'F1': low_vol_bull,
            'F2': high_vol_bear,
            'F3': normal
        }, index=dates)

        return returns

    @pytest.fixture
    def detector(self, sample_factor_returns):
        """Create RegimeDetector instance."""
        return RegimeDetector(sample_factor_returns)

    def test_initialization(self, sample_factor_returns):
        """Test detector initialization."""
        detector = RegimeDetector(sample_factor_returns)

        assert detector.n_factors == 3
        assert len(detector.factor_returns) == 500
        assert detector.hmm_model is None  # Not fitted yet

    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises ValueError."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty"):
            RegimeDetector(empty_df)

    def test_short_data_validation(self):
        """Test that short data series triggers validation check."""
        short_data = pd.DataFrame(
            np.random.randn(100, 3),
            columns=['F1', 'F2', 'F3'],
            index=pd.date_range('2023-01-01', periods=100, freq='D')
        )

        # Should create detector but with validation warning in logs
        detector = RegimeDetector(short_data)
        assert detector.n_factors == 3
        assert len(detector.factor_returns) == 100

    def test_fit_hmm(self, detector):
        """Test HMM fitting."""
        detector.fit_hmm(n_regimes=3)

        # Should have fitted model
        assert detector.hmm_model is not None

        # Should have regime labels
        assert len(detector.regime_labels) == 3

        # Should have regime history
        assert detector.regime_history is not None
        assert len(detector.regime_history) > 0

    def test_fit_hmm_different_regime_counts(self, detector):
        """Test HMM fitting with different regime counts."""
        for n_regimes in [2, 3, 4]:
            detector.fit_hmm(n_regimes=n_regimes)
            assert detector.hmm_model.n_components == n_regimes

    def test_fit_hmm_invalid_data(self, detector):
        """Test HMM fitting with insufficient data."""
        # Create detector with very short data
        short_data = pd.DataFrame(
            np.random.randn(50, 3),
            columns=['F1', 'F2', 'F3'],
            index=pd.date_range('2023-01-01', periods=50, freq='D')
        )
        short_detector = RegimeDetector(short_data)

        with pytest.raises(ValueError, match="Insufficient"):
            short_detector.fit_hmm()

    def test_detect_current_regime(self, detector):
        """Test current regime detection."""
        detector.fit_hmm(n_regimes=3)
        regime = detector.detect_current_regime()

        # Should return RegimeState
        assert isinstance(regime, RegimeState)

        # Should have expected attributes
        assert isinstance(regime.regime, MarketRegime)
        assert 0 <= regime.probability <= 1
        assert regime.volatility >= 0

    def test_detect_current_regime_not_fitted(self, detector):
        """Test that detect_current_regime raises error if not fitted."""
        with pytest.raises(RuntimeError, match="not fitted"):
            detector.detect_current_regime()

    def test_get_regime_probabilities(self, detector):
        """Test regime probability estimation."""
        detector.fit_hmm(n_regimes=3)
        probs = detector.get_regime_probabilities()

        # Should return dictionary
        assert isinstance(probs, dict)

        # Probabilities should sum to 1
        assert abs(sum(probs.values()) - 1.0) < 0.01

        # All keys should be MarketRegime
        for regime in probs.keys():
            assert isinstance(regime, MarketRegime)

    def test_get_regime_probabilities_not_fitted(self, detector):
        """Test that get_regime_probabilities raises error if not fitted."""
        with pytest.raises(RuntimeError, match="not fitted"):
            detector.get_regime_probabilities()

    def test_get_regime_optimal_factors(self, detector):
        """Test regime-based factor allocation."""
        detector.fit_hmm(n_regimes=3)
        current_regime = detector.detect_current_regime().regime

        optimal = detector.get_regime_optimal_factors(current_regime)

        # Should return dictionary
        assert isinstance(optimal, dict)

        # Should have at least one factor weight
        assert len(optimal) >= 1

        # Weights should sum to 1
        assert abs(sum(optimal.values()) - 1.0) < 0.01

    def test_get_regime_optimal_factors_not_fitted(self, detector):
        """Test that get_regime_optimal_factors raises error if not fitted."""
        with pytest.raises(RuntimeError, match="not fitted"):
            detector.get_regime_optimal_factors(MarketRegime.LOW_VOL_BULL)

    def test_generate_regime_signals(self, detector):
        """Test regime signal generation."""
        detector.fit_hmm(n_regimes=3)
        allocation = detector.generate_regime_signals()

        # Should return RegimeAllocation
        assert isinstance(allocation, RegimeAllocation)

        # Should have expected attributes
        assert isinstance(allocation.regime, MarketRegime)
        assert 0 <= allocation.risk_on_score <= 1
        assert isinstance(allocation.defensive_tilt, bool)
        assert isinstance(allocation.recommended_action, str)

    def test_analyze_regime_transitions(self, detector):
        """Test regime transition analysis."""
        detector.fit_hmm(n_regimes=3)
        transitions = detector.analyze_regime_transitions()

        # Should return DataFrame
        assert isinstance(transitions, pd.DataFrame)

        # Should be square matrix
        assert transitions.shape[0] == transitions.shape[1]

        # Rows should sum to 1 (probabilities)
        for idx in transitions.index:
            assert abs(transitions.loc[idx].sum() - 1.0) < 0.01

    def test_predict_regime(self, detector):
        """Test regime prediction."""
        detector.fit_hmm(n_regimes=3)
        predictions = detector.predict_regime(duration=5)

        # Should return list
        assert isinstance(predictions, list)
        assert len(predictions) == 5

        # All should be RegimeState
        for pred in predictions:
            assert isinstance(pred, RegimeState)

    def test_get_regime_summary(self, detector):
        """Test regime summary generation."""
        detector.fit_hmm(n_regimes=3)
        summary = detector.get_regime_summary()

        # Should return DataFrame
        assert isinstance(summary, pd.DataFrame)

        # Should have expected columns
        expected_cols = ['regime', 'count', 'pct_time', 'avg_return', 'avg_volatility', 'sharpe']
        for col in expected_cols:
            assert col in summary.columns


class TestSimpleRegimeDetector:
    """Test suite for SimpleRegimeDetector."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample factor returns."""
        np.random.seed(42)
        returns = pd.DataFrame({
            'F1': np.random.randn(100) * 0.01,
            'F2': np.random.randn(100) * 0.01
        }, index=pd.date_range('2023-01-01', periods=100, freq='D'))
        return returns

    def test_simple_detection(self, sample_returns):
        """Test simple regime detection."""
        detector = SimpleRegimeDetector(sample_returns)
        regime = detector.detect_current_regime()

        # Should return RegimeState
        assert isinstance(regime, RegimeState)

        # Should have valid regime
        assert isinstance(regime.regime, MarketRegime)

    def test_simple_optimal_factors(self, sample_returns):
        """Test simple optimal factor calculation."""
        detector = SimpleRegimeDetector(sample_returns)
        weights = detector.get_regime_optimal_factors(MarketRegime.LOW_VOL_BULL)

        # Should return dictionary with equal weights
        assert isinstance(weights, dict)
        assert len(weights) == 2
        assert abs(sum(weights.values()) - 1.0) < 0.01


class TestMarketRegime:
    """Test suite for MarketRegime enum."""

    def test_regime_values(self):
        """Test regime enum has expected values."""
        assert MarketRegime.LOW_VOL_BULL.value == 'low_volatility_bull'
        assert MarketRegime.HIGH_VOL_BULL.value == 'high_volatility_bull'
        assert MarketRegime.LOW_VOL_BEAR.value == 'low_volatility_bear'
        assert MarketRegime.HIGH_VOL_BEAR.value == 'high_volatility_bear'
        assert MarketRegime.TRANSITION.value == 'transition'
        assert MarketRegime.CRISIS.value == 'crisis'
        assert MarketRegime.UNKNOWN.value == 'unknown'


class TestRegimeState:
    """Test suite for RegimeState dataclass."""

    def test_regime_state_creation(self):
        """Test RegimeState dataclass creation."""
        state = RegimeState(
            regime=MarketRegime.LOW_VOL_BULL,
            probability=0.85,
            volatility=0.12,
            trend=0.001,
            description="Low volatility bull market"
        )

        assert state.regime == MarketRegime.LOW_VOL_BULL
        assert state.probability == 0.85
        assert state.description == "Low volatility bull market"


class TestRegimeAllocation:
    """Test suite for RegimeAllocation dataclass."""

    def test_regime_allocation_creation(self):
        """Test RegimeAllocation dataclass creation."""
        allocation = RegimeAllocation(
            regime=MarketRegime.LOW_VOL_BULL,
            factor_weights={'F1': 0.5, 'F2': 0.5},
            risk_on_score=0.8,
            defensive_tilt=False,
            recommended_action="Overweight momentum factors"
        )

        assert allocation.regime == MarketRegime.LOW_VOL_BULL
        assert allocation.risk_on_score == 0.8
        assert not allocation.defensive_tilt


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
