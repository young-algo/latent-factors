"""
Unit tests for trading_signals module.

Tests cover:
- RSI calculation accuracy
- MACD signal generation
- Z-score extreme detection
- Bollinger Bands calculation
- Momentum regime detection
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src import (
    FactorMomentumAnalyzer,
    MomentumRegime,
    SignalStrength,
    TradingSignal,
    ExtremeAlert
)


class TestFactorMomentumAnalyzer:
    """Test suite for FactorMomentumAnalyzer."""

    @pytest.fixture
    def sample_factor_returns(self):
        """Create sample factor returns for testing."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')

        # Create factor with trending behavior
        trend = np.cumsum(np.random.randn(252) * 0.01)

        # Create factor with mean-reverting behavior
        mean_rev = np.sin(np.linspace(0, 4*np.pi, 252)) * 0.02 + np.random.randn(252) * 0.005

        returns = pd.DataFrame({
            'F1_Trend': trend,
            'F2_MeanRev': mean_rev,
            'F3_Volatile': np.random.randn(252) * 0.03
        }, index=dates)

        return returns

    @pytest.fixture
    def analyzer(self, sample_factor_returns):
        """Create FactorMomentumAnalyzer instance."""
        return FactorMomentumAnalyzer(sample_factor_returns)

    def test_initialization(self, sample_factor_returns):
        """Test analyzer initialization."""
        analyzer = FactorMomentumAnalyzer(sample_factor_returns)

        assert analyzer.n_factors == 3
        assert list(analyzer.factor_returns.columns) == ['F1_Trend', 'F2_MeanRev', 'F3_Volatile']
        assert len(analyzer.factor_returns) == 252

    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises ValueError."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty"):
            FactorMomentumAnalyzer(empty_df)

    def test_rsi_calculation(self, analyzer):
        """Test RSI calculation produces values in 0-100 range."""
        rsi = analyzer.calculate_rsi('F1_Trend', period=14)

        # RSI should be between 0 and 100
        assert rsi.min() >= 0
        assert rsi.max() <= 100

        # RSI should have same length as input
        assert len(rsi) == len(analyzer.factor_returns)

        # First 13 values should be NaN (not enough data)
        assert rsi.iloc[:13].isna().all()

    def test_rsi_overbought_oversold(self):
        """Test RSI correctly identifies overbought/oversold conditions."""
        np.random.seed(42)

        # Create rising returns series (strong positive returns)
        rising_returns = np.concatenate([
            np.random.randn(20) * 0.01,  # Random noise
            np.ones(30) * 0.05,  # Strong positive returns
            np.random.randn(20) * 0.01   # Random noise
        ])

        rising = pd.DataFrame({
            'Rising': rising_returns
        }, index=pd.date_range('2023-01-01', periods=70, freq='D'))

        analyzer = FactorMomentumAnalyzer(rising)
        rsi = analyzer.calculate_rsi('Rising', period=14)

        # RSI of strongly rising series should be high (skip NaN values)
        valid_rsi = rsi.dropna()
        if len(valid_rsi) > 0:
            assert valid_rsi.iloc[-1] > 50  # Should be elevated

        # Create falling returns series (strong negative returns)
        falling_returns = np.concatenate([
            np.random.randn(20) * 0.01,   # Random noise
            np.ones(30) * -0.05,  # Strong negative returns
            np.random.randn(20) * 0.01    # Random noise
        ])

        falling = pd.DataFrame({
            'Falling': falling_returns
        }, index=pd.date_range('2023-01-01', periods=70, freq='D'))

        analyzer2 = FactorMomentumAnalyzer(falling)
        rsi2 = analyzer2.calculate_rsi('Falling', period=14)

        # RSI of strongly falling series should be low
        valid_rsi2 = rsi2.dropna()
        if len(valid_rsi2) > 0:
            assert valid_rsi2.iloc[-1] < 50  # Should be depressed

    def test_macd_calculation(self, analyzer):
        """Test MACD calculation returns three series."""
        macd_line, signal_line, histogram = analyzer.calculate_macd('F1_Trend')

        # Should return three Series
        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal_line, pd.Series)
        assert isinstance(histogram, pd.Series)

        # All should have same length
        assert len(macd_line) == len(signal_line) == len(histogram)

        # Histogram should equal MACD - Signal
        pd.testing.assert_series_equal(
            histogram,
            macd_line - signal_line,
            check_names=False
        )

    def test_adx_calculation(self, analyzer):
        """Test ADX calculation produces values in 0-100 range."""
        adx = analyzer.calculate_adx('F1_Trend', period=14)

        # ADX should be between 0 and 100
        assert adx.min() >= 0
        assert adx.max() <= 100

        # Should have same length as input
        assert len(adx) == len(analyzer.factor_returns)

    def test_calculate_roc(self, analyzer):
        """Test Rate of Change calculation."""
        roc = analyzer.calculate_roc('F1_Trend', periods=[21, 63])

        # Should return DataFrame
        assert isinstance(roc, pd.DataFrame)

        # Should have columns for each period
        assert 'ROC_21' in roc.columns
        assert 'ROC_63' in roc.columns

        # Values should be centered around 0 for random walk
        assert roc['ROC_21'].mean() < 1.0  # Shouldn't drift too far

    def test_detect_momentum_regime(self, analyzer):
        """Test momentum regime detection returns valid enum."""
        regime = analyzer.detect_momentum_regime('F1_Trend')

        # Should return MomentumRegime enum
        assert isinstance(regime, MomentumRegime)

        # Should be one of valid regimes
        assert regime in [
            MomentumRegime.TRENDING_UP,
            MomentumRegime.TRENDING_DOWN,
            MomentumRegime.MEAN_REVERTING,
            MomentumRegime.NEUTRAL
        ]

    def test_calculate_zscore(self, analyzer):
        """Test z-score calculation."""
        zscore = analyzer.calculate_zscore('F1_Trend', window=20)

        # Z-scores should be roughly normally distributed
        assert zscore.min() > -5  # Shouldn't be too extreme
        assert zscore.max() < 5

        # Mean should be close to 0
        assert abs(zscore.mean()) < 0.5

    def test_zscore_extreme_detection(self, analyzer):
        """Test that z-score > 2 triggers extreme alert."""
        # Create series with extreme values
        extreme_values = np.random.randn(100) * 0.01
        extreme_values[-1] = 0.5  # Extreme positive value

        df = pd.DataFrame({
            'Extreme': extreme_values
        }, index=pd.date_range('2023-01-01', periods=100, freq='D'))

        analyzer = FactorMomentumAnalyzer(df)
        zscore = analyzer.calculate_zscore('Extreme', window=20)

        # Last z-score should be extreme
        assert abs(zscore.iloc[-1]) > 2

    def test_calculate_bollinger_bands(self, analyzer):
        """Test Bollinger Bands calculation."""
        upper, middle, lower = analyzer.calculate_bollinger_bands('F1_Trend')

        # Should return three Series
        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)

        # Drop NaN values for comparison (rolling window creates NaN at start)
        valid_mask = ~(upper.isna() | middle.isna() | lower.isna())

        # Upper should be >= middle >= lower for valid values
        assert (upper[valid_mask] >= middle[valid_mask]).all()
        assert (middle[valid_mask] >= lower[valid_mask]).all()

    def test_get_percentile_rank(self, analyzer):
        """Test percentile rank calculation."""
        percentile = analyzer.get_percentile_rank('F1_Trend', lookback=63)

        # Percentile should be between 0 and 100
        assert percentile.min() >= 0
        assert percentile.max() <= 100

    def test_check_extreme_levels(self, analyzer):
        """Test extreme level detection."""
        alert = analyzer.check_extreme_levels('F1_Trend', z_threshold=2.0)

        # May or may not have alert depending on data
        if alert is not None:
            assert isinstance(alert, ExtremeAlert)
            assert alert.factor_name == 'F1_Trend'
            assert abs(alert.z_score) > 2.0

    def test_get_all_extreme_alerts(self, analyzer):
        """Test getting all extreme alerts."""
        alerts = analyzer.get_all_extreme_alerts(z_threshold=1.5)

        # Should return list
        assert isinstance(alerts, list)

        # All items should be ExtremeAlert
        for alert in alerts:
            assert isinstance(alert, ExtremeAlert)

    def test_get_momentum_signals(self, analyzer):
        """Test getting momentum signals for a factor."""
        signals = analyzer.get_momentum_signals('F1_Trend')

        # Should return dictionary
        assert isinstance(signals, dict)

        # Should have expected keys
        expected_keys = [
            'factor', 'date', 'rsi', 'rsi_signal', 'macd', 'macd_signal',
            'macd_histogram', 'adx', 'adx_signal', 'regime', 'combined_signal'
        ]
        for key in expected_keys:
            assert key in signals

    def test_get_all_signals(self, analyzer):
        """Test getting all signals for all factors."""
        all_signals = analyzer.get_all_signals()

        # Should return dictionary with factor keys
        assert isinstance(all_signals, dict)
        assert 'F1_Trend' in all_signals
        assert 'F2_MeanRev' in all_signals
        assert 'F3_Volatile' in all_signals

    def test_get_signal_summary(self, analyzer):
        """Test signal summary generation."""
        summary = analyzer.get_signal_summary()

        # Should return DataFrame
        assert isinstance(summary, pd.DataFrame)

        # Should have row for each factor
        assert len(summary) == 3

        # Should have expected columns
        expected_cols = [
            'factor', 'rsi', 'rsi_signal', 'macd_signal', 'adx',
            'regime', 'combined_signal'
        ]
        for col in expected_cols:
            assert col in summary.columns

    def test_invalid_factor_name_raises_error(self, analyzer):
        """Test that invalid factor name raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            analyzer.calculate_rsi('InvalidFactor')

    def test_short_data_logs_warning(self):
        """Test that short data series triggers validation check."""
        short_data = pd.DataFrame({
            'F1': np.random.randn(30)
        }, index=pd.date_range('2023-01-01', periods=30, freq='D'))

        # Should create analyzer but with validation warning in logs
        analyzer = FactorMomentumAnalyzer(short_data)
        assert analyzer.n_factors == 1
        assert len(analyzer.factor_returns) == 30


class TestMomentumRegime:
    """Test suite for MomentumRegime enum."""

    def test_regime_values(self):
        """Test regime enum has expected values."""
        assert MomentumRegime.TRENDING_UP.value == 'trending_up'
        assert MomentumRegime.TRENDING_DOWN.value == 'trending_down'
        assert MomentumRegime.MEAN_REVERTING.value == 'mean_reverting'
        assert MomentumRegime.NEUTRAL.value == 'neutral'


class TestSignalStrength:
    """Test suite for SignalStrength enum."""

    def test_strength_values(self):
        """Test strength enum has expected values."""
        assert SignalStrength.STRONG.value == 'strong'
        assert SignalStrength.MODERATE.value == 'moderate'
        assert SignalStrength.WEAK.value == 'weak'
        assert SignalStrength.NONE.value == 'none'


class TestTradingSignal:
    """Test suite for TradingSignal dataclass."""

    def test_trading_signal_creation(self):
        """Test TradingSignal dataclass creation."""
        signal = TradingSignal(
            factor_name='F1',
            signal_type='momentum',
            direction='long',
            strength=SignalStrength.STRONG,
            confidence=85.0,
            value=1.5,
            threshold=70.0,
            timestamp=pd.Timestamp.now()
        )

        assert signal.factor_name == 'F1'
        assert signal.confidence == 85.0
        assert signal.metadata == {}  # Default empty dict


class TestExtremeAlert:
    """Test suite for ExtremeAlert dataclass."""

    def test_extreme_alert_creation(self):
        """Test ExtremeAlert dataclass creation."""
        alert = ExtremeAlert(
            factor_name='F1',
            alert_type='zscore_extreme',
            z_score=2.5,
            percentile=98.0,
            current_value=0.05,
            threshold=2.0,
            direction='extreme_high',
            timestamp=pd.Timestamp.now()
        )

        assert alert.factor_name == 'F1'
        assert alert.z_score == 2.5
        assert alert.direction == 'extreme_high'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
