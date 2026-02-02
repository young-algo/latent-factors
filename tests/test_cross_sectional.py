"""
Unit tests for cross_sectional module.

Tests cover:
- Composite score calculation
- Decile ranking
- Long/short signal generation
- Factor exposure reporting
"""

import pytest
import numpy as np
import pandas as pd
from src import (
    CrossSectionalAnalyzer,
    SignalDirection,
    StyleFactor,
    StockSignal,
    StyleRotation
)


class TestCrossSectionalAnalyzer:
    """Test suite for CrossSectionalAnalyzer."""

    @pytest.fixture
    def sample_factor_loadings(self):
        """Create sample factor loadings for testing."""
        np.random.seed(42)

        # Create 50 stocks with 5 factors
        tickers = [f'STK{i:03d}' for i in range(1, 51)]

        # Create factors with different characteristics
        loadings = pd.DataFrame({
            'Value': np.random.randn(50),
            'Momentum': np.random.randn(50),
            'Quality': np.random.randn(50),
            'Size': np.random.randn(50),
            'Volatility': np.random.randn(50)
        }, index=tickers)

        return loadings

    @pytest.fixture
    def analyzer(self, sample_factor_loadings):
        """Create CrossSectionalAnalyzer instance."""
        return CrossSectionalAnalyzer(sample_factor_loadings)

    def test_initialization(self, sample_factor_loadings):
        """Test analyzer initialization."""
        analyzer = CrossSectionalAnalyzer(sample_factor_loadings)

        assert analyzer.n_stocks == 50
        assert analyzer.n_factors == 5
        assert list(analyzer.factor_loadings.index) == list(sample_factor_loadings.index)

    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises ValueError."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty"):
            CrossSectionalAnalyzer(empty_df)

    def test_small_universe_validation(self):
        """Test that small universe triggers validation check."""
        small_data = pd.DataFrame(
            np.random.randn(10, 3),
            columns=['F1', 'F2', 'F3'],
            index=[f'STK{i}' for i in range(1, 11)]
        )

        # Should create analyzer but with validation warning in logs
        analyzer = CrossSectionalAnalyzer(small_data)
        assert analyzer.n_stocks == 10
        assert analyzer.n_factors == 3

    def test_calculate_factor_scores_equal_weights(self, analyzer):
        """Test factor score calculation with equal weights."""
        scores = analyzer.calculate_factor_scores()

        # Should return Series
        assert isinstance(scores, pd.Series)

        # Should have same index as loadings
        assert len(scores) == len(analyzer.factor_loadings)

        # Should have same index
        assert list(scores.index) == list(analyzer.factor_loadings.index)

    def test_calculate_factor_scores_custom_weights(self, analyzer):
        """Test factor score calculation with custom weights."""
        weights = {
            'Value': 0.4,
            'Momentum': 0.3,
            'Quality': 0.2,
            'Size': 0.1
        }

        scores = analyzer.calculate_factor_scores(weights=weights)

        # Should return Series
        assert isinstance(scores, pd.Series)
        assert len(scores) == 50

    def test_calculate_factor_scores_invalid_factor(self, analyzer):
        """Test that invalid factor in weights raises error."""
        weights = {'InvalidFactor': 0.5, 'Value': 0.5}

        with pytest.raises(ValueError, match="not found"):
            analyzer.calculate_factor_scores(weights=weights)

    def test_calculate_factor_scores_rank_method(self, analyzer):
        """Test rank-based scoring method."""
        scores = analyzer.calculate_factor_scores(method='rank')

        # Scores should be between 0 and 1 (percentile ranks)
        assert scores.min() >= 0
        assert scores.max() <= 1

    def test_rank_universe(self, analyzer):
        """Test universe ranking."""
        scores = analyzer.calculate_factor_scores()
        rankings = analyzer.rank_universe(scores)

        # Should return DataFrame
        assert isinstance(rankings, pd.DataFrame)

        # Should have expected columns
        expected_cols = ['score', 'rank', 'decile', 'percentile']
        for col in expected_cols:
            assert col in rankings.columns

        # Ranks should be 1 to n_stocks
        assert rankings['rank'].min() == 1
        assert rankings['rank'].max() == 50

        # Deciles should be 1 to 10
        assert rankings['decile'].min() == 1
        assert rankings['decile'].max() == 10

    def test_rank_universe_quintiles(self, analyzer):
        """Test universe ranking with quintiles."""
        scores = analyzer.calculate_factor_scores()
        rankings = analyzer.rank_universe(scores, n_buckets=5)

        # Quintiles should be 1 to 5
        assert rankings['decile'].min() == 1
        assert rankings['decile'].max() == 5

    def test_generate_long_short_signals(self, analyzer):
        """Test long/short signal generation."""
        signals = analyzer.generate_long_short_signals(top_pct=0.1, bottom_pct=0.1)

        # Should return list
        assert isinstance(signals, list)

        # Should have 5 long + 5 short = 10 signals (10% of 50 each)
        assert len(signals) == 10

        # All should be StockSignal
        for sig in signals:
            assert isinstance(sig, StockSignal)

    def test_long_signals_in_top_decile(self, analyzer):
        """Test that long signals are from top decile."""
        signals = analyzer.generate_long_short_signals(top_pct=0.1, bottom_pct=0.1)

        longs = [s for s in signals if s.direction == SignalDirection.LONG]

        for sig in longs:
            assert sig.decile == 1  # Top decile
            assert sig.rank <= 5  # Top 10%

    def test_short_signals_in_bottom_decile(self, analyzer):
        """Test that short signals are from bottom decile."""
        signals = analyzer.generate_long_short_signals(top_pct=0.1, bottom_pct=0.1)

        shorts = [s for s in signals if s.direction == SignalDirection.SHORT]

        for sig in shorts:
            assert sig.decile == 10  # Bottom decile
            assert sig.rank >= 46  # Bottom 10%

    def test_get_factor_exposure_report(self, analyzer):
        """Test factor exposure report generation."""
        report = analyzer.get_factor_exposure_report('STK001')

        # Should return dictionary
        assert isinstance(report, dict)

        # Should have expected keys
        expected_keys = [
            'ticker', 'raw_exposures', 'standardized_exposures',
            'percentile_ranks', 'dominant_factor', 'style_classification'
        ]
        for key in expected_keys:
            assert key in report

        # Ticker should match
        assert report['ticker'] == 'STK001'

    def test_get_factor_exposure_report_invalid_ticker(self, analyzer):
        """Test that invalid ticker raises error."""
        with pytest.raises(ValueError, match="not found"):
            analyzer.get_factor_exposure_report('INVALID')

    def test_detect_style_rotation(self, analyzer):
        """Test style rotation detection."""
        rotation = analyzer.detect_style_rotation()

        # Should return StyleRotation
        assert isinstance(rotation, StyleRotation)

        # Should have expected attributes
        assert hasattr(rotation, 'value_vs_growth')
        assert hasattr(rotation, 'large_vs_small')
        assert hasattr(rotation, 'momentum_strength')
        assert hasattr(rotation, 'quality_bias')

    def test_get_sector_exposure(self, analyzer):
        """Test sector exposure calculation."""
        sector_mapping = {
            f'STK{i:03d}': 'Technology' if i <= 25 else 'Healthcare'
            for i in range(1, 51)
        }

        sector_exposure = analyzer.get_sector_exposure(sector_mapping)

        # Should return DataFrame
        assert isinstance(sector_exposure, pd.DataFrame)

        # Should have sectors as index
        assert 'Technology' in sector_exposure.index
        assert 'Healthcare' in sector_exposure.index

    def test_optimize_factor_weights_max_ic(self, analyzer):
        """Test factor weight optimization with max IC."""
        # Create fake target returns
        target_returns = pd.Series(
            np.random.randn(50),
            index=analyzer.factor_loadings.index
        )

        weights = analyzer.optimize_factor_weights(
            target_returns,
            method='max_ic'
        )

        # Should return dictionary
        assert isinstance(weights, dict)

        # Should have weight for each factor
        assert len(weights) == 5

        # Weights should sum to 1
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_get_portfolio_construction_weights(self, analyzer):
        """Test portfolio construction weights."""
        weights = analyzer.get_portfolio_construction_weights(
            long_pct=0.1,
            short_pct=0.1
        )

        # Should return Series
        assert isinstance(weights, pd.Series)

        # Should have same index as loadings
        assert len(weights) == 50

        # Long weights should be positive
        long_weights = weights[weights > 0]
        assert (long_weights > 0).all()

        # Short weights should be negative
        short_weights = weights[weights < 0]
        assert (short_weights < 0).all()

        # Should have 5 longs and 5 shorts (10% each)
        assert len(long_weights) == 5
        assert len(short_weights) == 5

    def test_get_portfolio_construction_weights_sum_to_zero(self, analyzer):
        """Test that portfolio weights are market neutral."""
        weights = analyzer.get_portfolio_construction_weights(
            long_pct=0.1,
            short_pct=0.1
        )

        # Market neutral: sum of weights should be 0
        assert abs(weights.sum()) < 0.01


class TestSignalDirection:
    """Test suite for SignalDirection enum."""

    def test_direction_values(self):
        """Test direction enum has expected values."""
        assert SignalDirection.LONG.value == 'long'
        assert SignalDirection.SHORT.value == 'short'
        assert SignalDirection.NEUTRAL.value == 'neutral'


class TestStyleFactor:
    """Test suite for StyleFactor enum."""

    def test_style_values(self):
        """Test style factor enum has expected values."""
        assert StyleFactor.VALUE.value == 'value'
        assert StyleFactor.GROWTH.value == 'growth'
        assert StyleFactor.MOMENTUM.value == 'momentum'
        assert StyleFactor.QUALITY.value == 'quality'
        assert StyleFactor.SIZE.value == 'size'
        assert StyleFactor.VOLATILITY.value == 'volatility'


class TestStockSignal:
    """Test suite for StockSignal dataclass."""

    def test_stock_signal_creation(self):
        """Test StockSignal dataclass creation."""
        signal = StockSignal(
            ticker='AAPL',
            direction=SignalDirection.LONG,
            composite_score=1.5,
            decile=1,
            rank=5,
            total_stocks=100,
            factor_breakdown={'Value': 0.5, 'Momentum': 1.0},
            confidence=0.85
        )

        assert signal.ticker == 'AAPL'
        assert signal.direction == SignalDirection.LONG
        assert signal.confidence == 0.85


class TestStyleRotation:
    """Test suite for StyleRotation dataclass."""

    def test_style_rotation_creation(self):
        """Test StyleRotation dataclass creation."""
        rotation = StyleRotation(
            value_vs_growth=0.5,
            large_vs_small=-0.3,
            momentum_strength=0.2,
            quality_bias=0.1,
            timestamp=pd.Timestamp.now()
        )

        assert rotation.value_vs_growth == 0.5
        assert rotation.large_vs_small == -0.3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
