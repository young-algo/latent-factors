"""
Phase 2 Enhancement Tests
=========================

Tests for the institutional quantitative framework upgrades:
1. Conditional Regime Optimization (RS-MVO)
2. Gradient Boosting Meta-Model
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Test configuration
np.random.seed(42)


class TestConditionalRegimeOptimization:
    """Tests for Regime-Switching Mean-Variance Optimization (RS-MVO)."""
    
    @pytest.fixture
    def sample_factor_returns(self):
        """Generate sample factor returns for testing."""
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='B')
        n = len(dates)
        
        # Generate correlated factor returns
        factors = ['momentum', 'value', 'quality', 'growth', 'low_vol']
        returns = pd.DataFrame(
            np.random.randn(n, len(factors)) * 0.01 + 0.0002,
            index=dates,
            columns=factors
        )
        return returns
    
    @pytest.fixture
    def fitted_detector(self, sample_factor_returns):
        """Create and fit a regime detector."""
        try:
            from src.regime_detection import RegimeDetector, MarketRegime
            
            detector = RegimeDetector(sample_factor_returns)
            detector.fit_hmm(n_regimes=3, random_state=42)
            return detector
        except ImportError as e:
            pytest.skip(f"Required module not available: {e}")
    
    def test_regime_masks_cached(self, fitted_detector):
        """Test that regime masks are properly cached after fitting."""
        assert hasattr(fitted_detector, '_regime_masks')
        assert len(fitted_detector._regime_masks) > 0
    
    def test_conditional_optimal_weights_basic(self, fitted_detector):
        """Test basic conditional optimization returns valid weights."""
        from src.regime_detection import MarketRegime
        
        weights = fitted_detector.get_conditional_optimal_weights(
            regime=MarketRegime.LOW_VOL_BULL,
            lookback_window=2520,
            min_observations=20,  # Lower for testing
            fallback_to_global=True
        )
        
        # Verify weights are valid
        assert isinstance(weights, dict)
        assert len(weights) > 0
        
        # Weights should sum to 1 (or close due to floating point)
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01
        
        # All weights should be non-negative
        assert all(w >= 0 for w in weights.values())
    
    def test_conditional_vs_heuristic_weights(self, fitted_detector):
        """Test that conditional optimization differs from heuristic weights."""
        from src.regime_detection import MarketRegime
        
        # Get conditional weights
        conditional_weights = fitted_detector.get_conditional_optimal_weights(
            regime=MarketRegime.LOW_VOL_BULL,
            lookback_window=2520,
            min_observations=20,
            fallback_to_global=True
        )
        
        # Get heuristic weights
        heuristic_weights = fitted_detector._get_heuristic_factor_weights(
            MarketRegime.LOW_VOL_BULL
        )
        
        # They should be different (data-driven vs rule-based)
        assert conditional_weights != heuristic_weights
    
    def test_global_fallback_when_insufficient_regime_data(self, fitted_detector):
        """Test fallback to global optimization when regime data is scarce."""
        from src.regime_detection import MarketRegime
        
        # Request weights with very high minimum observations
        weights = fitted_detector.get_conditional_optimal_weights(
            regime=MarketRegime.CRISIS,  # Usually rare
            lookback_window=100,
            min_observations=1000,  # Impossibly high
            fallback_to_global=True
        )
        
        # Should still return valid weights (fallback)
        assert isinstance(weights, dict)
        assert len(weights) > 0
        assert abs(sum(weights.values()) - 1.0) < 0.01
    
    def test_covariance_shrinkage(self, fitted_detector):
        """Test that covariance matrix shrinkage works for ill-conditioned data."""
        from src.regime_detection import MarketRegime
        
        # This test mainly ensures no exceptions are raised
        weights = fitted_detector.get_conditional_optimal_weights(
            regime=MarketRegime.TRANSITION,
            lookback_window=500,
            min_observations=30,
            fallback_to_global=True
        )
        
        assert isinstance(weights, dict)
        assert len(weights) > 0


class TestSharpeOptimizerNonContiguous:
    """Tests for SharpeOptimizer with non-contiguous (regime-filtered) indices."""
    
    @pytest.fixture
    def non_contiguous_returns(self):
        """Generate factor returns with non-contiguous index."""
        # Create returns with gaps (simulating regime-filtered data)
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='B')
        returns = pd.DataFrame(
            np.random.randn(1000, 3) * 0.01,
            index=dates,
            columns=['f1', 'f2', 'f3']
        )
        
        # Filter to create non-contiguous index (simulate regime filtering)
        mask = np.random.random(1000) > 0.5
        filtered_returns = returns[mask]
        return filtered_returns
    
    def test_optimizer_handles_reset_index(self, non_contiguous_returns):
        """Test that optimizer properly resets non-contiguous indices."""
        try:
            from src.factor_optimization import SharpeOptimizer
        except ImportError:
            pytest.skip("factor_optimization module not available")
        
        optimizer = SharpeOptimizer(
            factor_returns=non_contiguous_returns,
            reset_index=True
        )
        
        # Verify index was reset
        assert optimizer.returns.index[0] == 0
        assert len(optimizer.returns) == len(non_contiguous_returns)
    
    def test_optimization_with_filtered_data(self, non_contiguous_returns):
        """Test optimization on regime-filtered (non-contiguous) returns."""
        try:
            from src.factor_optimization import SharpeOptimizer
        except ImportError:
            pytest.skip("factor_optimization module not available")
        
        optimizer = SharpeOptimizer(
            factor_returns=non_contiguous_returns,
            reset_index=True
        )
        
        # Should complete without errors
        result = optimizer.optimize_blend(
            lookback=min(252, len(non_contiguous_returns)),
            methods=['sharpe', 'equal'],
            technique='differential',
            verbose=False
        )
        
        assert result is not None
        assert isinstance(result.optimal_weights, dict)
        assert abs(sum(result.optimal_weights.values()) - 1.0) < 0.01


class TestFeatureExtractor:
    """Tests for the FeatureExtractor helper class."""
    
    @pytest.fixture
    def sample_momentum_signals(self):
        """Generate sample momentum signal data."""
        return {
            'momentum': {
                'rsi': 65,
                'macd_signal': 'buy',
                'adx': 25,
                'combined_signal': 'buy'
            },
            'value': {
                'rsi': 45,
                'macd_signal': 'neutral',
                'adx': 15,
                'combined_signal': 'neutral'
            }
        }
    
    def test_momentum_feature_extraction(self, sample_momentum_signals):
        """Test extraction of momentum features."""
        try:
            from src.signal_aggregator import FeatureExtractor
        except ImportError:
            pytest.skip("signal_aggregator module not available")
        
        extractor = FeatureExtractor()
        features = extractor.extract_momentum_features(sample_momentum_signals)
        
        # Check features were extracted
        assert 'mom_momentum_rsi' in features
        assert 'mom_momentum_macd' in features
        assert 'mom_momentum_adx' in features
        
        # Check normalization
        assert -1 <= features['mom_momentum_rsi'] <= 1
        assert 0 <= features['mom_momentum_adx'] <= 1
    
    def test_regime_feature_extraction(self):
        """Test extraction of regime features."""
        try:
            from src.signal_aggregator import FeatureExtractor
            from src.regime_detection import RegimeState, MarketRegime
        except ImportError:
            pytest.skip("Required modules not available")
        
        extractor = FeatureExtractor()
        
        regime_state = RegimeState(
            regime=MarketRegime.LOW_VOL_BULL,
            probability=0.8,
            volatility=0.15,
            trend=0.001,
            description="Test"
        )
        
        features = extractor.extract_regime_features(
            regime_state,
            regime_probs={
                MarketRegime.LOW_VOL_BULL: 0.8,
                MarketRegime.HIGH_VOL_BEAR: 0.2
            }
        )
        
        assert 'regime_prob' in features
        assert 'regime_prob_low_volatility_bull' in features
        assert features['regime_prob'] == 0.8


class TestMetaModelAggregator:
    """Tests for the MetaModelAggregator (Gradient Boosting Meta-Model)."""
    
    @pytest.fixture
    def mock_frs(self):
        """Create a mock factor research system."""
        class MockFRS:
            pass
        return MockFRS()
    
    @pytest.fixture
    def sample_market_returns(self):
        """Generate sample market returns for label generation."""
        dates = pd.date_range(start='2020-01-01', periods=500, freq='B')
        returns = pd.Series(
            np.random.randn(500) * 0.01 + 0.0002,
            index=dates
        )
        return returns
    
    def test_metamodel_initialization(self, mock_frs):
        """Test MetaModelAggregator initialization."""
        try:
            from src.signal_aggregator import MetaModelAggregator
        except ImportError:
            pytest.skip("signal_aggregator module not available")
        
        aggregator = MetaModelAggregator(
            mock_frs,
            min_training_samples=100,
            prediction_horizon=5
        )
        
        assert aggregator.min_training_samples == 100
        assert aggregator.prediction_horizon == 5
        assert not aggregator._model_trained
    
    def test_set_market_returns(self, mock_frs, sample_market_returns):
        """Test setting market returns for label generation."""
        try:
            from src.signal_aggregator import MetaModelAggregator
        except ImportError:
            pytest.skip("signal_aggregator module not available")
        
        aggregator = MetaModelAggregator(mock_frs)
        aggregator.set_market_returns(sample_market_returns)
        
        assert aggregator.market_returns is not None
        assert len(aggregator.market_returns) == 500
    
    def test_label_generation(self, mock_frs, sample_market_returns):
        """Test binary label generation from forward returns."""
        try:
            from src.signal_aggregator import MetaModelAggregator
        except ImportError:
            pytest.skip("signal_aggregator module not available")
        
        aggregator = MetaModelAggregator(
            mock_frs,
            prediction_horizon=5
        )
        aggregator.set_market_returns(sample_market_returns)
        
        # Generate labels for some dates
        dates = sample_market_returns.index[:100]
        labels = aggregator._generate_labels(dates.tolist())
        
        assert len(labels) == 100
        assert all(l in [0, 1] for l in labels)
    
    def test_fallback_to_voting_when_untrained(self, mock_frs):
        """Test that aggregator falls back to voting when model is untrained."""
        try:
            from src.signal_aggregator import MetaModelAggregator
        except ImportError:
            pytest.skip("signal_aggregator module not available")
        
        aggregator = MetaModelAggregator(
            mock_frs,
            use_voting_fallback=True
        )
        
        # Should not raise error even though model is untrained
        result = aggregator.generate_meta_consensus()
        
        assert 'consensus_signal' in result
        assert 'probability_up' in result
        # Default probability when untrained
        assert result['probability_up'] == 0.5


class TestIntegration:
    """Integration tests combining multiple Phase 2 components."""
    
    def test_full_pipeline_with_regime_and_metamodel(self):
        """Test the full pipeline: regime detection -> conditional optimization -> meta-model."""
        try:
            from src.regime_detection import RegimeDetector, MarketRegime
            from src.factor_optimization import SharpeOptimizer
            from src.signal_aggregator import MetaModelAggregator
        except ImportError as e:
            pytest.skip(f"Required module not available: {e}")
        
        # Generate synthetic data
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='B')
        factor_returns = pd.DataFrame(
            np.random.randn(1000, 5) * 0.01 + 0.0002,
            index=dates,
            columns=['momentum', 'value', 'quality', 'growth', 'low_vol']
        )
        market_returns = pd.Series(
            np.random.randn(1000) * 0.01 + 0.0002,
            index=dates
        )
        
        # Step 1: Fit regime detector
        detector = RegimeDetector(factor_returns)
        detector.fit_hmm(n_regimes=3, random_state=42)
        
        # Step 2: Get conditional optimal weights
        current_regime = detector.detect_current_regime()
        weights = detector.get_conditional_optimal_weights(
            current_regime.regime,
            lookback_window=500,
            min_observations=50
        )
        
        # Verify weights are valid
        assert abs(sum(weights.values()) - 1.0) < 0.01
        
        # Step 3: Initialize meta-model aggregator
        # Note: We can't fully test training without proper signal sources
        class MockFRS:
            pass
        
        aggregator = MetaModelAggregator(
            MockFRS(),
            min_training_samples=50
        )
        aggregator.set_market_returns(market_returns)
        
        # Verify pipeline components work together
        assert detector.regime_history is not None
        assert isinstance(weights, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
