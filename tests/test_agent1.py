"""
Tests for Agent 1 — Macro & Market Intelligence System
========================================================
Comprehensive test suite covering all ML, data, and LLM components.
Run: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime


# ═══════════════════════════════════════════════════════
#  FEATURE ENGINE TESTS
# ═══════════════════════════════════════════════════════
class TestFeatureEngine:
    """Test feature engineering functions."""

    @pytest.fixture
    def benchmark_prices(self):
        np.random.seed(42)
        dates = pd.date_range(end=datetime.utcnow(), periods=300, freq="B")
        prices = 400 * np.exp(np.cumsum(np.random.normal(0.0003, 0.012, len(dates))))
        return pd.Series(prices, index=dates, name="SPY")

    @pytest.fixture
    def vix_prices(self):
        np.random.seed(42)
        dates = pd.date_range(end=datetime.utcnow(), periods=300, freq="B")
        vix = 18 + np.random.normal(0, 3, len(dates))
        vix = np.maximum(10, vix)
        return pd.Series(vix, index=dates, name="VIX")

    @pytest.fixture
    def cross_asset_prices(self, benchmark_prices):
        np.random.seed(42)
        dates = benchmark_prices.index
        assets = {"SPY": benchmark_prices}
        for name in ["QQQ", "TLT", "GLD"]:
            noise = np.random.normal(0.0002, 0.015, len(dates))
            assets[name] = pd.Series(100 * np.exp(np.cumsum(noise)), index=dates)
        return pd.DataFrame(assets)

    def test_compute_returns(self, benchmark_prices):
        from ml.feature_engine import FeatureEngine
        returns = FeatureEngine.compute_returns(benchmark_prices)
        assert isinstance(returns, pd.Series)
        assert len(returns) > 0
        assert len(returns) == len(benchmark_prices) - 1

    def test_compute_realized_volatility(self, benchmark_prices):
        from ml.feature_engine import FeatureEngine
        vol = FeatureEngine.compute_realized_volatility(benchmark_prices)
        assert isinstance(vol, pd.DataFrame)
        assert "vol_21d" in vol.columns
        assert vol["vol_21d"].dropna().iloc[-1] > 0

    def test_compute_rsi(self, benchmark_prices):
        from ml.feature_engine import FeatureEngine
        rsi = FeatureEngine.compute_rsi(benchmark_prices)
        assert isinstance(rsi, pd.Series)
        rsi_clean = rsi.dropna()
        if len(rsi_clean) > 0:
            assert rsi_clean.min() >= 0
            assert rsi_clean.max() <= 100

    def test_compute_macd(self, benchmark_prices):
        from ml.feature_engine import FeatureEngine
        macd = FeatureEngine.compute_macd(benchmark_prices)
        assert isinstance(macd, pd.DataFrame)
        assert "macd_line" in macd.columns
        assert "signal_line" in macd.columns
        assert "histogram" in macd.columns

    def test_compute_correlation_matrix(self, cross_asset_prices):
        from ml.feature_engine import FeatureEngine
        corr = FeatureEngine.compute_correlation_matrix(cross_asset_prices)
        assert isinstance(corr, pd.DataFrame)
        assert corr.shape[0] == corr.shape[1]
        # Diagonal should be 1.0
        for i in range(corr.shape[0]):
            assert abs(corr.iloc[i, i] - 1.0) < 1e-10

    def test_compute_drawdown(self, benchmark_prices):
        from ml.feature_engine import FeatureEngine
        dd = FeatureEngine.compute_drawdown(benchmark_prices)
        assert isinstance(dd, pd.DataFrame)
        assert "drawdown" in dd.columns
        assert dd["drawdown"].min() <= 0  # drawdowns are negative

    def test_build_features(self, benchmark_prices, cross_asset_prices, vix_prices):
        from ml.feature_engine import FeatureEngine
        features = FeatureEngine.build_features(
            benchmark_prices=benchmark_prices,
            all_close_prices=cross_asset_prices,
            vix_prices=vix_prices,
        )
        assert isinstance(features, dict)
        assert "returns" in features
        assert "volatility" in features
        assert "momentum" in features
        assert "correlations" in features
        assert "drawdown" in features

    def test_build_regime_features(self, benchmark_prices, vix_prices):
        from ml.feature_engine import FeatureEngine
        rf = FeatureEngine.build_regime_features(
            benchmark_prices=benchmark_prices,
            vix_prices=vix_prices,
        )
        assert isinstance(rf, pd.DataFrame)
        # May be empty after dropna if columns don't align; check columns exist
        assert len(rf.columns) > 5


# ═══════════════════════════════════════════════════════
#  REGIME DETECTOR TESTS
# ═══════════════════════════════════════════════════════
class TestRegimeDetector:
    """Test ensemble regime detection."""

    @pytest.fixture
    def regime_features(self):
        from ml.feature_engine import FeatureEngine
        np.random.seed(42)
        dates = pd.date_range(end=datetime.utcnow(), periods=300, freq="B")
        prices = 400 * np.exp(np.cumsum(np.random.normal(0.0003, 0.012, len(dates))))
        benchmark = pd.Series(prices, index=dates)
        vix = pd.Series(18 + np.random.normal(0, 3, len(dates)), index=dates)
        return FeatureEngine.build_regime_features(benchmark, vix)

    def test_detect_regime(self, regime_features):
        from ml.regime_detector import EnsembleRegimeDetector
        detector = EnsembleRegimeDetector()
        result = detector.detect_regime(regime_features)

        assert isinstance(result, dict)
        assert "primary_regime" in result
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1
        assert result["primary_regime"] in [
            "bull_low_vol", "bull_high_vol", "bear_low_vol", "bear_high_vol", "unknown"
        ]

    def test_hmm_detector(self, regime_features):
        from ml.regime_detector import HMMRegimeDetector
        detector = HMMRegimeDetector()
        result = detector.fit_predict(regime_features)

        assert isinstance(result, dict)
        assert "current_regime" in result
        assert "confidence" in result
        assert "regime_duration_days" in result

    def test_rf_classifier(self, regime_features):
        from ml.regime_detector import RFRegimeClassifier
        classifier = RFRegimeClassifier()
        result = classifier.fit_predict(regime_features)

        assert isinstance(result, dict)
        assert "current_regime" in result
        assert "confidence" in result


# ═══════════════════════════════════════════════════════
#  VOLATILITY CLASSIFIER TESTS
# ═══════════════════════════════════════════════════════
class TestVolatilityClassifier:
    """Test volatility state classification."""

    @pytest.fixture
    def features_and_vix(self):
        from ml.feature_engine import FeatureEngine
        np.random.seed(42)
        dates = pd.date_range(end=datetime.utcnow(), periods=300, freq="B")
        prices = 400 * np.exp(np.cumsum(np.random.normal(0.0003, 0.012, len(dates))))
        benchmark = pd.Series(prices, index=dates)
        vix = pd.Series(18 + np.random.normal(0, 3, len(dates)), index=dates)
        vix = np.maximum(10, vix)
        features = FeatureEngine.build_features(benchmark_prices=benchmark, vix_prices=vix)
        vix_df = pd.DataFrame({"close": vix})
        return features, vix_df

    def test_classify(self, features_and_vix):
        from ml.volatility_classifier import VolatilityClassifier
        features, vix_df = features_and_vix
        result = VolatilityClassifier.classify(features, vix_df)

        assert isinstance(result, dict)
        assert "current_state" in result
        assert result["current_state"] in [
            "extremely_low", "low", "normal", "elevated", "extreme"
        ]
        assert "vix_level" in result
        assert "vol_trend" in result
        assert 0 <= result["confidence"] <= 1

    def test_classify_vix_extreme(self):
        """Test classification with extreme VIX values."""
        from ml.volatility_classifier import VolatilityClassifier

        vix_close = pd.Series([40.0] * 300)
        result = VolatilityClassifier._classify_vix(vix_close)
        assert result["state"] == "extreme"

        vix_close = pd.Series([11.0] * 300)
        result = VolatilityClassifier._classify_vix(vix_close)
        assert result["state"] == "extremely_low"


# ═══════════════════════════════════════════════════════
#  MACRO ANALYZER TESTS
# ═══════════════════════════════════════════════════════
class TestMacroAnalyzer:
    """Test macroeconomic environment analysis."""

    @pytest.fixture
    def mock_macro_snapshot(self):
        return {
            "monetary_policy": {"state": "tightening", "confidence": 0.8},
            "inflation": {"state": "above_target", "confidence": 0.85,
                          "details": {"cpi_yoy_pct": 3.2, "inflation_momentum": "decelerating"}},
            "growth": {"state": "moderate_growth", "confidence": 0.75},
            "liquidity": {"state": "neutral", "confidence": 0.7,
                          "details": {"credit_stress": False}},
            "labor": {"state": "healthy", "confidence": 0.8},
            "derived_indicators": {
                "yield_curve": {
                    "inverted": True,
                    "spreads": {"10y_2y": -0.25, "10y_3m": -0.15},
                    "signal": "inverted_but_improving",
                }
            },
            "current_values": {
                "fed_funds_rate": {"value": 5.25},
                "treasury_10y": {"value": 4.35},
                "unemployment": {"value": 3.9},
            },
        }

    def test_analyze(self, mock_macro_snapshot):
        from ml.macro_analyzer import MacroAnalyzer
        result = MacroAnalyzer.analyze(mock_macro_snapshot)

        assert isinstance(result, dict)
        assert "macro_regime" in result
        assert result["macro_regime"] in [
            "risk_on_expansion", "stable_growth", "late_cycle",
            "stagflation", "recession", "recovery"
        ]
        assert -1 <= result["composite_score"] <= 1
        assert 0 <= result["confidence"] <= 1

    def test_risk_factors(self, mock_macro_snapshot):
        from ml.macro_analyzer import MacroAnalyzer
        result = MacroAnalyzer.analyze(mock_macro_snapshot)
        assert isinstance(result["risk_factors"], list)
        # Tightening + inverted yield curve should produce risk factors
        assert len(result["risk_factors"]) > 0


# ═══════════════════════════════════════════════════════
#  RISK DETECTOR TESTS
# ═══════════════════════════════════════════════════════
class TestRiskDetector:
    """Test systemic risk detection."""

    @pytest.fixture
    def risk_inputs(self):
        features = {
            "volatility": {"vol_zscore": pd.Series([0.5, 0.3, 0.1, -0.2, 0.0])},
            "correlations": {"median_correlation": 0.35, "matrix": pd.DataFrame()},
        }
        macro_analysis = {
            "macro_regime": "stable_growth",
            "yield_curve": {"inverted": False, "signal": "normal"},
            "key_indicators": {},
        }
        volatility_state = {
            "current_state": "normal",
            "vix_level": 18.0,
            "vol_trend": "stable",
        }
        return features, macro_analysis, volatility_state

    def test_detect(self, risk_inputs):
        from ml.risk_detector import SystemicRiskDetector
        features, macro, vol = risk_inputs
        result = SystemicRiskDetector.detect(features, macro, vol)

        assert isinstance(result, dict)
        assert "overall_risk_level" in result
        assert 0 <= result["overall_risk_level"] <= 1
        assert result["risk_category"] in ["low", "moderate", "elevated", "high", "critical"]

    def test_high_risk_scenario(self):
        """Test that extreme inputs produce high risk."""
        from ml.risk_detector import SystemicRiskDetector
        features = {
            "volatility": {"vol_zscore": pd.Series([3.0, 2.5, 2.0, 3.5, 4.0])},
            "correlations": {"median_correlation": 0.85, "matrix": pd.DataFrame()},
        }
        macro = {
            "macro_regime": "recession",
            "yield_curve": {"inverted": True, "spreads": {"10y_2y": -0.5}, "signal": "deepening_inversion"},
            "key_indicators": {"credit_spread_hy": 8.0},
        }
        vol_state = {"current_state": "extreme", "vix_level": 45.0, "vol_trend": "sharply_increasing"}

        result = SystemicRiskDetector.detect(features, macro, vol_state)
        assert result["overall_risk_level"] > 0.4
        assert result["risk_category"] in ["elevated", "high", "critical"]


# ═══════════════════════════════════════════════════════
#  SCHEMA VALIDATION TESTS
# ═══════════════════════════════════════════════════════
class TestSchema:
    """Test Pydantic output schema validation."""

    def test_valid_output(self):
        from schemas.agent1_output import validate_output
        output = {
            "timestamp": "2026-02-24T10:00:00Z",
            "data_freshness": "2026-02-24T09:55:00Z",
            "market_regime": {
                "primary_regime": "bull_low_vol", "confidence": 0.85,
                "hmm_regime": "bull_low_vol", "rf_regime": "bull_low_vol",
                "models_agree": True, "regime_duration_days": 10,
                "transition_probability": 0.2, "description": "test",
            },
            "volatility_state": {
                "current_state": "normal", "vix_level": 18.0,
                "realized_vol_percentile": 50.0, "vol_trend": "stable",
                "vol_of_vol": "normal", "term_structure": "unknown",
            },
            "macro_environment": {
                "macro_regime": "stable_growth", "monetary_policy": "tightening",
                "inflation_state": "above_target", "growth_state": "moderate_growth",
                "liquidity": "neutral", "composite_score": 0.1,
                "key_indicators": {}, "yield_curve": {},
            },
            "systemic_risk": {
                "overall_risk_level": 0.2, "risk_category": "low",
                "risk_signals": {}, "risk_assessment": "Low risk",
                "recommended_caution": False,
            },
            "cross_asset_analysis": {
                "correlation_state": "normal", "median_correlation": 0.3,
                "risk_appetite_index": 0.7, "key_correlations": {},
            },
            "llm_reasoning": {
                "market_narrative": "Markets stable.",
                "key_risks": [], "opportunities": [],
                "asset_class_outlook": {}, "sector_implications": {},
                "risk_budget_suggestion": {}, "confidence_level": 0.8,
                "uncertainty_factors": [],
            },
            "agent_metadata": {
                "agent_id": "agent1", "version": "1.0.0",
                "execution_time_ms": 1000, "llm_calls": 3,
                "llm_total_latency_ms": 500, "models_used": [],
                "data_sources": [],
            },
        }
        is_valid, error = validate_output(output)
        assert is_valid
        assert error is None

    def test_invalid_confidence(self):
        """Confidence must be 0-1."""
        from schemas.agent1_output import MarketRegime
        with pytest.raises(Exception):
            MarketRegime(primary_regime="test", confidence=1.5)


# ═══════════════════════════════════════════════════════
#  LLM CLIENT TESTS
# ═══════════════════════════════════════════════════════
class TestLLMClient:
    """Test LLM client abstraction."""

    def test_factory_creates_gemini(self):
        from llm.gemini_client import LLMFactory, GeminiClient
        client = LLMFactory.create("gemini")
        assert isinstance(client, GeminiClient)

    def test_factory_unknown_provider(self):
        from llm.gemini_client import LLMFactory
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            LLMFactory.create("nonexistent")

    def test_base_client_interface(self):
        from llm.gemini_client import BaseLLMClient
        # Ensure abstract methods exist
        assert hasattr(BaseLLMClient, "generate")
        assert hasattr(BaseLLMClient, "is_available")


# ═══════════════════════════════════════════════════════
#  FULL PIPELINE INTEGRATION TEST
# ═══════════════════════════════════════════════════════
class TestAgent1Pipeline:
    """Integration test for the full Agent 1 mock pipeline."""

    def test_mock_pipeline_runs(self):
        from agents.agent1_macro import Agent1MacroIntelligence
        agent = Agent1MacroIntelligence()
        output = agent.run(mock=True)

        assert isinstance(output, dict)
        assert "timestamp" in output
        assert "market_regime" in output
        assert "volatility_state" in output
        assert "macro_environment" in output
        assert "systemic_risk" in output
        assert "cross_asset_analysis" in output
        assert "llm_reasoning" in output
        assert "agent_metadata" in output

    def test_mock_output_regime(self):
        from agents.agent1_macro import Agent1MacroIntelligence
        agent = Agent1MacroIntelligence()
        output = agent.run(mock=True)

        regime = output["market_regime"]
        assert regime["primary_regime"] != "unknown"
        assert regime["confidence"] > 0.5
        assert regime["models_agree"] is True or regime["models_agree"] is False

    def test_mock_output_schema_valid(self):
        from agents.agent1_macro import Agent1MacroIntelligence
        from schemas.agent1_output import validate_output

        agent = Agent1MacroIntelligence()
        output = agent.run(mock=True)
        is_valid, error = validate_output(output)
        assert is_valid, f"Schema validation failed: {error}"
