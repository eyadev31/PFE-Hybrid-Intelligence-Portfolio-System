"""
Agent 3 -- Strategic Allocation & Optimization Agent Tests
=============================================================
Comprehensive tests covering:
  - Asset universe (returns, covariance, bounds)
  - Portfolio optimizer (MV, RP, CVaR, blending)
  - Monte Carlo simulator (10K scenarios)
  - Schema validation
  - Full pipeline integration
"""

import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ================================================================
#  FIXTURES
# ================================================================

@pytest.fixture
def mock_agent1():
    return {
        "market_regime": {
            "primary_regime": "bear_high_vol",
            "confidence": 0.82,
            "models_agree": True,
        },
        "volatility_state": {
            "current_state": "elevated",
            "vix_level": 28.5,
        },
        "systemic_risk": {
            "overall_risk_level": 0.55,
            "risk_category": "elevated",
        },
        "macro_environment": {
            "macro_regime": "contraction",
            "monetary_policy": "tightening",
            "inflation_state": "above_target",
            "growth_state": "slowing",
            "key_indicators": {"fed_funds_rate": 5.25},
            "yield_curve": {"inverted": True},
        },
        "cross_asset_analysis": {
            "median_correlation": 0.55,
            "key_correlations": {"SPY_GLD": 0.15},
        },
    }


@pytest.fixture
def mock_agent2_conservative():
    return {
        "risk_classification": {
            "risk_score": 0.20,
            "behavioral_type": "conservative_stable",
            "max_acceptable_drawdown": 0.08,
            "liquidity_preference": "high",
            "time_horizon": "short",
        },
        "behavioral_profile": {
            "consistency_score": 0.85,
            "emotional_stability": "stable",
        },
    }


@pytest.fixture
def mock_agent2_aggressive():
    return {
        "risk_classification": {
            "risk_score": 0.85,
            "behavioral_type": "aggressive_speculator",
            "max_acceptable_drawdown": 0.35,
            "liquidity_preference": "low",
            "time_horizon": "long",
        },
        "behavioral_profile": {
            "consistency_score": 0.70,
            "emotional_stability": "moderate",
        },
    }


@pytest.fixture
def mock_agent2_moderate():
    return {
        "risk_classification": {
            "risk_score": 0.55,
            "behavioral_type": "moderate_balanced",
            "max_acceptable_drawdown": 0.15,
            "liquidity_preference": "medium",
            "time_horizon": "medium",
        },
        "behavioral_profile": {
            "consistency_score": 0.78,
            "emotional_stability": "stable",
        },
    }


# ================================================================
#  ASSET UNIVERSE TESTS
# ================================================================

class TestAssetUniverse:
    def test_tickers(self):
        from ml.asset_universe import AssetUniverseManager
        mgr = AssetUniverseManager()
        assert mgr.tickers == ["SPY", "BND", "GLD", "BTC", "CASH"]
        assert mgr.n_assets == 5

    def test_expected_returns_default(self):
        from ml.asset_universe import AssetUniverseManager
        mgr = AssetUniverseManager()
        returns = mgr.get_expected_returns()
        assert len(returns) == 5
        assert all(r >= 0 for r in returns)

    def test_bear_regime_adjusts_returns(self, mock_agent1):
        from ml.asset_universe import AssetUniverseManager
        mgr = AssetUniverseManager()
        default = mgr.get_expected_returns()
        adjusted = mgr.get_expected_returns(agent1_output=mock_agent1)
        # In bear: SPY should decrease, GLD should increase
        assert adjusted[0] < default[0]  # SPY down
        assert adjusted[2] > default[2]  # GLD up

    def test_covariance_positive_definite(self, mock_agent1):
        from ml.asset_universe import AssetUniverseManager
        mgr = AssetUniverseManager()
        cov = mgr.get_covariance_matrix(agent1_output=mock_agent1)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert all(ev > -1e-6 for ev in eigenvalues)

    def test_bounds_conservative(self):
        from ml.asset_universe import AssetUniverseManager
        mgr = AssetUniverseManager()
        bounds = mgr.get_weight_bounds(risk_score=0.15)
        # Conservative: BTC max should be very low
        btc_idx = mgr.tickers.index("BTC")
        assert bounds[btc_idx][1] <= 0.05

    def test_bounds_aggressive(self):
        from ml.asset_universe import AssetUniverseManager
        mgr = AssetUniverseManager()
        bounds = mgr.get_weight_bounds(risk_score=0.90)
        btc_idx = mgr.tickers.index("BTC")
        assert bounds[btc_idx][1] >= 0.15


# ================================================================
#  PORTFOLIO OPTIMIZER TESTS
# ================================================================

class TestPortfolioOptimizer:
    def _get_optimizer(self):
        from ml.asset_universe import AssetUniverseManager
        mgr = AssetUniverseManager()
        mu = mgr.get_expected_returns()
        cov = mgr.get_covariance_matrix()
        return __import__("ml.portfolio_optimizer", fromlist=["PortfolioOptimizer"]).PortfolioOptimizer(
            expected_returns=mu, cov_matrix=cov, tickers=mgr.tickers
        )

    def test_mv_weights_sum_to_one(self):
        opt = self._get_optimizer()
        result = opt.mean_variance(risk_aversion=1.0)
        total = sum(result["weights"].values())
        assert abs(total - 1.0) < 0.01

    def test_mv_high_aversion_more_conservative(self):
        opt = self._get_optimizer()
        low_aversion = opt.mean_variance(risk_aversion=0.5)
        high_aversion = opt.mean_variance(risk_aversion=5.0)
        assert high_aversion["expected_volatility"] <= low_aversion["expected_volatility"]

    def test_risk_parity_weights_sum_to_one(self):
        opt = self._get_optimizer()
        result = opt.risk_parity()
        total = sum(result["weights"].values())
        assert abs(total - 1.0) < 0.01

    def test_risk_parity_equal_contributions(self):
        opt = self._get_optimizer()
        result = opt.risk_parity()
        rc = result.get("risk_contributions", {})
        if rc:
            values = list(rc.values())
            # Risk contributions should be roughly equal (within tolerance)
            assert max(values) - min(values) < 0.3

    def test_cvar_weights_sum_to_one(self):
        opt = self._get_optimizer()
        result = opt.cvar_constrained(max_cvar=0.15)
        total = sum(result["weights"].values())
        assert abs(total - 1.0) < 0.01

    def test_cvar_result_has_cvar(self):
        opt = self._get_optimizer()
        result = opt.cvar_constrained(max_cvar=0.15)
        assert "cvar_95" in result

    def test_profile_selection(self):
        opt = self._get_optimizer()
        result = opt.optimize_for_profile(risk_score=0.5, max_drawdown=0.15)
        assert "strategy_type" in result
        assert "alternatives" in result
        assert result["strategy_type"] in (
            "defensive", "defensive_growth", "balanced",
            "growth", "aggressive_growth", "max_growth"
        )

    def test_diversification_ratio(self):
        opt = self._get_optimizer()
        w = np.ones(5) / 5
        dr = opt.compute_diversification_ratio(w)
        assert dr >= 1.0  # Diversification should help


# ================================================================
#  MONTE CARLO TESTS
# ================================================================

class TestMonteCarlo:
    def test_simulation_runs(self):
        from ml.asset_universe import AssetUniverseManager
        from ml.monte_carlo import MonteCarloSimulator
        mgr = AssetUniverseManager()
        mu = mgr.get_expected_returns()
        cov = mgr.get_covariance_matrix()
        mc = MonteCarloSimulator(mu, cov, mgr.tickers)
        weights = np.array([0.35, 0.25, 0.15, 0.10, 0.15])
        result = mc.simulate(weights, n_simulations=1000, horizon_days=252)
        assert result["num_simulations"] == 1000
        assert "percentile_returns" in result
        assert "probability_of_loss" in result

    def test_percentile_ordering(self):
        from ml.asset_universe import AssetUniverseManager
        from ml.monte_carlo import MonteCarloSimulator
        mgr = AssetUniverseManager()
        mu = mgr.get_expected_returns()
        cov = mgr.get_covariance_matrix()
        mc = MonteCarloSimulator(mu, cov, mgr.tickers)
        weights = np.array([0.35, 0.25, 0.15, 0.10, 0.15])
        result = mc.simulate(weights, n_simulations=1000)
        p = result["percentile_returns"]
        assert p["p5"] <= p["p25"] <= p["p50"] <= p["p75"] <= p["p95"]

    def test_loss_probability_bounded(self):
        from ml.asset_universe import AssetUniverseManager
        from ml.monte_carlo import MonteCarloSimulator
        mgr = AssetUniverseManager()
        mu = mgr.get_expected_returns()
        cov = mgr.get_covariance_matrix()
        mc = MonteCarloSimulator(mu, cov, mgr.tickers)
        weights = np.array([0.35, 0.25, 0.15, 0.10, 0.15])
        result = mc.simulate(weights, n_simulations=1000)
        assert 0.0 <= result["probability_of_loss"] <= 1.0


# ================================================================
#  SCHEMA TESTS
# ================================================================

class TestSchemas:
    def test_agent3_output_schema(self):
        from schemas.agent3_output import Agent3Output
        data = {
            "timestamp": "2026-02-24T22:00:00Z",
            "allocation": [{
                "ticker": "SPY", "asset_class": "equity",
                "weight": 0.35, "expected_return": 0.10,
            }],
            "portfolio_metrics": {
                "expected_annual_return": 0.08,
                "expected_annual_volatility": 0.12,
                "sharpe_ratio": 0.67,
                "max_drawdown_estimate": 0.15,
            },
            "monte_carlo": {},
            "optimization": {
                "method_used": "blended",
                "strategy_type": "balanced",
            },
            "evolution_metrics": {
                "requires_rebalance": True,
                "max_drift_detected": 0.0,
                "portfolio_turnover": 0.0,
                "estimated_transaction_costs": 0.0,
            },
            "llm_explanation": {},
        }
        output = Agent3Output(**data)
        assert output.allocation[0].ticker == "SPY"

    def test_weight_validation(self):
        from schemas.agent3_output import validate_agent3_output
        valid, _ = validate_agent3_output({
            "timestamp": "2026-02-24T22:00:00Z",
            "allocation": [
                {"ticker": "SPY", "asset_class": "equity", "weight": 0.50},
                {"ticker": "BND", "asset_class": "bond", "weight": 0.50},
            ],
            "portfolio_metrics": {
                "expected_annual_return": 0.07,
                "expected_annual_volatility": 0.10,
                "sharpe_ratio": 0.60,
                "max_drawdown_estimate": 0.12,
            },
            "monte_carlo": {},
            "optimization": {
                "method_used": "mean_variance",
                "strategy_type": "balanced",
            },
            "evolution_metrics": {
                "requires_rebalance": True,
                "max_drift_detected": 0.0,
                "portfolio_turnover": 0.0,
                "estimated_transaction_costs": 0.0,
            },
            "llm_explanation": {},
        })
        assert valid


# ================================================================
#  FULL PIPELINE INTEGRATION
# ================================================================

class TestAgent3Pipeline:
    def test_mock_pipeline_runs(self):
        from agents.agent3_strategist import Agent3PortfolioStrategist
        agent = Agent3PortfolioStrategist()
        result = agent.run_mock()
        assert "allocation" in result
        assert "portfolio_metrics" in result
        assert "monte_carlo" in result

    def test_weights_sum_to_one(self):
        from agents.agent3_strategist import Agent3PortfolioStrategist
        agent = Agent3PortfolioStrategist()
        result = agent.run_mock()
        total = sum(a["weight"] for a in result["allocation"])
        assert abs(total - 1.0) < 0.02

    def test_conservative_allocation(self, mock_agent1, mock_agent2_conservative):
        from agents.agent3_strategist import Agent3PortfolioStrategist
        agent = Agent3PortfolioStrategist()
        result = agent.run_mock(
            agent1_output=mock_agent1,
            agent2_output=mock_agent2_conservative,
        )
        # Conservative should have more bonds/cash
        alloc = {a["ticker"]: a["weight"] for a in result["allocation"]}
        assert alloc.get("BND", 0) + alloc.get("CASH", 0) >= 0.30

    def test_aggressive_allocation(self, mock_agent1, mock_agent2_aggressive):
        from agents.agent3_strategist import Agent3PortfolioStrategist
        agent = Agent3PortfolioStrategist()
        result = agent.run_mock(
            agent1_output=mock_agent1,
            agent2_output=mock_agent2_aggressive,
        )
        # Aggressive should have more equity/crypto
        alloc = {a["ticker"]: a["weight"] for a in result["allocation"]}
        assert alloc.get("SPY", 0) + alloc.get("BTC", 0) >= 0.30

    def test_metrics_present(self):
        from agents.agent3_strategist import Agent3PortfolioStrategist
        agent = Agent3PortfolioStrategist()
        result = agent.run_mock()
        metrics = result["portfolio_metrics"]
        assert "sharpe_ratio" in metrics
        assert "max_drawdown_estimate" in metrics
        assert "cvar_95" in metrics

    def test_monte_carlo_present(self):
        from agents.agent3_strategist import Agent3PortfolioStrategist
        agent = Agent3PortfolioStrategist()
        result = agent.run_mock()
        mc = result["monte_carlo"]
        assert mc.get("num_simulations", 0) >= 1000
        assert "probability_of_loss" in mc

    def test_explanation_present(self):
        from agents.agent3_strategist import Agent3PortfolioStrategist
        agent = Agent3PortfolioStrategist()
        result = agent.run_mock()
        expl = result["llm_explanation"]
        assert "allocation_rationale" in expl
