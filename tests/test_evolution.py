import pytest
import numpy as np
from agents.agent3_strategist import Agent3PortfolioStrategist

class TestEvolutionEngine:
    @pytest.fixture
    def agent3(self):
        return Agent3PortfolioStrategist()

    @pytest.fixture
    def target_allocation(self):
        return [
            {"ticker": "SPY", "weight": 0.50},
            {"ticker": "BND", "weight": 0.30},
            {"ticker": "GLD", "weight": 0.10},
            {"ticker": "BTC", "weight": 0.05},
            {"ticker": "CASH", "weight": 0.05},
        ]

    @pytest.fixture
    def target_metrics(self):
        return {
            "expected_annual_return": 0.0800,
            "expected_annual_volatility": 0.1200,
            "sharpe_ratio": 0.50,
        }

    @pytest.fixture
    def cov_matrix(self):
        return np.eye(5) * 0.01

    @pytest.fixture
    def expected_returns(self):
        return np.array([0.10, 0.04, 0.06, 0.50, 0.02])

    def test_no_current_portfolio(self, agent3, target_allocation, target_metrics, cov_matrix, expected_returns):
        """Test initial deployment from scratch (no starting portfolio)."""
        evo, alloc, metrics = agent3._apply_evolution_engine(
            current_portfolio=None,
            target_allocation=target_allocation,
            target_metrics=target_metrics,
            regime="bull_low_vol",
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            rf_rate=0.04
        )
        
        assert evo["requires_rebalance"] is True
        assert evo["max_drift_detected"] == 0.0
        assert evo["portfolio_turnover"] == 0.0
        assert evo["estimated_transaction_costs"] == 0.0
        assert alloc == target_allocation

    def test_drift_under_threshold(self, agent3, target_allocation, target_metrics, cov_matrix, expected_returns):
        """Test that <5% drift does NOT trigger a rebalance."""
        # 3% drift on SPY/BND
        current = {
            "SPY": 0.53,
            "BND": 0.27,
            "GLD": 0.10,
            "BTC": 0.05,
            "CASH": 0.05
        }
        
        evo, alloc, metrics = agent3._apply_evolution_engine(
            current_portfolio=current,
            target_allocation=target_allocation,
            target_metrics=target_metrics,
            regime="bull_low_vol",
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            rf_rate=0.04
        )
        
        assert evo["requires_rebalance"] is False
        assert pytest.approx(evo["max_drift_detected"], 0.001) == 0.03
        assert evo["portfolio_turnover"] == 0.0
        assert evo["estimated_transaction_costs"] == 0.0
        
        # Allocations should be overridden to match CURRENT, not target
        assert alloc[0]["ticker"] == "SPY"
        assert alloc[0]["weight"] == 0.53
        assert alloc[1]["ticker"] == "BND"
        assert alloc[1]["weight"] == 0.27

    def test_drift_over_threshold(self, agent3, target_allocation, target_metrics, cov_matrix, expected_returns):
        """Test that >=5% drift DO trigger a rebalance and incurs costs."""
        # 6% drift on SPY/BND
        current = {
            "SPY": 0.56,
            "BND": 0.24,
            "GLD": 0.10,
            "BTC": 0.05,
            "CASH": 0.05
        }
        
        evo, alloc, metrics = agent3._apply_evolution_engine(
            current_portfolio=current,
            target_allocation=target_allocation,
            target_metrics=target_metrics,
            regime="bull_low_vol",
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            rf_rate=0.04
        )
        
        assert evo["requires_rebalance"] is True
        assert pytest.approx(evo["max_drift_detected"], 0.001) == 0.06
        
        # Turnover = sum(abs(diff)) / 2 = (0.06 + 0.06) / 2 = 0.06
        assert pytest.approx(evo["portfolio_turnover"], 0.001) == 0.06
        
        # Tx cost = 10 bps * turnover = 0.0010 * 0.06 = 0.00006
        assert pytest.approx(evo["estimated_transaction_costs"], 0.000001) == 0.00006
        
        # Expected return must be penalized by tx cost
        assert metrics["expected_annual_return"] == target_metrics["expected_annual_return"] - evo["estimated_transaction_costs"]
        
        # Allocations should SNAP to target
        assert alloc[0]["weight"] == 0.50

    def test_stressed_regime_forces_rebalance(self, agent3, target_allocation, target_metrics, cov_matrix, expected_returns):
        """Test that bear market crashes force rebalance even if drift is <5%."""
        # 1% drift
        current = {
            "SPY": 0.51,
            "BND": 0.29,
            "GLD": 0.10,
            "BTC": 0.05,
            "CASH": 0.05
        }
        
        evo, alloc, metrics = agent3._apply_evolution_engine(
            current_portfolio=current,
            target_allocation=target_allocation,
            target_metrics=target_metrics,
            regime="bear_high_vol",  # STRESS REGIME!
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            rf_rate=0.04
        )
        
        # Even though drift is only 1%, stressed regime forces a trade to ensure absolute safety alignment
        assert evo["requires_rebalance"] is True
        assert evo["max_drift_detected"] < 0.05
        assert alloc == target_allocation
