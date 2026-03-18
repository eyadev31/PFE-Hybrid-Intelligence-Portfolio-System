"""
Agent 4 -- Meta-Risk & Supervision Agent Tests
=============================================================
Comprehensive tests covering:
  - 5 risk audits (regime, profile, drawdown, concentration, coherence)
  - Allocation adjuster
  - Schema validation
  - Full pipeline integration (approve, adjust, reject scenarios)
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ================================================================
#  FIXTURES
# ================================================================

@pytest.fixture
def safe_agent1():
    """Bull regime, low risk — should pass most audits."""
    return {
        "market_regime": {
            "primary_regime": "bull_low_vol",
            "confidence": 0.90,
            "models_agree": True,
        },
        "volatility_state": {"current_state": "normal", "vix_level": 15},
        "systemic_risk": {"overall_risk_level": 0.10, "risk_category": "low"},
        "macro_environment": {"key_indicators": {"fed_funds_rate": 4.5}},
        "cross_asset_analysis": {"median_correlation": 0.20},
    }


@pytest.fixture
def dangerous_agent1():
    """Bear + high vol + elevated risk — should trigger many audits."""
    return {
        "market_regime": {
            "primary_regime": "bear_high_vol",
            "confidence": 0.85,
            "models_agree": False,
        },
        "volatility_state": {"current_state": "elevated", "vix_level": 32},
        "systemic_risk": {"overall_risk_level": 0.65, "risk_category": "elevated"},
        "macro_environment": {"key_indicators": {"fed_funds_rate": 5.5}},
        "cross_asset_analysis": {"median_correlation": 0.60},
    }


@pytest.fixture
def conservative_agent2():
    return {
        "risk_classification": {
            "risk_score": 0.25,
            "behavioral_type": "conservative_stable",
            "max_acceptable_drawdown": 0.08,
            "liquidity_preference": "high",
            "time_horizon": "short",
        },
        "behavioral_profile": {
            "consistency_score": 0.85,
            "emotional_stability": "stable",
            "contradiction_flags": [],
        },
    }


@pytest.fixture
def moderate_agent2():
    return {
        "risk_classification": {
            "risk_score": 0.55,
            "behavioral_type": "moderate_balanced",
            "max_acceptable_drawdown": 0.15,
            "liquidity_preference": "medium",
        },
        "behavioral_profile": {
            "consistency_score": 0.78,
            "emotional_stability": "stable",
            "contradiction_flags": [],
        },
    }


@pytest.fixture
def safe_agent3():
    """Balanced allocation — should pass most audits."""
    return {
        "allocation": [
            {"ticker": "SPY", "asset_class": "equity", "weight": 0.30,
             "risk_contribution": 0.35},
            {"ticker": "BND", "asset_class": "bond", "weight": 0.25,
             "risk_contribution": 0.05},
            {"ticker": "GLD", "asset_class": "commodity", "weight": 0.20,
             "risk_contribution": 0.15},
            {"ticker": "BTC", "asset_class": "crypto", "weight": 0.05,
             "risk_contribution": 0.25},
            {"ticker": "CASH", "asset_class": "cash", "weight": 0.20,
             "risk_contribution": 0.0},
        ],
        "portfolio_metrics": {
            "expected_annual_return": 0.065,
            "expected_annual_volatility": 0.09,
            "sharpe_ratio": 0.50,
            "max_drawdown_estimate": 0.08,
        },
        "monte_carlo": {
            "num_simulations": 10000,
            "median_max_drawdown": 0.07,
            "probability_of_loss": 0.18,
            "probability_of_severe_loss": 0.01,
            "simulation_var_95": 0.08,
            "simulation_cvar_95": 0.11,
            "worst_case_return": -0.15,
        },
        "optimization": {
            "method_used": "risk_parity",
            "strategy_type": "defensive_growth",
        },
        "session_id": "test_safe",
    }


@pytest.fixture
def dangerous_agent3():
    """Aggressive allocation — should trigger multiple audit failures."""
    return {
        "allocation": [
            {"ticker": "SPY", "asset_class": "equity", "weight": 0.50,
             "risk_contribution": 0.40},
            {"ticker": "BND", "asset_class": "bond", "weight": 0.05,
             "risk_contribution": 0.01},
            {"ticker": "GLD", "asset_class": "commodity", "weight": 0.10,
             "risk_contribution": 0.08},
            {"ticker": "BTC", "asset_class": "crypto", "weight": 0.30,
             "risk_contribution": 0.51},
            {"ticker": "CASH", "asset_class": "cash", "weight": 0.05,
             "risk_contribution": 0.0},
        ],
        "portfolio_metrics": {
            "expected_annual_return": 0.12,
            "expected_annual_volatility": 0.35,
            "sharpe_ratio": 0.23,
            "max_drawdown_estimate": 0.30,
        },
        "monte_carlo": {
            "num_simulations": 10000,
            "median_max_drawdown": 0.30,
            "probability_of_loss": 0.45,
            "probability_of_severe_loss": 0.25,
            "simulation_var_95": 0.30,
            "simulation_cvar_95": 0.42,
            "worst_case_return": -0.55,
        },
        "optimization": {
            "method_used": "mean_variance",
            "strategy_type": "aggressive_growth",
        },
        "session_id": "test_dangerous",
    }


# ================================================================
#  RISK AUDITOR TESTS
# ================================================================

class TestRegimeConsistency:
    def test_pass_in_bull_regime(self, safe_agent1, safe_agent3):
        from ml.risk_auditor import RiskAuditor
        result = RiskAuditor.audit_regime_consistency(safe_agent1, safe_agent3)
        assert result["verdict"] == "pass"

    def test_fail_aggressive_in_bear(self, dangerous_agent1, dangerous_agent3):
        from ml.risk_auditor import RiskAuditor
        result = RiskAuditor.audit_regime_consistency(dangerous_agent1, dangerous_agent3)
        assert result["verdict"] in ("fail", "warning")
        assert result["severity"] > 0.3

    def test_crypto_flagged_in_high_vol(self, dangerous_agent1, dangerous_agent3):
        from ml.risk_auditor import RiskAuditor
        result = RiskAuditor.audit_regime_consistency(dangerous_agent1, dangerous_agent3)
        assert "crypto" in result["finding"].lower() or "btc" in result["finding"].lower()


class TestProfileAlignment:
    def test_pass_balanced_moderate(self, moderate_agent2, safe_agent3):
        from ml.risk_auditor import RiskAuditor
        result = RiskAuditor.audit_profile_alignment(moderate_agent2, safe_agent3)
        assert result["verdict"] == "pass"

    def test_fail_aggressive_for_conservative(self, conservative_agent2, dangerous_agent3):
        from ml.risk_auditor import RiskAuditor
        result = RiskAuditor.audit_profile_alignment(conservative_agent2, dangerous_agent3)
        assert result["verdict"] in ("fail", "warning")

    def test_liquidity_check(self, conservative_agent2, dangerous_agent3):
        from ml.risk_auditor import RiskAuditor
        result = RiskAuditor.audit_profile_alignment(conservative_agent2, dangerous_agent3)
        # Conservative with high liquidity should flag low cash
        assert result["severity"] > 0


class TestDrawdownGuardrails:
    def test_pass_within_tolerance(self, moderate_agent2, safe_agent3):
        from ml.risk_auditor import RiskAuditor
        result = RiskAuditor.audit_drawdown_guardrails(moderate_agent2, safe_agent3)
        assert result["verdict"] == "pass"

    def test_fail_exceeds_tolerance(self, conservative_agent2, dangerous_agent3):
        from ml.risk_auditor import RiskAuditor
        result = RiskAuditor.audit_drawdown_guardrails(conservative_agent2, dangerous_agent3)
        assert result["verdict"] == "fail"
        assert result["severity"] > 0.5


class TestConcentration:
    def test_pass_diversified(self, safe_agent1, safe_agent3):
        from ml.risk_auditor import RiskAuditor
        result = RiskAuditor.audit_concentration(safe_agent1, safe_agent3)
        assert result["verdict"] == "pass"

    def test_fail_concentrated(self, dangerous_agent1, dangerous_agent3):
        from ml.risk_auditor import RiskAuditor
        result = RiskAuditor.audit_concentration(dangerous_agent1, dangerous_agent3)
        assert result["severity"] > 0.2

    def test_hhi_reported(self, safe_agent1, safe_agent3):
        from ml.risk_auditor import RiskAuditor
        result = RiskAuditor.audit_concentration(safe_agent1, safe_agent3)
        assert "hhi_index" in result["details"]


class TestCrossAgentCoherence:
    def test_pass_coherent(self, safe_agent1, moderate_agent2, safe_agent3):
        from ml.risk_auditor import RiskAuditor
        result = RiskAuditor.audit_cross_agent_coherence(
            safe_agent1, moderate_agent2, safe_agent3
        )
        assert result["verdict"] == "pass"

    def test_fail_bear_with_aggro(self, dangerous_agent1, moderate_agent2, dangerous_agent3):
        from ml.risk_auditor import RiskAuditor
        result = RiskAuditor.audit_cross_agent_coherence(
            dangerous_agent1, moderate_agent2, dangerous_agent3
        )
        assert result["verdict"] in ("fail", "warning")

    def test_all_audits_run(self, safe_agent1, moderate_agent2, safe_agent3):
        from ml.risk_auditor import RiskAuditor
        results = RiskAuditor.run_all_audits(safe_agent1, moderate_agent2, safe_agent3)
        assert len(results) == 5


# ================================================================
#  ALLOCATION ADJUSTER TESTS
# ================================================================

class TestAllocationAdjuster:
    def test_adjustments_sum_to_one(self, dangerous_agent1, conservative_agent2, dangerous_agent3):
        from ml.allocation_adjuster import AllocationAdjuster
        from ml.risk_auditor import RiskAuditor
        audits = RiskAuditor.run_all_audits(dangerous_agent1, conservative_agent2, dangerous_agent3)
        adjusted = AllocationAdjuster.adjust(
            dangerous_agent3["allocation"], audits,
            dangerous_agent1, conservative_agent2
        )
        total = sum(a["adjusted_weight"] for a in adjusted)
        assert abs(total - 1.0) < 0.02

    def test_crypto_reduced_in_bear(self, dangerous_agent1, conservative_agent2, dangerous_agent3):
        from ml.allocation_adjuster import AllocationAdjuster
        from ml.risk_auditor import RiskAuditor
        audits = RiskAuditor.run_all_audits(dangerous_agent1, conservative_agent2, dangerous_agent3)
        adjusted = AllocationAdjuster.adjust(
            dangerous_agent3["allocation"], audits,
            dangerous_agent1, conservative_agent2
        )
        btc = next(a for a in adjusted if a["ticker"] == "BTC")
        assert btc["adjusted_weight"] < btc["original_weight"]

    def test_no_negatives(self, dangerous_agent1, moderate_agent2, dangerous_agent3):
        from ml.allocation_adjuster import AllocationAdjuster
        from ml.risk_auditor import RiskAuditor
        audits = RiskAuditor.run_all_audits(dangerous_agent1, moderate_agent2, dangerous_agent3)
        adjusted = AllocationAdjuster.adjust(
            dangerous_agent3["allocation"], audits,
            dangerous_agent1, moderate_agent2
        )
        assert all(a["adjusted_weight"] >= 0 for a in adjusted)


# ================================================================
#  SCHEMA TESTS
# ================================================================

class TestSchemas:
    def test_agent4_output_schema(self):
        from schemas.agent4_output import Agent4Output
        data = {
            "timestamp": "2026-02-26T10:00:00Z",
            "validation_status": "approved",
            "risk_verdict": {"decision": "approved"},
        }
        output = Agent4Output(**data)
        assert output.validation_status == "approved"

    def test_validation(self):
        from schemas.agent4_output import validate_agent4_output
        valid, err = validate_agent4_output({
            "timestamp": "2026-02-26T10:00:00Z",
            "validation_status": "approved_with_adjustments",
            "risk_verdict": {"decision": "approved_with_adjustments"},
        })
        assert valid


# ================================================================
#  FULL PIPELINE INTEGRATION
# ================================================================

class TestAgent4Pipeline:
    def test_mock_pipeline_runs(self):
        from agents.agent4_supervisor import Agent4RiskSupervisor
        agent = Agent4RiskSupervisor()
        result = agent.run_mock()
        assert "validation_status" in result
        assert result["validation_status"] in (
            "approved", "approved_with_adjustments", "rejected"
        )

    def test_mock_flags_issues(self):
        """Mock mode uses intentionally bad allocation — should NOT approve as-is."""
        from agents.agent4_supervisor import Agent4RiskSupervisor
        agent = Agent4RiskSupervisor()
        result = agent.run_mock()
        assert result["validation_status"] in ("approved_with_adjustments", "rejected")
        assert result["audits_failed"] > 0 or result["audits_warned"] > 0

    def test_safe_approval(self, safe_agent1, moderate_agent2, safe_agent3):
        from agents.agent4_supervisor import Agent4RiskSupervisor
        agent = Agent4RiskSupervisor()
        result = agent.run(safe_agent1, moderate_agent2, safe_agent3)
        assert result["validation_status"] == "approved"
        assert result["original_allocation_preserved"] is True

    def test_dangerous_flagged(self, dangerous_agent1, conservative_agent2, dangerous_agent3):
        from agents.agent4_supervisor import Agent4RiskSupervisor
        agent = Agent4RiskSupervisor()
        result = agent.run(dangerous_agent1, conservative_agent2, dangerous_agent3)
        assert result["validation_status"] in ("approved_with_adjustments", "rejected")

    def test_audit_trail_present(self):
        from agents.agent4_supervisor import Agent4RiskSupervisor
        agent = Agent4RiskSupervisor()
        result = agent.run_mock()
        assert len(result["risk_audits"]) == 5
        assert result["total_audits"] == 5

    def test_verdict_has_reasoning(self):
        from agents.agent4_supervisor import Agent4RiskSupervisor
        agent = Agent4RiskSupervisor()
        result = agent.run_mock()
        verdict = result["risk_verdict"]
        assert "reasoning" in verdict
        assert len(verdict["reasoning"]) > 20

    def test_metadata_present(self):
        from agents.agent4_supervisor import Agent4RiskSupervisor
        agent = Agent4RiskSupervisor()
        result = agent.run_mock()
        meta = result["agent_metadata"]
        assert meta["agent_id"] == "agent4_risk_supervisor"
        assert len(meta["execution_log"]) >= 3
