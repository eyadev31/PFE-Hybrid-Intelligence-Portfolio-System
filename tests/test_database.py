"""
Database Integration Tests
============================
Tests covering:
  - Table creation & schema validation
  - User CRUD
  - Per-agent persistence (Agent 1-4)
  - Full pipeline persistence
  - Feature store
  - Event logging
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force in-memory SQLite for tests
os.environ["DATABASE_URL"] = "sqlite:///:memory:"


@pytest.fixture(autouse=True)
def setup_db():
    """Create fresh in-memory DB for each test."""
    from data.database import get_engine, init_db, _engine, _SessionFactory
    import data.database as db_mod
    # Reset globals
    db_mod._engine = None
    db_mod._SessionFactory = None
    init_db()
    from data.database import seed_assets
    seed_assets()
    yield
    db_mod._engine = None
    db_mod._SessionFactory = None


# ════════════════════════════════════════════════════════
#  SCHEMA & TABLE TESTS
# ════════════════════════════════════════════════════════

class TestSchema:
    def test_all_tables_created(self):
        from data.database import get_engine
        from sqlalchemy import inspect
        engine = get_engine()
        tables = inspect(engine).get_table_names()
        required = [
            "users", "user_profile", "dynamic_questions", "user_answers",
            "assets", "portfolios", "portfolio_allocations", "execution_orders",
            "system_events", "market_prices", "asset_features", "user_behavior_features",
        ]
        for t in required:
            assert t in tables, f"Missing table: {t}"

    def test_assets_seeded(self):
        from data.database import get_session
        from data.models import Asset
        with get_session() as session:
            assets = session.query(Asset).all()
            symbols = {a.symbol for a in assets}
            assert "SPY" in symbols
            assert "BTC" in symbols
            assert "GLD" in symbols
            assert "BND" in symbols
            assert "CASH" in symbols


# ════════════════════════════════════════════════════════
#  USER DAL TESTS
# ════════════════════════════════════════════════════════

class TestUserDAL:
    def test_create_user(self):
        from data.dal import UserDAL
        uid = UserDAL.create_user("test@example.com")
        assert uid is not None
        assert len(uid) == 36

    def test_get_user(self):
        from data.dal import UserDAL
        uid = UserDAL.create_user("get@test.com")
        user = UserDAL.get_user(uid)
        assert user["email"] == "get@test.com"

    def test_get_or_create_idempotent(self):
        from data.dal import UserDAL
        uid1 = UserDAL.get_or_create_user("same@test.com")
        uid2 = UserDAL.get_or_create_user("same@test.com")
        assert uid1 == uid2


# ════════════════════════════════════════════════════════
#  AGENT 1 DAL TESTS
# ════════════════════════════════════════════════════════

class TestAgent1DAL:
    def test_persist_market_regime(self):
        from data.dal import Agent1DAL, UserDAL, EventDAL
        uid = UserDAL.create_user("agent1_test@test.com")
        Agent1DAL.persist_market_regime(
            {"market_regime": {"primary_regime": "bull_low_vol"}},
            user_id=uid,
        )
        events = EventDAL.get_events(user_id=uid, event_type="market_regime_detected")
        assert len(events) == 1
        assert events[0]["metadata"]["regime"]["primary_regime"] == "bull_low_vol"

    def test_persist_questions(self):
        from data.dal import Agent1DAL
        from data.database import get_session
        from data.models import DynamicQuestion
        qids = Agent1DAL.persist_questions(
            questions=[
                {"question_text": "Test Q1", "category": "anchoring"},
                {"question_text": "Test Q2", "category": "herd"},
            ],
            session_id="test_session",
            regime_tag="bull",
        )
        assert len(qids) == 2
        with get_session() as session:
            assert session.query(DynamicQuestion).count() == 2


# ════════════════════════════════════════════════════════
#  AGENT 2 DAL TESTS
# ════════════════════════════════════════════════════════

class TestAgent2DAL:
    def test_persist_profile(self):
        from data.dal import Agent2DAL, UserDAL
        from data.database import get_session
        from data.models import User, UserProfile
        uid = UserDAL.create_user("agent2_test@test.com")
        Agent2DAL.persist_user_profile(uid, {
            "risk_classification": {
                "risk_score": 0.65,
                "behavioral_type": "growth_seeker",
                "max_acceptable_drawdown": 0.20,
                "liquidity_preference": "medium",
            },
            "behavioral_profile": {
                "emotional_stability": "stable",
                "consistency_score": 0.80,
            },
        })
        with get_session() as session:
            user = session.query(User).filter_by(id=uid).first()
            assert user.risk_score == 0.65
            profile = session.query(UserProfile).filter_by(user_id=uid).first()
            assert profile.behavioral_type == "growth_seeker"


# ════════════════════════════════════════════════════════
#  AGENT 3 DAL TESTS
# ════════════════════════════════════════════════════════

class TestAgent3DAL:
    def test_persist_portfolio(self):
        from data.dal import Agent3DAL, UserDAL
        from data.database import get_session
        from data.models import Portfolio, PortfolioAllocation
        uid = UserDAL.create_user("agent3_test@test.com")
        pid = Agent3DAL.persist_portfolio(uid, {
            "portfolio_metrics": {
                "expected_annual_return": 0.08,
                "expected_annual_volatility": 0.12,
                "sharpe_ratio": 0.55,
            },
            "optimization": {
                "method_used": "mean_variance",
                "strategy_type": "balanced",
            },
            "allocation": [
                {"ticker": "SPY", "weight": 0.40, "risk_contribution": 0.35},
                {"ticker": "BND", "weight": 0.30, "risk_contribution": 0.05},
                {"ticker": "GLD", "weight": 0.15, "risk_contribution": 0.10},
                {"ticker": "BTC", "weight": 0.05, "risk_contribution": 0.35},
                {"ticker": "CASH", "weight": 0.10, "risk_contribution": 0.00},
            ],
        })
        assert pid is not None
        with get_session() as session:
            portfolio = session.query(Portfolio).filter_by(id=pid).first()
            assert portfolio.strategy_type == "balanced"
            allocs = session.query(PortfolioAllocation).filter_by(portfolio_id=pid).all()
            assert len(allocs) == 5


# ════════════════════════════════════════════════════════
#  AGENT 4 DAL TESTS
# ════════════════════════════════════════════════════════

class TestAgent4DAL:
    def test_persist_risk_verdict(self):
        from data.dal import Agent3DAL, Agent4DAL, UserDAL, EventDAL
        from data.database import get_session
        from data.models import Portfolio
        uid = UserDAL.create_user("agent4_test@test.com")
        pid = Agent3DAL.persist_portfolio(uid, {
            "portfolio_metrics": {"sharpe_ratio": 0.50},
            "optimization": {"strategy_type": "balanced"},
            "allocation": [
                {"ticker": "SPY", "weight": 0.40},
                {"ticker": "BND", "weight": 0.30},
            ],
        })
        Agent4DAL.persist_risk_verdict(uid, pid, {
            "validation_status": "approved_with_adjustments",
            "overall_risk_level": "elevated",
            "confidence": 0.75,
            "original_allocation_preserved": False,
            "adjusted_allocation": [
                {"ticker": "SPY", "adjusted_weight": 0.35, "reason": "regime cap"},
                {"ticker": "BND", "adjusted_weight": 0.35, "reason": "defensive floor"},
            ],
            "risk_verdict": {"decision": "approved_with_adjustments", "reasoning": "Test"},
            "audits_passed": 3,
            "audits_failed": 2,
        })
        with get_session() as session:
            portfolio = session.query(Portfolio).filter_by(id=pid).first()
            assert portfolio.validation_status == "approved_with_adjustments"
        events = EventDAL.get_events(user_id=uid, event_type="portfolio_risk_verdict")
        assert len(events) == 1


# ════════════════════════════════════════════════════════
#  FULL PIPELINE TEST
# ════════════════════════════════════════════════════════

class TestPipelineDAL:
    def test_full_pipeline(self):
        from data.dal import PipelineDAL
        summary = PipelineDAL.persist_full_pipeline(
            email="pipeline@test.com",
            agent1_output={
                "market_regime": {"primary_regime": "bull_low_vol"},
                "volatility_state": {"current_state": "normal"},
                "systemic_risk": {"risk_category": "low"},
            },
            agent2_output={
                "phase2_profile": {
                    "risk_classification": {
                        "risk_score": 0.55,
                        "behavioral_type": "moderate_balanced",
                        "max_acceptable_drawdown": 0.15,
                    },
                    "behavioral_profile": {
                        "consistency_score": 0.80,
                        "emotional_stability": "stable",
                    },
                    "session_id": "test_pipeline",
                },
            },
            agent3_output={
                "portfolio_metrics": {"expected_annual_return": 0.065, "sharpe_ratio": 0.45},
                "optimization": {"strategy_type": "balanced", "method_used": "risk_parity"},
                "allocation": [
                    {"ticker": "SPY", "weight": 0.35, "risk_contribution": 0.30},
                    {"ticker": "BND", "weight": 0.25, "risk_contribution": 0.05},
                    {"ticker": "GLD", "weight": 0.20, "risk_contribution": 0.15},
                    {"ticker": "BTC", "weight": 0.05, "risk_contribution": 0.30},
                    {"ticker": "CASH", "weight": 0.15, "risk_contribution": 0.00},
                ],
                "session_id": "test_pipeline",
            },
            agent4_output={
                "validation_status": "approved",
                "overall_risk_level": "low",
                "confidence": 0.90,
                "original_allocation_preserved": True,
                "adjusted_allocation": [],
                "risk_verdict": {"decision": "approved", "reasoning": "All good"},
                "audits_passed": 5,
                "audits_failed": 0,
            },
        )
        assert summary["user_id"] is not None
        assert summary["portfolio_id"] is not None
        assert "agent1" in summary["agents_persisted"]
        assert "agent4" in summary["agents_persisted"]

    def test_event_logging(self):
        from data.dal import EventDAL
        EventDAL.log_event(
            event_type="market_shock_event",
            agent_id="agent1_macro_intelligence",
            severity="critical",
            event_data={"vix_spike": 35, "regime_change": True},
        )
        events = EventDAL.get_events(event_type="market_shock_event")
        assert len(events) == 1
        assert events[0]["severity"] == "critical"
