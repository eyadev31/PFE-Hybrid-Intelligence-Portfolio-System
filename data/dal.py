"""
Hybrid Intelligence Portfolio System -- Data Access Layer
===========================================================
Per-agent CRUD operations + Feature Store + Event Logging.

This module bridges the 4 agents to the database:

  Agent 1 → persist_market_regime(), persist_questions()
  Agent 2 → persist_user_profile(), persist_answers(), update_risk_score()
  Agent 3 → persist_portfolio(), persist_allocations()
  Agent 4 → persist_risk_verdict(), update_validation()
  All     → log_event()
"""

import json
import logging
from datetime import datetime
from typing import Optional

from data.database import get_session
from data.models import (
    User, UserProfile, DynamicQuestion, UserAnswer,
    Asset, Portfolio, PortfolioAllocation, ExecutionOrder,
    SystemEvent, MarketPrice, AssetFeature, UserBehaviorFeature,
)

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════
#  USER OPERATIONS
# ════════════════════════════════════════════════════════

class UserDAL:
    """User and profile data access."""

    @staticmethod
    def create_user(email: str, country: str = None) -> str:
        """Create a new user and return the user ID."""
        with get_session() as session:
            user = User(email=email, country=country)
            session.add(user)
            session.flush()
            user_id = user.id
            logger.info(f"Created user: {user_id} ({email})")
            return user_id

    @staticmethod
    def get_user(user_id: str) -> Optional[dict]:
        """Get user by ID."""
        with get_session() as session:
            user = session.query(User).filter_by(id=user_id).first()
            if not user:
                return None
            return {
                "id": user.id,
                "email": user.email,
                "risk_score": user.risk_score,
                "kyc_status": user.kyc_status,
                "is_verified": user.is_verified,
                "onboarding_completed": user.onboarding_completed,
                "created_at": str(user.created_at),
            }

    @staticmethod
    def get_or_create_user(email: str) -> str:
        """Get existing user ID or create a new one."""
        with get_session() as session:
            user = session.query(User).filter_by(email=email).first()
            if user:
                return user.id
        return UserDAL.create_user(email=email)


# ════════════════════════════════════════════════════════
#  AGENT 1: MARKET REGIME + DYNAMIC QUESTIONS
# ════════════════════════════════════════════════════════

class Agent1DAL:
    """Data access for Agent 1 — Macro & Market Intelligence."""

    @staticmethod
    def persist_market_regime(agent1_output: dict, user_id: str = None):
        """
        Log Agent 1's market regime detection as a system event.
        """
        with get_session() as session:
            event = SystemEvent(
                user_id=user_id,
                event_type="market_regime_detected",
                agent_id="agent1_macro_intelligence",
                severity="info",
                event_data={
                    "regime": agent1_output.get("market_regime", {}),
                    "volatility_state": agent1_output.get("volatility_state", {}),
                    "systemic_risk": agent1_output.get("systemic_risk", {}),
                    "macro_regime": agent1_output.get("macro_environment", {}).get("macro_regime"),
                    "timestamp": agent1_output.get("timestamp"),
                },
            )
            session.add(event)
            logger.info("Persisted Agent 1 market regime to system_events")

    @staticmethod
    def persist_questions(
        questions: list[dict],
        session_id: str,
        market_context: dict = None,
        regime_tag: str = None,
    ) -> list[str]:
        """
        Persist dynamically generated questions to the database.
        Returns list of question IDs.
        """
        question_ids = []
        with get_session() as session:
            for q in questions:
                dq = DynamicQuestion(
                    question_text=q.get("question_text", ""),
                    category=q.get("category", ""),
                    market_context=market_context or q.get("market_context"),
                    regime_tag=regime_tag or q.get("market_context_used", ""),
                    difficulty=q.get("difficulty"),
                    choices=q.get("choices"),
                    behavioral_insight=q.get("behavioral_insight"),
                    session_id=session_id,
                )
                session.add(dq)
                session.flush()
                question_ids.append(dq.id)

            logger.info(f"Persisted {len(question_ids)} questions for session {session_id}")
        return question_ids

    @staticmethod
    def persist_asset_features(features: dict):
        """
        Persist ML asset features to the feature store.
        features: {symbol: {feature_name: value}}
        """
        with get_session() as session:
            for symbol, feature_vals in features.items():
                asset = session.query(Asset).filter_by(symbol=symbol).first()
                if not asset:
                    continue
                af = AssetFeature(
                    asset_id=asset.id,
                    symbol=symbol,
                    rolling_volatility_30d=feature_vals.get("rolling_volatility_30d"),
                    rolling_return_90d=feature_vals.get("rolling_return_90d"),
                    beta=feature_vals.get("beta"),
                    momentum_score=feature_vals.get("momentum_score"),
                    macro_correlation=feature_vals.get("macro_correlation"),
                    sentiment_score=feature_vals.get("sentiment_score"),
                    rsi_14=feature_vals.get("rsi_14"),
                    macd_signal=feature_vals.get("macd_signal"),
                )
                session.add(af)
            logger.info(f"Persisted asset features for {len(features)} symbols")


# ════════════════════════════════════════════════════════
#  AGENT 2: USER PROFILE + ANSWERS + RISK SCORE
# ════════════════════════════════════════════════════════

class Agent2DAL:
    """Data access for Agent 2 — Cognitive & Behavioral Profiling."""

    @staticmethod
    def persist_answers(
        user_id: str,
        answers: list[dict],
        question_ids: list[str],
        session_id: str,
    ):
        """
        Persist user answers linked to their questions.
        """
        with get_session() as session:
            for i, answer in enumerate(answers):
                q_id = question_ids[i] if i < len(question_ids) else None
                ua = UserAnswer(
                    user_id=user_id,
                    question_id=q_id,
                    answer_value=answer.get("choice_id", answer.get("answer_value", "")),
                    numeric_score=answer.get("risk_signal", answer.get("numeric_score")),
                    response_time_ms=answer.get("response_time_ms"),
                    session_id=session_id,
                )
                session.add(ua)
            logger.info(f"Persisted {len(answers)} answers for user {user_id}")

    @staticmethod
    def persist_user_profile(user_id: str, agent2_output: dict):
        """
        Create or update user profile from Agent 2 phase2 output.
        Also updates users.risk_score.
        """
        # Normalize: accept full output or just phase2_profile
        profile_data = agent2_output
        if "phase2_profile" in agent2_output:
            profile_data = agent2_output["phase2_profile"]

        risk_cls = profile_data.get("risk_classification", {})
        beh_profile = profile_data.get("behavioral_profile", {})
        session_id = profile_data.get("session_id", "")

        with get_session() as session:
            # Update user risk score
            user = session.query(User).filter_by(id=user_id).first()
            if user:
                user.risk_score = risk_cls.get("risk_score")
                user.onboarding_completed = True

            # Create or update profile
            profile = session.query(UserProfile).filter_by(user_id=user_id).first()
            if profile is None:
                profile = UserProfile(user_id=user_id)
                session.add(profile)

            profile.behavioral_type = risk_cls.get("behavioral_type")
            profile.risk_score = risk_cls.get("risk_score")
            profile.max_acceptable_drawdown = risk_cls.get("max_acceptable_drawdown")
            profile.liquidity_preference = risk_cls.get("liquidity_preference")
            profile.emotional_stability = beh_profile.get("emotional_stability")
            profile.stress_response_pattern = beh_profile.get("stress_response_pattern")
            profile.consistency_score = beh_profile.get("consistency_score")
            profile.session_id = session_id

            time_horizon = risk_cls.get("time_horizon", "medium")
            horizon_map = {"short": 12, "medium": 36, "long": 60}
            profile.investment_horizon = horizon_map.get(time_horizon, 36)

            logger.info(
                f"Persisted profile for user {user_id}: "
                f"type={profile.behavioral_type}, score={profile.risk_score}"
            )

    @staticmethod
    def persist_behavior_features(user_id: str, agent2_output: dict):
        """Persist user behavioral features to the feature store."""
        profile_data = agent2_output
        if "phase2_profile" in agent2_output:
            profile_data = agent2_output["phase2_profile"]

        beh = profile_data.get("behavioral_profile", {})

        with get_session() as session:
            feature = UserBehaviorFeature(
                user_id=user_id,
                avg_response_time=beh.get("avg_response_time_ms"),
                risk_score_trend=0.0,  # computed over multiple sessions
                answer_consistency_score=beh.get("consistency_score"),
                bias_count=len(beh.get("detected_biases", [])),
                contradiction_count=len(beh.get("contradiction_flags", [])),
                emotional_stability_score=beh.get("emotional_stability_score"),
                sessions_completed=1,
            )
            session.add(feature)
            logger.info(f"Persisted behavior features for user {user_id}")


# ════════════════════════════════════════════════════════
#  AGENT 3: PORTFOLIO + ALLOCATIONS
# ════════════════════════════════════════════════════════

class Agent3DAL:
    """Data access for Agent 3 — Quantitative Portfolio Strategist."""

    @staticmethod
    def persist_portfolio(user_id: str, agent3_output: dict) -> str:
        """
        Persist full portfolio snapshot and per-asset allocations.
        Returns portfolio ID.
        """
        metrics = agent3_output.get("portfolio_metrics", {})
        opt = agent3_output.get("optimization", {})

        with get_session() as session:
            # Create portfolio record
            portfolio = Portfolio(
                user_id=user_id,
                expected_return=metrics.get("expected_annual_return"),
                expected_volatility=metrics.get("expected_annual_volatility"),
                sharpe_ratio=metrics.get("sharpe_ratio"),
                sortino_ratio=metrics.get("sortino_ratio"),
                max_drawdown_estimate=metrics.get("max_drawdown_estimate"),
                var_95=metrics.get("value_at_risk_95"),
                cvar_95=metrics.get("cvar_95"),
                optimization_method=opt.get("method_used"),
                strategy_type=opt.get("strategy_type"),
                market_regime=agent3_output.get("market_regime"),
                session_id=agent3_output.get("session_id"),
            )
            session.add(portfolio)
            session.flush()
            portfolio_id = portfolio.id

            # Create per-asset allocations
            for alloc in agent3_output.get("allocation", []):
                ticker = alloc.get("ticker", "")
                asset = session.query(Asset).filter_by(symbol=ticker).first()
                if not asset:
                    # Auto-create asset if not in registry
                    asset = Asset(
                        symbol=ticker,
                        name=ticker,
                        asset_type=alloc.get("asset_class", "unknown"),
                    )
                    session.add(asset)
                    session.flush()

                pa = PortfolioAllocation(
                    portfolio_id=portfolio_id,
                    asset_id=asset.id,
                    allocation_percentage=alloc.get("weight", 0),
                    expected_contribution=alloc.get("expected_return", 0),
                    risk_contribution=alloc.get("risk_contribution", 0),
                    rationale=alloc.get("rationale", ""),
                )
                session.add(pa)

            logger.info(
                f"Persisted portfolio {portfolio_id} for user {user_id}: "
                f"strategy={opt.get('strategy_type')}, "
                f"allocations={len(agent3_output.get('allocation', []))}"
            )
            return portfolio_id


# ════════════════════════════════════════════════════════
#  AGENT 4: RISK VERDICT + ADJUSTMENTS
# ════════════════════════════════════════════════════════

class Agent4DAL:
    """Data access for Agent 4 — Meta-Risk & Supervision."""

    @staticmethod
    def persist_risk_verdict(
        user_id: str,
        portfolio_id: str,
        agent4_output: dict,
    ):
        """
        Update portfolio with Agent 4 validation results
        and persist adjusted allocations.
        """
        with get_session() as session:
            # Update portfolio validation status
            portfolio = session.query(Portfolio).filter_by(id=portfolio_id).first()
            if portfolio:
                portfolio.validation_status = agent4_output.get("validation_status")
                portfolio.risk_level = agent4_output.get("overall_risk_level")
                portfolio.agent4_confidence = agent4_output.get("confidence")

            # If adjusted, update allocation weights
            adjusted = agent4_output.get("adjusted_allocation", [])
            if adjusted and not agent4_output.get("original_allocation_preserved", True):
                for adj in adjusted:
                    ticker = adj.get("ticker", "")
                    asset = session.query(Asset).filter_by(symbol=ticker).first()
                    if not asset:
                        continue

                    pa = session.query(PortfolioAllocation).filter_by(
                        portfolio_id=portfolio_id,
                        asset_id=asset.id,
                    ).first()
                    if pa:
                        pa.adjusted_percentage = adj.get("adjusted_weight")
                        pa.adjustment_reason = adj.get("reason", "")

            # Log as system event
            verdict = agent4_output.get("risk_verdict", {})
            severity = "critical" if agent4_output.get("validation_status") == "rejected" else "info"
            event = SystemEvent(
                user_id=user_id,
                event_type="portfolio_risk_verdict",
                agent_id="agent4_risk_supervisor",
                severity=severity,
                event_data={
                    "portfolio_id": portfolio_id,
                    "validation_status": agent4_output.get("validation_status"),
                    "confidence": agent4_output.get("confidence"),
                    "audits_passed": agent4_output.get("audits_passed"),
                    "audits_failed": agent4_output.get("audits_failed"),
                    "decision": verdict.get("decision"),
                    "reasoning_summary": verdict.get("reasoning", "")[:500],
                },
            )
            session.add(event)

            logger.info(
                f"Persisted risk verdict for portfolio {portfolio_id}: "
                f"status={agent4_output.get('validation_status')}"
            )


# ════════════════════════════════════════════════════════
#  SYSTEM EVENT LOGGING (ALL AGENTS)
# ════════════════════════════════════════════════════════

class EventDAL:
    """Audit-ready event logging for compliance and debugging."""

    @staticmethod
    def log_event(
        event_type: str,
        agent_id: str = None,
        user_id: str = None,
        severity: str = "info",
        event_data: dict = None,
    ):
        """
        Log a system event.

        Event types:
          - agent_execution
          - market_regime_detected
          - risk_reclassification
          - portfolio_generated
          - portfolio_risk_verdict
          - market_shock_event
          - order_submitted
          - order_executed
        """
        with get_session() as session:
            event = SystemEvent(
                user_id=user_id,
                event_type=event_type,
                agent_id=agent_id,
                severity=severity,
                event_data=event_data or {},
            )
            session.add(event)

    @staticmethod
    def get_events(
        user_id: str = None,
        event_type: str = None,
        limit: int = 50,
    ) -> list[dict]:
        """Retrieve events for audit or debugging."""
        with get_session() as session:
            query = session.query(SystemEvent)
            if user_id:
                query = query.filter_by(user_id=user_id)
            if event_type:
                query = query.filter_by(event_type=event_type)
            query = query.order_by(SystemEvent.created_at.desc()).limit(limit)

            return [
                {
                    "id": e.id,
                    "event_type": e.event_type,
                    "agent_id": e.agent_id,
                    "severity": e.severity,
                    "metadata": e.event_data,
                    "created_at": str(e.created_at),
                }
                for e in query.all()
            ]


# ════════════════════════════════════════════════════════
#  MARKET PRICES (LOCAL DEV — Timestream on AWS)
# ════════════════════════════════════════════════════════

class MarketPriceDAL:
    """Store and retrieve OHLCV data (local PostgreSQL/SQLite, Timestream on AWS)."""

    @staticmethod
    def persist_prices(symbol: str, prices_df, asset_type: str = "stock"):
        """
        Persist a DataFrame of OHLCV prices to the database.
        prices_df: DataFrame with DatetimeIndex and OHLCV columns.
        """
        with get_session() as session:
            asset = session.query(Asset).filter_by(symbol=symbol).first()
            asset_id = asset.id if asset else None

            count = 0
            for idx, row in prices_df.iterrows():
                mp = MarketPrice(
                    asset_id=asset_id,
                    symbol=symbol,
                    time=idx if hasattr(idx, 'date') else datetime.now(),
                    open=row.get("open"),
                    high=row.get("high"),
                    low=row.get("low"),
                    close=row.get("close"),
                    volume=row.get("volume"),
                    asset_type=asset_type,
                )
                session.add(mp)
                count += 1
            logger.info(f"Persisted {count} price records for {symbol}")

    @staticmethod
    def get_latest_prices(symbol: str, limit: int = 252) -> list[dict]:
        """Get latest N price records for a symbol."""
        with get_session() as session:
            prices = (
                session.query(MarketPrice)
                .filter_by(symbol=symbol)
                .order_by(MarketPrice.time.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "time": str(p.time),
                    "open": p.open,
                    "high": p.high,
                    "low": p.low,
                    "close": p.close,
                    "volume": p.volume,
                }
                for p in reversed(prices)
            ]


# ════════════════════════════════════════════════════════
#  FULL PIPELINE PERSISTENCE
# ════════════════════════════════════════════════════════

class PipelineDAL:
    """
    Orchestrates database persistence for the full 4-agent pipeline.

    Usage:
        user_id = PipelineDAL.persist_full_pipeline(
            email="investor@example.com",
            agent1_output=agent1_result,
            agent2_output=agent2_result,
            agent3_output=agent3_result,
            agent4_output=agent4_result,
        )
    """

    @staticmethod
    def persist_full_pipeline(
        email: str,
        agent1_output: dict = None,
        agent2_output: dict = None,
        agent3_output: dict = None,
        agent4_output: dict = None,
    ) -> dict:
        """
        Persist the entire pipeline output to the database.
        Returns summary with IDs.
        """
        # Step 1: Get or create user
        user_id = UserDAL.get_or_create_user(email=email)

        # Step 2: Agent 1 — market regime
        if agent1_output:
            Agent1DAL.persist_market_regime(agent1_output, user_id)
            EventDAL.log_event(
                event_type="agent_execution",
                agent_id="agent1_macro_intelligence",
                user_id=user_id,
                event_data={"regime": agent1_output.get("market_regime", {}).get("primary_regime")},
            )

        # Step 3: Agent 2 — behavioral profile
        session_id = ""
        if agent2_output:
            profile_data = agent2_output
            if "phase2_profile" in agent2_output:
                profile_data = agent2_output["phase2_profile"]
            session_id = profile_data.get("session_id", "")

            Agent2DAL.persist_user_profile(user_id, agent2_output)
            Agent2DAL.persist_behavior_features(user_id, agent2_output)

            # Persist questions if present
            if "phase1_questions" in agent2_output:
                questions = agent2_output["phase1_questions"].get("questions", [])
                if questions:
                    regime = agent2_output["phase1_questions"].get(
                        "market_calibration", {}
                    ).get("regime_used")
                    Agent1DAL.persist_questions(
                        questions=questions,
                        session_id=session_id,
                        regime_tag=regime,
                    )

            EventDAL.log_event(
                event_type="agent_execution",
                agent_id="agent2_behavioral_intelligence",
                user_id=user_id,
                event_data={
                    "risk_score": profile_data.get("risk_classification", {}).get("risk_score"),
                    "behavioral_type": profile_data.get("risk_classification", {}).get("behavioral_type"),
                },
            )

        # Step 4: Agent 3 — portfolio
        portfolio_id = None
        if agent3_output:
            portfolio_id = Agent3DAL.persist_portfolio(user_id, agent3_output)
            EventDAL.log_event(
                event_type="portfolio_generated",
                agent_id="agent3_portfolio_strategist",
                user_id=user_id,
                event_data={
                    "portfolio_id": portfolio_id,
                    "strategy": agent3_output.get("optimization", {}).get("strategy_type"),
                },
            )

        # Step 5: Agent 4 — risk verdict
        if agent4_output and portfolio_id:
            Agent4DAL.persist_risk_verdict(user_id, portfolio_id, agent4_output)

        summary = {
            "user_id": user_id,
            "session_id": session_id,
            "portfolio_id": portfolio_id,
            "agents_persisted": [
                "agent1" if agent1_output else None,
                "agent2" if agent2_output else None,
                "agent3" if agent3_output else None,
                "agent4" if agent4_output else None,
            ],
        }
        summary["agents_persisted"] = [a for a in summary["agents_persisted"] if a]

        logger.info(f"Full pipeline persisted: {summary}")
        return summary
