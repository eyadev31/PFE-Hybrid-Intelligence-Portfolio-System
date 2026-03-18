"""
Hybrid Intelligence Portfolio System -- Database ORM Models
================================================================
Production-grade SQLAlchemy ORM models matching dataset_shema.txt exactly.

Tables (PostgreSQL / AWS RDS):
  1. users              -- Authentication + risk score
  2. user_profile       -- Behavioral profile from Agent 2
  3. dynamic_questions  -- Market-calibrated questions from Agent 1/2
  4. user_answers       -- User responses for risk classification
  5. assets             -- Multi-asset registry (stocks, ETFs, crypto, forex, commodity)
  6. portfolios         -- Portfolio snapshots from Agent 3
  7. portfolio_allocations -- Per-asset allocation breakdown
  8. execution_orders   -- Binance/broker order tracking
  9. system_events      -- Audit-ready event logging

  + market_prices       -- OHLCV time-series (Timestream on AWS, local table for dev)
  + asset_features      -- ML feature store for Agent 1/3
  + user_behavior_features -- ML feature store for Agent 2
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, String, Float, Boolean, Integer, DateTime, Text,
    ForeignKey, Index, JSON, create_engine
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def generate_uuid():
    return str(uuid.uuid4())


# ════════════════════════════════════════════════════════
#  1. USERS TABLE
# ════════════════════════════════════════════════════════

class User(Base):
    """
    Core user table -- stores authentication info.
    risk_score is updated by Agent 2 after behavioral profiling.
    """
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_verified = Column(Boolean, default=False)
    country = Column(String(100), nullable=True)
    kyc_status = Column(String(50), default="pending")
    risk_score = Column(Float, nullable=True)  # Updated by Agent 2
    onboarding_completed = Column(Boolean, default=False)

    # Relationships
    profile = relationship("UserProfile", back_populates="user", uselist=False)
    answers = relationship("UserAnswer", back_populates="user")
    portfolios = relationship("Portfolio", back_populates="user")
    orders = relationship("ExecutionOrder", back_populates="user")
    events = relationship("SystemEvent", back_populates="user")


# ════════════════════════════════════════════════════════
#  2. USER_PROFILE TABLE
# ════════════════════════════════════════════════════════

class UserProfile(Base):
    """
    Stores structured profile extracted by Agent 2 (Behavioral Agent).
    Updated after each DAQ session.
    """
    __tablename__ = "user_profile"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    investment_horizon = Column(Integer, nullable=True)  # in months
    liquidity_preference = Column(String(50), nullable=True)
    income_level = Column(String(50), nullable=True)
    experience_level = Column(String(50), nullable=True)
    behavioral_type = Column(String(50), nullable=True)  # Agent 2 classification
    risk_score = Column(Float, nullable=True)
    max_acceptable_drawdown = Column(Float, nullable=True)
    emotional_stability = Column(String(50), nullable=True)
    stress_response_pattern = Column(String(50), nullable=True)
    consistency_score = Column(Float, nullable=True)
    session_id = Column(String(100), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="profile")


# ════════════════════════════════════════════════════════
#  3. DYNAMIC_QUESTIONS TABLE
# ════════════════════════════════════════════════════════

class DynamicQuestion(Base):
    """
    Agent 1 generates questions dynamically based on:
      - Market regime
      - Volatility
      - Macro signals
    market_context example: {"volatility": "high", "inflation": 3.4, "trend": "bearish"}
    """
    __tablename__ = "dynamic_questions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    question_text = Column(Text, nullable=False)
    category = Column(String(50), nullable=True)  # anchoring, herd_behavior, etc.
    market_context = Column(JSON, nullable=True)
    regime_tag = Column(String(50), nullable=True)
    difficulty = Column(Float, nullable=True)
    choices = Column(JSON, nullable=True)  # Array of choice objects
    behavioral_insight = Column(Text, nullable=True)
    session_id = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    answers = relationship("UserAnswer", back_populates="question")


# ════════════════════════════════════════════════════════
#  4. USER_ANSWERS TABLE
# ════════════════════════════════════════════════════════

class UserAnswer(Base):
    """
    Used by:
      - Agent 2 (Risk classifier)
      - Behavioral consistency analysis
    """
    __tablename__ = "user_answers"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    question_id = Column(String(36), ForeignKey("dynamic_questions.id"), nullable=False)
    answer_value = Column(Text, nullable=True)  # Choice ID (A, B, C, D)
    numeric_score = Column(Float, nullable=True)  # Risk signal of chosen answer
    response_time_ms = Column(Integer, nullable=True)  # For decision speed analysis
    session_id = Column(String(100), nullable=True)
    answered_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="answers")
    question = relationship("DynamicQuestion", back_populates="answers")


# ════════════════════════════════════════════════════════
#  5. ASSETS TABLE (Multi-Asset Support)
# ════════════════════════════════════════════════════════

class Asset(Base):
    """
    Multi-asset registry:
      BTCUSDT → crypto
      AAPL → stock
      GLD → ETF
      XAUUSD → commodity
      EUR/USD → forex
    """
    __tablename__ = "assets"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    symbol = Column(String(20), unique=True, nullable=False)
    name = Column(String(255), nullable=True)
    asset_type = Column(String(50), nullable=False)  # stock, etf, crypto, forex, commodity
    exchange = Column(String(50), nullable=True)
    base_currency = Column(String(10), nullable=True)
    quote_currency = Column(String(10), nullable=True)
    is_active = Column(Boolean, default=True)

    # Relationships
    allocations = relationship("PortfolioAllocation", back_populates="asset")
    orders = relationship("ExecutionOrder", back_populates="asset")
    features = relationship("AssetFeature", back_populates="asset")
    prices = relationship("MarketPrice", back_populates="asset")

    __table_args__ = (
        Index("idx_asset_symbol", "symbol"),
    )


# ════════════════════════════════════════════════════════
#  6. PORTFOLIOS TABLE
# ════════════════════════════════════════════════════════

class Portfolio(Base):
    """
    Each recommendation = new portfolio snapshot.
    Created by Agent 3, validated by Agent 4.
    """
    __tablename__ = "portfolios"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    total_value = Column(Float, nullable=True)
    expected_return = Column(Float, nullable=True)
    expected_volatility = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    sortino_ratio = Column(Float, nullable=True)
    max_drawdown_estimate = Column(Float, nullable=True)
    var_95 = Column(Float, nullable=True)
    cvar_95 = Column(Float, nullable=True)
    optimization_method = Column(String(50), nullable=True)
    strategy_type = Column(String(50), nullable=True)
    market_regime = Column(String(50), nullable=True)

    # Agent 4 validation
    validation_status = Column(String(50), nullable=True)  # approved/adjusted/rejected
    risk_level = Column(String(50), nullable=True)
    agent4_confidence = Column(Float, nullable=True)

    session_id = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="portfolios")
    allocations = relationship("PortfolioAllocation", back_populates="portfolio")

    __table_args__ = (
        Index("idx_portfolio_user_id", "user_id"),
    )


# ════════════════════════════════════════════════════════
#  7. PORTFOLIO_ALLOCATIONS TABLE
# ════════════════════════════════════════════════════════

class PortfolioAllocation(Base):
    """Stores per-asset allocation breakdown from Agent 3."""
    __tablename__ = "portfolio_allocations"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    portfolio_id = Column(String(36), ForeignKey("portfolios.id"), nullable=False)
    asset_id = Column(String(36), ForeignKey("assets.id"), nullable=False)
    allocation_percentage = Column(Float, nullable=False)
    expected_contribution = Column(Float, nullable=True)
    risk_contribution = Column(Float, nullable=True)
    rationale = Column(Text, nullable=True)

    # If Agent 4 adjusted
    adjusted_percentage = Column(Float, nullable=True)
    adjustment_reason = Column(Text, nullable=True)

    # Relationships
    portfolio = relationship("Portfolio", back_populates="allocations")
    asset = relationship("Asset", back_populates="allocations")


# ════════════════════════════════════════════════════════
#  8. EXECUTION_ORDERS TABLE
# ════════════════════════════════════════════════════════

class ExecutionOrder(Base):
    """
    If Binance integration enabled:
      - Store order ID from Binance
      - Track execution status
    """
    __tablename__ = "execution_orders"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    asset_id = Column(String(36), ForeignKey("assets.id"), nullable=False)
    portfolio_id = Column(String(36), ForeignKey("portfolios.id"), nullable=True)
    order_type = Column(String(20), nullable=False)  # buy / sell
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=True)
    total_value = Column(Float, nullable=True)
    status = Column(String(50), default="pending")  # pending/filled/cancelled/failed
    exchange = Column(String(50), nullable=True)
    exchange_order_id = Column(String(100), nullable=True)  # Binance order ID
    created_at = Column(DateTime, default=datetime.utcnow)
    executed_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="orders")
    asset = relationship("Asset", back_populates="orders")


# ════════════════════════════════════════════════════════
#  9. SYSTEM_EVENTS TABLE (Audit-Ready)
# ════════════════════════════════════════════════════════

class SystemEvent(Base):
    """
    Audit-ready event log. Tracks:
      - Portfolio generated
      - Risk reclassification
      - Market shock event
      - Order submitted
      - Agent execution records
    """
    __tablename__ = "system_events"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    event_type = Column(String(100), nullable=False)
    agent_id = Column(String(50), nullable=True)
    severity = Column(String(20), default="info")  # info/warning/critical
    event_data = Column(JSON, nullable=True)  # Renamed from metadata (reserved in SQLAlchemy)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="events")


# ════════════════════════════════════════════════════════
#  MARKET_PRICES TABLE (Local dev — Timestream on AWS)
# ════════════════════════════════════════════════════════

class MarketPrice(Base):
    """
    OHLCV time-series data.
    On AWS: use Amazon Timestream for production scale.
    Locally: stored in PostgreSQL/SQLite for development.
    """
    __tablename__ = "market_prices"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    asset_id = Column(String(36), ForeignKey("assets.id"), nullable=False)
    symbol = Column(String(20), nullable=False)
    time = Column(DateTime, nullable=False)
    open = Column(Float, nullable=True)
    high = Column(Float, nullable=True)
    low = Column(Float, nullable=True)
    close = Column(Float, nullable=True)
    volume = Column(Float, nullable=True)
    asset_type = Column(String(50), nullable=True)

    # Relationships
    asset = relationship("Asset", back_populates="prices")

    __table_args__ = (
        Index("idx_market_prices_time", "time"),
        Index("idx_market_prices_symbol", "symbol"),
    )


# ════════════════════════════════════════════════════════
#  ASSET_FEATURES TABLE (ML Feature Store)
# ════════════════════════════════════════════════════════

class AssetFeature(Base):
    """
    SageMaker Feature Store equivalent for local dev.
    Features computed by Agent 1 for consumption by Agent 3.
    """
    __tablename__ = "asset_features"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    asset_id = Column(String(36), ForeignKey("assets.id"), nullable=False)
    symbol = Column(String(20), nullable=False)
    computed_at = Column(DateTime, default=datetime.utcnow)

    # Features
    rolling_volatility_30d = Column(Float, nullable=True)
    rolling_return_90d = Column(Float, nullable=True)
    beta = Column(Float, nullable=True)
    momentum_score = Column(Float, nullable=True)
    macro_correlation = Column(Float, nullable=True)
    sentiment_score = Column(Float, nullable=True)
    rsi_14 = Column(Float, nullable=True)
    macd_signal = Column(Float, nullable=True)

    # Relationships
    asset = relationship("Asset", back_populates="features")


# ════════════════════════════════════════════════════════
#  USER_BEHAVIOR_FEATURES TABLE (ML Feature Store)
# ════════════════════════════════════════════════════════

class UserBehaviorFeature(Base):
    """
    SageMaker Feature Store equivalent for behavioral ML.
    Computed by Agent 2 for consumption by Agent 3/4.
    """
    __tablename__ = "user_behavior_features"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    computed_at = Column(DateTime, default=datetime.utcnow)

    # Features
    avg_response_time = Column(Float, nullable=True)
    risk_score_trend = Column(Float, nullable=True)
    answer_consistency_score = Column(Float, nullable=True)
    bias_count = Column(Integer, nullable=True)
    contradiction_count = Column(Integer, nullable=True)
    emotional_stability_score = Column(Float, nullable=True)
    sessions_completed = Column(Integer, default=0)
