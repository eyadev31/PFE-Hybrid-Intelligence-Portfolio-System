"""
Hybrid Intelligence Portfolio System -- Database Engine
=========================================================
Production-grade database connection management.

Supports:
  - PostgreSQL (production / AWS RDS)
  - SQLite (local development)

Config via environment variables:
  DATABASE_URL=postgresql://user:pass@host:5432/dbname   (production)
  DATABASE_URL=sqlite:///portfolio_system.db              (local dev)
"""

import os
import logging
from contextlib import contextmanager
from typing import Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session

from data.models import Base

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════
#  DATABASE CONFIGURATION
# ════════════════════════════════════════════════════════

def get_database_url() -> str:
    """
    Get database URL from environment or default to SQLite.

    For AWS RDS PostgreSQL:
      DATABASE_URL=postgresql://admin:password@your-rds-endpoint.amazonaws.com:5432/portfolio_db

    For local development:
      DATABASE_URL=sqlite:///portfolio_system.db
    """
    return os.getenv(
        "DATABASE_URL",
        "sqlite:///portfolio_system.db"
    )


# ════════════════════════════════════════════════════════
#  ENGINE & SESSION FACTORY
# ════════════════════════════════════════════════════════

_engine = None
_SessionFactory = None


def get_engine(url: Optional[str] = None):
    """Get or create the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        db_url = url or get_database_url()
        is_sqlite = db_url.startswith("sqlite")

        engine_kwargs = {
            "echo": os.getenv("DB_ECHO", "false").lower() == "true",
        }

        if not is_sqlite:
            # PostgreSQL connection pool settings (production)
            engine_kwargs.update({
                "pool_size": int(os.getenv("DB_POOL_SIZE", "10")),
                "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "20")),
                "pool_timeout": int(os.getenv("DB_POOL_TIMEOUT", "30")),
                "pool_recycle": int(os.getenv("DB_POOL_RECYCLE", "1800")),
                "pool_pre_ping": True,
            })

        _engine = create_engine(db_url, **engine_kwargs)

        # SQLite: enable WAL mode and foreign keys
        if is_sqlite:
            @event.listens_for(_engine, "connect")
            def set_sqlite_pragma(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        db_type = "PostgreSQL" if not is_sqlite else "SQLite"
        logger.info(f"Database engine created: {db_type}")

    return _engine


def get_session_factory():
    """Get or create the session factory."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine())
    return _SessionFactory


# ════════════════════════════════════════════════════════
#  SESSION CONTEXT MANAGERS
# ════════════════════════════════════════════════════════

@contextmanager
def get_session() -> Session:
    """
    Provide a transactional session scope.

    Usage:
        with get_session() as session:
            session.add(user)
            # auto-commits on exit, auto-rollbacks on exception
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ════════════════════════════════════════════════════════
#  DATABASE INITIALIZATION
# ════════════════════════════════════════════════════════

def init_db(url: Optional[str] = None):
    """
    Initialize the database -- create all tables.

    Call this once at startup or in migrations.
    """
    engine = get_engine(url)
    Base.metadata.create_all(engine)
    logger.info("Database initialized — all tables created")


def drop_db():
    """Drop all tables (for testing only)."""
    engine = get_engine()
    Base.metadata.drop_all(engine)
    logger.info("Database dropped — all tables removed")


def seed_assets():
    """
    Seed the assets table with the core portfolio universe
    and additional tracked assets.
    """
    from data.models import Asset

    SEED_ASSETS = [
        # Core 5-asset universe (Agent 3)
        {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust", "asset_type": "etf", "exchange": "NYSE", "base_currency": "USD"},
        {"symbol": "BND", "name": "Vanguard Total Bond Market ETF", "asset_type": "etf", "exchange": "NYSE", "base_currency": "USD"},
        {"symbol": "GLD", "name": "SPDR Gold Shares", "asset_type": "commodity", "exchange": "NYSE", "base_currency": "USD"},
        {"symbol": "BTC", "name": "Bitcoin", "asset_type": "crypto", "exchange": "Binance", "base_currency": "BTC", "quote_currency": "USDT"},
        {"symbol": "CASH", "name": "Cash Equivalent", "asset_type": "cash", "exchange": "N/A", "base_currency": "USD"},

        # Extended equity universe
        {"symbol": "QQQ", "name": "Invesco QQQ Trust", "asset_type": "etf", "exchange": "NASDAQ", "base_currency": "USD"},
        {"symbol": "IWM", "name": "iShares Russell 2000 ETF", "asset_type": "etf", "exchange": "NYSE", "base_currency": "USD"},
        {"symbol": "AAPL", "name": "Apple Inc.", "asset_type": "stock", "exchange": "NASDAQ", "base_currency": "USD"},
        {"symbol": "MSFT", "name": "Microsoft Corporation", "asset_type": "stock", "exchange": "NASDAQ", "base_currency": "USD"},
        {"symbol": "NVDA", "name": "NVIDIA Corporation", "asset_type": "stock", "exchange": "NASDAQ", "base_currency": "USD"},

        # Bonds & Fixed Income
        {"symbol": "TLT", "name": "iShares 20+ Year Treasury Bond ETF", "asset_type": "etf", "exchange": "NYSE", "base_currency": "USD"},
        {"symbol": "HYG", "name": "iShares iBoxx High Yield Corporate Bond ETF", "asset_type": "etf", "exchange": "NYSE", "base_currency": "USD"},

        # Additional crypto
        {"symbol": "ETH", "name": "Ethereum", "asset_type": "crypto", "exchange": "Binance", "base_currency": "ETH", "quote_currency": "USDT"},
        {"symbol": "SOL", "name": "Solana", "asset_type": "crypto", "exchange": "Binance", "base_currency": "SOL", "quote_currency": "USDT"},

        # Forex
        {"symbol": "EUR/USD", "name": "Euro / US Dollar", "asset_type": "forex", "exchange": "Forex", "base_currency": "EUR", "quote_currency": "USD"},
        {"symbol": "GBP/USD", "name": "British Pound / US Dollar", "asset_type": "forex", "exchange": "Forex", "base_currency": "GBP", "quote_currency": "USD"},

        # Commodities
        {"symbol": "SLV", "name": "iShares Silver Trust", "asset_type": "commodity", "exchange": "NYSE", "base_currency": "USD"},
        {"symbol": "USO", "name": "United States Oil Fund", "asset_type": "commodity", "exchange": "NYSE", "base_currency": "USD"},
    ]

    with get_session() as session:
        existing = {a.symbol for a in session.query(Asset).all()}
        added = 0
        for asset_data in SEED_ASSETS:
            if asset_data["symbol"] not in existing:
                session.add(Asset(**asset_data))
                added += 1
        logger.info(f"Seeded {added} assets ({len(existing)} already existed)")


def reset_db():
    """Full reset: drop + create + seed."""
    drop_db()
    init_db()
    seed_assets()
    logger.info("Database fully reset and seeded")
