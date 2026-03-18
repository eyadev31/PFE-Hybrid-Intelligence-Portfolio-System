"""
Hybrid Intelligence Portfolio System — Macroeconomic Data Layer
================================================================
Fetches institutional-grade macroeconomic indicators from FRED
(Federal Reserve Economic Data) — the gold standard for macro data.

Computes derived indicators:
  - Yield curve spreads
  - Inflation momentum
  - Leading indicator trends
  - Money supply growth rates
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from fredapi import Fred

from config.settings import MacroConfig, APIKeys

logger = logging.getLogger(__name__)


class MacroDataFetcher:
    """
    Fetches and processes macroeconomic data from FRED.
    
    Architecture:
      1. Raw data fetch with caching
      2. Derived indicator computation
      3. Normalization into unified macro snapshot
    """

    _fred_instance: Optional[Fred] = None
    _cache: dict[str, tuple[datetime, pd.Series]] = {}
    _cache_ttl = timedelta(hours=1)  # Macro data updates infrequently

    @classmethod
    def _get_fred(cls) -> Fred:
        """Lazy-initialize FRED client with API key validation."""
        if cls._fred_instance is None:
            key = APIKeys.FRED_API_KEY
            if not key or key == "your_fred_api_key_here":
                raise ValueError(
                    "FRED API key not configured. "
                    "Set FRED_API_KEY in your .env file. "
                    "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
                )
            cls._fred_instance = Fred(api_key=key)
        return cls._fred_instance

    @classmethod
    def _fetch_series(
        cls,
        series_id: str,
        lookback_years: int = MacroConfig.LOOKBACK_YEARS,
    ) -> pd.Series:
        """
        Fetch a single FRED series with caching.

        Args:
            series_id: FRED series identifier (e.g., 'FEDFUNDS')
            lookback_years: How many years of history to fetch

        Returns:
            pd.Series with DateTimeIndex
        """
        # Check cache
        cache_key = f"fred_{series_id}_{lookback_years}"
        if cache_key in cls._cache:
            cached_time, cached_data = cls._cache[cache_key]
            if datetime.utcnow() - cached_time < cls._cache_ttl:
                return cached_data.copy()

        try:
            fred = cls._get_fred()
            start_date = datetime.now() - timedelta(days=lookback_years * 365)
            series = fred.get_series(series_id, observation_start=start_date)

            if series is not None and not series.empty:
                series = series.dropna()
                cls._cache[cache_key] = (datetime.utcnow(), series.copy())
                logger.info(f"FRED: fetched {series_id} — {len(series)} observations")
                return series
            else:
                logger.warning(f"FRED: no data for {series_id}")
                return pd.Series(dtype=float)

        except Exception as e:
            logger.error(f"FRED API error for {series_id}: {e}")
            return pd.Series(dtype=float)

    @classmethod
    def fetch_all_series(cls) -> dict[str, pd.Series]:
        """
        Fetch all configured FRED series.

        Returns:
            Dict mapping human-readable name -> pd.Series
        """
        logger.info("Fetching all macroeconomic series from FRED...")
        result: dict[str, pd.Series] = {}

        for name, series_id in MacroConfig.SERIES.items():
            series = cls._fetch_series(series_id)
            if not series.empty:
                result[name] = series

        logger.info(f"FRED: fetched {len(result)}/{len(MacroConfig.SERIES)} series")
        return result

    @classmethod
    def compute_macro_snapshot(cls) -> dict:
        """
        Compute a comprehensive macroeconomic environment snapshot.

        Returns structured dict with:
          - Current values for all indicators
          - Derived indicators (spreads, growth rates, trends)
          - Macro regime classification
          - Confidence scores
        """
        logger.info("Computing macroeconomic environment snapshot...")
        raw = cls.fetch_all_series()

        snapshot = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "current_values": {},
            "derived_indicators": {},
            "monetary_policy": {},
            "inflation": {},
            "growth": {},
            "liquidity": {},
            "labor": {},
            "sentiment": {},
            "data_quality": {
                "total_series": len(MacroConfig.SERIES),
                "fetched_series": len(raw),
                "missing_series": [],
            },
        }

        # Track missing series
        for name in MacroConfig.SERIES:
            if name not in raw:
                snapshot["data_quality"]["missing_series"].append(name)

        # ── Current Values ───────────────────────────────
        for name, series in raw.items():
            if not series.empty:
                snapshot["current_values"][name] = {
                    "value": round(float(series.iloc[-1]), 4),
                    "date": str(series.index[-1].date()),
                    "prev_value": round(float(series.iloc[-2]), 4) if len(series) > 1 else None,
                }

        # ── Monetary Policy ──────────────────────────────
        snapshot["monetary_policy"] = cls._analyze_monetary_policy(raw)

        # ── Inflation ────────────────────────────────────
        snapshot["inflation"] = cls._analyze_inflation(raw)

        # ── Growth ───────────────────────────────────────
        snapshot["growth"] = cls._analyze_growth(raw)

        # ── Liquidity ────────────────────────────────────
        snapshot["liquidity"] = cls._analyze_liquidity(raw)

        # ── Labor Market ─────────────────────────────────
        snapshot["labor"] = cls._analyze_labor(raw)

        # ── Derived: Yield Curve ─────────────────────────
        snapshot["derived_indicators"]["yield_curve"] = cls._compute_yield_curve(raw)

        logger.info("Macro snapshot computed successfully")
        return snapshot

    # ─────────────────────────────────────────────────
    #  ANALYTICAL SUB-MODULES
    # ─────────────────────────────────────────────────

    @classmethod
    def _analyze_monetary_policy(cls, raw: dict[str, pd.Series]) -> dict:
        """Classify monetary policy stance from Fed Funds rate trajectory."""
        result = {"state": "unknown", "confidence": 0.0, "details": {}}

        fed_funds = raw.get("fed_funds_rate")
        if fed_funds is None or fed_funds.empty:
            return result

        current_rate = float(fed_funds.iloc[-1])
        result["details"]["current_fed_funds"] = round(current_rate, 2)

        # 6-month trajectory
        if len(fed_funds) >= 6:
            rate_6m_ago = float(fed_funds.iloc[-6]) if len(fed_funds) >= 6 else current_rate
            rate_change = current_rate - rate_6m_ago
            result["details"]["6m_change"] = round(rate_change, 2)

            if rate_change > 0.5:
                result["state"] = "aggressive_tightening"
                result["confidence"] = min(0.95, 0.7 + abs(rate_change) * 0.1)
            elif rate_change > 0.1:
                result["state"] = "tightening"
                result["confidence"] = 0.80
            elif rate_change < -0.5:
                result["state"] = "aggressive_easing"
                result["confidence"] = min(0.95, 0.7 + abs(rate_change) * 0.1)
            elif rate_change < -0.1:
                result["state"] = "easing"
                result["confidence"] = 0.80
            else:
                result["state"] = "neutral_hold"
                result["confidence"] = 0.70

        # Yield curve context
        t10y = raw.get("treasury_10y")
        t2y = raw.get("treasury_2y")
        if t10y is not None and t2y is not None and not t10y.empty and not t2y.empty:
            spread = float(t10y.iloc[-1]) - float(t2y.iloc[-1])
            result["details"]["yield_spread_10y2y"] = round(spread, 3)
            result["details"]["yield_curve_inverted"] = spread < 0

        return result

    @classmethod
    def _analyze_inflation(cls, raw: dict[str, pd.Series]) -> dict:
        """Classify inflation state from CPI and PCE data."""
        result = {"state": "unknown", "confidence": 0.0, "details": {}}

        cpi = raw.get("cpi_yoy")
        if cpi is None or cpi.empty:
            return result

        # CPI is reported as index; compute YoY % change
        if len(cpi) >= 12:
            current_cpi = float(cpi.iloc[-1])
            cpi_12m_ago = float(cpi.iloc[-12]) if len(cpi) >= 12 else current_cpi
            yoy_inflation = ((current_cpi / cpi_12m_ago) - 1) * 100
            result["details"]["cpi_yoy_pct"] = round(yoy_inflation, 2)

            # 3-month momentum
            if len(cpi) >= 3:
                cpi_3m_ago = float(cpi.iloc[-3])
                mom_3m = ((current_cpi / cpi_3m_ago) - 1) * 100 * 4  # annualized
                result["details"]["cpi_3m_annualized"] = round(mom_3m, 2)
                momentum = "accelerating" if mom_3m > yoy_inflation else "decelerating"
                result["details"]["inflation_momentum"] = momentum

            # Classify state
            if yoy_inflation < 0:
                result["state"] = "deflation"
                result["confidence"] = 0.90
            elif yoy_inflation < 1.5:
                result["state"] = "low_inflation"
                result["confidence"] = 0.85
            elif yoy_inflation < 2.5:
                result["state"] = "target_range"
                result["confidence"] = 0.85
            elif yoy_inflation < 4.0:
                result["state"] = "above_target"
                result["confidence"] = 0.80
            elif yoy_inflation < 6.0:
                result["state"] = "elevated"
                result["confidence"] = 0.85
            else:
                result["state"] = "high_inflation"
                result["confidence"] = 0.90

        # Breakeven inflation expectations
        breakeven = raw.get("breakeven_5y")
        if breakeven is not None and not breakeven.empty:
            result["details"]["breakeven_5y"] = round(float(breakeven.iloc[-1]), 2)

        return result

    @classmethod
    def _analyze_growth(cls, raw: dict[str, pd.Series]) -> dict:
        """Classify economic growth state."""
        result = {"state": "unknown", "confidence": 0.0, "details": {}}

        gdp = raw.get("gdp_growth")
        if gdp is not None and not gdp.empty:
            current_gdp = float(gdp.iloc[-1])
            result["details"]["gdp_growth_pct"] = round(current_gdp, 2)

            if current_gdp < -1.0:
                result["state"] = "recession"
                result["confidence"] = 0.90
            elif current_gdp < 0.5:
                result["state"] = "stagnation"
                result["confidence"] = 0.80
            elif current_gdp < 2.5:
                result["state"] = "moderate_growth"
                result["confidence"] = 0.75
            elif current_gdp < 4.0:
                result["state"] = "strong_growth"
                result["confidence"] = 0.80
            else:
                result["state"] = "overheating"
                result["confidence"] = 0.75

        # Industrial production momentum
        ip = raw.get("industrial_production")
        if ip is not None and len(ip) >= 12:
            ip_yoy = ((float(ip.iloc[-1]) / float(ip.iloc[-12])) - 1) * 100
            result["details"]["industrial_production_yoy"] = round(ip_yoy, 2)

        # Leading index
        leading = raw.get("leading_index")
        if leading is not None and not leading.empty:
            result["details"]["leading_index"] = round(float(leading.iloc[-1]), 2)
            if len(leading) >= 6:
                leading_trend = float(leading.iloc[-1]) - float(leading.iloc[-6])
                result["details"]["leading_index_6m_change"] = round(leading_trend, 2)

        return result

    @classmethod
    def _analyze_liquidity(cls, raw: dict[str, pd.Series]) -> dict:
        """Classify liquidity conditions."""
        result = {"state": "unknown", "confidence": 0.0, "details": {}}

        m2 = raw.get("m2_money_supply")
        if m2 is not None and len(m2) >= 12:
            m2_current = float(m2.iloc[-1])
            m2_12m = float(m2.iloc[-12])
            m2_growth = ((m2_current / m2_12m) - 1) * 100
            result["details"]["m2_yoy_growth"] = round(m2_growth, 2)

            if m2_growth < -2:
                result["state"] = "contraction"
                result["confidence"] = 0.85
            elif m2_growth < 2:
                result["state"] = "tight"
                result["confidence"] = 0.75
            elif m2_growth < 6:
                result["state"] = "neutral"
                result["confidence"] = 0.70
            elif m2_growth < 10:
                result["state"] = "accommodative"
                result["confidence"] = 0.75
            else:
                result["state"] = "flood"
                result["confidence"] = 0.85

        # Credit spreads
        hy_spread = raw.get("credit_spread_hy")
        if hy_spread is not None and not hy_spread.empty:
            spread_val = float(hy_spread.iloc[-1])
            result["details"]["hy_credit_spread"] = round(spread_val, 2)
            result["details"]["credit_stress"] = spread_val > 5.0

        ig_spread = raw.get("credit_spread_ig")
        if ig_spread is not None and not ig_spread.empty:
            result["details"]["ig_credit_spread"] = round(float(ig_spread.iloc[-1]), 2)

        return result

    @classmethod
    def _analyze_labor(cls, raw: dict[str, pd.Series]) -> dict:
        """Analyze labor market conditions."""
        result = {"state": "unknown", "confidence": 0.0, "details": {}}

        unemployment = raw.get("unemployment")
        if unemployment is not None and not unemployment.empty:
            current_rate = float(unemployment.iloc[-1])
            result["details"]["unemployment_rate"] = round(current_rate, 1)

            if len(unemployment) >= 12:
                rate_12m_ago = float(unemployment.iloc[-12])
                trend = current_rate - rate_12m_ago
                result["details"]["unemployment_12m_change"] = round(trend, 1)

                if current_rate < 4.0 and trend <= 0:
                    result["state"] = "tight_labor_market"
                    result["confidence"] = 0.85
                elif current_rate < 5.0:
                    result["state"] = "healthy"
                    result["confidence"] = 0.80
                elif trend > 0.5:
                    result["state"] = "deteriorating"
                    result["confidence"] = 0.75
                else:
                    result["state"] = "elevated_unemployment"
                    result["confidence"] = 0.70

        # Initial claims
        claims = raw.get("initial_claims")
        if claims is not None and not claims.empty:
            result["details"]["initial_claims"] = int(float(claims.iloc[-1]))
            if len(claims) >= 4:
                avg_4w = float(claims.iloc[-4:].mean())
                result["details"]["initial_claims_4w_avg"] = int(avg_4w)

        return result

    @classmethod
    def _compute_yield_curve(cls, raw: dict[str, pd.Series]) -> dict:
        """Compute yield curve analysis."""
        result = {"inverted": False, "spreads": {}, "signal": "neutral"}

        t10y = raw.get("treasury_10y")
        t2y = raw.get("treasury_2y")
        t3m = raw.get("treasury_3m")

        if t10y is not None and t2y is not None and not t10y.empty and not t2y.empty:
            spread_10y2y = float(t10y.iloc[-1]) - float(t2y.iloc[-1])
            result["spreads"]["10y_2y"] = round(spread_10y2y, 3)
            result["inverted"] = spread_10y2y < 0

            # Trend
            if len(t10y) > 21 and len(t2y) > 21:
                spread_1m_ago = float(t10y.iloc[-21]) - float(t2y.iloc[-21])
                result["spreads"]["10y_2y_1m_ago"] = round(spread_1m_ago, 3)
                if spread_10y2y < 0 and spread_10y2y < spread_1m_ago:
                    result["signal"] = "deepening_inversion"
                elif spread_10y2y < 0:
                    result["signal"] = "inverted_but_improving"
                elif spread_10y2y > 0 and spread_1m_ago < 0:
                    result["signal"] = "un-inverting"
                else:
                    result["signal"] = "normal"

        if t10y is not None and t3m is not None and not t10y.empty and not t3m.empty:
            spread_10y3m = float(t10y.iloc[-1]) - float(t3m.iloc[-1])
            result["spreads"]["10y_3m"] = round(spread_10y3m, 3)

        return result

    @classmethod
    def clear_cache(cls) -> None:
        """Clear cached FRED data."""
        cls._cache.clear()
        cls._fred_instance = None
        logger.info("Macro data cache cleared")
