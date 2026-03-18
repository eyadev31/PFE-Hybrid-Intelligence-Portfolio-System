"""
Adversarial Scenario Data Factory
====================================
Generates realistic synthetic market data that mimics specific historical
crisis periods. Each scenario produces data in the exact format that
Agent 1's ML pipeline expects (benchmark prices, VIX, cross-asset closes,
macro snapshots), so the REAL HMM + RF + VolatilityClassifier models
process them — not static mocks.

Government-grade adversarial testing:
  If the ML models cannot distinguish a 2008 crash from a 2021 bull run,
  the system possesses fake intelligence.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class ScenarioDataFactory:
    """
    Generates point-in-time synthetic market data for adversarial testing.
    Each method returns a dict with keys:
      - benchmark_prices: pd.Series (daily SPY-like closes)
      - vix_series: pd.Series (daily VIX closes)
      - cross_asset_closes: pd.DataFrame (SPY, QQQ, TLT, GLD, BTCUSDT)
      - macro_snapshot: dict (FRED-style macro indicators)
      - mock_answers: list[dict] (investor behavioral answers)
      - scenario_name: str
    """

    PERIODS = 300  # ~1.2 years of trading days

    @classmethod
    def _make_dates(cls, periods=None):
        periods = periods or cls.PERIODS
        return pd.date_range(end=datetime.utcnow(), periods=periods, freq="B")

    # ═══════════════════════════════════════════════════════
    #  SCENARIO 1: 2008-STYLE CREDIT CRASH
    # ═══════════════════════════════════════════════════════
    @classmethod
    def scenario_2008_crash(cls) -> dict:
        """
        Simulates a prolonged bear market with:
        - SPY declining ~45% over 12 months
        - VIX spiking to 60-80
        - All assets falling together (high cross-correlation)
        - Severe macro deterioration
        """
        np.random.seed(2008)
        dates = cls._make_dates()
        n = len(dates)

        # SPY: Strong initial decline, brief dead-cat bounce, then capitulation
        drift = np.concatenate([
            np.full(60, -0.003),    # Slow bleed phase
            np.full(40, -0.006),    # Acceleration
            np.full(20, 0.002),     # Dead-cat bounce
            np.full(80, -0.008),    # Capitulation
            np.full(n - 200, -0.001),  # Slow grind lower
        ])
        noise = np.random.normal(0, 0.025, n)
        spy = 1500 * np.exp(np.cumsum(drift + noise))
        benchmark = pd.Series(spy, index=dates, name="SPY")

        # VIX: Spikes massively during crisis
        vix_base = np.concatenate([
            np.linspace(18, 30, 60),
            np.linspace(30, 55, 40),
            np.linspace(55, 40, 20),
            np.linspace(40, 80, 80),
            np.linspace(80, 60, n - 200),
        ])
        vix = vix_base + np.random.normal(0, 3, n)
        vix = np.maximum(15, vix)
        vix_series = pd.Series(vix, index=dates, name="VIX")

        # Cross-assets: Everything crashes together (high correlation)
        assets = {}
        for name, base, corr_factor in [
            ("SPY", spy, 1.0),
            ("QQQ", 50, 0.95),   # Tech crashes harder
            ("TLT", 90, -0.3),   # Bonds rally (flight to safety)
            ("GLD", 800, -0.2),  # Gold rallies
            ("BTCUSDT", 10000, 0.8),  # Crypto crashes with equities
        ]:
            if name == "SPY":
                assets[name] = benchmark
            else:
                correlated_drift = drift * corr_factor
                asset_noise = np.random.normal(0, 0.02, n)
                assets[name] = pd.Series(
                    base * np.exp(np.cumsum(correlated_drift + asset_noise)),
                    index=dates,
                )

        macro = {
            "macro_regime": "recession",
            "monetary_policy_state": "emergency_easing",
            "inflation_state": "deflationary",
            "growth_state": "contraction",
            "liquidity_state": "frozen",
            "composite_score": -0.85,
            "key_indicators": {
                "fed_funds_rate": 0.25,
                "treasury_10y": 2.10,
                "treasury_2y": 0.80,
                "unemployment": 10.0,
            },
            "yield_curve": {
                "inverted": False,
                "spreads": {"10y_2y": 1.30, "10y_3m": 2.00},
                "signal": "steepening_recession",
            },
            "risk_factors": [
                "Credit markets frozen — interbank lending collapsed",
                "Unemployment surging above 10%",
                "Housing market in freefall — systemic contagion risk",
                "Global coordinated recession",
            ],
            "confidence": 0.90,
        }

        answers = cls._conservative_stable_answers()

        return {
            "benchmark_prices": benchmark,
            "vix_series": vix_series,
            "cross_asset_closes": pd.DataFrame(assets),
            "macro_snapshot": macro,
            "mock_answers": answers,
            "scenario_name": "2008_credit_crash",
        }

    # ═══════════════════════════════════════════════════════
    #  SCENARIO 2: 2020 COVID SHOCK (V-SHAPE)
    # ═══════════════════════════════════════════════════════
    @classmethod
    def scenario_2020_covid(cls) -> dict:
        """
        Simulates the COVID crash:
        - Extremely fast -34% drawdown in 23 trading days
        - Followed by a violent V-shape recovery
        - VIX spikes to 82 then rapidly declines
        - Massive fiscal/monetary stimulus
        """
        np.random.seed(2020)
        dates = cls._make_dates()
        n = len(dates)

        # SPY: Stable → crash → V-recovery
        drift = np.concatenate([
            np.full(200, 0.0005),    # Pre-COVID stability
            np.full(23, -0.018),     # COVID crash (23 days, -34%)
            np.full(77, 0.006),      # Aggressive recovery
        ])
        noise = np.random.normal(0, 0.015, n)
        spy = 3300 * np.exp(np.cumsum(drift + noise))
        benchmark = pd.Series(spy, index=dates, name="SPY")

        # VIX: Flash spike then decline
        vix_parts = np.concatenate([
            np.full(200, 14) + np.random.normal(0, 2, 200),
            np.linspace(14, 82, 23),
            np.linspace(82, 22, 77),
        ])
        vix = np.maximum(10, vix_parts + np.random.normal(0, 2, n))
        vix_series = pd.Series(vix, index=dates, name="VIX")

        assets = {}
        assets["SPY"] = benchmark
        for name, base, crash_mult, recovery_mult in [
            ("QQQ", 250, 0.85, 1.2),
            ("TLT", 140, -0.4, -0.2),
            ("GLD", 1550, -0.3, 0.5),
            ("BTCUSDT", 9000, 1.1, 1.5),
        ]:
            d = np.concatenate([
                np.full(200, 0.0004),
                np.full(23, -0.018 * crash_mult),
                np.full(77, 0.006 * recovery_mult),
            ])
            assets[name] = pd.Series(
                base * np.exp(np.cumsum(d + np.random.normal(0, 0.018, n))),
                index=dates,
            )

        macro = {
            "macro_regime": "crisis_recovery",
            "monetary_policy_state": "emergency_easing",
            "inflation_state": "below_target",
            "growth_state": "contraction",
            "liquidity_state": "flooded",
            "composite_score": -0.40,
            "key_indicators": {
                "fed_funds_rate": 0.0,
                "treasury_10y": 0.70,
                "treasury_2y": 0.15,
                "unemployment": 14.7,
            },
            "yield_curve": {
                "inverted": False,
                "spreads": {"10y_2y": 0.55, "10y_3m": 0.60},
                "signal": "steep_stimulus",
            },
            "risk_factors": [
                "Global pandemic lockdowns — GDP contracting at 30%+ annualized",
                "Unemployment at 14.7% — worst since Great Depression",
                "Unprecedented fiscal stimulus ($2.2T CARES Act)",
            ],
            "confidence": 0.70,
        }

        answers = cls._moderate_balanced_answers()

        return {
            "benchmark_prices": benchmark,
            "vix_series": vix_series,
            "cross_asset_closes": pd.DataFrame(assets),
            "macro_snapshot": macro,
            "mock_answers": answers,
            "scenario_name": "2020_covid_shock",
        }

    # ═══════════════════════════════════════════════════════
    #  SCENARIO 3: 2022 HIGH INFLATION TIGHTENING CYCLE
    # ═══════════════════════════════════════════════════════
    @classmethod
    def scenario_2022_inflation(cls) -> dict:
        """
        Simulates the 2022 inflation regime:
        - Equities AND bonds fall together (correlation breaks)
        - Slow grind lower, no dramatic crash
        - VIX elevated but not extreme (25-35)
        - Cash is king
        """
        np.random.seed(2022)
        dates = cls._make_dates()
        n = len(dates)

        # SPY: Slow sustained decline (-25%)
        drift = np.full(n, -0.001) + np.random.normal(0, 0.013, n)
        spy = 4800 * np.exp(np.cumsum(drift))
        benchmark = pd.Series(spy, index=dates, name="SPY")

        # VIX: Elevated but not panic
        vix = np.linspace(18, 32, n) + np.random.normal(0, 3, n)
        vix = np.clip(vix, 15, 40)
        vix_series = pd.Series(vix, index=dates, name="VIX")

        # KEY: Bonds ALSO fall (inflation destroys both asset classes)
        assets = {"SPY": benchmark}
        assets["QQQ"] = pd.Series(
            380 * np.exp(np.cumsum(np.full(n, -0.0015) + np.random.normal(0, 0.016, n))),
            index=dates,
        )
        # TLT DOWN — this is THE defining feature of 2022
        assets["TLT"] = pd.Series(
            150 * np.exp(np.cumsum(np.full(n, -0.0012) + np.random.normal(0, 0.008, n))),
            index=dates,
        )
        assets["GLD"] = pd.Series(
            1900 * np.exp(np.cumsum(np.full(n, 0.0001) + np.random.normal(0, 0.01, n))),
            index=dates,
        )
        assets["BTCUSDT"] = pd.Series(
            47000 * np.exp(np.cumsum(np.full(n, -0.003) + np.random.normal(0, 0.035, n))),
            index=dates,
        )

        macro = {
            "macro_regime": "stagflation_risk",
            "monetary_policy_state": "aggressive_tightening",
            "inflation_state": "runaway",
            "growth_state": "slowing",
            "liquidity_state": "tightening",
            "composite_score": -0.55,
            "key_indicators": {
                "fed_funds_rate": 4.75,
                "treasury_10y": 4.25,
                "treasury_2y": 4.70,
                "unemployment": 3.5,
            },
            "yield_curve": {
                "inverted": True,
                "spreads": {"10y_2y": -0.45, "10y_3m": -0.60},
                "signal": "deeply_inverted",
            },
            "risk_factors": [
                "CPI at 9.1% — 40-year high",
                "Fed hiking at fastest pace since Volcker era",
                "Bonds and equities falling in lockstep — 60/40 broken",
                "Yield curve deeply inverted — recession signal",
            ],
            "confidence": 0.80,
        }

        answers = cls._moderate_balanced_answers()

        return {
            "benchmark_prices": benchmark,
            "vix_series": vix_series,
            "cross_asset_closes": pd.DataFrame(assets),
            "macro_snapshot": macro,
            "mock_answers": answers,
            "scenario_name": "2022_inflation_tightening",
        }

    # ═══════════════════════════════════════════════════════
    #  SCENARIO 4: BULL MARKET EUPHORIA (2021 STYLE)
    # ═══════════════════════════════════════════════════════
    @classmethod
    def scenario_bull_euphoria(cls) -> dict:
        """
        Simulates a frothy bull market:
        - SPY +40% sustained rally
        - VIX crushed below 15
        - Crypto exploding (+200%)
        - Everything up, everything correlated to the upside
        """
        np.random.seed(2021)
        dates = cls._make_dates()
        n = len(dates)

        drift = np.full(n, 0.0015) + np.random.normal(0, 0.008, n)
        spy = 3700 * np.exp(np.cumsum(drift))
        benchmark = pd.Series(spy, index=dates, name="SPY")

        vix = np.linspace(22, 12, n) + np.random.normal(0, 1.5, n)
        vix = np.clip(vix, 9, 20)
        vix_series = pd.Series(vix, index=dates, name="VIX")

        assets = {"SPY": benchmark}
        assets["QQQ"] = pd.Series(
            310 * np.exp(np.cumsum(np.full(n, 0.002) + np.random.normal(0, 0.01, n))),
            index=dates,
        )
        assets["TLT"] = pd.Series(
            148 * np.exp(np.cumsum(np.full(n, -0.0003) + np.random.normal(0, 0.005, n))),
            index=dates,
        )
        assets["GLD"] = pd.Series(
            1750 * np.exp(np.cumsum(np.full(n, 0.0002) + np.random.normal(0, 0.008, n))),
            index=dates,
        )
        # Crypto moon
        assets["BTCUSDT"] = pd.Series(
            29000 * np.exp(np.cumsum(np.full(n, 0.004) + np.random.normal(0, 0.03, n))),
            index=dates,
        )

        macro = {
            "macro_regime": "expansion",
            "monetary_policy_state": "accommodative",
            "inflation_state": "above_target",
            "growth_state": "strong_growth",
            "liquidity_state": "abundant",
            "composite_score": 0.65,
            "key_indicators": {
                "fed_funds_rate": 0.0,
                "treasury_10y": 1.50,
                "treasury_2y": 0.25,
                "unemployment": 4.2,
            },
            "yield_curve": {
                "inverted": False,
                "spreads": {"10y_2y": 1.25, "10y_3m": 1.40},
                "signal": "healthy_steep",
            },
            "risk_factors": [
                "Euphoric sentiment — margin debt at all-time highs",
                "SPACs and meme stocks signal speculative excess",
            ],
            "confidence": 0.85,
        }

        answers = cls._aggressive_speculator_answers()

        return {
            "benchmark_prices": benchmark,
            "vix_series": vix_series,
            "cross_asset_closes": pd.DataFrame(assets),
            "macro_snapshot": macro,
            "mock_answers": answers,
            "scenario_name": "bull_market_euphoria",
        }

    # ═══════════════════════════════════════════════════════
    #  SCENARIO 5: USER OVERCONFIDENCE (LIAR DETECTION)
    # ═══════════════════════════════════════════════════════
    @classmethod
    def scenario_user_overconfidence(cls) -> dict:
        """
        Normal market conditions, BUT the user lies:
        - Claims to be aggressive
        - Behavioral answers reveal panic tendencies
        - Agent 2 must detect contradictions
        """
        np.random.seed(55)
        dates = cls._make_dates()
        n = len(dates)

        drift = np.full(n, 0.0003) + np.random.normal(0, 0.012, n)
        spy = 4400 * np.exp(np.cumsum(drift))
        benchmark = pd.Series(spy, index=dates, name="SPY")

        vix = 18 + np.random.normal(0, 3, n)
        vix = np.clip(vix, 10, 30)
        vix_series = pd.Series(vix, index=dates, name="VIX")

        assets = {"SPY": benchmark}
        for name, base in [("QQQ", 360), ("TLT", 100), ("GLD", 1800), ("BTCUSDT", 35000)]:
            assets[name] = pd.Series(
                base * np.exp(np.cumsum(np.random.normal(0.0002, 0.012, n))),
                index=dates,
            )

        macro = {
            "macro_regime": "stable_growth",
            "monetary_policy_state": "neutral",
            "inflation_state": "at_target",
            "growth_state": "moderate_growth",
            "liquidity_state": "neutral",
            "composite_score": 0.20,
            "key_indicators": {
                "fed_funds_rate": 3.50,
                "treasury_10y": 3.80,
                "treasury_2y": 3.60,
                "unemployment": 4.0,
            },
            "yield_curve": {
                "inverted": False,
                "spreads": {"10y_2y": 0.20, "10y_3m": 0.10},
                "signal": "flat",
            },
            "risk_factors": [],
            "confidence": 0.80,
        }

        # User LIES: says aggressive but answers show panic
        answers = cls._overconfident_liar_answers()

        return {
            "benchmark_prices": benchmark,
            "vix_series": vix_series,
            "cross_asset_closes": pd.DataFrame(assets),
            "macro_snapshot": macro,
            "mock_answers": answers,
            "scenario_name": "user_overconfidence",
        }

    # ═══════════════════════════════════════════════════════
    #  SCENARIO 6: MISSING DATA / SYSTEM DEGRADATION
    # ═══════════════════════════════════════════════════════
    @classmethod
    def scenario_missing_data(cls) -> dict:
        """
        Simulates API failures:
        - Only crypto data available (TwelveData down)
        - No VIX data
        - Empty macro snapshot (FRED unavailable)
        - System must degrade gracefully
        """
        np.random.seed(666)
        dates = cls._make_dates()
        n = len(dates)

        # Only BTCUSDT available as benchmark — high volatility
        btc_drift = np.random.normal(0.001, 0.035, n)
        btc = 30000 * np.exp(np.cumsum(btc_drift))
        benchmark = pd.Series(btc, index=dates, name="BTCUSDT")

        # NO VIX — None
        vix_series = None

        # Only crypto — no equities, no bonds, no gold
        assets = {
            "BTCUSDT": benchmark,
        }

        # Empty macro — FRED is down
        macro = {}

        answers = cls._conservative_anxious_answers()

        return {
            "benchmark_prices": benchmark,
            "vix_series": vix_series,
            "cross_asset_closes": pd.DataFrame(assets),
            "macro_snapshot": macro,
            "mock_answers": answers,
            "scenario_name": "missing_data_degraded",
        }

    # ═══════════════════════════════════════════════════════
    #  MOCK INVESTOR ANSWER PROFILES
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _conservative_stable_answers() -> list:
        return [
            {"question_id": "q1", "answer": "I would sell everything immediately", "response_time_seconds": 2.1},
            {"question_id": "q2", "answer": "Preserving capital is my absolute priority", "response_time_seconds": 3.5},
            {"question_id": "q3", "answer": "I cannot tolerate any losses beyond 5%", "response_time_seconds": 2.8},
            {"question_id": "q4", "answer": "I prefer guaranteed returns even if very low", "response_time_seconds": 4.0},
        ]

    @staticmethod
    def _moderate_balanced_answers() -> list:
        return [
            {"question_id": "q1", "answer": "I would rebalance to safer assets but not panic sell", "response_time_seconds": 5.2},
            {"question_id": "q2", "answer": "I accept moderate volatility for better long-term returns", "response_time_seconds": 4.0},
            {"question_id": "q3", "answer": "I can handle up to 15% drawdown before getting concerned", "response_time_seconds": 3.8},
            {"question_id": "q4", "answer": "A balanced mix of growth and safety suits me", "response_time_seconds": 6.1},
        ]

    @staticmethod
    def _aggressive_speculator_answers() -> list:
        return [
            {"question_id": "q1", "answer": "I would buy more during a crash — it's an opportunity", "response_time_seconds": 1.5},
            {"question_id": "q2", "answer": "Maximum growth is my only goal, I don't care about short-term losses", "response_time_seconds": 1.8},
            {"question_id": "q3", "answer": "I can handle 40%+ drawdowns without selling", "response_time_seconds": 1.2},
            {"question_id": "q4", "answer": "Put everything in crypto and tech — I want moonshots", "response_time_seconds": 0.9},
        ]

    @staticmethod
    def _conservative_anxious_answers() -> list:
        return [
            {"question_id": "q1", "answer": "I lose sleep when markets drop even 2%", "response_time_seconds": 8.5},
            {"question_id": "q2", "answer": "Safety is all that matters to me", "response_time_seconds": 7.2},
            {"question_id": "q3", "answer": "Any loss is unacceptable", "response_time_seconds": 9.0},
            {"question_id": "q4", "answer": "I want my money in a savings account, nothing else", "response_time_seconds": 6.5},
        ]

    @staticmethod
    def _overconfident_liar_answers() -> list:
        """
        User CLAIMS to be aggressive but their answers reveal deep anxiety.
        Key contradiction: fast response times (overconfidence) but
        answers showing panic behavior and loss aversion.
        """
        return [
            # Says aggressive but then reveals panic
            {"question_id": "q1", "answer": "I'm extremely aggressive, put it all in crypto", "response_time_seconds": 0.8},
            # Contradicts: would actually panic sell
            {"question_id": "q2", "answer": "If my portfolio drops 10% I would sell everything immediately", "response_time_seconds": 1.0},
            # Contradicts: cannot handle drawdown despite 'aggressive' claim
            {"question_id": "q3", "answer": "I absolutely cannot handle seeing red in my portfolio", "response_time_seconds": 0.7},
            # Final contradiction: wants safety guarantee alongside aggression
            {"question_id": "q4", "answer": "I want high returns but guarantee I won't lose money", "response_time_seconds": 0.5},
        ]
