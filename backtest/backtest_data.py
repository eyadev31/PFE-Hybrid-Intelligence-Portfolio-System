"""
Rolling Backtest Timeline Generator
=======================================
Generates 10 years of realistic multi-regime synthetic market data
for rolling historical backtesting.

Each month produces the same data format as ScenarioDataFactory:
  - benchmark_prices (pd.Series)
  - vix_series (pd.Series)
  - cross_asset_closes (pd.DataFrame)
  - macro_snapshot (dict)
  - forward_returns (dict) — actual next-month returns for scoring

The timeline cycles through historically accurate regime sequences:
  2014-2016: Bull Low Vol (post-crisis recovery)
  2016-2017: Sideways / Transition
  2017-2019: Bull High Vol (trade war)
  2020 Q1:   Crisis Crash (COVID)
  2020 Q2-2021: Bull Euphoria (stimulus)
  2022:      Bear + Inflation
  2023-2024: Recovery
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
#  REGIME DEFINITIONS
# ══════════════════════════════════════════════════════════

REGIME_TIMELINE = [
    # (label, months, spy_drift, spy_vol, vix_mean, macro_composite, description)
    ("bull_low_vol",    24, 0.0008,  0.008, 14, 0.50,  "2014-2016 Post-crisis recovery"),
    ("sideways",        12, 0.0001,  0.011, 18, 0.20,  "2016-2017 Election uncertainty"),
    ("bull_high_vol",   24, 0.0006,  0.015, 22, 0.35,  "2017-2019 Tax cuts + trade war"),
    ("bear_high_vol",    3, -0.015,  0.040, 70, -0.80, "2020 Q1 COVID crash"),
    ("bull_high_vol",   18, 0.0020,  0.018, 20, 0.30,  "2020-2021 Stimulus euphoria"),
    ("bear_low_vol",    12, -0.0010, 0.014, 28, -0.55, "2022 Inflation tightening"),
    ("bull_low_vol",    24, 0.0006,  0.010, 16, 0.40,  "2023-2024 AI recovery"),
]

# Macro snapshots per regime
MACRO_TEMPLATES = {
    "bull_low_vol": {
        "macro_regime": "expansion",
        "monetary_policy_state": "accommodative",
        "inflation_state": "at_target",
        "growth_state": "moderate_growth",
        "liquidity_state": "abundant",
        "key_indicators": {"fed_funds_rate": 1.50, "treasury_10y": 2.50,
                           "treasury_2y": 1.80, "unemployment": 4.5},
        "yield_curve": {"inverted": False, "spreads": {"10y_2y": 0.70, "10y_3m": 0.90},
                        "signal": "healthy"},
        "risk_factors": [],
        "confidence": 0.80,
    },
    "sideways": {
        "macro_regime": "stable_growth",
        "monetary_policy_state": "neutral",
        "inflation_state": "below_target",
        "growth_state": "moderate_growth",
        "liquidity_state": "neutral",
        "key_indicators": {"fed_funds_rate": 0.75, "treasury_10y": 2.30,
                           "treasury_2y": 1.20, "unemployment": 4.7},
        "yield_curve": {"inverted": False, "spreads": {"10y_2y": 1.10, "10y_3m": 1.30},
                        "signal": "flat"},
        "risk_factors": ["Political uncertainty"],
        "confidence": 0.70,
    },
    "bull_high_vol": {
        "macro_regime": "expansion",
        "monetary_policy_state": "tightening",
        "inflation_state": "above_target",
        "growth_state": "strong_growth",
        "liquidity_state": "neutral",
        "key_indicators": {"fed_funds_rate": 2.50, "treasury_10y": 3.00,
                           "treasury_2y": 2.80, "unemployment": 3.7},
        "yield_curve": {"inverted": False, "spreads": {"10y_2y": 0.20, "10y_3m": 0.10},
                        "signal": "flattening"},
        "risk_factors": ["Trade war escalation", "Rate hike cycle"],
        "confidence": 0.75,
    },
    "bear_high_vol": {
        "macro_regime": "recession",
        "monetary_policy_state": "emergency_easing",
        "inflation_state": "deflationary",
        "growth_state": "contraction",
        "liquidity_state": "frozen",
        "key_indicators": {"fed_funds_rate": 0.0, "treasury_10y": 0.70,
                           "treasury_2y": 0.15, "unemployment": 14.0},
        "yield_curve": {"inverted": False, "spreads": {"10y_2y": 0.55, "10y_3m": 0.60},
                        "signal": "steep_crisis"},
        "risk_factors": ["Systemic liquidity crisis", "Unemployment surge",
                         "GDP contracting at 30%+ annualized"],
        "confidence": 0.65,
    },
    "bear_low_vol": {
        "macro_regime": "stagflation_risk",
        "monetary_policy_state": "aggressive_tightening",
        "inflation_state": "runaway",
        "growth_state": "slowing",
        "liquidity_state": "tightening",
        "key_indicators": {"fed_funds_rate": 4.75, "treasury_10y": 4.25,
                           "treasury_2y": 4.70, "unemployment": 3.5},
        "yield_curve": {"inverted": True, "spreads": {"10y_2y": -0.45, "10y_3m": -0.60},
                        "signal": "deeply_inverted"},
        "risk_factors": ["CPI at 9%", "Fastest rate hikes since Volcker",
                         "Bonds and equities correlated downside"],
        "confidence": 0.80,
    },
}

# Cross-asset behavior per regime
ASSET_PARAMS = {
    # regime: {asset: (drift_mult, vol_mult)} relative to SPY
    "bull_low_vol": {
        "QQQ": (1.3, 1.2), "TLT": (-0.2, 0.4), "GLD": (0.1, 0.6), "BTCUSDT": (2.0, 3.0),
    },
    "sideways": {
        "QQQ": (0.8, 1.1), "TLT": (0.5, 0.5), "GLD": (0.3, 0.7), "BTCUSDT": (1.5, 2.5),
    },
    "bull_high_vol": {
        "QQQ": (1.5, 1.3), "TLT": (-0.3, 0.5), "GLD": (0.2, 0.8), "BTCUSDT": (3.0, 3.5),
    },
    "bear_high_vol": {
        "QQQ": (1.2, 1.1), "TLT": (-0.5, 0.6), "GLD": (-0.3, 0.8), "BTCUSDT": (0.9, 2.0),
    },
    "bear_low_vol": {
        "QQQ": (1.4, 1.2), "TLT": (0.8, 0.7), "GLD": (0.1, 0.8), "BTCUSDT": (2.5, 3.0),
    },
}


class TimelineGenerator:
    """
    Generates a 10-year month-by-month synthetic market timeline.
    Each window contains 252 trailing days of data (1 year lookback)
    plus forward returns for scoring.
    """

    LOOKBACK_DAYS = 252  # 1 year of trailing data per window
    TRADING_DAYS_PER_MONTH = 21

    @classmethod
    def generate_full_timeline(cls, seed: int = 42) -> list[dict]:
        """
        Generate the complete 10-year timeline as a list of monthly windows.

        Returns:
            List of dicts, each containing:
              - benchmark_prices: pd.Series (252 trailing days)
              - vix_series: pd.Series
              - cross_asset_closes: pd.DataFrame
              - macro_snapshot: dict
              - forward_returns: dict (next month actual returns per asset)
              - month_index: int
              - regime_label: str
              - period_description: str
        """
        np.random.seed(seed)

        # Step 1: Generate the full continuous price history
        total_months = sum(r[1] for r in REGIME_TIMELINE)
        total_days = total_months * cls.TRADING_DAYS_PER_MONTH
        buffer_days = cls.LOOKBACK_DAYS  # Extra days for lookback

        full_length = total_days + buffer_days
        dates = pd.date_range(end=datetime(2024, 12, 31), periods=full_length, freq="B")

        # Generate SPY continuous path
        spy_daily = np.zeros(full_length)
        vix_daily = np.zeros(full_length)
        regime_labels = [""] * full_length

        day_idx = buffer_days  # Start after the lookback buffer
        # Fill buffer with stable bull data
        for d in range(buffer_days):
            spy_daily[d] = np.random.normal(0.0004, 0.010)
            vix_daily[d] = 15 + np.random.normal(0, 2)
            regime_labels[d] = "bull_low_vol"

        for regime_label, months, spy_drift, spy_vol, vix_mean, macro_comp, desc in REGIME_TIMELINE:
            days_in_regime = months * cls.TRADING_DAYS_PER_MONTH
            end_idx = min(day_idx + days_in_regime, full_length)

            for d in range(day_idx, end_idx):
                spy_daily[d] = np.random.normal(spy_drift, spy_vol)
                vix_daily[d] = vix_mean + np.random.normal(0, vix_mean * 0.15)
                regime_labels[d] = regime_label

            day_idx = end_idx

        # Convert to price levels
        spy_prices = 1800 * np.exp(np.cumsum(spy_daily))
        vix_prices = np.clip(vix_daily, 9, 90)

        # Generate cross-asset prices
        asset_prices = {"SPY": spy_prices}
        base_prices = {"QQQ": 100, "TLT": 120, "GLD": 1200, "BTCUSDT": 500}

        for asset, base in base_prices.items():
            asset_daily = np.zeros(full_length)
            for d in range(full_length):
                regime = regime_labels[d]
                params = ASSET_PARAMS.get(regime, ASSET_PARAMS["bull_low_vol"])
                drift_mult, vol_mult = params.get(asset, (1.0, 1.0))
                asset_daily[d] = np.random.normal(
                    spy_daily[d] * drift_mult,
                    abs(spy_daily[d]) * vol_mult + 0.005
                )
            asset_prices[asset] = base * np.exp(np.cumsum(asset_daily))

        # Step 2: Slice into monthly windows
        windows = []
        month_idx = 0
        regime_day_cursor = buffer_days

        for regime_label, months, spy_drift, spy_vol, vix_mean, macro_comp, desc in REGIME_TIMELINE:
            macro = MACRO_TEMPLATES.get(regime_label, MACRO_TEMPLATES["bull_low_vol"]).copy()
            macro["composite_score"] = macro_comp

            for m in range(months):
                window_end = regime_day_cursor + cls.TRADING_DAYS_PER_MONTH
                if window_end >= full_length:
                    break

                # Lookback window (252 days ending at window_end)
                lb_start = max(0, window_end - cls.LOOKBACK_DAYS)
                lb_end = window_end

                window_dates = dates[lb_start:lb_end]
                benchmark = pd.Series(
                    spy_prices[lb_start:lb_end],
                    index=window_dates, name="SPY"
                )
                vix_series = pd.Series(
                    vix_prices[lb_start:lb_end],
                    index=window_dates, name="VIX"
                )

                cross_assets = {}
                for asset_name, prices_arr in asset_prices.items():
                    cross_assets[asset_name] = pd.Series(
                        prices_arr[lb_start:lb_end],
                        index=window_dates,
                    )
                cross_df = pd.DataFrame(cross_assets)

                # Forward returns (next month) for performance scoring
                fwd_start = window_end
                fwd_end = min(window_end + cls.TRADING_DAYS_PER_MONTH, full_length)
                forward_returns = {}
                if fwd_end > fwd_start:
                    for asset_name, prices_arr in asset_prices.items():
                        start_price = prices_arr[fwd_start - 1]
                        end_price = prices_arr[fwd_end - 1] if fwd_end <= full_length else prices_arr[-1]
                        forward_returns[asset_name] = (end_price / start_price) - 1.0

                # Map asset returns to portfolio tickers
                forward_returns_mapped = {
                    "SPY": forward_returns.get("SPY", 0.0),
                    "BND": forward_returns.get("TLT", 0.0) * 0.5,  # Bonds less volatile
                    "GLD": forward_returns.get("GLD", 0.0),
                    "BTC": forward_returns.get("BTCUSDT", 0.0),
                    "CASH": 0.003 / 12,  # Risk-free monthly
                }

                windows.append({
                    "benchmark_prices": benchmark,
                    "vix_series": vix_series,
                    "cross_asset_closes": cross_df,
                    "macro_snapshot": macro,
                    "forward_returns": forward_returns_mapped,
                    "month_index": month_idx,
                    "regime_label": regime_label,
                    "period_description": desc,
                    "scenario_name": f"backtest_month_{month_idx:03d}_{regime_label}",
                })

                regime_day_cursor = window_end
                month_idx += 1

        logger.info(f"Generated {len(windows)} monthly windows across {len(REGIME_TIMELINE)} regimes")
        return windows
