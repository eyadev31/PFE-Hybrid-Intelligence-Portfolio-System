"""
Rolling Historical Backtest Engine
=======================================
Replays the full 4-agent pipeline month-by-month across a 10-year
synthetic market timeline. Records every allocation decision, computes
forward returns using next-month data, and produces institutional-grade
performance metrics.

Government-grade requirement:
  "If you had trusted this system since 2014, what would have happened?"
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Optional

from backtest.backtest_data import TimelineGenerator
from agents.agent1_macro import Agent1MacroIntelligence
from agents.agent2_daq import Agent2BehavioralIntelligence
from agents.agent3_strategist import Agent3PortfolioStrategist
from agents.agent4_supervisor import Agent4RiskSupervisor

logger = logging.getLogger(__name__)


class BacktestResult:
    """Container for a single backtest month's result."""
    def __init__(self, month_index: int, regime: str, description: str,
                 weights: dict, forward_returns: dict, portfolio_return: float,
                 requires_rebalance: bool = True, turnover: float = 1.0):
        self.month_index = month_index
        self.regime = regime
        self.description = description
        self.weights = weights
        self.forward_returns = forward_returns
        self.portfolio_return = portfolio_return
        self.requires_rebalance = requires_rebalance
        self.turnover = turnover

    @property
    def defensive_ratio(self) -> float:
        """Fraction of portfolio in defensive assets (BND + CASH + GLD)."""
        return self.weights.get("BND", 0) + self.weights.get("CASH", 0) + self.weights.get("GLD", 0)


class BacktestEngine:
    """
    Rolling historical backtest engine.

    Replays the full Agent 1→2→3→4 pipeline at each monthly window,
    records the chosen allocation, then computes realized return using
    forward (next-month) market data.
    """

    def __init__(self):
        self.results: list[BacktestResult] = []
        self._agent1 = Agent1MacroIntelligence()
        self._agent2 = Agent2BehavioralIntelligence()

    def run_backtest(self, windows: list[dict] = None, max_months: int = None) -> dict:
        """
        Run the full rolling backtest.

        Args:
            windows: Pre-generated monthly windows (from TimelineGenerator).
                     If None, generates automatically.
            max_months: Limit months for faster testing. None = all.

        Returns:
            Performance metrics dict.
        """
        if windows is None:
            windows = TimelineGenerator.generate_full_timeline()

        if max_months:
            windows = windows[:max_months]

        total = len(windows)
        logger.info(f"Starting rolling backtest: {total} months")
        start_time = time.time()

        self.results.clear()
        current_weights = None

        for i, window in enumerate(windows):
            month_idx = window["month_index"]
            regime = window["regime_label"]
            desc = window["period_description"]

            logger.info(
                f"[Month {month_idx:3d}/{total}] {regime:15s} | {desc}"
            )

            try:
                # ── Agent 1: Real ML pipeline on this month's data ──
                agent1_output = self._agent1.run_scenario(window)

                # ── Agent 2: Fixed moderate investor profile ────────
                agent2_full = self._agent2.run_mock(
                    agent1_output=agent1_output,
                    bypass_llm=True
                )
                agent2_output = agent2_full.get("phase2_profile", agent2_full)

                # ── Agent 3: Real optimizer ─────────────────────────
                agent3 = Agent3PortfolioStrategist()
                agent3_output = agent3.run(
                    agent1_output=agent1_output,
                    agent2_output=agent2_output,
                    current_portfolio=current_weights,
                    bypass_llm=True
                )

                # ── Extract Evolution Metrics ───────────────────────
                evo = agent3_output.get("evolution_metrics", {})
                requires_rebalance = evo.get("requires_rebalance", True)
                turnover = evo.get("portfolio_turnover", 1.0)

                # ── Agent 4: Real CRO risk oversight ────────────────
                agent4 = Agent4RiskSupervisor()
                # Bypass LLM for Agent 4 during backtests to prevent massive API delays
                agent4._adjudicator._llm.is_available = lambda: False
                agent4_output = agent4.run(
                    agent1_output=agent1_output,
                    agent2_output=agent2_output,
                    agent3_output=agent3_output,
                )

                # ── Extract final weights ───────────────────────────
                weights = self._extract_weights(agent3_output, agent4_output)

                # ── Compute realized return ─────────────────────────
                fwd = window.get("forward_returns", {})
                portfolio_return = sum(
                    weights.get(ticker, 0) * fwd.get(ticker, 0)
                    for ticker in set(list(weights.keys()) + list(fwd.keys()))
                )

                self.results.append(BacktestResult(
                    month_index=month_idx,
                    regime=regime,
                    description=desc,
                    weights=weights,
                    forward_returns=fwd,
                    portfolio_return=portfolio_return,
                    requires_rebalance=requires_rebalance,
                    turnover=turnover,
                ))

                # ── Calculate Drifted Weights for Next Month Starts ──
                # As market moves during the month, the absolute value of each asset changes
                total_val = 0.0
                new_weights = {}
                for tk, w in weights.items():
                    ret = fwd.get(tk, 0.0)
                    val = w * (1.0 + ret)
                    new_weights[tk] = val
                    total_val += val
                
                if total_val > 0:
                    current_weights = {tk: v / total_val for tk, v in new_weights.items()}
                else:
                    current_weights = weights

            except Exception as e:
                logger.error(f"[Month {month_idx}] Pipeline failed: {e}")
                # Record zero return for failed months
                self.results.append(BacktestResult(
                    month_index=month_idx,
                    regime=regime,
                    description=desc,
                    weights={"CASH": 1.0},
                    forward_returns=window.get("forward_returns", {}),
                    portfolio_return=0.0,
                    requires_rebalance=True,
                    turnover=1.0,
                ))
                current_weights = None  # Reset on failure

        elapsed = time.time() - start_time
        logger.info(f"Backtest completed in {elapsed:.1f}s ({total} months)")

        return self.compute_metrics()

    def compute_metrics(self) -> dict:
        """
        Compute institutional-grade performance metrics from backtest results.

        Returns:
            Dict with annualized return, Sharpe, max drawdown, volatility,
            regime adaptation analysis.
        """
        if not self.results:
            return {"error": "No results to compute"}

        monthly_returns = np.array([r.portfolio_return for r in self.results])
        n_months = len(monthly_returns)

        # ── Cumulative Equity Curve ─────────────────────────
        cumulative = np.cumprod(1 + monthly_returns)

        # ── Annualized Return ───────────────────────────────
        total_return = cumulative[-1] - 1.0
        years = n_months / 12
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # ── Monthly Volatility → Annualized ─────────────────
        monthly_vol = np.std(monthly_returns)
        annualized_vol = monthly_vol * np.sqrt(12)

        # ── Sharpe Ratio ────────────────────────────────────
        monthly_rf = 0.04 / 12  # 4% annual risk-free
        excess_returns = monthly_returns - monthly_rf
        sharpe = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(12) if np.std(excess_returns) > 0 else 0

        # ── Max Drawdown ────────────────────────────────────
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        max_drawdown = np.max(drawdowns)

        # ── Sortino Ratio ───────────────────────────────────
        downside_returns = excess_returns[excess_returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(12) if len(downside_returns) > 0 else 1e-6
        sortino = (np.mean(excess_returns) * 12) / downside_vol

        # ── Calmar Ratio ────────────────────────────────────
        calmar = annualized_return / max_drawdown if max_drawdown > 0 else 0

        # ── Win Rate ────────────────────────────────────────
        win_rate = np.mean(monthly_returns > 0)

        # ── Evolution Statistics ────────────────────────────
        avg_turnover = np.mean([r.turnover for r in self.results])
        total_trades = sum(1 for r in self.results if r.requires_rebalance)
        trade_frequency = total_trades / n_months if n_months > 0 else 1.0

        # ── Regime Adaptation Analysis ──────────────────────
        regime_stats = self._compute_regime_stats()

        # ── Monthly Allocation Heatmap Data ─────────────────
        allocation_history = [
            {
                "month": r.month_index,
                "regime": r.regime,
                "SPY": r.weights.get("SPY", 0),
                "BND": r.weights.get("BND", 0),
                "GLD": r.weights.get("GLD", 0),
                "BTC": r.weights.get("BTC", 0),
                "CASH": r.weights.get("CASH", 0),
                "defensive_ratio": r.defensive_ratio,
                "return": r.portfolio_return,
            }
            for r in self.results
        ]

        return {
            "summary": {
                "total_months": n_months,
                "years": round(years, 1),
                "total_return": round(total_return, 4),
                "annualized_return": round(annualized_return, 4),
                "annualized_volatility": round(annualized_vol, 4),
                "sharpe_ratio": round(sharpe, 4),
                "sortino_ratio": round(sortino, 4),
                "calmar_ratio": round(calmar, 4),
                "max_drawdown": round(max_drawdown, 4),
                "win_rate": round(win_rate, 4),
                "best_month": round(float(np.max(monthly_returns)), 4),
                "worst_month": round(float(np.min(monthly_returns)), 4),
                "avg_monthly_turnover": round(float(avg_turnover), 4),
                "total_rebalances": total_trades,
                "rebalance_frequency": round(float(trade_frequency), 4),
            },
            "regime_adaptation": regime_stats,
            "allocation_history": allocation_history,
            "equity_curve": cumulative.tolist(),
        }

    def _compute_regime_stats(self) -> dict:
        """Compute per-regime performance and allocation analysis."""
        regime_groups = {}
        for r in self.results:
            if r.regime not in regime_groups:
                regime_groups[r.regime] = []
            regime_groups[r.regime].append(r)

        stats = {}
        for regime, results in regime_groups.items():
            returns = [r.portfolio_return for r in results]
            defensive_ratios = [r.defensive_ratio for r in results]

            stats[regime] = {
                "months": len(results),
                "avg_monthly_return": round(float(np.mean(returns)), 4),
                "volatility": round(float(np.std(returns)), 4),
                "avg_defensive_ratio": round(float(np.mean(defensive_ratios)), 4),
                "min_defensive_ratio": round(float(np.min(defensive_ratios)), 4),
                "max_defensive_ratio": round(float(np.max(defensive_ratios)), 4),
                "avg_turnover": round(float(np.mean([r.turnover for r in results])), 4),
                "worst_month": round(float(np.min(returns)), 4),
                "best_month": round(float(np.max(returns)), 4),
            }

        return stats

    @staticmethod
    def _extract_weights(agent3_output: dict, agent4_output: dict) -> dict:
        """Extract final portfolio weights (prefer Agent 4 adjusted if available)."""
        adjusted = agent4_output.get("adjusted_allocation", [])
        if adjusted:
            return {
                a["ticker"]: a.get("adjusted_weight", a.get("weight", 0))
                for a in adjusted
            }
        alloc = agent3_output.get("allocation", [])
        return {a["ticker"]: a["weight"] for a in alloc}

    def print_summary(self, metrics: dict) -> None:
        """Print a formatted backtest summary to the console."""
        s = metrics["summary"]
        print("\n" + "=" * 60)
        print("  ROLLING BACKTEST RESULTS")
        print("=" * 60)
        print(f"  Period:              {s['years']} years ({s['total_months']} months)")
        print(f"  Total Return:        {s['total_return']:+.2%}")
        print(f"  Annualized Return:   {s['annualized_return']:+.2%}")
        print(f"  Annualized Vol:      {s['annualized_volatility']:.2%}")
        print(f"  Sharpe Ratio:        {s['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio:       {s['sortino_ratio']:.2f}")
        print(f"  Calmar Ratio:        {s['calmar_ratio']:.2f}")
        print(f"  Max Drawdown:        {s['max_drawdown']:.2%}")
        print(f"  Win Rate:            {s['win_rate']:.0%}")
        print(f"  Best Month:          {s['best_month']:+.2%}")
        print(f"  Worst Month:         {s['worst_month']:+.2%}")
        print("-" * 60)
        print(f"  Rebalance Freq:      {s['rebalance_frequency']:.0%} ({s['total_rebalances']} trades in {s['total_months']} mo)")
        print(f"  Avg Turnover/Mo:     {s['avg_monthly_turnover']:.2%}")
        print("=" * 60)

        print("\n  REGIME ADAPTATION:")
        print("-" * 60)
        for regime, stats in metrics.get("regime_adaptation", {}).items():
            print(f"  {regime:20s} | {stats['months']:3d} mo | "
                  f"ret={stats['avg_monthly_return']:+.2%} | "
                  f"def={stats['avg_defensive_ratio']:.0%}")
        print("=" * 60 + "\n")
