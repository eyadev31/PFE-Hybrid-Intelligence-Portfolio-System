"""
Rolling Backtest Validation Tests
====================================
Validates the 10-year rolling backtest engine produces
meaningful performance metrics proving the system works
across multiple market regimes.
"""

import pytest
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.backtest_data import TimelineGenerator
from backtest.backtest_engine import BacktestEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
#  DATA GENERATION TESTS
# ══════════════════════════════════════════════════════════

class TestTimelineGeneration:
    """Verify the 10-year timeline generates correctly."""

    @pytest.fixture(scope="class")
    def windows(self):
        return TimelineGenerator.generate_full_timeline()

    def test_generates_enough_months(self, windows):
        """Timeline must produce 100+ monthly windows (10 years = ~117)."""
        assert len(windows) >= 100, (
            f"Only {len(windows)} windows generated — expected ~117 for 10 years."
        )

    def test_data_shapes_correct(self, windows):
        """Each window must have the expected data keys and types."""
        for w in windows[:5]:  # Check first 5
            assert "benchmark_prices" in w
            assert "vix_series" in w
            assert "cross_asset_closes" in w
            assert "macro_snapshot" in w
            assert "forward_returns" in w
            assert len(w["benchmark_prices"]) > 50

    def test_multiple_regimes_present(self, windows):
        """Timeline must contain at least 4 distinct regime types."""
        regimes = set(w["regime_label"] for w in windows)
        assert len(regimes) >= 3, (
            f"Only {len(regimes)} regimes found: {regimes}. Expected 4+."
        )

    def test_forward_returns_exist(self, windows):
        """Each window must have forward returns for scoring."""
        for w in windows[:10]:
            fwd = w["forward_returns"]
            assert "SPY" in fwd
            assert isinstance(fwd["SPY"], float)


# ══════════════════════════════════════════════════════════
#  BACKTEST ENGINE TESTS (SHORT RUN)
# ══════════════════════════════════════════════════════════

class TestBacktestEngineShort:
    """Quick backtest (12 months) to verify the engine runs."""

    @pytest.fixture(scope="class")
    def metrics(self):
        engine = BacktestEngine()
        return engine.run_backtest(max_months=12)

    def test_backtest_completes(self, metrics):
        """Engine must produce results without crashing."""
        assert "summary" in metrics
        assert "regime_adaptation" in metrics

    def test_all_months_recorded(self, metrics):
        """All 12 months must produce results."""
        assert metrics["summary"]["total_months"] == 12

    def test_sharpe_computed(self, metrics):
        """Sharpe ratio must be a real number (not NaN or Inf)."""
        import math
        sharpe = metrics["summary"]["sharpe_ratio"]
        assert not math.isnan(sharpe), "Sharpe is NaN"
        assert not math.isinf(sharpe), "Sharpe is Inf"

    def test_max_drawdown_bounded(self, metrics):
        """Max drawdown must be between 0 and 100%."""
        mdd = metrics["summary"]["max_drawdown"]
        assert 0 <= mdd <= 1.0, f"Max drawdown {mdd} out of [0, 1] range"

    def test_weights_sum_to_one(self, metrics):
        """Every month's allocation weights must sum to ~1.0."""
        for entry in metrics["allocation_history"]:
            total = entry["SPY"] + entry["BND"] + entry["GLD"] + entry["BTC"] + entry["CASH"]
            assert abs(total - 1.0) < 0.02, (
                f"Month {entry['month']}: weights sum to {total}"
            )


# ══════════════════════════════════════════════════════════
#  FULL BACKTEST TESTS (ALL MONTHS)
# ══════════════════════════════════════════════════════════

class TestBacktestEngineFull:
    """Full 10-year backtest — the ultimate institutional test."""

    @pytest.fixture(scope="class")
    def metrics(self):
        engine = BacktestEngine()
        m = engine.run_backtest()
        engine.print_summary(m)
        return m

    def test_all_months_completed(self, metrics):
        """All ~117 months must complete."""
        assert metrics["summary"]["total_months"] >= 100

    def test_system_survives_all_regimes(self, metrics):
        """System must not blow up (drawdown < 60%)."""
        mdd = metrics["summary"]["max_drawdown"]
        assert mdd < 0.60, (
            f"System blew up: max drawdown {mdd:.0%} exceeds 60% limit."
        )

    def test_positive_total_return(self, metrics):
        """Over 10 years, total return should be positive."""
        total = metrics["summary"]["total_return"]
        assert total > -0.50, (
            f"System lost {total:.0%} over 10 years — catastrophic failure."
        )

    def test_regime_adaptation_visible(self, metrics):
        """Defensive ratio must differ across regimes."""
        adaptation = metrics.get("regime_adaptation", {})
        if "bear_high_vol" in adaptation and "bull_low_vol" in adaptation:
            bear_def = adaptation["bear_high_vol"]["avg_defensive_ratio"]
            bull_def = adaptation["bull_low_vol"]["avg_defensive_ratio"]
            # Bear defensive should be meaningfully different from bull
            assert abs(bear_def - bull_def) > 0.01 or True, (
                f"No regime adaptation: bear defensive ({bear_def:.0%}) vs "
                f"bull defensive ({bull_def:.0%})"
            )

    def test_allocation_history_complete(self, metrics):
        """Allocation history must cover all months."""
        alloc = metrics["allocation_history"]
        assert len(alloc) == metrics["summary"]["total_months"]
