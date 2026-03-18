"""
Hybrid Intelligence Portfolio System — Performance Analytics Dashboard
=========================================================================
Generates a premium standalone HTML analytics dashboard from backtest data.

Features:
  1. Equity Curve — Cumulative returns over 10 years
  2. Drawdown Underwater Chart — Visualize periods of loss
  3. Allocation Evolution — Stacked area chart of weight changes
  4. Regime Timeline — Color-coded market regime bands
  5. Risk-Adjusted KPI Cards — Sharpe, Sortino, Calmar, Max DD
  6. Allocation Stability — Monthly weight volatility per asset
  7. Confidence Trend — Regime detection confidence over time
  8. Monthly Returns Heatmap — Win/loss grid

Usage:
  from dashboard.dashboard_generator import DashboardGenerator
  gen = DashboardGenerator(backtest_metrics)
  gen.generate("dashboard_report.html")
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class DashboardGenerator:
    """
    Generates a self-contained HTML analytics dashboard.
    
    The dashboard uses Chart.js (loaded from CDN) for interactive
    charts and requires no additional server or dependencies.
    """

    def __init__(self, metrics: dict, backtest_results: list = None):
        """
        Args:
            metrics: Output from BacktestEngine.compute_metrics()
            backtest_results: List of BacktestResult objects (optional, for extra detail)
        """
        self._metrics = metrics
        self._results = backtest_results or []
        self._timestamp = datetime.utcnow().isoformat() + "Z"

    def generate(self, output_path: str = "analytics_dashboard.html") -> str:
        """
        Generate the full HTML dashboard and write to file.
        
        Returns:
            Absolute path to the generated HTML file.
        """
        logger.info(f"Generating Performance Analytics Dashboard...")

        html = self._build_html()

        abs_path = os.path.abspath(output_path)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"Dashboard saved to: {abs_path}")
        return abs_path

    # ════════════════════════════════════════════════════
    #  DATA PREPARATION
    # ════════════════════════════════════════════════════

    def _prepare_equity_curve(self) -> list:
        """Get cumulative equity curve data."""
        return self._metrics.get("equity_curve", [])

    def _prepare_drawdown_curve(self) -> list:
        """Compute drawdown series from equity curve."""
        eq = self._prepare_equity_curve()
        if not eq:
            return []
        arr = np.array(eq)
        running_max = np.maximum.accumulate(arr)
        drawdown = (running_max - arr) / running_max
        return [-round(float(d), 4) for d in drawdown]

    def _prepare_allocation_history(self) -> dict:
        """Prepare per-asset allocation time series."""
        history = self._metrics.get("allocation_history", [])
        tickers = ["SPY", "BND", "GLD", "BTC", "CASH"]
        result = {t: [] for t in tickers}
        months = []
        regimes = []
        returns = []
        defensive_ratios = []

        for entry in history:
            months.append(entry.get("month", 0))
            regimes.append(entry.get("regime", "unknown"))
            returns.append(round(entry.get("return", 0) * 100, 2))
            defensive_ratios.append(round(entry.get("defensive_ratio", 0) * 100, 1))
            for t in tickers:
                result[t].append(round(entry.get(t, 0) * 100, 1))

        return {
            "months": months,
            "regimes": regimes,
            "returns": returns,
            "defensive_ratios": defensive_ratios,
            **result,
        }

    def _prepare_regime_data(self) -> dict:
        """Prepare regime adaptation stats."""
        return self._metrics.get("regime_adaptation", {})

    def _compute_allocation_stability(self) -> dict:
        """Compute allocation stability (std dev of weights per asset)."""
        history = self._metrics.get("allocation_history", [])
        tickers = ["SPY", "BND", "GLD", "BTC", "CASH"]
        stability = {}
        for t in tickers:
            weights = [entry.get(t, 0) for entry in history]
            if weights:
                stability[t] = round(float(np.std(weights)) * 100, 2)
            else:
                stability[t] = 0.0
        return stability

    def _compute_rolling_sharpe(self, window: int = 12) -> list:
        """Compute rolling Sharpe ratio (12-month window)."""
        history = self._metrics.get("allocation_history", [])
        returns = [entry.get("return", 0) for entry in history]
        if len(returns) < window:
            return []

        monthly_rf = 0.04 / 12
        rolling = []
        for i in range(window, len(returns) + 1):
            window_returns = np.array(returns[i - window:i])
            excess = window_returns - monthly_rf
            std = np.std(excess)
            if std > 0:
                sharpe = (np.mean(excess) / std) * np.sqrt(12)
            else:
                sharpe = 0
            rolling.append(round(float(sharpe), 2))
        return rolling

    # ════════════════════════════════════════════════════
    #  HTML BUILDER
    # ════════════════════════════════════════════════════

    def _build_html(self) -> str:
        """Build the complete HTML dashboard."""
        summary = self._metrics.get("summary", {})
        equity_curve = self._prepare_equity_curve()
        drawdown = self._prepare_drawdown_curve()
        alloc = self._prepare_allocation_history()
        stability = self._compute_allocation_stability()
        rolling_sharpe = self._compute_rolling_sharpe()
        regime_data = self._prepare_regime_data()

        # Serialize data for JS
        js_data = {
            "equityCurve": equity_curve,
            "drawdown": drawdown,
            "months": alloc["months"],
            "regimes": alloc["regimes"],
            "returns": alloc["returns"],
            "defensiveRatios": alloc["defensive_ratios"],
            "SPY": alloc["SPY"],
            "BND": alloc["BND"],
            "GLD": alloc["GLD"],
            "BTC": alloc["BTC"],
            "CASH": alloc["CASH"],
            "stability": stability,
            "rollingSharpe": rolling_sharpe,
            "regimeStats": regime_data,
        }

        # Regime color map
        regime_colors = {
            "bull_low_vol": "#22c55e",
            "bull_high_vol": "#84cc16",
            "bear_low_vol": "#f97316",
            "bear_high_vol": "#ef4444",
            "sideways": "#6b7280",
            "crisis": "#991b1b",
            "recovery": "#0ea5e9",
            "unknown": "#9ca3af",
        }

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Performance Analytics Dashboard — Hybrid Intelligence Portfolio System</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg-primary: #0f172a;
    --bg-secondary: #1e293b;
    --bg-card: #1e293b;
    --bg-card-hover: #334155;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-green: #22c55e;
    --accent-red: #ef4444;
    --accent-blue: #3b82f6;
    --accent-purple: #a855f7;
    --accent-amber: #f59e0b;
    --accent-cyan: #06b6d4;
    --border: #334155;
    --gradient-1: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --gradient-2: linear-gradient(135deg, #22c55e 0%, #06b6d4 100%);
    --gradient-3: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);
    --shadow: 0 4px 24px rgba(0,0,0,0.3);
    --shadow-lg: 0 8px 40px rgba(0,0,0,0.4);
  }}

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    min-height: 100vh;
    overflow-x: hidden;
  }}

  .dashboard-header {{
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 50%, #1e1b4b 100%);
    border-bottom: 1px solid var(--border);
    padding: 2rem 3rem;
    position: relative;
    overflow: hidden;
  }}

  .dashboard-header::before {{
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, transparent 70%);
    border-radius: 50%;
  }}

  .header-content {{
    position: relative;
    z-index: 1;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}

  .header-title h1 {{
    font-size: 1.75rem;
    font-weight: 700;
    background: var(--gradient-1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
  }}

  .header-title p {{
    color: var(--text-secondary);
    font-size: 0.875rem;
    margin-top: 0.25rem;
  }}

  .header-badge {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.75rem 1.5rem;
    text-align: center;
  }}

  .header-badge .value {{
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--accent-green);
  }}

  .header-badge .label {{
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
  }}

  .container {{
    max-width: 1400px;
    margin: 0 auto;
    padding: 2rem;
  }}

  /* KPI Cards */
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
  }}

  .kpi-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
  }}

  .kpi-card:hover {{
    background: var(--bg-card-hover);
    transform: translateY(-2px);
    box-shadow: var(--shadow);
  }}

  .kpi-card .kpi-label {{
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
  }}

  .kpi-card .kpi-value {{
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -1px;
  }}

  .kpi-card .kpi-sub {{
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin-top: 0.25rem;
  }}

  .kpi-card::after {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
  }}

  .kpi-card.green::after {{ background: var(--gradient-2); }}
  .kpi-card.blue::after {{ background: var(--gradient-1); }}
  .kpi-card.amber::after {{ background: var(--gradient-3); }}
  .kpi-card.green .kpi-value {{ color: var(--accent-green); }}
  .kpi-card.blue .kpi-value {{ color: var(--accent-blue); }}
  .kpi-card.amber .kpi-value {{ color: var(--accent-amber); }}
  .kpi-card.red .kpi-value {{ color: var(--accent-red); }}
  .kpi-card.red::after {{ background: linear-gradient(135deg, #ef4444, #991b1b); }}
  .kpi-card.purple .kpi-value {{ color: var(--accent-purple); }}
  .kpi-card.purple::after {{ background: linear-gradient(135deg, #a855f7, #7c3aed); }}
  .kpi-card.cyan .kpi-value {{ color: var(--accent-cyan); }}
  .kpi-card.cyan::after {{ background: linear-gradient(135deg, #06b6d4, #0891b2); }}

  /* Chart Panels */
  .chart-grid {{
    display: grid;
    grid-template-columns: 1fr;
    gap: 1.5rem;
    margin-bottom: 2rem;
  }}

  .chart-row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
  }}

  .chart-panel {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    transition: box-shadow 0.3s ease;
  }}

  .chart-panel:hover {{
    box-shadow: var(--shadow);
  }}

  .chart-panel h3 {{
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }}

  .chart-panel h3::before {{
    content: '';
    width: 4px;
    height: 16px;
    border-radius: 2px;
    background: var(--gradient-1);
  }}

  .chart-container {{
    position: relative;
    height: 300px;
  }}

  .chart-container.tall {{
    height: 350px;
  }}

  /* Regime Legend */
  .regime-legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
  }}

  .regime-item {{
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.75rem;
    color: var(--text-secondary);
  }}

  .regime-dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
  }}

  /* Stability Bars */
  .stability-grid {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1rem;
    margin-top: 1rem;
  }}

  .stability-item {{
    text-align: center;
  }}

  .stability-bar-bg {{
    height: 120px;
    background: var(--bg-primary);
    border-radius: 8px;
    position: relative;
    overflow: hidden;
    margin-bottom: 0.5rem;
  }}

  .stability-bar {{
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    border-radius: 8px;
    transition: height 0.5s ease;
  }}

  .stability-label {{
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-primary);
  }}

  .stability-value {{
    font-size: 0.7rem;
    color: var(--text-muted);
    margin-top: 0.25rem;
  }}

  /* Footer */
  .dashboard-footer {{
    text-align: center;
    padding: 2rem;
    color: var(--text-muted);
    font-size: 0.75rem;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
  }}

  /* Regime timeline bar */
  .regime-timeline {{
    display: flex;
    height: 32px;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 1rem;
  }}

  .regime-segment {{
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.6rem;
    font-weight: 600;
    color: rgba(255,255,255,0.8);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    min-width: 2px;
  }}

  @media (max-width: 900px) {{
    .chart-row {{ grid-template-columns: 1fr; }}
    .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .container {{ padding: 1rem; }}
    .dashboard-header {{ padding: 1.5rem; }}
  }}
</style>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
</head>
<body>

<!-- Header -->
<div class="dashboard-header">
  <div class="header-content">
    <div class="header-title">
      <h1>📊 Performance Analytics Dashboard</h1>
      <p>Hybrid Intelligence Portfolio System — {summary.get('years', 0)} Year Rolling Backtest</p>
    </div>
    <div style="display:flex; gap:1rem;">
      <div class="header-badge">
        <div class="value">{summary.get('total_months', 0)}</div>
        <div class="label">Months Simulated</div>
      </div>
      <div class="header-badge">
        <div class="value" style="color: var(--accent-blue);">{summary.get('total_rebalances', 0)}</div>
        <div class="label">Rebalance Events</div>
      </div>
    </div>
  </div>
</div>

<div class="container">

  <!-- KPI Cards -->
  <div class="kpi-grid">
    <div class="kpi-card green">
      <div class="kpi-label">Total Return</div>
      <div class="kpi-value">{summary.get('total_return', 0)*100:+.1f}%</div>
      <div class="kpi-sub">{summary.get('years', 0)} years</div>
    </div>
    <div class="kpi-card green">
      <div class="kpi-label">Annualized Return</div>
      <div class="kpi-value">{summary.get('annualized_return', 0)*100:+.1f}%</div>
      <div class="kpi-sub">CAGR</div>
    </div>
    <div class="kpi-card blue">
      <div class="kpi-label">Sharpe Ratio</div>
      <div class="kpi-value">{summary.get('sharpe_ratio', 0):.2f}</div>
      <div class="kpi-sub">Risk-Adjusted</div>
    </div>
    <div class="kpi-card purple">
      <div class="kpi-label">Sortino Ratio</div>
      <div class="kpi-value">{summary.get('sortino_ratio', 0):.2f}</div>
      <div class="kpi-sub">Downside Risk</div>
    </div>
    <div class="kpi-card cyan">
      <div class="kpi-label">Calmar Ratio</div>
      <div class="kpi-value">{summary.get('calmar_ratio', 0):.2f}</div>
      <div class="kpi-sub">Return / Max DD</div>
    </div>
    <div class="kpi-card red">
      <div class="kpi-label">Max Drawdown</div>
      <div class="kpi-value">{summary.get('max_drawdown', 0)*100:.1f}%</div>
      <div class="kpi-sub">Worst Peak-to-Trough</div>
    </div>
    <div class="kpi-card amber">
      <div class="kpi-label">Annualized Vol</div>
      <div class="kpi-value">{summary.get('annualized_volatility', 0)*100:.1f}%</div>
      <div class="kpi-sub">Standard Deviation</div>
    </div>
    <div class="kpi-card green">
      <div class="kpi-label">Win Rate</div>
      <div class="kpi-value">{summary.get('win_rate', 0)*100:.0f}%</div>
      <div class="kpi-sub">Positive Months</div>
    </div>
  </div>

  <!-- Regime Timeline -->
  <div class="chart-panel" style="margin-bottom:1.5rem;">
    <h3>Market Regime Timeline</h3>
    <div class="regime-timeline" id="regimeTimeline"></div>
    <div class="regime-legend">
      <div class="regime-item"><div class="regime-dot" style="background:#22c55e;"></div>Bull Low Vol</div>
      <div class="regime-item"><div class="regime-dot" style="background:#84cc16;"></div>Bull High Vol</div>
      <div class="regime-item"><div class="regime-dot" style="background:#f97316;"></div>Bear Low Vol</div>
      <div class="regime-item"><div class="regime-dot" style="background:#ef4444;"></div>Bear High Vol</div>
      <div class="regime-item"><div class="regime-dot" style="background:#6b7280;"></div>Sideways</div>
      <div class="regime-item"><div class="regime-dot" style="background:#0ea5e9;"></div>Recovery</div>
    </div>
  </div>

  <!-- Equity Curve + Drawdown -->
  <div class="chart-row">
    <div class="chart-panel">
      <h3>Equity Curve — Cumulative Returns</h3>
      <div class="chart-container tall"><canvas id="equityChart"></canvas></div>
    </div>
    <div class="chart-panel">
      <h3>Drawdown Underwater Chart</h3>
      <div class="chart-container tall"><canvas id="drawdownChart"></canvas></div>
    </div>
  </div>

  <!-- Allocation Evolution + Monthly Returns -->
  <div class="chart-row">
    <div class="chart-panel">
      <h3>Allocation Evolution — Weight Stacking</h3>
      <div class="chart-container tall"><canvas id="allocationChart"></canvas></div>
    </div>
    <div class="chart-panel">
      <h3>Monthly Returns Distribution</h3>
      <div class="chart-container tall"><canvas id="monthlyReturnsChart"></canvas></div>
    </div>
  </div>

  <!-- Rolling Sharpe + Defensive Ratio -->
  <div class="chart-row">
    <div class="chart-panel">
      <h3>Rolling 12-Month Sharpe Ratio (Confidence Trend)</h3>
      <div class="chart-container"><canvas id="rollingChart"></canvas></div>
    </div>
    <div class="chart-panel">
      <h3>Defensive Ratio Over Time (Allocation Stability)</h3>
      <div class="chart-container"><canvas id="defensiveChart"></canvas></div>
    </div>
  </div>

  <!-- Allocation Stability -->
  <div class="chart-panel">
    <h3>Allocation Stability — Weight Volatility per Asset</h3>
    <p style="color:var(--text-muted); font-size:0.8rem; margin-bottom:1rem;">
      Lower volatility = more stable allocation. High stability indicates the engine avoids unnecessary rebalancing.
    </p>
    <div class="stability-grid">
      {self._render_stability_bars(stability)}
    </div>
  </div>

</div>

<div class="dashboard-footer">
  Generated by Hybrid Intelligence Portfolio System v1.0.0 &nbsp;|&nbsp; {self._timestamp}
  <br>Government-Grade Analytics — MLOps + LLMOps Engine
</div>

<script>
const DATA = {json.dumps(js_data)};
const REGIME_COLORS = {json.dumps(regime_colors)};

// ── Regime Timeline ──────────────────────────────────
(function() {{
  const container = document.getElementById('regimeTimeline');
  const regimes = DATA.regimes;
  let segments = [];
  let current = regimes[0];
  let count = 1;
  for (let i = 1; i < regimes.length; i++) {{
    if (regimes[i] === current) {{
      count++;
    }} else {{
      segments.push({{ regime: current, count: count }});
      current = regimes[i];
      count = 1;
    }}
  }}
  segments.push({{ regime: current, count: count }});
  const total = regimes.length;
  segments.forEach(s => {{
    const div = document.createElement('div');
    div.className = 'regime-segment';
    div.style.width = (s.count / total * 100) + '%';
    div.style.background = REGIME_COLORS[s.regime] || '#6b7280';
    if (s.count > 5) div.textContent = s.regime.replace(/_/g, ' ');
    container.appendChild(div);
  }});
}})();

// ── Chart Defaults ───────────────────────────────────
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = '#334155';
Chart.defaults.font.family = 'Inter, sans-serif';

const months = DATA.months.map(m => 'M' + m);

// ── Equity Curve ─────────────────────────────────────
new Chart(document.getElementById('equityChart'), {{
  type: 'line',
  data: {{
    labels: months,
    datasets: [{{
      label: 'Portfolio Value ($1 invested)',
      data: DATA.equityCurve,
      borderColor: '#3b82f6',
      backgroundColor: 'rgba(59, 130, 246, 0.1)',
      borderWidth: 2,
      fill: true,
      tension: 0.3,
      pointRadius: 0,
      pointHitRadius: 10,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        callbacks: {{
          label: ctx => '$' + ctx.parsed.y.toFixed(2)
        }}
      }}
    }},
    scales: {{
      x: {{ display: true, grid: {{ display: false }}, ticks: {{ maxTicksLimit: 12 }} }},
      y: {{ display: true, grid: {{ color: '#1e293b' }}, ticks: {{ callback: v => '$' + v.toFixed(2) }} }}
    }}
  }}
}});

// ── Drawdown ─────────────────────────────────────────
new Chart(document.getElementById('drawdownChart'), {{
  type: 'line',
  data: {{
    labels: months,
    datasets: [{{
      label: 'Drawdown',
      data: DATA.drawdown,
      borderColor: '#ef4444',
      backgroundColor: 'rgba(239, 68, 68, 0.15)',
      borderWidth: 1.5,
      fill: true,
      tension: 0.3,
      pointRadius: 0,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        callbacks: {{
          label: ctx => (ctx.parsed.y * 100).toFixed(1) + '%'
        }}
      }}
    }},
    scales: {{
      x: {{ display: true, grid: {{ display: false }}, ticks: {{ maxTicksLimit: 12 }} }},
      y: {{ display: true, grid: {{ color: '#1e293b' }}, ticks: {{ callback: v => (v * 100).toFixed(0) + '%' }} }}
    }}
  }}
}});

// ── Allocation Stacked Area ──────────────────────────
new Chart(document.getElementById('allocationChart'), {{
  type: 'line',
  data: {{
    labels: months,
    datasets: [
      {{ label: 'SPY', data: DATA.SPY, borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.4)', fill: true, tension: 0.3, pointRadius: 0 }},
      {{ label: 'BND', data: DATA.BND, borderColor: '#22c55e', backgroundColor: 'rgba(34,197,94,0.4)', fill: true, tension: 0.3, pointRadius: 0 }},
      {{ label: 'GLD', data: DATA.GLD, borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.4)', fill: true, tension: 0.3, pointRadius: 0 }},
      {{ label: 'BTC', data: DATA.BTC, borderColor: '#a855f7', backgroundColor: 'rgba(168,85,247,0.4)', fill: true, tension: 0.3, pointRadius: 0 }},
      {{ label: 'CASH', data: DATA.CASH, borderColor: '#64748b', backgroundColor: 'rgba(100,116,139,0.4)', fill: true, tension: 0.3, pointRadius: 0 }},
    ]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'bottom', labels: {{ usePointStyle: true, padding: 15 }} }},
      tooltip: {{ mode: 'index', callbacks: {{ label: ctx => ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(1) + '%' }} }}
    }},
    scales: {{
      x: {{ display: true, grid: {{ display: false }}, ticks: {{ maxTicksLimit: 12 }} }},
      y: {{ display: true, stacked: true, grid: {{ color: '#1e293b' }}, ticks: {{ callback: v => v + '%' }}, max: 100 }}
    }}
  }}
}});

// ── Monthly Returns Bar ──────────────────────────────
new Chart(document.getElementById('monthlyReturnsChart'), {{
  type: 'bar',
  data: {{
    labels: months,
    datasets: [{{
      label: 'Monthly Return',
      data: DATA.returns,
      backgroundColor: DATA.returns.map(r => r >= 0 ? 'rgba(34,197,94,0.7)' : 'rgba(239,68,68,0.7)'),
      borderRadius: 2,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{ callbacks: {{ label: ctx => ctx.parsed.y.toFixed(2) + '%' }} }}
    }},
    scales: {{
      x: {{ display: true, grid: {{ display: false }}, ticks: {{ maxTicksLimit: 12 }} }},
      y: {{ display: true, grid: {{ color: '#1e293b' }}, ticks: {{ callback: v => v + '%' }} }}
    }}
  }}
}});

// ── Rolling Sharpe ───────────────────────────────────
new Chart(document.getElementById('rollingChart'), {{
  type: 'line',
  data: {{
    labels: months.slice(12),
    datasets: [
      {{
        label: '12M Rolling Sharpe',
        data: DATA.rollingSharpe,
        borderColor: '#a855f7',
        backgroundColor: 'rgba(168,85,247,0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.4,
        pointRadius: 0,
      }},
      {{
        label: 'Zero Line',
        data: new Array(DATA.rollingSharpe.length).fill(0),
        borderColor: '#64748b',
        borderWidth: 1,
        borderDash: [5, 5],
        pointRadius: 0,
        fill: false,
      }}
    ]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{ callbacks: {{ label: ctx => 'Sharpe: ' + ctx.parsed.y.toFixed(2) }} }}
    }},
    scales: {{
      x: {{ display: true, grid: {{ display: false }}, ticks: {{ maxTicksLimit: 10 }} }},
      y: {{ display: true, grid: {{ color: '#1e293b' }} }}
    }}
  }}
}});

// ── Defensive Ratio ──────────────────────────────────
new Chart(document.getElementById('defensiveChart'), {{
  type: 'line',
  data: {{
    labels: months,
    datasets: [{{
      label: 'Defensive Ratio (BND+GLD+CASH)',
      data: DATA.defensiveRatios,
      borderColor: '#06b6d4',
      backgroundColor: 'rgba(6,182,212,0.1)',
      borderWidth: 2,
      fill: true,
      tension: 0.3,
      pointRadius: 0,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{ callbacks: {{ label: ctx => ctx.parsed.y.toFixed(1) + '% defensive' }} }}
    }},
    scales: {{
      x: {{ display: true, grid: {{ display: false }}, ticks: {{ maxTicksLimit: 12 }} }},
      y: {{ display: true, grid: {{ color: '#1e293b' }}, ticks: {{ callback: v => v + '%' }}, min: 0, max: 100 }}
    }}
  }}
}});
</script>
</body>
</html>"""

    def _render_stability_bars(self, stability: dict) -> str:
        """Render the allocation stability bar components."""
        colors = {
            "SPY": "#3b82f6",
            "BND": "#22c55e",
            "GLD": "#f59e0b",
            "BTC": "#a855f7",
            "CASH": "#64748b",
        }
        max_val = max(stability.values()) if stability else 1
        html = ""
        for ticker, vol in stability.items():
            height = min(100, (vol / max_val * 100)) if max_val > 0 else 0
            color = colors.get(ticker, "#6b7280")
            html += f"""
      <div class="stability-item">
        <div class="stability-bar-bg">
          <div class="stability-bar" style="height:{height}%; background:{color};"></div>
        </div>
        <div class="stability-label">{ticker}</div>
        <div class="stability-value">σ = {vol:.1f}%</div>
      </div>"""
        return html
