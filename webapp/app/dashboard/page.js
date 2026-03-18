'use client';

import { useState, useEffect } from 'react';
import { useAuth } from '../../context/AuthContext';
import api from '../../lib/api';
import { useRouter } from 'next/navigation';
import { Pie } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    ArcElement,
    Tooltip,
    Legend
} from 'chart.js';
import { ShieldAlert, ShieldCheck, TrendingUp, Cpu, ExternalLink } from 'lucide-react';

ChartJS.register(ArcElement, Tooltip, Legend);

export default function DashboardPage() {
    const { user } = useAuth();
    const router = useRouter();

    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    // Auto-fetch dashboard data
    useEffect(() => {
        fetchDashboard();
    }, [user]);

    const fetchDashboard = async () => {
        try {
            const res = await api.get('/dashboard/summary');
            setData(res.data);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to load dashboard.');
        } finally {
            setLoading(false);
        }
    };

    const handleRebalance = async () => {
        if (!user?.binance_connected) {
            alert("Must connect Binance API to auto-trade!");
            router.push('/connect');
            return;
        }

        // In a real scenario, this would loop through allocations and send limit/market orders.
        // For safety, we just show a mockup execution logic block
        if (confirm("Execute portfolio rebalancing on Binance? This will place real or testnet orders.")) {
            alert("Trade execution requested! (Demo Mode: Refer to api/routes/portfolio.py '/trade' route)");
        }
    };

    if (!user) return null;

    if (loading) return <div className="flex-center mt-4">Loading Dashboard...</div>;

    if (error) return <div className="container text-danger mt-4">{error}</div>;

    if (data && !data.has_recommendation) {
        return (
            <div className="container text-center mt-4 glass-panel" style={{ padding: '3rem' }}>
                <Cpu size={48} color="var(--text-muted)" style={{ margin: '0 auto 1.5rem' }} />
                <h2>No Portfolio Generated Yet</h2>
                <p className="mb-4 text-muted">Complete the behavioral assessment to get your personalized AI portfolio.</p>
                <button className="btn-primary" onClick={() => router.push('/questionnaire')}>Start DAQ Session</button>
            </div>
        );
    }

    // Chart Setup
    const allocation = data.allocation || [];
    const chartData = {
        labels: allocation.map(a => a.asset),
        datasets: [{
            data: allocation.map(a => a.weight * 100),
            backgroundColor: [
                '#6366f1', // SPY
                '#10b981', // BND
                '#f59e0b', // GLD
                '#8b5cf6', // BTC
                '#64748b', // CASH
            ],
            borderWidth: 0,
            hoverOffset: 4
        }]
    };

    const isRiskRejected = data.risk_verdict?.status === 'REJECTED';

    return (
        <div className="container" style={{ paddingBottom: '3rem' }}>
            <div className="flex-between mb-4">
                <h2>Portfolio Command Center</h2>
                <div style={{ display: 'flex', gap: '1rem' }}>
                    <button className="btn-secondary" onClick={() => router.push('/analytics')}>
                        <TrendingUp size={16} /> View Analytics
                    </button>
                    {!isRiskRejected && (
                        <button className="btn-primary" onClick={handleRebalance}>
                            <ExternalLink size={16} /> Auto-Rebalance
                        </button>
                    )}
                </div>
            </div>

            {isRiskRejected && (
                <div className="bg-danger text-danger mb-4 glass-panel" style={{ padding: '1.5rem', display: 'flex', gap: '1rem', alignItems: 'center' }}>
                    <ShieldAlert size={32} />
                    <div>
                        <strong>CRO Intercept: Risk Limit Exceeded</strong>
                        <p style={{ margin: 0, fontSize: '0.9rem' }}>Agent 4 declined execution. Re-run assessment or review guardrails.</p>
                    </div>
                </div>
            )}

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '2rem' }}>

                {/* Allocation Chart */}
                <div className="glass-panel" style={{ padding: '2rem' }}>
                    <h3 className="mb-3">Target Allocation</h3>
                    <div style={{ height: '280px', display: 'flex', justifyContent: 'center' }}>
                        <Pie data={chartData} options={{ maintainAspectRatio: false, plugins: { legend: { position: 'right', labels: { color: '#f0f2f5' } } } }} />
                    </div>
                    <div className="mt-4 pt-4 text-center" style={{ borderTop: '1px solid var(--border-light)' }}>
                        <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Strategy: </span>
                        <strong style={{ color: 'var(--accent-primary)', textTransform: 'capitalize' }}>{data.strategy?.replace('_', ' ')}</strong>
                    </div>
                </div>

                {/* Intelligence Summary */}
                <div className="glass-panel" style={{ padding: '2rem' }}>
                    <h3 className="mb-4">Intelligence Synthesis</h3>

                    <div style={{ marginBottom: '1.5rem' }}>
                        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>Agent 1: Market Regime</div>
                        <div className="text-warning" style={{ fontWeight: 600, fontSize: '1.1rem' }}>{data.market_regime?.toUpperCase()}</div>
                    </div>

                    <div style={{ marginBottom: '1.5rem' }}>
                        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>Agent 2: Risk Profile</div>
                        <div style={{ fontWeight: 600, fontSize: '1.1rem' }}>{data.risk_profile?.behavioral_type?.replace('_', ' ').toUpperCase()}</div>
                        <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>{data.risk_profile?.investor_personality}</div>
                    </div>

                    <div style={{ marginBottom: '1.5rem' }}>
                        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>Agent 4: Risk CRO</div>
                        <div className={data.risk_verdict?.status === 'APPROVED' ? 'text-success' : 'text-danger'} style={{ fontWeight: 600, fontSize: '1.1rem', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                            {data.risk_verdict?.status === 'APPROVED' ? <ShieldCheck size={18} /> : <ShieldAlert size={18} />}
                            {data.risk_verdict?.status}
                        </div>
                    </div>
                </div>

                {/* Financial Metrics */}
                <div className="glass-panel" style={{ padding: '2rem' }}>
                    <h3 className="mb-4">Monte Carlo Projections</h3>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                        <div>
                            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>Expected Return</div>
                            <div className="text-success" style={{ fontWeight: 700, fontSize: '1.5rem' }}>+{(data.portfolio_metrics?.expected_return * 100).toFixed(2)}%</div>
                        </div>

                        <div>
                            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>Estimated Risk (Vol)</div>
                            <div style={{ fontWeight: 700, fontSize: '1.5rem' }}>{(data.portfolio_metrics?.volatility * 100).toFixed(2)}%</div>
                        </div>

                        <div>
                            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>Median Max Drawdown</div>
                            <div className="text-danger" style={{ fontWeight: 700, fontSize: '1.5rem' }}>-{(data.monte_carlo?.max_drawdown * 100).toFixed(2)}%</div>
                        </div>

                        <div>
                            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>Sharpe Ratio</div>
                            <div style={{ fontWeight: 700, fontSize: '1.5rem' }}>{data.portfolio_metrics?.sharpe_ratio?.toFixed(2) || '0.00'}</div>
                        </div>
                    </div>
                </div>

            </div>
        </div>
    );
}
