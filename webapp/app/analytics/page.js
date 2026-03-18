'use client';

import { useState, useEffect } from 'react';
import { useAuth } from '../../context/AuthContext';
import api from '../../lib/api';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler
} from 'chart.js';
import { TrendingUp, Activity, BarChart2 } from 'lucide-react';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler
);

export default function AnalyticsPage() {
    const { user } = useAuth();
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        fetchAnalytics();
    }, [user]);

    const fetchAnalytics = async () => {
        try {
            const res = await api.get('/dashboard/analytics');
            setData(res.data);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to load analytics.');
        } finally {
            setLoading(false);
        }
    };

    if (!user) return null;

    if (loading) return <div className="flex-center mt-4">Loading Analytics (Running historical simulations)...</div>;

    if (error) return <div className="container text-danger mt-4">{error}</div>;

    if (!data?.equity_curve || data.equity_curve.length === 0) {
        return <div className="container mt-4 text-muted">Analytics data is running...</div>;
    }

    const equityCurve = {
        labels: data.equity_curve.map((_, i) => `Month ${i + 1}`),
        datasets: [
            {
                label: 'Portfolio Value ($)',
                data: data.equity_curve,
                borderColor: 'rgb(99, 102, 241)',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                borderWidth: 2,
                pointRadius: 0,
                fill: true,
                tension: 0.4
            }
        ]
    };

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#a0a6cc' } },
            x: { grid: { display: false }, ticks: { color: '#a0a6cc' } }
        },
        plugins: {
            legend: { display: false }
        }
    };

    return (
        <div className="container" style={{ paddingBottom: '3rem' }}>
            <div className="flex-between mb-4">
                <h2>Performance Analytics</h2>
                <div className="text-secondary" style={{ fontSize: '0.9rem' }}>Based on engine backtest simulation</div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1.5rem', marginBottom: '2rem' }}>
                <div className="glass-panel text-center" style={{ padding: '1.5rem' }}>
                    <TrendingUp size={24} color="var(--accent-primary)" style={{ margin: '0 auto 0.5rem' }} />
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Backtest Strategy Return</div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--status-success)', marginTop: '0.5rem' }}>
                        +{(data.summary?.total_return * 100).toFixed(2)}%
                    </div>
                </div>

                <div className="glass-panel text-center" style={{ padding: '1.5rem' }}>
                    <Activity size={24} color="var(--accent-secondary)" style={{ margin: '0 auto 0.5rem' }} />
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Tested Win Rate</div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 700, marginTop: '0.5rem' }}>
                        {(data.summary?.win_rate * 100).toFixed(1)}%
                    </div>
                </div>

                <div className="glass-panel text-center" style={{ padding: '1.5rem' }}>
                    <BarChart2 size={24} color="var(--status-warning)" style={{ margin: '0 auto 0.5rem' }} />
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Simulated Downside (Max DD)</div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--status-danger)', marginTop: '0.5rem' }}>
                        -{(data.summary?.max_drawdown * 100).toFixed(2)}%
                    </div>
                </div>
            </div>

            <div className="glass-panel" style={{ padding: '2rem' }}>
                <h3 className="mb-4">Expected Equity Curve Growth</h3>
                <div style={{ height: '350px' }}>
                    <Line data={equityCurve} options={chartOptions} />
                </div>
            </div>
        </div>
    );
}
