'use client';

import { useState, useEffect } from 'react';
import { useAuth } from '../../context/AuthContext';
import api from '../../lib/api';
import { useRouter } from 'next/navigation';
import { Wallet, Key, ShieldCheck, RefreshCw, AlertCircle, ArrowRight } from 'lucide-react';

export default function ConnectPage() {
    const { user, setUser } = useAuth();
    const router = useRouter();

    const [apiKey, setApiKey] = useState('');
    const [apiSecret, setApiSecret] = useState('');
    const [testnet, setTestnet] = useState(true);

    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');

    const [balances, setBalances] = useState(null);
    const [loadingBalances, setLoadingBalances] = useState(false);

    useEffect(() => {
        if (user?.binance_connected) {
            fetchBalances();
        }
    }, [user]);

    const fetchBalances = async () => {
        setLoadingBalances(true);
        try {
            const res = await api.get('/portfolio/balances');
            setBalances(res.data);
        } catch (err) {
            console.error(err);
        } finally {
            setLoadingBalances(false);
        }
    };

    const handleConnect = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');
        setSuccess('');

        try {
            await api.post('/portfolio/connect-binance', {
                api_key: apiKey,
                api_secret: apiSecret,
                testnet
            });

            setSuccess('Binance connected successfully!');
            // Update local user state
            setUser({ ...user, binance_connected: true });
            fetchBalances();
        } catch (err) {
            setError(err.response?.data?.detail || 'Connection failed. Check your API keys.');
        } finally {
            setLoading(false);
        }
    };

    if (!user) return null;

    return (
        <div className="container">
            <div className="flex-between mb-4">
                <div>
                    <h2>Binance Integration</h2>
                    <p style={{ color: 'var(--text-secondary)' }}>Connect your exchange account for real-time portfolio management.</p>
                </div>
                {user?.binance_connected && (
                    <div className="bg-success text-success" style={{ padding: '0.5rem 1rem', borderRadius: 'var(--radius-full)', fontSize: '0.875rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <ShieldCheck size={16} /> Connected
                    </div>
                )}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '2rem' }}>

                {/* Connection Form */}
                <div className="glass-panel" style={{ padding: '2rem' }}>
                    <h3 className="flex-center" style={{ justifyContent: 'flex-start', gap: '0.5rem', marginBottom: '1.5rem' }}>
                        <Key size={20} color="var(--accent-primary)" /> API Credentials
                    </h3>

                    {error && (
                        <div className="bg-danger text-danger mb-3" style={{ padding: '0.75rem', borderRadius: 'var(--radius-sm)', fontSize: '0.9rem', display: 'flex', gap: '0.5rem' }}>
                            <AlertCircle size={18} /> {error}
                        </div>
                    )}

                    {success && (
                        <div className="bg-success text-success mb-3" style={{ padding: '0.75rem', borderRadius: 'var(--radius-sm)', fontSize: '0.9rem', display: 'flex', gap: '0.5rem' }}>
                            <ShieldCheck size={18} /> {success}
                        </div>
                    )}

                    <form onSubmit={handleConnect}>
                        <div className="input-group">
                            <label className="input-label">API Key</label>
                            <input
                                type="password"
                                className="input-field"
                                value={apiKey}
                                onChange={(e) => setApiKey(e.target.value)}
                                placeholder="Enter Binance API Key"
                                required={!user?.binance_connected}
                            />
                        </div>

                        <div className="input-group">
                            <label className="input-label">API Secret</label>
                            <input
                                type="password"
                                className="input-field"
                                value={apiSecret}
                                onChange={(e) => setApiSecret(e.target.value)}
                                placeholder="Enter Binance API Secret"
                                required={!user?.binance_connected}
                            />
                        </div>

                        <div className="input-group" style={{ flexDirection: 'row', alignItems: 'center', gap: '1rem', marginTop: '1rem' }}>
                            <input
                                type="checkbox"
                                id="testnet"
                                checked={testnet}
                                onChange={(e) => setTestnet(e.target.checked)}
                                style={{ width: '1.2rem', height: '1.2rem', accentColor: 'var(--accent-primary)' }}
                            />
                            <label htmlFor="testnet" style={{ cursor: 'pointer', fontWeight: 500 }}>Use Binance Testnet</label>
                        </div>

                        <button type="submit" className="btn-primary mt-3" style={{ width: '100%' }} disabled={loading}>
                            {loading ? <RefreshCw size={18} className="animate-spin" /> : <ShieldCheck size={18} />}
                            {user?.binance_connected ? 'Update Connection' : 'Connect Account'}
                        </button>
                    </form>

                    <div className="mt-4" style={{ padding: '1rem', background: 'rgba(255,255,255,0.03)', borderRadius: 'var(--radius-md)', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                        <strong>Security Notice:</strong> Your keys are stored encrypted. We only require READ and SPOT TRADING permissions. Do NOT enable withdrawals.
                    </div>
                </div>

                {/* Balances Display */}
                <div className="glass-panel" style={{ padding: '2rem' }}>
                    <div className="flex-between mb-4">
                        <h3 className="flex-center" style={{ justifyContent: 'flex-start', gap: '0.5rem', margin: 0 }}>
                            <Wallet size={20} color="var(--accent-primary)" /> Live Balances
                        </h3>
                        {user?.binance_connected && (
                            <button className="btn-secondary" style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }} onClick={fetchBalances} disabled={loadingBalances}>
                                <RefreshCw size={14} className={loadingBalances ? 'animate-spin' : ''} /> Refresh
                            </button>
                        )}
                    </div>

                    {!user?.binance_connected ? (
                        <div className="flex-center text-center" style={{ height: '200px', flexDirection: 'column', color: 'var(--text-muted)' }}>
                            <Wallet size={48} style={{ opacity: 0.2, marginBottom: '1rem' }} />
                            <p>Connect your account to see balances.</p>
                        </div>
                    ) : loadingBalances && !balances ? (
                        <div className="flex-center text-center" style={{ height: '200px', flexDirection: 'column', color: 'var(--text-muted)' }}>
                            <RefreshCw size={32} className="animate-spin mb-2" />
                            <p>Fetching account data...</p>
                        </div>
                    ) : balances ? (
                        <>
                            <div style={{ padding: '1.5rem', background: 'rgba(0,0,0,0.2)', borderRadius: 'var(--radius-lg)', marginBottom: '1.5rem', textAlign: 'center' }}>
                                <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>Total Portfolio Value</div>
                                <div style={{ fontSize: '2.5rem', fontWeight: 700, color: 'white' }}>${balances.total_value_usdt?.toLocaleString() || '0.00'}</div>
                            </div>

                            <div style={{ maxHeight: '300px', overflowY: 'auto', paddingRight: '0.5rem' }}>
                                {balances.positions?.length > 0 ? (
                                    balances.positions.map((pos) => (
                                        <div key={pos.asset} className="flex-between" style={{ padding: '1rem', borderBottom: '1px solid var(--border-color)' }}>
                                            <div>
                                                <div style={{ fontWeight: 600, fontSize: '1.1rem' }}>{pos.asset}</div>
                                                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{pos.quantity}</div>
                                            </div>
                                            <div className="text-right">
                                                <div style={{ fontWeight: 600 }}>${pos.value_usdt.toLocaleString()}</div>
                                                <div style={{ fontSize: '0.8rem', color: 'var(--text-success)' }}>{(pos.weight * 100).toFixed(1)}% weight</div>
                                            </div>
                                        </div>
                                    ))
                                ) : (
                                    <p className="text-center mt-4">No assets found in account.</p>
                                )}
                            </div>

                            <button
                                className="btn-primary mt-4"
                                style={{ width: '100%' }}
                                onClick={() => router.push('/questionnaire')}
                            >
                                Start Portfolio Strategy Assessment <ArrowRight size={18} />
                            </button>
                        </>
                    ) : null}
                </div>
            </div>
        </div>
    );
}
