'use client';

import Link from 'next/link';
import { useAuth } from '../context/AuthContext';
import { LayoutDashboard, Wallet, LogOut, TrendingUp, Cpu } from 'lucide-react';
import { usePathname } from 'next/navigation';

export default function Navbar() {
    const { user, logout } = useAuth();
    const pathname = usePathname();

    if (!user) return null; // Don't show nav on login screen

    return (
        <nav style={{
            background: 'var(--bg-card)',
            borderBottom: '1px solid var(--border-color)',
            padding: '1rem 0',
            position: 'sticky',
            top: 0,
            zIndex: 50,
            backdropFilter: 'blur(12px)'
        }}>
            <div className="container flex-between">
                <Link href="/dashboard" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 700, fontSize: '1.2rem', color: 'var(--text-primary)' }}>
                    <Cpu className="text-accent" style={{ color: 'var(--accent-primary)' }} />
                    Hybrid Intelligence
                </Link>

                <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'center' }}>
                    <Link href="/dashboard" style={{
                        color: pathname === '/dashboard' ? 'var(--accent-primary)' : 'var(--text-secondary)',
                        display: 'flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.9rem', fontWeight: 600
                    }}>
                        <LayoutDashboard size={18} /> Dashboard
                    </Link>
                    <Link href="/analytics" style={{
                        color: pathname === '/analytics' ? 'var(--accent-primary)' : 'var(--text-secondary)',
                        display: 'flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.9rem', fontWeight: 600
                    }}>
                        <TrendingUp size={18} /> Analytics
                    </Link>
                    <Link href="/connect" style={{
                        color: pathname === '/connect' ? 'var(--accent-primary)' : 'var(--text-secondary)',
                        display: 'flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.9rem', fontWeight: 600
                    }}>
                        <Wallet size={18} /> {user.binance_connected ? 'Wallet Connected' : 'Connect API'}
                    </Link>

                    <div style={{ borderLeft: '1px solid var(--border-light)', height: '24px', margin: '0 0.5rem' }}></div>

                    <button onClick={logout} style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                        <LogOut size={18} />
                    </button>
                </div>
            </div>
        </nav>
    );
}
