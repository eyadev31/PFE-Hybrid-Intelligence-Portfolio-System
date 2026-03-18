'use client';

import { useState, useEffect } from 'react';
import { useAuth } from '../../context/AuthContext';
import api from '../../lib/api';
import { useRouter } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import { BrainCircuit, Activity, AlertTriangle, ArrowRight, ShieldCheck } from 'lucide-react';

export default function QuestionnairePage() {
    const { user } = useAuth();
    const router = useRouter();

    const [session, setSession] = useState(null);
    const [questions, setQuestions] = useState([]);
    const [marketContext, setMarketContext] = useState(null);

    const [currentIndex, setCurrentIndex] = useState(0);
    const [answers, setAnswers] = useState([]);

    const [loading, setLoading] = useState(true);
    const [submitting, setSubmitting] = useState(false);
    const [error, setError] = useState('');

    useEffect(() => {
        startDaq();
    }, []);

    const startDaq = async () => {
        setLoading(true);
        try {
            // Pass mock=false to use real live data from Agent 1
            const res = await api.post('/daq/start', { mock: false });
            setSession(res.data.session_id);
            setQuestions(res.data.questions);
            setMarketContext(res.data.market_context);
        } catch (err) {
            setError('Failed to start DAQ: ' + (err.response?.data?.detail || err.message));
        } finally {
            setLoading(false);
        }
    };

    const handleSelect = async (choiceId) => {
        const q = questions[currentIndex];

        const newAnswer = {
            question_id: q.question_id,
            selected_choice_id: choiceId,
            time_taken_ms: 2500, // Simulated for now
            changed_answer: false
        };

        const updatedAnswers = [...answers, newAnswer];
        setAnswers(updatedAnswers);

        if (currentIndex < questions.length - 1) {
            setCurrentIndex(currentIndex + 1);
        } else {
            // Submit all
            submitAnswers(updatedAnswers);
        }
    };

    const submitAnswers = async (finalAnswers) => {
        setSubmitting(true);
        try {
            await api.post('/daq/submit', {
                session_id: session,
                answers: finalAnswers
            });
            router.push('/dashboard');
        } catch (err) {
            setError('Failed to submit: ' + (err.response?.data?.detail || err.message));
            setSubmitting(false);
        }
    };

    if (!user) return null;

    if (loading) {
        return (
            <div className="flex-center" style={{ height: '70vh', flexDirection: 'column' }}>
                <BrainCircuit size={64} className="text-accent animate-pulse-glow" style={{ color: 'var(--accent-primary)', marginBottom: '1.5rem', borderRadius: '50%' }} />
                <h3>Agent 1: Analyzing Global Markets</h3>
                <p style={{ color: 'var(--text-secondary)' }}>Generating context-aware behavioral scenarios...</p>
            </div>
        );
    }

    if (submitting) {
        return (
            <div className="flex-center" style={{ height: '70vh', flexDirection: 'column' }}>
                <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem' }}>
                    <BrainCircuit size={48} color="var(--accent-secondary)" className="animate-pulse" />
                    <Activity size={48} color="var(--accent-primary)" className="animate-pulse" style={{ animationDelay: '0.2s' }} />
                    <ShieldCheck size={48} color="var(--status-success)" className="animate-pulse" style={{ animationDelay: '0.4s' }} />
                </div>
                <h3 className="mb-2">Optimizing Portfolio...</h3>
                <ul style={{ color: 'var(--text-secondary)', listStyle: 'none', textAlign: 'center', lineHeight: '2' }}>
                    <li>✓ Agent 2: Profiling behavioral consistency</li>
                    <li>✓ Agent 3: Running Monte Carlo optimization</li>
                    <li className="text-accent" style={{ color: 'var(--accent-primary)' }}>○ Agent 4: Chief Risk Officer running guardrail audits...</li>
                </ul>
            </div>
        );
    }

    if (error) {
        return (
            <div className="container mt-4">
                <div className="bg-danger text-danger glass-panel" style={{ padding: '2rem', textAlign: 'center' }}>
                    <AlertTriangle size={48} style={{ margin: '0 auto 1rem' }} />
                    <h3>Optimization Failed</h3>
                    <p>{error}</p>
                    <button className="btn-primary mt-3" onClick={startDaq}>Retry Assessment</button>
                </div>
            </div>
        );
    }

    const currentQ = questions[currentIndex];

    return (
        <div className="container" style={{ maxWidth: '800px' }}>
            <div className="mb-4">
                <h2 className="flex-center" style={{ justifyContent: 'flex-start', gap: '0.5rem' }}>
                    <BrainCircuit size={28} color="var(--accent-primary)" />
                    Behavioral Assessment
                </h2>

                {marketContext && (
                    <div className="flex-between mt-2" style={{ padding: '0.75rem 1rem', background: 'rgba(255,255,255,0.05)', borderRadius: 'var(--radius-md)' }}>
                        <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                            <strong>Market Agent Context:</strong> Regime = <span style={{ color: 'var(--status-warning)' }}>{marketContext.regime.toUpperCase()}</span>
                        </span>
                        <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                            Question {currentIndex + 1} of {questions.length}
                        </span>
                    </div>
                )}
            </div>

            <div style={{ position: 'relative', height: '400px' }}>
                <AnimatePresence mode="wait">
                    <motion.div
                        key={currentIndex}
                        initial={{ opacity: 0, x: 50 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -50 }}
                        transition={{ duration: 0.3 }}
                        className="glass-panel"
                        style={{ padding: '2.5rem', position: 'absolute', width: '100%', top: 0 }}
                    >
                        <h3 className="mb-4 text-center" style={{ fontSize: '1.4rem', lineHeight: '1.6' }}>
                            {currentQ.text}
                        </h3>

                        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '1rem' }}>
                            {currentQ.choices.map((choice) => (
                                <button
                                    key={choice.id}
                                    className="btn-secondary"
                                    style={{
                                        padding: '1.25rem',
                                        textAlign: 'left',
                                        height: 'auto',
                                        display: 'flex',
                                        justifyContent: 'space-between',
                                        alignItems: 'center'
                                    }}
                                    onClick={() => handleSelect(choice.id)}
                                >
                                    <span style={{ fontSize: '1.05rem' }}>{choice.text}</span>
                                    <ArrowRight size={18} style={{ opacity: 0.5 }} />
                                </button>
                            ))}
                        </div>
                    </motion.div>
                </AnimatePresence>
            </div>

            {/* Progress Bar */}
            <div className="mt-4" style={{ background: 'rgba(255,255,255,0.1)', height: '8px', borderRadius: '4px', overflow: 'hidden' }}>
                <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${((currentIndex) / questions.length) * 100}%` }}
                    transition={{ duration: 0.3 }}
                    style={{ background: 'var(--accent-primary)', height: '100%' }}
                />
            </div>
        </div>
    );
}
