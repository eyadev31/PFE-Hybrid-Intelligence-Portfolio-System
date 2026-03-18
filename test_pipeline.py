"""
End-to-End Pipeline Test
Tests the full flow: Register → DAQ Start → Submit Answers → Results
"""
import requests
import time
import json

BASE = "http://localhost:8000"

def test_pipeline():
    print("=" * 60)
    print("  END-TO-END PIPELINE TEST")
    print("=" * 60)

    # 1. Register / Login
    print("\n[STEP 1] Registering user...")
    r = requests.post(
        f"{BASE}/api/auth/register",
        json={"email": "e2e@gov.com", "password": "test123", "name": "E2E Test"},
    )
    if r.status_code != 200:
        r = requests.post(
            f"{BASE}/api/auth/login",
            json={"email": "e2e@gov.com", "password": "test123"},
        )
    
    if r.status_code != 200:
        print(f"  FAILED: {r.status_code} {r.text}")
        return
    
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print(f"  OK - Token: {token[:25]}...")

    # 2. Health check
    print("\n[STEP 2] Health check...")
    r_health = requests.get(f"{BASE}/api/health")
    print(f"  Status: {r_health.status_code} - {r_health.json()}")

    # 3. DAQ Start (Agent 1 + Agent 2 Phase 1)
    print("\n[STEP 3] Starting DAQ session (Agent 1 + Agent 2 Phase 1)...")
    print("  This runs Agent 1 (macro analysis) and generates questions...")
    t0 = time.time()
    r2 = requests.post(
        f"{BASE}/api/daq/start",
        json={"mock": True},
        headers=headers,
        timeout=120,
    )
    elapsed = time.time() - t0
    print(f"  Status: {r2.status_code} (took {elapsed:.1f}s)")

    if r2.status_code != 200:
        print(f"  FAILED: {r2.text[:500]}")
        return

    data = r2.json()
    session_id = data["session_id"]
    questions = data["questions"]
    market = data["market_context"]

    print(f"  Session ID: {session_id}")
    print(f"  Questions generated: {len(questions)}")
    print(f"  Market regime: {market['regime']}")
    print(f"  Regime confidence: {market['confidence']}")

    for i, q in enumerate(questions):
        q_text = q.get("question_text", "N/A")[:80]
        num_opts = len(q.get("options", []))
        print(f"  Q{i+1}: {q_text}... ({num_opts} options)")

    # 4. Submit answers (Agent 2 Phase 2 + Agent 3 + Agent 4)
    print("\n[STEP 4] Submitting answers (Agent 2P2 + Agent 3 + Agent 4)...")
    print("  This runs the full 3-agent pipeline...")

    answers = []
    for q in questions:
        opts = q.get("options", [])
        if opts:
            val = opts[0].get("value", "1")
            answers.append({"question_id": q["question_id"], "selected_value": val})

    t1 = time.time()
    r3 = requests.post(
        f"{BASE}/api/daq/submit",
        json={"session_id": session_id, "answers": answers},
        headers=headers,
        timeout=120,
    )
    elapsed2 = time.time() - t1
    print(f"  Status: {r3.status_code} (took {elapsed2:.1f}s)")

    if r3.status_code != 200:
        print(f"  FAILED: {r3.text[:500]}")
        return

    result = r3.json()
    print(f"  Pipeline status: {result.get('status')}")

    # Check agent outputs
    a1 = result.get("agent1_output", {})
    a2 = result.get("agent2_output", {})
    a3 = result.get("agent3_output", {})
    a4 = result.get("agent4_output", {})

    print(f"\n  --- Agent 1 (Macro) ---")
    regime = a1.get("market_regime", {})
    print(f"  Regime: {regime.get('primary_regime', 'N/A')}")
    print(f"  Confidence: {regime.get('confidence', 0)}")

    print(f"\n  --- Agent 2 (Behavioral) ---")
    risk = a2.get("risk_classification", {})
    print(f"  Risk score: {risk.get('risk_score', 'N/A')}")
    print(f"  Behavioral type: {risk.get('behavioral_type', 'N/A')}")
    print(f"  Personality: {risk.get('investor_personality', 'N/A')}")

    print(f"\n  --- Agent 3 (Portfolio) ---")
    alloc = a3.get("allocation", [])
    metrics = a3.get("portfolio_metrics", {})
    print(f"  Allocation ({len(alloc)} assets):")
    for a in alloc:
        w = a.get("weight", 0) * 100
        print(f"    {a.get('asset', '?')}: {w:.1f}%")
    print(f"  Expected return: {metrics.get('expected_annual_return', 0)*100:.1f}%")
    print(f"  Volatility: {metrics.get('annual_volatility', 0)*100:.1f}%")
    print(f"  Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  CVaR 95: {metrics.get('cvar_95', 0)*100:.1f}%")

    mc = a3.get("monte_carlo", {})
    if mc:
        print(f"  Monte Carlo median: {mc.get('median_annual_return', 0)*100:.1f}%")
        print(f"  Prob of loss: {mc.get('probability_of_loss', 0)*100:.1f}%")

    print(f"\n  --- Agent 4 (Risk Supervisor) ---")
    verdict = a4.get("final_verdict", {})
    print(f"  Status: {verdict.get('status', 'N/A')}")
    print(f"  Risk level: {verdict.get('risk_level', 'N/A')}")
    print(f"  Confidence: {verdict.get('confidence', 0)}")

    # 5. Dashboard summary
    print("\n[STEP 5] Checking dashboard summary...")
    r4 = requests.get(f"{BASE}/api/dashboard/summary", headers=headers)
    print(f"  Status: {r4.status_code}")
    if r4.status_code == 200:
        dash = r4.json()
        print(f"  Has recommendation: {dash.get('has_recommendation')}")
        print(f"  Strategy: {dash.get('strategy', 'N/A')}")

    print("\n" + "=" * 60)
    print("  ALL 4 AGENTS EXECUTED SUCCESSFULLY!")
    total = elapsed + elapsed2
    print(f"  Total pipeline time: {total:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    test_pipeline()
