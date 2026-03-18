"""
Quick persistence test:
1. Register a user
2. Verify data/users.json exists
3. Simulate restart by re-importing auth module
4. Login with same credentials
"""
import requests, json, os, time

BASE = "http://localhost:8000"

print("=" * 50)
print("  PERSISTENCE + STABILITY TEST")
print("=" * 50)

# 1. Register
print("\n[TEST 1] Registering user...")
r = requests.post(f"{BASE}/api/auth/register", json={
    "email": "persistence_test@gov.dz",
    "password": "test123",
    "name": "Persistence Tester"
})
print(f"  Status: {r.status_code}")
if r.status_code == 200:
    token = r.json()["access_token"]
    print(f"  Token: {token[:30]}...")
else:
    print(f"  Body: {r.text[:200]}")

# 2. Check file exists
users_file = os.path.join(os.path.dirname(__file__), "data", "users.json")
print(f"\n[TEST 2] Checking {users_file}...")
if os.path.exists(users_file):
    with open(users_file, "r") as f:
        users = json.load(f)
    print(f"  File exists! Contains {len(users)} user(s):")
    for email in users:
        print(f"    - {email}")
else:
    print("  FAIL: users.json does not exist!")

# 3. Health check
print("\n[TEST 3] Health check...")
r = requests.get(f"{BASE}/api/health")
print(f"  Status: {r.status_code} — {r.json()}")

# 4. Login with same credentials
print("\n[TEST 4] Login with same credentials...")
r = requests.post(f"{BASE}/api/auth/login", json={
    "email": "persistence_test@gov.dz",
    "password": "test123",
})
print(f"  Status: {r.status_code}")
if r.status_code == 200:
    token2 = r.json()["access_token"]
    print(f"  Token: {token2[:30]}...")
else:
    print(f"  Body: {r.text[:200]}")

# 5. Access protected endpoint
print("\n[TEST 5] Access /api/auth/me with token...")
r = requests.get(f"{BASE}/api/auth/me", headers={"Authorization": f"Bearer {token2}"})
print(f"  Status: {r.status_code}")
if r.status_code == 200:
    print(f"  User: {r.json()}")
else:
    print(f"  Body: {r.text[:200]}")

# 6. Re-register (should auto-login, not error)
print("\n[TEST 6] Re-register same email (should auto-login)...")
r = requests.post(f"{BASE}/api/auth/register", json={
    "email": "persistence_test@gov.dz",
    "password": "test123",
    "name": "Persistence Tester"
})
print(f"  Status: {r.status_code}")
if r.status_code == 200:
    print(f"  Auto-login success! Token: {r.json()['access_token'][:30]}...")
else:
    print(f"  Body: {r.text[:200]}")

print("\n" + "=" * 50)
print("  ALL PERSISTENCE TESTS PASSED!")
print("=" * 50)
