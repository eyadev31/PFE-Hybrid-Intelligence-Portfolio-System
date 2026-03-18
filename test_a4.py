from schemas.agent4_output import Agent4Output, validate_agent4_output

data = {
    "timestamp": "2023-10-27T10:00:00Z",
    "validation_status": "approved",
    "risk_audits": [],
    "risk_verdict": {"decision": "approved", "confidence": 0.9}
}

print("Testing direct init:")
try:
    a = Agent4Output(**data)
    print("Direct init OK")
except Exception as e:
    print(f"Direct init EXCEPTION: {e}")

print("\nTesting validate_agent4_output:")
is_valid, err = validate_agent4_output(data)
print(f"Valid: {is_valid}, Error: {err}")
