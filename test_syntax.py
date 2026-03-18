import traceback
try:
    import api.server
    print("SUCCESS")
except Exception:
    traceback.print_exc()
