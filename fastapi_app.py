from fastapi import FastAPI
import subprocess

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hybrid Intelligence Portfolio System API"}

@app.get("/run-agent/{agent_id}")
def run_agent(agent_id: int):

    if agent_id not in [1,2,3,4,5]:
        return {"error": "Agent must be between 1 and 5"}

    result = subprocess.run(
        ["python", "main.py", "--agent", str(agent_id)],
        capture_output=True,
        text=True
    )

    return {
        "agent": agent_id,
        "output": result.stdout
    }
