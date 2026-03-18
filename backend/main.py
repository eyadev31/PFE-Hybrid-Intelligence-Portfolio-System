from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DATA_CACHE_TTL_MINUTES = os.getenv("DATA_CACHE_TTL_MINUTES", "15")


def check_environment():
    required_keys = [
        GEMINI_API_KEY,
        GROQ_API_KEY,
        FRED_API_KEY,
        TWELVEDATA_API_KEY,
        NEWS_API_KEY,
    ]

    if not all(required_keys):
        raise EnvironmentError("One or more API keys are missing in the .env file.")

    print("Environment variables loaded successfully.")


def main():
    check_environment()
    print("Hybrid Intelligence Portfolio System started")
    print("LLM Provider:", LLM_PROVIDER)
    print("LLM Model:", LLM_MODEL)


if __name__ == "__main__":
    main()
