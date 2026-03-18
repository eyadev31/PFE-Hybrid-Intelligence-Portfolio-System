<div align="center">
  <h1>🧠 Hybrid Intelligence Portfolio System</h1>
  <p><em>An Advanced Autonomous Multi-Agent Asset Management Framework</em></p>
  <p><strong>Status:</strong> Projet de Fin d'Études</p>
</div>
<img width="1408" height="768" alt="Gemini_Generated_Image_5h3vhl5h3vhl5h3v (1)" src="https://github.com/user-attachments/assets/751d73e2-90d4-4076-aaaa-1fa0d4bd2b11" />

<hr />

## 🌟 Executive Summary
The **Hybrid Intelligence Portfolio System** represents a paradigm shift in autonomous wealth management. By bridging the gap between rigorous quantitative finance— utilizing Hidden Markov Models (HMM), Random Forests, and Monte Carlo simulations—and cutting-edge Large Language Model (LLM) orchestration, this system creates a deeply resilient, adaptive, and self-supervising investment pipeline.

Designed meticulously for academic and governmental review, this architecture proves the feasibility of fully autonomous financial intelligence operating in non-stationary markets, with built-in systemic risk supervision and cognitive behavioral user profiling.

---

## 🏗️ Architecture: The 5-Agent Paradigm
The system is decentralized into five highly specialized, autonomous AI agents. Rather than relying on a single monolithic model, these agents operate chronologically, forming an immutable "Chain of Thought" data pipeline. 

### 📡 Agent 1: Macro & Market Intelligence
**The Sentinel.** Ingests multi-asset telemetry from Binance and macroeconomic indicators from FRED. It employs a mathematical Ensemble Regime Detector (HMM + RF) and Volatility Classifiers to identify broader structural market regimes. Finally, an LLM synthesizes these hard-quant metrics into a structured financial narrative.

### 🧠 Agent 2: DAQ (Cognitive & Behavioral Profiling)
**The Psychologist.** Translates standard user risk-questionnaires into a highly dynamic psychometric matrix. It identifies behavioral biases (loss aversion, overconfidence) and establishes a psychological baseline to ensure the portfolio matches the true cognitive risk appetite of the investor, not just their stated goals.

### ♟️ Agent 3: Portfolio Strategist
**The Allocator.** The core quantitative engine. It receives the market regime from Agent 1 and the behavioral constraints from Agent 2, subsequently deploying advanced mathematical optimizers (Mean-Variance, Black-Litterman abstractions) to compute dynamic capital asset weightings perfectly tuned to the current timeline.

### 🛡️ Agent 4: Meta-Risk & Supervision
**The Auditor.** An overriding meta-cognitive supervisor. Agent 4 runs adversarial scenarios and Monte Carlo simulations on Agent 3's proposed allocation. If the allocation breaches systemic risk thresholds, Agent 4 possesses unilateral authority to veto the allocation, enforcing an emergency shift to cash equivalents or safe-haven assets.

### 📰 Agent 5: News Sentiment Intelligence
**The Wall Street Wire.** A 10-step, state-of-the-art Natural Language Processing (NLP) pipeline. It scrapes global news feeds, generates vector embeddings (using `sentence-transformers`), performs semantic deduplication, and scores financial impact using `ProsusAI/finbert`. It specializes in detecting Black Swan macro events before they impact the primary price-action models.

---

## 💻 Technology Stack & Engineering

### ⚙️ Backend & Machine Learning Ecosystem
- **Core Languages:** Python 3.10+
- **Web API Engine:** FastAPI, Uvicorn (RESTful architectures with extreme concurrency)
- **Quantitative Engine:** `scikit-learn`, `hmmlearn`, `pandas`, `numpy`, `scipy`
- **NLP & LLM Generation:** Google Gemini AI API, HuggingFace (`transformers`, `sentence-transformers`), `spaCy`
- **Ingestion Nodes:** Requests, FRED API, TwelveData, Binance API

### 📱 Frontend Mobile Dashboard
- **Framework:** React Native 0.83+
- **Build System:** Expo 55, EAS (Expo Application Services)
- **UI & Animations:** React Native Reanimated, Expo Blur, Expo Linear Gradient
- **Navigation:** Expo Router

---

## ⚙️ Resilience & Honesty in Design (Graceful Degradation)
In the spirit of absolute engineering honesty, the system is designed with **Graceful Degradation**. Financial APIs (like FRED or News providers) often experience rate limits or downtime. If an API request fails, the Agents seamlessly fall back to synthetic Mock Data or secondary quantitative estimation constraints. The pipeline state flags structural "degraded quality" to the user, yet the orchestration *never crashes*.

---

## 🚀 Getting Started

### 1. API & Backend Initialization
```bash
# Install critical ML/AI dependencies
pip install -r requirements.txt

# Configure environmental keys (Rename .env.example to .env)
# Needs: GEMINI_API_KEY, FRED_API_KEY, TWELVEDATA_API_KEY

# Validate API Keys and environment health
python main.py --validate

# Spin up the FastAPI Service on port 8000
python main.py --init-db
uvicorn api.server:app --reload --port 8000
```

### 2. Orchestration Simulation (CLI)
You can manually test the pipeline's reasoning chain straight from the terminal:
```bash
# Run Agent 1 (Macro) using Mock/Synthetic Data
python main.py --agent 1 --mock

# Run the full Orchestrator with persistence to database
python main.py --orchestrate --mock --persist
```

### 3. Mobile Client Deployment
```bash
cd mobile
npm install

# Start the Expo bundler for iOS/Android
npx expo start
```
# backend
cd C:\Users\yassi\OneDrive\Desktop\pfa f
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000

# mobile
C:\Users\yassi\OneDrive\Desktop\pfa f\mobile>npx expo start -c


npx eas-cli build -p android --profile preview

---

<div align="center">
  <i>Developed to the highest academic standards. Validated for intelligence resilience.</i>
</div>
