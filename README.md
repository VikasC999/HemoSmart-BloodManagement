# HemoSmart

AI-assisted blood transfusion prediction, guideline-grounded explanation, demand forecasting, and donor coordination.

## Architecture

- `agents/`, `rag/`, `adapters/`, `schema/`, `models/`, `data/` — the ML/RAG/multi-agent core (XGBoost transfusion prediction, RAG explanation over WHO guidelines via FAISS, Prophet/LSTM demand forecasting, PDF/CSV/manual intake adapters, CrewAI 6-agent pipeline, Thompson-Sampling donor selection).
- `backend/` — FastAPI layer wrapping the above as HTTP endpoints.
- `frontend/` — React (Vite) dashboard for prediction, explanation, forecasting, chat, inventory, and donor alerts.

## Local setup

**Backend** (Python 3.10+; a `.venv` using Python 3.11 is already set up in this repo):

```bash
# from repo root
cp backend/.env.example backend/.env   # fill in GROQ_API_KEY
export GROQ_API_KEY=your_key_here      # or `source` a .env loader of your choice
.venv/bin/uvicorn backend.main:app --reload --port 8000
```

Run uvicorn from the **repo root**, not from inside `backend/` — `backend`, `agents`, `rag`, `adapters`, and `schema` are imported as sibling top-level packages, which only resolves correctly when the process's working directory is the repo root.

**Frontend**:

```bash
cd frontend
npm install
npm run dev
```

Opens at `http://localhost:5173`, talking to the backend at `http://localhost:8000` (see `frontend/.env.local`).

## What's simulated vs. real

- **LLM provider**: Groq (cloud, free tier) is used throughout, not Ollama — a free-tier host can't run a local Ollama server. Set `GROQ_API_KEY` (console.groq.com).
- **Forecasting**: LSTM is the better-performing model (MAE 3.35 vs Prophet's 3.48) but requires a separate TensorFlow virtual environment (see `agents/lstm_forecast_standalone.py`) due to a dependency conflict with CrewAI's requirements. It runs via an isolated subprocess when `HEMOSMART_LSTM_PYTHON` points at that venv; otherwise the system automatically and silently falls back to Prophet. The containerized/deployed version intentionally does not set up the LSTM sidecar (not worth it for a free-tier single-service host) and always serves Prophet forecasts — this is a scoped decision, not a bug.
- **Donor alerts**: donor selection (Thompson Sampling / Multi-Armed Bandit) is fully real. Actually dispatching an SMS/WhatsApp alert is a print/log stub — real dispatch would need a paid API (Twilio, WhatsApp Business).
- **Storage**: `agents/donors.json` and `agents/inventory.json` back donor/inventory state for now; a Postgres migration is planned (see project plan).
