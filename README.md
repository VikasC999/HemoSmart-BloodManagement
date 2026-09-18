# HemoSmart

AI-assisted blood transfusion prediction, guideline-grounded explanation, demand forecasting, and donor coordination.

## Architecture

- `agents/`, `rag/`, `adapters/`, `schema/`, `models/`, `data/` — the ML/RAG/multi-agent core (XGBoost transfusion prediction, RAG explanation over WHO guidelines via FAISS, Prophet/LSTM demand forecasting, PDF/CSV/manual intake adapters, CrewAI 6-agent pipeline, Thompson-Sampling donor selection).
- `db/` — SQLAlchemy models and session, shared by the backend and the standalone agent scripts.
- `backend/` — FastAPI layer wrapping the above as HTTP endpoints.
- `frontend/` — React (Vite) dashboard for prediction, explanation, forecasting, chat, inventory, and donor alerts.

## Local setup

**Backend** (Python 3.10+; a `.venv` using Python 3.11 is already set up in this repo):

```bash
# from repo root
brew install postgresql@16 && brew services start postgresql@16
createdb hemosmart   # only needed once

cp backend/.env.example backend/.env   # fill in GROQ_API_KEY
export GROQ_API_KEY=your_key_here      # or `source` a .env loader of your choice
.venv/bin/uvicorn backend.main:app --reload --port 8000
```

Tables are created automatically on first run (see `db/database.py`) — no separate migration step. `DATABASE_URL` defaults to the local database above; set it explicitly (see `backend/.env.example`) to point at Neon/production instead.

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
- **Storage**: donor and inventory state now live in Postgres (`db/models.py`), replacing the Day 1 JSON files — auto-seeded with the same simulated data on first run. `users`, `audit_log`, and `forecasts` tables exist already but aren't written to until auth (Day 3) and audit logging (Day 5) land.
- **File upload predictions**: `POST /api/predict/pdf` (one patient per report) and `POST /api/predict/csv` (batch, one prediction per row) are wired up and persist to Postgres; there's no upload UI in the frontend yet (planned for Day 6) — use `curl -F file=@report.pdf ...` or the FastAPI docs at `/docs` in the meantime.
