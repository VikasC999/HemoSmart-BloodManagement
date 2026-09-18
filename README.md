# HemoSmart

AI-assisted blood transfusion prediction, guideline-grounded explanation, demand forecasting, and donor coordination.

## Architecture

- `agents/`, `rag/`, `adapters/`, `schema/`, `models/`, `data/` — the ML/RAG/multi-agent core (XGBoost transfusion prediction, RAG explanation over WHO guidelines via FAISS, Prophet/LSTM demand forecasting, PDF/CSV/manual intake adapters, CrewAI 6-agent pipeline, Thompson-Sampling donor selection).
- `db/` — SQLAlchemy models and session, shared by the backend and the standalone agent scripts.
- `backend/` — FastAPI layer wrapping the above as HTTP endpoints, with JWT auth + role-based access control.
- `frontend/` — React (Vite) dashboard for prediction, explanation, forecasting, chat, inventory, and donor alerts, behind a login screen.
- `Dockerfile` — single-container backend image (see **Docker** below).

## Local setup

**Backend** (Python 3.10+; a `.venv` using Python 3.11 is already set up in this repo):

```bash
# from repo root
brew install postgresql@16 redis && brew services start postgresql@16 && brew services start redis
createdb hemosmart   # only needed once

cp backend/.env.example backend/.env   # fill in GROQ_API_KEY, JWT_SECRET_KEY, ADMIN_EMAIL/PASSWORD
export GROQ_API_KEY=your_key_here      # or `source` a .env loader of your choice
.venv/bin/uvicorn backend.main:app --reload --port 8000
```

Tables are created automatically on first run (see `db/database.py`) — no separate migration step. An Admin account is also auto-created on first run (see `backend/services/bootstrap.py`) — log in with it (`POST /api/auth/login`) to get a token for every other route. `DATABASE_URL`/`REDIS_URL` default to the local services above; set them explicitly (see `backend/.env.example`) to point at Neon/Upstash in production. Redis is optional — caching degrades gracefully (with a logged warning) if it's unreachable.

Run uvicorn from the **repo root**, not from inside `backend/` — `backend`, `agents`, `rag`, `adapters`, and `schema` are imported as sibling top-level packages, which only resolves correctly when the process's working directory is the repo root.

**Frontend**:

```bash
cd frontend
npm install
npm run dev
```

Opens at `http://localhost:5173`, talking to the backend at `http://localhost:8000` (see `frontend/.env.local`).

## Docker

```bash
docker build -t hemosmart-backend .
docker run -p 8000:8000 \
  -e GROQ_API_KEY=... -e JWT_SECRET_KEY=... -e ADMIN_EMAIL=... -e ADMIN_PASSWORD=... \
  -e DATABASE_URL=postgresql+psycopg://user:pass@host/hemosmart \
  -e REDIS_URL=redis://host:6379/0 \
  hemosmart-backend
```

Single container, no LSTM sidecar (see below). Built and verified locally via [Colima](https://github.com/abiosoft/colima) (`brew install docker colima && colima start`) — a lightweight, license-free Docker runtime for macOS that doesn't need Docker Desktop. To test against a local Postgres/Redis from inside the container, use `host.docker.internal` in place of `localhost` in the connection strings, and make sure Postgres accepts password-authenticated connections from the Colima subnet (`listen_addresses = '*'` in `postgresql.conf`, plus a `host ... scram-sha-256` line in `pg_hba.conf` — local Postgres otherwise only trusts `127.0.0.1`, which a container isn't).

The image installs a CPU-only build of `torch` before the rest of `backend/requirements.txt` — `sentence-transformers` and `crewai`'s dependency tree both pull in `torch`, and pip's default wheel is the CUDA/GPU build (several GB of unused `nvidia-*` packages, since every model in this project runs CPU inference). This cuts the image from ~4GB to ~1.3GB.

## What's simulated vs. real

- **LLM provider**: Groq (cloud, free tier) is used throughout, not Ollama — a free-tier host can't run a local Ollama server. Set `GROQ_API_KEY` (console.groq.com).
- **Forecasting**: LSTM is the better-performing model (MAE 3.35 vs Prophet's 3.48) but requires a separate TensorFlow virtual environment (see `agents/lstm_forecast_standalone.py`) due to a dependency conflict with CrewAI's requirements. It runs via an isolated subprocess when `HEMOSMART_LSTM_PYTHON` points at that venv; otherwise the system automatically and silently falls back to Prophet. The containerized/deployed version intentionally does not set up the LSTM sidecar (not worth it for a free-tier single-service host) and always serves Prophet forecasts — this is a scoped decision, not a bug.
- **Donor alerts**: donor selection (Thompson Sampling / Multi-Armed Bandit) is fully real. Actually dispatching an SMS/WhatsApp alert is a print/log stub — real dispatch would need a paid API (Twilio, WhatsApp Business).
- **Storage**: donor and inventory state live in Postgres (`db/models.py`), replacing the Day 1 JSON files — auto-seeded with the same simulated data on first run. `audit_log` and `forecasts` tables exist already but aren't written to until audit logging (Day 5) lands.
- **Auth**: JWT + 5-role RBAC is enforced on every route. The frontend's login screen is intentionally minimal (functional, not the role-based dashboards planned for Day 6) — everyone sees the same panels today; the backend is what actually restricts what each role can do.
- **Caching**: chat session state and forecast responses are cached in Redis; both degrade gracefully (logged warning, no error) if Redis is unreachable — it's a performance layer, not a dependency.
- **File upload predictions**: `POST /api/predict/pdf` (one patient per report) and `POST /api/predict/csv` (batch, one prediction per row) are wired up and persist to Postgres; there's no upload UI in the frontend yet (planned for Day 6) — use `curl -F file=@report.pdf ...` or the FastAPI docs at `/docs` in the meantime.
