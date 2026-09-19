---
title: HemoSmart API
emoji: 🩸
colorFrom: red
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
---

# HemoSmart

AI-assisted blood transfusion prediction, guideline-grounded explanation, demand forecasting, and donor coordination.

## Architecture

- `agents/`, `rag/`, `adapters/`, `schema/`, `models/`, `data/` — the ML/RAG/multi-agent core (XGBoost transfusion prediction, RAG explanation over WHO guidelines via FAISS, Prophet/LSTM demand forecasting, PDF/CSV/manual intake adapters, CrewAI 6-agent pipeline, Thompson-Sampling donor selection).
- `db/` — SQLAlchemy models and session, shared by the backend and the standalone agent scripts.
- `backend/` — FastAPI layer wrapping the above as HTTP endpoints, with JWT auth, role-based access control, and audit logging.
- `mcp_server/` — a real Model Context Protocol server exposing HemoSmart's own tools (predict, explain, check inventory, alert donors) to any MCP-compatible AI client (Claude Desktop, Claude Code, etc.) — see **MCP server** below.
- `integrations/mcp/` — a plain adapter interface for external hospital systems (named for continuity with the project report, not related to the Model Context Protocol above — see that module's docstring for why those are different things).
- `frontend/` — React (Vite) app with a separate dashboard per role (Admin, Blood Bank Manager, Hospital Staff, Donor Coordinator, Auditor): prediction, explanation, forecasting, chat, inventory, donor alerts and management, patient history, file upload, and an audit log viewer.
- `scripts/seed.py` — puts the database into a known demo state (see **Seed data** below).
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

The image installs a CPU-only build of `torch` before the rest of `backend/requirements.txt` — `sentence-transformers` and `crewai`'s dependency tree both pull in `torch`, and pip's default wheel is the CUDA/GPU build (several GB of unused `nvidia-*` packages, since every model in this project runs CPU inference). This cuts the image to ~1.3GB compressed (~4.9GB unpacked on disk; `docker images` shows the latter).

The image runs as a non-root user (uid 1000) and has the RAG embedding model baked in at build time, so a cold start never downloads anything. **Memory:** the backend needs ~600MB resident (torch + sentence-transformers + Prophet + CrewAI), so free tiers capped at 512MB (e.g. Render) will OOM-kill it — see **Deployment**.

## Seed data

```bash
python -m scripts.seed                 # idempotent: adds whatever is missing
python -m scripts.seed --reset --yes   # wipes ALL data, then re-seeds
```

Creates the Admin (from `ADMIN_EMAIL`/`ADMIN_PASSWORD`), one demo account per other role (`manager@`, `staff@`, `coordinator@`, `auditor@hemosmart.app`, password from `DEMO_PASSWORD`, default `demo-pass-123`), the 50-donor pool, default inventory, and 8 sample patients scored by the real model (4 transfusion, 4 not). It targets whatever `DATABASE_URL` points at and prints the host first — `--reset` on the deployed Neon database wipes production, so it refuses to run without `--yes`.

## Deployment

| Piece | Host | Why |
|---|---|---|
| Backend (Docker) | [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker) | Free 16GB RAM. The backend needs ~600MB, which rules out 512MB free tiers. Sleeps after ~48h idle. |
| Postgres | [Neon](https://neon.tech) | Free serverless Postgres. |
| Redis | [Upstash](https://upstash.com) | Free; optional — caching degrades gracefully without it. |
| Frontend | [Vercel](https://vercel.com) | Free static hosting for the Vite build. |

**Order matters** — each step needs a URL from the previous one:

1. **Neon**: create a project, copy the connection string, and change the scheme to `postgresql+psycopg://` (keep `?sslmode=require`). This is `DATABASE_URL`.
2. **Upstash**: create a Redis database and copy its `rediss://default:<password>@<host>:<port>` URL. This is `REDIS_URL`.
3. **Hugging Face**: create a Space (SDK: **Docker**), then add the Space as a git remote and push:
   ```bash
   git remote add space https://huggingface.co/spaces/<user>/<space-name>
   git push space main
   ```
   The `sdk: docker` / `app_port: 8000` frontmatter at the top of this README is what tells Spaces how to run it. Under **Settings → Variables and secrets**, add these as *secrets*: `DATABASE_URL`, `REDIS_URL`, `GROQ_API_KEY`, `JWT_SECRET_KEY`, `ADMIN_EMAIL`, `ADMIN_PASSWORD`, `DEMO_PASSWORD`. Leave `CORS_ORIGINS` until step 5. The backend URL is `https://<user>-<space-name>.hf.space`.
4. **Vercel**: import the GitHub repo, set **Root Directory** to `frontend`, and add the env var `VITE_API_URL` = the Space URL (no trailing slash). Vite is auto-detected. Note the resulting `https://….vercel.app` URL.
5. Back on Hugging Face, add `CORS_ORIGINS` = the Vercel URL and let the Space restart.
6. **Seed**: from your machine, run `DATABASE_URL=<neon url> python -m scripts.seed`.

The first request after the Space has been idle takes a while (container wake-up plus model loading); subsequent ones are fast.

## MCP server

HemoSmart exposes its own tools — `predict_transfusion`, `explain_prediction`, `check_inventory`, `alert_donors` — as a real [Model Context Protocol](https://modelcontextprotocol.io) server, so any MCP-compatible AI client can connect and use them directly, not just the REST API or the built-in chat agent.

- **Remote (HTTP)**: mounted into the FastAPI app at `/mcp` — reachable at `http://localhost:8000/mcp` locally, or the deployed URL in production. Verified with the real `mcp` client SDK (session initialize → list tools → call tool), not just written and assumed to work.
- **Local (stdio)**: `python -m mcp_server.stdio_main`, for a client that launches it as a subprocess (see the docstring in `mcp_server/stdio_main.py` for a Claude Desktop config example).

No auth on the `/mcp` path today — a known simplification, same as the rest of this project names what's simplified rather than hiding it.

This is the correct use of MCP here: the protocol connects LLM clients to tools. It's deliberately *not* used to represent an external hospital system — `integrations/mcp/` (a plain adapter interface) is the right pattern for that kind of system-to-system integration, which MCP was never designed for. `POST /api/mcp/sync/{hospital_id}` demonstrates that adapter boundary against a fixture-backed mock hospital (`integrations/mcp/mock_hospital.py`).

## What's simulated vs. real

- **LLM provider**: Groq (cloud, free tier) is used throughout, not Ollama — a free-tier host can't run a local Ollama server. Set `GROQ_API_KEY` (console.groq.com).
- **Forecasting**: LSTM is the better-performing model (MAE 3.35 vs Prophet's 3.48) but requires a separate TensorFlow virtual environment (see `agents/lstm_forecast_standalone.py`) due to a dependency conflict with CrewAI's requirements. It runs via an isolated subprocess when `HEMOSMART_LSTM_PYTHON` points at that venv; otherwise the system automatically and silently falls back to Prophet. The containerized/deployed version intentionally does not set up the LSTM sidecar (not worth it for a free-tier single-service host) and always serves Prophet forecasts — this is a scoped decision, not a bug.
- **Donor alerts**: donor selection (Thompson Sampling / Multi-Armed Bandit) is fully real. Actually dispatching an SMS/WhatsApp alert is a print/log stub — real dispatch would need a paid API (Twilio, WhatsApp Business).
- **Storage**: donor and inventory state live in Postgres (`db/models.py`), replacing the Day 1 JSON files — auto-seeded with the same simulated data on first run. `forecasts` exists in the schema but isn't written to yet.
- **Audit logging**: every `/api/*` request logs to `audit_log` (method/path/status/user/timestamp); prediction creation, donor alerts, and inventory changes additionally log structured detail (e.g. previous vs. new stock level). Read via `GET /api/audit-log` (Auditor/Admin only), and shown in the Auditor/Admin dashboards.
- **Auth**: JWT + 5-role RBAC is enforced on every route. The frontend shows each role its own dashboard, but the backend is what actually enforces access — a hidden button is a convenience, a 403 is the guarantee.
- **Caching**: chat session state and forecast responses are cached in Redis; both degrade gracefully (logged warning, no error) if Redis is unreachable — it's a performance layer, not a dependency.
- **File upload predictions**: `POST /api/predict/pdf` (one patient per report) and `POST /api/predict/csv` (batch, one prediction per row) are wired up and persist to Postgres; the frontend's upload card drives both.
