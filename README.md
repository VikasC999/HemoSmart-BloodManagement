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

A wastage-reducing, autonomous blood management system: component-level transfusion prediction, unit-level inventory with FEFO allocation, a demand-supply gap projection, and a multi-agent loop that alerts donors before shortages and redistributes blood before it expires — every action grounded in the WHO *Clinical Use of Blood* handbook via RAG.

## How the autonomous loop works

`agents/autonomy/` runs on a timer (`HEMOSMART_AUTONOMY_INTERVAL_MINUTES`, default 60; `0` disables) or on demand (`POST /api/autonomy/run`). An Orchestrator runs specialist agents over a shared blackboard, each logging what it observed, decided and did:

1. **Donor (feedback)** — resolves donor replies: the Thompson-Sampling bandit updates, donated units enter stock, unanswered alerts expire after 48h.
2. **Inventory** — sweeps expired units (recorded as wastage) and computes stock vs par.
3. **Forecast** — Prophet demand per component, split across blood groups.
4. **Planner** — replays the next 7 days with the *same* allocation policy used at issue time (compatible near-expiry units first, then own group, FEFO), giving projected **shortages** and units projected to **expire unused**.
5. **Donor (outreach)** — bandit-selected alerts sized to each gap (whole blood covers PRBC + FFP, apheresis covers platelets); escalates what donors can't cover.
6. **Logistics** — offers near-expiry units the hospital won't use in time to a partner hospital (`integrations/mcp/`).
7. **Explanation** — cites the WHO handbook page behind each kind of action and writes a brief (LLM if available, template otherwise).

Each agent step commits on its own and failures are contained: a failed forecast falls back to historical averages, a failed Planner skips outreach, a failed LLM falls back to the template brief. The whole cycle is stored (`agent_cycles`) and shown in the dashboard.

Every design choice and trade-off, with how to revert it, is logged in [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md).

## Architecture

- `bloodbank/` — the blood-bank domain: catalog (groups, components, shelf lives, ABO/RhD compatibility), wastage-aware FEFO allocation, the gap projection, the stock service, and the forecaster.
- `agents/autonomy/` — the autonomous multi-agent loop (above).
- `agents/`, `rag/`, `adapters/`, `schema/`, `models/`, `data/` — prediction (one XGBoost model per component, with per-patient TreeSHAP), RAG explanation with page citations over the WHO handbook (FAISS), PDF/CSV/manual intake adapters, the CrewAI PDF-case pipeline (`agents/crew_agents.py`, CLI), the chat agent, and Thompson-Sampling donor selection.
- `db/` — SQLAlchemy models and session, shared by the backend and the standalone agent scripts.
- `backend/` — FastAPI layer wrapping the above as HTTP endpoints, with JWT auth, role-based access control, and audit logging.
- `mcp_server/` — a real Model Context Protocol server exposing HemoSmart's own tools (predict, explain, check inventory, supply outlook, alert donors) to any MCP-compatible AI client (Claude Desktop, Claude Code, etc.) — see **MCP server** below.
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

Tables are created automatically on first run (see `db/database.py`) — no separate migration step. An Admin account is also auto-created on first run (see `backend/services/bootstrap.py`) — log in with it (`POST /api/auth/login`) to get a token for every other route. `POST /api/auth/register` is Admin-only: staff accounts are provisioned by an Admin, never self-registered. `DATABASE_URL`/`REDIS_URL` default to the local services above; set them explicitly (see `backend/.env.example`) to point at Neon/Upstash in production. Redis is optional — caching degrades gracefully (with a logged warning) if it's unreachable.

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

## Training the models

All data is synthetic (`data/create_dataset.py`); retrain after changing it:

```bash
python data/create_dataset.py     # patient_data.csv, component_demand.csv
python models/train_xgboost.py    # xgb_components.pkl + per-component SHAP plots
python models/train_prophet.py    # prophet_components.pkl + demand_profile.json
python rag/setup_rag.py           # FAISS index with page metadata
```

Clinical thresholds, doses and urgency rules live in one place, `schema/clinical_thresholds.py`, shared by the label process (`data/label_process.py`) and the explanation layer so they can never disagree. After retraining, re-seed (`python -m scripts.seed --reset --yes`) so the stored sample predictions come from the new models.

## Tests

```bash
pip install -r backend/requirements-dev.txt
python -m pytest
```

Runs against a throwaway SQLite database, never your Postgres.

## Evaluation

How the Phase I report's claims line up with the code: [docs/REPORT_ALIGNMENT.md](docs/REPORT_ALIGNMENT.md). Viva walkthrough: [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md).

Each claim is measured by a script in `evaluation/` that writes its tables to `evaluation/results/` (`.md` + `.json`). All of them run on a throwaway database. Results are on synthetic data; methods and caveats are in [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md) (D45–D54).

```bash
python -m evaluation.simulate               # 90-day wastage/shortage ablation, 3 seeds (~1.5 min)
python -m evaluation.sensitivity            # outreach window x alert cap sweep (~8 min)
python -m evaluation.prediction_benchmark   # model vs threshold rule, 5-fold CV (~30 s)
python -m evaluation.forecast_backtest      # Prophet vs baselines vs LSTM, rolling origin (~3 min)
python -m evaluation.bandit_regret          # Thompson vs random/greedy/oracle donor selection (~15 s)
python -m evaluation.rag_faithfulness       # retrieval hit@2/MRR; add --llm for faithfulness (20 Groq calls)
python -m evaluation.latency                # NFR-01 response times through the real app (5 Groq calls; --no-llm to skip)
python -m evaluation.figures                # report figures -> docs/figures/ (from the results above)
```

Headline results:

| What | Result |
|---|---|
| **Wastage & shortages** (90 days vs a manual blood bank) | Units short 96 → **43** (−55%); units expired on site 46.7 → **11.0** (−76%); total units not used here (expired + redistributed) 46.7 → 47.7, so no extra collection is hidden by shipping it out. |
| **Prediction** (vs a static threshold rule) | F1 0.654 → **0.705** (PRBC), 0.685 → **0.783** (Platelets): the model learns the surgery-dependent thresholds the static rule ignores, within ~0.01 of the best possible on this data. No gain on FFP. Random Forest performs equivalently. |
| **Forecasting** (7-day MAE) | Prophet is within 0.07–0.15 units/day of the noise floor. Platelets in dengue season: 2.87 vs 3.35 for a 4-week weekday mean. The LSTM is no better than Prophet. |
| **Response times** | Prediction 0.01 s, forecast ≤ 0.35 s, RAG explanation 2.05 s median (one free-tier Groq call took 17 s). |
| **Donor selection** | Thompson sampling: 19% more donations than random calls. Plain greedy is ~2% better still (an open decision). |
| **RAG** | Plasma-guideline retrieval hit@2 0.00 → 1.00 after switching to handbook vocabulary. Explanations crediting WHO with a number its excerpts don't contain: 2 → 1 (borderline) of 10 cases; 1 direction slip remains. A small sample, so read it as a direction of travel rather than a significant result. |

## Seed data

```bash
python -m scripts.seed                 # idempotent: adds whatever is missing
python -m scripts.seed --reset --yes   # wipes ALL data, then re-seeds
python -m scripts.rescore              # after retraining: add fresh predictions, keep history (non-destructive)
```

Creates the Admin (from `ADMIN_EMAIL`/`ADMIN_PASSWORD`), one demo account per other role (`manager@`, `staff@`, `coordinator@`, `auditor@hemosmart.app`, password from `DEMO_PASSWORD`, default `demo-pass-123`), the 400-donor pool, a unit-level inventory dated relative to *now* (with a critical O- red-cell shortage and a near-expiry platelet surplus for the loop to find), and 8 sample patients with blood groups, scored by the real models.

Schema changes are applied on startup: missing tables are created and missing *nullable* columns added (`db/database.py`) — a database from before unit-level inventory upgrades in place. The old `inventory` table is no longer used. It targets whatever `DATABASE_URL` points at and prints the host first — `--reset` on the deployed Neon database wipes production, so it refuses to run without `--yes`.

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

HemoSmart exposes its own tools — `predict_transfusion`, `explain_prediction`, `check_inventory`, `supply_outlook`, `alert_donors` — as a real [Model Context Protocol](https://modelcontextprotocol.io) server, so any MCP-compatible AI client can connect and use them directly, not just the REST API or the built-in chat agent.

- **Remote (HTTP)**: mounted into the FastAPI app at `/mcp` — reachable at `http://localhost:8000/mcp` locally, or the deployed URL in production. Verified with the real `mcp` client SDK (session initialize → list tools → call tool), not just written and assumed to work.
- **Local (stdio)**: `python -m mcp_server.stdio_main`, for a client that launches it as a subprocess (see the docstring in `mcp_server/stdio_main.py` for a Claude Desktop config example).

The HTTP `/mcp` path requires the same JWT as the REST API (`Authorization: Bearer <token>` from `POST /api/auth/login`), restricted to Admin and Blood Bank Manager — the roles that hold every permission the tools expose. The stdio server is for a trusted local process and has no auth.

This is the correct use of MCP here: the protocol connects LLM clients to tools. It's deliberately *not* used to represent an external hospital system — `integrations/mcp/` (a plain adapter interface) is the right pattern for that kind of system-to-system integration, which MCP was never designed for. `POST /api/mcp/sync/{hospital_id}` demonstrates that adapter boundary against a fixture-backed mock hospital (`integrations/mcp/mock_hospital.py`).

## What's simulated vs. real

- **LLM provider**: Groq (cloud, free tier) is used throughout, not Ollama — a free-tier host can't run a local Ollama server. Set `GROQ_API_KEY` (console.groq.com).
- **Data**: all synthetic. Patient labels are simulated clinician decisions (`data/label_process.py`): surgery-dependent thresholds, a soft boundary and 1% record noise. Prediction metrics therefore measure how well the model recovers that process, not clinical validity.
- **Thresholds**: `schema/clinical_thresholds.py` is HemoSmart's local protocol. Some of its numbers (INR > 1.5 for FFP, platelets < 100k for cardiac surgery) are not in the WHO handbook the explanations cite, so explanations call them "HemoSmart protocol thresholds" and credit WHO only with what a retrieved page says.
- **Forecasting**: Prophet per component, disaggregated by blood group. The LSTM is benchmarked in `evaluation/forecast_backtest.py` (a small PyTorch model, no TensorFlow env) but not served; the old TensorFlow scripts (`agents/lstm_forecast_standalone.py`, `models/train_lstm_colab.ipynb`) are kept for reference.
- **Routine supply**: the gap projection assumes routine voluntary collection brings in a fixed share of average demand (`ROUTINE_SUPPLY_SHARE` in `bloodbank/catalog.py`).
- **Donor alerts**: donor selection (Thompson Sampling / Multi-Armed Bandit) and its feedback loop are fully real. Dispatching an SMS/WhatsApp alert is a print/log stub. Donor replies are recorded by a Donor Coordinator (`POST /api/donors/alerts/{id}/response`), or, with `HEMOSMART_SIMULATE_DONOR_RESPONSES=1` (default), sampled from each synthetic donor's hidden response rate — which the bandit never sees.
- **Partner hospital**: redistribution goes through the adapter interface to a mock partner whose acceptance policy is simulated (`integrations/mcp/mock_hospital.py`).
- **Storage**: donors, blood units, transfusions, predictions (with components, SHAP, explanation and citations), and agent cycles live in Postgres (`db/models.py`). `forecasts` exists in the schema but isn't written to yet.
- **Audit logging**: every `/api/*` request logs to `audit_log` (method/path/status/user/timestamp); prediction creation, donor alerts, and inventory changes additionally log structured detail (e.g. previous vs. new stock level). Read via `GET /api/audit-log` (Auditor/Admin only), and shown in the Auditor/Admin dashboards.
- **Auth**: JWT + 5-role RBAC is enforced on every route. The frontend shows each role its own dashboard, but the backend is what actually enforces access — a hidden button is a convenience, a 403 is the guarantee.
- **Caching**: chat session state and forecast responses are cached in Redis; both degrade gracefully (logged warning, no error) if Redis is unreachable — it's a performance layer, not a dependency.
- **File upload predictions**: `POST /api/predict/pdf` (one patient per report) and `POST /api/predict/csv` (batch, one prediction per row) are wired up and persist to Postgres; the frontend's upload card drives both.
