---
title: HemoSmart API
emoji: 🩸
colorFrom: red
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
---

<div align="center">

# 🩸 HemoSmart

### A wastage-reducing, autonomous blood management system built on RAG and multi-agent AI

Predicts which blood components a patient will need, projects the supply gap, and runs a multi-agent loop that **alerts donors before a shortage and redistributes blood before it expires**, with every action grounded in the WHO *Clinical Use of Blood* handbook.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-SHAP-EB6F2F)
![CrewAI](https://img.shields.io/badge/CrewAI-LLM%20agents-7A5C00)
![MCP](https://img.shields.io/badge/MCP-server-292621)
![Tests](https://img.shields.io/badge/tests-285%20passing-3D6B43)

Final-year project · SJBIT, Computer Science & Engineering (Data Science) · 2025–26

[Results](#results) · [Architecture](#architecture) · [Quick start](#quick-start) · [Documentation](#documentation)

</div>

---

## The problem

Blood banks face two failures at once: patients can wait on shortages, while screened units expire unused. Platelets last only 5 days, so a shortage of one group can sit beside expiring stock of another. Manual stock review struggles to see the gap coming, and donors are called too late. HemoSmart is built to close that loop automatically.

## What HemoSmart does

| | |
|---|---|
| 🧠 **Component-level prediction** | One XGBoost model per component (PRBC, platelets, FFP) with per-patient TreeSHAP, so a clinician sees *why* |
| 📚 **RAG explanations with citations** | Every explanation cites the WHO handbook page it came from; local protocol thresholds are labelled as such, never credited to WHO |
| 🩸 **Unit-level inventory** | Real units with expiry dates, ABO/RhD compatibility, and FEFO allocation that prefers near-expiry units |
| 📈 **Demand–supply gap** | Prophet forecasts per component, replayed day by day with the *same* allocation policy used at issue time, giving projected shortages and units that will expire unused |
| 🤖 **Autonomous agent loop** | Seven specialist agents on a shared blackboard collect donor replies, sweep expiry, forecast, plan, alert donors, redistribute to a partner hospital and explain, with no human clicking a button |
| 📄 **CBC report to full case** | Upload a PDF, confirm the extracted values, and six CrewAI agents process the case, each answer checked against an independently computed value |
| 🔐 **Role-aware assistant** | A chat assistant that answers only what the signed-in role's pages allow, checked in code before any data is read or any LLM is called |
| 👁️ **Agent monitor** | A live 2D "agent office" showing which agents are working, waiting, degraded or stuck, how long each step takes, and what needs a person |

## Results

All results are on **synthetic data**; every number is reproducible with one command (see [Evaluation](#evaluation)).

<p align="center">
  <img src="docs/figures/fig_simulation.png" alt="90-day simulation: expired units and unmet demand across four configurations" width="860">
</p>

| What | Result |
|---|---|
| **Wastage and shortages** (90 days vs a manual blood bank) | Units short 96 → **43** (−55%). Units expired on site 46.7 → **11.0** (−76%). Total units not used here (expired + redistributed) stays level (46.7 → 47.7), so no extra collection is hidden by shipping it out. |
| **Prediction** (vs a static threshold rule) | F1 0.654 → **0.705** (PRBC) and 0.685 → **0.783** (platelets): the model learns the surgery-dependent thresholds the static rule ignores, within ~0.01 of the best possible on this data. No gain on FFP. Random Forest performs equivalently. |
| **Forecasting** (7-day MAE) | Prophet is within 0.07–0.15 units/day of the noise floor. Platelets in dengue season: 2.87 vs 3.35 for a 4-week weekday mean. The LSTM is no better than Prophet. |
| **Response times** | Prediction 0.01 s, forecast ≤ 0.35 s, RAG explanation 2.05 s median (one free-tier Groq call took 17 s). |
| **Donor selection** | Thompson sampling gives 19% more donations than random calls. Plain greedy is ~2% better still, an open decision recorded in [D53](docs/DESIGN_DECISIONS.md). |
| **RAG** | Plasma-guideline retrieval hit@2 0.00 → 1.00 after switching to handbook vocabulary. A small sample, so read it as a direction of travel rather than a significant result. |

## Architecture

<p align="center">
  <img src="docs/figures/fig_architecture.png" alt="HemoSmart architecture: dashboards, MCP clients and hospital systems over a FastAPI backend with two agent layers" width="860">
</p>

HemoSmart has **two deliberately different agent layers**:

- **Layer 1, the autonomous supply loop** uses deterministic Python agents. They have real side effects (alerts, transfers, stock changes), so they must be testable and repeatable, and a 90-day simulation needs thousands of cycles. The LLM is used only for language: the cycle brief.
- **Layer 2, the patient case pipeline** uses CrewAI LLM agents. Each agent's answer is compared with an independently computed value and falls back to the verified one on any mismatch, so an LLM mistake can't reach a clinician.

### The supply loop

`agents/autonomy/` runs on a timer (`HEMOSMART_AUTONOMY_INTERVAL_MINUTES`, default 60; `0` disables) or on demand (`POST /api/autonomy/run`). An Orchestrator runs the agents in order over a shared blackboard:

1. **Donor (replies)**: resolves donor replies, updates the Thompson-sampling bandit, adds donated units to stock, and expires alerts unanswered after 48 h.
2. **Inventory**: sweeps expired units (recorded as wastage) and computes stock against par.
3. **Forecast**: Prophet demand per component, split across blood groups.
4. **Planner**: replays the next 7 days with the issue-time allocation policy, giving projected shortages and units that will expire unused.
5. **Donor (outreach)**: bandit-selected alerts sized to each gap (whole blood covers PRBC and FFP, apheresis covers platelets); escalates what donors can't cover.
6. **Logistics**: offers near-expiry units the hospital won't use in time to a partner hospital.
7. **Explanation**: cites the WHO handbook page behind each kind of action and writes a brief (LLM if available, template otherwise).

Each step commits on its own and failures are contained: a failed forecast falls back to historical averages, a failed Planner skips outreach, a failed LLM falls back to the template brief. The cycle is marked *degraded* and stored in `agent_cycles`.

### Security and governance

- **One permission matrix** ([schema/permissions.py](schema/permissions.py)) is shared by every REST route and the chat assistant, so the dashboard and the chatbot can't disagree about who may see what. A test fails if any route bypasses it.
- **Five roles**: Admin, Blood Bank Manager, Hospital Staff, Donor Coordinator, Auditor. JWT on every route; the backend enforces access, a hidden button is only a convenience.
- **The assistant decides access in code**, per question, before any tool or LLM runs. Claiming "I am admin" changes nothing because the role comes from the login. Donor alerts sent from chat need an explicit "yes".
- **Data minimisation**: no patient identifiers are stored, donor phones are masked to the last 4 digits, the audit log records chat decisions but never message text, and the agent monitor's telemetry is allow-listed so lab values, filenames and exception messages can't reach it.
- **Audit log** of every `/api/*` request, readable by the Auditor and Admin only.

### The agent monitor

A read-only operations view: a flat 2D office with a room per agent layer, a desk per agent, and a task panel. It answers *are the agents actually running?* with a scheduler heartbeat, stuck-step and orphaned-cycle detection, per-step timing, rate-limit waits and the share of LLM answers that checked out. It links to the page where a person acts and never acts itself. Design: [D71–D75](docs/DESIGN_DECISIONS.md) and [docs/2dvision.md](docs/2dvision.md).

### Repository layout

| Path | Purpose |
|---|---|
| `bloodbank/` | The blood-bank domain: catalog (groups, components, shelf lives, ABO/RhD), wastage-aware FEFO allocation, gap projection, stock service, forecaster |
| `agents/` | The autonomous loop (`autonomy/`), the CrewAI case pipeline, the chat assistant, agent tools, telemetry |
| `rag/` | FAISS retrieval over the WHO handbook with page metadata, explanation generator, LLM client |
| `models/`, `data/` | XGBoost and Prophet training, synthetic dataset and label process |
| `backend/` | FastAPI app: routes, JWT auth, RBAC, audit logging, services |
| `db/` | SQLAlchemy models and session, shared by the backend and the agent scripts |
| `mcp_server/` | A real Model Context Protocol server exposing HemoSmart's tools |
| `integrations/mcp/` | Plain adapter interface for external hospital systems (not the Model Context Protocol; see its docstring) |
| `frontend/` | React (Vite) app with a workspace per role |
| `evaluation/` | Scripts behind every reported number |
| `docs/` | Design decisions, report alignment, demo script, figures |
| `scripts/seed.py` | Puts the database into a known demo state |

## Technology

| Layer | Stack |
|---|---|
| Prediction | XGBoost (one model per component), TreeSHAP |
| Forecasting | Prophet (an LSTM is benchmarked, not served) |
| Retrieval | FAISS, sentence-transformers, WHO *Clinical Use of Blood* handbook |
| LLM | Groq (`openai/gpt-oss-20b`), used for language only |
| Agents | Deterministic Python blackboard agents; CrewAI for the case pipeline |
| Donor selection | Thompson-sampling multi-armed bandit |
| Backend | FastAPI, SQLAlchemy, PostgreSQL, Redis (optional cache), JWT |
| Frontend | React 18, Vite, Recharts, lucide-react |
| Integration | Model Context Protocol server (HTTP and stdio) |

## Quick start

**Backend** (Python 3.10+; this repo uses a `.venv` on Python 3.11):

```bash
# from the repo root
brew install postgresql@16 redis && brew services start postgresql@16 && brew services start redis
createdb hemosmart   # once

cp backend/.env.example backend/.env   # fill in GROQ_API_KEY, JWT_SECRET_KEY, ADMIN_EMAIL/PASSWORD
.venv/bin/uvicorn backend.main:app --reload --port 8000
```

Run uvicorn from the **repo root**, not from `backend/`: `backend`, `agents`, `rag`, `adapters` and `schema` are sibling top-level packages that only import correctly from there.

Tables are created automatically on first run and an Admin account is bootstrapped from `ADMIN_EMAIL`/`ADMIN_PASSWORD`; log in with `POST /api/auth/login`. `POST /api/auth/register` is Admin-only, so staff accounts are provisioned, never self-registered. `DATABASE_URL` and `REDIS_URL` default to the local services above. Redis is optional: caching degrades gracefully with a logged warning.

**Seed demo data**:

```bash
python -m scripts.seed                 # idempotent: adds whatever is missing
python -m scripts.seed --reset --yes   # wipes ALL data, then re-seeds
python -m scripts.rescore              # after retraining: add fresh predictions, keep history
```

This creates the Admin, one demo account per other role (`manager@`, `staff@`, `coordinator@`, `auditor@hemosmart.app`, password from `DEMO_PASSWORD`, default `demo-pass-123`), a 400-donor pool, a unit-level inventory dated relative to *now* (with a critical O− red-cell shortage and a near-expiry platelet surplus for the loop to find), and 8 sample patients scored by the real models. `--reset` targets whatever `DATABASE_URL` points at, so it refuses to run without `--yes`.

**Frontend**:

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173, talking to the backend at :8000
```

### What each role sees

| Role | Workspace |
|---|---|
| Blood Bank Manager | Supply overview, inventory and demand, agent activity and monitor, prediction, donor outreach, patients, assistant |
| Hospital Staff | Prediction, CBC upload (including the AI-agent case), patients, assistant |
| Donor Coordinator | Donor responses, donor outreach, donors, inventory (read-only), assistant |
| Auditor | Agent activity and monitor (read-only), audit log, patients, assistant |
| Admin | Everything, grouped by area |

## Tests

```bash
pip install -r backend/requirements-dev.txt
python -m pytest        # 285 tests
```

Tests run against a throwaway SQLite database, never your Postgres. They cover the permission matrix for every role and route, the autonomous loop end to end, the agent monitor's events, redaction and alert rules, clinical thresholds, and the donor bandit.

## Evaluation

Each claim is measured by a script in `evaluation/` that writes its tables to `evaluation/results/` (`.md` and `.json`) on a throwaway database. Methods and caveats are in [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md) (D45–D54).

```bash
python -m evaluation.simulate               # 90-day wastage/shortage ablation, 3 seeds (~1.5 min)
python -m evaluation.sensitivity            # outreach window x alert cap sweep (~8 min)
python -m evaluation.prediction_benchmark   # model vs threshold rule, 5-fold CV (~30 s)
python -m evaluation.forecast_backtest      # Prophet vs baselines vs LSTM, rolling origin (~3 min)
python -m evaluation.bandit_regret          # Thompson vs random/greedy/oracle donor selection (~15 s)
python -m evaluation.rag_faithfulness       # retrieval hit@2/MRR; add --llm for faithfulness (20 Groq calls)
python -m evaluation.latency                # NFR-01 response times through the real app (5 Groq calls; --no-llm to skip)
python -m evaluation.figures                # report figures -> docs/figures/
```

<details>
<summary><b>More figures</b> (prediction, forecast, donor selection, sensitivity)</summary>

<p align="center">
  <img src="docs/figures/fig_prediction.png" alt="Prediction: model against a static threshold rule" width="780"><br>
  <img src="docs/figures/fig_forecast.png" alt="Demand forecast against baselines" width="780"><br>
  <img src="docs/figures/fig_bandit.png" alt="Donor selection: Thompson sampling against baselines" width="780"><br>
  <img src="docs/figures/fig_sensitivity.png" alt="Outreach window sensitivity" width="780">
</p>

</details>

### Training the models

All data is synthetic (`data/create_dataset.py`); retrain after changing it:

```bash
python data/create_dataset.py     # patient_data.csv, component_demand.csv
python models/train_xgboost.py    # xgb_components.pkl + per-component SHAP plots
python models/train_prophet.py    # prophet_components.pkl + demand_profile.json
python rag/setup_rag.py           # FAISS index with page metadata
```

Clinical thresholds, doses and urgency rules live in one place, `schema/clinical_thresholds.py`, shared by the label process and the explanation layer so they can never disagree. After retraining, re-seed so stored sample predictions come from the new models.

## Honest limitations

HemoSmart is a research prototype, and it says where it is simulated:

- **All data is synthetic.** Patient labels are simulated clinician decisions (surgery-dependent thresholds, a soft boundary, 1% noise), so prediction metrics measure how well the model recovers that process, not clinical validity.
- **Some thresholds are local protocol, not WHO.** INR > 1.5 for FFP and platelets < 100k for cardiac surgery are not in the handbook the explanations cite, so explanations call them "HemoSmart protocol thresholds".
- **Donor alert dispatch is a print/log stub.** Donor selection and its feedback loop are real; with `HEMOSMART_SIMULATE_DONOR_RESPONSES=1` (default) replies are sampled from each synthetic donor's hidden response rate, which the bandit never sees.
- **The partner hospital is a mock** whose acceptance policy is simulated. Redistribution benefit assumes the partner uses what it accepts.
- **Routine supply** in the gap projection is a fixed share of average demand (`ROUTINE_SUPPLY_SHARE`).
- **RAG evaluation is small**: 4 query families with hand-labelled pages and 10 LLM cases checked by regex.
- **Decision support, not diagnosis.** Predictions and explanations inform a clinician; they never decide.

The full list, with how to revert each design choice, is in [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md).

## MCP server

HemoSmart exposes its own tools (`predict_transfusion`, `explain_prediction`, `check_inventory`, `supply_outlook`, `alert_donors`) as a real [Model Context Protocol](https://modelcontextprotocol.io) server, so any MCP-compatible AI client can use them directly.

- **Remote (HTTP)**: mounted at `/mcp` (`http://localhost:8000/mcp` locally). It requires the same JWT as the REST API, restricted to Admin and Blood Bank Manager, the roles that hold every permission the tools expose. Verified with the real `mcp` client SDK (initialize, list tools, call tool).
- **Local (stdio)**: `python -m mcp_server.stdio_main`, for a client that launches it as a subprocess. It is for a trusted local process and has no auth.

MCP connects LLM clients to tools. It is deliberately *not* used to represent an external hospital system: `integrations/mcp/` is a plain adapter interface for that, demonstrated against a fixture-backed mock hospital by `POST /api/mcp/sync/{hospital_id}`.

## Docker

```bash
docker build -t hemosmart-backend .
docker run -p 8000:8000 \
  -e GROQ_API_KEY=... -e JWT_SECRET_KEY=... -e ADMIN_EMAIL=... -e ADMIN_PASSWORD=... \
  -e DATABASE_URL=postgresql+psycopg://user:pass@host/hemosmart \
  -e REDIS_URL=redis://host:6379/0 \
  hemosmart-backend
```

A single container with no sidecar. The image installs a CPU-only `torch` (every model here runs CPU inference), which keeps it at ~1.3 GB compressed, runs as a non-root user, and bakes in the RAG embedding model so a cold start never downloads anything. The backend needs ~600 MB resident, so free tiers capped at 512 MB (such as Render) will OOM-kill it.

<details>
<summary><b>Testing the container against a local Postgres/Redis</b></summary>

Built and verified with [Colima](https://github.com/abiosoft/colima) (`brew install docker colima && colima start`), a license-free Docker runtime for macOS. Use `host.docker.internal` in place of `localhost` in the connection strings, and make sure Postgres accepts password-authenticated connections from the Colima subnet (`listen_addresses = '*'` in `postgresql.conf`, plus a `host ... scram-sha-256` line in `pg_hba.conf`); local Postgres otherwise only trusts `127.0.0.1`.

</details>

## Deployment

| Piece | Host | Why |
|---|---|---|
| Backend (Docker) | [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker) | Free 16 GB RAM; the backend needs ~600 MB. Sleeps after ~48 h idle. |
| Postgres | [Neon](https://neon.tech) | Free serverless Postgres |
| Redis | [Upstash](https://upstash.com) | Free; optional |
| Frontend | [Vercel](https://vercel.com) | Free static hosting for the Vite build |

**Order matters**, since each step needs a URL from the previous one:

1. **Neon**: create a project, copy the connection string, and change the scheme to `postgresql+psycopg://` (keep `?sslmode=require`). This is `DATABASE_URL`.
2. **Upstash**: create a Redis database and copy its `rediss://default:<password>@<host>:<port>` URL. This is `REDIS_URL`.
3. **Hugging Face**: create a Space (SDK: **Docker**), add it as a git remote and push:
   ```bash
   git remote add space https://huggingface.co/spaces/<user>/<space-name>
   git push space main
   ```
   The `sdk: docker` / `app_port: 8000` frontmatter at the top of this README is what tells Spaces how to run it. Under **Settings → Variables and secrets**, add `DATABASE_URL`, `REDIS_URL`, `GROQ_API_KEY`, `JWT_SECRET_KEY`, `ADMIN_EMAIL`, `ADMIN_PASSWORD` and `DEMO_PASSWORD` as *secrets*. Leave `CORS_ORIGINS` until step 5. The backend URL is `https://<user>-<space-name>.hf.space`.
4. **Vercel**: import the GitHub repo, set **Root Directory** to `frontend`, and add `VITE_API_URL` = the Space URL (no trailing slash). Note the resulting `https://….vercel.app` URL.
5. Back on Hugging Face, add `CORS_ORIGINS` = the Vercel URL and let the Space restart.
6. **Seed**: from your machine, run `DATABASE_URL=<neon url> python -m scripts.seed`.

The first request after the Space has been idle is slow (container wake-up plus model loading); later ones are fast.

## Documentation

| Document | What it covers |
|---|---|
| [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md) | Every design choice (D1–D75): what, why, alternatives, trade-offs, and how to revert |
| [docs/REPORT_ALIGNMENT.md](docs/REPORT_ALIGNMENT.md) | The Phase I report's claims, checked one by one against the code |
| [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) | A ~12-minute viva walkthrough with likely examiner questions and answers |
| [docs/2dvision.md](docs/2dvision.md) | The agent monitor's design: events, redaction, status rules, build plan |

## Notes on configuration

- **LLM provider**: Groq (cloud, free tier) throughout, because a free-tier host can't run a local Ollama server. Set `GROQ_API_KEY` ([console.groq.com](https://console.groq.com)).
- **Storage**: donors, blood units, transfusions, predictions (with components, SHAP, explanation and citations), agent cycles and agent events live in Postgres. Missing tables are created, and missing *nullable* columns added, on startup (`db/database.py`).
- **Agent telemetry**: `HEMOSMART_AGENT_EVENTS=0` turns it off; `HEMOSMART_AGENT_EVENTS_DAYS` sets retention (default 14).
- **Chat sessions and forecast responses** are cached in Redis and degrade gracefully without it.
