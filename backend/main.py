"""
HemoSmart FastAPI backend.

Run from the REPO ROOT (not from inside backend/), so that `backend`,
`agents`, `rag`, `adapters`, and `schema` are all importable as
top-level sibling packages, matching the sys.path convention those
modules already assume internally:

    uvicorn backend.main:app --reload --port 8000
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Loads backend/.env if present, before any module below reads os.environ.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agents import telemetry
from agents.autonomy import scheduler
from agents.autonomy.orchestrator import mark_orphaned_cycles
from agents.tools import _load_models, set_explanation_generator
from backend.middleware.audit import AuditLogMiddleware
from backend.middleware.mcp_auth import MCPAuthMiddleware
from backend.services.bootstrap import ensure_admin_user
from db.database import init_db
from mcp_server.server import mcp as mcp_server
from rag.explain import ExplanationGenerator

from backend.routes import (
    agents_monitor, audit, auth, autonomy, cases, chat, donors, explain, forecast, inventory, mcp_sync, patients, predict,
    transfusions,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the XGBoost models/label encoder and build the FAISS-backed
    # explanation generator ONCE at startup, not per-request -- both
    # are expensive to construct (see agents/tools.py, rag/explain.py).
    init_db()  # idempotent; also runs at db/database.py import time
    ensure_admin_user()
    _load_models()
    app.state.explanation_generator = ExplanationGenerator()
    set_explanation_generator(app.state.explanation_generator)
    # A crash mid-run leaves a cycle "running" and its telemetry open
    # forever; close both out as failed before the monitor reads them.
    mark_orphaned_cycles()
    telemetry.close_open_runs()
    telemetry.prune()
    # The autonomous wastage-reduction loop (agents/autonomy/) on a timer.
    autonomy_task = scheduler.start(app)
    # The mounted MCP app (see below) has its own lifespan that starts
    # its session manager -- mounting a sub-app's routes doesn't run its
    # lifespan automatically, so it's entered explicitly here instead.
    try:
        async with mcp_server.session_manager.run():
            yield
    finally:
        if autonomy_task:
            autonomy_task.cancel()


app = FastAPI(title="HemoSmart API", lifespan=lifespan)

# Comma-separated list of allowed frontend origins, e.g.
# "https://hemosmart.vercel.app". Unset means the local Vite dev server
# only -- never "*", so a forgotten setting fails closed, not open.
_DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"
_cors_origins = [
    o.strip() for o in os.environ.get("CORS_ORIGINS", _DEFAULT_CORS_ORIGINS).split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AuditLogMiddleware)

app.include_router(audit.router)
app.include_router(auth.router)
app.include_router(predict.router)
app.include_router(explain.router)
app.include_router(forecast.router)
app.include_router(chat.router)
app.include_router(donors.router)
app.include_router(inventory.router)
app.include_router(patients.router)
app.include_router(transfusions.router)
app.include_router(autonomy.router)
app.include_router(cases.router)
app.include_router(mcp_sync.router)
app.include_router(agents_monitor.router)

# Exposes predict/explain/check_inventory/alert_donors as MCP tools at
# /mcp, so any MCP-compatible client (Claude Desktop, Claude Code, a
# remote agent) can connect to this backend directly -- with the same
# JWT the REST API uses, restricted to roles holding every permission
# those tools expose (see backend/middleware/mcp_auth.py).
app.mount("/mcp", MCPAuthMiddleware(mcp_server.streamable_http_app()))


@app.get("/api/health")
def health():
    return {"status": "ok"}
