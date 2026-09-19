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

from agents.tools import _load_xgb_model
from backend.middleware.audit import AuditLogMiddleware
from backend.services.bootstrap import ensure_admin_user
from db.database import init_db
from mcp_server.server import mcp as mcp_server
from rag.explain import ExplanationGenerator

from backend.routes import audit, auth, chat, donors, explain, forecast, inventory, mcp_sync, patients, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the XGBoost model/label encoder and build the FAISS-backed
    # explanation generator ONCE at startup, not per-request -- both
    # are expensive to construct (see agents/tools.py, rag/explain.py).
    init_db()  # idempotent; also runs at db/database.py import time
    ensure_admin_user()
    _load_xgb_model()
    app.state.explanation_generator = ExplanationGenerator()
    # The mounted MCP app (see below) has its own lifespan that starts
    # its session manager -- mounting a sub-app's routes doesn't run its
    # lifespan automatically, so it's entered explicitly here instead.
    async with mcp_server.session_manager.run():
        yield


app = FastAPI(title="HemoSmart API", lifespan=lifespan)

# Comma-separated list of allowed frontend origins, e.g.
# "https://hemosmart.vercel.app". Unset means "*", which is fine for
# local dev but should always be set in production.
_cors_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]

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
app.include_router(mcp_sync.router)

# Exposes predict/explain/check_inventory/alert_donors as MCP tools at
# /mcp, so any MCP-compatible client (Claude Desktop, Claude Code, a
# remote agent) can connect to this deployed backend directly. No auth
# on this path today -- a known simplification, consistent with how
# the rest of this project names what's simplified rather than hiding
# it (see README's "what's simulated vs real").
app.mount("/mcp", mcp_server.streamable_http_app())


@app.get("/api/health")
def health():
    return {"status": "ok"}
