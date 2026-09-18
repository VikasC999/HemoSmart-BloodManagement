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
from backend.services.bootstrap import ensure_admin_user
from db.database import init_db
from rag.explain import ExplanationGenerator

from backend.routes import auth, chat, donors, explain, forecast, inventory, patients, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the XGBoost model/label encoder and build the FAISS-backed
    # explanation generator ONCE at startup, not per-request -- both
    # are expensive to construct (see agents/tools.py, rag/explain.py).
    init_db()  # idempotent; also runs at db/database.py import time
    ensure_admin_user()
    _load_xgb_model()
    app.state.explanation_generator = ExplanationGenerator()
    yield


app = FastAPI(title="HemoSmart API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to the deployed frontend origin later
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(predict.router)
app.include_router(explain.router)
app.include_router(forecast.router)
app.include_router(chat.router)
app.include_router(donors.router)
app.include_router(inventory.router)
app.include_router(patients.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
