"""
HemoSmart - Database Connection
-----------------------------------
Single source of truth for the SQLAlchemy engine/session, shared by the
FastAPI backend AND the standalone agent scripts (agents/donor_module.py,
agents/tools.py) -- both used to read/write agents/donors.json and
agents/inventory.json directly; they now read/write Postgres through
this module instead, with no change to their own function signatures.

Local dev: defaults to a local Postgres database (`createdb hemosmart`),
which Homebrew's postgresql@16 sets up with trust auth, so no
username/password is needed. Deployment: set DATABASE_URL to the Neon
connection string (see backend/.env.example) -- nothing else changes.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://localhost/hemosmart",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def init_db() -> None:
    """Creates any missing tables. Safe to call repeatedly (idempotent)."""
    from db import models  # noqa: F401 -- registers models on Base before create_all
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yields a session, closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Tables are created at import time so any entrypoint (FastAPI, a CLI
# script run directly, a one-off migration) has them ready without an
# explicit setup step.
init_db()
