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

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://localhost/hemosmart",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def init_db() -> None:
    """Creates any missing tables and columns. Safe to call repeatedly."""
    from db import models  # noqa: F401 -- registers models on Base before create_all
    Base.metadata.create_all(bind=engine)
    _add_missing_columns()


def _add_missing_columns() -> None:
    """
    create_all() only creates missing TABLES -- it never alters an
    existing one, so a column added to a model later would be missing
    from any database created before it. This adds such columns.

    A deliberately small stand-in for a migration tool (Alembic): it only
    ever ADDS nullable columns, never alters or drops anything, so it
    can't damage existing data.
    """
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue
            existing = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing:
                    continue
                if not column.nullable:
                    raise RuntimeError(
                        f"Can't auto-add non-nullable column {table.name}.{column.name}; "
                        f"make it nullable or migrate this database by hand."
                    )
                ddl_type = column.type.compile(dialect=engine.dialect)
                conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN {column.name} {ddl_type}'))
                print(f"[db] Added column {table.name}.{column.name}")


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
