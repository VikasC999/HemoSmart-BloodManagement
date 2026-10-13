"""
Points the app at a throwaway SQLite database BEFORE anything imports
db.database (which builds its engine and creates tables at import time),
so the test suite never touches the local Postgres demo data.
"""

import os
import tempfile

_DB_DIR = tempfile.mkdtemp(prefix="hemosmart-tests-")
os.environ["DATABASE_URL"] = f"sqlite:///{os.path.join(_DB_DIR, 'test.db')}"
os.environ["JWT_SECRET_KEY"] = "test-secret"

import pytest  # noqa: E402

from backend.services.auth import create_access_token, hash_password  # noqa: E402
from db.database import SessionLocal  # noqa: E402
from db.models import AuditLog, User  # noqa: E402

ROLES = ("Admin", "Blood Bank Manager", "Hospital Staff", "Donor Coordinator", "Auditor")


@pytest.fixture
def users():
    """One user per role. Returns {role: bearer token}."""
    with SessionLocal() as db:
        db.query(AuditLog).delete()
        db.query(User).delete()
        for role in ROLES:
            db.add(User(email=_email(role), hashed_password=hash_password("pw"), role=role))
        db.commit()
    yield {role: create_access_token(_email(role), role) for role in ROLES}
    with SessionLocal() as db:
        db.query(AuditLog).delete()
        db.query(User).delete()
        db.commit()


def _email(role: str) -> str:
    return role.lower().replace(" ", "-") + "@hemosmart.app"


def auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}
