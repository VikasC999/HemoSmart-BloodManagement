"""
Creates the first Admin account on startup if none exists yet --
otherwise there's no way to log in at all on a fresh database (every
route needs a token, and /api/auth/register can't hand out Admin to
just anyone). Credentials come from env vars so the real password
never lives in source control; the fallback default is loud on
purpose so it's obvious in logs that it needs changing.
"""

import os
import secrets

from backend.services.auth import hash_password
from db.database import SessionLocal
from db.models import User


def ensure_admin_user() -> None:
    admin_email = os.environ.get("ADMIN_EMAIL", "admin@hemosmart.local")
    admin_password = os.environ.get("ADMIN_PASSWORD", "")

    with SessionLocal() as db:
        if db.query(User).filter(User.role == "Admin").first():
            return

        if admin_password in ("", "change-me-admin"):
            # Never fall back to a password that's printed in this repo.
            admin_password = secrets.token_urlsafe(12)
            print(f"[bootstrap] ADMIN_PASSWORD unset -- generated one-time password: {admin_password}")

        db.add(User(
            email=admin_email,
            hashed_password=hash_password(admin_password),
            role="Admin",
        ))
        db.commit()
        print(
            f"[bootstrap] Created initial Admin account ({admin_email}). "
            f"Set ADMIN_EMAIL/ADMIN_PASSWORD env vars to override the default credentials."
        )
