"""
Password hashing + JWT creation/verification.

Uses `bcrypt` directly rather than `passlib[bcrypt]` (as originally
scoped) -- passlib 1.7.4's bcrypt backend breaks against bcrypt>=4.1
(it probes a `__about__.__version__` attribute bcrypt no longer ships),
a known incompatibility between the two current package versions.
Calling bcrypt directly is one function each way and avoids the
version conflict entirely.
"""

import os
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt

# The placeholder shipped in backend/.env.example -- copying that file
# verbatim must not leave tokens signed with a publicly known secret.
_PLACEHOLDER_SECRETS = {"", "change_me_to_a_random_64_char_hex_string"}

JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "")
if JWT_SECRET_KEY in _PLACEHOLDER_SECRETS:
    # No hardcoded fallback: anyone who has read this repo could forge
    # an Admin token with it. A random per-process secret is safe; the
    # only cost is that tokens stop working when the server restarts.
    JWT_SECRET_KEY = secrets.token_hex(32)
    print(
        "[auth] JWT_SECRET_KEY is unset or the .env.example placeholder -- "
        "using a random per-process secret. Logins will not survive a "
        "restart; set JWT_SECRET_KEY in backend/.env to fix that."
    )
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 12  # 12 hours


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def create_access_token(subject: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": subject, "role": role, "exp": expire}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Raises jose.JWTError if the token is invalid, tampered with, or expired."""
    return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])


__all__ = ["hash_password", "verify_password", "create_access_token", "decode_access_token", "JWTError"]
