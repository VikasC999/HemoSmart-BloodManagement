"""
Audit logging middleware.

Writes one audit_log row per /api/* request: method, path, status,
timestamp, and the calling user (resolved from the bearer token, if
any -- an unauthenticated call, e.g. a failed login, still logs with a
null user_id rather than being skipped).

Routes that touch something worth a structured record (a prediction,
a donor alert, an inventory change) enrich that same row instead of
a second one being written: set request.state.audit_action /
audit_resource_type / audit_resource_id / audit_details before
returning, and the middleware picks them up after call_next() runs.

Never blocks or fails the request -- a logging failure is a bug in
observability, not a reason to break the actual operation.
"""

from typing import Optional

from fastapi import Request
from jose import JWTError
from starlette.middleware.base import BaseHTTPMiddleware

from backend.services.auth import decode_access_token
from db.database import SessionLocal
from db.models import AuditLog, User


def _resolve_user_id(request: Request) -> Optional[int]:
    auth_header = request.headers.get("authorization", "")
    if not auth_header.lower().startswith("bearer "):
        return None

    try:
        payload = decode_access_token(auth_header[7:])
    except JWTError:
        return None

    email = payload.get("sub")
    if not email:
        return None

    with SessionLocal() as db:
        user = db.query(User).filter(User.email == email).first()
        return user.id if user else None


class AuditLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not request.url.path.startswith("/api/"):
            return await call_next(request)

        response = await call_next(request)

        try:
            user_id = _resolve_user_id(request)
            action = getattr(request.state, "audit_action", f"{request.method} {request.url.path}")

            with SessionLocal() as db:
                db.add(AuditLog(
                    user_id=user_id,
                    action=action,
                    resource_type=getattr(request.state, "audit_resource_type", None),
                    resource_id=getattr(request.state, "audit_resource_id", None),
                    ip_address=request.client.host if request.client else None,
                    details=getattr(request.state, "audit_details", None),
                ))
                db.commit()
        except Exception as exc:
            print(f"[audit] Failed to log request: {exc}")

        return response
