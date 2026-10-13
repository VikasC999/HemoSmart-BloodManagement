"""
Bearer-token guard for the HTTP-mounted MCP server (/mcp).

The MCP tools (predict, explain, check inventory, alert donors) are the
same operations the REST API protects with RBAC, so the MCP path must
not be a way around it. An MCP client authenticates exactly like the
frontend: log in via POST /api/auth/login and send the token as
`Authorization: Bearer <token>`.

Only roles that hold EVERY permission the MCP tools expose may connect
-- Blood Bank Manager (predict + inventory + donor alerts) and Admin.
MCP authorizes per connection, not per tool call, so a narrower role
would otherwise reach tools its REST equivalent would refuse.

Pure ASGI (not BaseHTTPMiddleware) so MCP's streamed responses pass
through untouched once a request is allowed.
"""

import json

from jose import JWTError

from backend.services.auth import decode_access_token
from db.database import SessionLocal
from db.models import User

MCP_ALLOWED_ROLES = {"Admin", "Blood Bank Manager"}


async def _reject(send, status: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode()
    headers = [(b"content-type", b"application/json")]
    if status == 401:
        headers.append((b"www-authenticate", b"Bearer"))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


def _role_for_token(token: str):
    """Returns the caller's CURRENT role from the database, or None if the
    token is invalid or its user no longer exists. The role claim inside
    the token isn't trusted on its own -- a demoted user's old token
    must stop working here too."""
    try:
        email = decode_access_token(token).get("sub")
    except JWTError:
        return None
    if not email:
        return None
    with SessionLocal() as db:
        user = db.query(User).filter(User.email == email).first()
        return user.role if user else None


class MCPAuthMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        headers = dict(scope.get("headers") or [])
        auth = headers.get(b"authorization", b"").decode()
        if not auth.lower().startswith("bearer "):
            return await _reject(send, 401, "Missing bearer token.")

        role = _role_for_token(auth[7:])
        if role is None:
            return await _reject(send, 401, "Could not validate credentials.")
        if role not in MCP_ALLOWED_ROLES:
            return await _reject(send, 403, f"Role '{role}' is not permitted to use the MCP server.")

        return await self.app(scope, receive, send)
