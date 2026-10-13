import asyncio

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from backend.dependencies.rbac import require_role
from backend.middleware.mcp_auth import MCPAuthMiddleware
from backend.routes import auth
from conftest import auth_header

app = FastAPI()
app.include_router(auth.router)


@app.get("/manager-only", dependencies=[Depends(require_role("Blood Bank Manager"))])
def manager_only():
    return {"ok": True}


client = TestClient(app)
NEW_USER = {"email": "new@hemosmart.app", "password": "pw", "role": "Admin"}


def test_register_requires_a_token(users):
    assert client.post("/api/auth/register", json=NEW_USER).status_code == 401


@pytest.mark.parametrize("role", ["Blood Bank Manager", "Hospital Staff", "Donor Coordinator", "Auditor"])
def test_register_refuses_non_admins(users, role):
    # the old privilege escalation: anyone could self-register as Admin
    res = client.post("/api/auth/register", json=NEW_USER, headers=auth_header(users[role]))
    assert res.status_code == 403


def test_admin_can_provision_users(users):
    res = client.post("/api/auth/register", json=NEW_USER, headers=auth_header(users["Admin"]))
    assert res.status_code == 200
    login = client.post("/api/auth/login", data={"username": NEW_USER["email"], "password": "pw"})
    assert login.status_code == 200


def test_login_rejects_wrong_password(users):
    res = client.post("/api/auth/login", data={"username": "admin@hemosmart.app", "password": "nope"})
    assert res.status_code == 401


def test_tampered_token_rejected(users):
    assert client.get("/api/auth/me", headers=auth_header(users["Admin"] + "x")).status_code == 401


@pytest.mark.parametrize("role, status", [
    ("Blood Bank Manager", 200), ("Admin", 200), ("Hospital Staff", 403), ("Auditor", 403),
])
def test_require_role(users, role, status):
    assert client.get("/manager-only", headers=auth_header(users[role])).status_code == status


# ----------------------------------------------------------------------
# /mcp guard
# ----------------------------------------------------------------------
def _call_mcp_guard(headers: dict) -> int:
    """Runs one HTTP request through MCPAuthMiddleware in front of a stub
    app that always answers 200. Returns the status the client saw."""
    async def stub_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b""}

    scope = {
        "type": "http", "method": "POST", "path": "/",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    }
    asyncio.run(MCPAuthMiddleware(stub_app)(scope, receive, send))
    return sent[0]["status"]


def test_mcp_rejects_missing_token(users):
    assert _call_mcp_guard({}) == 401


def test_mcp_rejects_invalid_token(users):
    assert _call_mcp_guard(auth_header("not-a-jwt")) == 401


@pytest.mark.parametrize("role, status", [
    ("Admin", 200), ("Blood Bank Manager", 200),
    ("Hospital Staff", 403), ("Donor Coordinator", 403), ("Auditor", 403),
])
def test_mcp_role_gate(users, role, status):
    assert _call_mcp_guard(auth_header(users[role])) == status
