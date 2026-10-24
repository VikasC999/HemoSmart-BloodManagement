"""
Per-session ChatSession storage for the /api/chat endpoint.

agents.chat_agent.ChatSession was designed for a single-user CLI loop
(agents/chat_agent.py's run_chat()) and has no concept of a session ID
on its own. This module is the thin layer that gives it one, so a
multi-user FastAPI backend can hold one conversation's state per
session ID.

Day 1: plain in-memory dict, lost on restart. Day 4: Redis -- but a
ChatSession object itself isn't serializable (it holds a live LLM
client), so what's actually persisted is just its state (last_patient,
last_prediction, a donor alert awaiting confirmation). A fresh
ChatSession is constructed per request and that state is restored onto
it before `.handle()` runs, then saved back after.

Sessions belong to a user (D67): the key includes the user's id, so a
session_id copied from someone else finds nothing, and the stored role
must match the user's current role or the state is dropped -- a pending
donor alert can't survive a role change.
"""

import json
import uuid
from typing import Optional, Tuple

from agents.chat_agent import ChatSession
from backend.services.cache import cache_get, cache_set
from schema.patient_schema import PatientRecord

SESSION_TTL_SECONDS = 60 * 60 * 2  # 2 hours


def _key(user_id: int, session_id: str) -> str:
    return f"chat_session:{user_id}:{session_id}"


def get_or_create_session(session_id: Optional[str], user) -> Tuple[str, ChatSession]:
    new_id = session_id or str(uuid.uuid4())
    session = ChatSession(role=user.role)

    raw = cache_get(_key(user.id, new_id))
    if raw:
        data = json.loads(raw)
        if data.get("role") == user.role:
            if data.get("last_patient"):
                session.last_patient = PatientRecord(**data["last_patient"])
            session.last_prediction = data.get("last_prediction")
            session.pending_action = data.get("pending_action")

    return new_id, session


def save_session(session_id: str, session: ChatSession, user) -> None:
    data = {
        "role": user.role,
        "last_patient": session.last_patient.model_dump() if session.last_patient else None,
        "last_prediction": session.last_prediction,
        "pending_action": session.pending_action,
    }
    cache_set(_key(user.id, session_id), json.dumps(data), SESSION_TTL_SECONDS)
