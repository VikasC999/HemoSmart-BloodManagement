"""
Per-session ChatSession storage for the /api/chat endpoint.

agents.chat_agent.ChatSession was designed for a single-user CLI loop
(agents/chat_agent.py's run_chat()) and has no concept of a session ID
on its own. This module is the thin layer that gives it one, so a
multi-user FastAPI backend can hold one ChatSession per conversation.

Day 1: plain in-memory dict, lost on restart -- fine for a local demo.
Day 4+: swap the internals for Redis (Upstash) without changing the
two functions below, so routes/chat.py never needs to change.
"""

import uuid
from typing import Optional, Tuple

from agents.chat_agent import ChatSession

_sessions: dict[str, ChatSession] = {}


def get_or_create_session(session_id: Optional[str]) -> Tuple[str, ChatSession]:
    if session_id and session_id in _sessions:
        return session_id, _sessions[session_id]

    new_id = session_id or str(uuid.uuid4())
    session = ChatSession()
    _sessions[new_id] = session
    return new_id, session
