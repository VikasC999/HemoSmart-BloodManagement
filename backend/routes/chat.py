"""
The assistant, for every role (FR-08). What each role may ask is
decided in code by agents/chat_agent.py against schema/permissions.py;
this route ties the conversation to the logged-in user and writes one
structured audit record per message (D68). The message text itself is
not stored, only what was asked about and whether it was allowed.
"""

from fastapi import APIRouter, Depends, Request

from backend.dependencies.rbac import require_permission
from backend.schemas.requests import ChatRequest, ChatResponse
from backend.services.session_store import get_or_create_session, save_session
from db.models import User
from schema.permissions import chat_topics_for, suggestions_for

router = APIRouter()
can_chat = require_permission("chat:use")


@router.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request, user: User = Depends(can_chat)):
    session_id, session = get_or_create_session(payload.session_id, user)
    result = session.handle(payload.message)
    save_session(session_id, session, user)

    request.state.audit_action = "chat_query"
    request.state.audit_resource_type = "chat"
    request.state.audit_resource_id = session_id
    request.state.audit_details = {
        "role": user.role,
        "intents": result["intents"],
        "decisions": result["decisions"],
        "outcome": result["outcome"],
        "llm_used": result["llm_used"],
        "degraded": result["degraded"],
        "injection_suspected": result["injection_suspected"],
        "action_taken": result["action_taken"],
        "message_chars": len(payload.message),
    }
    return {"session_id": session_id, **result}


@router.get("/api/chat/capabilities")
def capabilities(user: User = Depends(can_chat)):
    """What this role can ask, for the assistant's empty state."""
    return {"role": user.role, "can_ask_about": chat_topics_for(user.role),
            "suggestions": suggestions_for(user.role)}
