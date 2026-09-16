from fastapi import APIRouter, HTTPException

from backend.schemas.requests import ChatRequest
from backend.services.session_store import get_or_create_session

router = APIRouter()


@router.post("/api/chat")
def chat(payload: ChatRequest):
    session_id, session = get_or_create_session(payload.session_id)
    try:
        reply = session.handle(payload.message)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Chat agent failed ({exc.__class__.__name__}): {exc}",
        )
    return {"session_id": session_id, "reply": reply}
