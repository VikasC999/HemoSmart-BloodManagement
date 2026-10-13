from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from adapters.manual_adapter import ManualEntryAdapter
from backend.dependencies.rbac import require_role
from backend.schemas.requests import ExplainRequest
from db.database import get_db
from db.models import Prediction

router = APIRouter()


@router.post("/api/explain", dependencies=[Depends(require_role("Hospital Staff", "Blood Bank Manager"))])
def explain(payload: ExplainRequest, request: Request, db: Session = Depends(get_db)):
    adapter = ManualEntryAdapter()
    records = adapter.safe_parse(payload.patient.model_dump())
    if not records:
        raise HTTPException(status_code=422, detail="Could not parse patient data.")

    record = records[0]
    generator = request.app.state.explanation_generator
    try:
        result = generator.explain_with_sources(record, payload.prediction)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM explanation failed ({exc.__class__.__name__}): {exc}",
        )

    if payload.prediction_id is not None:
        prediction = db.query(Prediction).filter(Prediction.id == payload.prediction_id).first()
        if prediction is not None:
            prediction.explanation = result["explanation"]
            prediction.citations = result["citations"]
            db.commit()

    return {
        "threshold_summary": result["threshold_facts"],
        "llm_explanation": result["explanation"],
        "citations": result["citations"],
    }
