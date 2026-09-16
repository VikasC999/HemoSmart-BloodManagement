from fastapi import APIRouter, HTTPException, Request

from adapters.manual_adapter import ManualEntryAdapter
from backend.schemas.requests import ExplainRequest
from rag.explain import check_thresholds

router = APIRouter()


@router.post("/api/explain")
def explain(payload: ExplainRequest, request: Request):
    adapter = ManualEntryAdapter()
    records = adapter.safe_parse(payload.patient.model_dump())
    if not records:
        raise HTTPException(status_code=422, detail="Could not parse patient data.")

    record = records[0]
    threshold_summary = check_thresholds(record)

    generator = request.app.state.explanation_generator
    try:
        llm_explanation = generator.explain(record, payload.prediction)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM explanation failed ({exc.__class__.__name__}): {exc}",
        )

    return {
        "threshold_summary": threshold_summary,
        "llm_explanation": llm_explanation,
    }
