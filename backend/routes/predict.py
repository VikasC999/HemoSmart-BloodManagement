from fastapi import APIRouter, HTTPException

from adapters.manual_adapter import ManualEntryAdapter
from agents.tools import predict_transfusion
from backend.schemas.requests import ManualPatientInput
from schema.patient_schema import PredictionResult

router = APIRouter()


@router.post("/api/predict", response_model=PredictionResult)
def predict(payload: ManualPatientInput):
    adapter = ManualEntryAdapter()
    records = adapter.safe_parse(payload.model_dump())
    if not records:
        raise HTTPException(status_code=422, detail="Could not parse patient data.")

    record = records[0]
    result = predict_transfusion(record)
    return PredictionResult(
        transfusion_needed=result["transfusion_needed"],
        confidence=result["confidence"],
        source_format=record.source_format,
    )
