import os
import tempfile
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from adapters.csv_adapter import CSVExportAdapter
from adapters.manual_adapter import ManualEntryAdapter
from adapters.pdf_adapter import PDFReportAdapter
from agents.tools import predict_transfusion
from backend.dependencies.rbac import require_role
from backend.schemas.requests import ManualPatientInput
from backend.services.persistence import persist_prediction as _persist
from db.database import get_db
from db.models import User
from rag.llm_client import get_llm_client
from schema.patient_schema import PredictionResult

router = APIRouter()
allowed_roles = require_role("Hospital Staff", "Blood Bank Manager")


@router.post("/api/predict", response_model=PredictionResult)
def predict(
    payload: ManualPatientInput, request: Request,
    db: Session = Depends(get_db), user: User = Depends(allowed_roles),
):
    adapter = ManualEntryAdapter()
    records = adapter.safe_parse(payload.model_dump())
    if not records:
        raise HTTPException(status_code=422, detail="Could not parse patient data.")

    record = records[0]
    result = predict_transfusion(record)
    _persist(db, record, result, created_by=user.id)

    request.state.audit_action = "prediction_created"
    request.state.audit_resource_type = "prediction"
    request.state.audit_details = {
        "surgery_type": record.surgery_type,
        "source_format": record.source_format,
        "transfusion_needed": result["transfusion_needed"],
        "confidence": result["confidence"],
    }

    return PredictionResult(
        transfusion_needed=result["transfusion_needed"],
        confidence=result["confidence"],
        source_format=record.source_format,
    )


@router.post("/api/predict/pdf", response_model=PredictionResult)
async def predict_from_pdf(
    request: Request, file: UploadFile = File(...),
    db: Session = Depends(get_db), user: User = Depends(allowed_roles),
):
    """Upload a CBC lab report PDF -- extracted via PyMuPDF + LLM into a
    validated PatientRecord, then predicted exactly like manual entry."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        llm = get_llm_client(provider=os.environ.get("HEMOSMART_LLM_PROVIDER", "groq"))
        adapter = PDFReportAdapter(llm_client=llm)
        records = adapter.safe_parse(tmp_path)
    finally:
        os.unlink(tmp_path)

    if not records:
        raise HTTPException(
            status_code=422,
            detail="Could not extract patient data from this PDF. Try manual entry instead.",
        )

    record = records[0]
    result = predict_transfusion(record)
    _persist(db, record, result, created_by=user.id)

    request.state.audit_action = "prediction_created"
    request.state.audit_resource_type = "prediction"
    request.state.audit_details = {
        "surgery_type": record.surgery_type,
        "source_format": record.source_format,
        "transfusion_needed": result["transfusion_needed"],
        "confidence": result["confidence"],
    }

    return PredictionResult(
        transfusion_needed=result["transfusion_needed"],
        confidence=result["confidence"],
        source_format=record.source_format,
    )


@router.post("/api/predict/csv", response_model=List[PredictionResult])
async def predict_from_csv(
    request: Request, file: UploadFile = File(...),
    db: Session = Depends(get_db), user: User = Depends(allowed_roles),
):
    """Upload a hospital's bulk CSV export -- one prediction per row,
    since a CSV export is naturally a batch of patients (unlike a PDF,
    which is always one patient's report)."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        adapter = CSVExportAdapter()
        records = adapter.safe_parse(tmp_path)
    finally:
        os.unlink(tmp_path)

    if not records:
        raise HTTPException(
            status_code=422,
            detail="Could not parse any valid rows from this CSV.",
        )

    results = []
    for record in records:
        result = predict_transfusion(record)
        _persist(db, record, result, created_by=user.id)
        results.append(PredictionResult(
            transfusion_needed=result["transfusion_needed"],
            confidence=result["confidence"],
            source_format=record.source_format,
        ))

    request.state.audit_action = "prediction_created_batch"
    request.state.audit_resource_type = "prediction"
    request.state.audit_details = {
        "source_format": "csv",
        "count": len(results),
        "transfusion_needed_count": sum(1 for r in results if r.transfusion_needed),
    }

    return results
