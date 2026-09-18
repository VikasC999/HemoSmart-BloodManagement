import os
import tempfile
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from adapters.csv_adapter import CSVExportAdapter
from adapters.manual_adapter import ManualEntryAdapter
from adapters.pdf_adapter import PDFReportAdapter
from agents.tools import predict_transfusion
from backend.schemas.requests import ManualPatientInput
from db.database import get_db
from db.models import Patient, Prediction
from rag.llm_client import get_llm_client
from schema.patient_schema import PatientRecord, PredictionResult

router = APIRouter()


def _persist(db: Session, record: PatientRecord, result: dict) -> None:
    """Saves the patient + prediction so it survives past this request
    (Day 1's predictions vanished on restart -- there was nowhere to put
    them). Failures here are logged, not raised -- a DB hiccup shouldn't
    block a clinician from getting their prediction back."""
    try:
        patient_row = Patient(
            hemoglobin=record.hemoglobin,
            platelets=record.platelets,
            inr=record.INR,
            age=record.age,
            surgery_type=record.surgery_type,
            source_format=record.source_format,
            source_hospital=record.source_hospital,
        )
        db.add(patient_row)
        db.flush()  # assigns patient_row.id without committing yet
        db.add(Prediction(
            patient_id=patient_row.id,
            transfusion_needed=result["transfusion_needed"],
            confidence=result["confidence"],
        ))
        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"[predict] Failed to persist patient/prediction: {exc}")


@router.post("/api/predict", response_model=PredictionResult)
def predict(payload: ManualPatientInput, db: Session = Depends(get_db)):
    adapter = ManualEntryAdapter()
    records = adapter.safe_parse(payload.model_dump())
    if not records:
        raise HTTPException(status_code=422, detail="Could not parse patient data.")

    record = records[0]
    result = predict_transfusion(record)
    _persist(db, record, result)

    return PredictionResult(
        transfusion_needed=result["transfusion_needed"],
        confidence=result["confidence"],
        source_format=record.source_format,
    )


@router.post("/api/predict/pdf", response_model=PredictionResult)
async def predict_from_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
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
    _persist(db, record, result)

    return PredictionResult(
        transfusion_needed=result["transfusion_needed"],
        confidence=result["confidence"],
        source_format=record.source_format,
    )


@router.post("/api/predict/csv", response_model=List[PredictionResult])
async def predict_from_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
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
        _persist(db, record, result)
        results.append(PredictionResult(
            transfusion_needed=result["transfusion_needed"],
            confidence=result["confidence"],
            source_format=record.source_format,
        ))
    return results
