"""
Shared patient/prediction persistence, used by both the direct predict
routes (backend/routes/predict.py) and the MCP-pattern hospital sync
route (backend/routes/mcp_sync.py) -- every intake path ends up here so
a prediction is recorded the same way regardless of where the patient
data came from.
"""

from typing import Optional

from sqlalchemy.orm import Session

from db.models import Patient, Prediction
from schema.patient_schema import PatientRecord


def persist_prediction(db: Session, record: PatientRecord, result: dict, created_by: Optional[int] = None) -> None:
    """Failures here are logged, not raised -- a DB hiccup shouldn't
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
            created_by=created_by,
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
        print(f"[persist] Failed to persist patient/prediction: {exc}")
