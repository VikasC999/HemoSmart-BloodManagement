from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.dependencies.rbac import require_role
from db.database import get_db
from db.models import Patient, Prediction

router = APIRouter()
allowed_roles = require_role("Hospital Staff", "Blood Bank Manager", "Auditor")


class PredictionOut(BaseModel):
    id: int
    transfusion_needed: bool
    confidence: float
    created_at: str

    class Config:
        from_attributes = True


class PatientOut(BaseModel):
    id: int
    hemoglobin: float
    platelets: int
    inr: float
    age: int
    surgery_type: str
    source_format: str
    source_hospital: str
    created_at: str
    latest_prediction: Optional[PredictionOut] = None

    class Config:
        from_attributes = True


@router.get("/api/patients", response_model=List[PatientOut], dependencies=[Depends(allowed_roles)])
def list_patients(limit: int = 50, db: Session = Depends(get_db)):
    """Most recent patients first -- backs the Day 6 case-list view so
    predictions persist across page reloads instead of vanishing."""
    patients = (
        db.query(Patient)
        .order_by(Patient.created_at.desc())
        .limit(limit)
        .all()
    )
    out = []
    for p in patients:
        latest = p.predictions[-1] if p.predictions else None
        out.append(PatientOut(
            id=p.id,
            hemoglobin=p.hemoglobin,
            platelets=p.platelets,
            inr=p.inr,
            age=p.age,
            surgery_type=p.surgery_type,
            source_format=p.source_format,
            source_hospital=p.source_hospital,
            created_at=p.created_at.isoformat(),
            latest_prediction=(
                PredictionOut(
                    id=latest.id,
                    transfusion_needed=latest.transfusion_needed,
                    confidence=latest.confidence,
                    created_at=latest.created_at.isoformat(),
                )
                if latest else None
            ),
        ))
    return out


@router.get(
    "/api/patients/{patient_id}/predictions",
    response_model=List[PredictionOut],
    dependencies=[Depends(allowed_roles)],
)
def patient_predictions(patient_id: int, db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found.")

    return [
        PredictionOut(
            id=pr.id,
            transfusion_needed=pr.transfusion_needed,
            confidence=pr.confidence,
            created_at=pr.created_at.isoformat(),
        )
        for pr in patient.predictions
    ]
