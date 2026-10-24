from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from agents.tools import list_recent_patients, prediction_to_dict
from backend.dependencies.rbac import require_permission
from db.database import get_db
from db.models import Patient, Prediction

router = APIRouter()
allowed_roles = require_permission("patients:read")


class PredictionOut(BaseModel):
    id: int
    transfusion_needed: bool
    confidence: float
    urgency: Optional[str] = None
    components: Optional[list] = None
    explanation: Optional[str] = None
    citations: Optional[list] = None
    created_at: str

    model_config = ConfigDict(from_attributes=True)


class PatientOut(BaseModel):
    id: int
    hemoglobin: float
    platelets: int
    inr: float
    age: int
    surgery_type: str
    blood_group: Optional[str]
    source_format: str
    source_hospital: str
    created_at: str
    latest_prediction: Optional[PredictionOut] = None

    model_config = ConfigDict(from_attributes=True)


@router.get("/api/patients", response_model=List[PatientOut], dependencies=[Depends(allowed_roles)])
def list_patients(limit: int = 50, db: Session = Depends(get_db)):
    """Most recent patients first -- backs the Day 6 case-list view so
    predictions persist across page reloads instead of vanishing."""
    return list_recent_patients(limit, db=db)


@router.get(
    "/api/patients/{patient_id}/predictions",
    response_model=List[PredictionOut],
    dependencies=[Depends(allowed_roles)],
)
def patient_predictions(patient_id: int, db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found.")

    return [_prediction_out(pr) for pr in patient.predictions]


def _prediction_out(pr: Prediction) -> PredictionOut:
    return PredictionOut(**prediction_to_dict(pr))
