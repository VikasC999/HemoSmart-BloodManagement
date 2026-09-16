"""
Request bodies for the FastAPI layer. These mirror the dict shape
adapters/manual_adapter.py already expects (hemoglobin, platelets,
INR, age, surgery_type, hospital_name) -- kept separate from
schema.patient_schema.PatientRecord because the API accepts the raw
"hospital_name" key, while PatientRecord's field is "source_hospital"
(set by the adapter, not by the caller).
"""

from typing import Optional
from pydantic import BaseModel


class ManualPatientInput(BaseModel):
    hemoglobin: float
    platelets: int
    INR: float
    age: int
    surgery_type: str
    hospital_name: Optional[str] = "unknown"


class ExplainRequest(BaseModel):
    patient: ManualPatientInput
    prediction: bool


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class DonorAlertRequest(BaseModel):
    blood_type: str
