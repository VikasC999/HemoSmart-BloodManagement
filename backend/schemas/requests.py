"""
Request bodies for the FastAPI layer. These mirror the dict shape
adapters/manual_adapter.py already expects (hemoglobin, platelets,
INR, age, surgery_type, hospital_name) -- kept separate from
schema.patient_schema.PatientRecord because the API accepts the raw
"hospital_name" key, while PatientRecord's field is "source_hospital"
(set by the adapter, not by the caller).
"""

from typing import List, Literal, Optional
from pydantic import BaseModel


class ManualPatientInput(BaseModel):
    hemoglobin: float
    platelets: int
    INR: float
    age: int
    surgery_type: str
    blood_group: Optional[str] = None
    hospital_name: Optional[str] = "unknown"
    # "pdf" when the values were extracted from a report and confirmed in
    # the form (POST /api/predict/pdf/extract), so the record keeps its
    # provenance; anything else typed in is "manual".
    source_format: Literal["manual", "pdf"] = "manual"


class ExplainRequest(BaseModel):
    patient: ManualPatientInput
    prediction: bool
    # When given, the explanation and its citations are saved onto that
    # prediction, so the case record keeps the reasoning it was shown.
    prediction_id: Optional[int] = None


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class ChatPart(BaseModel):
    intent: str
    allowed: bool  # False = refused by the role check, in code (D66)
    text: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    parts: List[ChatPart]
    role: str
    degraded: bool  # True when the LLM was unavailable and a fallback answered
    can_ask_about: List[str]


class DonorAlertRequest(BaseModel):
    blood_type: str
    component: str = "PRBC"
