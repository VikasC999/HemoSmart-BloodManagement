"""
HemoSmart - Canonical Patient Schema
----------------------------------------
This is THE single source of truth for what a "patient record" looks like
inside HemoSmart, regardless of which hospital or format it came from.

Every input format (PDF report, manual entry, CSV export, future HL7/FHIR
feed) must be converted into this schema by an adapter BEFORE it reaches
the ML model, the RAG engine, or the backend API.

    Hospital A (HL7)  ---\
    Hospital B (PDF)  ----\
    Hospital C (CSV)  -----> adapters/*.py --> PatientRecord --> XGBoost model
    Hospital D (API)  ----/
    Manual entry form -/

Nothing downstream should ever read a raw hospital format directly.
If you need a new hospital format supported, write ONE new adapter that
outputs a PatientRecord -- nothing else in the system needs to change.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional

from bloodbank.catalog import BLOOD_GROUPS


VALID_SURGERY_TYPES = ("Cardiac", "Orthopedic", "General", "Emergency")


class PatientRecord(BaseModel):
    """
    The canonical, validated shape of a single patient's data.
    All fields are required and range-checked against real physiological
    limits, so a bad adapter (or bad OCR read) fails loudly here instead
    of silently poisoning a prediction later.
    """

    hemoglobin: float = Field(..., ge=2.0, le=22.0,
                               description="g/dL")
    platelets: int = Field(..., ge=1000, le=1_000_000,
                            description="per microliter")
    INR: float = Field(..., ge=0.5, le=10.0,
                        description="International Normalized Ratio")
    age: int = Field(..., ge=0, le=120)
    surgery_type: str
    # ABO/RhD group -- not a model feature (it doesn't change whether a
    # patient needs blood), but required to allocate compatible units.
    blood_group: Optional[str] = None

    # Optional metadata -- not used by the model, but useful for
    # traceability (which hospital/format this record came from).
    source_format: str = Field(default="unknown",
                                description="e.g. 'pdf', 'manual', 'csv'")
    source_hospital: str = Field(default="unknown")

    @field_validator("platelets", mode="before")
    @classmethod
    def normalize_platelet_units(cls, v):
        # Many labs report platelets in x10^3/uL ("128 x10^3/uL") rather
        # than per uL ("128,000 /uL"). A count below 1,000/uL is not a
        # real measurement, so a value in that range can only mean the
        # x10^3 convention -- convert it here, once, for every adapter.
        if isinstance(v, (int, float)) and 0 < v < 1000:
            return int(round(v * 1000))
        return v

    @field_validator("blood_group", mode="before")
    @classmethod
    def check_blood_group(cls, v):
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        cleaned = str(v).strip().upper().replace("POS", "+").replace("NEG", "-").replace(" ", "")
        if cleaned not in BLOOD_GROUPS:
            raise ValueError(f"blood_group must be one of {BLOOD_GROUPS}, got '{v}'")
        return cleaned

    @field_validator("surgery_type")
    @classmethod
    def check_surgery_type(cls, v: str) -> str:
        # Normalize case/whitespace differences between hospitals
        # ("cardiac", " Cardiac ", "CARDIAC" all mean the same thing)
        cleaned = v.strip().title()
        if cleaned not in VALID_SURGERY_TYPES:
            raise ValueError(
                f"surgery_type must be one of {VALID_SURGERY_TYPES}, "
                f"got '{v}'. Map unrecognized hospital-specific surgery "
                f"codes to the nearest category inside the adapter."
            )
        return cleaned

    def to_model_features(self, label_encoder) -> list:
        """
        Converts this record into the exact ordered feature list the
        trained XGBoost model expects. Keeping this conversion in ONE
        place (not duplicated in the backend, the agents, etc.) avoids
        feature-order bugs.
        """
        return [
            self.hemoglobin,
            self.platelets,
            self.INR,
            self.age,
            int(label_encoder.transform([self.surgery_type])[0]),
        ]


class ComponentRecommendation(BaseModel):
    component: str
    probability: float
    needed: bool
    units: int


class PredictionResult(BaseModel):
    """
    The canonical shape of what the ML models hand back to the backend /
    frontend. `confidence` is P(any transfusion); `components` breaks it
    down per component with a dose; `shap` gives each feature's
    contribution (log-odds) to each component's prediction.
    """
    transfusion_needed: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    components: List[ComponentRecommendation] = []
    urgency: str = "Low"
    shap: Dict[str, Dict[str, float]] = {}
    blood_group: Optional[str] = None
    patient_id: Optional[int] = None
    prediction_id: Optional[int] = None
    explanation: str = ""
    source_format: str = "unknown"