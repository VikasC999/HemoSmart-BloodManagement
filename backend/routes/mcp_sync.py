"""
Makes the hospital adapter boundary (integrations/mcp/) tangible:
fetches a patient record from a registered external hospital system,
predicts against it exactly like any other intake path, and persists
it the same way manual/PDF/CSV entry does.

Not to be confused with mcp_server/ -- that's the real Model Context
Protocol server this backend also exposes at /mcp for LLM clients.
This route is plain REST, named for the adapter pattern it demonstrates.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from agents.tools import predict_transfusion
from backend.dependencies.rbac import require_role
from backend.services.persistence import persist_prediction
from db.database import get_db
from db.models import User
from integrations.mcp.mock_hospital import MockHospitalAdapter
from schema.patient_schema import PredictionResult

router = APIRouter()
allowed_roles = require_role("Hospital Staff", "Blood Bank Manager")

_ADAPTERS = {
    "mock-regional-hospital": MockHospitalAdapter(),
}


class MCPSyncRequest(BaseModel):
    external_patient_id: str


@router.post("/api/mcp/sync/{hospital_id}", response_model=PredictionResult)
def sync_from_hospital(
    hospital_id: str, payload: MCPSyncRequest, request: Request,
    db: Session = Depends(get_db), user: User = Depends(allowed_roles),
):
    adapter = _ADAPTERS.get(hospital_id)
    if adapter is None:
        raise HTTPException(status_code=404, detail=f"No adapter registered for hospital_id='{hospital_id}'.")

    try:
        record = adapter.fetch_patient_record(payload.external_patient_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    result = predict_transfusion(record)
    persist_prediction(db, record, result, created_by=user.id)

    request.state.audit_action = "mcp_hospital_sync"
    request.state.audit_resource_type = "prediction"
    request.state.audit_resource_id = payload.external_patient_id
    request.state.audit_details = {
        "hospital_id": hospital_id,
        "external_patient_id": payload.external_patient_id,
        "transfusion_needed": result["transfusion_needed"],
    }

    return PredictionResult(
        transfusion_needed=result["transfusion_needed"],
        confidence=result["confidence"],
        source_format=record.source_format,
    )
