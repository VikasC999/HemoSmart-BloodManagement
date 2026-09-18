from fastapi import APIRouter, Depends, Request

from agents.tools import send_donor_alert
from backend.dependencies.rbac import require_role
from backend.schemas.requests import DonorAlertRequest

router = APIRouter()


@router.post("/api/donors/alert", dependencies=[Depends(require_role("Blood Bank Manager", "Donor Coordinator"))])
def alert_donors(payload: DonorAlertRequest, request: Request):
    message = send_donor_alert(payload.blood_type)

    request.state.audit_action = "donor_alert_sent"
    request.state.audit_resource_type = "donor_alert"
    request.state.audit_resource_id = payload.blood_type
    request.state.audit_details = {"blood_type": payload.blood_type, "message": message}

    return {"message": message}
