from fastapi import APIRouter, Depends

from agents.tools import send_donor_alert
from backend.dependencies.rbac import require_role
from backend.schemas.requests import DonorAlertRequest

router = APIRouter()


@router.post("/api/donors/alert", dependencies=[Depends(require_role("Blood Bank Manager", "Donor Coordinator"))])
def alert_donors(payload: DonorAlertRequest):
    message = send_donor_alert(payload.blood_type)
    return {"message": message}
