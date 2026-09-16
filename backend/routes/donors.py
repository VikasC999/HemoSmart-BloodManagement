from fastapi import APIRouter

from agents.tools import send_donor_alert
from backend.schemas.requests import DonorAlertRequest

router = APIRouter()


@router.post("/api/donors/alert")
def alert_donors(payload: DonorAlertRequest):
    message = send_donor_alert(payload.blood_type)
    return {"message": message}
