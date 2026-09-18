from typing import List, Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from agents.donor_module import load_donors
from agents.tools import send_donor_alert
from backend.dependencies.rbac import require_role
from backend.schemas.requests import DonorAlertRequest
from db.database import get_db
from db.models import DonorAlert

router = APIRouter()


class DonorOut(BaseModel):
    donor_id: str
    name: str
    phone: str
    blood_type: str
    last_donation_date: str
    alerts_sent: int
    alerts_responded: int


class DonorAlertOut(BaseModel):
    donor_id: str
    blood_type: str
    sent_at: str
    method: str


@router.get(
    "/api/donors",
    response_model=List[DonorOut],
    dependencies=[Depends(require_role("Blood Bank Manager", "Donor Coordinator"))],
)
def list_donors(blood_type: Optional[str] = None):
    donors = load_donors()
    if blood_type:
        donors = [d for d in donors if d["blood_type"] == blood_type]
    return donors


@router.get(
    "/api/donors/alerts/recent",
    response_model=List[DonorAlertOut],
    dependencies=[Depends(require_role("Blood Bank Manager", "Donor Coordinator"))],
)
def recent_donor_alerts(limit: int = 10, db: Session = Depends(get_db)):
    """Backs the alerts panel -- unlike /api/audit-log (Auditor/Admin
    only), this is scoped to the donor_alerts table directly so Blood
    Bank Manager/Donor Coordinator can see it without Auditor access.
    """
    rows = db.query(DonorAlert).order_by(DonorAlert.sent_at.desc()).limit(limit).all()
    return [
        DonorAlertOut(
            donor_id=r.donor_id, blood_type=r.blood_type,
            sent_at=r.sent_at.isoformat(), method=r.method,
        )
        for r in rows
    ]


@router.post("/api/donors/alert", dependencies=[Depends(require_role("Blood Bank Manager", "Donor Coordinator"))])
def alert_donors(payload: DonorAlertRequest, request: Request):
    message = send_donor_alert(payload.blood_type)

    request.state.audit_action = "donor_alert_sent"
    request.state.audit_resource_type = "donor_alert"
    request.state.audit_resource_id = payload.blood_type
    request.state.audit_details = {"blood_type": payload.blood_type, "message": message}

    return {"message": message}
