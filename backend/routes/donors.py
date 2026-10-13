from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from agents.donor_module import load_donors, posterior_mean, record_donor_response, send_intelligent_donor_alerts
from bloodbank.catalog import BLOOD_GROUPS, COMPONENTS
from bloodbank.stock import utcnow
from backend.dependencies.rbac import require_role
from backend.schemas.requests import DonorAlertRequest
from db.database import get_db
from db.models import DonorAlert
from schema.privacy import mask_identifier

router = APIRouter()
donor_roles = require_role("Blood Bank Manager", "Donor Coordinator")


class DonorOut(BaseModel):
    donor_id: str
    name: str
    phone: str  # masked to the last 4 digits (NFR-03)
    blood_type: str
    last_donation_date: str
    alerts_sent: int
    alerts_responded: int
    expected_response_rate: float  # the bandit's current belief (Beta posterior mean)


class DonorAlertOut(BaseModel):
    id: int
    donor_id: str
    blood_type: str
    component: Optional[str]
    sent_at: str
    method: str
    responded: Optional[bool]
    responded_at: Optional[str]
    cycle_id: Optional[int]  # set when the autonomous loop sent it


class DonorResponseRequest(BaseModel):
    responded: bool


@router.get("/api/donors", response_model=List[DonorOut], dependencies=[Depends(donor_roles)])
def list_donors(blood_type: Optional[str] = None):
    donors = load_donors()
    if blood_type:
        donors = [d for d in donors if d["blood_type"] == blood_type]
    return [{**d, "phone": mask_identifier(d["phone"]), "expected_response_rate": round(posterior_mean(d), 3)}
            for d in donors]


@router.get("/api/donors/alerts/recent", response_model=List[DonorAlertOut], dependencies=[Depends(donor_roles)])
def recent_donor_alerts(limit: int = 10, pending_only: bool = False, db: Session = Depends(get_db)):
    """Backs the alerts panel and the coordinator's response queue --
    scoped to donor_alerts directly so Blood Bank Manager/Donor
    Coordinator can see it without Auditor access."""
    q = db.query(DonorAlert)
    if pending_only:
        q = q.filter(DonorAlert.responded.is_(None))
    rows = q.order_by(DonorAlert.sent_at.desc()).limit(limit).all()
    return [
        DonorAlertOut(
            id=r.id, donor_id=r.donor_id, blood_type=r.blood_type, component=r.component,
            sent_at=r.sent_at.isoformat(), method=r.method, responded=r.responded,
            responded_at=r.responded_at.isoformat() if r.responded_at else None, cycle_id=r.cycle_id,
        )
        for r in rows
    ]


@router.post("/api/donors/alert", dependencies=[Depends(donor_roles)])
def alert_donors(payload: DonorAlertRequest, request: Request):
    if payload.blood_type not in BLOOD_GROUPS or payload.component not in COMPONENTS:
        raise HTTPException(status_code=422, detail=f"blood_type in {BLOOD_GROUPS}, component in {COMPONENTS}")
    result = send_intelligent_donor_alerts(payload.blood_type, component=payload.component)

    request.state.audit_action = "donor_alert_sent"
    request.state.audit_resource_type = "donor_alert"
    request.state.audit_resource_id = payload.blood_type
    request.state.audit_details = {"blood_type": payload.blood_type, "component": payload.component,
                                   "donors": [d["donor_id"] for d in result["alerted_donors"]]}
    return result


@router.post("/api/donors/alerts/{alert_id}/response", dependencies=[Depends(donor_roles)])
def record_response(alert_id: int, payload: DonorResponseRequest, request: Request, db: Session = Depends(get_db)):
    """Records a donor's real reply -- the bandit's learning signal. A
    donor who responds donates: their units enter stock immediately."""
    try:
        outcome = record_donor_response(db, alert_id, payload.responded, utcnow())
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    db.commit()

    request.state.audit_action = "donor_response_recorded"
    request.state.audit_resource_type = "donor_alert"
    request.state.audit_resource_id = str(alert_id)
    request.state.audit_details = outcome
    return outcome
