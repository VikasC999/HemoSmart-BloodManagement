from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from agents.donor_module import record_donor_response, send_intelligent_donor_alerts
from agents.tools import list_donors_masked, recent_donor_alerts as _recent_donor_alerts
from bloodbank.catalog import BLOOD_GROUPS, COMPONENTS
from bloodbank.stock import utcnow
from backend.dependencies.rbac import require_permission
from backend.schemas.requests import DonorAlertRequest
from db.database import get_db

router = APIRouter()
can_read = require_permission("donors:read")
can_alert = require_permission("donor_alerts:send")
can_record = require_permission("donor_responses:write")


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


@router.get("/api/donors", response_model=List[DonorOut], dependencies=[Depends(can_read)])
def list_donors(blood_type: Optional[str] = None):
    return list_donors_masked(blood_type)


@router.get("/api/donors/alerts/recent", response_model=List[DonorAlertOut], dependencies=[Depends(can_read)])
def recent_donor_alerts(limit: int = 10, pending_only: bool = False, db: Session = Depends(get_db)):
    """Backs the alerts panel and the coordinator's response queue --
    scoped to donor_alerts directly so Blood Bank Manager/Donor
    Coordinator can see it without Auditor access."""
    return _recent_donor_alerts(limit, pending_only, db=db)


@router.post("/api/donors/alert", dependencies=[Depends(can_alert)])
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


@router.post("/api/donors/alerts/{alert_id}/response", dependencies=[Depends(can_record)])
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
