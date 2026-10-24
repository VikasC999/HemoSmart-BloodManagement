"""
The transfusion workflow -- where a prediction turns into blood leaving
the fridge (report Section 1.6, Step 9):

    POST /api/transfusions               reserve units FEFO for a patient
    POST /api/transfusions/{id}/confirm  transfused -> units issued
    POST /api/transfusions/{id}/cancel   not needed -> units released

Reservation is where wastage-aware allocation happens: compatible
near-expiry units go first, from any compatible group (see
bloodbank/allocation.py), so a unit about to expire is used rather than
wasted.
"""

from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from bloodbank.allocation import is_near_expiry
from bloodbank.catalog import COMPONENTS
from bloodbank.stock import cancel_transfusion, confirm_transfusion, reserve_for_transfusion, utcnow
from backend.dependencies.rbac import require_permission
from db.database import get_db
from db.models import BloodUnit, Patient, Prediction, Transfusion, User

router = APIRouter()
clinical_roles = require_permission("transfusions:write")


class TransfusionRequest(BaseModel):
    patient_id: int
    prediction_id: Optional[int] = None
    # Defaults to the patient's recorded group
    blood_group: Optional[str] = None
    # Defaults to the doses in the prediction's component recommendation
    components: Optional[Dict[str, int]] = None


def _transfusion_out(t: Transfusion, db: Session) -> dict:
    units = db.query(BloodUnit).filter(BloodUnit.transfusion_id == t.id).order_by(BloodUnit.component).all()
    now = utcnow()
    return {
        "id": t.id,
        "patient_id": t.patient_id,
        "prediction_id": t.prediction_id,
        "recipient_group": t.recipient_group,
        "status": t.status,
        "requested": t.requested,
        "shortfall": t.shortfall or {},
        "created_at": t.created_at.isoformat() if t.created_at else None,
        "units": [
            {
                "unit_id": u.unit_id, "blood_group": u.blood_group, "component": u.component,
                "expires_at": u.expires_at.isoformat(), "status": u.status,
                "near_expiry": is_near_expiry(u, now),
                "cross_group": u.blood_group != t.recipient_group,
            }
            for u in units
        ],
    }


@router.post("/api/transfusions")
def request_transfusion(
    payload: TransfusionRequest, request: Request,
    db: Session = Depends(get_db), user: User = Depends(clinical_roles),
):
    patient = db.query(Patient).filter(Patient.id == payload.patient_id).first()
    if patient is None:
        raise HTTPException(status_code=404, detail=f"No patient {payload.patient_id}")
    group = payload.blood_group or patient.blood_group
    if not group:
        raise HTTPException(status_code=422, detail="Patient's blood group is unknown -- provide blood_group.")

    components = payload.components
    if components is None:
        prediction = db.query(Prediction).filter(Prediction.id == payload.prediction_id).first() \
            if payload.prediction_id else None
        if prediction is None or not prediction.components:
            raise HTTPException(status_code=422, detail="Provide components, or a prediction_id with a recommendation.")
        components = {c["component"]: c["units"] for c in prediction.components if c["needed"]}
    unknown = set(components) - set(COMPONENTS)
    if unknown or not any(v > 0 for v in components.values()):
        raise HTTPException(status_code=422, detail=f"components must request units of {COMPONENTS}")

    try:
        t = reserve_for_transfusion(db, patient.id, group, components, utcnow(),
                                    prediction_id=payload.prediction_id, created_by=user.id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if patient.blood_group is None:
        patient.blood_group = group
    db.commit()

    out = _transfusion_out(t, db)
    request.state.audit_action = "transfusion_reserved"
    request.state.audit_resource_type = "transfusion"
    request.state.audit_resource_id = str(t.id)
    request.state.audit_details = {"patient_id": patient.id, "requested": components,
                                   "unit_ids": [u["unit_id"] for u in out["units"]],
                                   "shortfall": out["shortfall"]}
    return out


def _transition(fn, transfusion_id: int, action: str, request: Request, db: Session):
    try:
        t = fn(db, transfusion_id, utcnow())
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    db.commit()
    request.state.audit_action = action
    request.state.audit_resource_type = "transfusion"
    request.state.audit_resource_id = str(transfusion_id)
    return _transfusion_out(t, db)


@router.post("/api/transfusions/{transfusion_id}/confirm", dependencies=[Depends(clinical_roles)])
def confirm(transfusion_id: int, request: Request, db: Session = Depends(get_db)):
    return _transition(confirm_transfusion, transfusion_id, "transfusion_issued", request, db)


@router.post("/api/transfusions/{transfusion_id}/cancel", dependencies=[Depends(clinical_roles)])
def cancel(transfusion_id: int, request: Request, db: Session = Depends(get_db)):
    return _transition(cancel_transfusion, transfusion_id, "transfusion_cancelled", request, db)


@router.get(
    "/api/transfusions", response_model=List[dict],
    dependencies=[Depends(require_permission("transfusions:read"))],
)
def list_transfusions(limit: int = 20, patient_id: Optional[int] = None, db: Session = Depends(get_db)):
    q = db.query(Transfusion)
    if patient_id is not None:
        q = q.filter(Transfusion.patient_id == patient_id)
    return [_transfusion_out(t, db) for t in q.order_by(Transfusion.created_at.desc()).limit(limit).all()]
