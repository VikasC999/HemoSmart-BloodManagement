"""
Unit-level blood inventory. Stock is never "set" to a number any more --
it changes only through events that leave a trace: units received,
reserved/issued for a patient, expired, transferred out, or discarded.
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from agents.tools import check_inventory
from bloodbank.allocation import is_near_expiry
from bloodbank.stock import discard_unit, receive_units, utcnow
from db.database import get_db
from db.models import BloodUnit
from backend.dependencies.rbac import require_permission

router = APIRouter()


class ReceiveUnitsRequest(BaseModel):
    blood_group: str
    component: str
    count: int = Field(..., ge=1, le=200)
    collected_at: Optional[datetime] = None  # defaults to now


class UnitOut(BaseModel):
    unit_id: str
    blood_group: str
    component: str
    collected_at: str
    expires_at: str
    status: str
    near_expiry: bool
    hours_to_expiry: float


def _unit_out(u: BloodUnit, now: datetime) -> UnitOut:
    return UnitOut(
        unit_id=u.unit_id, blood_group=u.blood_group, component=u.component,
        collected_at=u.collected_at.isoformat(), expires_at=u.expires_at.isoformat(),
        status=u.status, near_expiry=is_near_expiry(u, now),
        hours_to_expiry=round((u.expires_at - now).total_seconds() / 3600, 1),
    )


@router.get("/api/inventory", dependencies=[Depends(require_permission("inventory:read"))])
def inventory(blood_group: Optional[str] = None, component: Optional[str] = None):
    try:
        return check_inventory(blood_group, component)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get(
    "/api/inventory/units", response_model=List[UnitOut],
    dependencies=[Depends(require_permission("inventory:read"))],
)
def list_units(
    blood_group: Optional[str] = None, component: Optional[str] = None,
    status: str = "available", limit: int = 200, db: Session = Depends(get_db),
):
    """Units soonest-to-expire first -- the order FEFO will issue them in."""
    q = db.query(BloodUnit).filter(BloodUnit.status == status)
    if blood_group:
        q = q.filter(BloodUnit.blood_group == blood_group)
    if component:
        q = q.filter(BloodUnit.component == component)
    now = utcnow()
    return [_unit_out(u, now) for u in q.order_by(BloodUnit.expires_at).limit(limit).all()]


@router.post("/api/inventory/units", dependencies=[Depends(require_permission("inventory:write"))])
def receive(payload: ReceiveUnitsRequest, request: Request, db: Session = Depends(get_db)):
    """Records units received (a collection drive, a supplier delivery).
    Expiry is computed from the collection time and the component's
    shelf life -- never typed in by hand."""
    now = utcnow()
    collected_at = payload.collected_at.replace(tzinfo=None) if payload.collected_at else now
    if collected_at > now:
        raise HTTPException(status_code=422, detail="collected_at can't be in the future.")
    try:
        units = receive_units(db, payload.blood_group, payload.component, payload.count, collected_at)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    db.commit()

    request.state.audit_action = "inventory_received"
    request.state.audit_resource_type = "inventory"
    request.state.audit_resource_id = f"{payload.blood_group} {payload.component}"
    request.state.audit_details = {**payload.model_dump(mode="json"), "unit_ids": [u.unit_id for u in units]}
    return {"received": [_unit_out(u, now) for u in units]}


@router.post("/api/inventory/units/{unit_id}/discard", dependencies=[Depends(require_permission("inventory:write"))])
def discard(unit_id: str, request: Request, db: Session = Depends(get_db)):
    """Removes a unit from stock for a reason other than expiry (failed
    QC, broken bag) -- recorded separately from expiry wastage."""
    try:
        unit = discard_unit(db, unit_id, utcnow())
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    db.commit()

    request.state.audit_action = "inventory_discarded"
    request.state.audit_resource_type = "blood_unit"
    request.state.audit_resource_id = unit_id
    return {"unit_id": unit.unit_id, "status": unit.status}
