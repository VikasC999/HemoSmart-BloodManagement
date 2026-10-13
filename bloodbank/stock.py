"""
HemoSmart - Blood Stock Service
-----------------------------------
Every change to physical blood stock goes through here: receiving
units, reserving/issuing them for a transfusion, sweeping expired units,
and transferring near-expiry surplus out. Each function takes the DB
session and an explicit `now`, so the autonomous loop can run against
real time or a simulated clock with the same code.

Status lifecycle of a unit:
    available -> reserved -> issued
              -> expired      (wastage)
              -> transferred  (redistributed before expiry -- wastage avoided)
              -> discarded    (manual, e.g. failed QC)
"""

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import func
from sqlalchemy.orm import Session

from bloodbank.allocation import choose_units, is_near_expiry
from bloodbank.catalog import BLOOD_GROUPS, COMPONENTS, SHELF_LIFE_DAYS
from bloodbank.forecast import par_level
from db.models import BloodUnit, Transfusion

_UNIT_PREFIX = {"PRBC": "RBC", "Platelets": "PLT", "FFP": "FFP"}
_LIVE_STATUSES = ("available", "reserved")


def utcnow() -> datetime:
    """Naive UTC, matching how every DateTime column here is stored."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _new_unit_id(component: str) -> str:
    return f"{_UNIT_PREFIX[component]}-{uuid.uuid4().hex[:8].upper()}"


def _validate(group: str, component: str) -> None:
    if group not in BLOOD_GROUPS:
        raise ValueError(f"Unknown blood group: {group}")
    if component not in COMPONENTS:
        raise ValueError(f"Unknown component: {component}")


# ----------------------------------------------------------------------
# Receiving stock
# ----------------------------------------------------------------------
def receive_units(db: Session, group: str, component: str, count: int, collected_at: datetime,
                  source: str = "stock", donor_id: str = None) -> list:
    _validate(group, component)
    expires_at = collected_at + timedelta(days=SHELF_LIFE_DAYS[component])
    units = [
        BloodUnit(
            unit_id=_new_unit_id(component), blood_group=group, component=component,
            collected_at=collected_at, expires_at=expires_at, status="available",
            status_changed_at=collected_at, source=source, donor_id=donor_id,
        )
        for _ in range(count)
    ]
    db.add_all(units)
    db.flush()
    return units


def discard_unit(db: Session, unit_id: str, now: datetime) -> BloodUnit:
    unit = db.query(BloodUnit).filter(BloodUnit.unit_id == unit_id).first()
    if unit is None:
        raise LookupError(f"No unit {unit_id}")
    if unit.status != "available":
        raise ValueError(f"Unit {unit_id} is {unit.status}, not available")
    unit.status, unit.status_changed_at = "discarded", now
    return unit


# ----------------------------------------------------------------------
# Reading stock
# ----------------------------------------------------------------------
def available_units(db: Session, now: datetime, group: str = None, component: str = None) -> list:
    q = db.query(BloodUnit).filter(BloodUnit.status == "available", BloodUnit.expires_at > now)
    if group:
        q = q.filter(BloodUnit.blood_group == group)
    if component:
        q = q.filter(BloodUnit.component == component)
    # id breaks expiry ties, so allocation (and the simulation) is reproducible
    return q.order_by(BloodUnit.expires_at, BloodUnit.id).all()


def stock_status(available: int, par: float) -> str:
    if available >= par:
        return "Sufficient"
    if available >= par / 2:
        return "Low"
    return "Critical"


def stock_summary(db: Session, now: datetime) -> dict:
    """Per group x component: units available, near-expiry, par and status."""
    units = available_units(db, now)
    grid = {g: {c: {"available": 0, "near_expiry": 0} for c in COMPONENTS} for g in BLOOD_GROUPS}
    for u in units:
        cell = grid[u.blood_group][u.component]
        cell["available"] += 1
        cell["near_expiry"] += is_near_expiry(u, now)

    low_stock = []
    for g in BLOOD_GROUPS:
        for c in COMPONENTS:
            cell = grid[g][c]
            cell["par"] = par_level(g, c)
            cell["status"] = stock_status(cell["available"], cell["par"])
            if cell["status"] != "Sufficient":
                low_stock.append({"blood_group": g, "component": c, **cell})

    return {
        "as_of": now.isoformat(),
        "inventory": grid,
        "low_stock": low_stock,
        "any_shortage": any(i["status"] == "Critical" for i in low_stock),
        "near_expiry_total": sum(grid[g][c]["near_expiry"] for g in BLOOD_GROUPS for c in COMPONENTS),
        "total_available": len(units),
    }


# ----------------------------------------------------------------------
# Transfusion workflow: reserve (FEFO) -> issue | cancel
# ----------------------------------------------------------------------
def reserve_for_transfusion(db: Session, patient_id: int, recipient_group: str, request: dict, now: datetime,
                            prediction_id: int = None, created_by: int = None) -> Transfusion:
    """Reserves the best units per bloodbank.allocation for each requested
    component. A partial reservation is still made when stock is short --
    the shortfall is recorded so the caller (and the Planner agent) can
    see it, rather than the whole request failing."""
    if recipient_group not in BLOOD_GROUPS:
        raise ValueError(f"Unknown blood group: {recipient_group}")
    transfusion = Transfusion(
        patient_id=patient_id, prediction_id=prediction_id, recipient_group=recipient_group,
        status="reserved", requested=request, created_by=created_by, created_at=now,
    )
    db.add(transfusion)
    db.flush()

    shortfall = {}
    for component, count in request.items():
        if count <= 0:
            continue
        pool = available_units(db, now, component=component)
        chosen = choose_units(pool, recipient_group, component, count, now)
        for unit in chosen:
            unit.status, unit.status_changed_at, unit.transfusion_id = "reserved", now, transfusion.id
        if len(chosen) < count:
            shortfall[component] = count - len(chosen)
    transfusion.shortfall = shortfall or None
    db.flush()
    return transfusion


def _units_of(db: Session, transfusion: Transfusion) -> list:
    return db.query(BloodUnit).filter(BloodUnit.transfusion_id == transfusion.id).all()


def confirm_transfusion(db: Session, transfusion_id: int, now: datetime) -> Transfusion:
    t = _get_open(db, transfusion_id)
    for unit in _units_of(db, t):
        if unit.expires_at <= now:
            raise ValueError(f"Unit {unit.unit_id} expired while reserved -- cancel and re-request.")
        unit.status, unit.status_changed_at = "issued", now
    t.status, t.updated_at = "issued", now
    return t


def cancel_transfusion(db: Session, transfusion_id: int, now: datetime) -> Transfusion:
    t = _get_open(db, transfusion_id)
    for unit in _units_of(db, t):
        unit.status, unit.status_changed_at, unit.transfusion_id = "available", now, None
    t.status, t.updated_at = "cancelled", now
    return t


def _get_open(db: Session, transfusion_id: int) -> Transfusion:
    t = db.query(Transfusion).filter(Transfusion.id == transfusion_id).first()
    if t is None:
        raise LookupError(f"No transfusion {transfusion_id}")
    if t.status != "reserved":
        raise ValueError(f"Transfusion {transfusion_id} is already {t.status}")
    return t


# ----------------------------------------------------------------------
# Wastage: expiry sweep, redistribution, and measurement
# ----------------------------------------------------------------------
def sweep_expired(db: Session, now: datetime) -> list:
    """Marks every unit past its expiry as expired -- this IS the wastage
    HemoSmart measures. Reserved units count too: a unit that expired
    while held for a patient was still wasted."""
    expired = (
        db.query(BloodUnit)
        .filter(BloodUnit.status.in_(_LIVE_STATUSES), BloodUnit.expires_at <= now)
        .all()
    )
    for unit in expired:
        unit.status, unit.status_changed_at = "expired", now
    return expired


def transfer_units(db: Session, units: list, partner: str, now: datetime) -> None:
    for unit in units:
        unit.status, unit.status_changed_at, unit.transferred_to = "transferred", now, partner


def wastage_stats(db: Session, since: datetime, until: datetime) -> dict:
    """Unit outcomes in a time window -- the system's headline metric."""
    rows = (
        db.query(BloodUnit.status, BloodUnit.component, func.count())
        .filter(BloodUnit.status.in_(("issued", "expired", "transferred", "discarded")),
                BloodUnit.status_changed_at > since, BloodUnit.status_changed_at <= until)
        .group_by(BloodUnit.status, BloodUnit.component)
        .all()
    )
    stats = {s: {c: 0 for c in COMPONENTS} for s in ("issued", "expired", "transferred", "discarded")}
    for status, component, n in rows:
        stats[status][component] = n
    totals = {s: sum(v.values()) for s, v in stats.items()}
    used = totals["issued"] + totals["expired"] + totals["transferred"]
    return {
        "since": since.isoformat(), "until": until.isoformat(),
        "by_status": stats, "totals": totals,
        "wastage_rate": round(totals["expired"] / used, 4) if used else None,
    }
