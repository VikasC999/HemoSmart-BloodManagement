"""
The autonomous loop's API: trigger a cycle now, read what past cycles
observed/decided/did, and the wastage metrics the loop exists to move.
"""

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from agents.autonomy.orchestrator import CycleAlreadyRunning, cycle_to_dict, run_cycle
from agents.autonomy.scheduler import interval_minutes
from agents.tools import latest_autonomy_cycle
from bloodbank.stock import utcnow, wastage_stats
from backend.dependencies.rbac import require_permission
from db.database import get_db
from db.models import AgentCycle

router = APIRouter()
readers = require_permission("autonomy:read")


@router.post("/api/autonomy/run", dependencies=[Depends(require_permission("autonomy:run"))])
async def run_now(request: Request):
    generator = getattr(request.app.state, "explanation_generator", None)
    try:
        cycle = await run_in_threadpool(
            run_cycle, trigger="manual",
            retriever=generator.retriever if generator else None,
            llm=generator.llm if generator else None,
        )
    except CycleAlreadyRunning as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    request.state.audit_action = "autonomy_cycle_run"
    request.state.audit_resource_type = "agent_cycle"
    request.state.audit_resource_id = str(cycle["id"])
    request.state.audit_details = cycle["summary"]
    return cycle


@router.get("/api/autonomy/cycles", dependencies=[Depends(readers)])
def list_cycles(limit: int = Query(10, ge=1, le=100), db: Session = Depends(get_db)):
    rows = db.query(AgentCycle).order_by(AgentCycle.id.desc()).limit(limit).all()
    return [cycle_to_dict(c) for c in rows]


@router.get("/api/autonomy/status", dependencies=[Depends(readers)])
def status(db: Session = Depends(get_db)):
    return {
        "scheduler_interval_minutes": interval_minutes(),
        "last_cycle": latest_autonomy_cycle(db=db),
    }


@router.get("/api/autonomy/metrics", dependencies=[Depends(readers)])
def metrics(days: int = Query(30, ge=1, le=365), db: Session = Depends(get_db)):
    """Unit outcomes over the last `days`: issued, expired (wastage),
    transferred (wastage avoided), discarded -- per component."""
    now = utcnow()
    return wastage_stats(db, now - timedelta(days=days), now)
