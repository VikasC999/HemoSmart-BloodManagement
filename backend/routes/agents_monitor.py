"""
The agent monitor's API (D74): are the agents running, how long each
step takes, how often the LLM agents' answers check out, and what needs
a person. Read-only -- every action it points to lives on its own page,
behind that page's permission.

    GET /api/agents/overview          cards, open runs, alerts, attention, metric strip
    GET /api/agents/runs              recent runs
    GET /api/agents/runs/{run_id}     one run's step timeline
    GET /api/agents/metrics           per-agent aggregates over `days`

The page polls the overview (D73), so these are cheap queries over at
most a week of agent_events rows.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from agents.autonomy.scheduler import interval_minutes
from backend.dependencies.rbac import require_permission
from backend.services import agent_monitor
from bloodbank.stock import utcnow
from db.database import get_db

router = APIRouter()
monitors = require_permission("agents:monitor")


@router.get("/api/agents/overview", dependencies=[Depends(monitors)])
def overview(db: Session = Depends(get_db)):
    return agent_monitor.overview(db, utcnow(), interval_minutes())


@router.get("/api/agents/runs", dependencies=[Depends(monitors)])
def runs(layer: str = Query(None, pattern="^(supply_loop|case_pipeline)$"),
         limit: int = Query(20, ge=1, le=100), db: Session = Depends(get_db)):
    return agent_monitor.recent_runs(db, utcnow(), layer, limit)


@router.get("/api/agents/runs/{run_id}", dependencies=[Depends(monitors)])
def run_detail(run_id: str, db: Session = Depends(get_db)):
    detail = agent_monitor.run_detail(db, run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="No such agent run (runs are kept for 14 days).")
    return detail


@router.get("/api/agents/metrics", dependencies=[Depends(monitors)])
def metrics(days: int = Query(7, ge=1, le=30), db: Session = Depends(get_db)):
    return agent_monitor.metrics(db, utcnow(), days)
