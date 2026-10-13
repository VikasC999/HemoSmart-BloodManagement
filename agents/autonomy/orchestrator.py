"""
HemoSmart - Autonomous Loop Orchestrator
--------------------------------------------
Runs one cycle of the wastage-reducing loop, end to end, without a human
in it:

    Donor feedback -> Inventory -> Forecast -> Planner -> Donor outreach
                   -> Logistics -> Explanation

The Orchestrator owns sequencing and failure handling, not decisions --
those belong to the agents (agents.py). Each agent step commits on its
own, so one agent failing can't roll back another's real-world actions
(an alert already sent stays recorded). A failed step is logged, the
cycle is marked "degraded", and the Orchestrator routes around it:
  - Forecast fails  -> Planner runs on historical average demand
  - Planner fails   -> outreach and redistribution are skipped this cycle
  - RAG/LLM fails   -> actions are recorded without citations / with a
                       template brief

Triggered three ways: the background scheduler (scheduler.py), a manual
run from the dashboard (POST /api/autonomy/run), or a simulation that
passes its own clock.
"""

import os
import random
import threading
from datetime import datetime

from agents.autonomy.agents import (
    DonorAgent,
    ExplanationAgent,
    ForecastAgent,
    InventoryAgent,
    LogisticsAgent,
    PlannerAgent,
)
from agents.autonomy.blackboard import Blackboard, _jsonable
from bloodbank.forecast import get_forecaster
from bloodbank.stock import utcnow
from db.database import SessionLocal
from db.models import AgentCycle
from integrations.mcp.mock_hospital import MockHospitalAdapter

_cycle_lock = threading.Lock()


class CycleAlreadyRunning(RuntimeError):
    pass


def simulate_donor_responses_enabled() -> bool:
    return os.environ.get("HEMOSMART_SIMULATE_DONOR_RESPONSES", "1") != "0"


def run_cycle(
    now: datetime = None,
    mode: str = "live",
    trigger: str = "manual",
    horizon_days: int = 7,
    forecaster=None,
    retriever=None,
    llm=None,
    partner=None,
    simulate_responses: bool = None,
    rng: random.Random = None,
    session_factory=SessionLocal,
) -> dict:
    """Runs one cycle and returns it as a dict (same shape the API serves)."""
    if not _cycle_lock.acquire(blocking=False):
        raise CycleAlreadyRunning("An autonomy cycle is already running.")
    try:
        return _run(now, mode, trigger, horizon_days, forecaster, retriever, llm, partner,
                    simulate_responses, rng, session_factory)
    finally:
        _cycle_lock.release()


def _run(now, mode, trigger, horizon_days, forecaster, retriever, llm, partner,
         simulate_responses, rng, session_factory) -> dict:
    now = now or utcnow()
    if simulate_responses is None:
        simulate_responses = simulate_donor_responses_enabled()

    with session_factory() as db:
        cycle = AgentCycle(mode=mode, trigger=trigger, clock=now, started_at=utcnow(), status="running")
        db.add(cycle)
        db.commit()

        bb = Blackboard(now=now, cycle_id=cycle.id, mode=mode, horizon_days=horizon_days)
        donor = DonorAgent(simulate_responses, rng or random.Random(cycle.id))
        forecast = ForecastAgent(forecaster or get_forecaster())
        planner_ok = True

        # (agent, step, needs the Planner's output)
        steps = [
            ("Donor", donor.collect_feedback, False),
            ("Inventory", InventoryAgent().run, False),
            ("Forecast", forecast.run, False),
            ("Planner", PlannerAgent().run, False),
            ("Donor", donor.act, True),
            ("Logistics", LogisticsAgent(partner or MockHospitalAdapter()).run, True),
            ("Explanation", ExplanationAgent(retriever, llm).run, False),
        ]
        degraded = False
        for name, step, needs_plan in steps:
            if needs_plan and not planner_ok:
                bb.log(name, "fallback", "Skipped -- the Planner produced no plan this cycle.")
                continue
            try:
                step(bb, db)
                db.commit()
            except Exception as exc:
                db.rollback()
                degraded = True
                bb.log(name, "error", f"{name} agent failed: {exc.__class__.__name__}: {exc}")
                if name == "Forecast":
                    forecast.fallback(bb)
                elif name == "Planner":
                    planner_ok = False

        cycle = db.get(AgentCycle, cycle.id)
        cycle.finished_at = utcnow()
        cycle.status = "degraded" if degraded else "ok"
        cycle.summary = _summary(bb)
        cycle.events = bb.events
        cycle.brief = bb.brief
        db.commit()
        return cycle_to_dict(cycle)


def _summary(bb: Blackboard) -> dict:
    return _jsonable({
        "expired_units": len(bb.expired_units),
        "donor_replies": len(bb.donor_feedback),
        "donations": sum(o["responded"] for o in bb.donor_feedback),
        "units_donated": sum(len(o["units_received"]) for o in bb.donor_feedback),
        "shortages": len(bb.shortages),
        "outreach_needs": len(bb.needs),
        "donors_alerted": sum(len(a["alerted"]) for a in bb.alerts),
        "escalations": len(bb.escalations),
        "wastage_risk_units": sum(len(r["units"]) for r in bb.wastage_risks),
        "transferred_units": len(bb.transfers),
        "errors": len(bb.events_of("error")),
    })


def cycle_to_dict(cycle: AgentCycle) -> dict:
    return {
        "id": cycle.id,
        "mode": cycle.mode,
        "trigger": cycle.trigger,
        "clock": cycle.clock.isoformat(),
        "started_at": cycle.started_at.isoformat() if cycle.started_at else None,
        "finished_at": cycle.finished_at.isoformat() if cycle.finished_at else None,
        "status": cycle.status,
        "summary": cycle.summary,
        "events": cycle.events or [],
        "brief": cycle.brief,
    }
