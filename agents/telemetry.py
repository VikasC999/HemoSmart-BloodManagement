"""
HemoSmart - Agent Telemetry (D71, D72)
-----------------------------------------
Small, structured, patient-data-free events for the agent monitor: a
step started or finished, an LLM rate-limit wait, a scheduler
heartbeat. Stored in agent_events; read by backend/services/agent_monitor.py.

Sits next to the agents, not inside them: the orchestrator, scheduler
and case route call emit()/start_step(), and no agent logic depends on
them. Two rules:
  - Never raises, and writes through its own session -- a telemetry
    failure can't break or roll back an agent's work (the same rule as
    the audit middleware).
  - Payloads are allow-listed (ALLOWED_KEYS); every other key is
    dropped. Lab values, filenames, donor IDs, chat text and exception
    messages never reach the table.

Off when run_id is None (simulation runs pass None), or with
HEMOSMART_AGENT_EVENTS=0.
"""

import contextvars
import os
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

ALLOWED_KEYS = frozenset({
    # counts
    "expired_units", "donors_alerted", "transferred_units", "escalations", "units",
    "attempt", "wait_s", "recommended_alerts", "interval_min",
    # categories
    "blood_group", "component", "outcome", "trigger", "dispatch_allowed",
    "exc_class",  # the exception CLASS name only, never its message
})

RUN = "Run"  # the `agent` of run-level events (run_started / run_finished)

_current_run = contextvars.ContextVar("hemosmart_agent_run", default=(None, None))


def enabled() -> bool:
    return os.environ.get("HEMOSMART_AGENT_EVENTS", "1") != "0"


def retention_days() -> float:
    return float(os.environ.get("HEMOSMART_AGENT_EVENTS_DAYS", "14"))


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def safe_data(data: dict) -> dict:
    """Keeps only allow-listed keys with plain scalar values."""
    return {k: v for k, v in data.items()
            if k in ALLOWED_KEYS and (v is None or isinstance(v, (bool, int, float, str)))}


def emit(run_id, layer, agent, step, type, status=None, duration_ms=None, llm_calls=None, **data) -> None:
    if run_id is None or not enabled():
        return
    try:
        from db.database import SessionLocal
        from db.models import AgentEvent
        with SessionLocal() as db:
            db.add(AgentEvent(
                run_id=run_id, layer=layer, agent=agent, step=step, type=type, status=status,
                ts=utcnow(), duration_ms=duration_ms, llm_calls=llm_calls, data=safe_data(data) or None,
            ))
            db.commit()
    except Exception as exc:  # observability must never break the operation
        print(f"[telemetry] Event not recorded: {exc.__class__.__name__}")


def _llm_calls(hook: bool) -> int:
    try:
        from rag.llm_client import llm_call_count
        return llm_call_count(hook=hook)
    except Exception:
        return 0


class Step:
    """One agent step: start_step() records step_started; finish()
    records step_finished with the duration and the LLM-call delta."""

    def __init__(self, run_id, layer, agent, step, hook_llm=False):
        self.run_id, self.layer, self.agent, self.step = run_id, layer, agent, step
        self.hook_llm = hook_llm
        self.started = time.monotonic()
        self.calls_before = _llm_calls(hook_llm)
        self.finished = False
        emit(run_id, layer, agent, step, "step_started")

    def finish(self, status: str, **data) -> None:
        if self.finished:
            return
        self.finished = True
        emit(self.run_id, self.layer, self.agent, self.step, "step_finished", status=status,
             duration_ms=int((time.monotonic() - self.started) * 1000),
             llm_calls=_llm_calls(self.hook_llm) - self.calls_before, **data)


def start_step(run_id, layer, agent, step, hook_llm=False) -> Step:
    return Step(run_id, layer, agent, step, hook_llm)


@contextmanager
def run_context(run_id, layer):
    """Binds the current run to this thread, so code deep inside a
    pipeline (the CrewAI rate-limit retry) can report without the run
    ID being passed down every call."""
    token = _current_run.set((run_id, layer))
    try:
        yield
    finally:
        _current_run.reset(token)


def emit_current(agent, step, type, status=None, **data) -> None:
    run_id, layer = _current_run.get()
    emit(run_id, layer, agent, step, type, status, **data)


def prune(days: float = None, now: datetime = None) -> int:
    """Deletes events older than the retention window. Never raises."""
    try:
        from db.database import SessionLocal
        from db.models import AgentEvent
        cutoff = (now or utcnow()) - timedelta(days=retention_days() if days is None else days)
        with SessionLocal() as db:
            n = db.query(AgentEvent).filter(AgentEvent.ts < cutoff).delete(synchronize_session=False)
            db.commit()
            return n
    except Exception as exc:
        print(f"[telemetry] Prune failed: {exc.__class__.__name__}")
        return 0


def close_open_runs(outcome: str = "interrupted") -> int:
    """At startup: a run with no run_finished belonged to a process that
    died, so it is closed as failed instead of looking busy forever."""
    try:
        from db.database import SessionLocal
        from db.models import AgentEvent
        with SessionLocal() as db:
            started = db.query(AgentEvent.run_id, AgentEvent.layer).filter(AgentEvent.type == "run_started").all()
            finished = {r for (r,) in db.query(AgentEvent.run_id).filter(AgentEvent.type == "run_finished")}
        open_runs = [(r, layer) for r, layer in started if r not in finished]
        for run_id, layer in open_runs:
            emit(run_id, layer, RUN, None, "run_finished", status="failed", outcome=outcome)
        return len(open_runs)
    except Exception as exc:
        print(f"[telemetry] Couldn't close open runs: {exc.__class__.__name__}")
        return 0
