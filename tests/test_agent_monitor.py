"""
The agent monitor (D71-D75): what the agents emit, that nothing patient-
identifying gets into it, who may read it, and the status/alert rules
under a fixed clock.
"""

import sys
import tempfile
import time
import types
from datetime import timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agents import telemetry
from agents.autonomy.orchestrator import mark_orphaned_cycles, run_cycle
from backend.middleware.audit import AuditLogMiddleware
from backend.routes import agents_monitor, cases
from backend.services import agent_monitor as monitor
from bloodbank.catalog import BLOOD_GROUPS, COMPONENTS
from bloodbank.stock import utcnow
from conftest import auth_header
from db.database import SessionLocal
from db.models import AgentCycle, AgentEvent, AuditLog, DonorAlert
from schema.permissions import denial

app = FastAPI()
app.add_middleware(AuditLogMiddleware)
app.include_router(agents_monitor.router)
app.include_router(cases.router)
client = TestClient(app)


class StubForecaster:
    model_name = "stub"

    def __init__(self, fail=False):
        self.fail = fail

    def group_forecast(self, start, days, totals=None):
        return {(g, c): [0.0] * days for g in BLOOD_GROUPS for c in COMPONENTS}

    def component_forecast(self, start, days):
        if self.fail:
            raise RuntimeError("model file missing")
        return {c: [{"date": str(start + timedelta(days=i)), "units": 0, "lower": 0, "upper": 0}
                    for i in range(days)] for c in COMPONENTS}


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.setattr("agents.autonomy.agents.routine_inflow", lambda: {})
    monkeypatch.delenv("HEMOSMART_AGENT_EVENTS", raising=False)
    _wipe()
    yield
    _wipe()


def _wipe():
    with SessionLocal() as db:
        for model in (AgentEvent, DonorAlert, AgentCycle):
            db.query(model).delete()
        db.commit()


def _events(run_id=None):
    with SessionLocal() as db:
        q = db.query(AgentEvent).order_by(AgentEvent.id)
        if run_id:
            q = q.filter(AgentEvent.run_id == run_id)
        return [monitor.event_dict(e) for e in q]


def _cycle(**kw):
    return run_cycle(now=utcnow(), forecaster=kw.pop("forecaster", StubForecaster()), simulate_responses=False, **kw)


# ----------------------------------------------------------------------
# Supply loop
# ----------------------------------------------------------------------
def test_cycle_emits_run_and_step_events():
    cycle = _cycle()
    events = _events(f"cycle:{cycle['id']}")
    types_ = [e["type"] for e in events]
    assert types_[0] == "run_started" and types_[-1] == "run_finished"
    assert types_.count("step_started") == 7 and types_.count("step_finished") == 7
    assert {e["layer"] for e in events} == {"supply_loop"}
    finished = [e for e in events if e["type"] == "step_finished"]
    assert all(e["duration_ms"] is not None for e in finished)
    assert events[-1]["status"] == cycle["status"]


def test_forecast_failure_is_an_error_step_and_a_degraded_run():
    cycle = _cycle(forecaster=StubForecaster(fail=True))
    events = _events(f"cycle:{cycle['id']}")
    forecast = next(e for e in events if e["type"] == "step_finished" and e["agent"] == "Forecast")
    assert forecast["status"] == "error" and forecast["data"] == {"exc_class": "RuntimeError"}
    assert events[-1]["status"] == "degraded" == cycle["status"]


def test_planner_failure_skips_the_dependent_steps(monkeypatch):
    def boom(self, bb, db):
        raise ValueError("no plan")
    monkeypatch.setattr("agents.autonomy.orchestrator.PlannerAgent.run", boom)
    cycle = _cycle()
    finished = {(e["agent"], e["step"]): e["status"] for e in _events(f"cycle:{cycle['id']}")
                if e["type"] == "step_finished"}
    assert finished[("Planner", "plan supply")] == "error"
    assert finished[("Donor", "alert donors")] == "skipped"
    assert finished[("Logistics", "redistribute")] == "skipped"


def test_simulation_and_the_off_switch_write_nothing(monkeypatch):
    for _ in range(50):
        run_cycle(now=utcnow(), mode="simulation", trigger="simulation", forecaster=StubForecaster(),
                  simulate_responses=False)
    assert _events() == []
    monkeypatch.setenv("HEMOSMART_AGENT_EVENTS", "0")
    _cycle()
    assert _events() == []


# ----------------------------------------------------------------------
# Case pipeline
# ----------------------------------------------------------------------
def _fake_pipeline(monkeypatch, fail_at=None):
    def run_full_pipeline(pdf_path, progress=None, allow_dispatch=True, confirmed=None):
        for agent in cases.AGENTS:
            progress(agent, "running", None)
            if agent == fail_at:
                raise ValueError("hemoglobin=7.2 is not valid")
            progress(agent, "fell_back" if agent == "RAG" else "verified", {"hemoglobin": 7.2})
        return {"action": {"recommended": ["O- PRBC"]},
                "patient_record": {"hemoglobin": 7.2, "platelets": 65000, "INR": 1.9, "age": 58,
                                   "surgery_type": "Emergency", "blood_group": "O-"}}
    monkeypatch.setitem(sys.modules, "agents.crew_agents",
                        types.SimpleNamespace(run_full_pipeline=run_full_pipeline))
    monkeypatch.setattr(cases, "predict_transfusion", lambda record: {"transfusion_needed": True})
    monkeypatch.setattr(cases, "persist_prediction", lambda db, record, prediction, created_by: {"patient_id": 1})


def _run_case(job):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(b"%PDF")
    cases._run_lock.acquire()
    cases._run(job, tmp.name, user_id=None)


def _job(dispatch_allowed=False):
    return {"job_id": "abc123def456", "status": "queued", "dispatch_allowed": dispatch_allowed,
            "steps": [{"agent": a, "status": "pending", "detail": None} for a in cases.AGENTS],
            "result": None, "error": None, "llm_calls": None, "seconds": None}


def test_case_progress_becomes_step_events(monkeypatch):
    _fake_pipeline(monkeypatch)
    _run_case(_job())
    events = _events("case:abc123def456")
    finished = {e["agent"]: e["status"] for e in events if e["type"] == "step_finished"}
    assert finished == {a: ("fell_back" if a == "RAG" else "verified") for a in cases.AGENTS}
    assert events[-1]["type"] == "run_finished" and events[-1]["status"] == "done"
    assert events[-1]["data"]["recommended_alerts"] == 1
    assert all("hemoglobin" not in (e["data"] or {}) for e in events)


def test_a_failing_case_step_is_recorded_as_an_error(monkeypatch):
    _fake_pipeline(monkeypatch, fail_at="Inventory")
    _run_case(_job())
    events = _events("case:abc123def456")
    inventory = next(e for e in events if e["type"] == "step_finished" and e["agent"] == "Inventory")
    assert inventory["status"] == "error" and inventory["data"] == {"exc_class": "ValueError"}
    assert events[-1]["status"] == "failed"
    assert "7.2" not in str(events)  # the exception message never reaches the table


def test_rate_limit_wait_is_reported(monkeypatch):
    crew = pytest.importorskip("agents.crew_agents")
    attempts = []

    class FakeCrew:
        def __init__(self, **kw):
            pass

        def kickoff(self):
            attempts.append(1)
            if len(attempts) == 1:
                raise Exception("RateLimitError: Rate limit reached ... Please try again in 12s.")
            return types.SimpleNamespace(tasks_output=["ok"])

    monkeypatch.setattr(crew, "Crew", FakeCrew)
    monkeypatch.setattr(crew.time, "sleep", lambda s: None)
    agent = types.SimpleNamespace(tools=None)
    with telemetry.run_context("case:wait", "case_pipeline"):
        value, ok, _ = crew._run_isolated_task("Prediction", agent, None, False, lambda out: True)
    assert ok and value is True
    (wait,) = [e for e in _events("case:wait") if e["type"] == "llm_wait"]
    assert wait["agent"] == "Prediction" and wait["data"] == {"wait_s": 13, "attempt": 1}


# ----------------------------------------------------------------------
# Redaction (D72)
# ----------------------------------------------------------------------
def test_emit_keeps_only_allow_listed_keys():
    telemetry.emit("case:x", "case_pipeline", "Extraction", "Extraction", "step_finished", status="error",
                   hemoglobin=7.2, platelets=65000, filename="Ravi_Kumar_CBC.pdf", pdf_path="/tmp/Ravi.pdf",
                   donor_id="D-0042", phone="+91 98765 43210", message="hemoglobin 7.2 out of range",
                   exc_class="ValidationError", blood_group="O-", nested={"x": 1}, units=2)
    (row,) = _events()
    assert row["data"] == {"exc_class": "ValidationError", "blood_group": "O-", "units": 2}


def test_a_patient_named_upload_leaves_no_trace(users, monkeypatch):
    _fake_pipeline(monkeypatch)
    response = client.post("/api/cases", headers=auth_header(users["Hospital Staff"]),
                           files={"file": ("Ravi_Kumar_CBC.pdf", b"%PDF-1.4 fake", "application/pdf")})
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    for _ in range(100):
        if cases._jobs[job_id]["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert cases._jobs[job_id]["status"] == "done"
    with SessionLocal() as db:
        audit = [(a.action, a.resource_id, a.details) for a in db.query(AuditLog)]
    assert ("agent_case_started", job_id, {"size_bytes": 13, "dispatch_allowed": False}) in audit
    for blob in (str(audit), str(_events()), str(cases._jobs[job_id])):
        assert "Ravi" not in blob and "Kumar" not in blob


# ----------------------------------------------------------------------
# Permissions (D74)
# ----------------------------------------------------------------------
ENDPOINTS = ("/api/agents/overview", "/api/agents/runs", "/api/agents/runs/cycle:1", "/api/agents/metrics")


@pytest.mark.parametrize("path", ENDPOINTS)
@pytest.mark.parametrize("role, allowed", [
    ("Admin", True), ("Blood Bank Manager", True), ("Auditor", True),
    ("Hospital Staff", False), ("Donor Coordinator", False),
])
def test_monitor_permissions(users, role, allowed, path):
    telemetry.emit("cycle:1", "supply_loop", telemetry.RUN, None, "run_started", trigger="manual")
    response = client.get(path, headers=auth_header(users[role]))
    if allowed:
        assert response.status_code == 200, response.text
    else:
        assert response.status_code == 403
        assert response.json()["detail"] == denial(role, "agents:monitor")


@pytest.mark.parametrize("path", ENDPOINTS)
def test_monitor_needs_a_token(path):
    assert client.get(path).status_code == 401


def test_overview_after_a_real_cycle(users):
    cycle = _cycle()
    body = client.get("/api/agents/overview", headers=auth_header(users["Auditor"])).json()
    supply = next(l for l in body["layers"] if l["layer"] == "supply_loop")
    assert [c["agent"] for c in supply["cards"]] == list(monitor.LAYERS["supply_loop"]["agents"])
    assert body["has_open_run"] is False
    assert supply["last_run"]["run_id"] == f"cycle:{cycle['id']}" and supply["runs_24h"] == 1
    runs = client.get("/api/agents/runs", headers=auth_header(users["Auditor"])).json()
    assert runs[0]["run_id"] == f"cycle:{cycle['id']}" and runs[0]["steps_done"] == 7
    detail = client.get(f"/api/agents/runs/cycle:{cycle['id']}", headers=auth_header(users["Auditor"])).json()
    assert detail["cycle_id"] == cycle["id"] and len(detail["steps"]) == 7


# ----------------------------------------------------------------------
# Status and alerts, under a fixed clock (D75)
# ----------------------------------------------------------------------
NOW = utcnow().replace(microsecond=0)
_ids = iter(range(1, 10_000))


def ev(type_, ago_s, agent="Extraction", status=None, run_id="case:r1", layer="case_pipeline", step=None, **data):
    return {"id": next(_ids), "run_id": run_id, "layer": layer, "agent": agent,
            "step": step or (agent if type_ != "run_started" else None), "type": type_, "status": status,
            "ts": NOW - timedelta(seconds=ago_s), "duration_ms": 1000 if type_ == "step_finished" else None,
            "llm_calls": None, "data": data}


def card(events, agent="Extraction", layer="case_pipeline", now=NOW):
    runs = monitor.summarise_runs(events)
    return next(c for c in monitor.build_cards(events, runs, now) if c["agent"] == agent and c["layer"] == layer)


def test_status_working_waiting_and_finished():
    start = [ev("run_started", 100, agent=telemetry.RUN), ev("step_started", 90)]
    assert card(start)["status"] == "working"
    assert card(start)["current"]["done"] == 0 and card(start)["current"]["total"] == 6

    waiting = start + [ev("llm_wait", 5, wait_s=12)]
    assert card(waiting)["status"] == "waiting" and card(waiting)["retry_in_s"] == 7
    assert card(start + [ev("llm_wait", 60, wait_s=12)])["status"] == "working"  # retry long due

    assert card(start + [ev("step_finished", 1, status="verified")])["status"] == "idle"
    assert card(start + [ev("step_finished", 1, status="fell_back")])["status"] == "degraded"
    assert card(start + [ev("step_finished", 1, status="error")])["status"] == "failed"
    assert card([])["status"] == "idle"


def test_a_step_whose_run_was_closed_under_it_is_failed():
    events = [ev("run_started", 100, agent=telemetry.RUN), ev("step_started", 90),
              ev("run_finished", 10, agent=telemetry.RUN, status="failed")]
    c = card(events)
    assert c["status"] == "failed" and c["current"] is None


def test_stuck_alert_uses_the_layer_threshold():
    events = [ev("run_started", 400, agent=telemetry.RUN), ev("step_started", 390)]  # 6.5 min
    cards = monitor.build_cards(events, monitor.summarise_runs(events), NOW)
    off = {"state": "off"}
    assert [a["kind"] for a in monitor.build_alerts(cards, off, [], events, NOW)] == ["stuck"]
    young = [ev("run_started", 300, agent=telemetry.RUN), ev("step_started", 290)]  # 4.8 min
    cards = monitor.build_cards(young, monitor.summarise_runs(young), NOW)
    assert monitor.build_alerts(cards, off, [], young, NOW) == []


def test_scheduler_heartbeat_states():
    beat = [ev("heartbeat", 66 * 60, agent="Scheduler", run_id="scheduler", layer="scheduler", status="ok")]
    assert monitor.scheduler_state(beat, 60, NOW)["state"] == "late"
    assert monitor.scheduler_state(beat, 90, NOW)["state"] == "ok"
    assert monitor.scheduler_state(beat, 0, NOW)["state"] == "off"
    assert monitor.scheduler_state([], 60, NOW)["state"] == "unknown"
    alerts = monitor.build_alerts([], monitor.scheduler_state(beat, 60, NOW), [], [], NOW)
    assert [a["kind"] for a in alerts] == ["missed_tick"]
    assert monitor.build_alerts([], monitor.scheduler_state(beat, 0, NOW), [], [], NOW) == []


def test_repeated_fallback_needs_three_runs_in_a_row():
    def runs(statuses):
        out = []
        for i, status in enumerate(statuses):
            out += [ev("run_started", 1000 - i * 100, agent=telemetry.RUN, run_id=f"case:{i}"),
                    ev("step_finished", 990 - i * 100, status=status, run_id=f"case:{i}"),
                    ev("run_finished", 980 - i * 100, agent=telemetry.RUN, run_id=f"case:{i}", status="done")]
        return out
    kinds = lambda e: [a["kind"] for a in monitor.build_alerts([], {"state": "off"}, [], e, NOW)]
    assert kinds(runs(["fell_back"] * 3)) == ["repeated_fallback"]
    assert kinds(runs(["fell_back", "verified", "fell_back", "fell_back"])) == []


def test_orphaned_cycle_is_flagged_then_marked_failed_at_startup():
    with SessionLocal() as db:
        db.add(AgentCycle(mode="live", trigger="manual", clock=NOW, started_at=NOW - timedelta(minutes=11),
                          status="running"))
        db.add(AgentCycle(mode="live", trigger="manual", clock=NOW, started_at=NOW - timedelta(minutes=2),
                          status="running"))
        db.commit()
        orphans = monitor.orphaned_cycles(db, NOW)
    assert len(orphans) == 1
    assert [a["kind"] for a in monitor.build_alerts([], {"state": "off"}, orphans, [], NOW)] == ["orphaned_cycle"]
    assert mark_orphaned_cycles(now=NOW) == 1
    with SessionLocal() as db:
        assert sorted(c.status for c in db.query(AgentCycle)) == ["failed", "running"]


def test_open_runs_are_closed_at_startup():
    telemetry.emit("case:dead", "case_pipeline", telemetry.RUN, None, "run_started")
    telemetry.emit("case:done", "case_pipeline", telemetry.RUN, None, "run_started")
    telemetry.emit("case:done", "case_pipeline", telemetry.RUN, None, "run_finished", status="done")
    assert telemetry.close_open_runs() == 1
    runs = monitor.summarise_runs(_events())
    assert runs["case:dead"]["status"] == "failed" and not runs["case:dead"]["open"]


def test_timeline_has_one_bar_per_step_with_its_waits():
    events = [ev("run_started", 100, agent=telemetry.RUN), ev("step_started", 90), ev("llm_wait", 80, wait_s=12),
              ev("step_finished", 60, status="verified"),
              ev("step_finished", 50, agent="Action", status="skipped")]
    steps = monitor.run_timeline(events)
    assert [(s["agent"], s["status"]) for s in steps] == [("Extraction", "verified"), ("Action", "skipped")]
    assert steps[0]["waits"] == [{"at": (NOW - timedelta(seconds=80)).isoformat(), "seconds": 12}]


# ----------------------------------------------------------------------
# Retention
# ----------------------------------------------------------------------
def test_prune_removes_only_old_rows():
    telemetry.emit("cycle:1", "supply_loop", telemetry.RUN, None, "run_started")
    telemetry.emit("cycle:2", "supply_loop", telemetry.RUN, None, "run_started")
    with SessionLocal() as db:
        old = db.query(AgentEvent).filter(AgentEvent.run_id == "cycle:1").one()
        old.ts = utcnow() - timedelta(days=15)
        db.commit()
    assert telemetry.prune(days=14) == 1
    assert [e["run_id"] for e in _events()] == ["cycle:2"]
