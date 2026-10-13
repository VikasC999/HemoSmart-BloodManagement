"""
The autonomous loop and the stock service, end to end against the test
database, with a deterministic stub forecaster and no LLM/RAG.
"""

import random
from datetime import timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agents.autonomy.orchestrator import run_cycle
from agents.donor_module import simulate_donors
from bloodbank.catalog import BLOOD_GROUPS, COMPONENTS
from bloodbank.stock import (
    cancel_transfusion,
    confirm_transfusion,
    receive_units,
    reserve_for_transfusion,
    sweep_expired,
    utcnow,
    wastage_stats,
)
from backend.routes import predict, transfusions
from conftest import auth_header
from db.database import SessionLocal
from db.models import AgentCycle, BloodUnit, Donor, DonorAlert, Patient, Prediction, Transfusion

HORIZON = 7


class StubForecaster:
    model_name = "stub"

    def __init__(self, by_key=None, fail=False):
        self.by_key = by_key or {}
        self.fail = fail

    def group_forecast(self, start, days, totals=None):
        return {(g, c): [self.by_key.get((g, c), 0.0)] * days for g in BLOOD_GROUPS for c in COMPONENTS}

    def component_forecast(self, start, days):
        if self.fail:
            raise RuntimeError("model file missing")
        by_key = self.group_forecast(start, days)
        return {c: [{"date": str(start + timedelta(days=i)), "units": sum(by_key[(g, c)][i] for g in BLOOD_GROUPS),
                     "lower": 0, "upper": 0} for i in range(days)] for c in COMPONENTS}


@pytest.fixture(autouse=True)
def clean_db(monkeypatch):
    # Routine inflow would blur these hand-built scenarios
    monkeypatch.setattr("agents.autonomy.agents.routine_inflow", lambda: {})
    _wipe()
    yield
    _wipe()


def _wipe():
    with SessionLocal() as db:
        for model in (DonorAlert, BloodUnit, Transfusion, Prediction, Patient, AgentCycle, Donor):
            db.query(model).delete()
        db.commit()


def _add(group, component, count, age_days, now):
    with SessionLocal() as db:
        units = receive_units(db, group, component, count, now - timedelta(days=age_days))
        ids = [u.unit_id for u in units]
        db.commit()
        return ids


def _status(unit_ids):
    with SessionLocal() as db:
        return {u.unit_id: u.status for u in db.query(BloodUnit).filter(BloodUnit.unit_id.in_(unit_ids))}


# ----------------------------------------------------------------------
# Stock service
# ----------------------------------------------------------------------
def _patient(db, group="A+"):
    p = Patient(hemoglobin=7, platelets=200000, inr=1, age=40, surgery_type="General", blood_group=group)
    db.add(p)
    db.flush()
    return p


def test_reserve_uses_near_expiry_compatible_units_and_records_shortfall():
    now = utcnow()
    expiring = _add("O-", "PRBC", 1, age_days=37, now=now)   # 5 days left: near expiry
    fresh_own = _add("A+", "PRBC", 1, age_days=1, now=now)
    with SessionLocal() as db:
        t = reserve_for_transfusion(db, _patient(db).id, "A+", {"PRBC": 3}, now)
        db.commit()
        reserved = [u.unit_id for u in db.query(BloodUnit).filter(BloodUnit.transfusion_id == t.id)]
        assert set(reserved) == set(expiring + fresh_own)
        assert t.shortfall == {"PRBC": 1}


def test_confirm_issues_and_cancel_releases():
    now = utcnow()
    ids = _add("B+", "FFP", 2, age_days=10, now=now)
    with SessionLocal() as db:
        pid = _patient(db, "B+").id
        t1 = reserve_for_transfusion(db, pid, "B+", {"FFP": 1}, now)
        t2 = reserve_for_transfusion(db, pid, "B+", {"FFP": 1}, now)
        confirm_transfusion(db, t1.id, now)
        cancel_transfusion(db, t2.id, now)
        db.commit()
        with pytest.raises(ValueError):
            confirm_transfusion(db, t1.id, now)  # already issued
    assert sorted(_status(ids).values()) == ["available", "issued"]


def test_sweep_marks_expired_units_as_wastage():
    now = utcnow()
    old = _add("O+", "Platelets", 2, age_days=6, now=now)   # past 5-day shelf life
    ok = _add("O+", "Platelets", 1, age_days=1, now=now)
    with SessionLocal() as db:
        assert {u.unit_id for u in sweep_expired(db, now)} == set(old)
        db.commit()
        stats = wastage_stats(db, now - timedelta(hours=1), now)
    assert stats["totals"]["expired"] == 2
    assert _status(ok) == {ok[0]: "available"}


# ----------------------------------------------------------------------
# Autonomous cycle
# ----------------------------------------------------------------------
def test_cycle_sweeps_alerts_and_redistributes():
    now = utcnow()
    expired = _add("A+", "PRBC", 1, age_days=43, now=now)                 # already past expiry
    surplus = _add("B+", "Platelets", 4, age_days=4.2, now=now)           # ~19h left, no demand
    forecast = StubForecaster({("O-", "PRBC"): 1.0})                      # O- demand, zero O- stock

    cycle = run_cycle(now=now, forecaster=forecast, simulate_responses=False)

    assert cycle["status"] == "ok"
    assert cycle["summary"]["expired_units"] == 1
    assert cycle["summary"]["shortages"] >= 1
    assert cycle["summary"]["donors_alerted"] > 0
    assert _status(expired)[expired[0]] == "expired"
    assert set(_status(surplus).values()) == {"transferred"}
    kinds = {(e["agent"], e["kind"]) for e in cycle["events"]}
    assert {("Inventory", "act"), ("Planner", "decide"), ("Donor", "act"), ("Logistics", "act")} <= kinds
    with SessionLocal() as db:
        alerts = db.query(DonorAlert).all()
        assert all(a.blood_type == "O-" and a.cycle_id == cycle["id"] for a in alerts)


def test_donor_replies_feed_the_bandit_and_restock():
    now = utcnow()
    forecast = StubForecaster({("O-", "PRBC"): 1.0})
    seeded = {d["donor_id"]: d for d in simulate_donors()}  # the pool the cycle seeds
    first = run_cycle(now=now, forecaster=forecast, simulate_responses=True, rng=random.Random(1))
    alerted = first["summary"]["donors_alerted"]
    assert alerted > 0

    second = run_cycle(now=now + timedelta(hours=6), forecaster=forecast,
                       simulate_responses=True, rng=random.Random(1))

    assert second["summary"]["donor_replies"] == alerted
    with SessionLocal() as db:
        donated = db.query(BloodUnit).filter(BloodUnit.source == "donation").count()
        alerts = db.query(DonorAlert).all()
        donors = {d.donor_id: d for d in db.query(Donor)}
    responders = {a.donor_id for a in alerts if a.responded}
    assert donated == second["summary"]["units_donated"] == 2 * len(responders)  # whole blood -> PRBC + FFP

    # Bandit bookkeeping: each alert is one trial (counted when sent, so
    # still-pending cycle-2 alerts count too), each reply one success.
    for donor_id in {a.donor_id for a in alerts}:
        d, s = donors[donor_id], seeded[donor_id]
        mine = [a for a in alerts if a.donor_id == donor_id]
        assert d.alerts_sent == s["alerts_sent"] + len(mine)
        assert d.alerts_responded == s["alerts_responded"] + sum(bool(a.responded) for a in mine)
        if donor_id in responders:
            assert d.last_donation_date == (now + timedelta(hours=6)).date()  # eligibility clock reset


def test_forecast_failure_falls_back_and_cycle_completes():
    cycle = run_cycle(now=utcnow(), forecaster=StubForecaster(fail=True), simulate_responses=False)
    assert cycle["status"] == "degraded"
    kinds = [(e["agent"], e["kind"]) for e in cycle["events"]]
    assert ("Forecast", "error") in kinds
    assert ("Forecast", "fallback") in kinds
    assert ("Planner", "error") not in kinds   # planned on historical averages instead
    assert cycle["brief"]


def test_quiet_when_nothing_to_do():
    now = utcnow()
    _add("O+", "PRBC", 10, age_days=1, now=now)
    cycle = run_cycle(now=now, forecaster=StubForecaster({("O+", "PRBC"): 1.0}), simulate_responses=False)
    assert cycle["summary"]["shortages"] == 0
    assert cycle["summary"]["donors_alerted"] == 0
    assert cycle["summary"]["transferred_units"] == 0


# ----------------------------------------------------------------------
# Prediction -> transfusion workflow over HTTP
# ----------------------------------------------------------------------
app = FastAPI()
app.include_router(predict.router)
app.include_router(transfusions.router)
client = TestClient(app)


def test_prediction_to_issued_transfusion(users):
    now = utcnow()
    _add("O+", "PRBC", 5, age_days=3, now=now)
    _add("O+", "FFP", 3, age_days=30, now=now)
    staff = auth_header(users["Hospital Staff"])

    pred = client.post("/api/predict", headers=staff, json={
        "hemoglobin": 6.8, "platelets": 220000, "INR": 1.9, "age": 61,
        "surgery_type": "General", "blood_group": "O+",
    }).json()
    assert pred["prediction_id"] and pred["urgency"] == "Critical"

    t = client.post("/api/transfusions", headers=staff,
                    json={"patient_id": pred["patient_id"], "prediction_id": pred["prediction_id"]}).json()
    assert t["status"] == "reserved"
    assert t["requested"] == {"PRBC": 2, "FFP": 2}
    assert len(t["units"]) == 4 and t["shortfall"] == {}

    done = client.post(f"/api/transfusions/{t['id']}/confirm", headers=staff).json()
    assert done["status"] == "issued"
    assert {u["status"] for u in done["units"]} == {"issued"}


def test_transfusion_needs_a_blood_group(users):
    staff = auth_header(users["Hospital Staff"])
    pred = client.post("/api/predict", headers=staff, json={
        "hemoglobin": 6.8, "platelets": 220000, "INR": 1.0, "age": 61, "surgery_type": "General",
    }).json()
    res = client.post("/api/transfusions", headers=staff,
                      json={"patient_id": pred["patient_id"], "prediction_id": pred["prediction_id"]})
    assert res.status_code == 422


def test_stale_alerts_expire_before_simulated_replies():
    # Regression: alerts older than the response window are resolved as
    # non-responses first; the simulator must not then see them as still
    # pending (sessions don't autoflush) and try to resolve them twice.
    now = utcnow()
    first = run_cycle(now=now - timedelta(days=5), forecaster=StubForecaster({("O-", "PRBC"): 1.0}),
                      simulate_responses=False)
    assert first["summary"]["donors_alerted"] > 0

    cycle = run_cycle(now=now, forecaster=StubForecaster(), simulate_responses=True, rng=random.Random(3))
    assert cycle["status"] == "ok"
    assert cycle["summary"]["donor_replies"] == first["summary"]["donors_alerted"]
    assert cycle["summary"]["donations"] == 0  # 5-day-old alerts are non-responses, not donations
