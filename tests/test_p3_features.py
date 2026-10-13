"""P3: identifier masking (NFR-03), the confirmed PDF flow (FR-03), and
the RAG faithfulness checker's number handling."""

import os
from datetime import date

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routes import donors, predict
from conftest import auth_header
from db.database import SessionLocal
from db.models import Donor, Patient
from schema.patient_schema import PatientRecord
from schema.privacy import mask_identifier

app = FastAPI()
app.include_router(predict.router)
app.include_router(donors.router)
client = TestClient(app)

PATIENT = dict(hemoglobin=6.8, platelets=210_000, INR=1.1, age=64, surgery_type="General", blood_group="O+")


# ----------------------------------------------------------------------
# Masking
# ----------------------------------------------------------------------
@pytest.mark.parametrize("value, masked", [
    ("+91XXXXX01358", "•••••••••1358"),
    ("1234", "••••"),       # too short to show anything
    ("", ""),
    (None, None),
])
def test_mask_identifier(value, masked):
    assert mask_identifier(value) == masked


def test_donor_list_masks_phone_numbers(users):
    with SessionLocal() as db:
        db.query(Donor).filter(Donor.donor_id == "D9001").delete()
        db.add(Donor(donor_id="D9001", name="Test Donor", phone="+919876543210", blood_type="O-",
                     last_donation_date=date(2026, 1, 1), alerts_sent=0, alerts_responded=0))
        db.commit()
    res = client.get("/api/donors", headers=auth_header(users["Donor Coordinator"]))
    assert res.status_code == 200
    phones = [d["phone"] for d in res.json() if d["donor_id"] == "D9001"]
    assert phones == ["•••••••••3210"]
    with SessionLocal() as db:
        db.query(Donor).filter(Donor.donor_id == "D9001").delete()
        db.commit()


# ----------------------------------------------------------------------
# Confirmed PDF flow: extract (no prediction, nothing saved) -> confirm
# ----------------------------------------------------------------------
def test_pdf_extract_returns_values_without_predicting(users, monkeypatch):
    monkeypatch.setattr(predict, "_extract_pdf", lambda path: PatientRecord(**PATIENT, source_format="pdf"))
    with SessionLocal() as db:
        before = db.query(Patient).count()
    res = client.post("/api/predict/pdf/extract", files={"file": ("r.pdf", b"%PDF-1.4", "application/pdf")},
                      headers=auth_header(users["Hospital Staff"]))
    assert res.status_code == 200
    body = res.json()
    assert body["hemoglobin"] == 6.8 and body["source_format"] == "pdf"
    assert "transfusion_needed" not in body
    with SessionLocal() as db:
        assert db.query(Patient).count() == before  # nothing persisted until confirmed


def test_confirmed_pdf_values_keep_their_provenance(users):
    res = client.post("/api/predict", json={**PATIENT, "source_format": "pdf"},
                      headers=auth_header(users["Hospital Staff"]))
    assert res.status_code == 200
    with SessionLocal() as db:
        patient = db.query(Patient).filter(Patient.id == res.json()["patient_id"]).one()
        assert patient.source_format == "pdf"


def test_source_format_is_restricted(users):
    res = client.post("/api/predict", json={**PATIENT, "source_format": "csv"},
                      headers=auth_header(users["Hospital Staff"]))
    assert res.status_code == 422


# ----------------------------------------------------------------------
# RAG faithfulness checker (evaluation/rag_faithfulness.py)
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def faithfulness():
    # The evaluation package repoints DATABASE_URL at its own sandbox;
    # restore ours so later tests (and subprocesses) are unaffected.
    saved = os.environ["DATABASE_URL"]
    import evaluation.rag_faithfulness as module
    os.environ["DATABASE_URL"] = saved
    return module


@pytest.mark.parametrize("text, numbers", [
    ("32 000 /µL", {32000.0}),
    ("32 × 10³ /µL", {32000.0}),
    ("≥ 50 × 10^9/L", {50000.0}),
    ("below 50 x 109/L", {50000.0}),       # PDF-flattened superscript
    ("260,000 and INR 1.5", {260000.0, 1.5}),
    ("give 2 units", set()),                # small bare integers aren't threshold claims
])
def test_checker_parses_number_formats(faithfulness, text, numbers):
    assert faithfulness._numbers(text) == numbers


def test_checker_flags_misattribution_and_contradiction(faithfulness):
    case = dict(hemoglobin=14.1, platelets=260_000, INR=1.1, age=47, surgery_type="Orthopedic")
    facts = "INR 1.1 is AT OR BELOW 1.5"
    sources = ["fresh frozen plasma indications: coagulation factor deficiency"]
    bad = faithfulness.check("According to WHO guidance, FFP is indicated when INR exceeds 1.5 [1]. "
                             "The INR is above its threshold.", case, facts, sources)
    assert bad["misattributed"] == [1.5]
    assert bad["contradictions"] == ["ffp"]
    good = faithfulness.check("HemoSmart's protocol threshold for INR is 1.5; the INR of 1.1 is at or below it.",
                              case, facts, sources)
    assert good["misattributed"] == [] and good["contradictions"] == []
