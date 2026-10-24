"""
HemoSmart - Agent Tool Functions
------------------------------------
Plain Python functions containing the actual logic each agent calls --
shared by the REST API, the chat agent, the MCP server, the CrewAI case
pipeline and the autonomous loop (agents/autonomy/), so every interface
gives the same answer.

    predict_transfusion()      component-level prediction + dose + urgency + SHAP
    generate_explanation()     RAG explanation grounded in the WHO handbook
    check_inventory()          unit-level stock status (bloodbank/stock.py)
    get_supply_outlook()       forecast + shortage/wastage projection
    send_donor_alert()         bandit-selected donor alerts
    extract_from_pdf()         CBC report -> PatientRecord

Read helpers shared by the REST routes and the chat assistant, so a
role sees the same records on a page and in chat:

    list_recent_patients()     patients + their latest prediction
    list_donors_masked()       donor pool, phones masked (NFR-03)
    recent_donor_alerts()      alerts, optionally only unanswered ones
    latest_autonomy_cycle()    what the last agent cycle did
    recent_audit_entries()     audit log rows, newest first
    answer_guideline_question() WHO handbook Q&A (RAG), no patient data
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import pickle
from contextlib import contextmanager

import numpy as np
import xgboost as xgb

from bloodbank.catalog import BLOOD_GROUPS, COMPONENTS
from db.database import SessionLocal
from schema.clinical_thresholds import recommended_units, urgency

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


@contextmanager
def _session(db=None):
    """Uses the caller's DB session (a route's) or opens one (chat, MCP)."""
    if db is not None:
        yield db
    else:
        with SessionLocal() as own:
            yield own


# ----------------------------------------------------------------------
# Prediction Agent's tool
# ----------------------------------------------------------------------
_bundle = None
_label_encoder = None


def _load_models():
    """
    Loads the per-component XGBoost models + label encoder once and
    caches them at module level (unpickling per request would add real
    latency to every API call).
    """
    global _bundle, _label_encoder
    if _bundle is None or _label_encoder is None:
        with open(os.path.join(MODEL_DIR, "xgb_components.pkl"), "rb") as f:
            _bundle = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
            _label_encoder = pickle.load(f)
    return _bundle, _label_encoder


def _shap_contributions(model, features: list, names: list) -> dict:
    """Exact TreeSHAP contributions (log-odds) for one patient, computed
    by XGBoost itself -- no separate `shap` dependency at serving time.
    Positive values push toward "needs this component"."""
    dmatrix = xgb.DMatrix(np.array([features], dtype=float), feature_names=names)
    contribs = model.get_booster().predict(dmatrix, pred_contribs=True)[0]
    out = {name: round(float(v), 4) for name, v in zip(names, contribs[:-1])}
    out["_base"] = round(float(contribs[-1]), 4)
    return out


def predict_transfusion(patient_record) -> dict:
    """
    patient_record: a PatientRecord instance (schema.patient_schema)

    Returns transfusion_needed, confidence (= P(any component needed)),
    per-component probability + dose, urgency, and per-component SHAP.
    """
    bundle, le = _load_models()
    names = bundle["features"]
    features = patient_record.to_model_features(le)

    probs = {
        c: float(bundle["models"][c].predict_proba(np.array([features], dtype=float))[0][1])
        for c in COMPONENTS
    }
    p_any = 1 - math.prod(1 - p for p in probs.values())
    transfusion_needed = p_any >= 0.5
    needed = {c for c, p in probs.items() if p >= 0.5}
    if transfusion_needed and not needed:
        # Several individually-uncertain components can add up to a
        # likely transfusion -- recommend the most likely one rather
        # than "transfusion needed" with nothing to give.
        needed = {max(probs, key=probs.get)}

    r = patient_record
    components = [
        {
            "component": c,
            "probability": round(probs[c], 4),
            "needed": c in needed,
            "units": recommended_units(c, r.hemoglobin, r.platelets, r.INR, r.surgery_type) if c in needed else 0,
        }
        for c in COMPONENTS
    ]

    return {
        "transfusion_needed": transfusion_needed,
        "confidence": round(p_any, 4),
        "components": components,
        "urgency": urgency(r.hemoglobin, r.platelets, r.INR, r.surgery_type, transfusion_needed),
        "shap": {c: _shap_contributions(bundle["models"][c], features, names) for c in COMPONENTS},
        "blood_group": r.blood_group,
    }


# ----------------------------------------------------------------------
# Explanation Agent's tool
# ----------------------------------------------------------------------
_explanation_generator = None


def set_explanation_generator(generator) -> None:
    """Shares one loaded ExplanationGenerator (embedding model + FAISS
    index) across the process -- the API sets it once at startup."""
    global _explanation_generator
    _explanation_generator = generator


def generate_explanation(
    patient_record, prediction: bool,
    llm_provider: str = os.environ.get("HEMOSMART_LLM_PROVIDER", "groq"),
) -> str:
    """
    Wraps rag.explain.ExplanationGenerator -- retrieves WHO guideline
    context via FAISS, checks thresholds deterministically, and asks
    the LLM to phrase the explanation, citing handbook pages. Built once
    and reused: loading the embedding model per call cost seconds each.
    """
    global _explanation_generator
    if _explanation_generator is None:
        from rag.explain import ExplanationGenerator
        _explanation_generator = ExplanationGenerator(llm_provider=llm_provider)
    return _explanation_generator.explain(patient_record, prediction)


# ----------------------------------------------------------------------
# Inventory Agent's tool
# ----------------------------------------------------------------------
def check_inventory(blood_group: str = None, component: str = None) -> dict:
    """
    Unit-level stock status. With no arguments: the full group x
    component grid plus everything below par. With a group (and
    optionally a component): just that slice.
    """
    from bloodbank.stock import stock_summary, utcnow

    if blood_group and blood_group not in BLOOD_GROUPS:
        raise ValueError(f"Unknown blood group: {blood_group}")
    if component and component not in COMPONENTS:
        raise ValueError(f"Unknown component: {component}")

    with SessionLocal() as db:
        summary = stock_summary(db, utcnow())
    if not blood_group:
        return summary

    cells = summary["inventory"][blood_group]
    if component:
        cells = {component: cells[component]}
    return {
        "blood_group": blood_group,
        "components": cells,
        "low_stock": any(c["status"] != "Sufficient" for c in cells.values()),
    }


# ----------------------------------------------------------------------
# Forecast / Planner tools
# ----------------------------------------------------------------------
def get_supply_outlook(days: int = 7) -> dict:
    """Forecast demand + day-by-day projection of shortages and of units
    that will expire unused (see bloodbank/outlook.py)."""
    from bloodbank.outlook import compute_outlook
    from bloodbank.stock import utcnow

    with SessionLocal() as db:
        return compute_outlook(db, utcnow(), days)


# ----------------------------------------------------------------------
# Extraction Agent's tool
# ----------------------------------------------------------------------
def extract_from_pdf(
    pdf_path: str,
    llm_provider: str = os.environ.get("HEMOSMART_LLM_PROVIDER", "groq"),
):
    """
    Wraps adapters.pdf_adapter.PDFReportAdapter -- extracts CBC report
    text, converts it to structured fields via the LLM, and returns a
    validated PatientRecord. Returns None if extraction fails (caller
    should fall back to manual entry, per the adapter's design).
    """
    from adapters.pdf_adapter import PDFReportAdapter
    from rag.llm_client import get_llm_client

    llm = get_llm_client(provider=llm_provider)
    adapter = PDFReportAdapter(llm_client=llm)
    records = adapter.safe_parse(pdf_path)
    return records[0] if records else None


# ----------------------------------------------------------------------
# Donor Alert Agent's tool
# ----------------------------------------------------------------------
def send_donor_alert(blood_type: str, component: str = "PRBC", k: int = 3) -> str:
    """
    Sends personalized alerts to the top eligible, most-likely-to-respond
    donors, selected via a Multi-Armed Bandit (Thompson Sampling) -- see
    agents/donor_module.py. `component` decides the donation asked for:
    whole blood for PRBC/FFP, apheresis for platelets.
    """
    from agents.donor_module import send_intelligent_donor_alerts
    result = send_intelligent_donor_alerts(blood_type, component=component, k=k)
    return result["message"]


# ----------------------------------------------------------------------
# Read helpers (routes + chat)
# ----------------------------------------------------------------------
def prediction_to_dict(pr) -> dict:
    return {
        "id": pr.id,
        "transfusion_needed": pr.transfusion_needed,
        "confidence": pr.confidence,
        "urgency": pr.urgency,
        "components": pr.components,
        "explanation": pr.explanation or None,
        "citations": pr.citations,
        "created_at": pr.created_at.isoformat(),
    }


def list_recent_patients(limit: int = 50, db=None) -> list:
    """Most recent patients first, each with their latest prediction."""
    from db.models import Patient

    with _session(db) as s:
        patients = s.query(Patient).order_by(Patient.created_at.desc()).limit(limit).all()
        return [
            {
                "id": p.id, "hemoglobin": p.hemoglobin, "platelets": p.platelets, "inr": p.inr, "age": p.age,
                "surgery_type": p.surgery_type, "blood_group": p.blood_group, "source_format": p.source_format,
                "source_hospital": p.source_hospital, "created_at": p.created_at.isoformat(),
                "latest_prediction": prediction_to_dict(p.predictions[-1]) if p.predictions else None,
            }
            for p in patients
        ]


def list_donors_masked(blood_type: str = None) -> list:
    """The donor pool with phones masked to the last 4 digits and the
    bandit's current belief about each donor's reply rate."""
    from agents.donor_module import load_donors, posterior_mean
    from schema.privacy import mask_identifier

    donors = load_donors()
    if blood_type:
        donors = [d for d in donors if d["blood_type"] == blood_type]
    return [{**d, "phone": mask_identifier(d["phone"]), "expected_response_rate": round(posterior_mean(d), 3)}
            for d in donors]


def recent_donor_alerts(limit: int = 10, pending_only: bool = False, db=None) -> list:
    from db.models import DonorAlert

    with _session(db) as s:
        q = s.query(DonorAlert)
        if pending_only:
            q = q.filter(DonorAlert.responded.is_(None))
        rows = q.order_by(DonorAlert.sent_at.desc()).limit(limit).all()
        return [
            {
                "id": r.id, "donor_id": r.donor_id, "blood_type": r.blood_type, "component": r.component,
                "sent_at": r.sent_at.isoformat(), "method": r.method, "responded": r.responded,
                "responded_at": r.responded_at.isoformat() if r.responded_at else None, "cycle_id": r.cycle_id,
            }
            for r in rows
        ]


def latest_autonomy_cycle(db=None):
    from agents.autonomy.orchestrator import cycle_to_dict
    from db.models import AgentCycle

    with _session(db) as s:
        last = s.query(AgentCycle).order_by(AgentCycle.id.desc()).first()
        return cycle_to_dict(last) if last else None


def recent_audit_entries(limit: int = 100, action: str = None, db=None) -> list:
    from db.models import AuditLog

    with _session(db) as s:
        q = s.query(AuditLog)
        if action:
            q = q.filter(AuditLog.action == action)
        rows = q.order_by(AuditLog.timestamp.desc()).limit(limit).all()
        return [
            {
                "id": r.id, "user_id": r.user_id, "action": r.action, "resource_type": r.resource_type,
                "resource_id": r.resource_id, "ip_address": r.ip_address,
                "timestamp": r.timestamp.isoformat(), "details": r.details,
            }
            for r in rows
        ]


GUIDELINE_PROMPT = """{role_scope}
Answer the question using ONLY the WHO handbook excerpts below. Cite
excerpts as [n]. If the excerpts don't answer it, say so. Do not give
advice about a specific patient. Keep it under 120 words.

Excerpts:
{context}

Question: {question}
"""


def answer_guideline_question(question: str, llm=None, role_scope: str = "") -> dict:
    """RAG over the WHO handbook for general questions (no patient data).
    If the LLM fails, returns the retrieved excerpts with degraded=True."""
    global _explanation_generator
    if _explanation_generator is None:
        from rag.explain import ExplanationGenerator
        _explanation_generator = ExplanationGenerator()
    citations = _explanation_generator.retriever.cite(question, k=3)
    context = "\n\n".join(f"[{c['ref']}] (page {c['page']}) {c['excerpt']}" for c in citations)
    try:
        llm = llm or _explanation_generator.llm
        answer = llm.generate(GUIDELINE_PROMPT.format(role_scope=role_scope, context=context, question=question))
        return {"answer": answer.strip(), "citations": citations, "degraded": False}
    except Exception:
        return {"answer": "The AI model is unavailable, so here are the most relevant handbook passages.",
                "citations": citations, "degraded": True}
