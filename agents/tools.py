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
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import pickle

import numpy as np
import xgboost as xgb

from bloodbank.catalog import BLOOD_GROUPS, COMPONENTS
from db.database import SessionLocal
from schema.clinical_thresholds import recommended_units, urgency

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


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
