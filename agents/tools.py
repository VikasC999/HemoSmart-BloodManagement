"""
HemoSmart - Agent Tool Functions
------------------------------------
Plain Python functions containing the actual logic each CrewAI agent
will call. Kept separate from CrewAI's Agent/Task/Crew wrapping so the
logic itself can be tested independently of the agent framework.

    Prediction Agent    -> predict_transfusion()
    Explanation Agent   -> generate_explanation()
    Inventory Agent     -> check_inventory()
    Donor Alert Agent   -> send_donor_alert()
    Orchestrator Agent  -> calls the above in sequence, no logic of its own
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import json

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# In a real deployment, this reads from PostgreSQL (Person C's database).
# Until that's wired up, a local JSON file stands in as the inventory
# store so the Inventory Agent has something real to check against.
INVENTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inventory.json")

DEFAULT_INVENTORY = {
    "A+": 45, "A-": 12, "B+": 38, "B-": 8,
    "O+": 52, "O-": 15, "AB+": 20, "AB-": 5,
}

LOW_STOCK_THRESHOLD = 15  # units; below this, a blood type is "low"


# ----------------------------------------------------------------------
# Prediction Agent's tool
# ----------------------------------------------------------------------
def predict_transfusion(patient_record) -> dict:
    """
    patient_record: a PatientRecord instance (schema.patient_schema)
    Returns: {"transfusion_needed": bool, "confidence": float}
    """
    xgb_path = os.path.join(MODEL_DIR, "xgb_model.pkl")
    le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

    with open(xgb_path, "rb") as f:
        model = pickle.load(f)
    with open(le_path, "rb") as f:
        le = pickle.load(f)

    features = patient_record.to_model_features(le)
    prediction = bool(model.predict([features])[0])
    confidence = float(model.predict_proba([features])[0][1])

    return {"transfusion_needed": prediction, "confidence": round(confidence, 4)}


# ----------------------------------------------------------------------
# Explanation Agent's tool
# ----------------------------------------------------------------------
def generate_explanation(patient_record, prediction: bool, llm_provider: str = "ollama") -> str:
    """
    Wraps rag.explain.ExplanationGenerator -- retrieves WHO guideline
    context via FAISS, checks thresholds deterministically, and asks
    the LLM to phrase the explanation. See rag/explain.py for details.
    """
    from rag.explain import ExplanationGenerator
    generator = ExplanationGenerator(llm_provider=llm_provider)
    return generator.explain(patient_record, prediction)


# ----------------------------------------------------------------------
# Inventory Agent's tool
# ----------------------------------------------------------------------
def _load_inventory() -> dict:
    if os.path.exists(INVENTORY_FILE):
        with open(INVENTORY_FILE, "r") as f:
            return json.load(f)
    return dict(DEFAULT_INVENTORY)


def check_inventory(blood_type: str = None) -> dict:
    """
    Returns current stock levels and flags any blood types below the
    low-stock threshold. If blood_type is given, checks only that type.
    """
    inventory = _load_inventory()

    if blood_type:
        if blood_type not in inventory:
            raise ValueError(f"Unknown blood type: {blood_type}")
        units = inventory[blood_type]
        return {
            "blood_type": blood_type,
            "units": units,
            "low_stock": units < LOW_STOCK_THRESHOLD,
        }

    low_stock_types = {
        bt: units for bt, units in inventory.items() if units < LOW_STOCK_THRESHOLD
    }
    return {
        "inventory": inventory,
        "low_stock_types": low_stock_types,
        "any_shortage": len(low_stock_types) > 0,
    }


# ----------------------------------------------------------------------
# Extraction Agent's tool
# ----------------------------------------------------------------------
def extract_from_pdf(pdf_path: str, llm_provider: str = "ollama"):
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



def send_donor_alert(blood_type: str, units_needed: int = None) -> str:
    """
    Sends personalized alerts to the top eligible, most-likely-to-respond
    donors, selected via a Multi-Armed Bandit (Thompson Sampling) -- see
    agents/donor_module.py for the full explanation. This replaces a naive
    "alert everyone" approach with the intelligent donor coordination
    described in the project report's Donor Module (Section 4.3.6).
    """
    from agents.donor_module import send_intelligent_donor_alerts
    result = send_intelligent_donor_alerts(blood_type, k=3)
    return result["message"]