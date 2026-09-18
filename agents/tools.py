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
import subprocess

from db.database import SessionLocal
from db.models import InventoryItem

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

DEFAULT_INVENTORY = {
    "A+": 45, "A-": 12, "B+": 38, "B-": 8,
    "O+": 52, "O-": 15, "AB+": 20, "AB-": 5,
}

LOW_STOCK_THRESHOLD = 15  # units; below this, a blood type is "low"


# ----------------------------------------------------------------------
# Prediction Agent's tool
# ----------------------------------------------------------------------
_xgb_model = None
_label_encoder = None


def _load_xgb_model():
    """
    Loads the trained XGBoost model + label encoder once and caches them
    at module level. predict_transfusion() used to pickle.load() both
    files on every single call, which is fine for a CLI script but adds
    real latency to every API request once this is wrapped by FastAPI.
    """
    global _xgb_model, _label_encoder
    if _xgb_model is None or _label_encoder is None:
        xgb_path = os.path.join(MODEL_DIR, "xgb_model.pkl")
        le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
        with open(xgb_path, "rb") as f:
            _xgb_model = pickle.load(f)
        with open(le_path, "rb") as f:
            _label_encoder = pickle.load(f)
    return _xgb_model, _label_encoder


def predict_transfusion(patient_record) -> dict:
    """
    patient_record: a PatientRecord instance (schema.patient_schema)
    Returns: {"transfusion_needed": bool, "confidence": float}
    """
    model, le = _load_xgb_model()

    features = patient_record.to_model_features(le)
    prediction = bool(model.predict([features])[0])
    confidence = float(model.predict_proba([features])[0][1])

    return {"transfusion_needed": prediction, "confidence": round(confidence, 4)}


# ----------------------------------------------------------------------
# Explanation Agent's tool
# ----------------------------------------------------------------------
def generate_explanation(
    patient_record, prediction: bool,
    llm_provider: str = os.environ.get("HEMOSMART_LLM_PROVIDER", "groq"),
) -> str:
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
    """Reads stock levels from Postgres, seeding from DEFAULT_INVENTORY
    the first time the table is empty (was previously agents/inventory.json)."""
    with SessionLocal() as session:
        rows = session.query(InventoryItem).all()
        if rows:
            return {r.blood_type: r.units for r in rows}

        for blood_type, units in DEFAULT_INVENTORY.items():
            session.add(InventoryItem(blood_type=blood_type, units=units))
        session.commit()
        return dict(DEFAULT_INVENTORY)


def _save_inventory(inventory: dict) -> None:
    with SessionLocal() as session:
        existing = {r.blood_type: r for r in session.query(InventoryItem).all()}
        for blood_type, units in inventory.items():
            row = existing.get(blood_type)
            if row is None:
                session.add(InventoryItem(blood_type=blood_type, units=units))
            else:
                row.units = units
        session.commit()


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



DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
LSTM_SEQ_LENGTH = 30  # must match the sequence length used in models/train_lstm_colab.py

# Path to the Python interpreter INSIDE the separate LSTM venv (see
# agents/lstm_forecast_standalone.py's setup instructions). Override
# with the HEMOSMART_LSTM_PYTHON environment variable if your venv
# lives somewhere else.
LSTM_VENV_PYTHON = os.environ.get(
    "HEMOSMART_LSTM_PYTHON",
    os.path.join(os.path.dirname(MODEL_DIR), "lstm_env", "Scripts", "python.exe"),
)
LSTM_STANDALONE_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "lstm_forecast_standalone.py"
)


def _forecast_with_lstm(days: int):
    """
    Runs LSTM inference in a SEPARATE, isolated Python environment via
    subprocess, rather than importing TensorFlow directly in this
    process. This exists because TensorFlow's installed build requires
    an older NumPy/protobuf combination than CrewAI's dependencies
    require -- both cannot be reliably satisfied in one environment.
    Isolating them via a subprocess boundary removes the conflict
    entirely: this process never imports TensorFlow at all.

    Returns None (triggering fallback to Prophet) if the LSTM venv
    doesn't exist yet, or if the subprocess call fails for any reason.
    See agents/lstm_forecast_standalone.py's docstring for one-time
    setup instructions (create the venv, install tensorflow there).
    """
    if not os.path.exists(LSTM_VENV_PYTHON):
        print(
            f"[FORECAST] LSTM venv not found at {LSTM_VENV_PYTHON}. "
            f"Falling back to Prophet. (See agents/lstm_forecast_standalone.py "
            f"for one-time setup instructions to enable LSTM.)"
        )
        return None

    try:
        result = subprocess.run(
            [LSTM_VENV_PYTHON, LSTM_STANDALONE_SCRIPT, "--days", str(days)],
            capture_output=True, text=True, timeout=60,
        )
        output = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
        parsed = json.loads(output)
        if "error" in parsed:
            print(f"[FORECAST] LSTM subprocess reported an error: {parsed['error']}. Falling back to Prophet.")
            return None
        return parsed
    except Exception as exc:
        print(f"[FORECAST] LSTM subprocess call failed ({exc.__class__.__name__}: {exc}). Falling back to Prophet.")
        return None


def get_blood_demand_forecast(days: int = 7) -> dict:
    """
    Forecasting Module (matches the report's Level 2 DFD, block 3C:
    "Forecasting Module (LSTM / Prophet)"). Tries the LSTM model first
    since it was empirically shown to outperform Prophet after tuning
    (MAE 3.35 vs 3.48 -- see models/train_lstm_colab.py comparison).
    Falls back to Prophet if the LSTM model/scaler files aren't
    present (e.g. LSTM was trained on Colab but files weren't copied
    down yet), and returns a clear error if neither is available.
    """
    lstm_result = _forecast_with_lstm(days)
    if lstm_result is not None:
        return lstm_result

    prophet_path = os.path.join(MODEL_DIR, "prophet_model.pkl")
    if not os.path.exists(prophet_path):
        return {
            "error": (
                "No trained forecasting model found. Either place "
                "lstm_model.h5 + lstm_scaler.pkl (from Colab) or "
                "prophet_model.pkl (from models/train_prophet.py) "
                "in the models/ folder."
            )
        }

    with open(prophet_path, "rb") as f:
        model = pickle.load(f)

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    result = forecast[["ds", "yhat"]].tail(days)

    return {
        "forecast_days": days,
        "model_used": "Prophet",
        "predicted_units_per_day": [
            {"date": str(row.ds.date()), "predicted_units": round(row.yhat, 1)}
            for row in result.itertuples()
        ],
        "average_daily_demand": round(float(result["yhat"].mean()), 1),
    }


# ----------------------------------------------------------------------
# Donor Alert Agent's tool
# ----------------------------------------------------------------------
def send_donor_alert(blood_type: str) -> str:
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