"""
HemoSmart - Interactive Chat Agent (Orchestrator)
--------------------------------------------------------
This is the top layer in the architecture diagram: a user types a
free-text request, and the Orchestrator decides which specialist
agent(s) to invoke, manages any state needed between turns (e.g. "the
patient we just discussed"), and returns a conversational response.

Routing uses a hybrid approach, not a single LLM call for everything:
  1. Fast keyword/pattern matching for common, unambiguous phrasings
     (e.g. a message containing "inventory" or a blood type clearly
     means the Inventory Agent -- no need to spend an LLM call
     deciding something a keyword already answers correctly).
  2. LLM-based intent classification as the fallback for anything the
     keyword rules don't confidently match.
This mirrors the earlier lesson from rag/explain.py: let code handle
what code can decide reliably, and only use the LLM for genuine
language understanding.

Run:
    python agents/chat_agent.py
"""

import sys, os, re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema.patient_schema import PatientRecord
from agents.tools import (
    predict_transfusion,
    generate_explanation,
    check_inventory,
    send_donor_alert,
    extract_from_pdf,
    get_blood_demand_forecast,
    DEFAULT_INVENTORY,
)
from rag.llm_client import get_llm_client

VALID_BLOOD_TYPES = set(DEFAULT_INVENTORY.keys())

INTENT_CLASSIFICATION_PROMPT = """
Classify the user's request into exactly ONE of these categories:
  - predict       (asking whether a patient needs a transfusion, given lab values)
  - extract       (mentions uploading, parsing, or reading a PDF/lab report file)
  - explain       (asking WHY a prediction was made, or for guideline reasoning)
  - inventory     (asking about blood stock levels)
  - forecast      (asking about future/predicted blood demand, e.g. "next week", "7-day forecast")
  - action        (asking to alert donors or take action on a shortage)
  - unclear       (none of the above fit, or missing required information)

Respond with ONLY the single category word, nothing else.

User request: "{message}"
"""


class ChatSession:
    """
    Holds conversation state across turns -- specifically, the last
    patient discussed and their prediction, so a follow-up like "why?"
    doesn't require the user to repeat all the lab values again.
    """

    def __init__(self, llm_provider="ollama"):
        self.llm = get_llm_client(provider=llm_provider)
        self.last_patient: PatientRecord = None
        self.last_prediction: bool = None

    # ------------------------------------------------------------
    # Intent routing
    # ------------------------------------------------------------
    def classify_intent(self, message: str) -> str:
        lowered = message.lower()

        # Fast path: keyword rules for unambiguous cases
        if re.search(r"\.pdf|upload|scan|lab report", lowered):
            return "extract"
        if any(k in lowered for k in ["forecast", "demand", "next week", "predicted demand", "how much blood will"]):
            return "forecast"
        if any(bt.lower() in lowered for bt in VALID_BLOOD_TYPES) or "inventory" in lowered or "stock" in lowered:
            if "alert" in lowered or "notify" in lowered or "donor" in lowered:
                return "action"
            return "inventory"
        if "why" in lowered or "explain" in lowered or "reason" in lowered:
            return "explain"
        if any(k in lowered for k in ["hemoglobin", "platelet", "inr", "transfusion need", "predict"]):
            return "predict"

        # Fallback: ask the LLM
        prompt = INTENT_CLASSIFICATION_PROMPT.format(message=message)
        response = self.llm.generate(prompt).strip().lower()
        for valid in ("predict", "extract", "explain", "inventory", "forecast", "action"):
            if valid in response:
                return valid
        return "unclear"

    # ------------------------------------------------------------
    # Extraction helpers -- pulling lab values out of free text
    # ------------------------------------------------------------
    def _extract_lab_values_from_text(self, message: str) -> dict:
        """
        Simple regex extraction for values typed directly in chat
        (e.g. "hemoglobin 7.2, platelets 65000, INR 1.9, age 58,
        emergency surgery"). This is NOT the PDF extraction pipeline --
        just enough to let a user type values conversationally.
        """
        values = {}
        hb = re.search(r"(?:h(?:ae|e)moglobin|hb)\s*[:=]?\s*(\d+\.?\d*)", message, re.I)
        plt = re.search(r"(?:platelets?|plt)\s*[:=]?\s*(\d+)", message, re.I)
        inr = re.search(r"inr\s*[:=]?\s*(\d+\.?\d*)", message, re.I)
        age = re.search(r"age\s*[:=]?\s*(\d+)", message, re.I)

        if hb: values["hemoglobin"] = float(hb.group(1))
        if plt: values["platelets"] = int(plt.group(1))
        if inr: values["INR"] = float(inr.group(1))
        if age: values["age"] = int(age.group(1))

        for st in ["Cardiac", "Orthopedic", "General", "Emergency"]:
            if st.lower() in message.lower():
                values["surgery_type"] = st
                break

        return values

    # ------------------------------------------------------------
    # Main entry point: one user message -> one response
    # ------------------------------------------------------------
    def handle(self, message: str) -> str:
        intent = self.classify_intent(message)

        if intent == "predict":
            values = self._extract_lab_values_from_text(message)
            required = {"hemoglobin", "platelets", "INR", "age", "surgery_type"}
            missing = required - values.keys()
            if missing:
                return (
                    f"I need more information to predict: missing {', '.join(missing)}. "
                    f"Please provide hemoglobin, platelets, INR, age, and surgery type."
                )
            record = PatientRecord(**values)
            result = predict_transfusion(record)
            self.last_patient = record
            self.last_prediction = result["transfusion_needed"]
            verdict = "needs" if result["transfusion_needed"] else "does not need"
            return (
                f"Prediction: this patient {verdict} a transfusion "
                f"(confidence: {result['confidence']:.1%}). Ask me 'why?' for the clinical reasoning."
            )

        elif intent == "explain":
            if self.last_patient is None:
                return "I don't have a recent prediction to explain. Please give me patient values first."
            explanation = generate_explanation(self.last_patient, self.last_prediction)
            return explanation

        elif intent == "extract":
            pdf_match = re.search(r"[\w./\\-]+\.pdf", message, re.I)
            if not pdf_match:
                return "Please provide the path to the PDF report you'd like me to read."
            pdf_path = pdf_match.group(0)
            record = extract_from_pdf(pdf_path)
            if record is None:
                return (
                    "I couldn't extract data from that PDF reliably. "
                    "Please use manual entry instead."
                )
            self.last_patient = record
            return (
                f"Extracted from report: hemoglobin={record.hemoglobin}, "
                f"platelets={record.platelets}, INR={record.INR}, age={record.age}, "
                f"surgery_type={record.surgery_type}. Say 'predict' to get a transfusion prediction."
            )

        elif intent == "inventory":
            bt_match = next((bt for bt in VALID_BLOOD_TYPES if bt.lower() in message.lower()), None)
            result = check_inventory(bt_match)
            return str(result)

        elif intent == "forecast":
            days_match = re.search(r"(\d+)\s*-?\s*day", message, re.I)
            days = int(days_match.group(1)) if days_match else 7  # report specifies 7-day forecast by default
            result = get_blood_demand_forecast(days)
            if "error" in result:
                return result["error"]
            lines = "\n".join(
                f"  {d['date']}: {d['predicted_units']} units"
                for d in result["predicted_units_per_day"]
            )
            return (
                f"{result['forecast_days']}-day blood demand forecast "
                f"(model: {result['model_used']}, average {result['average_daily_demand']} units/day):\n{lines}"
            )

        elif intent == "action":
            bt_match = next((bt for bt in VALID_BLOOD_TYPES if bt.lower() in message.lower()), None)
            if not bt_match:
                return "Which blood type should I send a donor alert for?"
            return send_donor_alert(bt_match)

        else:
            return (
                "I can help with: predicting transfusion need, explaining a prediction, "
                "extracting data from a lab report PDF, checking blood inventory, "
                "forecasting blood demand, or alerting donors for a shortage. "
                "Could you rephrase your request?"
            )


def run_chat():
    print("HemoSmart Assistant (type 'quit' to exit)")
    print("-" * 50)
    session = ChatSession(llm_provider="ollama")

    while True:
        message = input("\nYou: ").strip()
        if message.lower() in ("quit", "exit"):
            break
        response = session.handle(message)
        print(f"HemoSmart: {response}")


if __name__ == "__main__":
    run_chat()
