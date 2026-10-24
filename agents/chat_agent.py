"""
HemoSmart - Interactive Chat Agent (Orchestrator)
--------------------------------------------------------
This is the top layer in the architecture diagram: a user types a
free-text request, and the Orchestrator decides which specialist
agent(s) to invoke, manages any state needed between turns (e.g. "the
patient we just discussed"), and returns a conversational response.

Every role has the assistant, and it answers only from what that role
may see (FR-08, D66). Per message:

  1. Split it into clauses ("O- stock and the audit log" is two).
  2. Classify each clause: keyword rules first; the LLM only for a
     clause no rule matches, and its answer must be one label from a
     closed set.
  3. Check the clause's intent against the role's capabilities
     (schema/permissions.py) -- IN CODE, before any tool runs. A
     refused clause gets a fixed "Not authorised" line: no data is
     read and no LLM is called for it.
  4. Run the allowed tools and phrase their results with templates.
     The LLM only ever sees data that already passed step 3.

The role comes from the logged-in user (the database, via the JWT),
never from the message, so "I am admin" or "ignore your rules" changes
nothing; such messages are only flagged for the audit log.

Sending donor alerts is the one write action. It is never done in the
same turn it is asked for: the assistant states what it will send and
waits for "yes" (D66).

Run (as Admin, the CLI has no login):
    python agents/chat_agent.py
"""

import sys, os, re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema.patient_schema import PatientRecord
from schema.permissions import (
    INTENT_CAPABILITY, LABELS, chat_topics_for, denial, is_permitted, suggestions_for,
)
from agents.tools import (
    predict_transfusion,
    generate_explanation,
    check_inventory,
    send_donor_alert,
    get_supply_outlook,
    list_recent_patients,
    list_donors_masked,
    recent_donor_alerts,
    latest_autonomy_cycle,
    recent_audit_entries,
    answer_guideline_question,
)
from bloodbank.catalog import BLOOD_GROUPS
from rag.llm_client import get_llm_client

VALID_BLOOD_TYPES = set(BLOOD_GROUPS)
_COMPONENT_WORDS = {"platelet": "Platelets", "plasma": "FFP", "ffp": "FFP", "red cell": "PRBC", "prbc": "PRBC"}
_FORECAST_WORDS = ("forecast", "demand", "next week", "shortage", "outlook", "waste", "wastage", "expire",
                   "how much blood will")
_LAB_VALUES = re.compile(r"(?:h(?:ae|e)moglobin|hb)\s*[:=]?\s*\d", re.I)
_YES = re.compile(r"(yes|y|yeah|yep|confirm|send( it)?|go ahead|ok(ay)?)[.!]?", re.I)
_NO = re.compile(r"(no|n|nope|cancel|stop|don'?t)[.!]?", re.I)
# Recorded for the audit log only -- never changes a decision.
_INJECTION = re.compile(
    r"ignore (all|any|previous|prior|your|the)|you are now|pretend|system prompt|override|bypass|jailbreak"
    r"|\bi am (an? |the )?(admin|administrator|auditor|manager|blood bank manager|coordinator)\b"
    r"|\bas (an? |the )?(admin|administrator|auditor)\b",
    re.I,
)

INTENTS = ("predict", "explain", "inventory", "forecast", "action", "donors", "donor_responses",
           "patients", "autonomy", "audit", "guidelines", "help")


def _find_blood_group(message: str):
    """Longest match first, so "AB+" isn't read as "B+"."""
    upper = message.upper()
    for bt in sorted(VALID_BLOOD_TYPES, key=len, reverse=True):
        if re.search(rf"(?<![A-Z]){re.escape(bt)}", upper):
            return bt
    return None


def _find_component(message: str):
    lowered = message.lower()
    return next((c for word, c in _COMPONENT_WORDS.items() if word in lowered), None)


def keyword_intent(text: str):
    """Fast path: an intent from unambiguous phrasing, or None."""
    lowered = text.lower()
    if _LAB_VALUES.search(text):
        # Lab values typed in -> a prediction, even if a blood group is
        # mentioned too ("hb 7.2 ... O+" is a patient, not a stock query).
        return "predict"
    if re.search(r"\baudit|access log|who accessed|refused|denied|not authori[sz]ed", lowered):
        return "audit"
    if re.search(r"agent cycle|autonom|last cycle|agents? (do|did|run|ran)|agent activity|wastage rate", lowered):
        return "autonomy"
    if re.search(r"\bdonors?\b|\bdonations?\b", lowered):
        if re.search(r"\b(alert|notify|contact|call|message|request)\b", lowered):
            return "action"
        if re.search(r"respon|repl|pending|answered", lowered):
            return "donor_responses"
        return "donors"
    if re.search(r"\b(alert|notify)\b", lowered) and _find_blood_group(text):
        return "action"
    if re.search(r"pending (alerts|replies|responses)|\brepl(y|ies)\b", lowered):
        return "donor_responses"
    if any(k in lowered for k in _FORECAST_WORDS):
        return "forecast"
    if re.search(r"\bpatients?\b|recent predictions|case list|prediction history", lowered):
        return "patients"
    if re.search(r"guideline|handbook|\bthresholds?\b|recommendation", lowered) or re.search(r"\bWHO\b", text):
        return "guidelines"
    if _find_blood_group(text) or re.search(r"inventory|\bstock|units? (available|left)|do we have", lowered):
        return "inventory"
    if re.search(r"what can (i|you)|\bhelp\b|capabilit|what do you do", lowered):
        return "help"
    if re.search(r"\b(why|explain|reason)", lowered):
        return "explain"
    if any(k in lowered for k in ["hemoglobin", "platelet", "inr", "transfusion need", "predict"]):
        return "predict"
    return None


def split_clauses(message: str) -> list:
    """'O- stock and the audit log' -> two clauses, so each is checked
    on its own. A clause is only split on "and"/"also" when every piece
    has a clear, distinct intent, so "platelets and plasma for O-"
    stays one question. Lab values are never split."""
    out = []
    for segment in (s.strip(" ,.") for s in re.split(r"[?;\n]+", message)):
        if not segment:
            continue
        pieces = [p.strip(" ,.") for p in re.split(r"\s+(?:and also|and|also|plus|then)\s+", segment, flags=re.I)]
        pieces = [p for p in pieces if p]
        if len(pieces) > 1 and not _LAB_VALUES.search(segment):
            intents = [keyword_intent(p) for p in pieces]
            if all(intents) and len(set(intents)) > 1:
                out.extend(pieces)
                continue
        out.append(segment)
    return out or [message.strip()]


INTENT_CLASSIFICATION_PROMPT = """
Classify the user's request into exactly ONE of these categories:
  - predict          (whether a patient needs a transfusion, given lab values)
  - explain          (WHY the last prediction was made)
  - inventory        (current blood stock levels)
  - forecast         (future demand, shortages, or blood that may expire)
  - action           (alert / contact donors)
  - donors           (the donor list or a donor's details)
  - donor_responses  (donor replies to alerts, pending alerts)
  - patients         (recent patients and their predictions)
  - autonomy         (what the autonomous agents did, agent cycles, wastage metrics)
  - audit            (the audit log, who accessed what, refused requests)
  - guidelines       (general transfusion guideline questions, not about a specific patient)
  - help             (what the assistant can do)
  - unclear          (none of the above fit)

Respond with ONLY the single category word, nothing else.

User request: "{message}"
"""


class ChatSession:
    """
    One user's conversation. Holds the last patient discussed (so a
    follow-up "why?" needs no values repeated) and a donor alert waiting
    for confirmation. The role is fixed per session; session_store resets
    the state if the user's role changes.
    """

    def __init__(self, role: str = "Admin", llm_provider=None, llm=None):
        self.role = role
        self._llm_provider = llm_provider or os.environ.get("HEMOSMART_LLM_PROVIDER", "groq")
        self._llm = llm
        self.last_patient: PatientRecord = None
        self.last_prediction: bool = None
        self.pending_action: dict = None
        self._reset_turn()

    def _reset_turn(self):
        self._degraded = False
        self._llm_used = False
        self._action_taken = None

    # ------------------------------------------------------------
    # LLM access: lazy, and a failure degrades instead of erroring
    # ------------------------------------------------------------
    def _generate(self, prompt: str):
        try:
            if self._llm is None:
                self._llm = get_llm_client(provider=self._llm_provider)
            self._llm_used = True
            return self._llm.generate(prompt)
        except Exception:
            self._degraded = True
            return None

    def role_scope(self) -> str:
        """Prefixed to LLM prompts. Defence in depth only: access is
        decided in code before the LLM is involved."""
        return (f"You are assisting a {self.role} at a hospital blood bank. "
                f"You may only discuss: {', '.join(chat_topics_for(self.role))}.")

    # ------------------------------------------------------------
    # Intent routing
    # ------------------------------------------------------------
    def classify_intent(self, message: str) -> str:
        intent = keyword_intent(message)
        if intent:
            return intent
        response = self._generate(INTENT_CLASSIFICATION_PROMPT.format(message=message))
        if not response:
            return "unclear"
        label = response.strip().strip(".`'\"").lower()
        if label in INTENTS:
            return label
        # Longest first, so "donor_responses" isn't read as "donors".
        return next((i for i in sorted(INTENTS, key=len, reverse=True) if re.search(rf"\b{i}\b", label)),
                    "unclear")

    # ------------------------------------------------------------
    # Extraction helpers -- pulling lab values out of free text
    # ------------------------------------------------------------
    def _extract_lab_values_from_text(self, message: str) -> dict:
        """
        Simple regex extraction for values typed directly in chat
        (e.g. "hemoglobin 7.2, platelets 65000, INR 1.9, age 58,
        emergency surgery"). PDFs go through the CBC upload page, where
        the clinician confirms extracted values (FR-03, D69).
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

        group = _find_blood_group(message)
        if group:
            values["blood_group"] = group

        return values

    # ------------------------------------------------------------
    # Main entry point: one user message -> one structured reply
    # ------------------------------------------------------------
    def handle(self, message: str) -> dict:
        """
        Returns {reply, parts, role, degraded, intents, decisions,
        outcome, llm_used, injection_suspected, action_taken,
        can_ask_about}. `parts` has one {intent, allowed, text} per
        clause; the rest feeds the audit log (backend/routes/chat.py).
        """
        self._reset_turn()
        text = message.strip()
        pending, self.pending_action = self.pending_action, None

        if pending and _YES.fullmatch(text):
            parts = [self._confirm_alert(pending)]
        elif pending and _NO.fullmatch(text):
            parts = [self._part("action", "Cancelled. No donors were alerted.")]
        elif _YES.fullmatch(text) or _NO.fullmatch(text):
            parts = [self._part("help", "There's nothing waiting for confirmation.")]
        else:
            parts = [self._answer(clause) for clause in split_clauses(text)]
        return self._reply(parts, injection=bool(_INJECTION.search(text)))

    def _part(self, intent: str, text: str, allowed: bool = True) -> dict:
        return {"intent": intent, "allowed": allowed, "text": text}

    def _answer(self, clause: str) -> dict:
        intent = self.classify_intent(clause)
        capability = INTENT_CAPABILITY.get(intent)
        if capability is None or not is_permitted(self.role, capability):
            return self._part(intent, denial(self.role, capability), allowed=False)
        try:
            return self._part(intent, getattr(self, f"_do_{intent}")(clause))
        except Exception as exc:
            return self._part(intent, f"Sorry, I couldn't get {LABELS.get(capability, 'that')} right now "
                                      f"({exc.__class__.__name__}).")

    def _reply(self, parts: list, injection: bool) -> dict:
        decisions = ["allowed" if p["allowed"] else "denied" for p in parts]
        outcome = ("answered" if "denied" not in decisions
                   else "denied" if "allowed" not in decisions else "partial")
        lines = [p["text"] for p in parts]
        if outcome != "answered":
            lines.append(f"As {self.role}, you can ask about: {', '.join(chat_topics_for(self.role))}.")
        return {
            "reply": "\n\n".join(lines),
            "parts": parts,
            "role": self.role,
            "degraded": self._degraded,
            "intents": [p["intent"] for p in parts],
            "decisions": decisions,
            "outcome": outcome,
            "llm_used": self._llm_used,
            "injection_suspected": injection,
            "action_taken": self._action_taken,
            "can_ask_about": chat_topics_for(self.role),
        }

    # ------------------------------------------------------------
    # One handler per intent. Each runs only after the permission check.
    # ------------------------------------------------------------
    def _do_help(self, clause: str) -> str:
        topics = "\n".join(f"  - {t}" for t in chat_topics_for(self.role))
        examples = "\n".join(f'  "{s}"' for s in suggestions_for(self.role))
        note = "The AI model is unavailable right now, so I can only answer clear questions. " if self._degraded else ""
        return f"{note}As {self.role}, you can ask me about:\n{topics}\n\nFor example:\n{examples}"

    _do_unclear = _do_help

    def _do_predict(self, clause: str) -> str:
        values = self._extract_lab_values_from_text(clause)
        required = {"hemoglobin", "platelets", "INR", "age", "surgery_type"}
        missing = required - values.keys()
        if missing:
            return (
                f"I need more information to predict: missing {', '.join(sorted(missing))}. "
                f"Please provide hemoglobin, platelets, INR, age, and surgery type."
            )
        record = PatientRecord(**values)
        result = predict_transfusion(record)
        self.last_patient = record
        self.last_prediction = result["transfusion_needed"]
        if not result["transfusion_needed"]:
            return (
                f"Prediction: this patient does not need a transfusion "
                f"(P(transfusion) {result['confidence']:.1%}). Ask me 'why?' for the clinical reasoning."
            )
        plan = ", ".join(f"{c['component']} x{c['units']}" for c in result["components"] if c["needed"])
        return (
            f"Prediction: transfusion needed (P {result['confidence']:.1%}), urgency {result['urgency']}. "
            f"Recommended: {plan}. Ask me 'why?' for the guideline reasoning."
        )

    def _do_explain(self, clause: str) -> str:
        if self.last_patient is None:
            return "I don't have a recent prediction to explain. Please give me patient values first."
        self._llm_used = True
        try:
            return generate_explanation(self.last_patient, self.last_prediction)
        except Exception:
            from schema.clinical_thresholds import describe_threshold_facts
            self._degraded = True
            return ("The AI explanation is unavailable right now. Rule-based protocol checks:\n"
                    + describe_threshold_facts(self.last_patient))

    def _do_inventory(self, clause: str) -> str:
        bt_match = _find_blood_group(clause)
        if bt_match:
            result = check_inventory(bt_match, _find_component(clause))
            return f"{bt_match} stock:\n" + "\n".join(
                f"  {c}: {cell['available']} units ({cell['status']}, {cell['near_expiry']} near expiry)"
                for c, cell in result["components"].items()
            )

        result = check_inventory()
        if not result["low_stock"]:
            return f"All {result['total_available']} units are at or above par for every group and component."
        lines = "\n".join(
            f"  {i['blood_group']} {i['component']}: {i['available']} units ({i['status']}, par {i['par']})"
            for i in result["low_stock"]
        )
        return (
            f"{result['total_available']} units available, {result['near_expiry_total']} near expiry. "
            f"Below par:\n{lines}"
        )

    def _do_forecast(self, clause: str) -> str:
        days_match = re.search(r"(\d+)\s*-?\s*day", clause, re.I)
        days = int(days_match.group(1)) if days_match else 7  # report specifies 7-day forecast by default
        days = min(max(days, 1), 30)
        result = get_supply_outlook(days)
        totals = {
            c: round(sum(d["units"] for d in daily), 1) for c, daily in result["component_forecast"].items()
        }
        lines = [f"{days}-day demand forecast: " + ", ".join(f"{c} {n}" for c, n in totals.items()) + " units."]
        for s in result["shortages"]:
            lines.append(f"  SHORTAGE {s['blood_group']} {s['component']}: {s['projected_shortage']} unit(s) "
                         f"from {s['first_shortage_date']}")
        for w in result["wastage_risks"]:
            lines.append(f"  WASTAGE RISK {w['blood_group']} {w['component']}: "
                         f"{w['projected_wastage']} unit(s) may expire unused")
        if len(lines) == 1:
            lines.append("No shortages or wastage risks projected.")
        return "\n".join(lines)

    def _do_action(self, clause: str) -> str:
        """Asks first; _confirm_alert sends on the next turn's "yes"."""
        bt_match = _find_blood_group(clause)
        if not bt_match:
            return "Which blood group should I alert donors for? For example: alert donors for O- platelets."
        component = _find_component(clause) or "PRBC"
        self.pending_action = {"type": "donor_alert", "blood_group": bt_match, "component": component}
        return (f"I'll alert the 3 eligible {bt_match} donors most likely to respond, for {component}. "
                f'Reply "yes" to send, or "no" to cancel.')

    def _confirm_alert(self, pending: dict) -> dict:
        # Re-checked at send time: the stored role is the session's, and
        # session_store discards state if the user's role has changed.
        if not is_permitted(self.role, "donor_alerts:send"):
            return self._part("action", denial(self.role, "donor_alerts:send"), allowed=False)
        message = send_donor_alert(pending["blood_group"], pending["component"])
        self._action_taken = {"donor_alert": {"blood_group": pending["blood_group"],
                                              "component": pending["component"]}}
        return self._part("action", message)

    def _do_donors(self, clause: str) -> str:
        group = _find_blood_group(clause)
        donors = list_donors_masked(group)
        if not donors:
            return f"No donors{' for ' + group if group else ''} in the pool."
        top = sorted(donors, key=lambda d: d["expected_response_rate"], reverse=True)[:5]
        lines = "\n".join(
            f"  {d['donor_id']} ({d['blood_type']}), phone {d['phone']}, last donated {d['last_donation_date']}, "
            f"reply rate {d['expected_response_rate']:.0%}"
            for d in top
        )
        return f"{len(donors)} donor(s){' for ' + group if group else ''}. Most likely to reply:\n{lines}"

    def _do_donor_responses(self, clause: str) -> str:
        alerts = recent_donor_alerts(limit=10, pending_only=True)
        if not alerts:
            return "No donor alerts are waiting for a reply."
        lines = "\n".join(
            f"  Alert #{a['id']}: donor {a['donor_id']} ({a['blood_type']} {a['component'] or ''}), "
            f"sent {a['sent_at'][:16].replace('T', ' ')}"
            for a in alerts
        )
        return f"{len(alerts)} alert(s) waiting for a reply (record replies on the Donor responses page):\n{lines}"

    def _do_patients(self, clause: str) -> str:
        patients = list_recent_patients(limit=5)
        if not patients:
            return "No patients recorded yet."
        lines = []
        for p in patients:
            pr = p["latest_prediction"]
            verdict = ("no prediction" if pr is None
                       else f"transfusion needed ({pr['urgency']})" if pr["transfusion_needed"]
                       else "no transfusion needed")
            lines.append(f"  Patient #{p['id']} ({p['blood_group'] or 'group unknown'}, {p['surgery_type']}): "
                         f"Hb {p['hemoglobin']}, platelets {p['platelets']}, INR {p['inr']} -> {verdict}")
        return "Most recent patients:\n" + "\n".join(lines)

    def _do_autonomy(self, clause: str) -> str:
        cycle = latest_autonomy_cycle()
        if cycle is None:
            return "The autonomous agents haven't run a cycle yet."
        summary = cycle.get("summary") or {}
        facts = ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in summary.items()
                          if isinstance(v, (int, float, str)))
        return (f"Last agent cycle #{cycle['id']} ({cycle['trigger']}, {cycle['status']}) "
                f"finished {(cycle['finished_at'] or cycle['started_at'] or '')[:16].replace('T', ' ')}."
                + (f"\n  {facts}" if facts else ""))

    def _do_audit(self, clause: str) -> str:
        if re.search(r"refused|denied|not authori[sz]ed", clause, re.I):
            rows = [r for r in recent_audit_entries(limit=200, action="chat_query")
                    if (r["details"] or {}).get("outcome") in ("denied", "partial")][:10]
            if not rows:
                return "No refused chat requests on record."
            return "Recent refused chat requests:\n" + "\n".join(
                f"  {r['timestamp'][:16].replace('T', ' ')}: user #{r['user_id']} "
                f"({r['details'].get('role')}) asked about {', '.join(_denied_intents(r['details']))}"
                for r in rows
            )
        rows = recent_audit_entries(limit=10)
        if not rows:
            return "The audit log is empty."
        return "Latest audit entries:\n" + "\n".join(
            f"  {r['timestamp'][:16].replace('T', ' ')}: user #{r['user_id']} {r['action']}" for r in rows
        )

    def _do_guidelines(self, clause: str) -> str:
        self._llm_used = True
        result = answer_guideline_question(clause, role_scope=self.role_scope())
        if result["degraded"]:
            self._degraded = True
            body = "\n".join(f"[{c['ref']}] p.{c['page']}: {c['excerpt']}" for c in result["citations"])
            return f"{result['answer']}\n{body}"
        sources = "\n".join(f"[{c['ref']}] {c['source']}, p.{c['page']}" for c in result["citations"])
        return f"{result['answer']}\n\nSources:\n{sources}"


def _denied_intents(details: dict) -> list:
    pairs = zip(details.get("intents", []), details.get("decisions", []))
    return [i for i, d in pairs if d == "denied"] or ["(unknown)"]


def run_chat():
    print("HemoSmart Assistant, as Admin (type 'quit' to exit)")
    print("-" * 50)
    session = ChatSession(role="Admin")

    while True:
        message = input("\nYou: ").strip()
        if message.lower() in ("quit", "exit"):
            break
        print(f"HemoSmart: {session.handle(message)['reply']}")


if __name__ == "__main__":
    run_chat()
