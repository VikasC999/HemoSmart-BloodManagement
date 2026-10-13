"""
HemoSmart - Clinical Transfusion Thresholds
-----------------------------------------------
The ONE place the guideline thresholds live. Before this module existed,
the dataset labeling rule (data/create_dataset.py) used a platelet
threshold of 80,000/uL while the explanation layer (rag/explain.py)
used 50,000/uL -- so for a patient at 65,000/uL the model said
"transfuse" while the explanation said "does NOT cross threshold".

Everything that reasons about thresholds -- the synthetic label
generator, the deterministic facts the RAG explanation is grounded on,
and the tests -- imports from here, so they cannot drift apart again.

These are HemoSmart's LOCAL PROTOCOL, written from the placeholder
guideline text in rag/guideline_reference.py. They are not all stated in
the WHO handbook the RAG index cites (rag/who_guidelines.pdf). Where each
number stands against the handbook:
  - RBC: Hb < 8 g/dL for surgical patients, < 10 g/dL in emergency, 9 with
    cardiac surgery. Handbook p.160: "approximately 7-8 g/dl" for minor
    surgery in a healthy patient, higher with cardiorespiratory disease or
    major surgery; p.121: maintain Hb > 9 g/dl in a moderate bleed.
    Consistent, but the exact 9 and 10 are our choice.
  - Platelets: < 50,000/uL. Handbook p.77/p.162: 50 x 10^9/L WITH
    microvascular bleeding; 10-20 x 10^9/L prophylactically (p.102, p.147).
    The 100,000/uL for cardiac surgery is NOT in the handbook.
  - FFP: INR > 1.5. NOT in the handbook, which says INR < 2.0 before most
    surgery (p.162) and lists FFP for coagulation factor deficiencies
    (p.30).
So explanations must call these "HemoSmart's protocol thresholds" and
only attribute to the WHO handbook what a retrieved passage actually says
(checked by evaluation/rag_faithfulness.py).

The thresholds depend on the surgical context. The context-free version
(Hb 8/10, platelets 50k for everyone) is static_guideline_flags() below:
the "simple threshold rule" the prediction benchmark
(evaluation/prediction_benchmark.py) compares the model against.
"""

import math

HB_THRESHOLD_ELECTIVE = 8.0     # g/dL
HB_THRESHOLD_CARDIAC = 9.0      # g/dL, cardiovascular disease
HB_THRESHOLD_EMERGENCY = 10.0   # g/dL
PLATELET_THRESHOLD = 50_000     # per microliter
PLATELET_THRESHOLD_HIGH_BLEEDING_RISK = 100_000
INR_THRESHOLD = 1.5

HIGH_BLEEDING_RISK_SURGERY = {"Cardiac"}


def hemoglobin_threshold(surgery_type: str) -> float:
    if surgery_type == "Emergency":
        return HB_THRESHOLD_EMERGENCY
    if surgery_type == "Cardiac":
        return HB_THRESHOLD_CARDIAC
    return HB_THRESHOLD_ELECTIVE


def platelet_threshold(surgery_type: str) -> int:
    if surgery_type in HIGH_BLEEDING_RISK_SURGERY:
        return PLATELET_THRESHOLD_HIGH_BLEEDING_RISK
    return PLATELET_THRESHOLD


def guideline_flags(hemoglobin: float, platelets: int, inr: float, surgery_type: str) -> dict:
    """Which guideline thresholds this patient crosses, one bool per component."""
    return {
        "rbc": hemoglobin < hemoglobin_threshold(surgery_type),
        "platelets": platelets < platelet_threshold(surgery_type),
        "ffp": inr > INR_THRESHOLD,
    }


def static_guideline_flags(hemoglobin: float, platelets: int, inr: float, surgery_type: str) -> dict:
    """The context-free rule: one Hb threshold (10 only for Emergency),
    one platelet threshold for everyone. Benchmark baseline only."""
    return {
        "rbc": hemoglobin < (HB_THRESHOLD_EMERGENCY if surgery_type == "Emergency" else HB_THRESHOLD_ELECTIVE),
        "platelets": platelets < PLATELET_THRESHOLD,
        "ffp": inr > INR_THRESHOLD,
    }


# guideline_flags key -> the blood component that criterion calls for
FLAG_COMPONENT = {"rbc": "PRBC", "platelets": "Platelets", "ffp": "FFP"}


def transfusion_indicated(hemoglobin: float, platelets: int, inr: float, surgery_type: str) -> bool:
    return any(guideline_flags(hemoglobin, platelets, inr, surgery_type).values())


# ----------------------------------------------------------------------
# Dose and urgency -- protocol rules, not model output. The model says
# WHICH components a patient needs; these say how much and how fast,
# deterministically, so the explanation can state exactly why.
# Adult doses assume ~70 kg (body weight isn't collected).
# ----------------------------------------------------------------------
HB_RISE_PER_PRBC_UNIT = 1.0      # g/dL per unit in an average adult
MAX_PRBC_UNITS = 4               # reassess after this; don't order blindly
SEVERE_THROMBOCYTOPENIA = 20_000
HIGH_INR = 2.0

CRITICAL_HB = 7.0
CRITICAL_PLATELETS = 20_000
CRITICAL_INR = 2.5


def recommended_units(component: str, hemoglobin: float, platelets: int, inr: float, surgery_type: str) -> int:
    if component == "PRBC":
        # Enough to lift Hb back over the threshold, single-unit minimum
        # (patient blood management: transfuse, then reassess).
        deficit = hemoglobin_threshold(surgery_type) - hemoglobin
        return max(1, min(MAX_PRBC_UNITS, math.ceil(deficit / HB_RISE_PER_PRBC_UNIT)))
    if component == "Platelets":
        return 2 if platelets < SEVERE_THROMBOCYTOPENIA else 1  # adult therapeutic doses
    if component == "FFP":
        return 4 if inr > HIGH_INR else 2  # ~15 mL/kg vs a smaller correction, 250 mL units
    raise ValueError(f"Unknown component: {component}")


def urgency(hemoglobin: float, platelets: int, inr: float, surgery_type: str, transfusion_needed: bool) -> str:
    """Critical / Moderate / Low, per FR-02's 'based on parameter thresholds'."""
    if not transfusion_needed:
        return "Low"
    if (
        hemoglobin < CRITICAL_HB
        or platelets < CRITICAL_PLATELETS
        or inr > CRITICAL_INR
        or surgery_type == "Emergency"
    ):
        return "Critical"
    return "Moderate"


def describe_threshold_facts(patient_record) -> str:
    """
    Deterministically checks each lab value against the thresholds above
    in plain Python -- NOT via the LLM. Small LLMs have been observed to
    make numeric comparison errors (e.g. claiming 65,000 < 50,000) even
    when the correct numbers are right in front of them. Computing this
    in code guarantees correctness; the LLM is only used to phrase the
    already-correct facts into natural language.
    """
    hb = patient_record.hemoglobin
    plt = patient_record.platelets
    inr = patient_record.INR
    surgery = patient_record.surgery_type
    hb_threshold = hemoglobin_threshold(surgery)
    plt_threshold = platelet_threshold(surgery)
    flags = guideline_flags(hb, plt, inr, surgery)

    facts = []
    if flags["rbc"]:
        facts.append(
            f"- Hemoglobin {hb} g/dL is BELOW the {hb_threshold} g/dL threshold "
            f"for {surgery} surgery -> crosses the HemoSmart protocol threshold."
        )
    else:
        facts.append(
            f"- Hemoglobin {hb} g/dL is AT OR ABOVE the {hb_threshold} g/dL threshold "
            f"for {surgery} surgery -> does NOT cross the HemoSmart protocol threshold."
        )

    if flags["platelets"]:
        facts.append(
            f"- Platelets {plt}/microliter is BELOW the {plt_threshold:,} threshold "
            f"for {surgery} surgery -> crosses the HemoSmart protocol threshold."
        )
    else:
        facts.append(
            f"- Platelets {plt}/microliter is AT OR ABOVE the {plt_threshold:,} threshold "
            f"for {surgery} surgery -> does NOT cross the HemoSmart protocol threshold."
        )

    if flags["ffp"]:
        facts.append(
            f"- INR {inr} is ABOVE {INR_THRESHOLD} -> crosses the HemoSmart protocol threshold for FFP transfusion."
        )
    else:
        facts.append(
            f"- INR {inr} is AT OR BELOW {INR_THRESHOLD} -> does NOT cross the HemoSmart protocol threshold for FFP transfusion."
        )

    return "\n".join(facts)
