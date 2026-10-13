import os

import pandas as pd
import pytest

from data.label_process import decision_probabilities
from schema.clinical_thresholds import (
    PLATELET_THRESHOLD,
    describe_threshold_facts,
    guideline_flags,
    platelet_threshold,
    recommended_units,
    static_guideline_flags,
    transfusion_indicated,
)
from schema.patient_schema import PatientRecord

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _record(**overrides):
    values = dict(hemoglobin=13.0, platelets=250_000, INR=1.0, age=50, surgery_type="General")
    values.update(overrides)
    return PatientRecord(**values)


def test_dataset_labels_follow_the_shared_thresholds():
    # The model is trained on these labels and the explanation is built
    # from the same thresholds -- if they diverge, the model and its
    # explanation contradict each other (the old 80k vs 50k bug). Labels
    # are noisy clinician decisions now, so "follow" means: mostly agree,
    # and agree better with the context-aware rule than the static one.
    df = pd.read_csv(os.path.join(REPO_ROOT, "data", "patient_data.csv"))
    context = df.apply(
        lambda r: int(transfusion_indicated(r.hemoglobin, r.platelets, r.INR, r.surgery_type)), axis=1
    )
    static = df.apply(
        lambda r: int(any(static_guideline_flags(r.hemoglobin, r.platelets, r.INR, r.surgery_type).values())),
        axis=1,
    )
    agreement = (context == df.transfusion_needed).mean()
    assert 0.85 < agreement < 1.0  # noisy, but not a coin flip -- and not circular
    assert agreement > (static == df.transfusion_needed).mean()


def test_label_probability_is_centred_on_the_guideline_threshold():
    p = decision_probabilities([8.0, 6.0, 11.0], [250_000] * 3, [1.0] * 3, ["General"] * 3)["PRBC"]
    assert abs(p[0] - 0.5) < 1e-9
    assert p[1] > 0.95 and p[2] < 0.05


def test_dataset_exercises_every_criterion():
    df = pd.read_csv(os.path.join(REPO_ROOT, "data", "patient_data.csv"))
    flags = df.apply(
        lambda r: guideline_flags(r.hemoglobin, r.platelets, r.INR, r.surgery_type), axis=1
    )
    for criterion in ("rbc", "platelets", "ffp"):
        assert sum(f[criterion] for f in flags) >= 50, f"too few '{criterion}' cases for the model to learn"


@pytest.mark.parametrize("platelets, crosses", [
    (PLATELET_THRESHOLD - 1, True),
    (PLATELET_THRESHOLD, False),
    (65_000, False),  # the case the 80k/50k mismatch used to get wrong
])
def test_platelet_facts_match_flags(platelets, crosses):
    facts = describe_threshold_facts(_record(platelets=platelets))
    platelet_line = next(line for line in facts.splitlines() if "Platelets" in line)
    assert ("BELOW" in platelet_line) == crosses


def test_emergency_uses_higher_hemoglobin_threshold():
    assert guideline_flags(9.5, 250_000, 1.0, "Emergency")["rbc"]
    assert not guideline_flags(9.5, 250_000, 1.0, "Cardiac")["rbc"]


def test_cardiac_surgery_uses_context_thresholds():
    # Guideline text: Hb 9-10 with cardiovascular disease, platelets
    # 100k for high-bleeding-risk procedures
    assert guideline_flags(8.5, 250_000, 1.0, "Cardiac")["rbc"]
    assert not guideline_flags(8.5, 250_000, 1.0, "General")["rbc"]
    assert guideline_flags(13, 80_000, 1.0, "Cardiac")["platelets"]
    assert not static_guideline_flags(13, 80_000, 1.0, "Cardiac")["platelets"]
    assert platelet_threshold("Orthopedic") == PLATELET_THRESHOLD
    assert recommended_units("PRBC", 8.5, 250_000, 1.0, "Cardiac") == 1
    facts = describe_threshold_facts(_record(platelets=80_000, surgery_type="Cardiac"))
    assert "BELOW the 100,000 threshold for Cardiac" in facts


def test_explanation_reexports_shared_facts():
    from rag.explain import check_thresholds
    assert check_thresholds is describe_threshold_facts
