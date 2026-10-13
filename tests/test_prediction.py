import math

from agents.tools import predict_transfusion
from schema.patient_schema import PatientRecord


def _predict(**values):
    base = dict(hemoglobin=13.0, platelets=250_000, INR=1.0, age=50, surgery_type="General")
    return predict_transfusion(PatientRecord(**{**base, **values}))


def _needed(result):
    return {c["component"]: c["units"] for c in result["components"] if c["needed"]}


def test_healthy_patient_needs_nothing():
    r = _predict()
    assert not r["transfusion_needed"]
    assert _needed(r) == {}
    assert r["urgency"] == "Low"


def test_components_follow_the_failing_parameter():
    assert set(_needed(_predict(hemoglobin=6.5))) == {"PRBC"}
    assert set(_needed(_predict(platelets=30_000))) == {"Platelets"}
    assert set(_needed(_predict(INR=2.2))) == {"FFP"}


def test_trauma_case_gets_multiple_components_with_doses():
    r = _predict(hemoglobin=7.2, platelets=40_000, INR=1.9, surgery_type="Emergency", blood_group="O+")
    assert r["transfusion_needed"]
    assert _needed(r) == {"PRBC": 3, "Platelets": 1, "FFP": 2}
    assert r["urgency"] == "Critical"
    assert r["blood_group"] == "O+"


def test_confidence_is_probability_of_any_component():
    r = _predict(hemoglobin=6.5)
    p_any = 1 - math.prod(1 - c["probability"] for c in r["components"])
    assert abs(r["confidence"] - p_any) < 1e-3


def test_shap_contributions_add_up_to_the_model_output():
    # TreeSHAP is exact: base + contributions = the model's log-odds
    r = _predict(hemoglobin=6.5)
    for c in r["components"]:
        contribs = dict(r["shap"][c["component"]])
        log_odds = contribs.pop("_base") + sum(contribs.values())
        assert abs(1 / (1 + math.exp(-log_odds)) - c["probability"]) < 1e-3


def test_low_hemoglobin_drives_the_prbc_shap():
    shap_prbc = dict(_predict(hemoglobin=6.5)["shap"]["PRBC"])
    shap_prbc.pop("_base")
    assert max(shap_prbc, key=shap_prbc.get) == "hemoglobin"
