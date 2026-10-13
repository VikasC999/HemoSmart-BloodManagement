import pytest
from pydantic import ValidationError

from schema.patient_schema import PatientRecord

VALID = dict(hemoglobin=7.2, platelets=65_000, INR=1.9, age=58, surgery_type="Emergency")


def test_valid_record():
    assert PatientRecord(**VALID).platelets == 65_000


def test_surgery_type_is_normalized():
    assert PatientRecord(**{**VALID, "surgery_type": "  cardiac "}).surgery_type == "Cardiac"


def test_unknown_surgery_type_rejected():
    with pytest.raises(ValidationError):
        PatientRecord(**{**VALID, "surgery_type": "Neuro"})


@pytest.mark.parametrize("field, value", [
    ("hemoglobin", 1.0), ("hemoglobin", 30.0), ("INR", 0.1), ("age", 150), ("platelets", 2_000_000),
])
def test_out_of_range_rejected(field, value):
    with pytest.raises(ValidationError):
        PatientRecord(**{**VALID, field: value})


@pytest.mark.parametrize("reported, expected", [(128, 128_000), (65.5, 65_500), (128_000, 128_000)])
def test_platelets_in_thousands_are_normalized(reported, expected):
    # e.g. sample_cbc_report_2.pdf reports "Plt 128 x10^3/uL"
    assert PatientRecord(**{**VALID, "platelets": reported}).platelets == expected
