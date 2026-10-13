"""
A fixture-backed HospitalSystemAdapter standing in for a second,
external hospital system -- proves the adapter boundary works
end-to-end without a real hospital integration to plug into.
"""

import sys
import os
from datetime import timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from integrations.mcp.base import HospitalSystemAdapter
from schema.patient_schema import PatientRecord

_FIXTURE_PATIENTS = {
    "EXT-1001": {"hemoglobin": 8.4, "platelets": 142000, "INR": 1.2, "age": 45, "surgery_type": "General", "blood_group": "B+"},
    "EXT-1002": {"hemoglobin": 6.9, "platelets": 58000, "INR": 2.1, "age": 67, "surgery_type": "Emergency", "blood_group": "O-"},
    "EXT-1003": {"hemoglobin": 12.1, "platelets": 210000, "INR": 0.9, "age": 33, "surgery_type": "Orthopedic", "blood_group": "A+"},
}


class MockHospitalAdapter(HospitalSystemAdapter):
    hospital_id = "mock-regional-hospital"

    def fetch_patient_record(self, external_patient_id: str) -> PatientRecord:
        data = _FIXTURE_PATIENTS.get(external_patient_id)
        if data is None:
            raise ValueError(f"No record found for {external_patient_id} at {self.hospital_id}")
        return PatientRecord(**data, source_format="mcp_hospital", source_hospital=self.hospital_id)

    # SIMULATED acceptance policy: there is no real partner deciding, so
    # this stands in for one -- it accepts units with enough shelf life
    # left to be issued on arrival, up to its daily intake capacity.
    MIN_REMAINING_HOURS = 12
    MAX_UNITS_PER_OFFER = 6

    def push_inventory_update(self, blood_type: str, units: int) -> bool:
        print(f"[mock_hospital] Would push inventory update to {self.hospital_id}: {blood_type} -> {units} units")
        return True

    def offer_surplus_units(self, units: list, now=None) -> list:
        accepted = []
        for unit in sorted(units, key=lambda u: u["expires_at"]):
            if len(accepted) >= self.MAX_UNITS_PER_OFFER:
                break
            if now is None or unit["expires_at"] - now >= timedelta(hours=self.MIN_REMAINING_HOURS):
                accepted.append(unit["unit_id"])
        return accepted
