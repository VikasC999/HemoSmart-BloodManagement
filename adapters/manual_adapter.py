"""
Manual Entry Adapter
-----------------------
Converts data typed directly into the dashboard form into a PatientRecord.
This is the guaranteed fallback: if the PDF adapter fails (bad scan,
unusual report layout), the doctor can always type values in manually
and the system still works.
"""

from typing import List, Dict
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema.patient_schema import PatientRecord
from adapters.base import BaseAdapter


class ManualEntryAdapter(BaseAdapter):
    source_format = "manual"

    def parse(self, raw_input: Dict) -> List[PatientRecord]:
        """
        raw_input: dict coming straight from the React form, e.g.
            {
                "hemoglobin": 9.2, "platelets": 145000,
                "INR": 1.3, "age": 54, "surgery_type": "Cardiac"
            }
        """
        record = PatientRecord(
            hemoglobin=raw_input["hemoglobin"],
            platelets=raw_input["platelets"],
            INR=raw_input["INR"],
            age=raw_input["age"],
            surgery_type=raw_input["surgery_type"],
            blood_group=raw_input.get("blood_group"),
            source_format=raw_input.get("source_format") or self.source_format,
            source_hospital=raw_input.get("hospital_name", "unknown"),
        )
        return [record]