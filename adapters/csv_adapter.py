"""
CSV Export Adapter
---------------------
Converts a hospital's bulk CSV export (e.g. from a legacy Lab
Information System) into a list of PatientRecords.

Different hospitals name their columns differently -- this adapter
takes a column_map so the SAME code works for any hospital's CSV
layout, instead of writing one adapter per hospital.
"""

import pandas as pd
from typing import List, Dict, Optional
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema.patient_schema import PatientRecord
from adapters.base import BaseAdapter

# Default mapping: hospital's column name -> our schema field name.
# Override per-hospital by passing a custom column_map.
DEFAULT_COLUMN_MAP = {
    "Hb": "hemoglobin",
    "Hemoglobin": "hemoglobin",
    "PLT": "platelets",
    "Platelets": "platelets",
    "INR": "INR",
    "Age": "age",
    "Surgery": "surgery_type",
    "SurgeryType": "surgery_type",
    "BloodGroup": "blood_group",
    "Blood Group": "blood_group",
    "blood_group": "blood_group",
}


class CSVExportAdapter(BaseAdapter):
    source_format = "csv"

    def __init__(self, column_map: Optional[Dict[str, str]] = None,
                 hospital_name: str = "unknown"):
        self.column_map = column_map or DEFAULT_COLUMN_MAP
        self.hospital_name = hospital_name

    def parse(self, raw_input: str) -> List[PatientRecord]:
        """raw_input: path to the hospital's CSV export."""
        df = pd.read_csv(raw_input)

        # Rename whatever columns this hospital used into our schema names
        rename_map = {
            col: self.column_map[col]
            for col in df.columns if col in self.column_map
        }
        df = df.rename(columns=rename_map)

        required = {"hemoglobin", "platelets", "INR", "age", "surgery_type"}
        missing_cols = required - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"CSV is missing required columns after mapping: "
                f"{missing_cols}. Update column_map for this hospital."
            )

        records = []
        for _, row in df.iterrows():
            try:
                records.append(PatientRecord(
                    hemoglobin=row["hemoglobin"],
                    platelets=row["platelets"],
                    INR=row["INR"],
                    age=row["age"],
                    surgery_type=row["surgery_type"],
                    blood_group=row.get("blood_group") if pd.notna(row.get("blood_group")) else None,
                    source_format=self.source_format,
                    source_hospital=self.hospital_name,
                ))
            except Exception as exc:
                # Skip and log a single bad row instead of failing
                # the entire hospital's batch import.
                print(f"[csv adapter] Skipped one row: {exc}")
        return records