"""
Hospital System Adapter -- the interface any external hospital
system integration implements.

Named "mcp" for continuity with the project report's "MCP notification
service" framing (Section 3.8.5), but this is a plain adapter
interface, not the Model Context Protocol -- see mcp_server/ for the
real MCP server this project also exposes (which correctly uses MCP
for what it's for: connecting LLM clients to tools).

MCP was never designed to model system-to-system hospital
interoperability -- HL7 or FHIR would be the real-world protocol for
that. This adapter pattern is the honestly-scoped stand-in: a real,
working interface contract for "fetch a patient record from another
hospital's system" and "push an inventory update to it," backed by a
mock implementation (mock_hospital.py) rather than a live integration
-- there is no real second hospital system for a class project to
integrate with; the point is that the contract itself is real and
that a genuine implementation could be swapped in behind it without
touching any calling code.
"""

from abc import ABC, abstractmethod

from schema.patient_schema import PatientRecord


class HospitalSystemAdapter(ABC):
    hospital_id: str = "unknown"

    @abstractmethod
    def fetch_patient_record(self, external_patient_id: str) -> PatientRecord:
        """Pulls a patient record from the external hospital system,
        already converted into HemoSmart's canonical schema."""
        raise NotImplementedError

    @abstractmethod
    def push_inventory_update(self, blood_type: str, units: int) -> bool:
        """Pushes a local inventory change to the external hospital
        system -- e.g. so a shared regional blood bank ledger stays in
        sync. Returns True if the push succeeded."""
        raise NotImplementedError
