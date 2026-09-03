"""
PDF Report Adapter
---------------------
Converts an uploaded CBC lab report PDF into a PatientRecord.

Pipeline:
    PDF bytes -> raw text (PyMuPDF) -> structured JSON (local LLM via
    Ollama, or Groq) -> validated PatientRecord

If text extraction fails (e.g. the PDF is a scanned image with no
selectable text), this raises so the caller can fall back to
ManualEntryAdapter -- we never silently guess patient values.

This is Person B's module -- kept here so the adapter pattern is visible
end-to-end, but the LLM prompt/integration is Person B's to build out.
"""

import json
from typing import List
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema.patient_schema import PatientRecord
from adapters.base import BaseAdapter

EXTRACTION_PROMPT = """
Extract the following values from this CBC (Complete Blood Count) lab
report and return ONLY a valid JSON object with these exact keys:
hemoglobin (float, g/dL), platelets (int, per microliter),
INR (float), age (int), surgery_type (one of: Cardiac, Orthopedic,
General, Emergency).

If a value is not present in the report, use null for that field.
Return ONLY the JSON object. No explanation, no markdown formatting.

Report text:
{report_text}
"""


class PDFReportAdapter(BaseAdapter):
    source_format = "pdf"

    def __init__(self, llm_client=None):
        """
        llm_client: any object with a .generate(prompt) -> str method.
        Pass in an Ollama or Groq wrapper here -- kept generic so
        swapping the LLM provider doesn't touch this file.
        """
        self.llm_client = llm_client

    def extract_text(self, pdf_path: str) -> str:
        import pymupdf
        doc = pymupdf.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        if not text.strip():
            raise ValueError(
                "No extractable text found in PDF -- likely a scanned "
                "image. Fall back to ManualEntryAdapter or add OCR."
            )
        return text

    @staticmethod
    def _extract_json(raw_response: str) -> dict:
        """
        LLMs frequently ignore "return ONLY JSON" instructions and wrap
        the output in markdown code fences (```json ... ```) or add a
        stray sentence before/after. Strip that instead of trusting the
        instruction was followed literally -- same lesson as the
        threshold-checking fix in rag/explain.py: don't trust an LLM to
        do something exactly when code can guarantee it instead.
        """
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()

        # If there's still leading/trailing text, extract the {...} block
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise json.JSONDecodeError("No JSON object found", cleaned, 0)
        cleaned = cleaned[start:end + 1]

        return json.loads(cleaned)

    def parse(self, raw_input: str) -> List[PatientRecord]:
        """raw_input: path to the uploaded PDF file."""
        if self.llm_client is None:
            raise RuntimeError(
                "PDFReportAdapter requires an llm_client (Ollama/Groq) "
                "to convert extracted text into structured fields."
            )

        report_text = self.extract_text(raw_input)
        prompt = EXTRACTION_PROMPT.format(report_text=report_text)
        response_text = self.llm_client.generate(prompt)

        try:
            fields = self._extract_json(response_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM did not return valid JSON: {exc}. "
                f"Raw response: {response_text[:200]}"
            )

        missing = [k for k in
                   ("hemoglobin", "platelets", "INR", "age", "surgery_type")
                   if fields.get(k) is None]
        if missing:
            raise ValueError(
                f"Extraction incomplete, missing fields: {missing}. "
                f"Route this patient to manual entry instead."
            )

        record = PatientRecord(
            hemoglobin=fields["hemoglobin"],
            platelets=fields["platelets"],
            INR=fields["INR"],
            age=fields["age"],
            surgery_type=fields["surgery_type"],
            source_format=self.source_format,
        )
        return [record]
