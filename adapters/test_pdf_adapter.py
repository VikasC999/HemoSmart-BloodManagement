"""
HemoSmart - PDF Adapter Integration Test (Person B)
---------------------------------------------------------
Tests the full PDF extraction pipeline:
    CBC report PDF -> PyMuPDF text -> LLM -> JSON -> validated PatientRecord

Requires:
    - sample_cbc_report.pdf in the same folder (a synthetic test report)
    - Ollama running locally with a model pulled (e.g. llama3.1:8b)

Run:
    python adapters/test_pdf_adapter.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.pdf_adapter import PDFReportAdapter
from rag.llm_client import get_llm_client

SAMPLE_PDF_PATH = os.path.join(os.path.dirname(__file__), "sample_cbc_report.pdf")


def main():
    if not os.path.exists(SAMPLE_PDF_PATH):
        print(f"Sample PDF not found at {SAMPLE_PDF_PATH}")
        print("Place a test CBC report PDF there, or update SAMPLE_PDF_PATH.")
        return

    print("Loading LLM client (Ollama, llama3.1:8b)...")
    llm = get_llm_client(provider="ollama", model="llama3.1:8b")

    adapter = PDFReportAdapter(llm_client=llm)

    print(f"Extracting from: {SAMPLE_PDF_PATH}\n")
    records = adapter.safe_parse(SAMPLE_PDF_PATH)

    if not records:
        print("FAILED: No records extracted. Check the error message above.")
        print("This is the expected fallback path -- route to ManualEntryAdapter.")
        return

    record = records[0]
    print("SUCCESS. Extracted and validated PatientRecord:")
    print(f"  hemoglobin:   {record.hemoglobin}")
    print(f"  platelets:    {record.platelets}")
    print(f"  INR:          {record.INR}")
    print(f"  age:          {record.age}")
    print(f"  surgery_type: {record.surgery_type}")
    print(f"  source:       {record.source_format}")

    print("\n(Compare these against the actual values printed on the source PDF.)")


if __name__ == "__main__":
    main()