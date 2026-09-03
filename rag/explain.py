"""
HemoSmart - RAG Explanation Generator (Person B)
-------------------------------------------------------
This is the actual "RAG" step: given a prediction + patient data,
retrieve relevant guideline chunks from FAISS, then ask the LLM to
explain the prediction USING those retrieved chunks as grounding
context (not from the LLM's own unguided medical knowledge).

Usage:
    generator = ExplanationGenerator(llm_provider="ollama")
    explanation = generator.explain(patient_record, prediction=True)
"""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from llm_client import get_llm_client

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "rag/faiss_index"

EXPLANATION_PROMPT_TEMPLATE = """
You are assisting a clinician by explaining an AI transfusion prediction.

Patient data:
  Hemoglobin: {hemoglobin} g/dL
  Platelets: {platelets} per microliter
  INR: {inr}
  Age: {age}
  Surgery type: {surgery_type}

AI Prediction: {prediction_text}

The following threshold checks have ALREADY been computed and verified
-- treat them as fact, do not recompute or re-judge these comparisons:
{threshold_facts}

Relevant clinical guideline text (for wording/context only):
{context}

Using ONLY the threshold facts above as the source of truth, write a
2-3 sentence clinical explanation of why the prediction is consistent
with the guidelines. Do not state that any value crosses a threshold
unless the facts above say it does. Do not invent additional criteria.
"""


def check_thresholds(patient_record) -> str:
    """
    Deterministically checks each lab value against guideline thresholds
    in plain Python -- NOT via the LLM. Small LLMs have been observed to
    make numeric comparison errors (e.g. claiming 65,000 < 50,000) even
    when the correct numbers are right in front of them. Computing this
    in code guarantees correctness; the LLM is only used to phrase the
    already-correct facts into natural language.
    """
    facts = []

    hb = patient_record.hemoglobin
    hb_threshold = 10.0 if patient_record.surgery_type == "Emergency" else 8.0
    if hb < hb_threshold:
        facts.append(
            f"- Hemoglobin {hb} g/dL is BELOW the {hb_threshold} g/dL threshold "
            f"for {patient_record.surgery_type} surgery -> crosses guideline threshold."
        )
    else:
        facts.append(
            f"- Hemoglobin {hb} g/dL is AT OR ABOVE the {hb_threshold} g/dL threshold "
            f"for {patient_record.surgery_type} surgery -> does NOT cross guideline threshold."
        )

    plt = patient_record.platelets
    if plt < 50000:
        facts.append(
            f"- Platelets {plt}/microliter is BELOW 50,000 -> crosses guideline threshold."
        )
    else:
        facts.append(
            f"- Platelets {plt}/microliter is AT OR ABOVE 50,000 -> does NOT cross guideline threshold."
        )

    inr = patient_record.INR
    if inr > 1.5:
        facts.append(
            f"- INR {inr} is ABOVE 1.5 -> crosses guideline threshold for FFP transfusion."
        )
    else:
        facts.append(
            f"- INR {inr} is AT OR BELOW 1.5 -> does NOT cross guideline threshold for FFP transfusion."
        )

    return "\n".join(facts)


class ExplanationGenerator:
    def __init__(self, llm_provider: str = "ollama", top_k: int = 5):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = FAISS.load_local(
            INDEX_PATH, self.embeddings, allow_dangerous_deserialization=True
        )
        self.llm = get_llm_client(provider=llm_provider)
        self.top_k = top_k

    def retrieve_context(self, query: str) -> str:
        docs = self.vectorstore.similarity_search(query, k=self.top_k)
        return "\n\n".join(d.page_content for d in docs)

    def retrieve_context_per_parameter(self, patient_record) -> str:
        """
        Runs one targeted retrieval query per clinical parameter, instead
        of a single combined query. This matters because small embedding
        models match on overall topic similarity, not numeric values --
        a query mentioning "INR 1.9" does not reliably retrieve a chunk
        about "INR exceeds 1.5" through similarity search alone, since the
        two phrases aren't semantically close as sentences. Retrieving
        per-parameter guarantees each lab value's relevant guideline
        section is included, regardless of embedding quality.
        """
        sub_queries = [
            f"hemoglobin transfusion threshold for {patient_record.surgery_type} surgery",
            "platelet transfusion threshold before surgery",
            "INR fresh frozen plasma transfusion threshold",
            f"{patient_record.surgery_type} surgery transfusion guidelines",
        ]

        seen = set()
        ordered_chunks = []
        for sub_query in sub_queries:
            docs = self.vectorstore.similarity_search(sub_query, k=2)
            for d in docs:
                if d.page_content not in seen:
                    seen.add(d.page_content)
                    ordered_chunks.append(d.page_content)

        return "\n\n".join(ordered_chunks)

    def explain(self, patient_record, prediction: bool, verbose: bool = False) -> str:
        """
        patient_record: a PatientRecord instance (from schema.patient_schema)
        prediction: the boolean output from the XGBoost model
        verbose: if True, prints which guideline chunks were retrieved,
                 so retrieval quality can be checked rather than assumed
        """
        prediction_text = "Transfusion needed" if prediction else "No transfusion needed"

        context = self.retrieve_context_per_parameter(patient_record)
        threshold_facts = check_thresholds(patient_record)

        if verbose:
            print("=== Retrieved guideline chunks ===")
            print(context)
            print("=== Computed threshold facts (ground truth) ===")
            print(threshold_facts)
            print("=" * 40)

        prompt = EXPLANATION_PROMPT_TEMPLATE.format(
            hemoglobin=patient_record.hemoglobin,
            platelets=patient_record.platelets,
            inr=patient_record.INR,
            age=patient_record.age,
            surgery_type=patient_record.surgery_type,
            prediction_text=prediction_text,
            threshold_facts=threshold_facts,
            context=context,
        )

        return self.llm.generate(prompt)


if __name__ == "__main__":
    # Quick manual test -- requires rag/setup_rag.py to have been run first,
    # and either Ollama running locally or GROQ_API_KEY set.
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from schema.patient_schema import PatientRecord

    test_patient = PatientRecord(
        hemoglobin=7.2, platelets=65000, INR=1.9,
        age=58, surgery_type="Emergency",
    )

    generator = ExplanationGenerator(llm_provider="ollama")
    explanation = generator.explain(test_patient, prediction=True, verbose=True)
    print("Explanation:\n", explanation)