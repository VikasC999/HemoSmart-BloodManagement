"""
HemoSmart - RAG Explanation Generator
-----------------------------------------
This is the actual "RAG" step: given a prediction + patient data,
retrieve relevant WHO handbook pages from FAISS, then ask the LLM to
explain the prediction USING those passages as grounding (not its own
unguided medical knowledge). Threshold comparisons are computed in code
(schema/clinical_thresholds.py) and called HemoSmart's protocol; WHO is
credited only with what a retrieved page says (DESIGN_DECISIONS D52).

Usage:
    generator = ExplanationGenerator()            # Groq by default
    result = generator.explain_with_sources(patient_record, prediction=True)
    # -> {"explanation", "threshold_facts", "citations"}
"""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_client import get_llm_client
from schema.clinical_thresholds import describe_threshold_facts

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index")

# Two kinds of grounding, kept apart on purpose: HemoSmart's protocol
# thresholds (computed in code, schema/clinical_thresholds.py) and the
# WHO handbook passages. Several protocol numbers (INR 1.5, platelets
# 100k for cardiac surgery) are NOT in the handbook, and the first
# version of this prompt let the LLM write "WHO guidelines state INR >
# 1.5..." -- see evaluation/rag_faithfulness.py and DESIGN_DECISIONS D52.
EXPLANATION_PROMPT_TEMPLATE = """
You are assisting a clinician by explaining an AI transfusion prediction.

Patient data:
  Hemoglobin: {hemoglobin} g/dL
  Platelets: {platelets} per microliter
  INR: {inr}
  Age: {age}
  Surgery type: {surgery_type}

AI Prediction: {prediction_text}

HemoSmart protocol checks (computed in code -- treat as fact, do not
recompute or re-judge them):
{threshold_facts}

Numbered WHO handbook excerpts:
{context}

Write a 2-3 sentence clinical explanation of the prediction. Rules:
1. Call the thresholds above "HemoSmart's protocol threshold(s)". Never
   say WHO, the handbook or "guidelines" set a number unless that exact
   number appears in an excerpt you cite.
2. When an excerpt supports a point, cite it with its bracketed number,
   e.g. [1], and only say what the excerpt itself says.
3. Do not state that any value crosses a threshold unless the protocol
   checks say it does. Do not invent criteria, numbers or sources.
"""


def parameter_queries(surgery_type: str) -> list:
    """One retrieval query per parameter, in the handbook's own words
    (British spelling, x 10^9/L units). Measured against the earlier
    generic queries in evaluation/rag_faithfulness.py."""
    return [
        f"preoperative haemoglobin level threshold for {surgery_type.lower()} surgery",
        "give platelet concentrates when the platelet count falls below 50 x 10^9/L",
        "fresh frozen plasma indications coagulation factor deficiency INR before surgery",
        f"{surgery_type.lower()} surgery blood loss transfusion",
    ]


# Thresholds live in schema/clinical_thresholds.py, shared with the
# dataset labeling rule so the explanation and the model can't disagree.
check_thresholds = describe_threshold_facts


def _citation(doc, ref: int, excerpt_chars: int = 320) -> dict:
    text = " ".join(doc.page_content.split())
    return {
        "ref": ref,
        "source": doc.metadata.get("source", "guideline"),
        "page": doc.metadata.get("page"),
        "excerpt": text[:excerpt_chars] + ("..." if len(text) > excerpt_chars else ""),
    }


class GuidelineRetriever:
    """
    Semantic search over the indexed WHO handbook, returning citable
    passages (source, page, excerpt). Loaded once and shared: by the
    explanation generator below, and by the autonomous loop's
    Explanation agent to ground each automated action in the handbook.
    """

    def __init__(self, vectorstore=None):
        if vectorstore is None:
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        self.vectorstore = vectorstore

    def search(self, query: str, k: int = 2) -> list:
        return self.vectorstore.similarity_search(query, k=k)

    def cite(self, query: str, k: int = 2) -> list:
        return [_citation(d, i + 1) for i, d in enumerate(self.search(query, k))]


class ExplanationGenerator:
    def __init__(
        self,
        llm_provider: str = os.environ.get("HEMOSMART_LLM_PROVIDER", "groq"),
        retriever: GuidelineRetriever = None,
        prompt_template: str = EXPLANATION_PROMPT_TEMPLATE,
        queries=parameter_queries,
        llm=None,
    ):
        self.retriever = retriever or GuidelineRetriever()
        self.llm = llm or get_llm_client(provider=llm_provider)
        self.prompt_template = prompt_template
        self.queries = queries

    def retrieve_sources_per_parameter(self, patient_record) -> list:
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
        seen = set()
        ordered = []
        for sub_query in self.queries(patient_record.surgery_type):
            for d in self.retriever.search(sub_query, k=2):
                if d.page_content not in seen:
                    seen.add(d.page_content)
                    ordered.append(d)
        return ordered

    def explain_with_sources(self, patient_record, prediction: bool, verbose: bool = False) -> dict:
        """
        patient_record: a PatientRecord instance (from schema.patient_schema)
        prediction: the transfusion decision from the models
        verbose: if True, prints which guideline chunks were retrieved,
                 so retrieval quality can be checked rather than assumed

        Returns {"explanation", "threshold_facts", "citations"} -- each
        citation is a numbered handbook excerpt with its page, matching
        the [n] markers the LLM is asked to use.
        """
        prediction_text = "Transfusion needed" if prediction else "No transfusion needed"

        docs = self.retrieve_sources_per_parameter(patient_record)
        citations = [_citation(d, i + 1) for i, d in enumerate(docs)]
        context = "\n\n".join(
            f"[{c['ref']}] (page {c['page']}) {d.page_content}" for c, d in zip(citations, docs)
        )
        threshold_facts = check_thresholds(patient_record)

        if verbose:
            print("=== Retrieved guideline chunks ===")
            print(context)
            print("=== Computed threshold facts (ground truth) ===")
            print(threshold_facts)
            print("=" * 40)

        prompt = self.prompt_template.format(
            hemoglobin=patient_record.hemoglobin,
            platelets=patient_record.platelets,
            inr=patient_record.INR,
            age=patient_record.age,
            surgery_type=patient_record.surgery_type,
            prediction_text=prediction_text,
            threshold_facts=threshold_facts,
            context=context,
        )

        return {
            "explanation": self.llm.generate(prompt),
            "threshold_facts": threshold_facts,
            "citations": citations,
        }

    def explain(self, patient_record, prediction: bool, verbose: bool = False) -> str:
        """Text-only form, for callers that just need the explanation
        (chat, MCP, CrewAI): the explanation followed by its sources."""
        result = self.explain_with_sources(patient_record, prediction, verbose)
        sources = "\n".join(f"[{c['ref']}] {c['source']}, p.{c['page']}" for c in result["citations"])
        return f"{result['explanation']}\n\nSources:\n{sources}"


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