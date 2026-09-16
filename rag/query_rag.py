"""
HemoSmart - RAG Retrieval Test (Person B)
----------------------------------------------
Sanity-checks that FAISS retrieval actually returns relevant guideline
chunks BEFORE wiring in the LLM. If retrieval quality is bad here, the
LLM's explanations will be bad too, no matter how good the prompt is --
so this step is worth testing in isolation first.

Run after setup_rag.py:
    python rag/query_rag.py
"""

import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index")

# A few realistic test queries mirroring what the Prediction Agent
# would actually send -- covering the main clinical scenarios.
TEST_QUERIES = [
    "Patient has hemoglobin 7.2, platelets 65000, INR 1.9, Emergency surgery. Transfusion needed.",
    "Patient has platelets 42000 before major surgery.",
    "Pediatric patient with hemoglobin 6.5.",
    "Patient has normal hemoglobin 13.5, no bleeding, INR 1.0. No transfusion needed.",
]


def test_retrieval(k=3):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )

    for query in TEST_QUERIES:
        print("=" * 70)
        print("QUERY:", query)
        print("-" * 70)
        results = vectorstore.similarity_search(query, k=k)
        for i, doc in enumerate(results, 1):
            print(f"[{i}] {doc.page_content[:150]}...")
        print()


if __name__ == "__main__":
    test_retrieval()