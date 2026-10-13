"""
HemoSmart - RAG Engine Setup (Person B)
-------------------------------------------
Builds the retrieval knowledge base: chunks transfusion guideline text,
converts chunks to embeddings, and stores them in a local FAISS index
for fast similarity search at prediction time.

Supports TWO sources, in priority order:
  1. A real WHO guideline PDF, if present at GUIDELINE_PDF_PATH
     (download from who.int and place it there -- see instructions below)
  2. The short placeholder text in guideline_reference.py, used only
     as a fallback so the pipeline still runs before the real PDF exists

Run once (or whenever the guideline source changes):
    python rag/setup_rag.py

Requires:
    pip install langchain langchain-community faiss-cpu sentence-transformers pymupdf

Output:
    rag/faiss_index/   -> the saved vector index, loaded later by query_rag.py

HOW TO USE THE REAL WHO GUIDELINES:
    1. Go to who.int and search "WHO Clinical Transfusion Guidelines"
       (or a national blood transfusion guideline PDF from a source you trust)
    2. Download the PDF and save it as: rag/who_guidelines.pdf
    3. Re-run this script -- it will automatically detect and use it
       instead of the placeholder text, with no other code changes needed
"""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from guideline_reference import TRANSFUSION_GUIDELINES

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RAG_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(RAG_DIR, "faiss_index")
GUIDELINE_PDF_PATH = os.path.join(RAG_DIR, "who_guidelines.pdf")

# Real documents are much longer and denser than the placeholder text,
# so chunks need to be bigger to hold a complete clinical point, with
# more overlap so a threshold number doesn't get split away from the
# sentence that gives it context.
PDF_CHUNK_SIZE = 800
PDF_CHUNK_OVERLAP = 120

PLACEHOLDER_CHUNK_SIZE = 400
PLACEHOLDER_CHUNK_OVERLAP = 50


SOURCE_TITLE = "WHO, The Clinical Use of Blood (Handbook)"


def load_guideline_documents() -> tuple[list, str]:
    """
    Returns (documents, source_label), one Document per PDF page with its
    page number in metadata -- so every retrieved chunk can be cited
    back to the exact handbook page it came from. Prefers the real PDF.
    """
    if os.path.exists(GUIDELINE_PDF_PATH):
        import pymupdf  # the modern import; `fitz` is now a deprecated alias
        print(f"Found real guideline PDF -> {GUIDELINE_PDF_PATH}")
        doc = pymupdf.open(GUIDELINE_PDF_PATH)
        documents = [
            Document(page_content=page.get_text(), metadata={"source": SOURCE_TITLE, "page": i + 1})
            for i, page in enumerate(doc)
        ]
        doc.close()
        print(f"Extracted {sum(len(d.page_content) for d in documents)} characters from {len(documents)} pages")
        return documents, "pdf"
    else:
        print(f"No PDF found at {GUIDELINE_PDF_PATH} -- using placeholder guideline text.")
        print("(See the docstring at the top of this file for how to add the real WHO PDF.)")
        return [Document(page_content=TRANSFUSION_GUIDELINES,
                         metadata={"source": "HemoSmart placeholder guideline text", "page": None})], "placeholder"


def build_index():
    documents, source = load_guideline_documents()

    chunk_size = PDF_CHUNK_SIZE if source == "pdf" else PLACEHOLDER_CHUNK_SIZE
    chunk_overlap = PDF_CHUNK_OVERLAP if source == "pdf" else PLACEHOLDER_CHUNK_OVERLAP

    print(f"Splitting guideline text into chunks (source={source}, chunk_size={chunk_size})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(documents)
    # Drop near-empty chunks (common artifact of PDF extraction: headers,
    # page numbers, stray whitespace lines)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 40]
    print(f"Created {len(chunks)} usable chunks")

    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    print("(First run downloads the model, ~90MB, takes a few minutes)")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)
    print(f"Saved FAISS index -> {INDEX_PATH}")

    if source == "placeholder":
        print(
            "\nNOTE: This index was built from PLACEHOLDER guideline text, "
            "not the real WHO document. Add rag/who_guidelines.pdf and "
            "re-run this script before using results in your final report."
        )

    return vectorstore, chunks


if __name__ == "__main__":
    build_index()
    print("\nRAG engine ready. Run rag/query_rag.py to test retrieval.")