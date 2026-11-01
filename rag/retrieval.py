"""
HemoSmart - Retrieval building blocks for the WHO handbook
--------------------------------------------------------------
Chunkers, a dense index, BM25, rank fusion and a cross-encoder reranker,
compared in evaluation/rag_retrieval.py (D88/D89). Every chunk is a
langchain Document with the handbook page (and, for section chunks, the
section title) in its metadata, so citations keep working.
"""

import math
import re

from langchain_core.documents import Document

SOURCE_TITLE = "WHO, The Clinical Use of Blood (Handbook)"
# bge models are trained to see this before a query (not before passages)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MIN_CHUNK_CHARS = 40  # drops PDF artefacts: page numbers, stray headers


def _splitter(size: int, overlap: int):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " "])


def page_chunks(pdf_path: str, size: int = 800, overlap: int = 120) -> list:
    """The original chunking: each page split on its own, no structure."""
    import pymupdf
    with pymupdf.open(pdf_path) as doc:
        pages = [Document(page_content=p.get_text(), metadata={"source": SOURCE_TITLE, "page": i + 1})
                 for i, p in enumerate(doc)]
    return [c for c in _splitter(size, overlap).split_documents(pages) if len(c.page_content.strip()) > MIN_CHUNK_CHARS]


def _is_heading(line: dict) -> bool:
    spans = [s for s in line["spans"] if s["text"].strip()]
    if not spans:
        return False
    text = "".join(s["text"] for s in spans).strip()
    font, size = spans[0]["font"], spans[0]["size"]
    # Section headings are FranklinGothic-Heavy >= 10.5 pt; box titles are upper-case Demi
    caps = text.split("(")[0].strip()  # "PLATELET CONCENTRATES (prepared from ...)"
    return ("Heavy" in font and size >= 10.4) or ("Demi" in font and len(caps) >= 8 and caps == caps.upper()
                                                  and any(c.isalpha() for c in caps))


def _handbook_header(line: dict, text: str):
    """2001 handbook layout: "skip" for the printed page number, "chapter"
    for the chapter name running vertically in the margin, else None."""
    x0, y0, x1, y1 = line["bbox"]
    if line["spans"][0]["size"] >= 13 and text.isdigit():
        return "skip"
    if y1 - y0 > 2 * (x1 - x0):
        return "chapter"
    return None


def page_sections(page, is_heading=_is_heading, is_header=_handbook_header) -> tuple:
    """(running chapter header, [(heading or None, text), ...]) for one page,
    in reading order. The running header and page number are left out.
    is_heading / is_header describe the document's layout (defaults: the
    2001 handbook)."""
    chapter, sections, title, buf, prev_heading = None, [], None, [], False
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(s["text"] for s in line["spans"] if "Dingbats" not in s["font"]).strip()
            if not text:
                continue
            kind = is_header(line, text)
            if kind == "skip":
                continue
            if kind == "chapter":
                chapter = chapter or text
                continue
            if is_heading(line):
                if title and not buf and prev_heading:  # a heading wrapped onto a second line
                    title = f"{title} {text}"
                else:
                    sections.append((title, "\n".join(buf)))
                    title, buf = text, []
                prev_heading = True
            else:
                buf.append(text)
                prev_heading = False
    sections.append((title, "\n".join(buf)))
    return chapter, [(t, b) for t, b in sections if b.strip() or t]


def section_chunks(pdf_path: str, size: int = 800, overlap: int = None, min_section: int = 150,
                   source: str = SOURCE_TITLE, is_heading=_is_heading, is_header=_handbook_header) -> list:
    """
    Chunks that follow the handbook's structure: a new chunk starts at each
    heading, never crosses a page (so the cited page is exact), and keeps
    a list or table whole when it fits in `size`. Sections shorter than
    `min_section` are merged into the next one on the page. Each chunk is
    prefixed with its section title, also kept as metadata["section"]
    (a section continuing from the previous page keeps its title).
    """
    import pymupdf
    overlap = overlap if overlap is not None else size // 7
    splitter = _splitter(size, overlap)
    chunks, carried = [], None
    with pymupdf.open(pdf_path) as doc:
        for pno, page in enumerate(doc):
            chapter, sections = page_sections(page, is_heading, is_header)
            merged, pending = [], None
            for title, body in sections:
                title = title or carried
                if pending:
                    title, body = pending[0], f"{pending[1]}\n{title or ''}\n{body}".strip()
                    pending = None
                if len(body) < min_section:
                    pending = (title, body)
                else:
                    merged.append((title, body))
            if pending:
                merged.append(pending)
            for title, body in merged:
                carried = title or carried
                section = title or chapter or ""
                for piece in splitter.split_text(body) if len(body) > size else [body]:
                    text = f"{section}\n{piece}".strip() if section else piece
                    if len(piece.strip()) > MIN_CHUNK_CHARS:
                        chunks.append(Document(page_content=text, metadata={
                            "source": source, "page": pno + 1, "section": section, "chapter": chapter}))
    return chunks


class DenseIndex:
    """Cosine search over chunk embeddings (sentence-transformers)."""

    def __init__(self, chunks: list, model_name: str, query_prefix: str = ""):
        from sentence_transformers import SentenceTransformer
        self.model, self.prefix = SentenceTransformer(model_name), query_prefix
        self.vectors = self.model.encode([c.page_content for c in chunks], normalize_embeddings=True,
                                         batch_size=64)

    def top(self, query: str, n: int) -> list:
        q = self.model.encode([self.prefix + query], normalize_embeddings=True)[0]
        return [int(i) for i in (-(self.vectors @ q)).argsort()[:n]]


def _tokens(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """Okapi BM25 over the chunk texts (k1=1.5, b=0.75)."""

    def __init__(self, texts: list, k1: float = 1.5, b: float = 0.75):
        self.docs = [_tokens(t) for t in texts]
        self.k1, self.b = k1, b
        self.avgdl = sum(len(d) for d in self.docs) / len(self.docs)
        df = {}
        for d in self.docs:
            for w in set(d):
                df[w] = df.get(w, 0) + 1
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.tf = [{w: d.count(w) for w in set(d)} for d in self.docs]

    def scores(self, query: str) -> list:
        q = _tokens(query)
        out = []
        for tf, d in zip(self.tf, self.docs):
            norm = self.k1 * (1 - self.b + self.b * len(d) / self.avgdl)
            out.append(sum(self.idf.get(w, 0) * tf[w] * (self.k1 + 1) / (tf[w] + norm) for w in q if w in tf))
        return out

    def top(self, query: str, n: int) -> list:
        s = self.scores(query)
        return sorted(range(len(s)), key=lambda i: -s[i])[:n]


def reciprocal_rank_fusion(rankings: list, k: int = 60) -> list:
    """Merges ranked lists of ids: score = sum of 1 / (k + rank)."""
    score = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking, 1):
            score[i] = score.get(i, 0.0) + 1 / (k + rank)
    return sorted(score, key=lambda i: -score[i])


class Reranker:
    """Re-scores (query, passage) pairs with a cross-encoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: list) -> list:
        if not docs:
            return []
        scores = self.model.predict([(query, d.page_content) for d in docs])
        return [d for _, d in sorted(zip(scores, docs), key=lambda x: -x[0])]
