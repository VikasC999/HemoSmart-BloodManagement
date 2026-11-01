"""
Retrieval building blocks (rag/retrieval.py), the production retriever's
rerank path (rag/explain.py), and the retrieval metrics
(evaluation/rag_retrieval.py). No models are downloaded here.
"""

import math
import os

import pytest
from langchain_core.documents import Document

from rag.explain import RERANK_CANDIDATES, GuidelineRetriever, citation_label
from rag.retrieval import BM25, _is_heading, reciprocal_rank_fusion, section_chunks

PDF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag", "who_guidelines.pdf")


@pytest.fixture(scope="module")
def metrics():
    saved = os.environ["DATABASE_URL"]  # the evaluation package repoints it; restore ours
    import evaluation.rag_retrieval as module
    os.environ["DATABASE_URL"] = saved
    return module


def test_rrf_rewards_agreement_between_rankings():
    assert reciprocal_rank_fusion([[1, 2, 3], [3, 1]]) == [1, 3, 2]


def test_bm25_ranks_the_matching_text_first():
    bm25 = BM25(["fresh frozen plasma dose", "platelet concentrates storage at 22 degrees", "red cell suspension"])
    assert bm25.top("how are platelet concentrates stored", 1) == [1]
    assert bm25.scores("unrelated words")[0] == 0


def _line(text, font, size):
    return {"spans": [{"text": text, "font": font, "size": size}]}


def test_heading_detection():
    assert _is_heading(_line("Platelet concentrates", "FranklinGothic-Heavy", 12.0))
    assert _is_heading(_line("PLATELET CONCENTRATES (prepared from whole blood donations)", "FranklinGothic-Demi", 9.8))
    assert not _is_heading(_line("Should be issued from the blood bank", "FranklinGothic-Book", 10.0))
    assert not _is_heading(_line("never", "FranklinGothic-Demi", 10.0))  # emphasis, not a heading


@pytest.mark.skipif(not os.path.exists(PDF), reason="handbook PDF not present")
def test_section_chunks_keep_page_and_section():
    chunks = section_chunks(PDF, 500)
    assert all(c.metadata["page"] for c in chunks)
    assert all(c.metadata["section"] for c in chunks if c.metadata["page"] > 1)  # page 1 is the cover
    p51 = {c.metadata["section"] for c in chunks if c.metadata["page"] == 51}
    assert {"Whole blood and red cells", "Platelet concentrates"} <= p51
    assert all(len(c.page_content) <= 500 + len(c.metadata["section"] or "") + 1 for c in chunks)


class _Store:
    def __init__(self, docs):
        self.docs, self.asked = docs, None

    def similarity_search(self, query, k):
        self.asked = k
        return self.docs[:k]


class _ReverseReranker:
    def rerank(self, query, docs):
        return list(reversed(docs))


def test_retriever_reranks_candidates_and_cites_the_section():
    docs = [Document(page_content=f"chunk {i}", metadata={"source": "WHO", "page": i, "section": f"S{i}"})
            for i in range(30)]
    store = _Store(docs)
    r = GuidelineRetriever(vectorstore=store, reranker=_ReverseReranker())
    top = r.search("q", k=2)
    assert store.asked == RERANK_CANDIDATES and [d.metadata["page"] for d in top] == [19, 18]
    assert citation_label(r.cite("q", k=1)[0]) == "WHO, p.19, S19"
    assert [d.metadata["page"] for d in GuidelineRetriever(vectorstore=store, reranker=False).search("q", 2)] == [0, 1]


def test_ranking_metrics(metrics):
    s = metrics.score_ranking([5, 9, 9, 3], {9, 3})
    assert (s["hit@1"], s["hit@2"], s["hit@5"], s["mrr@5"]) == (0.0, 1.0, 1.0, 0.5)
    ideal = 1 + 1 / math.log2(3)
    assert s["ndcg@5"] == pytest.approx((1 / math.log2(3) + 1 / math.log2(5)) / ideal)  # the repeat of 9 adds nothing
    assert metrics.score_ranking([1, 2], {7})["mrr@5"] == 0.0
    assert metrics.paired_bootstrap([0.5, 1.0], [0.5, 1.0]) == (0.0, 0.0, 0.0)


def test_eval_set_is_frozen_and_split(metrics):
    items = metrics.load_eval_set()
    assert len(items) == 45 and sum(i["split"] == "dev" for i in items) == 15
    assert all(i["pages"] and i["question"].endswith("?") for i in items)


def test_nli_support_counts_only_cited_sentences():
    saved = os.environ["DATABASE_URL"]
    import evaluation.rag_faithfulness as rf
    os.environ["DATABASE_URL"] = saved
    verdict = {"A": "entailment", "B": "contradiction", "C": "neutral"}
    stub = lambda pairs: [verdict[p[0]] for p in pairs]  # noqa: E731  premise letter decides
    text = "Platelets are stored warm [1]. FFP is frozen [2][3]. No citation here. Bad ref [9]."
    out = rf.nli_support(text, ["A", "B", "C"], predict=stub)
    assert (out["cited_sentences"], out["entailed"], out["contradicted"]) == (2, 1, 1)
    assert rf.nli_support("Mixed [2][1].", ["A", "B"], predict=stub)["entailed"] == 1  # any entailing passage wins


class _ScriptedLLM:
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def _generator(llm):
    from rag.explain import ExplanationGenerator
    docs = [Document(page_content=f"chunk {i}", metadata={"source": "WHO", "page": i}) for i in range(30)]
    return ExplanationGenerator(retriever=GuidelineRetriever(vectorstore=_Store(docs), reranker=False), llm=llm)


def _record():
    from schema.patient_schema import PatientRecord
    return PatientRecord(hemoglobin=7.0, platelets=200000, INR=1.0, age=60, surgery_type="General",
                         source_format="manual")


def test_prompt_states_each_verdict_so_values_are_not_merged():
    llm = _ScriptedLLM("Hemoglobin is below HemoSmart's protocol threshold [1].")
    _generator(llm).explain_with_sources(_record(), True)
    assert ("crosses HemoSmart's protocol threshold: hemoglobin; within it: platelets, INR."
            in llm.prompts[0])


def test_empty_llm_reply_is_retried_then_refused():
    llm = _ScriptedLLM("", "Hemoglobin is low [1].")
    assert _generator(llm).explain_with_sources(_record(), True)["explanation"] == "Hemoglobin is low [1]."
    with pytest.raises(ValueError):
        _generator(_ScriptedLLM("  ", "")).explain_with_sources(_record(), True)
