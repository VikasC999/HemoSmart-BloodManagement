"""
HemoSmart - RAG Retrieval Experiment
----------------------------------------
Which retrieval pipeline finds the handbook page that answers a question?
Scored on a frozen, hand-labelled set (evaluation/rag_eval_set.json,
D88): 45 questions, each with the WHO handbook page(s) that answer it,
split 15 dev / 30 test before any variant was run.

Variants (alone and combined):
  chunking   "page"     current: RecursiveCharacterTextSplitter per page
                        (800 chars / 120 overlap)
             "section"  split at the handbook's headings, within a page,
                        section title kept in metadata and prefixed to
                        the text; chunk size tuned on dev
  embedder   MiniLM     sentence-transformers/all-MiniLM-L6-v2 (current)
             bge-small  BAAI/bge-small-en-v1.5, with its query prefix
  retrieval  dense      cosine top-k
             hybrid     BM25 + dense, reciprocal rank fusion
             +rerank    top-20 candidates rescored by a cross-encoder
                        (cross-encoder/ms-marco-MiniLM-L-6-v2)

Rule: every config is scored on DEV; the highest dev MRR@5 wins. Ties
(within 0.01) go to the config needing no new embedding model, then to
the lower latency. (The first run broke ties by latency alone; latency
differences of ~10 ms flipped between runs, so the tie-break now uses
a deterministic cost first -- D89 records both runs.) The winner and the
baseline are then scored on TEST, with a paired bootstrap 95% CI.

Metrics, on the top 5 retrieved chunks: hit@1/2/5 (a relevant page in
the top n), MRR@5, nDCG@5 (binary page relevance, each page counted
once), and CPU retrieval latency per query (median of 3 runs; mean and p95 over
questions), models warm.

Run (from the repo root):
    python -m evaluation.rag_retrieval

Writes evaluation/results/rag_retrieval.md and .json.
"""

import json
import math
import os
import time
from datetime import datetime

import numpy as np

from rag.retrieval import (BGE_QUERY_PREFIX, DenseIndex, Reranker, page_chunks, reciprocal_rank_fusion,
                           section_chunks, BM25)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "evaluation", "results")
EVAL_SET = os.path.join(REPO_ROOT, "evaluation", "rag_eval_set.json")
PDF = os.path.join(REPO_ROOT, "rag", "who_guidelines.pdf")
K = 5
CANDIDATES = 20
EMBEDDERS = {"minilm": ("sentence-transformers/all-MiniLM-L6-v2", ""),
             "bge-small": ("BAAI/bge-small-en-v1.5", BGE_QUERY_PREFIX)}
CHUNKINGS = {"page-800": ("page", 800), "section-500": ("section", 500), "section-800": ("section", 800),
             "section-1200": ("section", 1200)}
MODES = ("dense", "hybrid", "dense+rerank", "hybrid+rerank")
NEW_MODEL = {"minilm": 0, "bge-small": 1}  # tie-break: 1 = a model production doesn't load yet
BOOTSTRAP = 10_000
BASELINE = "page-800 · minilm · dense"  # the original production pipeline


def load_eval_set(path: str = EVAL_SET) -> list:
    with open(path) as f:
        return json.load(f)["items"]


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def score_ranking(pages: list, relevant: set, k: int = K) -> dict:
    """pages: the page of each retrieved chunk, best first."""
    top = pages[:k]
    first = next((i + 1 for i, p in enumerate(top) if p in relevant), None)
    seen, dcg = set(), 0.0
    for i, p in enumerate(top):
        if p in relevant and p not in seen:
            seen.add(p)
            dcg += 1 / math.log2(i + 2)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return {
        "hit@1": float(first == 1), "hit@2": float(first is not None and first <= 2),
        "hit@5": float(first is not None), "mrr@5": 1 / first if first else 0.0,
        "ndcg@5": dcg / ideal if ideal else 0.0,
    }


def summarize(rows: list) -> dict:
    return {m: round(float(np.mean([r[m] for r in rows])), 3) for m in rows[0]}


def paired_bootstrap(a: list, b: list, n: int = BOOTSTRAP, seed: int = 0) -> tuple:
    """95% CI of mean(b - a) over questions, resampled with replacement."""
    diff = np.array(b) - np.array(a)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (n, len(diff)))].mean(axis=1)
    return round(float(diff.mean()), 3), round(float(np.percentile(means, 2.5)), 3), \
        round(float(np.percentile(means, 97.5)), 3)


# ----------------------------------------------------------------------
# Pipelines
# ----------------------------------------------------------------------
class Pipeline:
    """One config: chunks -> dense (+ BM25) -> optional cross-encoder."""

    def __init__(self, chunks: list, dense: DenseIndex, mode: str, reranker: Reranker = None):
        self.chunks, self.dense, self.mode, self.reranker = chunks, dense, mode, reranker
        self.bm25 = BM25([c.page_content for c in chunks]) if mode.startswith("hybrid") else None

    def search(self, query: str, k: int = K) -> list:
        n = CANDIDATES if self.mode.endswith("rerank") else k
        ranked = self.dense.top(query, max(n, CANDIDATES) if self.bm25 else n)
        if self.bm25:
            ranked = reciprocal_rank_fusion([ranked, self.bm25.top(query, CANDIDATES)])[:n]
        docs = [self.chunks[i] for i in ranked]
        if self.reranker:
            docs = self.reranker.rerank(query, docs)
        return docs[:k]


class Production:
    """The deployed retriever (rag/explain.GuidelineRetriever) as-is."""

    def __init__(self):
        from rag.explain import GuidelineRetriever
        self.retriever = GuidelineRetriever()

    def search(self, query: str, k: int = K) -> list:
        return self.retriever.search(query, k=k)


def evaluate(pipeline, items: list) -> dict:
    pipeline.search("warm-up query")
    rows, times = [], []
    for it in items:
        runs = []
        for _ in range(3):  # median of 3: one-off stalls (GC, other processes) shouldn't set p95
            t = time.perf_counter()
            docs = pipeline.search(it["question"], K)
            runs.append((time.perf_counter() - t) * 1000)
        times.append(float(np.median(runs)))
        rows.append(score_ranking([d.metadata.get("page") for d in docs], set(it["pages"])))
    return {**summarize(rows), "latency_ms_mean": round(float(np.mean(times)), 1),
            "latency_ms_p95": round(float(np.percentile(times, 95)), 1), "per_question": rows}


def run() -> dict:
    items = load_eval_set()
    dev = [i for i in items if i["split"] == "dev"]
    test = [i for i in items if i["split"] == "test"]
    reranker = Reranker()
    results, pipelines = {}, {}
    for cname, (kind, size) in CHUNKINGS.items():
        chunks = page_chunks(PDF, size) if kind == "page" else section_chunks(PDF, size)
        for ename, (model, prefix) in EMBEDDERS.items():
            dense = DenseIndex(chunks, model, prefix)
            for mode in MODES:
                name = f"{cname} · {ename} · {mode}"
                pipelines[name] = Pipeline(chunks, dense, mode, reranker if "rerank" in mode else None)
                results[name] = {"dev": evaluate(pipelines[name], dev), "chunks": len(chunks)}
                print(f"{name:<45} dev MRR@5 {results[name]['dev']['mrr@5']:.3f} "
                      f"hit@2 {results[name]['dev']['hit@2']:.2f}  {results[name]['dev']['latency_ms_mean']:.0f} ms",
                      flush=True)

    best_mrr = max(r["dev"]["mrr@5"] for r in results.values())
    near = [n for n in results if results[n]["dev"]["mrr@5"] >= best_mrr - 0.01]
    winner = min(near, key=lambda n: (NEW_MODEL[n.split(" · ")[1]], results[n]["dev"]["latency_ms_mean"]))

    results[BASELINE]["test"] = evaluate(pipelines[BASELINE], test)
    results[winner]["test"] = evaluate(pipelines[winner], test)
    # The deployed retriever must reproduce the winner (same chunks, model and reranker)
    production = Production()
    results["deployed (rag/explain.GuidelineRetriever)"] = {"dev": evaluate(production, dev),
                                                          "test": evaluate(production, test)}
    base_t, win_t = results[BASELINE]["test"], results[winner]["test"]
    ci = {m: paired_bootstrap([r[m] for r in base_t["per_question"]], [r[m] for r in win_t["per_question"]])
          for m in ("mrr@5", "hit@2", "ndcg@5")}
    return {"results": results, "winner": winner, "test_ci": ci, "n_dev": len(dev), "n_test": len(test)}


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
METRICS = ("hit@1", "hit@2", "hit@5", "mrr@5", "ndcg@5")


def _row(name: str, cell: dict) -> str:
    return (f"| {name} | " + " | ".join(f"{cell[m]:.3f}" for m in METRICS)
            + f" | {cell['latency_ms_mean']:.0f} / {cell['latency_ms_p95']:.0f} |")


def to_markdown(out: dict, meta: dict) -> str:
    r, w = out["results"], out["winner"]
    head = ["| Config | " + " | ".join(METRICS) + " | Latency ms (mean / p95) |",
            "|---|" + "---|" * (len(METRICS) + 1)]
    lines = [
        "# HemoSmart — RAG Retrieval Experiment",
        "",
        f"Frozen eval set `evaluation/rag_eval_set.json`: {out['n_dev']} dev / {out['n_test']} test questions with "
        f"hand-labelled handbook pages. Top {K} chunks scored; CPU latency per query with models loaded. "
        f"Generated by `python -m evaluation.rag_retrieval` on {meta['generated_at']}.",
        "",
        "Rule: highest dev MRR@5 wins; ties within 0.01 go to the config needing no new embedding model, then "
        f"to the lower latency. Baseline = `{BASELINE}`, the original production pipeline. Only the baseline, "
        "the winner and the deployed retriever are scored on test.",
        "",
        "## Dev (all configs)",
        "",
        *head,
    ]
    for name, cell in sorted(r.items(), key=lambda x: -x[1]["dev"]["mrr@5"]):
        label = f"**{name}** (winner)" if name == w else f"{name} (baseline)" if name == BASELINE else name
        lines.append(_row(label, cell["dev"]))
    lines += ["", f"## Test (once): baseline vs the dev winner `{w}`", "", *head,
              _row(f"{BASELINE} (baseline)", r[BASELINE]["test"]), _row(w, r[w]["test"]),
              _row("deployed (rag/explain.GuidelineRetriever)", r["deployed (rag/explain.GuidelineRetriever)"]["test"]),
              "",
              "Paired bootstrap 95% CI of (winner − baseline) over test questions:", ""]
    for m, (d, lo, hi) in out["test_ci"].items():
        lines.append(f"- {m}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    lines += ["", "Labels were written by one labeller (the coding agent) from the handbook text; with 30 test "
                  "questions one question moves hit@k by 0.033."]
    return "\n".join(lines) + "\n"


def main() -> None:
    out = run()
    meta = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "k": K, "candidates": CANDIDATES}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "rag_retrieval.json"), "w") as f:
        json.dump({"meta": meta, **out}, f, indent=1)
    md = to_markdown(out, meta)
    with open(os.path.join(RESULTS_DIR, "rag_retrieval.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
