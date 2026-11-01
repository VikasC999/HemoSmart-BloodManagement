"""
HemoSmart - Experiment: add WHO's 2021 educational modules to the RAG corpus?
--------------------------------------------------------------------------------
EXPERIMENT ONLY -- production (rag/faiss_index, rag/explain.py) is not
changed. Builds two experimental indexes with the production pipeline
(section-aware 500-char chunks, MiniLM, FAISS top 20 + cross-encoder
rerank, rag/explain.GuidelineRetriever):

  combined      2001 handbook + 2021 modules (each chunk's `source` names
                its document, so citations do too)
  2021-only     the modules alone

and scores them against production (2001 only) on two frozen question
sets: the original 45 (evaluation/rag_eval_set.json, D88) and 18 new ones
answerable from the modules (evaluation/rag_eval_set_2021.json, frozen
before this script first ran). A hit needs the right (document, page).

With --llm it also runs the 10 fixed explanation cases of
evaluation/rag_faithfulness.py through ExplanationGenerator with the
production and the combined retriever, twice each (40 Groq calls,
cached per call in EXPLANATION_CACHE so a rate-limited run resumes).

Source: WHO & ISBT, Educational modules on clinical use of blood (2021),
CC BY-NC-SA 3.0 IGO, https://www.who.int/publications/i/item/9789240033733

Run (from the repo root):
    python -m evaluation.rag_2021_experiment          # retrieval only
    python -m evaluation.rag_2021_experiment --llm    # + explanations

Writes evaluation/results/rag_2021_experiment.md and .json.
"""

import argparse
import json
import os
import time
from datetime import datetime

# XGBoost must load before FAISS/torch on macOS (two OpenMP runtimes).
from agents.tools import predict_transfusion  # noqa: I001
import numpy as np

import evaluation.rag_faithfulness as rf
from evaluation.rag_retrieval import paired_bootstrap, score_ranking
from rag.explain import EMBEDDING_MODEL, ExplanationGenerator, GuidelineRetriever, citation_label
from rag.retrieval import SOURCE_TITLE, Reranker, section_chunks
from schema.patient_schema import PatientRecord

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "evaluation", "results")
HANDBOOK = os.path.join(REPO_ROOT, "rag", "who_guidelines.pdf")
MODULES = os.path.join(REPO_ROOT, "data", "external", "who_clinical_use_modules_2021.pdf")
EXP_INDEX = os.path.join(REPO_ROOT, "rag", "faiss_index_2021exp")
MODULES_TITLE = "WHO & ISBT, Educational modules on clinical use of blood (2021)"
SOURCES = {"who2001": SOURCE_TITLE, "who2021": MODULES_TITLE}
SETS = {"2001 set (45)": "rag_eval_set.json", "2021 set (18)": "rag_eval_set_2021.json"}
K = 5
METRICS = ("hit@1", "hit@2", "hit@5", "mrr@5", "ndcg@5")
RUNS = 2
# Each finished explanation is saved here, so a run stopped by Groq's free-tier daily token limit
# (200k tokens; this experiment needs ~40 x 5k) resumes instead of starting over.
EXPLANATION_CACHE = os.path.join(RESULTS_DIR, "rag_2021_experiment_explanations.json")


# ----------------------------------------------------------------------
# The 2021 modules' layout (MyriadPro): headings are Bold >= 12 pt or
# Semibold >= 10.5 pt; the running header sits in a band at the top.
# ----------------------------------------------------------------------
def modules_heading(line: dict) -> bool:
    spans = [s for s in line["spans"] if s["text"].strip()]
    if not spans or line["bbox"][3] <= 50:
        return False
    font, size = spans[0]["font"], spans[0]["size"]
    return ("Bold" in font and size >= 12) or ("Semibold" in font and size >= 10.5)


def modules_header(line: dict, text: str):
    if line["bbox"][3] <= 50:
        return "chapter" if text.upper().startswith("CHAPTER") else "skip"
    return None


def modules_chunks(size: int = 500) -> list:
    return section_chunks(MODULES, size, source=MODULES_TITLE, is_heading=modules_heading, is_header=modules_header)


# ----------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------
def build_retrievers(reranker) -> dict:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    handbook, modules = section_chunks(HANDBOOK, 500), modules_chunks(500)
    out = {"production (2001)": GuidelineRetriever(reranker=reranker)}
    for name, chunks in (("combined", handbook + modules), ("2021-only", modules)):
        path = os.path.join(EXP_INDEX, name)
        store = FAISS.from_documents(chunks, embeddings)
        store.save_local(path)
        with open(os.path.join(path, "index_info.json"), "w") as f:
            json.dump({"embedding_model": EMBEDDING_MODEL, "chunking": "section-500", "chunks": len(chunks),
                       "experiment": "evaluation/rag_2021_experiment.py -- not used by production"}, f, indent=1)
        out[name] = GuidelineRetriever(vectorstore=store, reranker=reranker)
    return out, {"2001 handbook": len(handbook), "2021 modules": len(modules)}


def load_items(fname: str) -> list:
    with open(os.path.join(REPO_ROOT, "evaluation", fname)) as f:
        items = json.load(f)["items"]
    for it in items:
        it["relevant"] = {(SOURCES[it.get("source", "who2001")], p) for p in it["pages"]}
    return items


def evaluate(retriever, items: list) -> dict:
    retriever.search("warm-up", k=K)
    rows, times, share_2021 = [], [], []
    for it in items:
        runs = []
        for _ in range(3):
            t = time.perf_counter()
            docs = retriever.search(it["question"], k=K)
            runs.append((time.perf_counter() - t) * 1000)
        times.append(float(np.median(runs)))
        keys = [(d.metadata.get("source"), d.metadata.get("page")) for d in docs]
        rows.append(score_ranking(keys, it["relevant"]))
        share_2021.append(sum(s == MODULES_TITLE for s, _ in keys[:2]) / 2)
    return {**{m: round(float(np.mean([r[m] for r in rows])), 3) for m in METRICS},
            "top2_from_2021": round(float(np.mean(share_2021)), 3),
            "latency_ms_mean": round(float(np.mean(times)), 1), "latency_ms_p95": round(float(np.percentile(times, 95)), 1),
            "per_question": rows}


def retrieval(retrievers: dict) -> dict:
    out = {}
    for set_name, fname in SETS.items():
        items = load_items(fname)
        out[set_name] = {}
        for split in ("test", "all"):
            subset = [i for i in items if split == "all" or i["split"] == split]
            out[set_name][split] = {name: evaluate(r, subset) for name, r in retrievers.items()}
            out[set_name][split]["n"] = len(subset)
        test = out[set_name]["test"]
        out[set_name]["ci_vs_production"] = {
            name: {m: paired_bootstrap([r[m] for r in test["production (2001)"]["per_question"]],
                                       [r[m] for r in test[name]["per_question"]]) for m in ("mrr@5", "hit@2")}
            for name in ("combined", "2021-only")}
        print(set_name, {n: test[n]["mrr@5"] for n in retrievers}, flush=True)
    return out


# ----------------------------------------------------------------------
# Explanations (--llm)
# ----------------------------------------------------------------------
def explanations(retrievers: dict) -> dict:
    cache = json.load(open(EXPLANATION_CACHE)) if os.path.exists(EXPLANATION_CACHE) else {}
    out = {}
    for name in ("production (2001)", "combined"):
        generator = ExplanationGenerator(retriever=retrievers[name])
        runs = []
        for run in range(RUNS):
            rows = []
            for case in rf.CASES:
                record = PatientRecord(**case)
                key = f"{name}|{run}|{rf.CASES.index(case)}"
                if key not in cache:
                    needed = predict_transfusion(record)["transfusion_needed"]
                    r = rf._generate(generator, record, needed)
                    cache[key] = {"explanation": r["explanation"], "citations": r["citations"]}
                    with open(EXPLANATION_CACHE, "w") as f:
                        json.dump(cache, f, indent=1)
                result = cache[key]
                docs = generator.retrieve_sources_per_parameter(record)
                sources = [d.page_content for d in docs]
                checks = rf.check(result["explanation"], case, rf.describe_threshold_facts(record), sources)
                refs = [int(n) for n in rf._CITE.findall(result["explanation"]) if 1 <= int(n) <= len(docs)]
                rows.append({"case": case, "explanation": result["explanation"], **checks,
                             "nli": rf.nli_support(result["explanation"], sources),
                             "citations": [citation_label(c) for c in result["citations"]],
                             "cited_2021": sum(docs[n - 1].metadata.get("source") == MODULES_TITLE for n in refs),
                             "provided_2021": sum(d.metadata.get("source") == MODULES_TITLE for d in docs)})
                print(f"{name:<18} run {run + 1} {case['surgery_type']:<10} contra {checks['contradictions']} "
                      f"misattr {checks['misattributed']} cited_2021 {rows[-1]['cited_2021']}", flush=True)
            runs.append({
                "with_misattribution": sum(bool(r["misattributed"]) for r in rows),
                "with_unsupported_number": sum(bool(r["unsupported"]) for r in rows),
                "with_contradiction": sum(bool(r["contradictions"]) for r in rows),
                "with_bad_citation": sum(bool(r["bad_citations"]) for r in rows),
                "citing_a_source": sum(r["cites_any"] for r in rows),
                "citing_2021": sum(r["cited_2021"] > 0 for r in rows),
                "passages_from_2021": sum(r["provided_2021"] for r in rows),
                "passages_total": sum(len(r["citations"]) for r in rows),
                "nli_contradicted": sum(r["nli"]["contradicted"] for r in rows),
                "cited_sentences": sum(r["nli"]["cited_sentences"] for r in rows),
                "rows": rows,
            })
        out[name] = runs
    return out


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
def _cell(c: dict) -> str:
    return " | ".join(f"{c[m]:.3f}" for m in METRICS) + f" | {c['top2_from_2021']:.2f} | " \
        f"{c['latency_ms_mean']:.0f} / {c['latency_ms_p95']:.0f}"


def to_markdown(out: dict, meta: dict) -> str:
    lines = [
        "# HemoSmart — Experiment: adding WHO's 2021 educational modules to the RAG corpus",
        "",
        "**Experiment only; production is unchanged.** Same pipeline as production (section-aware 500-char chunks, "
        "MiniLM, FAISS top 20 + cross-encoder rerank). Chunks: " + ", ".join(f"{k} {v}" for k, v in out["chunks"].items())
        + f". Generated by `python -m evaluation.rag_2021_experiment{' --llm' if 'explanations' in out else ''}` on "
        f"{meta['generated_at']}.",
        "",
        "A hit needs the labelled (document, page). Question sets are frozen: `rag_eval_set.json` (D88, sha256 "
        f"`{meta['sha256']['rag_eval_set.json'][:12]}…`) and `rag_eval_set_2021.json` (sha256 "
        f"`{meta['sha256']['rag_eval_set_2021.json'][:12]}…`, written before this experiment first ran).",
        "",
    ]
    for set_name, r in out["retrieval"].items():
        for split in ("test", "all"):
            lines += [f"## {set_name}: {split} split (n = {r[split]['n']})", "",
                      "| Corpus | " + " | ".join(METRICS) + " | Top-2 from 2021 | Latency ms (mean / p95) |",
                      "|---|" + "---|" * (len(METRICS) + 2)]
            for name in ("production (2001)", "combined", "2021-only"):
                lines.append(f"| {name} | {_cell(r[split][name])} |")
            lines.append("")
        lines += ["Paired bootstrap 95% CI vs production, test split:", ""]
        for name, ci in r["ci_vs_production"].items():
            lines.append(f"- {name}: " + "; ".join(f"{m} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]" for m, (d, lo, hi) in ci.items()))
        lines.append("")
    lines += [
        "## Reading the 2001-set drop (post hoc, not a metric)",
        "",
        "The 2001 set credits only 2001 pages. Of the 7 test questions production answers at rank 1 but the combined "
        "corpus does not, the combined top hit is a 2021 page that answers the question in 4 (q06 red cell "
        "concentrate p.54; q20 and q21 cryoprecipitate p.59; q44 obstetric haemorrhage pp.139–140), partly in 2 "
        "(q09 platelets p.60 gives post-pooling time, not storage; q32 p.68 gives donor screening, not the tests) and "
        "not in 1 (q01 adult Hb threshold, p.115 is the malaria section). Judged after seeing the results, so it "
        "explains the drop rather than replacing the frozen score.",
        "",
        "## Numeric thresholds stated in the 2021 modules (PDF pages)",
        "",
        "- Red cells, adults (p.44): Hb < 7 g/dl for stable hospitalized adults incl. critically ill; 7.5 g/dl cardiac "
        "surgery; 8 g/dl orthopaedic surgery or stable cardiovascular disease; 7 g/dl acute upper GI bleed; transfusion "
        "rarely indicated above 10 g/dl.",
        "- Red cells, children (p.44) and neonates (p.150): < 7 g/dl stable critically ill children; 7–10 g/dl with "
        "acute brain injury; neonatal table by respiratory support (Hb ≤ 7–10 g/dl).",
        "- Anaemia definitions (p.29: men < 13, non-pregnant women < 12 g/dl; p.132: pregnancy < 11 / 10.5 g/dl by "
        "trimester) -- definitions, explicitly not transfusion triggers.",
        "- Platelets (p.165, Box 7.4, children; 'based on adult literature'): prophylaxis ≤ 10 × 10⁹/L stable, ≤ 20 with "
        "bleeding risk; bleeding ≤ 10 / 30 / 50 (minor / moderate / severe), ≤ 100 at a critical site; lumbar "
        "puncture ≤ 40; liver biopsy and other surgery ≤ 50; minor surgery ≤ 20. Neonates ≤ 25 (NAIT ≤ 30). "
        "Expected increment per adult dose 15–40 × 10⁹/L (pp.60, 101).",
        "- Plasma/INR (p.174): correct INR to < 2.0 before surgery (vitamin K 0.5 mg IV if surgery can wait ≥ 6 h, "
        "otherwise plasma up to 15 ml/kg); plasma initial dose 15–20 ml/kg (p.58). DIC: platelets < 50 × 10⁹/L (p.142).",
        "- No adult platelet table outside the paediatric box, and no INR > 1.5 trigger for FFP.",
        "",
    ]
    if "explanations" not in out:
        done = len(json.load(open(EXPLANATION_CACHE))) if os.path.exists(EXPLANATION_CACHE) else 0
        lines += ["## Explanations: not yet run", "",
                  f"The explanation comparison (10 cases × {RUNS} runs × 2 corpora = 40 Groq calls, ~200k tokens) "
                  f"needs about one full day of Groq's free-tier token budget (200k tokens/day, shared with the app). "
                  f"{done} of 40 explanations are cached in `{os.path.relpath(EXPLANATION_CACHE, REPO_ROOT)}`; "
                  "`python -m evaluation.rag_2021_experiment --llm` resumes from there.", ""]
    else:
        e = out["explanations"]
        lines += [f"## Explanations: 10 fixed cases × {RUNS} runs (Groq)", "",
                  "| Per run (out of 10) | " + " | ".join(f"{n} run {i + 1}" for n in e for i in range(RUNS)) + " |",
                  "|---|" + "---|" * (len(e) * RUNS)]
        for key, label in (("with_misattribution", "number credited to WHO that no passage contains"),
                           ("with_unsupported_number", "number from nowhere"),
                           ("with_contradiction", "threshold claim contradicting the protocol checks"),
                           ("with_bad_citation", "citation to a source not provided"),
                           ("citing_a_source", "cites at least one passage"),
                           ("citing_2021", "cites a 2021-module passage"),
                           ("passages_from_2021", "passages given from 2021 (of all given)"),
                           ("nli_contradicted", "cited sentences NLI-contradicted (of cited)")):
            cells = []
            for n in e:
                for run in e[n]:
                    v = run[key]
                    if key == "passages_from_2021":
                        v = f"{v} / {run['passages_total']}"
                    elif key == "nli_contradicted":
                        v = f"{v} / {run['cited_sentences']}"
                    cells.append(str(v))
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines += ["", "### Examples (run 1)", ""]
        for i in meta["example_cases"]:
            case = rf.CASES[i]
            lines += [f"**Case {i + 1}: {case['surgery_type']}, Hb {case['hemoglobin']}, platelets {case['platelets']}, "
                      f"INR {case['INR']}**", ""]
            for n in e:
                row = e[n][0]["rows"][i]
                lines += [f"- *{n}*: {row['explanation']}", f"  - sources: " + "; ".join(row["citations"])]
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm", action="store_true", help="also run the explanation cases (40 Groq calls)")
    args = parser.parse_args()
    import hashlib
    meta = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "example_cases": [0, 3, 5],
            "sha256": {f: hashlib.sha256(open(os.path.join(REPO_ROOT, "evaluation", f), "rb").read()).hexdigest()
                       for f in SETS.values()}}
    retrievers, chunks = build_retrievers(Reranker())
    out = {"chunks": chunks, "retrieval": retrieval(retrievers)}
    if args.llm:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(REPO_ROOT, "backend", ".env"))
        out["explanations"] = explanations(retrievers)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "rag_2021_experiment.json"), "w") as f:
        json.dump({"meta": meta, **out}, f, indent=1)
    md = to_markdown(out, meta)
    with open(os.path.join(RESULTS_DIR, "rag_2021_experiment.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
