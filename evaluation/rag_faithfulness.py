"""
HemoSmart - RAG Retrieval & Faithfulness Evaluation
-------------------------------------------------------
Two questions about the explanation pipeline (rag/explain.py):

1. RETRIEVAL -- do the per-parameter queries fetch the handbook pages
   that actually discuss that decision? Ground truth is a hand-labelled
   set of WHO handbook pages (RELEVANT_PAGES below, each checked by
   reading the page). Scored as hit@2 (the explanation uses the top 2
   per query) and MRR@5. No LLM calls.

2. FAITHFULNESS (--llm) -- does the LLM's explanation stay within what
   its grounding says? Checked automatically on a fixed set of patient
   cases, for each prompt version:
     contradiction   says a value crosses a threshold the protocol
                     checks say it doesn't (or vice versa)
     misattribution  credits WHO / the handbook / "guidelines" with a
                     number that isn't in any cited excerpt
     unsupported     uses a number that's in neither the patient data,
                     the protocol checks, nor the retrieved excerpts
     bad citation    cites [n] for a source that wasn't provided
   Each case costs one LLM call per prompt version (Groq).

Compares the first version ("legacy": generic American-spelling queries,
original prompt) with the current one (handbook-vocabulary queries,
prompt that separates HemoSmart's protocol from the handbook).

Run (from the repo root):
    python -m evaluation.rag_faithfulness          # retrieval only
    python -m evaluation.rag_faithfulness --llm    # + faithfulness

Writes evaluation/results/rag_faithfulness.md and .json.
"""

import argparse
import json
import os
import re
import time
from datetime import datetime

# XGBoost must load before FAISS/torch: the other order segfaults on
# macOS (two OpenMP runtimes). The backend imports in this order too.
from agents.tools import predict_transfusion  # noqa: I001
from rag.explain import EXPLANATION_PROMPT_TEMPLATE, ExplanationGenerator, GuidelineRetriever, parameter_queries
from schema.clinical_thresholds import describe_threshold_facts, guideline_flags
from schema.patient_schema import PatientRecord

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "evaluation", "results")
SURGERY_TYPES = ("General", "Orthopedic", "Cardiac", "Emergency")

# Hand-labelled: handbook pages (PDF page numbers, as cited) that state a
# transfusion criterion for this parameter. See DESIGN_DECISIONS D52.
RELEVANT_PAGES = {
    "rbc": {98, 121, 160},              # adults <7 g/dl; bleed, maintain >9; preoperative 7-8 g/dl
    "platelets": {77, 102, 147, 162},   # 50 x 10^9/L with bleeding; 10-20 prophylactic; surgery
    "ffp": {30, 118, 155, 162},         # FFP indications; massive transfusion; INR < 2.0 before surgery
    "surgery": set(range(158, 177)),    # the "Surgery & anaesthesia" chapter
}
QUERY_PARAMETER = ("rbc", "platelets", "ffp", "surgery")  # order of the query lists

LEGACY_PROMPT = """
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

Numbered guideline sources (WHO handbook excerpts):
{context}

Using ONLY the threshold facts above as the source of truth, write a
2-3 sentence clinical explanation of why the prediction is consistent
with the guidelines. Cite the sources you rely on with their bracketed
numbers, e.g. [1]. Do not state that any value crosses a threshold
unless the facts above say it does. Do not invent additional criteria
or sources.
"""


def legacy_queries(surgery_type: str) -> list:
    return [
        f"hemoglobin transfusion threshold for {surgery_type} surgery",
        "platelet transfusion threshold before surgery",
        "INR fresh frozen plasma transfusion threshold",
        f"{surgery_type} surgery transfusion guidelines",
    ]


VERSIONS = {
    "legacy": {"queries": legacy_queries, "prompt": LEGACY_PROMPT},
    "current": {"queries": parameter_queries, "prompt": EXPLANATION_PROMPT_TEMPLATE},
}

# Fixed cases covering each component, combinations, borderline values
# and context thresholds (cardiac).
CASES = [
    dict(hemoglobin=6.8, platelets=210_000, INR=1.1, age=64, surgery_type="General"),
    dict(hemoglobin=12.9, platelets=32_000, INR=1.0, age=41, surgery_type="Orthopedic"),
    dict(hemoglobin=13.4, platelets=240_000, INR=2.3, age=70, surgery_type="General"),
    dict(hemoglobin=7.2, platelets=45_000, INR=1.9, age=35, surgery_type="Emergency"),
    dict(hemoglobin=8.6, platelets=180_000, INR=1.0, age=72, surgery_type="Cardiac"),
    dict(hemoglobin=13.0, platelets=85_000, INR=1.2, age=60, surgery_type="Cardiac"),
    dict(hemoglobin=9.4, platelets=150_000, INR=1.3, age=29, surgery_type="Emergency"),
    dict(hemoglobin=8.1, platelets=52_000, INR=1.5, age=55, surgery_type="General"),
    dict(hemoglobin=14.1, platelets=260_000, INR=1.0, age=47, surgery_type="Orthopedic"),
    dict(hemoglobin=10.5, platelets=130_000, INR=1.6, age=81, surgery_type="Cardiac"),
]


# ----------------------------------------------------------------------
# 1. Retrieval
# ----------------------------------------------------------------------
def retrieval_scores(retriever: GuidelineRetriever) -> dict:
    out = {}
    for version, cfg in VERSIONS.items():
        rows = {p: {"hit_at_2": [], "rr_at_5": [], "pages": []} for p in QUERY_PARAMETER}
        for surgery in SURGERY_TYPES:
            for param, query in zip(QUERY_PARAMETER, cfg["queries"](surgery)):
                pages = [d.metadata.get("page") for d in retriever.search(query, k=5)]
                relevant = RELEVANT_PAGES[param]
                rank = next((i + 1 for i, p in enumerate(pages) if p in relevant), None)
                rows[param]["hit_at_2"].append(any(p in relevant for p in pages[:2]))
                rows[param]["rr_at_5"].append(1 / rank if rank else 0.0)
                rows[param]["pages"].append(pages[:2])
        out[version] = {
            p: {"hit_at_2": round(sum(r["hit_at_2"]) / len(r["hit_at_2"]), 3),
                "mrr_at_5": round(sum(r["rr_at_5"]) / len(r["rr_at_5"]), 3),
                "top2_pages_example": r["pages"][0]}
            for p, r in rows.items()
        }
    return out


# ----------------------------------------------------------------------
# 2. Faithfulness checks (deterministic, on the LLM's text)
# ----------------------------------------------------------------------
_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w])")
# "50 x 10^9/L", "50 ×10⁹/L", "50 x 109/L" (PDF-flattened superscript) and
# "32 × 10³/µL" are all thousands per microliter
_SCI = re.compile(r"(\d+(?:\.\d+)?)\s?[x×]\s?10\s?(?:\^\s?)?(9|⁹|3|³)(?:\s?/\s?[Lµu]L?)?", re.I)
_GROUPED = re.compile(r"(?<=\d)[,\u00a0\u202f\u2009 ](?=\d{3}(?!\d))")  # 32 000 / 32,000
_ATTRIBUTION = re.compile(r"\bWHO\b|handbook|guideline", re.I)
_CITE = re.compile(r"\[(\d+)\]")
_PARAM_WORDS = {"rbc": r"h(a)?emoglobin|\bHb\b", "platelets": r"platelet", "ffp": r"\bINR\b"}
# Wording that says a value crosses its threshold, per direction. RBC and
# platelets cross by being low; INR by being high.
_SAYS_LOW = re.compile(r"\b(below|under|less than|falls short)\b")
_SAYS_HIGH = re.compile(r"\b(above|exceeds?|over|greater than)\b")
_NEGATED = re.compile(r"\b(not|no|neither|nor|within|at or above|at or below|normal)\b|[≥≤]")


def _normalise(text: str) -> str:
    text = _CITE.sub("", text)
    text = _SCI.sub(lambda m: f"{float(m.group(1)) * 1000:g}", text)
    return _GROUPED.sub("", text)


def _numbers(text: str) -> set:
    """Numeric values in text, with unit forms unified. Bare small
    integers (1-4: doses, list items, "2-3 sentences") are skipped --
    they aren't threshold claims."""
    out = set()
    for m in _NUM.finditer(_normalise(text)):
        token = m.group(1)
        v = float(token)
        if "." not in token and v <= 4:
            continue
        out.add(v)
    return out


def _sentences(text: str) -> list:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _clauses(sentence: str) -> list:
    return [c for c in re.split(r"[,;]|\band\b|\bwhile\b|\bbut\b", sentence) if c.strip()]


def _contradicts(clause: str, param: str, crosses: bool) -> bool:
    """True if a clause about one parameter claims the opposite of the
    protocol check. Only clauses that talk about a threshold count."""
    lowered = clause.lower()
    if not re.search(r"threshold|cut-?off|cutoff|limit", lowered):
        return False
    low, high, negated = _SAYS_LOW.search(lowered), _SAYS_HIGH.search(lowered), _NEGATED.search(clause)
    if param == "ffp":
        claims_cross = bool(high) and not negated and not low
        claims_not = bool(low or negated) and not high
    else:
        claims_cross = bool(low) and not negated and not high
        claims_not = bool(high or negated) and not low
    return (claims_cross and not crosses) or (claims_not and crosses and not claims_cross)


def check(explanation: str, record: dict, facts: str, sources: list) -> dict:
    source_nums = _numbers(" ".join(sources))
    record_nums = {float(v) for v in record.values() if isinstance(v, (int, float))}
    allowed = _numbers(facts) | source_nums | record_nums
    flags = guideline_flags(record["hemoglobin"], record["platelets"], record["INR"], record["surgery_type"])

    misattributed, unsupported, contradictions = [], [], []
    for sentence in _sentences(explanation):
        nums = _numbers(sentence)
        if _ATTRIBUTION.search(sentence):
            misattributed += [n for n in nums if n not in source_nums and n not in record_nums]
        unsupported += [n for n in nums if n not in allowed]
        for clause in _clauses(sentence):
            for param, words in _PARAM_WORDS.items():
                if re.search(words, clause, re.I) and _contradicts(clause, param, flags[param]):
                    contradictions.append(param)

    refs = [int(n) for n in _CITE.findall(explanation)]
    return {
        "misattributed": sorted(set(misattributed)),
        "unsupported": sorted(set(unsupported)),
        "contradictions": sorted(set(contradictions)),
        "bad_citations": sorted({n for n in refs if n < 1 or n > len(sources)}),
        "cites_any": bool(refs),
    }


def _generate(generator: ExplanationGenerator, record: PatientRecord, prediction: bool, retries: int = 6) -> dict:
    for attempt in range(retries):
        try:
            return generator.explain_with_sources(record, prediction=prediction)
        except Exception as e:  # rate limit on the free tier: wait and retry
            if attempt == retries - 1:
                raise
            wait = 20 * (attempt + 1)
            print(f"  LLM error ({str(e)[:80]}...), retrying in {wait}s", flush=True)
            time.sleep(wait)


def _score(version: str, explanations: list, retriever: GuidelineRetriever) -> dict:
    """Runs the checks on one version's explanations (same order as
    CASES). Retrieval and threshold facts are deterministic, so saved
    explanations can be re-scored without new LLM calls (--rescore)."""
    cfg = VERSIONS[version]
    generator = ExplanationGenerator(retriever=retriever, prompt_template=cfg["prompt"],
                                     queries=cfg["queries"], llm=object())
    rows = []
    for case, explanation in zip(CASES, explanations):
        record = PatientRecord(**case)
        sources = [d.page_content for d in generator.retrieve_sources_per_parameter(record)]
        checks = check(explanation, case, describe_threshold_facts(record), sources)
        rows.append({"case": case, "explanation": explanation, **checks})
        print(f"{version:<8} {case['surgery_type']:<10} Hb {case['hemoglobin']:<5} -> "
              f"misattr {checks['misattributed']} unsupported {checks['unsupported']} "
              f"contra {checks['contradictions']}", flush=True)
    return {
        "cases": len(rows),
        "with_misattribution": sum(bool(r["misattributed"]) for r in rows),
        "with_unsupported_number": sum(bool(r["unsupported"]) for r in rows),
        "with_contradiction": sum(bool(r["contradictions"]) for r in rows),
        "with_bad_citation": sum(bool(r["bad_citations"]) for r in rows),
        "citing_a_source": sum(r["cites_any"] for r in rows),
        "rows": rows,
    }


def faithfulness(retriever: GuidelineRetriever) -> dict:
    out = {}
    for version, cfg in VERSIONS.items():
        generator = ExplanationGenerator(retriever=retriever, prompt_template=cfg["prompt"], queries=cfg["queries"])
        explanations = []
        for case in CASES:
            record = PatientRecord(**case)
            needed = predict_transfusion(record)["transfusion_needed"]
            explanations.append(_generate(generator, record, needed)["explanation"])
        out[version] = _score(version, explanations, retriever)
    return out


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
def to_markdown(result: dict, meta: dict) -> str:
    lines = [
        "# HemoSmart — RAG Retrieval & Faithfulness",
        "",
        f"WHO *Clinical Use of Blood* handbook index (rag/faiss_index). Generated by "
        f"`python -m evaluation.rag_faithfulness{' --llm' if 'faithfulness' in result else ''}` "
        f"on {meta['generated_at']}.",
        "",
        "## Retrieval (no LLM)",
        "",
        "Each parameter's query run for all 4 surgery types; relevance = hand-labelled handbook pages "
        "(`RELEVANT_PAGES`).",
        "",
        "| Parameter | Legacy hit@2 | Legacy MRR@5 | Current hit@2 | Current MRR@5 |",
        "|---|---|---|---|---|",
    ]
    r = result["retrieval"]
    for p in QUERY_PARAMETER:
        lines.append(f"| {p} | {r['legacy'][p]['hit_at_2']:.2f} | {r['legacy'][p]['mrr_at_5']:.2f} | "
                     f"{r['current'][p]['hit_at_2']:.2f} | {r['current'][p]['mrr_at_5']:.2f} |")
    if "faithfulness" in result:
        f = result["faithfulness"]
        lines += [
            "",
            f"## Faithfulness (LLM: {meta.get('model', 'groq')}, {f['legacy']['cases']} fixed cases)",
            "",
            "| Explanations with… | Legacy | Current |",
            "|---|---|---|",
        ]
        for key, label in (("with_misattribution", "a number credited to WHO/guidelines that no excerpt contains"),
                           ("with_unsupported_number", "a number from nowhere in the grounding"),
                           ("with_contradiction", "a threshold claim contradicting the protocol checks"),
                           ("with_bad_citation", "a citation to a source that wasn't provided"),
                           ("citing_a_source", "at least one [n] citation (higher is better)")):
            lines.append(f"| {label} | {f['legacy'][key]} | {f['current'][key]} |")
        lines += ["", "Checks are regex-based and conservative: they catch numeric misattribution, not every "
                      "unsupported qualitative claim. Example explanations are in the .json file."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm", action="store_true", help="also run the faithfulness check (LLM calls)")
    parser.add_argument("--rescore", action="store_true",
                        help="re-run the checks on the explanations saved in rag_faithfulness.json (no LLM calls)")
    args = parser.parse_args()

    retriever = GuidelineRetriever()
    result = {"retrieval": retrieval_scores(retriever)}
    print(json.dumps(result["retrieval"], indent=1))
    meta = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M")}
    if args.rescore:
        with open(os.path.join(RESULTS_DIR, "rag_faithfulness.json")) as fh:
            saved = json.load(fh)
        result["faithfulness"] = {
            v: _score(v, [r["explanation"] for r in saved["faithfulness"][v]["rows"]], retriever)
            for v in VERSIONS
        }
        meta["model"] = saved["meta"].get("model")
        meta["explanations_generated_at"] = saved["meta"].get("explanations_generated_at",
                                                              saved["meta"]["generated_at"])
    elif args.llm:
        # GROQ_API_KEY lives in backend/.env. load_dotenv doesn't override
        # variables already set, so the throwaway DATABASE_URL survives.
        from dotenv import load_dotenv
        load_dotenv(os.path.join(REPO_ROOT, "backend", ".env"))
        result["faithfulness"] = faithfulness(retriever)
        meta["model"] = "openai/gpt-oss-20b via Groq"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "rag_faithfulness.json"), "w") as fh:
        json.dump({"meta": meta, **result}, fh, indent=2)
    md = to_markdown(result, meta)
    with open(os.path.join(RESULTS_DIR, "rag_faithfulness.md"), "w") as fh:
        fh.write(md)
    print(md)


if __name__ == "__main__":
    main()
