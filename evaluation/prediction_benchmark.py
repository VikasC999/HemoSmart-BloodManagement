"""
HemoSmart - Transfusion Prediction Benchmark
------------------------------------------------
Is the ML model better than a simple threshold rule, and how far is it
from the best any method could do on this data?

Compared, per component (PRBC / Platelets / FFP) and for "any
transfusion", with 5-fold stratified cross-validation on
data/patient_data.csv:

  static rule        one Hb threshold (10 g/dL only for Emergency), one
                     platelet threshold (50k) for everyone, INR > 1.5 --
                     what a simple threshold system does
  logistic reg.      a standard linear ML baseline on the same features
  random forest      the other tree ensemble the report names (300 trees)
  XGBoost            the deployed model (models/train_xgboost.make_model)
  ceiling            the true label probabilities from the label process
                     (data/label_process.py); thresholded at 0.5 this is
                     the context-aware guideline rule, which is
                     Bayes-optimal here. No model can beat it on average.

Labels are simulated clinician decisions, so these numbers measure how
well each method recovers the simulated decision process -- not
clinical accuracy on real patients.

Run (from the repo root):
    python -m evaluation.prediction_benchmark

Writes evaluation/results/prediction_benchmark.md and .json.
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bloodbank.catalog import COMPONENTS
from data.label_process import decision_probabilities
from models.train_xgboost import LABELS, make_model
from schema.clinical_thresholds import FLAG_COMPONENT, guideline_flags, static_guideline_flags

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "evaluation", "results")
NUMERIC = ["hemoglobin", "platelets", "INR", "age"]
SURGERY_TYPES = ["Cardiac", "Emergency", "General", "Orthopedic"]
METHODS = ("static_rule", "logistic", "random_forest", "xgboost", "ceiling")
METHOD_LABELS = {
    "static_rule": "Static threshold rule",
    "logistic": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost (HemoSmart)",
    "ceiling": "Ceiling (true label process)",
}
FOLDS = 5


def _xgb_features(df: pd.DataFrame) -> pd.DataFrame:
    # Same encoding as training: LabelEncoder on sorted surgery types
    out = df[NUMERIC].copy()
    out["surgery_type"] = df["surgery_type"].map({s: i for i, s in enumerate(SURGERY_TYPES)})
    return out[["hemoglobin", "platelets", "INR", "age", "surgery_type"]]


def _logistic():
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(categories=[SURGERY_TYPES]), ["surgery_type"]),
    ])
    return make_pipeline(pre, LogisticRegression(max_iter=1000))


def _static_scores(df: pd.DataFrame) -> dict:
    flags = [static_guideline_flags(r.hemoglobin, r.platelets, r.INR, r.surgery_type) for r in df.itertuples()]
    return {FLAG_COMPONENT[k]: np.array([float(f[k]) for f in flags]) for k in FLAG_COMPONENT}


def _metrics(y, score, has_probabilities: bool) -> dict:
    pred = (score >= 0.5).astype(int)
    out = {
        "accuracy": accuracy_score(y, pred),
        "sensitivity": recall_score(y, pred, zero_division=0),
        "specificity": recall_score(1 - y, 1 - pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }
    if has_probabilities:
        out["auroc"] = roc_auc_score(y, score)
        out["brier"] = brier_score_loss(y, score)
    return out


def run() -> dict:
    df = pd.read_csv(os.path.join(REPO_ROOT, "data", "patient_data.csv"))
    truth = decision_probabilities(df.hemoglobin, df.platelets, df.INR, df.surgery_type)
    folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

    per_fold = {m: {t: [] for t in (*COMPONENTS, "Any")} for m in METHODS}
    for train, test in folds.split(df, df.transfusion_needed):
        tr, te = df.iloc[train], df.iloc[test]
        scores = {m: {} for m in METHODS}
        scores["static_rule"] = _static_scores(te)
        for c in COMPONENTS:
            y_tr = tr[LABELS[c]]
            xgb = make_model().fit(_xgb_features(tr), y_tr)
            scores["xgboost"][c] = xgb.predict_proba(_xgb_features(te))[:, 1]
            scores["logistic"][c] = _logistic().fit(tr, y_tr).predict_proba(te)[:, 1]
            rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1)
            scores["random_forest"][c] = rf.fit(_xgb_features(tr), y_tr).predict_proba(_xgb_features(te))[:, 1]
            scores["ceiling"][c] = truth[c][test]

        for m in METHODS:
            # "Any transfusion" exactly as inference derives it (agents/tools.py)
            scores[m]["Any"] = 1 - np.prod([1 - scores[m][c] for c in COMPONENTS], axis=0)
            for c in COMPONENTS:
                per_fold[m][c].append(_metrics(te[LABELS[c]].values, scores[m][c], m != "static_rule"))
            per_fold[m]["Any"].append(_metrics(te.transfusion_needed.values, scores[m]["Any"], m != "static_rule"))

    summary = {}
    for m in METHODS:
        summary[m] = {}
        for target, rows in per_fold[m].items():
            summary[m][target] = {
                k: {"mean": round(float(np.mean([r[k] for r in rows])), 4),
                    "std": round(float(np.std([r[k] for r in rows])), 4)}
                for k in rows[0]
            }

    # Patients whose surgical context changes the guideline decision
    context_cases = {c: 0 for c in COMPONENTS}
    for r in df.itertuples():
        static = static_guideline_flags(r.hemoglobin, r.platelets, r.INR, r.surgery_type)
        context = guideline_flags(r.hemoglobin, r.platelets, r.INR, r.surgery_type)
        for k, c in FLAG_COMPONENT.items():
            context_cases[c] += context[k] and not static[k]
    return {"summary": summary, "context_cases": context_cases, "n": len(df),
            "positives": {c: int(df[LABELS[c]].sum()) for c in COMPONENTS}}


def to_markdown(result: dict, meta: dict) -> str:
    s = result["summary"]
    lines = [
        "# HemoSmart — Transfusion Prediction Benchmark",
        "",
        f"{result['n']} synthetic patients, {FOLDS}-fold stratified cross-validation, mean ± std across folds. "
        f"Generated by `python -m evaluation.prediction_benchmark` on {meta['generated_at']}.",
        "",
        "Labels are simulated clinician decisions (`data/label_process.py`): context-dependent guideline "
        "thresholds, a soft boundary, 1% record noise. Positives: "
        + ", ".join(f"{c} {n}" for c, n in result["positives"].items()) + ".",
        "",
    ]
    for target in (*COMPONENTS, "Any"):
        title = "Any transfusion" if target == "Any" else target
        lines += [f"## {title}", "", "| Method | Accuracy | Sensitivity | Specificity | F1 | AUROC | Brier |",
                  "|---|---|---|---|---|---|---|"]
        for m in METHODS:
            cell = s[m][target]
            fmt = lambda k: f"{cell[k]['mean']:.3f} ± {cell[k]['std']:.3f}" if k in cell else "—"  # noqa: E731
            lines.append(f"| {METHOD_LABELS[m]} | " + " | ".join(
                fmt(k) for k in ("accuracy", "sensitivity", "specificity", "f1", "auroc", "brier")) + " |")
        lines.append("")
    lines += [
        "Patients whose surgical context changes the guideline decision (the context-aware rule says transfuse, "
        "the static rule doesn't): " + ", ".join(f"{c} {n}" for c, n in result["context_cases"].items()) + ". "
        "This is where a model that sees surgery type can beat the static rule; FFP has no context "
        "adjustment, so there it can only tie.",
        "",
        "Rules output yes/no, so AUROC and Brier are reported only for probabilistic methods. "
        "The ceiling is not a method: it's the probability the label generator itself used, so it bounds what "
        "any model can achieve on this data.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    result = run()
    meta = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "folds": FOLDS}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "prediction_benchmark.json"), "w") as f:
        json.dump({"meta": meta, **result}, f, indent=2)
    md = to_markdown(result, meta)
    with open(os.path.join(RESULTS_DIR, "prediction_benchmark.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
