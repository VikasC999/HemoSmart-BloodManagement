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

Also reported: each method's F1/AUROC as a share of the ceiling's, the
"always no" accuracy (why raw accuracy misleads here), and a clinical
operating point -- the threshold, chosen on out-of-fold predictions
inside each training fold, that keeps sensitivity >= 0.90.

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
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bloodbank.catalog import COMPONENTS
from data.label_process import RECORD_NOISE, decision_probabilities
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
TARGET_SENSITIVITY = 0.90
OPERATING_METHODS = ("xgboost", "ceiling")


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


def threshold_for_sensitivity(y, score, target: float = TARGET_SENSITIVITY) -> float:
    """Highest threshold t with sensitivity(score >= t) >= target."""
    pos = np.sort(np.asarray(score)[np.asarray(y) == 1])[::-1]
    return float(pos[int(np.ceil(target * len(pos))) - 1])


def _operating_point(y, score, threshold: float) -> dict:
    pred = (score >= threshold).astype(int)
    return {
        "threshold": threshold,
        "sensitivity": recall_score(y, pred, zero_division=0),
        "specificity": recall_score(1 - y, 1 - pred, zero_division=0),
        "ppv": precision_score(y, pred, zero_division=0),
    }


def share_of_ceiling(summary: dict) -> dict:
    """{method: {target: {"f1": share, "auroc": share or None}}}, means only."""
    ceiling = summary["ceiling"]
    return {
        m: {t: {k: round(cell[k]["mean"] / ceiling[t][k]["mean"], 3) if k in cell else None
                for k in ("f1", "auroc")}
            for t, cell in summary[m].items()}
        for m in METHODS if m != "ceiling"
    }


def _mean_std(rows: list) -> dict:
    return {k: {"mean": round(float(np.mean([r[k] for r in rows])), 4),
                "std": round(float(np.std([r[k] for r in rows])), 4)}
            for k in rows[0]}


def run() -> dict:
    df = pd.read_csv(os.path.join(REPO_ROOT, "data", "patient_data.csv"))
    truth = decision_probabilities(df.hemoglobin, df.platelets, df.INR, df.surgery_type)
    folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

    per_fold = {m: {t: [] for t in (*COMPONENTS, "Any")} for m in METHODS}
    operating = {m: {t: [] for t in (*COMPONENTS, "Any")} for m in OPERATING_METHODS}
    for train, test in folds.split(df, df.transfusion_needed):
        tr, te = df.iloc[train], df.iloc[test]
        scores = {m: {} for m in METHODS}
        # Training-fold scores the thresholds are chosen on: out-of-fold for XGBoost (in-sample
        # probabilities are overconfident), the label probabilities themselves for the ceiling.
        train_scores = {"xgboost": {}, "ceiling": {}}
        inner = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
        scores["static_rule"] = _static_scores(te)
        for c in COMPONENTS:
            y_tr = tr[LABELS[c]]
            xgb = make_model().fit(_xgb_features(tr), y_tr)
            scores["xgboost"][c] = xgb.predict_proba(_xgb_features(te))[:, 1]
            train_scores["xgboost"][c] = cross_val_predict(
                make_model(), _xgb_features(tr), y_tr, cv=inner, method="predict_proba")[:, 1]
            train_scores["ceiling"][c] = truth[c][train]
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

        for m in OPERATING_METHODS:
            train_scores[m]["Any"] = 1 - np.prod([1 - train_scores[m][c] for c in COMPONENTS], axis=0)
            for t in (*COMPONENTS, "Any"):
                label = "transfusion_needed" if t == "Any" else LABELS[t]
                threshold = threshold_for_sensitivity(tr[label].values, train_scores[m][t])
                operating[m][t].append(_operating_point(te[label].values, scores[m][t], threshold))

    summary = {m: {t: _mean_std(rows) for t, rows in per_fold[m].items()} for m in METHODS}
    operating_point = {m: {t: _mean_std(rows) for t, rows in operating[m].items()} for m in OPERATING_METHODS}
    labels = {**{c: LABELS[c] for c in COMPONENTS}, "Any": "transfusion_needed"}
    always_no_accuracy = {t: round(float(1 - df[col].mean()), 4) for t, col in labels.items()}
    # Positives the label process produced only through record noise (clinical probability ~0)
    noise_positives = {c: round(float((truth[c][df[LABELS[c]] == 1] <= RECORD_NOISE + 1e-3).mean()), 3)
                       for c in COMPONENTS}

    # Patients whose surgical context changes the guideline decision
    context_cases = {c: 0 for c in COMPONENTS}
    for r in df.itertuples():
        static = static_guideline_flags(r.hemoglobin, r.platelets, r.INR, r.surgery_type)
        context = guideline_flags(r.hemoglobin, r.platelets, r.INR, r.surgery_type)
        for k, c in FLAG_COMPONENT.items():
            context_cases[c] += context[k] and not static[k]
    return {"summary": summary, "share_of_ceiling": share_of_ceiling(summary),
            "operating_point": operating_point, "always_no_accuracy": always_no_accuracy,
            "noise_positives": noise_positives,
            "context_cases": context_cases, "n": len(df),
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
        lines += [f"## {title}", "",
                  "| Method | Accuracy | Sensitivity | Specificity | F1 | AUROC | Brier | F1 / ceiling | AUROC / ceiling |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for m in METHODS:
            cell = s[m][target]
            share = result["share_of_ceiling"].get(m, {}).get(target, {})
            fmt = lambda k: f"{cell[k]['mean']:.3f} ± {cell[k]['std']:.3f}" if k in cell else "—"  # noqa: E731
            pct = lambda k: f"{share[k] * 100:.0f}%" if share.get(k) is not None else "—"  # noqa: E731
            lines.append(f"| {METHOD_LABELS[m]} | " + " | ".join(
                fmt(k) for k in ("accuracy", "sensitivity", "specificity", "f1", "auroc", "brier"))
                + f" | {pct('f1')} | {pct('auroc')} |")
        lines += ["", f"Always answering \"no\" scores accuracy {result['always_no_accuracy'][target]:.3f} here "
                  "(sensitivity 0, F1 0).", ""]
    lines += [
        "## Why accuracy alone is misleading",
        "",
        "Most patients need no transfusion of a given component, so a model that never recommends one is "
        "already \"accurate\": " + ", ".join(
            f"{'Any' if t == 'Any' else t} {v:.3f}" for t, v in result["always_no_accuracy"].items())
        + ". Every method above sits within a few points of that, which says nothing about the patients who "
        "do need blood. Sensitivity, F1, AUROC and the share of the ceiling are the numbers to read.",
        "",
        f"## Clinical operating point (sensitivity ≥ {TARGET_SENSITIVITY:.2f})",
        "",
        "Missing a needed transfusion is worse than an extra crossmatch, so a clinical deployment would not "
        "use the default 0.5 cut-off. Inside each outer fold the threshold is chosen on the training fold "
        "only (XGBoost: out-of-fold predictions from a 5-fold inner CV; ceiling: the label probabilities) as "
        f"the highest one that keeps training sensitivity ≥ {TARGET_SENSITIVITY:.2f}; the test fold is then "
        "scored at that threshold. Mean ± std across the outer folds.",
        "",
        "| Target | Method | Threshold | Sensitivity | Specificity | PPV |",
        "|---|---|---|---|---|---|",
    ]
    for target in (*COMPONENTS, "Any"):
        for m in OPERATING_METHODS:
            cell = result["operating_point"][m][target]
            lines.append(f"| {'Any transfusion' if target == 'Any' else target} | {METHOD_LABELS[m]} | " + " | ".join(
                f"{cell[k]['mean']:.3f} ± {cell[k]['std']:.3f}" for k in ("threshold", "sensitivity", "specificity", "ppv"))
                + " |")
    lines += [
        "",
        "PPV is the share of flagged patients who truly need the component: what reaching 90% sensitivity "
        "costs in extra crossmatches. The ceiling row shows how much of that cost is in the data itself.",
        "",
        "Read the rows with a large noise share with care: "
        + ", ".join(f"{c} {v * 100:.0f}%" for c, v in result["noise_positives"].items())
        + f" of positives exist only through the label process's {RECORD_NOISE:.0%} record noise (clinical "
        "probability ~0, indistinguishable from patients who don't need blood). Reaching 90% sensitivity "
        "means reaching into that noise, so specificity drops for every method, the ceiling included, "
        "the more so the larger that share and the rarer the component. "
        "On real data this cost would depend on how noisy real transfusion records are.",
        "",
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
