"""
HemoSmart - Transfusion Prediction Models (XGBoost + SHAP)
-------------------------------------------------------------
Trains one XGBoost classifier per blood component (PRBC, Platelets,
FFP) on the simulated patient dataset. A patient can need several
components at once (e.g. a bleeding trauma patient: red cells AND
plasma), so this is three independent binary models -- multi-label --
rather than one multi-class model that could only ever pick one.

"Transfusion needed" is then derived at inference as
P(any component) = 1 - prod(1 - p_component), see agents/tools.py.

Requires: data/patient_data.csv (run data/create_dataset.py first)

Run:
    python models/train_xgboost.py

Outputs:
    models/xgb_components.pkl          -> {"features": [...], "models": {component: model}}
    models/label_encoder.pkl           -> encoder for surgery_type
    models/shap_summary_<component>.png
    models/metrics.txt                 -> per-component and overall metrics
"""

import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")  # no display needed, just save the plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bloodbank.catalog import COMPONENTS

FEATURES = ["hemoglobin", "platelets", "INR", "age", "surgery_type"]
LABELS = {"PRBC": "needs_prbc", "Platelets": "needs_platelets", "FFP": "needs_ffp"}


def make_model() -> XGBClassifier:
    """The classifier configuration, shared with evaluation/prediction_benchmark.py."""
    return XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
    )


def main() -> None:
    # ----------------------------------------------------------------------
    # 1. Load + preprocess
    # ----------------------------------------------------------------------
    data = pd.read_csv("data/patient_data.csv")
    print("Loaded dataset:", data.shape)

    le = LabelEncoder()
    data["surgery_type"] = le.fit_transform(data["surgery_type"])

    X = data[FEATURES]
    train_idx, test_idx = train_test_split(
        data.index, test_size=0.2, random_state=42, stratify=data["transfusion_needed"]
    )
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]

    # ----------------------------------------------------------------------
    # 2. One model per component
    # ----------------------------------------------------------------------
    models = {}
    report_lines = []
    test_probs = {}
    for component in COMPONENTS:
        y = data[LABELS[component]]
        model = make_model()
        model.fit(X_train, y.loc[train_idx])
        models[component] = model

        y_pred = model.predict(X_test)
        test_probs[component] = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y.loc[test_idx], y_pred)
        print(f"\n[{component}] accuracy: {acc:.4f}")
        report_lines.append(f"=== {component} (label: {LABELS[component]}) ===\nAccuracy: {acc:.4f}\n")
        report_lines.append(classification_report(y.loc[test_idx], y_pred))

        # SHAP summary per component
        explainer = shap.TreeExplainer(model)
        shap.summary_plot(explainer.shap_values(X_test), X_test, show=False)
        plt.title(f"SHAP - {component}")
        plt.tight_layout()
        plt.savefig(f"models/shap_summary_{component}.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Overall "any transfusion" decision, exactly as inference derives it
    p_any = 1 - np.prod([1 - test_probs[c] for c in COMPONENTS], axis=0)
    y_any = data.loc[test_idx, "transfusion_needed"]
    any_pred = (p_any >= 0.5).astype(int)
    acc_any = accuracy_score(y_any, any_pred)
    print(f"\n[ANY TRANSFUSION] accuracy: {acc_any:.4f}")
    report_lines.insert(0, f"=== Any transfusion (derived) ===\nAccuracy: {acc_any:.4f}\n"
                           + classification_report(y_any, any_pred) + "\n")

    with open("models/metrics.txt", "w") as f:
        f.write("\n".join(report_lines))

    # ----------------------------------------------------------------------
    # 3. Save models + encoder
    # ----------------------------------------------------------------------
    with open("models/xgb_components.pkl", "wb") as f:
        pickle.dump({"features": FEATURES, "models": models}, f)

    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print("\nSaved models/xgb_components.pkl and models/label_encoder.pkl")


if __name__ == "__main__":
    main()
