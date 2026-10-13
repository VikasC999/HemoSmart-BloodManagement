"""
HemoSmart - Simulated Transfusion Decisions (the label process)
-------------------------------------------------------------------
How the synthetic patients' labels are made. Shared by
data/create_dataset.py (to generate them) and
evaluation/prediction_benchmark.py (to know the best any model could do).

Earlier labels were a hard threshold rule on the features, so the model
"learned" a rule it could read straight off the data and scored 100%.
Real transfusion records aren't like that. Here a label is a simulated
clinician's decision:

  1. Context-dependent thresholds -- the guideline text says Hb 9-10
     with cardiovascular disease and platelets 100k for high-bleeding-risk
     procedures (schema/clinical_thresholds.py). A context-free rule
     misses those patients; a model given surgery_type can learn them.
  2. A soft boundary -- near a threshold clinicians disagree (symptoms,
     trend, bleeding). P(transfuse) is a logistic curve centred on the
     threshold rather than a step.
  3. Record noise -- a small share of decisions go the other way
     (documentation errors, information the dataset doesn't capture).

Because the curve is centred on the context threshold, the
context-aware guideline rule is also the Bayes-optimal classifier:
no model can beat it on this data, only match it. The benchmark
reports it as the ceiling.
"""

import numpy as np

from schema.clinical_thresholds import INR_THRESHOLD, hemoglobin_threshold, platelet_threshold

# Width of the zone where clinicians disagree (one logistic scale unit)
HB_SCALE = 0.3          # g/dL
PLATELET_SCALE = 6_000  # per microliter
INR_SCALE = 0.08
RECORD_NOISE = 0.01     # share of decisions flipped


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def decision_probabilities(hemoglobin, platelets, inr, surgery_type) -> dict:
    """P(a clinician orders each component), arrays in, arrays out.
    Includes the record-noise flip, so these are the true label
    probabilities."""
    surgery_type = np.asarray(surgery_type)
    hb_t = np.vectorize(hemoglobin_threshold)(surgery_type)
    plt_t = np.vectorize(platelet_threshold)(surgery_type)
    clinical = {
        "PRBC": _sigmoid((hb_t - np.asarray(hemoglobin)) / HB_SCALE),
        "Platelets": _sigmoid((plt_t - np.asarray(platelets)) / PLATELET_SCALE),
        "FFP": _sigmoid((np.asarray(inr) - INR_THRESHOLD) / INR_SCALE),
    }
    return {c: p * (1 - RECORD_NOISE) + (1 - p) * RECORD_NOISE for c, p in clinical.items()}


def sample_labels(hemoglobin, platelets, inr, surgery_type, rng: np.random.Generator) -> dict:
    """{component: 0/1 array} drawn from decision_probabilities."""
    probs = decision_probabilities(hemoglobin, platelets, inr, surgery_type)
    return {c: (rng.random(len(p)) < p).astype(int) for c, p in probs.items()}
