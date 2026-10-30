"""
HemoSmart - Donor Response on Real Data
-------------------------------------------
Everything else about donors is simulated. This checks the idea behind
the donor bandit against real donors: does a Beta posterior built from
a donor's history rank who will donate next?

Data: the public UCI "Blood Transfusion Service Center" dataset (Yeh et
al. 2009), 748 donors from a blood transfusion service center in
Hsin-Chu City, Taiwan. Per donor: Recency (months since last donation),
Frequency (total donations), Monetary (c.c. donated), Time (months since
first donation), and whether they donated in March 2007 -- the outcome.

Mapping to the bandit's Beta(1 + responses, 1 + alerts - responses):
  responses = Frequency
  alerts    = donation opportunities since the first donation, one per
              eligibility window: floor(Time / 3) + 1, with 3 months =
              agents/donor_module.MIN_DONATION_INTERVAL_DAYS. A donor who
              gave more often than that (shorter local intervals) gets
              alerts = Frequency, i.e. responded to every opportunity.
The dataset records donations, not alerts, so this is an assumption --
the closest defensible proxy, not the bandit's real input.

Compared, on the same 5 stratified folds (mean ± std):
  random                 the reference: expected AUROC 0.5, lift 1
  Beta posterior mean    the bandit's greedy score, no fitted parameters
  logistic regression    on Recency, Frequency, Time (Monetary is exactly
                         250 c.c. x Frequency, so it adds nothing)

Run (from the repo root):
    python -m evaluation.donor_real_data

Downloads the dataset to data/external/transfusion.data if missing.
Writes evaluation/results/donor_real_data.md and .json.
"""

import json
import os
import urllib.error
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from agents.donor_module import MIN_DONATION_INTERVAL_DAYS, RESPONSE_RATE_PRIOR

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "evaluation", "results")
DATA_PATH = os.path.join(REPO_ROOT, "data", "external", "transfusion.data")
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data"
COLUMNS = ["recency", "frequency", "monetary", "time", "donated"]
INTERVAL_MONTHS = round(MIN_DONATION_INTERVAL_DAYS / 30)
FEATURES = ["recency", "frequency", "time"]
FOLDS = 5
TOP_SHARE = 0.10  # top decile
SIMULATOR_MEAN_RATE = RESPONSE_RATE_PRIOR[0] / sum(RESPONSE_RATE_PRIOR)  # simulate_donors()
METHODS = ("random", "posterior", "logistic")
METHOD_LABELS = {
    "random": "Random order",
    "posterior": "Beta posterior mean (bandit idea)",
    "logistic": "Logistic regression (R, F, T)",
}


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            urllib.request.urlretrieve(DATA_URL, path)
        except (urllib.error.URLError, OSError) as exc:
            raise SystemExit(f"Could not download the UCI dataset ({exc}). Download {DATA_URL} "
                             f"by hand and save it as {path}.")
    df = pd.read_csv(path)
    df.columns = COLUMNS
    return df


def beta_history(df: pd.DataFrame) -> pd.DataFrame:
    """The mapping above, as the bandit's alerts_sent / alerts_responded."""
    alerts = np.maximum(df.frequency, df.time // INTERVAL_MONTHS + 1)
    return pd.DataFrame({"alerts_sent": alerts, "alerts_responded": df.frequency})


def precision_at(y, score, share: float = TOP_SHARE, rng: np.random.Generator = None) -> float:
    """Share of positives among the top `share` by score; ties broken at random."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    k = max(1, int(round(share * len(y))))
    tiebreak = (rng or np.random.default_rng(0)).random(len(y))
    top = np.lexsort((tiebreak, -score))[:k]
    return float(y[top].mean())


def run() -> dict:
    df = load_dataset()
    history = beta_history(df)
    # The deployed posterior_mean() on the mapped history
    posterior = (1 + history.alerts_responded) / (2 + history.alerts_sent)
    folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    rng = np.random.default_rng(42)

    rows = {m: [] for m in METHODS}
    for train, test in folds.split(df, df.donated):
        y = df.donated.values[test]
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(df.iloc[train][FEATURES], df.donated.values[train])
        scores = {
            "random": rng.random(len(test)),
            "posterior": posterior.values[test],
            "logistic": model.predict_proba(df.iloc[test][FEATURES])[:, 1],
        }
        for m, s in scores.items():
            p = precision_at(y, s, rng=rng)
            rows[m].append({"auroc": roc_auc_score(y, s), "precision_top_decile": p, "lift": p / y.mean()})

    summary = {m: {k: {"mean": round(float(np.mean([r[k] for r in rs])), 4),
                       "std": round(float(np.std([r[k] for r in rs])), 4)} for k in rs[0]}
               for m, rs in rows.items()}
    return {
        "summary": summary,
        "n": len(df),
        "response_rate": round(float(df.donated.mean()), 4),
        "mean_posterior": round(float(posterior.mean()), 4),
        "clamped": int((df.frequency > df.time // INTERVAL_MONTHS + 1).sum()),
    }


def to_markdown(result: dict, meta: dict) -> str:
    s = result["summary"]
    gap = SIMULATOR_MEAN_RATE - result["response_rate"]
    lines = [
        "# HemoSmart — Donor Response on Real Data",
        "",
        f"UCI Blood Transfusion Service Center dataset (Yeh et al. 2009), {result['n']} real donors. "
        f"Outcome: donated in March 2007. {FOLDS}-fold stratified cross-validation, mean ± std across folds. "
        f"Generated by `python -m evaluation.donor_real_data` on {meta['generated_at']}.",
        "",
        "## Ranking who donates next",
        "",
        f"| Method | AUROC | Precision, top {TOP_SHARE:.0%} | Lift over base rate |",
        "|---|---|---|---|",
    ]
    for m in METHODS:
        cell = s[m]
        lines.append(f"| {METHOD_LABELS[m]} | " + " | ".join(
            f"{cell[k]['mean']:.3f} ± {cell[k]['std']:.3f}" for k in ("auroc", "precision_top_decile", "lift")) + " |")
    lines += [
        "",
        f"Beta prior from history: responses = Frequency, alerts = floor(Time / {INTERVAL_MONTHS}) + 1 donation "
        f"opportunities (one per {MIN_DONATION_INTERVAL_DAYS}-day eligibility window), raised to Frequency for "
        f"the {result['clamped']} donors who gave more often than that. The dataset records donations, not "
        "alerts, so this mapping is an assumption. The posterior has no fitted parameters; the logistic "
        "regression is trained on each training fold. Monetary is exactly 250 c.c. × Frequency, so it is left out.",
        "",
        "The posterior ignores Recency (months since the last donation), which the logistic regression uses; "
        "how much that matters is the gap between the two rows.",
        "",
        "## Response rate: real vs simulated",
        "",
        f"Real donors: **{result['response_rate'] * 100:.1f}%** donated in the target month. The posterior "
        f"means from the mapped history average {result['mean_posterior'] * 100:.1f}%, about twice that: past "
        "giving overstates future response, mostly because many donors have lapsed (high Recency).",
        "",
        f"The simulator used to draw each donor's hidden reply rate from Beta(2, 2), mean 50%. Because of this "
        f"result it now draws from Beta{RESPONSE_RATE_PRIOR}, mean **{SIMULATOR_MEAN_RATE * 100:.1f}%** "
        f"({gap * 100:+.1f} points from the real rate; docs/DESIGN_DECISIONS.md D83). That match is by "
        "construction, not a validation, and a month's donation and a reply to one alert are not the same "
        "quantity: the simulated rate is an anchored assumption.",
        "",
        "The random row is one random ordering per fold, so it scatters around AUROC 0.5 and lift 1.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    result = run()
    meta = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "folds": FOLDS, "source": DATA_URL}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "donor_real_data.json"), "w") as f:
        json.dump({"meta": meta, **result}, f, indent=2)
    md = to_markdown(result, meta)
    with open(os.path.join(RESULTS_DIR, "donor_real_data.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
