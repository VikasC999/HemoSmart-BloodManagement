"""
HemoSmart - Donor Selection (Bandit) Evaluation
---------------------------------------------------
Does Thompson-Sampling donor selection get more donations per alert than
the alternatives, and how close is it to the best possible choice?

Each simulated day the system needs ALERTS_PER_DAY donors contacted. It
picks them from the currently eligible pool (a donor who donated waits
90 days), each contacted donor replies with their hidden true rate, and
the policy updates its counts -- the same feedback loop as
agents/donor_module.py. The donor pool, hidden rates and prior history
come from simulate_donors(); Thompson selection is the deployed
select_top_donors().

Policies:
  random         anyone eligible (what the baseline in simulate.py does)
  greedy         highest posterior mean, no exploration
  epsilon-greedy greedy, but a random donor 10% of the time
  thompson       HemoSmart (agents/donor_module.select_top_donors)
  oracle         knows every donor's true rate -- the reference for regret

Three scenarios:
  history     the pool's seeded prior alerts, as deployed
  cold start  no history -- where exploration matters most
  drift       as "history", but on day 180 a third of donors' true rates
              are redrawn (people move, change jobs, lose interest) --
              what a learner that stops exploring can't adapt to

Run (from the repo root):
    python -m evaluation.bandit_regret

Writes evaluation/results/bandit_regret.md and .json.
"""

import json
import os
import random
from datetime import datetime, timedelta

import numpy as np

from agents.donor_module import MIN_DONATION_INTERVAL_DAYS, posterior_mean, select_top_donors, simulate_donors

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "evaluation", "results")
DAYS = 365
ALERTS_PER_DAY = 5
SEEDS = range(1, 21)
EPSILON = 0.1
POLICIES = ("random", "greedy", "epsilon_greedy", "thompson", "oracle")
POLICY_LABELS = {
    "random": "Random", "greedy": "Greedy (posterior mean)", "epsilon_greedy": f"ε-greedy (ε={EPSILON})",
    "thompson": "Thompson sampling (HemoSmart)", "oracle": "Oracle (knows true rates)",
}
CHECKPOINTS = (30, 90, 180, 365)
SCENARIOS = {
    "history": "With the pool's prior alert history (as deployed)",
    "cold_start": "Cold start (no history)",
    "drift": "Drift: a third of donors' true rates change on day 180",
}
DRIFT_DAY, DRIFT_SHARE = 180, 1 / 3


def _pick(policy: str, eligible: list, k: int, rng: random.Random) -> list:
    if policy == "random":
        return rng.sample(eligible, k)
    if policy == "oracle":
        return sorted(eligible, key=lambda d: d["simulated_response_rate"], reverse=True)[:k]
    if policy == "thompson":
        return select_top_donors(eligible, k)
    # greedy / epsilon-greedy; random tie-break so untried donors aren't ordered by id
    ranked = sorted(eligible, key=lambda d: (posterior_mean(d), rng.random()), reverse=True)
    if policy == "greedy":
        return ranked[:k]
    chosen = []
    pool = list(ranked)
    for _ in range(k):
        d = rng.choice(pool) if rng.random() < EPSILON else pool[0]
        chosen.append(d)
        pool.remove(d)
    return chosen


def run_one(policy: str, seed: int, scenario: str) -> list:
    start = datetime(2026, 1, 1)
    donors = simulate_donors(n=400, seed=seed, today=start)
    drift_rng = random.Random(seed + 3000)
    if scenario == "cold_start":
        for d in donors:
            d["alerts_sent"] = d["alerts_responded"] = 0
    for d in donors:
        d["last"] = datetime.strptime(d["last_donation_date"], "%Y-%m-%d")

    random.seed(seed)                 # Thompson's sampler (global, as deployed)
    rng = random.Random(seed + 1000)  # other policies' randomness
    reply_rng = random.Random(seed + 2000)
    cumulative, total = [], 0
    for day in range(DAYS):
        today = start + timedelta(days=day)
        if scenario == "drift" and day == DRIFT_DAY:
            for d in drift_rng.sample(donors, int(len(donors) * DRIFT_SHARE)):
                d["simulated_response_rate"] = drift_rng.betavariate(2, 2)
        eligible = [d for d in donors if (today - d["last"]).days >= MIN_DONATION_INTERVAL_DAYS]
        for d in _pick(policy, eligible, min(ALERTS_PER_DAY, len(eligible)), rng):
            replied = reply_rng.random() < d["simulated_response_rate"]
            d["alerts_sent"] += 1
            if replied:
                d["alerts_responded"] += 1
                d["last"] = today
                total += 1
        cumulative.append(total)
    return cumulative


def run() -> dict:
    out = {}
    for start_kind in SCENARIOS:
        curves = {p: np.array([run_one(p, s, start_kind) for s in SEEDS]) for p in POLICIES}
        oracle = curves["oracle"]
        out[start_kind] = {}
        for p in POLICIES:
            c = curves[p]
            alerts = ALERTS_PER_DAY * DAYS
            out[start_kind][p] = {
                "donations": {"mean": round(float(c[:, -1].mean()), 1), "std": round(float(c[:, -1].std()), 1)},
                "response_rate": round(float(c[:, -1].mean() / alerts), 4),
                "alerts_per_donation": round(float(alerts / c[:, -1].mean()), 2),
                "regret_at": {d: round(float((oracle[:, d - 1] - c[:, d - 1]).mean()), 1) for d in CHECKPOINTS},
                "mean_curve": [round(float(v), 2) for v in c.mean(axis=0)],
            }
        print(start_kind, {p: out[start_kind][p]["donations"]["mean"] for p in POLICIES}, flush=True)
    return out


def to_markdown(result: dict, meta: dict) -> str:
    lines = [
        "# HemoSmart — Donor Selection (Bandit) Evaluation",
        "",
        f"400 simulated donors, {ALERTS_PER_DAY} alerts/day for {DAYS} days, {len(SEEDS)} seeds (mean ± std). "
        f"A donor who donates is ineligible for {MIN_DONATION_INTERVAL_DAYS} days. "
        f"Generated by `python -m evaluation.bandit_regret` on {meta['generated_at']}.",
        "",
    ]
    for kind, title in SCENARIOS.items():
        lines += [f"## {title}", "",
                  "| Policy | Donations | Response rate | Alerts per donation | "
                  + " | ".join(f"Regret @ day {d}" for d in CHECKPOINTS) + " |",
                  "|---|---|---|---|" + "---|" * len(CHECKPOINTS)]
        for p in POLICIES:
            r = result[kind][p]
            lines.append(
                f"| {POLICY_LABELS[p]} | {r['donations']['mean']} ± {r['donations']['std']} | "
                f"{r['response_rate'] * 100:.1f}% | {r['alerts_per_donation']} | "
                + " | ".join(str(r["regret_at"][d]) for d in CHECKPOINTS) + " |")
        lines.append("")
    lines += [
        "Regret = donations the oracle got minus donations the policy got, cumulative to that day "
        "(lower is better). The oracle ranks by true rate every day; with eligibility gaps that is a strong "
        "reference, not a proven optimum, so slightly negative regret is possible.",
        "",
        "Hidden response rates are drawn from Beta(2, 2) by `simulate_donors()`; real donor behaviour "
        "(time of day, distance, fatigue from repeated alerts) isn't modelled.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    result = run()
    meta = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "days": DAYS,
            "alerts_per_day": ALERTS_PER_DAY, "seeds": list(SEEDS)}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "bandit_regret.json"), "w") as f:
        json.dump({"meta": meta, **result}, f, indent=2)
    md = to_markdown(result, meta)
    with open(os.path.join(RESULTS_DIR, "bandit_regret.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
