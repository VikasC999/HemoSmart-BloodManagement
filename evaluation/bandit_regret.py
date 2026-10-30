"""
HemoSmart - Donor Selection (Bandit) Evaluation
---------------------------------------------------
Does Thompson-Sampling donor selection get more donations per alert than
the alternatives, and how close is it to the best possible choice?

Each simulated day the system needs ALERTS_PER_DAY donors contacted. It
picks them from the currently eligible pool (a donor who donated waits
90 days, 120 for women -- required_interval_days), each contacted donor replies with their hidden true rate, and
the policy updates its counts -- the same feedback loop as
agents/donor_module.py. The donor pool, hidden rates and prior history
come from simulate_donors(); Thompson selection is the deployed
select_top_donors().

Policies:
  random         anyone eligible (what the baseline in simulate.py does)
  greedy         highest posterior mean, no exploration
  epsilon-greedy greedy, but a random donor 10% of the time
  thompson       HemoSmart (agents/donor_module.select_top_donors)
  oracle         knows every donor's true rate (with fatigue, the
                 fatigued rate of the day) -- the reference for regret

Scenarios without fatigue (the D53 results):
  history     the pool's seeded prior alerts, as deployed
  cold start  no history -- where exploration matters most
  drift       as "history", but on day 180 a third of donors' true rates
              are redrawn (people move, change jobs, lose interest) --
              what a learner that stops exploring can't adapt to
The realistic donor model (D85), each also with a contact cooldown
(no donor alerted twice within COOLDOWN_DAYS) layered on every policy:
  fatigue           as "history", but each alert lowers that donor's
                    reply rate for a while (donor_module.DONOR_FATIGUE)
  fatigue + new     as "fatigue", plus new donors with no history joining
                    through the year (donor_module.NEW_DONORS_PER_YEAR)

Run (from the repo root):
    python -m evaluation.bandit_regret

Writes evaluation/results/bandit_regret.md and .json.
"""

import json
import os
import random
from datetime import datetime, timedelta

import numpy as np

from bloodbank import catalog
from agents.donor_module import (DONOR_FATIGUE, NEW_DONORS_PER_YEAR, RESPONSE_RATE_PRIOR, arriving_donors,
                                 fatigued_rate, posterior_mean, required_interval_days, select_top_donors,
                                 simulate_donors)

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
    "fatigue": "Realistic: contact fatigue",
    "fatigue_new": "Realistic: contact fatigue + new donors joining (decides the production policy)",
}
REALISTIC = ("fatigue", "fatigue_new")
DRIFT_DAY, DRIFT_SHARE = 180, 1 / 3
COOLDOWN_DAYS = catalog.CONTACT_COOLDOWN_DAYS  # the production cooldown, tried on the realistic scenarios


def variants(scenario: str) -> list:
    """(policy, cooldown days) pairs run in a scenario."""
    cooldowns = (0, COOLDOWN_DAYS) if scenario in REALISTIC else (0,)
    return [(p, c) for c in cooldowns for p in POLICIES]


def _key(policy: str, cooldown: int) -> str:
    return f"{policy}+cooldown{cooldown}" if cooldown else policy


def _pick(policy: str, eligible: list, k: int, rng: random.Random) -> list:
    if policy == "random":
        return rng.sample(eligible, k)
    if policy == "oracle":
        return sorted(eligible, key=lambda d: d["rate_today"], reverse=True)[:k]
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


def run_one(policy: str, seed: int, scenario: str, cooldown: int = 0) -> tuple:
    """Returns (cumulative donations per day, max alerts sent to one donor)."""
    start = datetime(2026, 1, 1)
    donors = simulate_donors(n=400, seed=seed, today=start)
    drift_rng = random.Random(seed + 3000)
    arrival_rng = np.random.default_rng(seed + 4000)
    fatigue = DONOR_FATIGUE if scenario in REALISTIC else False
    if scenario == "cold_start":
        for d in donors:
            d["alerts_sent"] = d["alerts_responded"] = 0

    def prepare(ds):
        for d in ds:
            d["last"] = datetime.strptime(d["last_donation_date"], "%Y-%m-%d")
            d["alert_days"], d["new_alerts"] = [], 0
    prepare(donors)

    random.seed(seed)                 # Thompson's sampler (global, as deployed)
    rng = random.Random(seed + 1000)  # other policies' randomness
    reply_rng = random.Random(seed + 2000)
    cumulative, total = [], 0
    for day in range(DAYS):
        today = start + timedelta(days=day)
        if scenario == "drift" and day == DRIFT_DAY:
            for d in drift_rng.sample(donors, int(len(donors) * DRIFT_SHARE)):
                d["simulated_response_rate"] = drift_rng.betavariate(*RESPONSE_RATE_PRIOR)
        if scenario == "fatigue_new":
            joined = arriving_donors(int(arrival_rng.poisson(NEW_DONORS_PER_YEAR / DAYS)), seed * 1000 + day,
                                     today, prefix=f"N{day:03d}")
            prepare(joined)
            donors += joined
        eligible = [d for d in donors
                    if (today - d["last"]).days >= required_interval_days("PRBC", None, d["sex"])
                    and not (cooldown and d["alert_days"] and day - d["alert_days"][-1] < cooldown)]
        for d in eligible:
            d["rate_today"] = fatigued_rate(d["simulated_response_rate"],
                                            [day - a for a in d["alert_days"]], fatigue)
        for d in _pick(policy, eligible, min(ALERTS_PER_DAY, len(eligible)), rng):
            replied = reply_rng.random() < d["rate_today"]
            d["alerts_sent"] += 1
            d["alert_days"].append(day)
            if replied:
                d["alerts_responded"] += 1
                d["last"] = today
                total += 1
        cumulative.append(total)
    return cumulative, max(len(d["alert_days"]) for d in donors)


def run() -> dict:
    out = {}
    for scenario in SCENARIOS:
        runs = {_key(p, c): [run_one(p, s, scenario, c) for s in SEEDS] for p, c in variants(scenario)}
        curves = {k: np.array([r[0] for r in rs]) for k, rs in runs.items()}
        oracle = curves["oracle"]
        out[scenario] = {}
        for k, c in curves.items():
            alerts = ALERTS_PER_DAY * DAYS
            out[scenario][k] = {
                "donations": {"mean": round(float(c[:, -1].mean()), 1), "std": round(float(c[:, -1].std()), 1)},
                "share_of_oracle": round(float(c[:, -1].mean() / oracle[:, -1].mean()), 4),
                "response_rate": round(float(c[:, -1].mean() / alerts), 4),
                "alerts_per_donation": round(float(alerts / c[:, -1].mean()), 2),
                "max_alerts_one_donor": round(float(np.mean([r[1] for r in runs[k]])), 1),
                "regret_at": {d: round(float((oracle[:, d - 1] - c[:, d - 1]).mean()), 1) for d in CHECKPOINTS},
                "mean_curve": [round(float(v), 2) for v in c.mean(axis=0)],
            }
        print(scenario, {k: v["donations"]["mean"] for k, v in out[scenario].items()}, flush=True)
    out["decision"] = choose_policy(out["fatigue_new"])
    return out


def choose_policy(rows: dict) -> dict:
    """The production rule (D85): among the learning policies, with or
    without the cooldown, the most donations wins (with ALERTS_PER_DAY
    fixed, that is also the fewest alerts per donation). Thompson, the
    incumbent, is kept unless the winner beats it by more than the
    seed-to-seed std of the winner's donations."""
    learners = {k: r for k, r in rows.items() if not k.startswith(("random", "oracle"))}
    best = max(learners, key=lambda k: learners[k]["donations"]["mean"])
    incumbent = max((k for k in learners if k.startswith("thompson")),
                    key=lambda k: learners[k]["donations"]["mean"])
    margin = learners[best]["donations"]["mean"] - learners[incumbent]["donations"]["mean"]
    keep = margin <= learners[best]["donations"]["std"]
    return {"best": best, "best_thompson": incumbent, "margin": round(margin, 1),
            "chosen": incumbent if keep else best}


def to_markdown(result: dict, meta: dict) -> str:
    lines = [
        "# HemoSmart — Donor Selection (Bandit) Evaluation",
        "",
        f"400 simulated donors, {ALERTS_PER_DAY} alerts/day for {DAYS} days, {len(SEEDS)} seeds (mean ± std). "
        "A donor who donates is ineligible for 90 days (120 for women, per NBTC 2017). "
        f"Generated by `python -m evaluation.bandit_regret` on {meta['generated_at']}.",
        "",
    ]
    for kind, title in SCENARIOS.items():
        lines += [f"## {title}", "",
                  "| Policy | Donations | % of oracle donations | Response rate | Alerts per donation | "
                  "Max alerts to one donor | " + " | ".join(f"Regret @ day {d}" for d in CHECKPOINTS) + " |",
                  "|---|---|---|---|---|---|" + "---|" * len(CHECKPOINTS)]
        for p, c in variants(kind):
            r = result[kind][_key(p, c)]
            label = POLICY_LABELS[p] + (f" + {c}-day cooldown" if c else "")
            lines.append(
                f"| {label} | {r['donations']['mean']} ± {r['donations']['std']} | "
                f"{r['share_of_oracle'] * 100:.1f}% | "
                f"{r['response_rate'] * 100:.1f}% | {r['alerts_per_donation']} | {r['max_alerts_one_donor']} | "
                + " | ".join(str(r["regret_at"][d]) for d in CHECKPOINTS) + " |")
        lines.append("")
    dec = result["decision"]
    lines += [
        "## Production policy",
        "",
        "Rule (D85): in the realistic scenario (fatigue + new donors), among the learning policies with or "
        "without the cooldown, the most donations wins; alerts per day are fixed, so that is also the fewest "
        "alerts per donation. Thompson sampling, the incumbent, is kept unless the winner beats its best "
        "variant by more than the winner's seed-to-seed std.",
        "",
        f"Best: **{dec['best']}**; best Thompson variant: **{dec['best_thompson']}**; margin {dec['margin']} "
        f"donations. Chosen: **{dec['chosen']}**.",
        "",
    ]
    lines += [
        "Regret = donations the oracle got minus donations the policy got, cumulative to that day "
        "(lower is better). The oracle ranks by true rate every day; with eligibility gaps that is a strong "
        "reference, not a proven optimum, so slightly negative regret is possible.",
        "",
        f"Hidden response rates are drawn from Beta{RESPONSE_RATE_PRIOR} (mean "
        f"{RESPONSE_RATE_PRIOR[0] / sum(RESPONSE_RATE_PRIOR):.0%}, anchored to the real UCI donors' 23.8%) by "
        "`simulate_donors()`, blood types by population share. Fatigue: each earlier alert multiplies the "
        f"reply rate by {DONOR_FATIGUE['decay_per_alert']}, its effect halving every "
        f"{DONOR_FATIGUE['half_life_days']} days; new donors: {NEW_DONORS_PER_YEAR}/year. Both are assumptions "
        "(no study of alert frequency vs response was found). Time of day, distance and donors leaving the "
        "registry aren't modelled.",
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
