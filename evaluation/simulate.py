"""
HemoSmart - 90-Day Wastage & Shortage Simulation
----------------------------------------------------
Does HemoSmart actually reduce blood wastage without causing shortages?
This replays N days of hospital operation under different policies and
measures what happens to every unit.

THE WORLD (identical for every policy, per seed):
  - daily patient demand per blood group x component, sampled from the
    same process the forecaster was trained on (data/demand_process.py),
    including the post-monsoon dengue platelet spike;
  - routine donations arriving each day (Poisson around
    ROUTINE_SUPPLY_SHARE of average demand -- the same rate the Planner
    assumes);
  - the same starting stock (scripts/seed.py's inventory, dated to the
    start) and the same donor pool with the same hidden reply rates.

THE POLICIES (an ablation -- each adds one HemoSmart capability):
  baseline      Manual blood bank: issue own group first-in-first-out
                (other compatible groups only when own group is out);
                when a group/component turns Critical, phone 3 random
                eligible donors. No forecasting, no redistribution.
  allocation    + HemoSmart's wastage-aware allocation (near-expiry
                compatible units first) -- donors still reactive/random.
  agents        + the autonomous loop: forecast -> gap projection ->
                outreach sized to the gap, Thompson-Sampling donor
                choice with feedback. No redistribution.
  hemosmart     + Logistics agent offering near-expiry surplus to the
                partner hospital. The full system.

Everything runs through the real code (bloodbank/, agents/autonomy/,
agents/donor_module.py) against a throwaway SQLite database -- this
script never touches DATABASE_URL.

Run (from the repo root):
    python -m evaluation.simulate                       # 90 days, 3 seeds
    python -m evaluation.simulate --days 30 --seeds 1   # quick check

Writes evaluation/results/simulation.json and simulation.md.
"""

import argparse
import json
import os
import random
import sys
import tempfile
import time
from datetime import datetime, timedelta

# Always a throwaway database -- set before anything imports db.database.
_DB_DIR = tempfile.mkdtemp(prefix="hemosmart-sim-")
os.environ["DATABASE_URL"] = f"sqlite:///{os.path.join(_DB_DIR, 'sim.db')}"
os.environ.setdefault("JWT_SECRET_KEY", "simulation")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from agents.autonomy.orchestrator import run_cycle  # noqa: E402
from agents.donor_module import alert_donors, ensure_donor_pool, expire_unanswered_alerts, simulate_responses  # noqa: E402
from bloodbank.allocation import rank_units  # noqa: E402
from bloodbank.catalog import BLOOD_GROUPS, COMPONENTS, ROUTINE_SUPPLY_SHARE, compatible_donor_groups  # noqa: E402
from bloodbank.forecast import get_forecaster, load_demand_profile  # noqa: E402
from bloodbank.stock import available_units, receive_units, stock_summary, sweep_expired  # noqa: E402
from data.demand_process import sample_demand  # noqa: E402
from db.database import SessionLocal  # noqa: E402
from db.models import AgentCycle, BloodUnit, Donor, DonorAlert, Patient, Prediction, Transfusion  # noqa: E402
from integrations.mcp.mock_hospital import MockHospitalAdapter  # noqa: E402
from scripts.seed import _seed_inventory  # noqa: E402

POLICIES = ("baseline", "allocation", "agents", "hemosmart")
DEFAULT_START = datetime(2026, 8, 1, 8, 0)  # covers the dengue peak (mid-September)
BASELINE_ALERTS_PER_CRITICAL = 3
RESULTS_DIR = os.path.join(REPO_ROOT, "evaluation", "results")


class NoPartner(MockHospitalAdapter):
    """Redistribution switched off: every offer is declined."""
    def offer_surplus_units(self, units, now=None):
        return []


def random_selector(eligible, k):
    """Baseline donor choice: no learning, anyone eligible."""
    return random.sample(eligible, k)


def manual_rank(units, recipient, component, now):
    """Baseline issuing: own group oldest-first; other compatible groups
    (oldest-first) only when own group has nothing."""
    compatible = set(compatible_donor_groups(recipient, component))
    candidates = [u for u in units if u.blood_group in compatible and u.expires_at > now]
    return sorted(candidates, key=lambda u: (u.blood_group != recipient, u.expires_at))


# ----------------------------------------------------------------------
# World
# ----------------------------------------------------------------------
def build_world(start: datetime, days: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start.date(), periods=days, freq="D")
    demand = sample_demand(dates, rng)
    mean_daily = load_demand_profile()["mean_daily"]
    supply = {
        (g, c): rng.poisson(mean_daily[c].get(g, 0.0) * ROUTINE_SUPPLY_SHARE[c], days)
        for c in COMPONENTS for g in BLOOD_GROUPS
    }
    return {"demand": demand, "supply": supply}


def _wipe() -> None:
    with SessionLocal() as db:
        for model in (DonorAlert, BloodUnit, Transfusion, Prediction, Patient, AgentCycle, Donor):
            db.query(model).delete()
        db.commit()


# ----------------------------------------------------------------------
# One day of operation
# ----------------------------------------------------------------------
def _manual_morning(db, now, response_rng) -> None:
    sweep_expired(db, now)
    expire_unanswered_alerts(db, now)
    simulate_responses(db, now, response_rng)
    for cell in stock_summary(db, now)["low_stock"]:
        if cell["status"] == "Critical":
            alert_donors(db, cell["blood_group"], cell["component"], BASELINE_ALERTS_PER_CRITICAL,
                         now, selector=random_selector)
    db.commit()


def _serve_demand(db, now, day, world, ranker) -> dict:
    """Issues the day's demand; returns {(group, component): units unmet}."""
    unmet = {}
    for component in COMPONENTS:
        pool = available_units(db, now, component=component)
        recipients = sorted(BLOOD_GROUPS, key=lambda g: len(compatible_donor_groups(g, component)))
        for group in recipients:
            for _ in range(int(world["demand"][(group, component)][day])):
                ranked = ranker(pool, group, component, now)
                if not ranked:
                    unmet[(group, component)] = unmet.get((group, component), 0) + 1
                    continue
                unit = ranked[0]
                unit.status, unit.status_changed_at = "issued", now
                pool.remove(unit)
    db.commit()
    return unmet


def run_policy(policy: str, world: dict, start: datetime, days: int, seed: int) -> dict:
    _wipe()
    random.seed(seed)  # the bandit's Thompson sampler, and the baseline's random choice
    response_rng = random.Random(seed + 1)
    forecaster = get_forecaster()
    partner = MockHospitalAdapter() if policy == "hemosmart" else NoPartner()
    autonomous = policy in ("agents", "hemosmart")
    ranker = manual_rank if policy == "baseline" else rank_units

    with SessionLocal() as db:
        ensure_donor_pool(db, today=start)
        db.commit()
    _seed_inventory(now=start, rng=random.Random(2026))

    daily = []
    shortage_units = {c: 0 for c in COMPONENTS}
    cycle_seconds = []
    for day in range(days):
        now = start + timedelta(days=day)

        if autonomous:
            t = time.time()
            run_cycle(now=now, mode="simulation", trigger="simulation", forecaster=forecaster,
                      partner=partner, simulate_responses=True, rng=response_rng)
            cycle_seconds.append(time.time() - t)
        else:
            with SessionLocal() as db:
                _manual_morning(db, now, response_rng)

        with SessionLocal() as db:
            for (group, component), arrivals in world["supply"].items():
                if arrivals[day]:
                    receive_units(db, group, component, int(arrivals[day]), now, source="routine")
            db.commit()
            unmet = _serve_demand(db, now + timedelta(hours=4), day, world, ranker)
        for (_, component), n in unmet.items():
            shortage_units[component] += n

        daily.append({"day": day, "date": str(now.date()), "unmet": sum(unmet.values())})

    end = start + timedelta(days=days)
    with SessionLocal() as db:
        sweep_expired(db, end)  # anything that expired on the last day
        db.commit()
        outcome = _outcomes(db, start, end)
        alerts = db.query(DonorAlert).count()
        donations = db.query(DonorAlert).filter(DonorAlert.responded.is_(True)).count()
        expired_by_day = _expired_by_day(db, start, days)

    for d in daily:
        d["expired"] = expired_by_day.get(d["date"], 0)

    total_demand = int(sum(arr.sum() for arr in world["demand"].values()))
    expired = sum(outcome["expired"].values())
    issued = sum(outcome["issued"].values())
    transferred = sum(outcome["transferred"].values())
    return {
        "policy": policy,
        "seed": seed,
        "demand_units": total_demand,
        "issued": issued,
        "shortage_units": sum(shortage_units.values()),
        "shortage_by_component": shortage_units,
        "service_level": round(issued / total_demand, 4) if total_demand else None,
        "expired": expired,
        "expired_by_component": outcome["expired"],
        "transferred": transferred,
        "not_used_here": expired + transferred,
        "wastage_rate": round(expired / (issued + expired + transferred), 4),
        "donor_alerts": alerts,
        "donations": donations,
        "alerts_per_donation": round(alerts / donations, 2) if donations else None,
        "mean_cycle_seconds": round(sum(cycle_seconds) / len(cycle_seconds), 2) if cycle_seconds else None,
        "daily": daily,
    }


def _outcomes(db, start, end) -> dict:
    out = {s: {c: 0 for c in COMPONENTS} for s in ("issued", "expired", "transferred")}
    rows = (db.query(BloodUnit.status, BloodUnit.component)
            .filter(BloodUnit.status.in_(tuple(out)), BloodUnit.status_changed_at >= start,
                    BloodUnit.status_changed_at <= end).all())
    for status, component in rows:
        out[status][component] += 1
    return out


def _expired_by_day(db, start, days) -> dict:
    counts = {}
    for (changed,) in db.query(BloodUnit.status_changed_at).filter(BloodUnit.status == "expired"):
        counts[str(changed.date())] = counts.get(str(changed.date()), 0) + 1
    return counts


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
METRICS = [
    ("expired", "Units expired (wastage)", "lower"),
    ("wastage_rate", "Wastage rate", "lower"),
    ("transferred", "Units redistributed", "info"),
    ("not_used_here", "Units not used here (expired + redistributed)", "lower"),
    ("shortage_units", "Units short (unmet demand)", "lower"),
    ("service_level", "Service level (demand met)", "higher"),
    ("donor_alerts", "Donor alerts sent", "info"),
    ("donations", "Donations received", "info"),
    ("alerts_per_donation", "Alerts per donation", "lower"),
]


def summarize(runs: list) -> dict:
    summary = {}
    for policy in POLICIES:
        rows = [r for r in runs if r["policy"] == policy]
        if not rows:
            continue
        summary[policy] = {}
        for key, _, _ in METRICS:
            vals = [r[key] for r in rows if r[key] is not None]
            summary[policy][key] = {
                "mean": round(float(np.mean(vals)), 4) if vals else None,
                "std": round(float(np.std(vals)), 4) if len(vals) > 1 else 0.0,
            }
        for comp_key in ("expired_by_component", "shortage_by_component"):
            summary[policy][comp_key] = {
                c: round(float(np.mean([r[comp_key][c] for r in rows])), 1) for c in COMPONENTS
            }
    return summary


def _fmt(cell: dict, key: str) -> str:
    if cell["mean"] is None:
        return "—"
    if key in ("wastage_rate", "service_level"):
        text = f"{cell['mean'] * 100:.1f}%"
        return text + (f" ± {cell['std'] * 100:.1f}" if cell["std"] else "")
    if key == "alerts_per_donation":
        return f"{cell['mean']:.2f}" + (f" ± {cell['std']:.2f}" if cell["std"] else "")
    return f"{cell['mean']:.1f}" + (f" ± {cell['std']:.1f}" if cell["std"] else "")


def to_markdown(summary: dict, meta: dict) -> str:
    policies = [p for p in POLICIES if p in summary]
    lines = [
        "# HemoSmart — Wastage & Shortage Simulation",
        "",
        f"{meta['days']} days from {meta['start']}, {len(meta['seeds'])} seed(s) {meta['seeds']}; "
        f"mean ± std across seeds. Generated by `python -m evaluation.simulate` "
        f"on {meta['generated_at']}.",
        "",
        "| Metric | " + " | ".join(policies) + " |",
        "|---|" + "---|" * len(policies),
    ]
    for key, label, _ in METRICS:
        lines.append(f"| {label} | " + " | ".join(_fmt(summary[p][key], key) for p in policies) + " |")
    lines += ["", "**Expired units by component (mean)**", "",
              "| Component | " + " | ".join(policies) + " |", "|---|" + "---|" * len(policies)]
    for c in COMPONENTS:
        lines.append(f"| {c} | " + " | ".join(str(summary[p]["expired_by_component"][c]) for p in policies) + " |")
    lines += ["", "**Units short by component (mean)**", "",
              "| Component | " + " | ".join(policies) + " |", "|---|" + "---|" * len(policies)]
    for c in COMPONENTS:
        lines.append(f"| {c} | " + " | ".join(str(summary[p]["shortage_by_component"][c]) for p in policies) + " |")
    lines += [
        "",
        "Policies: **baseline** = manual (own-group FIFO, reactive random donor calls); "
        "**allocation** = + wastage-aware allocation; **agents** = + autonomous forecast/gap/bandit "
        "loop; **hemosmart** = + redistribution to the partner hospital. "
        "Redistributed units are reported separately, not counted as used.",
        "",
        "Simulated, not measured: donor replies (hidden per-donor rates), the partner's acceptance, "
        "and routine supply (Poisson around the Planner's assumed rate). See docs/DESIGN_DECISIONS.md.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--start", default=DEFAULT_START.strftime("%Y-%m-%d"))
    parser.add_argument("--policies", default=",".join(POLICIES))
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(hour=8)
    seeds = list(range(1, args.seeds + 1))
    policies = [p for p in args.policies.split(",") if p in POLICIES]
    runs = []
    for seed in seeds:
        world = build_world(start, args.days, seed)
        for policy in policies:
            t = time.time()
            result = run_policy(policy, world, start, args.days, seed)
            runs.append(result)
            print(f"seed {seed} {policy:<10} expired {result['expired']:>4}  short {result['shortage_units']:>4}  "
                  f"transferred {result['transferred']:>4}  alerts {result['donor_alerts']:>4}  "
                  f"({time.time() - t:.0f}s)", flush=True)

    meta = {"days": args.days, "start": str(start.date()), "seeds": seeds,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M")}
    summary = summarize(runs)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "simulation.json"), "w") as f:
        json.dump({"meta": meta, "summary": summary, "runs": runs}, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "simulation.md"), "w") as f:
        f.write(to_markdown(summary, meta))
    print(f"\nWrote {RESULTS_DIR}/simulation.md and simulation.json")


if __name__ == "__main__":
    main()
