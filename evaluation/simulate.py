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
    assumes -- times the April-May dip in data/demand_process.supply_factor,
    which the Planner doesn't know about);
  - the same starting stock (scripts/seed.py's inventory, dated to the
    start) and the same donor pool with the same hidden reply rates;
  - the realistic donor model (D85): replies fall with recent contact
    (donor_module.DONOR_FATIGUE), and new donors with no history join on
    the same days for every policy (NEW_DONORS_PER_YEAR).

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

Also scores the Planner as a shortage detector (autonomous policies):
did each real shortage (group x component x day with unmet demand)
appear in a cycle's projected shortages 1+ / 2+ days earlier, and how
many projected shortages really happened within the horizon.

Writes evaluation/results/simulation.json and simulation.md.

    python -m evaluation.simulate --planning-sweep      # mean vs upper-quantile planning

runs the full HemoSmart policy with the Planner projecting the mean or an
upper quantile of demand (bloodbank/projection.quantile_demand) and
writes evaluation/results/planning_quantile.md and .json.

    python -m evaluation.simulate --cooldown-sweep      # donor contact cooldown

runs the full HemoSmart policy with catalog.CONTACT_COOLDOWN_DAYS at each
of COOLDOWNS and writes evaluation/results/cooldown_sweep.md and .json.
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

from agents.autonomy.agents import PlannerAgent  # noqa: E402
from agents.autonomy.orchestrator import run_cycle  # noqa: E402
from agents.donor_module import (NEW_DONORS_PER_YEAR, add_donors, alert_donors, arriving_donors,  # noqa: E402
                                 ensure_donor_pool, expire_unanswered_alerts, simulate_responses)
from bloodbank.allocation import rank_units  # noqa: E402
from bloodbank.catalog import BLOOD_GROUPS, COMPONENTS, ROUTINE_SUPPLY_SHARE, compatible_donor_groups  # noqa: E402
from bloodbank.forecast import get_forecaster, load_demand_profile  # noqa: E402
from bloodbank.stock import available_units, receive_units, stock_summary, sweep_expired  # noqa: E402
from data.demand_process import sample_demand, supply_factor  # noqa: E402
from db.database import SessionLocal  # noqa: E402
from db.models import AgentCycle, BloodUnit, Donor, DonorAlert, Patient, Prediction, Transfusion  # noqa: E402
from integrations.mcp.mock_hospital import MockHospitalAdapter  # noqa: E402
from scripts.seed import _seed_inventory  # noqa: E402

POLICIES = ("baseline", "allocation", "agents", "hemosmart")
DEFAULT_START = datetime(2026, 8, 1, 8, 0)  # covers the dengue peak (mid-September)
BASELINE_ALERTS_PER_CRITICAL = 3
RESULTS_DIR = os.path.join(REPO_ROOT, "evaluation", "results")
HORIZON_DAYS = 7  # the Planner's projection horizon (run_cycle's default)
PLANNING_QUANTILES = (None, 0.7, 0.8, 0.9)  # None = mean (= the 50th percentile under the approximation)
# A level replaces the mean only if units short fall by more than the mean's seed-to-seed std, and units
# not used here (expired + redistributed), alerts sent and alerts per donation each rise by at most this.
MAX_COST_RISE = 0.10
COOLDOWNS = (0, 3, 7, 14)
D45_SHORTAGE_WEIGHT = 3  # evaluation/sensitivity.SHORTAGE_WEIGHT, shown for comparison only


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
    seasonal = supply_factor(dates)
    supply = {
        (g, c): rng.poisson(mean_daily[c].get(g, 0.0) * ROUTINE_SUPPLY_SHARE[c] * seasonal)
        for c in COMPONENTS for g in BLOOD_GROUPS
    }
    arrivals = np.random.default_rng(seed + 500).poisson(NEW_DONORS_PER_YEAR / 365, days)
    return {"demand": demand, "supply": supply, "arrivals": arrivals}


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
    warnings, unmet_cells = [], set()
    for day in range(days):
        now = start + timedelta(days=day)
        if world["arrivals"][day]:
            with SessionLocal() as db:
                add_donors(db, arriving_donors(int(world["arrivals"][day]), seed * 1000 + day, now, f"N{day:03d}"))
                db.commit()

        if autonomous:
            t = time.time()
            cycle = run_cycle(now=now, mode="simulation", trigger="simulation", horizon_days=HORIZON_DAYS,
                              forecaster=forecaster, partner=partner, simulate_responses=True, rng=response_rng)
            cycle_seconds.append(time.time() - t)
            warnings += projected_shortages(cycle["events"], day, now)
        else:
            with SessionLocal() as db:
                _manual_morning(db, now, response_rng)

        with SessionLocal() as db:
            for (group, component), arrivals in world["supply"].items():
                if arrivals[day]:
                    receive_units(db, group, component, int(arrivals[day]), now, source="routine")
            db.commit()
            unmet = _serve_demand(db, now + timedelta(hours=4), day, world, ranker)
        for (group, component), n in unmet.items():
            shortage_units[component] += n
            unmet_cells.add((group, component, day))

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
        # Same definition as wastage_rate, per component: expired / every unit that left stock here
        "wastage_rate_by_component": {
            c: round(outcome["expired"][c] / handled, 4) if (handled := sum(outcome[k][c] for k in outcome)) else 0.0
            for c in COMPONENTS},
        "transferred": transferred,
        "not_used_here": expired + transferred,
        "wastage_rate": round(expired / (issued + expired + transferred), 4),
        "donor_alerts": alerts,
        "donations": donations,
        "alerts_per_donation": round(alerts / donations, 2) if donations else None,
        "mean_cycle_seconds": round(sum(cycle_seconds) / len(cycle_seconds), 2) if cycle_seconds else None,
        "shortage_detection": shortage_detection(warnings, unmet_cells, days) if autonomous else None,
        "daily": daily,
    }


def projected_shortages(events: list, day: int, now: datetime) -> list:
    """The Planner's full-horizon projected shortages in one cycle's event
    log, as (cycle day, group, component, first shortage day)."""
    out = []
    for e in events:
        data = e.get("data", {})
        # Shortage events carry first_shortage_date; outreach needs carry it too, plus window_days.
        if e["agent"] == "Planner" and e["kind"] == "decide" and "first_shortage_date" in data \
                and "window_days" not in data:
            first = datetime.strptime(data["first_shortage_date"], "%Y-%m-%d").date()
            out.append((day, data["blood_group"], data["component"], day + (first - now.date()).days))
    return out


def shortage_detection(warnings: list, unmet_cells: set, days: int, horizon: int = HORIZON_DAYS) -> dict:
    """
    warnings: (cycle day t, group, component, first shortage day f); a
    warning covers day D when f <= D <= t + horizon - 1.
    unmet_cells: {(group, component, day)} where real demand went unmet.
    Lead time of a real shortage = days between the EARLIEST cycle whose
    warning covered it and the day itself (0 = that morning's cycle).
    Precision: share of warnings (first day inside the run) followed by a
    real shortage of that group/component within the warning's window.
    """
    leads = []
    for g, c, d in unmet_cells:
        covering = [d - t for t, wg, wc, f in warnings if (wg, wc) == (g, c) and t <= d and f <= d < t + horizon]
        leads.append(max(covering) if covering else None)
    scored = [w for w in warnings if w[3] < days]
    hits = sum(any((g, c, d) in unmet_cells for d in range(f, min(t + horizon, days))) for t, g, c, f in scored)
    detected = [lead for lead in leads if lead is not None]
    return {
        "shortages": len(unmet_cells),
        "warnings": len(scored),
        "recall_0d": round(len(detected) / len(leads), 4) if leads else None,
        "recall_1d": round(sum(lead >= 1 for lead in detected) / len(leads), 4) if leads else None,
        "recall_2d": round(sum(lead >= 2 for lead in detected) / len(leads), 4) if leads else None,
        "precision": round(hits / len(scored), 4) if scored else None,
        "mean_lead_days": round(float(np.mean(detected)), 2) if detected else None,
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
        summary[policy]["wastage_rate_by_component"] = {
            c: round(float(np.mean([r["wastage_rate_by_component"][c] for r in rows])), 4) for c in COMPONENTS}
        if rows[0]["shortage_detection"]:
            detection = [r["shortage_detection"] for r in rows]
            summary[policy]["shortage_detection"] = {}
            for key in detection[0]:
                vals = [d[key] for d in detection if d[key] is not None]
                summary[policy]["shortage_detection"][key] = {
                    "mean": round(float(np.mean(vals)), 4) if vals else None,
                    "std": round(float(np.std(vals)), 4) if len(vals) > 1 else 0.0,
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
    lines += ["", "**Wastage rate by component (expired / units issued, expired or redistributed; mean)**", "",
              "| Component | " + " | ".join(policies) + " |", "|---|" + "---|" * len(policies)]
    for c in COMPONENTS:
        lines.append(f"| {c} | " + " | ".join(
            f"{summary[p]['wastage_rate_by_component'][c] * 100:.1f}%" for p in policies) + " |")
    lines += ["", "**Units short by component (mean)**", "",
              "| Component | " + " | ".join(policies) + " |", "|---|" + "---|" * len(policies)]
    for c in COMPONENTS:
        lines.append(f"| {c} | " + " | ".join(str(summary[p]["shortage_by_component"][c]) for p in policies) + " |")
    detecting = [p for p in policies if "shortage_detection" in summary[p]]
    if detecting:
        lines += ["", "**Shortage detection by the Planner (autonomous policies)**", "",
                  "| Metric | " + " | ".join(detecting) + " |", "|---|" + "---|" * len(detecting)]
        for key, label in (("shortages", "Real shortages (group × component × day)"),
                           ("warnings", "Projected shortages (one per cycle per group/component)"),
                           ("recall_0d", "Recall, projected at all (incl. the same morning)"),
                           ("recall_1d", "Recall, projected ≥ 1 day earlier"),
                           ("recall_2d", "Recall, projected ≥ 2 days earlier"),
                           ("precision", f"Precision (happened within the {HORIZON_DAYS}-day window)"),
                           ("mean_lead_days", "Mean lead time, days (projected shortages only)")):
            cells = [summary[p]["shortage_detection"][key] for p in detecting]
            lines.append(f"| {label} | " + " | ".join(
                _fmt(c, "service_level" if key.startswith(("recall", "precision")) else "alerts_per_donation"
                     if key == "mean_lead_days" else key) for c in cells) + " |")
        lines += ["",
                  "A real shortage counts as projected L days earlier if the cycle L or more days before it "
                  "listed that group/component as short with a first shortage day on or before it, within the "
                  f"{HORIZON_DAYS}-day horizon. Lead 0 is the same morning's cycle, a few hours before demand. "
                  "Precision understates the Planner: every projected shortage triggers donor outreach, so a "
                  "warning that worked (donors came, the shortage never happened) counts here as a false alarm. "
                  "The Planner projects expected (mean) demand, so a shortage caused by a one-day demand spike "
                  "above the forecast is not projected at all; planning on an upper quantile catches more of them "
                  "at a cost in wastage (`--planning-sweep`, docs/DESIGN_DECISIONS.md D81)."]
    lines += [
        "",
        "Policies: **baseline** = manual (own-group FIFO, reactive random donor calls); "
        "**allocation** = + wastage-aware allocation; **agents** = + autonomous forecast/gap/bandit "
        "loop; **hemosmart** = + redistribution to the partner hospital. "
        "Redistributed units are reported separately, not counted as used.",
        "",
        "Simulated, not measured: donor replies (hidden per-donor rates, mean ~23%), the partner's acceptance, "
        "and routine supply (Poisson around the Planner's assumed rate, with an April-May dip that this "
        "August-October window doesn't reach). See docs/DESIGN_DECISIONS.md D83.",
    ]
    return "\n".join(lines) + "\n"


def choose_planning_level(rows: list, max_rise: float = MAX_COST_RISE):
    """rows: [{"quantile", "summary"}] with the mean (quantile None) among them.
    Returns the quantile with the fewest units short that passes the rule
    above, or None (keep the mean)."""
    base = next(r["summary"] for r in rows if r["quantile"] is None)
    m = lambda s, k: s[k]["mean"]  # noqa: E731

    def passes(s):
        return (m(s, "shortage_units") < m(base, "shortage_units") - base["shortage_units"]["std"]
                and all(m(s, k) <= m(base, k) * (1 + max_rise)
                        for k in ("not_used_here", "donor_alerts", "alerts_per_donation")))
    ok = [r for r in rows if r["quantile"] is not None and passes(r["summary"])]
    return min(ok, key=lambda r: m(r["summary"], "shortage_units"))["quantile"] if ok else None


def _level(q) -> str:
    return "mean" if q is None else f"{q * 100:.0f}th percentile"


def planning_sweep(start: datetime, days: int, seeds: list) -> dict:
    worlds = {s: build_world(start, days, s) for s in seeds}
    original = PlannerAgent.DEMAND_QUANTILE
    rows = []
    try:
        for q in PLANNING_QUANTILES:
            PlannerAgent.DEMAND_QUANTILE = q
            runs = [run_policy("hemosmart", worlds[s], start, days, s) for s in seeds]
            rows.append({"quantile": q, "summary": summarize(runs)["hemosmart"]})
            print(f"{_level(q):<16} short {rows[-1]['summary']['shortage_units']['mean']:.1f}  "
                  f"not used here {rows[-1]['summary']['not_used_here']['mean']:.1f}", flush=True)
    finally:
        PlannerAgent.DEMAND_QUANTILE = original
    return {"rows": rows, "chosen": choose_planning_level(rows), "current_default": original}


def choose_cooldown(rows: list) -> int:
    """The longest cooldown whose units short don't exceed no cooldown's (D85)."""
    none = next(r["summary"] for r in rows if r["cooldown"] == 0)["shortage_units"]["mean"]
    return max(r["cooldown"] for r in rows if r["summary"]["shortage_units"]["mean"] <= none)


def cooldown_sweep(start: datetime, days: int, seeds: list) -> dict:
    from bloodbank import catalog
    worlds = {s: build_world(start, days, s) for s in seeds}
    original = catalog.CONTACT_COOLDOWN_DAYS
    rows = []
    try:
        for k in COOLDOWNS:
            catalog.CONTACT_COOLDOWN_DAYS = k
            runs = [run_policy("hemosmart", worlds[s], start, days, s) for s in seeds]
            rows.append({"cooldown": k, "summary": summarize(runs)["hemosmart"]})
            print(f"cooldown {k:>2}  short {rows[-1]['summary']['shortage_units']['mean']:.1f}  "
                  f"alerts {rows[-1]['summary']['donor_alerts']['mean']:.0f}", flush=True)
    finally:
        catalog.CONTACT_COOLDOWN_DAYS = original
    return {"rows": rows, "chosen": choose_cooldown(rows), "current_default": original}


def cooldown_markdown(result: dict, meta: dict) -> str:
    cols = [("shortage_units", "Units short"), ("service_level", "Service level"), ("expired", "Expired"),
            ("transferred", "Redistributed"), ("donor_alerts", "Alerts sent"), ("donations", "Donations"),
            ("alerts_per_donation", "Alerts / donation")]
    lines = [
        "# HemoSmart — Donor Contact Cooldown",
        "",
        f"Full HemoSmart policy with the realistic donor model (fatigue + new donors, D85), {meta['days']} days "
        f"from {meta['start']}, {len(meta['seeds'])} seeds {meta['seeds']}, mean ± std; the same worlds for every "
        f"row. Generated by `python -m evaluation.simulate --cooldown-sweep` on {meta['generated_at']}.",
        "",
        "| Cooldown (days) | " + " | ".join(label for _, label in cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for r in result["rows"]:
        lines.append(f"| {r['cooldown']} | " + " | ".join(_fmt(r["summary"][k], k) for k, _ in cols) + " |")
    lines += [
        "",
        "**Rule:** the longest cooldown whose units short don't exceed no cooldown's (a shortage harms a "
        "patient; among those, a longer cooldown means fewer alerts to tired donors). "
        f"**Result:** {result['chosen']} days. Production setting (`bloodbank/catalog.CONTACT_COOLDOWN_DAYS`) at "
        f"run time: {result['current_default']} days.",
        "",
        "The rule was written after a first run showed the 14-day cooldown (chosen in the bandit evaluation, "
        "where alerts per day are fixed) raising units short here; it is stated so the choice can be checked.",
    ]
    return "\n".join(lines) + "\n"


def sweep_markdown(result: dict, meta: dict) -> str:
    cols = [("shortage_units", "Units short"), ("expired", "Expired"), ("transferred", "Redistributed"),
            ("not_used_here", "Not used here"), ("donor_alerts", "Alerts sent"),
            ("alerts_per_donation", "Alerts / donation")]
    detect = [("shortages", "Shortage events"), ("recall_0d", "Recall (any lead)"), ("recall_1d", "Recall ≥ 1 d"), ("recall_2d", "Recall ≥ 2 d"),
              ("precision", "Precision"), ("mean_lead_days", "Mean lead (d)")]
    lines = [
        "# HemoSmart — Planning on Mean vs Upper-Quantile Demand",
        "",
        f"Full HemoSmart policy, {meta['days']} days from {meta['start']}, {len(meta['seeds'])} seeds "
        f"{meta['seeds']}, mean ± std across seeds; the same simulated worlds for every level. Generated by "
        f"`python -m evaluation.simulate --planning-sweep` on {meta['generated_at']}.",
        "",
        "The Planner's gap projection serves either the forecast mean or the q-th percentile of cumulative "
        "demand per blood group, C(d) = M(d) + z·√M(d) (a normal approximation to Poisson counts; "
        "`bloodbank/projection.quantile_demand`). Prophet's 80% band isn't used because it is per component "
        "and can't be split by group share. Safety stock stays on the mean.",
        "",
        "| Planning demand | " + " | ".join(label for _, label in cols) + f" | D45 cost ({D45_SHORTAGE_WEIGHT}×short + not used) | "
        + " | ".join(label for _, label in detect) + " |",
        "|---|" + "---|" * (len(cols + detect) + 1),
    ]
    for r in result["rows"]:
        s, d = r["summary"], r["summary"]["shortage_detection"]
        cost = D45_SHORTAGE_WEIGHT * s["shortage_units"]["mean"] + s["not_used_here"]["mean"]
        lines.append(f"| {_level(r['quantile'])} | "
                     + " | ".join(_fmt(s[k], k) for k, _ in cols) + f" | {cost:.1f} | "
                     + " | ".join(_fmt(d[k], {"mean_lead_days": "alerts_per_donation", "shortages": "shortages"}
                                       .get(k, "service_level")) for k, _ in detect) + " |")
    chosen = result["chosen"]
    lines += [
        "",
        f"**Rule:** a quantile replaces the mean only if units short fall by more than the mean's "
        f"seed-to-seed std, while units not used here (expired + redistributed), alerts sent and alerts per "
        f"donation each rise by at most {MAX_COST_RISE:.0%}. Among levels that pass, the fewest units short wins.",
        "",
        f"**Result:** {'no level passes; the Planner keeps planning on the mean' if chosen is None else 'the ' + _level(chosen) + ' passes and is the best that does'}. "
        f"Default in `bloodbank/catalog.PLANNING_DEMAND_QUANTILE` at run time: {_level(result['current_default'])}.",
        "",
        f"The D45 cost column is the rule `evaluation/sensitivity.py` used to pick the outreach window "
        f"({D45_SHORTAGE_WEIGHT}× units short + units not used here). It is shown, not used, here: by it an upper "
        "quantile can look better, because it prices a unit short at three wasted donations. Which rule is right "
        "is a clinical judgement, not something this simulation can settle.",
        "",
        "Detection recall is computed on the shortages that still happen, so with few left (an upper quantile "
        "prevents most) it is noisy and not comparable across rows; units short is the outcome to compare.",
        "",
        "Simulated demand is Poisson by construction (`data/demand_process.py`), so the Poisson approximation "
        "fits this world exactly; real demand is usually overdispersed, which would make these quantiles too low.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--start", default=DEFAULT_START.strftime("%Y-%m-%d"))
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--planning-sweep", action="store_true", help="mean vs upper-quantile planning")
    parser.add_argument("--cooldown-sweep", action="store_true", help="donor contact cooldown lengths")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(hour=8)
    seeds = list(range(1, args.seeds + 1))
    if args.planning_sweep:
        meta = {"days": args.days, "start": str(start.date()), "seeds": seeds,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M")}
        result = planning_sweep(start, args.days, seeds)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, "planning_quantile.json"), "w") as f:
            json.dump({"meta": meta, **result}, f, indent=2)
        with open(os.path.join(RESULTS_DIR, "planning_quantile.md"), "w") as f:
            f.write(sweep_markdown(result, meta))
        print(f"\nWrote {RESULTS_DIR}/planning_quantile.md and planning_quantile.json")
        return
    if args.cooldown_sweep:
        meta = {"days": args.days, "start": str(start.date()), "seeds": seeds,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M")}
        result = cooldown_sweep(start, args.days, seeds)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, "cooldown_sweep.json"), "w") as f:
            json.dump({"meta": meta, **result}, f, indent=2)
        with open(os.path.join(RESULTS_DIR, "cooldown_sweep.md"), "w") as f:
            f.write(cooldown_markdown(result, meta))
        print(f"\nWrote {RESULTS_DIR}/cooldown_sweep.md and cooldown_sweep.json")
        return
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
