"""
HemoSmart - Demand-Supply Gap Projection
--------------------------------------------
The report's central metric: the gap between forecast demand and what
will actually be usable. A raw "units in stock minus units forecast"
gets it wrong for perishable stock -- 10 platelet units in stock do not
cover 10 units of demand spread over 7 days if 6 of them expire on day 2.

So this replays the horizon day by day, per component, running the SAME
allocation policy the blood bank issues units with
(bloodbank/allocation.py): each recipient group's forecast demand is
served from compatible stock -- near-expiry units from any compatible
group first, then own group, FEFO -- and whatever expires unused is
counted as projected waste. Both failure modes come out of one pass:

  projected_shortage -- recipients' demand stock can't serve -> donor alerts
  projected_wastage  -- units that expire unused              -> redistribution

Simulating the real policy matters for wastage: an O- platelet patient
can receive A-/B-/AB- platelets, so rare-group units that would expire
in isolation are projected as used, not wasted -- and a rare group's
"shortage" that compatible stock covers doesn't send donors out for
units that would then expire on the shelf.

Each day, recipients with the fewest compatible donor groups are served
first (they have no fallback). Expected routine supply
(bloodbank.catalog.ROUTINE_SUPPLY_SHARE) arrives as fresh units; only
REAL units are ever reported as at risk, since only they can be moved.
Forecasts are fractional expected units; each group's fraction carries
over day to day, so 0.4 units/day consumes a unit every 2-3 days
instead of rounding to zero forever.

Optionally (demand_quantile) the replay serves an upper quantile of
demand instead of the mean, so shortages from busier-than-average days
are projected too -- see quantile_demand(). Safety stock and the
reported forecast_demand stay on the mean, so buffers aren't stacked.
"""

import math
from datetime import datetime, timedelta
from statistics import NormalDist
from types import SimpleNamespace

from bloodbank.allocation import rank_units
from bloodbank.catalog import BLOOD_GROUPS, COMPONENTS, compatible_donor_groups


def quantile_demand(daily: list, quantile: float = None) -> list:
    """
    Daily demand whose running total is the `quantile` of cumulative
    demand, by a normal approximation to Poisson demand:
        C(d) = M(d) + z * sqrt(M(d)),  M(d) = expected units through day d.
    Applied to the running total, not day by day: summing daily upper
    quantiles would plan for every day of the week being a bad day.
    Poisson because group-level demand is a count with no group-level
    Prophet band (the band is per component, and a band can't be split by
    group share); it ignores forecast error in the mean itself.
    None (or 0.5) returns the mean unchanged.
    """
    if quantile is None or quantile == 0.5:
        return list(daily)
    z = NormalDist().inv_cdf(quantile)
    out, previous, running = [], 0.0, 0.0
    for units in daily:
        running += units
        upper = max(running + z * math.sqrt(running), previous)  # z < 0 must not make it shrink
        out.append(upper - previous)
        previous = upper
    return out


def project_component(component: str, units: list, demand: dict, now: datetime,
                      inflow: dict = None, shelf_life_days: int = None, demand_quantile: float = None) -> dict:
    """
    units: available units of this component (with blood_group, expires_at)
    demand: {recipient group: [expected units per day]}, day 0 = next 24h
    inflow: {group: expected routine units/day}; shelf_life_days required with it
    demand_quantile: plan for this quantile of demand (None = the mean)
    Returns {group: projection}.
    """
    inflow = inflow or {}
    mean_demand = demand
    demand = {g: quantile_demand(d, demand_quantile) for g, d in demand.items()}
    horizon = max((len(v) for v in demand.values()), default=0)
    stock = [u for u in units if u.expires_at > now]
    own_supply = {g: sum(u.blood_group == g for u in stock) for g in BLOOD_GROUPS}
    recipients = sorted(BLOOD_GROUPS, key=lambda g: len(compatible_donor_groups(g, component)))

    carry = {g: 0.0 for g in BLOOD_GROUPS}
    inflow_carry = {g: 0.0 for g in BLOOD_GROUPS}
    shortage = {g: 0 for g in BLOOD_GROUPS}
    first_shortage = {g: None for g in BLOOD_GROUPS}
    wasted = {g: [] for g in BLOOD_GROUPS}
    daily_unmet = {g: [] for g in BLOOD_GROUPS}
    end_stock = {g: [] for g in BLOOD_GROUPS}  # own-group units left at each day's end

    for day in range(horizon):
        day_start = now + timedelta(days=day)
        day_end = day_start + timedelta(days=1)

        for g in BLOOD_GROUPS:
            inflow_carry[g] += inflow.get(g, 0.0)
            while inflow_carry[g] >= 1:
                stock.append(SimpleNamespace(
                    blood_group=g, component=component, virtual=True,
                    expires_at=day_start + timedelta(days=shelf_life_days),
                ))
                inflow_carry[g] -= 1

        for g in recipients:
            daily = demand.get(g, [])
            daily_unmet[g].append(0)
            carry[g] += daily[day] if day < len(daily) else 0.0
            while carry[g] >= 1:
                best = rank_units(stock, g, component, day_start)
                if not best:
                    break
                stock.remove(best[0])
                carry[g] -= 1
            if carry[g] >= 1:  # demand left over with no compatible stock
                unmet = math.floor(carry[g])
                shortage[g] += unmet
                daily_unmet[g][-1] = unmet
                carry[g] -= unmet
                if first_shortage[g] is None:
                    first_shortage[g] = day

        for u in [u for u in stock if u.expires_at <= day_end]:
            stock.remove(u)
            if not getattr(u, "virtual", False):
                wasted[u.blood_group].append(u)
        for g in BLOOD_GROUPS:
            end_stock[g].append(sum(u.blood_group == g for u in stock))

    out = {}
    for g in BLOOD_GROUPS:
        total = sum(mean_demand.get(g, []))
        avg = total / horizon if horizon else 0.0
        out[g] = {
            "supply_now": own_supply[g],
            "forecast_demand": round(total, 1),
            "planned_demand": round(sum(demand.get(g, [])), 1),
            "projected_shortage": shortage[g],
            "projected_wastage": len(wasted[g]),
            "at_risk_units": sorted(wasted[g], key=lambda u: u.expires_at),
            "first_shortage_day": first_shortage[g],
            "daily_unmet": daily_unmet[g],
            "end_stock": end_stock[g],
            "days_of_cover": round(own_supply[g] / avg, 1) if avg > 0 else None,
        }
    return out


def project(units, forecast: dict, now: datetime, inflow: dict = None, shelf_life: dict = None,
            demand_quantile: float = None) -> dict:
    """
    units: available units (objects with blood_group/component/expires_at)
    forecast: {(group, component): [daily expected units]}
    inflow: {(group, component): expected routine units/day}, optional
    shelf_life: {component: days}, required with inflow
    demand_quantile: plan for this quantile of demand (None = the mean)
    Returns {(group, component): projection} for every group of every
    component that appears in the forecast or in stock.
    """
    inflow = inflow or {}
    # A fixed order (not set iteration, which varies with Python's hash
    # seed) keeps downstream decisions -- and the simulation -- reproducible.
    present = {c for _, c in forecast} | {u.component for u in units}
    components = [c for c in COMPONENTS if c in present] + sorted(present - set(COMPONENTS))
    result = {}
    for c in components:
        per_group = project_component(
            c,
            [u for u in units if u.component == c],
            {g: d for (g, comp), d in forecast.items() if comp == c},
            now,
            inflow={g: v for (g, comp), v in inflow.items() if comp == c},
            shelf_life_days=shelf_life[c] if shelf_life else None,
            demand_quantile=demand_quantile,
        )
        for g, p in per_group.items():
            result[(g, c)] = p
    return result
