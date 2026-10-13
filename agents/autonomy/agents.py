"""
The specialist agents of HemoSmart's autonomous loop.

Each agent owns one responsibility, reads what earlier agents posted on
the Blackboard, and posts its own results plus an event for every
observation (observe), decision (decide) and action (act). Decisions
are made in code -- deterministic, testable, and fast enough to replay
months of simulated operation. Language is used only where language is
the product: the Explanation agent grounds each action in the WHO
handbook via RAG and writes the cycle brief.

    Donor (feedback)  resolve donor replies -> bandit update + donated units
    Inventory         expiry sweep (wastage) + stock status
    Forecast          demand per group x component over the horizon
    Planner           day-by-day projection -> shortages + units at risk of expiring
    Donor (outreach)  bandit-selected alerts sized to each shortage
    Logistics         offer at-risk near-expiry units to a partner hospital
    Explanation       guideline citations per action + cycle brief
"""

import math
import random
from datetime import timedelta

from agents.donor_module import (
    alert_donors,
    eligible_candidates,
    expire_unanswered_alerts,
    pending_alerts,
    posterior_mean,
    simulate_responses,
)
from bloodbank.allocation import is_near_expiry
from bloodbank.catalog import BLOOD_GROUPS, COMPONENTS, OUTREACH_WINDOW_DAYS, SAFETY_STOCK_DAYS, SHELF_LIFE_DAYS
from bloodbank.forecast import load_demand_profile, routine_inflow
from bloodbank.projection import project
from bloodbank.stock import available_units, stock_summary, sweep_expired, transfer_units


def _need_kind(need: dict) -> str:
    """Why outreach fired, in words: unmet demand, a safety-stock top-up, or both."""
    unmet, safety = need.get("unmet", 0), need.get("safety_gap", 0)
    if unmet and safety:
        return f"{unmet} unmet + {safety} to restore safety stock"
    if unmet:
        return "unmet demand"
    return "safety-stock top-up, no unmet demand"

class DonorAgent:
    """Coordinates donors in two phases: feedback (start of cycle) and
    outreach (after the Planner has found shortages)."""
    name = "Donor"
    MAX_ALERTS_PER_SHORTAGE = 16  # per group/donation type per cycle; see evaluation/results/sensitivity.md
    MIN_ASSUMED_RESPONSE_RATE = 0.1

    def __init__(self, simulate_responses: bool, rng: random.Random):
        self.simulate = simulate_responses
        self.rng = rng

    def collect_feedback(self, bb, db) -> None:
        # Alerts past the response window are non-responses in either
        # mode -- a weeks-old alert must not turn into a donation today.
        outcomes = expire_unanswered_alerts(db, bb.now)
        if self.simulate:
            outcomes += simulate_responses(db, bb.now, self.rng, current_cycle_id=bb.cycle_id)
        bb.donor_feedback = outcomes
        if not outcomes:
            bb.log(self.name, "observe", "No donor alerts awaiting a reply.")
            return
        responded = [o for o in outcomes if o["responded"]]
        units = [u for o in responded for u in o["units_received"]]
        bb.log(
            self.name, "act",
            f"{len(responded)} of {len(outcomes)} alerted donors responded; "
            f"{len(units)} donated units entered stock. Bandit posteriors updated.",
            responded=[o["donor_id"] for o in responded],
            no_response=[o["donor_id"] for o in outcomes if not o["responded"]],
            units_received=units,
            source="simulated replies + expired alerts" if self.simulate else "expired alerts (no reply in 48h)",
        )

    @staticmethod
    def _donation_needs(shortages: list) -> list:
        """One whole-blood donation yields a PRBC AND an FFP unit, so a
        group short of both needs max(PRBC gap, FFP gap) whole-blood
        donors, not the sum. Platelets come from separate apheresis
        donations."""
        needs = {}
        for s in shortages:
            donation = "Platelets" if s["component"] == "Platelets" else "PRBC"
            key = (s["blood_group"], donation)
            need = needs.setdefault(key, {**s, "component": donation, "covers": []})
            need["covers"].append(s["component"])
            need["projected_shortage"] = max(need["projected_shortage"], s["projected_shortage"])
            need["first_shortage_date"] = min(need["first_shortage_date"], s["first_shortage_date"])
        return sorted(needs.values(), key=lambda n: (n["first_shortage_date"], -n["projected_shortage"]))

    def act(self, bb, db) -> None:
        if not bb.needs:
            bb.log(self.name, "observe", "No gap inside the outreach window -- no donor outreach needed.")
            return
        for s in self._donation_needs(bb.needs):
            group, component, gap = s["blood_group"], s["component"], s["projected_shortage"]
            label = f"{group} {'/'.join(s['covers'])}"
            pending = len(pending_alerts(db, group, component))
            candidates = eligible_candidates(db, group, component, bb.now)

            if not candidates and pending == 0:
                bb.escalations.append({**s, "uncovered": gap})
                bb.log(self.name, "decide",
                       f"{label}: outreach need of {gap} unit(s) but no eligible donors -- "
                       f"escalating to the regional blood bank.", **s)
                continue

            # Size the outreach to the gap: with a ~40% expected reply
            # rate, a 2-unit gap needs ~5 alerts, not 3 and not 50.
            rate = (sum(posterior_mean(d) for d in candidates) / len(candidates)) if candidates else 0.5
            needed = math.ceil(gap / max(rate, self.MIN_ASSUMED_RESPONSE_RATE))
            to_send = min(self.MAX_ALERTS_PER_SHORTAGE, needed - pending)
            if to_send <= 0:
                bb.log(self.name, "decide",
                       f"{label}: {pending} alert(s) already pending cover the "
                       f"{gap}-unit gap -- not re-alerting.", **s)
                continue

            result = alert_donors(db, group, component, to_send, bb.now, cycle_id=bb.cycle_id)
            sent = len(result["alerted_donors"])
            bb.alerts.append({**s, "alerted": result["alerted_donors"]})
            bb.log(
                self.name, "act",
                f"{label}: outreach need of {gap} unit(s) ({_need_kind(s)}; stock first runs low "
                f"{s['first_shortage_date']}) -- "
                f"alerted {sent} donor(s) for {'apheresis' if component == 'Platelets' else 'whole blood'} "
                f"(Thompson sampling over {result['eligible_count']} eligible; expected reply rate {rate:.0%}).",
                donors=[a["donor_id"] for a in result["alerted_donors"]],
            )
            expected_units = (sent + pending) * rate
            if expected_units < gap:
                uncovered = math.ceil(gap - expected_units)
                bb.escalations.append({**s, "uncovered": uncovered})
                bb.log(self.name, "decide",
                       f"{label}: expected donor yield (~{expected_units:.1f}) won't close the gap -- "
                       f"escalating ~{uncovered} unit(s) to the regional blood bank.", **s)


class InventoryAgent:
    name = "Inventory"

    def run(self, bb, db) -> None:
        expired = sweep_expired(db, bb.now)
        bb.expired_units = [
            {"unit_id": u.unit_id, "blood_group": u.blood_group, "component": u.component,
             "expires_at": u.expires_at}
            for u in expired
        ]
        if expired:
            by_key = {}
            for u in expired:
                by_key[f"{u.blood_group} {u.component}"] = by_key.get(f"{u.blood_group} {u.component}", 0) + 1
            bb.log(self.name, "act", f"{len(expired)} unit(s) passed expiry and were recorded as wastage.",
                   by_group_component=by_key, unit_ids=[u.unit_id for u in expired])

        bb.stock = stock_summary(db, bb.now)
        below_par = [f"{i['blood_group']} {i['component']}" for i in bb.stock["low_stock"]]
        bb.log(
            self.name, "observe",
            f"{bb.stock['total_available']} units available, {bb.stock['near_expiry_total']} near expiry, "
            f"{len(below_par)} group/component(s) below par.",
            below_par=below_par,
        )


class ForecastAgent:
    name = "Forecast"

    def __init__(self, forecaster):
        self.forecaster = forecaster

    def run(self, bb, db) -> None:
        start = bb.now.date()
        bb.forecast_totals = self.forecaster.component_forecast(start, bb.horizon_days)
        bb.forecast_by_key = self.forecaster.group_forecast(start, bb.horizon_days, bb.forecast_totals)
        totals = {c: round(sum(d["units"] for d in days), 1) for c, days in bb.forecast_totals.items()}
        bb.log(self.name, "observe",
               f"Forecast demand over the next {bb.horizon_days} days: "
               + ", ".join(f"{c} {n}" for c, n in totals.items()) + " units.",
               model=self.forecaster.model_name, totals=totals)

    def fallback(self, bb) -> None:
        """If the model fails, plan on historical average demand rather
        than not planning at all."""
        mean_daily = load_demand_profile()["mean_daily"]
        bb.forecast_by_key = {
            (g, c): [mean_daily[c].get(g, 0.0)] * bb.horizon_days for c in COMPONENTS for g in BLOOD_GROUPS
        }
        bb.log(self.name, "fallback",
               "Forecast model unavailable -- planning on historical average daily demand instead.")


class PlannerAgent:
    name = "Planner"

    def run(self, bb, db) -> None:
        if bb.forecast_by_key is None:
            raise RuntimeError("no forecast available to plan against")
        units = available_units(db, bb.now)
        bb.projections = project(units, bb.forecast_by_key, bb.now,
                                 inflow=routine_inflow(), shelf_life=SHELF_LIFE_DAYS)

        for (group, component), p in bb.projections.items():
            if p["projected_shortage"] > 0:
                day = p["first_shortage_day"]
                bb.shortages.append({
                    "blood_group": group, "component": component,
                    "projected_shortage": p["projected_shortage"],
                    "first_shortage_date": str(bb.now.date() + timedelta(days=day)),
                    "supply_now": p["supply_now"], "forecast_demand": p["forecast_demand"],
                })
            need = self._actionable_need(component, p, bb.horizon_days)
            if need:
                bb.needs.append({
                    "blood_group": group, "component": component, **need,
                    "supply_now": p["supply_now"], "forecast_demand": p["forecast_demand"],
                    "first_shortage_date": str(bb.now.date() + timedelta(days=need["first_day"])),
                })
            if p["projected_wastage"] > 0:
                bb.wastage_risks.append({
                    "blood_group": group, "component": component,
                    "units": p["at_risk_units"], "supply_now": p["supply_now"],
                    "forecast_demand": p["forecast_demand"],
                })
        bb.shortages.sort(key=lambda s: (s["first_shortage_date"], -s["projected_shortage"]))
        bb.needs.sort(key=lambda n: (n["first_shortage_date"], -n["projected_shortage"]))

        for s in bb.shortages:
            bb.log(self.name, "decide",
                   f"{s['blood_group']} {s['component']}: {s['supply_now']} in stock vs "
                   f"{s['forecast_demand']} forecast -> short {s['projected_shortage']} unit(s) "
                   f"from {s['first_shortage_date']}.", **s)
        for r in bb.wastage_risks:
            bb.log(self.name, "decide",
                   f"{r['blood_group']} {r['component']}: {len(r['units'])} unit(s) projected to "
                   f"expire unused within {bb.horizon_days} days.",
                   blood_group=r["blood_group"], component=r["component"],
                   unit_ids=[u.unit_id for u in r["units"]])
        for n in bb.needs:
            bb.log(self.name, "decide",
                   f"{n['blood_group']} {n['component']}: act now for {n['projected_shortage']} unit(s) "
                   f"within {n['window_days']} day(s) ({n['unmet']} unmet + {n['safety_gap']} to restore "
                   f"safety stock of {n['safety_stock']}).", **n)
        if not bb.shortages and not bb.wastage_risks:
            bb.log(self.name, "observe", "Stock matches forecast demand -- no shortages or wastage risks projected.")

    @staticmethod
    def _actionable_need(component: str, p: dict, horizon: int):
        """
        What donor outreach should cover NOW for one group/component:
        unmet demand inside the outreach window (just-in-time for
        perishables), plus whatever it takes to keep projected stock at
        the safety level over that window. None if nothing is needed.
        """
        window = min(OUTREACH_WINDOW_DAYS[component], len(p["daily_unmet"]))
        if window == 0:
            return None
        avg_demand = p["forecast_demand"] / horizon if horizon else 0.0
        safety = round(SAFETY_STOCK_DAYS[component] * avg_demand, 1)
        unmet = sum(p["daily_unmet"][:window])
        lowest = min(p["end_stock"][:window])
        safety_gap = max(0, math.ceil(safety - lowest)) if safety >= 0.5 else 0
        total = unmet + safety_gap
        if total <= 0:
            return None
        first_day = next((d for d in range(window)
                          if p["daily_unmet"][d] or p["end_stock"][d] < safety), 0)
        return {"projected_shortage": total, "unmet": unmet, "safety_gap": safety_gap,
                "safety_stock": safety, "window_days": window, "first_day": first_day}


class LogisticsAgent:
    """Turns projected waste into supply elsewhere: near-expiry units the
    Planner says won't be used here in time are offered to a partner."""
    name = "Logistics"

    def __init__(self, partner):
        self.partner = partner

    def run(self, bb, db) -> None:
        offer = [u for r in bb.wastage_risks for u in r["units"] if is_near_expiry(u, bb.now)]
        if not offer:
            if bb.wastage_risks:
                bb.log(self.name, "decide",
                       "Units at risk aren't near expiry yet -- holding them; FEFO allocation issues them first.")
            return

        accepted_ids = set(self.partner.offer_surplus_units(
            [{"unit_id": u.unit_id, "blood_group": u.blood_group, "component": u.component,
              "expires_at": u.expires_at} for u in offer],
            now=bb.now,
        ))
        accepted = [u for u in offer if u.unit_id in accepted_ids]
        declined = [u for u in offer if u.unit_id not in accepted_ids]
        transfer_units(db, accepted, self.partner.hospital_id, bb.now)

        for u in accepted:
            bb.transfers.append({"unit_id": u.unit_id, "blood_group": u.blood_group,
                                 "component": u.component, "expires_at": u.expires_at,
                                 "partner": self.partner.hospital_id})
        if accepted:
            bb.log(self.name, "act",
                   f"Transferred {len(accepted)} near-expiry unit(s) to {self.partner.hospital_id} "
                   f"before they could expire unused here.",
                   unit_ids=[u.unit_id for u in accepted])
        if declined:
            bb.log(self.name, "decide",
                   f"{len(declined)} near-expiry unit(s) declined by {self.partner.hospital_id} -- "
                   f"flagged issue-first for local FEFO allocation.",
                   unit_ids=[u.unit_id for u in declined])


class ExplanationAgent:
    """RAG: grounds each kind of automated action in the WHO handbook, then
    writes a short brief for the Blood Bank Manager. The LLM only phrases
    -- every number in the brief comes from the other agents' facts."""
    name = "Explanation"

    STORAGE_QUERY = {
        "PRBC": "red cell concentrate storage between +2 C and +6 C blood bank refrigerator",
        "Platelets": "platelet concentrate storage up to 72 hours at 20 C to 24 C with agitation",
        "FFP": "fresh frozen plasma storage at -25 C or colder for up to 1 year",
    }
    SUPPLY_QUERY = "blood bank should ensure that blood can be provided quickly when needed"

    BRIEF_PROMPT = """You are briefing a hospital Blood Bank Manager on one cycle of an automated
blood supply system. Write 3-4 plain sentences summarizing what happened and
what needs attention. Use ONLY these verified facts -- do not invent numbers:
{facts}

Guideline sources you may cite with their bracketed numbers:
{sources}
"""

    def __init__(self, retriever=None, llm=None):
        self.retriever = retriever
        self.llm = llm

    def _topics(self, bb) -> list:
        topics = []
        for c in sorted({u["component"] for u in bb.expired_units} | {t["component"] for t in bb.transfers}):
            topics.append((f"{c} shelf life (expiry sweep / redistribution)", self.STORAGE_QUERY[c]))
        if bb.shortages:
            topics.append(("supply continuity (shortage response)", self.SUPPLY_QUERY))
        return topics

    def run(self, bb, db) -> None:
        sources = []
        if self.retriever is None:
            bb.log(self.name, "fallback", "Guideline retriever unavailable -- actions recorded without citations.")
        else:
            for label, query in self._topics(bb):
                cites = self.retriever.cite(query, k=1)
                for c in cites:
                    c["ref"] = len(sources) + 1
                    sources.append(c)
                bb.log(self.name, "explain", f"Grounded {label} in the WHO handbook.",
                       citations=[{"ref": c["ref"], "page": c["page"], "excerpt": c["excerpt"]} for c in cites])

        facts = self._facts(bb)
        bb.brief = self._template_brief(facts)
        if self.llm is not None:
            try:
                bb.brief = self.llm.generate(self.BRIEF_PROMPT.format(
                    facts="\n".join(f"- {f}" for f in facts),
                    sources="\n".join(f"[{s['ref']}] p.{s['page']}: {s['excerpt']}" for s in sources) or "(none)",
                )).strip()
            except Exception as exc:
                bb.log(self.name, "fallback", f"LLM brief failed ({exc.__class__.__name__}) -- using the template brief.")
        bb.log(self.name, "explain", "Cycle brief written.", brief=bb.brief)

    @staticmethod
    def _facts(bb) -> list:
        facts = [f"Clock: {bb.now:%Y-%m-%d %H:%M} UTC; horizon {bb.horizon_days} days."]
        if bb.donor_feedback:
            r = sum(o["responded"] for o in bb.donor_feedback)
            facts.append(f"{r} of {len(bb.donor_feedback)} pending donor alerts were answered with a donation.")
        facts.append(f"{len(bb.expired_units)} unit(s) expired this cycle (recorded as wastage).")
        if bb.shortages:
            facts.append(f"{len(bb.shortages)} projected shortage(s): " + "; ".join(
                f"{s['blood_group']} {s['component']} {s['projected_shortage']} unit(s) from {s['first_shortage_date']}"
                for s in bb.shortages) + ".")
        if bb.alerts:
            # An outreach need is unmet demand inside the outreach window,
            # a safety-stock top-up, or both -- not necessarily a shortage.
            facts.append(f"{sum(len(a['alerted']) for a in bb.alerts)} donor(s) alerted for "
                         f"{len(bb.alerts)} outreach need(s): " + "; ".join(
                             f"{a['blood_group']} {'/'.join(a.get('covers', [a['component']]))} "
                             f"{a['projected_shortage']} unit(s), {_need_kind(a)}" for a in bb.alerts) + ".")
        if bb.escalations:
            facts.append("Escalated to the regional blood bank (donors alone won't cover it): "
                         + ", ".join(f"{e['blood_group']} {'/'.join(e['covers'])} ~{e['uncovered']}"
                                     for e in bb.escalations) + " unit(s).")
        if bb.transfers:
            facts.append(f"{len(bb.transfers)} near-expiry unit(s) transferred to a partner hospital.")
        if not bb.shortages and not bb.wastage_risks:
            facts.append("No shortages or wastage risks projected.")
        return facts

    @staticmethod
    def _template_brief(facts: list) -> str:
        return " ".join(facts)
