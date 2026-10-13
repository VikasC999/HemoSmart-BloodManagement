"""
HemoSmart - Supply Outlook
------------------------------
Forecast + gap projection in one place: what demand is coming, and for
every (blood group, component) whether current stock will fall short of
it or expire unused within the horizon. The REST API, the chat agent,
the MCP server and the autonomous Planner agent all call this, so they
can never disagree about the outlook.
"""

from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from bloodbank.catalog import BLOOD_GROUPS, COMPONENTS, SHELF_LIFE_DAYS
from bloodbank.forecast import get_forecaster, routine_inflow
from bloodbank.projection import project
from bloodbank.stock import available_units

DEFAULT_HORIZON_DAYS = 7


def compute_outlook(db: Session, now: datetime, days: int = DEFAULT_HORIZON_DAYS, forecaster=None,
                    totals: dict = None) -> dict:
    """`totals`: a precomputed component forecast for this start date and
    horizon (the expensive part, safe to cache); stock is always read fresh."""
    forecaster = forecaster or get_forecaster()
    start = now.date()
    totals = totals or forecaster.component_forecast(start, days)
    by_key = forecaster.group_forecast(start, days, totals)
    units = available_units(db, now)
    projections = project(units, by_key, now, inflow=routine_inflow(), shelf_life=SHELF_LIFE_DAYS)

    rows = []
    for group in BLOOD_GROUPS:
        for component in COMPONENTS:
            p = projections[(group, component)]
            rows.append({
                "blood_group": group,
                "component": component,
                "supply_now": p["supply_now"],
                "forecast_demand": p["forecast_demand"],
                "projected_shortage": p["projected_shortage"],
                "projected_wastage": p["projected_wastage"],
                "first_shortage_date": (
                    str(start + timedelta(days=p["first_shortage_day"]))
                    if p["first_shortage_day"] is not None else None
                ),
                "days_of_cover": p["days_of_cover"],
                "at_risk_unit_ids": [u.unit_id for u in p["at_risk_units"]],
            })

    return {
        "as_of": now.isoformat(),
        "horizon_days": days,
        "model_used": forecaster.model_name,
        "component_forecast": totals,
        "projection": rows,
        "shortages": [r for r in rows if r["projected_shortage"] > 0],
        "wastage_risks": [r for r in rows if r["projected_wastage"] > 0],
    }
