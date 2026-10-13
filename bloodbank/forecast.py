"""
HemoSmart - Demand Forecaster
---------------------------------
Loads the per-component Prophet models and the demand profile trained by
models/train_prophet.py, and turns them into what the rest of the system
needs: expected units per day for every (blood group, component) over a
horizon starting at any date (the models were trained on history ending
`trained_through`, and Prophet extrapolates trend + seasonality forward).
"""

import json
import os
import pickle
from datetime import date, timedelta
from functools import lru_cache

import pandas as pd

from bloodbank.catalog import BLOOD_GROUPS, COMPONENTS, DAYS_OF_COVER, MIN_PAR_UNITS, ROUTINE_SUPPLY_SHARE

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


class DemandForecaster:
    model_name = "Prophet (per component) + group-share disaggregation"

    def __init__(self, models: dict = None, profile: dict = None):
        if models is None:
            with open(os.path.join(MODEL_DIR, "prophet_components.pkl"), "rb") as f:
                models = pickle.load(f)
        if profile is None:
            profile = load_demand_profile()
        self.models = models
        self.profile = profile

    def component_forecast(self, start: date, days: int) -> dict:
        """{component: [{"date", "units", "lower", "upper"}]} -- hospital-wide totals."""
        future = pd.DataFrame({"ds": pd.date_range(start, periods=days, freq="D")})
        out = {}
        for component in COMPONENTS:
            fc = self.models[component].predict(future)
            out[component] = [
                {
                    "date": str(row.ds.date()),
                    "units": round(max(row.yhat, 0.0), 2),
                    "lower": round(max(row.yhat_lower, 0.0), 2),
                    "upper": round(max(row.yhat_upper, 0.0), 2),
                }
                for row in fc.itertuples()
            ]
        return out

    def group_forecast(self, start: date, days: int, totals: dict = None) -> dict:
        """{(group, component): [expected units per day]}"""
        totals = totals or self.component_forecast(start, days)
        shares = self.profile["group_share"]
        return {
            (group, component): [day["units"] * shares[component].get(group, 0.0) for day in totals[component]]
            for component in COMPONENTS
            for group in BLOOD_GROUPS
        }


@lru_cache(maxsize=1)
def load_demand_profile() -> dict:
    with open(os.path.join(MODEL_DIR, "demand_profile.json")) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_forecaster() -> DemandForecaster:
    return DemandForecaster()


def par_level(group: str, component: str) -> float:
    """Target stock: DAYS_OF_COVER days of average demand, with a floor."""
    mean_daily = load_demand_profile()["mean_daily"][component].get(group, 0.0)
    return max(MIN_PAR_UNITS, round(mean_daily * DAYS_OF_COVER[component], 1))


def routine_inflow() -> dict:
    """{(group, component): expected routine units/day} -- see
    bloodbank.catalog.ROUTINE_SUPPLY_SHARE."""
    mean_daily = load_demand_profile()["mean_daily"]
    return {
        (g, c): mean_daily[c].get(g, 0.0) * ROUTINE_SUPPLY_SHARE[c]
        for c in COMPONENTS for g in BLOOD_GROUPS
    }


def horizon_dates(start: date, days: int) -> list:
    return [str(start + timedelta(days=i)) for i in range(days)]
