"""
HemoSmart - Synthetic Blood Demand Process
----------------------------------------------
The one definition of how daily blood demand is generated, per blood
group x component. data/create_dataset.py samples the training history
from it; evaluation/simulate.py samples the "true" future the policies
are tested against -- so the simulated world and the forecaster's
training data come from the same process.
"""

import numpy as np
import pandas as pd

# Approximate ABO/RhD distribution of the Indian population.
GROUP_SHARE = {
    "O+": 0.36, "B+": 0.31, "A+": 0.21, "AB+": 0.07,
    "O-": 0.02, "B-": 0.015, "A-": 0.01, "AB-": 0.005,
}
BASE_DAILY_UNITS = {"PRBC": 20.0, "Platelets": 7.0, "FFP": 6.0}
ORIGIN = pd.Timestamp("2024-01-01")  # phase reference for the PRBC yearly wave


def expected_demand(dates) -> dict:
    """{component: array of expected hospital-wide units per day}."""
    dates = pd.DatetimeIndex(dates)
    day_index = np.asarray((dates - ORIGIN).days)
    day_of_year = np.asarray(dates.dayofyear)
    # Scheduled surgery happens on weekdays.
    weekday_factor = np.where(np.asarray(dates.dayofweek) < 5, 1.15, 0.65)
    # Post-monsoon dengue season (peaking mid-September) drives platelet
    # demand up sharply -- the seasonal spike forecasting has to anticipate.
    dengue_wave = np.exp(-0.5 * ((day_of_year - 258) / 25.0) ** 2)
    component_factor = {
        "PRBC": 1 + 0.08 * np.sin(day_index * 2 * np.pi / 365),
        "Platelets": 1 + 0.9 * dengue_wave,
        "FFP": np.ones(len(dates)),
    }
    return {c: base * weekday_factor * component_factor[c] for c, base in BASE_DAILY_UNITS.items()}


# Routine donations fall in the summer holidays (college donors away). A Mumbai
# hospital reported 30-40% fewer donors in Apr-May 2016 (news report, Medical
# Dialogues); a moderate 20% dip is an assumption, not a measured rate.
# {month: share of normal routine supply}; set to {} to switch it off.
SUPPLY_SEASONAL_FACTOR = {4: 0.8, 5: 0.8}


def supply_factor(dates) -> np.ndarray:
    """Share of normal routine supply arriving on each date."""
    return np.array([SUPPLY_SEASONAL_FACTOR.get(m, 1.0) for m in pd.DatetimeIndex(dates).month])


def sample_demand(dates, rng: np.random.Generator) -> dict:
    """{(group, component): array of units demanded per day}. Draw order
    (component, then group) is fixed, so a given rng is reproducible."""
    expected = expected_demand(dates)
    return {
        (group, component): rng.poisson(expected[component] * share)
        for component in BASE_DAILY_UNITS
        for group, share in GROUP_SHARE.items()
    }
