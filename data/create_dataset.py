"""
HemoSmart - Dataset Simulation
--------------------------------
Generates the datasets since real hospital patient data cannot be used
(privacy / HIPAA constraints):

1. patient_data.csv      -> transfusion prediction models (XGBoost, one
                            per component: PRBC / Platelets / FFP)
2. blood_demand.csv      -> aggregate daily demand (legacy LSTM experiment)
3. component_demand.csv  -> daily units used per blood group x component,
                            for the forecasting + gap projection pipeline

Both are simulated using medically plausible ranges (based on WHO
reference ranges), NOT real patient records.

Run:
    python data/create_dataset.py
"""

import os
import sys

import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.demand_process import sample_demand
from data.label_process import sample_labels

np.random.seed(42)

# ----------------------------------------------------------------------
# 1. Patient transfusion dataset
# ----------------------------------------------------------------------
N_PATIENTS = 5000

# Most patients are clinically "normal" with a minority presenting
# abnormal values -> use normal distributions centered on healthy
# reference values (clipped to valid physiological ranges), instead
# of uniform, which unrealistically overloads the abnormal end.
hemoglobin = np.clip(np.random.normal(12.5, 2.2, N_PATIENTS), 5.0, 16.0)
platelets = np.clip(np.random.normal(250000, 70000, N_PATIENTS), 50000, 400000).astype(int)
inr = np.clip(np.random.normal(1.1, 0.35, N_PATIENTS), 0.8, 4.0)
age = np.random.randint(18, 90, N_PATIENTS)
surgery_type = np.random.choice(
    ["Cardiac", "Orthopedic", "General", "Emergency"],
    N_PATIENTS,
    p=[0.20, 0.25, 0.40, 0.15],  # Emergency should be a minority of cases
)

# A thrombocytopenic subgroup (e.g. dengue, chemotherapy, liver disease).
# The healthy-centered draw above is clipped at 50,000/uL, so without
# this no patient would ever fall below the platelet threshold and the model
# could never learn the platelet criterion. Drawn from a separate RNG so
# the global sequence -- and therefore blood_demand.csv below -- is
# unchanged.
_subgroup_rng = np.random.default_rng(7)
thrombocytopenic = _subgroup_rng.random(N_PATIENTS) < 0.08
platelets[thrombocytopenic] = _subgroup_rng.integers(
    10_000, 120_000, thrombocytopenic.sum()
)

# Round first, then label, so the labels are a function of the stored values.
hemoglobin = np.round(hemoglobin, 2)
inr = np.round(inr, 2)

# Labels are simulated clinician decisions (data/label_process.py):
# context-dependent guideline thresholds, a soft boundary where
# clinicians disagree, and a little record noise -- not a hard rule the
# model could read straight off the features. Own RNG, so the datasets
# below are unaffected.
labels = sample_labels(hemoglobin, platelets, inr, surgery_type, np.random.default_rng(13))
needs_prbc, needs_platelets, needs_ffp = labels["PRBC"], labels["Platelets"], labels["FFP"]
transfusion_needed = (needs_prbc | needs_platelets | needs_ffp).astype(int)

patient_data = pd.DataFrame({
    "hemoglobin": hemoglobin,
    "platelets": platelets,
    "INR": inr,
    "age": age,
    "surgery_type": surgery_type,
    "needs_prbc": needs_prbc,
    "needs_platelets": needs_platelets,
    "needs_ffp": needs_ffp,
    "transfusion_needed": transfusion_needed,
})

patient_data.to_csv("data/patient_data.csv", index=False)
print(f"patient_data.csv created -> {patient_data.shape[0]} rows")
print("Transfusion rate:", round(patient_data['transfusion_needed'].mean() * 100, 1), "%")

# ----------------------------------------------------------------------
# 2. Blood demand time-series dataset (2 years, daily)
# ----------------------------------------------------------------------
dates = pd.date_range(start="2024-01-01", end="2025-12-31", freq="D")
n_days = len(dates)

# Base demand + weekly seasonality (higher on weekdays, scheduled surgeries)
# + yearly seasonality (slight seasonal wave) + random noise
day_of_week_effect = np.where(pd.Series(dates).dt.dayofweek < 5, 5, -3)
yearly_wave = 5 * np.sin(np.arange(n_days) * (2 * np.pi / 365))
weekly_wave = 3 * np.sin(np.arange(n_days) * (2 * np.pi / 7))
noise = np.random.normal(0, 4, n_days)

base_demand = 30
units_used = (
    base_demand + day_of_week_effect + yearly_wave + weekly_wave + noise
)
units_used = np.clip(units_used, 5, None).round().astype(int)

blood_demand = pd.DataFrame({
    "ds": dates,       # Prophet requires columns named 'ds' and 'y'
    "y": units_used,
})

blood_demand.to_csv("data/blood_demand.csv", index=False)
print(f"blood_demand.csv created -> {blood_demand.shape[0]} rows")
print("Average daily units used:", round(blood_demand['y'].mean(), 1))

# ----------------------------------------------------------------------
# 3. Component demand: daily units used per blood group x component
# ----------------------------------------------------------------------
# What the gap projection actually needs: platelets and red cells are
# not interchangeable, and an O- shortage is not covered by B+ stock.
# The process itself lives in data/demand_process.py (shared with the
# evaluation simulator). Own RNG, so the two datasets above are
# unaffected by this section.
_demand_rng = np.random.default_rng(11)

rows = [
    pd.DataFrame({"ds": dates, "blood_group": group, "component": component, "units": units})
    for (group, component), units in sample_demand(dates, _demand_rng).items()
]

component_demand = pd.concat(rows, ignore_index=True)
component_demand.to_csv("data/component_demand.csv", index=False)
print(f"component_demand.csv created -> {component_demand.shape[0]} rows")
print(component_demand.groupby("component")["units"].sum().div(n_days).round(1).to_string())
