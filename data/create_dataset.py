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

All are simulated, NOT real patient records. Every patient parameter, its
source and the realism checks against NHANES are in data/DATA_CARD.md.

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

# Calibration targets and sources: data/DATA_CARD.md. The global
# np.random calls below consume exactly the same draws as before (one
# normal for Hb, one for platelets, one for INR, then age and surgery), so
# blood_demand.csv further down is unchanged; the new variables (sex, the
# peri-operative drop) come from their own RNG.
_periop_rng = np.random.default_rng(17)

# Sex: generator-internal for Hb, stored for the realism check, NOT a model
# input (transfusion thresholds aren't sex-specific). 50/50 is an assumption:
# no source found for the sex mix of Indian surgical patients.
sex = np.where(_periop_rng.random(N_PATIENTS) < 0.5, "M", "F")

# Pre-operative Hb by sex. NHANES 2021-23 adults are M 14.4-15.0, F 13.0-13.3
# g/dL (SD ~1.2-1.4); we shift down to Indian levels: targets are anaemia
# (WHO 2024 cut-offs M<13, F<12) ~25% in men (NFHS-5: 25.0%) and ~30-35%
# overall (ISOS, Fowler 2018: 30.1%, LMIC 32.5%). Women land at ~43%, below
# NFHS-5's 57% (women 15-49, general population). The shift is our
# calibration, not a published number.
HB_PREOP_MEAN = {"M": 14.0, "F": 12.3}
HB_PREOP_SD = 1.6
hemoglobin_preop = np.clip(
    np.vectorize(HB_PREOP_MEAN.get)(sex) + HB_PREOP_SD * np.random.normal(0, 1, N_PATIENTS), 6.0, 18.5
)
platelets = np.clip(np.random.normal(250000, 70000, N_PATIENTS), 50000, 450000).astype(int)  # NHANES 256k (68k), 95th pct ~370-430k
# INR: right-skewed, a healthy core of 0.8-1.2 with a tail (anticoagulants,
# liver disease). Shifted lognormal, median 1.02, ~4% above 1.5. That share
# is an ESTIMATE: no clean published figure for pre-op INR > 1.5 was found
# (one Mexican elective cohort: abnormal PT 9.5%).
inr = np.clip(0.75 + np.exp(np.log(0.27) + 0.58 * np.random.normal(0, 1, N_PATIENTS)), 0.8, 4.0)
age = np.random.randint(18, 90, N_PATIENTS)
surgery_type = np.random.choice(
    ["Cardiac", "Orthopedic", "General", "Emergency"],
    N_PATIENTS,
    p=[0.20, 0.25, 0.40, 0.15],  # Emergency should be a minority of cases
)

# Peri-operative Hb drop (blood loss, haemodilution on bypass): the stored
# `hemoglobin` is the value at the transfusion decision, pre-op Hb minus a
# surgery-specific drop, so surgery changes the patient's Hb, not only the
# threshold. Gamma draws (never negative, right-skewed: most patients lose a
# little, a few bleed a lot). Mean/SD in g/dL, tuned so PRBC rates land in
# the published ranges with the protocol thresholds (9 cardiac, 8 elective,
# 10 emergency): cardiac 40-60% (TRICS III 52-73%), orthopaedic 10-20%
# (NSQIP THA 22%, TKA 18%; with TXA ~16%), general 5-15% (Indian MSBOS ~15%),
# emergency 15-25% (level-I trauma ~17%). The drop sizes are calibrated to
# those rates; only the cardiac 3-4 g/dL is close to a reported figure.
HB_DROP = {"Cardiac": (4.0, 1.0), "Orthopedic": (3.0, 0.8), "General": (2.0, 1.0), "Emergency": (1.2, 1.5)}
_drop_mean = np.vectorize(lambda s: HB_DROP[s][0])(surgery_type)
_drop_sd = np.vectorize(lambda s: HB_DROP[s][1])(surgery_type)
hb_drop = _periop_rng.gamma((_drop_mean / _drop_sd) ** 2, _drop_sd ** 2 / _drop_mean)
hemoglobin = np.clip(hemoglobin_preop - hb_drop, 4.0, 18.5)

# A thrombocytopenic subgroup (e.g. dengue, chemotherapy, liver disease).
# The healthy-centered draw above is clipped at 50,000/uL, so without
# this no patient would ever fall below the platelet threshold and the model
# could never learn the platelet criterion. Drawn from a separate RNG so
# the global sequence -- and therefore blood_demand.csv below -- is
# unchanged. The 8% share has no published source (NHANES <150k: 2-4%).
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
    "hemoglobin_preop": np.round(hemoglobin_preop, 2),  # not a model input; realism check only
    "sex": sex,                                          # not a model input; realism check only
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
