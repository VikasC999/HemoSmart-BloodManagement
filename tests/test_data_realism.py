"""The patient generator's realism targets (data/DATA_CARD.md), checked on the
seeded sample it writes (data/patient_data.csv). No database."""

import os

import numpy as np
import pandas as pd
import pytest

from evaluation.data_realism import weighted_ks, weighted_quantile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(os.path.join(REPO_ROOT, "data", "patient_data.csv"))


def test_inr_is_right_skewed_with_a_small_tail(df):
    assert 0.02 < (df.INR > 1.5).mean() < 0.06
    assert df.INR.mean() > df.INR.median()


def test_prbc_rates_by_surgery_land_in_published_ranges(df):
    rate = df.groupby("surgery_type").needs_prbc.mean()
    assert 0.40 <= rate["Cardiac"] <= 0.60
    assert 0.15 <= rate["Emergency"] <= 0.25
    assert 0.10 <= rate["Orthopedic"] <= 0.20
    assert 0.05 <= rate["General"] <= 0.15
    assert rate["Cardiac"] > rate["Emergency"] > max(rate["Orthopedic"], rate["General"])


def test_preop_hemoglobin_differs_by_sex(df):
    means = df.groupby("sex").hemoglobin_preop.mean()
    assert 1.4 < means["M"] - means["F"] < 2.0
    anaemic = df.hemoglobin_preop < df.sex.map({"M": 13.0, "F": 12.0})
    assert 0.25 < anaemic.mean() < 0.40  # ISOS 30%, NFHS-5 25% men / 57% women


def test_decision_hemoglobin_is_preop_minus_a_non_negative_drop(df):
    assert (df.hemoglobin <= df.hemoglobin_preop + 0.01).all()
    drop = (df.hemoglobin_preop - df.hemoglobin).groupby(df.surgery_type).mean()
    assert drop["Cardiac"] > drop["Orthopedic"] > drop["General"]


def test_sex_is_not_a_model_input():
    from models.train_xgboost import FEATURES
    assert FEATURES == ["hemoglobin", "platelets", "INR", "age", "surgery_type"]


def test_weighted_helpers():
    x = np.arange(1, 101, dtype=float)
    assert weighted_quantile(x, np.ones(100), 50) == pytest.approx(50.5, abs=1)
    assert weighted_ks(x, x, np.ones(100)) == 0.0
    assert weighted_ks(x, x + 1000, np.ones(100)) == 1.0
