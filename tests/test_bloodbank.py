"""Pure blood-bank logic: compatibility, wastage-aware allocation, gap projection, dosing."""

import math
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from bloodbank.allocation import choose_units, is_near_expiry, rank_units
from bloodbank.catalog import compatible_donor_groups
from bloodbank.projection import project, project_component
from schema.clinical_thresholds import recommended_units, urgency

NOW = datetime(2026, 10, 13, 8, 0)


def unit(group, component="PRBC", days_left=20.0, uid=None):
    return SimpleNamespace(unit_id=uid or f"{group}-{component}-{days_left}", blood_group=group,
                           component=component, expires_at=NOW + timedelta(days=days_left))


# ----------------------------------------------------------------------
# Compatibility
# ----------------------------------------------------------------------
@pytest.mark.parametrize("recipient, component, expected", [
    ("O-", "PRBC", {"O-"}),
    ("AB+", "PRBC", {"A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"}),
    ("A+", "PRBC", {"A+", "A-", "O+", "O-"}),
    ("A+", "FFP", {"A+", "A-", "AB+", "AB-"}),        # plasma: the reverse rule
    ("AB-", "FFP", {"AB+", "AB-"}),
    ("O-", "Platelets", {"O-", "A-", "B-", "AB-"}),   # RhD-negative only
])
def test_compatible_donor_groups(recipient, component, expected):
    groups = compatible_donor_groups(recipient, component)
    assert set(groups) == expected
    assert groups[0] == recipient  # identical group first


def test_unknown_group_rejected():
    with pytest.raises(ValueError):
        compatible_donor_groups("C+", "PRBC")


# ----------------------------------------------------------------------
# Allocation (FEFO, wastage-aware)
# ----------------------------------------------------------------------
def test_near_expiry_window_is_component_specific():
    assert is_near_expiry(unit("O+", "Platelets", days_left=1.5), NOW)
    assert not is_near_expiry(unit("O+", "Platelets", days_left=3), NOW)
    assert is_near_expiry(unit("O+", "PRBC", days_left=6), NOW)
    assert not is_near_expiry(unit("O+", "PRBC", days_left=10), NOW)


def test_near_expiry_compatible_unit_beats_fresh_identical():
    # An O- red cell unit expiring in 3 days goes to an A+ patient before
    # a fresh A+ unit -- used now rather than wasted.
    fresh_own = unit("A+", days_left=30)
    expiring_o_neg = unit("O-", days_left=3)
    assert rank_units([fresh_own, expiring_o_neg], "A+", "PRBC", NOW)[0] is expiring_o_neg


def test_otherwise_identical_group_first_then_fefo():
    # Without expiry pressure, keep universal O- for patients who need it
    o_neg = unit("O-", days_left=15)
    own_later = unit("A+", days_left=25)
    own_sooner = unit("A+", days_left=12)
    assert rank_units([o_neg, own_later, own_sooner], "A+", "PRBC", NOW) == [own_sooner, own_later, o_neg]


def test_incompatible_expired_and_wrong_component_excluded():
    units = [unit("B+"), unit("A+", days_left=-1), unit("A+", "FFP"), unit("A+", uid="ok")]
    assert [u.unit_id for u in rank_units(units, "A+", "PRBC", NOW)] == ["ok"]


def test_choose_units_returns_what_stock_allows():
    assert len(choose_units([unit("O-"), unit("O-", days_left=9)], "O-", "PRBC", 5, NOW)) == 2


# ----------------------------------------------------------------------
# Gap projection
# ----------------------------------------------------------------------
def test_perishable_stock_is_both_short_and_wasted():
    # 6 platelet units expiring within 2 days cannot cover 1/day for a week:
    # some are wasted early, and demand goes unmet later.
    units = [unit("A+", "Platelets", days_left=1.5, uid=f"p{i}") for i in range(6)]
    p = project_component("Platelets", units, {"A+": [1.0] * 7}, NOW)["A+"]
    assert p["projected_wastage"] == 4   # 2 used before expiry, 4 expire
    assert p["projected_shortage"] == 5  # days 2..6 unmet
    assert p["first_shortage_day"] == 2
    assert {u.unit_id for u in p["at_risk_units"]} <= {f"p{i}" for i in range(6)}


def test_compatible_substitution_prevents_waste_and_false_shortage():
    # O- platelet patients can receive AB- platelets: projecting groups in
    # isolation would show an O- shortage AND AB- waste.
    units = [unit("AB-", "Platelets", days_left=4, uid=f"ab{i}") for i in range(3)]
    p = project_component("Platelets", units, {"O-": [1.0, 1.0, 1.0]}, NOW)
    assert p["O-"]["projected_shortage"] == 0
    assert p["AB-"]["projected_wastage"] == 0


def test_routine_inflow_closes_gap_and_is_never_reported_at_risk():
    no_inflow = project_component("PRBC", [], {"B+": [1.0] * 7}, NOW)["B+"]
    with_inflow = project_component("PRBC", [], {"B+": [1.0] * 7}, NOW,
                                    inflow={"B+": 1.0}, shelf_life_days=42)["B+"]
    assert no_inflow["projected_shortage"] == 7
    assert with_inflow["projected_shortage"] == 0
    assert with_inflow["at_risk_units"] == []


def test_fractional_demand_carries_over():
    p = project_component("PRBC", [], {"AB-": [0.4] * 7}, NOW)["AB-"]
    assert p["projected_shortage"] == math.floor(0.4 * 7)


def test_project_covers_every_group_of_each_component():
    result = project([unit("O+")], {("O+", "PRBC"): [0.0] * 3}, NOW)
    assert len([k for k in result if k[1] == "PRBC"]) == 8


# ----------------------------------------------------------------------
# Dose + urgency rules
# ----------------------------------------------------------------------
@pytest.mark.parametrize("component, hb, plt, inr, surgery, units", [
    ("PRBC", 7.5, 200_000, 1.0, "General", 1),     # 0.5 below 8 -> single unit
    ("PRBC", 6.2, 200_000, 1.0, "General", 2),
    ("PRBC", 4.0, 200_000, 1.0, "Emergency", 4),   # capped
    ("Platelets", 13, 15_000, 1.0, "General", 2),
    ("Platelets", 13, 40_000, 1.0, "General", 1),
    ("FFP", 13, 200_000, 2.4, "General", 4),
    ("FFP", 13, 200_000, 1.7, "General", 2),
])
def test_recommended_units(component, hb, plt, inr, surgery, units):
    assert recommended_units(component, hb, plt, inr, surgery) == units


def test_urgency():
    assert urgency(13, 250_000, 1.0, "General", False) == "Low"
    assert urgency(7.5, 250_000, 1.0, "General", True) == "Moderate"
    assert urgency(6.5, 250_000, 1.0, "General", True) == "Critical"
    assert urgency(9.5, 250_000, 1.0, "Emergency", True) == "Critical"
