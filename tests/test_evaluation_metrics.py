"""
The accuracy metrics the evaluation scripts report: operating-point
threshold, share of ceiling, shortage detection, and the real-data
donor mapping. Pure functions, no data files or network.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def ev():
    # The evaluation modules repoint DATABASE_URL at their own sandbox; restore ours.
    saved = os.environ["DATABASE_URL"]
    import evaluation.donor_real_data as donor
    import evaluation.prediction_benchmark as bench
    import evaluation.simulate as sim
    os.environ["DATABASE_URL"] = saved
    return bench, sim, donor


def test_threshold_keeps_target_sensitivity(ev):
    bench = ev[0]
    y = np.array([1] * 10 + [0] * 5)
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.95, 0.5, 0.2, 0.01, 0.0])
    t = bench.threshold_for_sensitivity(y, score, 0.9)
    assert t == 0.1                       # 9 of 10 positives at or above it
    assert ((score >= t) & (y == 1)).sum() / 10 >= 0.9
    assert bench.threshold_for_sensitivity(y, score, 1.0) == 0.05


def test_share_of_ceiling(ev):
    bench = ev[0]
    cell = lambda f1, auroc=None: {"f1": {"mean": f1}, **({"auroc": {"mean": auroc}} if auroc else {})}  # noqa: E731
    summary = {m: {"PRBC": cell(0.5, 0.8)} for m in bench.METHODS}
    summary["static_rule"] = {"PRBC": cell(0.4)}
    summary["ceiling"] = {"PRBC": cell(0.8, 0.9)}
    share = bench.share_of_ceiling(summary)
    assert "ceiling" not in share
    assert share["xgboost"]["PRBC"] == {"f1": 0.625, "auroc": 0.889}
    assert share["static_rule"]["PRBC"] == {"f1": 0.5, "auroc": None}


def test_shortage_detection_recall_precision_lead(ev):
    sim = ev[1]
    # (cycle day, group, component, first shortage day)
    warnings = [
        (2, "O-", "Platelets", 5),   # covers day 5 three days ahead -> hit
        (4, "O-", "Platelets", 5),
        (0, "A+", "PRBC", 1),        # nothing happened -> false alarm
        (8, "B+", "FFP", 12),        # first day past the run -> not scored
    ]
    unmet = {("O-", "Platelets", 5), ("AB+", "FFP", 3)}  # the second was never projected
    out = sim.shortage_detection(warnings, unmet, days=10)
    assert out["shortages"] == 2 and out["warnings"] == 3
    assert out["recall_0d"] == out["recall_1d"] == out["recall_2d"] == 0.5
    assert out["precision"] == round(2 / 3, 4)
    assert out["mean_lead_days"] == 3.0
    # A warning only reaches as far as its horizon
    assert sim.shortage_detection([(0, "O-", "Platelets", 0)], {("O-", "Platelets", 7)}, 10)["recall_0d"] == 0


def test_projected_shortages_reads_only_shortage_events(ev):
    sim = ev[1]
    now = datetime(2026, 8, 3, 8)
    events = [
        {"agent": "Planner", "kind": "decide", "message": "",
         "data": {"blood_group": "O-", "component": "Platelets", "first_shortage_date": "2026-08-05"}},
        {"agent": "Planner", "kind": "decide", "message": "",   # an outreach need, not a shortage
         "data": {"blood_group": "O-", "component": "Platelets", "first_shortage_date": "2026-08-03",
                  "window_days": 2}},
        {"agent": "Inventory", "kind": "observe", "message": ""},
    ]
    assert sim.projected_shortages(events, 2, now) == [(2, "O-", "Platelets", 4)]


def test_beta_history_mapping_and_precision_at(ev):
    donor = ev[2]
    df = pd.DataFrame({"frequency": [2, 30], "time": [10, 40]})
    h = donor.beta_history(df)
    assert h.alerts_sent.tolist() == [4, 30]       # floor(10/3)+1; clamped to Frequency
    assert h.alerts_responded.tolist() == [2, 30]
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    assert donor.precision_at(y, np.arange(10)[::-1], share=0.2) == 0.5


def test_planning_level_rule_needs_fewer_shortages_without_more_waste(ev):
    sim = ev[1]
    row = lambda q, short, not_used, alerts=100, per=2.0: {"quantile": q, "summary": {  # noqa: E731
        "shortage_units": {"mean": short, "std": 5.0}, "not_used_here": {"mean": not_used},
        "donor_alerts": {"mean": alerts}, "alerts_per_donation": {"mean": per}}}
    base = row(None, 40, 50)
    assert sim.choose_planning_level([base, row(0.7, 10, 90)]) is None      # waste nearly doubles
    assert sim.choose_planning_level([base, row(0.7, 37, 50)]) is None      # within seed noise
    assert sim.choose_planning_level([base, row(0.7, 20, 54), row(0.8, 15, 54, alerts=120)]) == 0.7


def test_cooldown_rule_takes_the_longest_that_does_not_add_shortages(ev):
    sim = ev[1]
    row = lambda k, short: {"cooldown": k, "summary": {"shortage_units": {"mean": short}}}  # noqa: E731
    assert sim.choose_cooldown([row(0, 104), row(3, 93), row(7, 98), row(14, 169)]) == 7
    assert sim.choose_cooldown([row(0, 40), row(7, 90)]) == 0
