"""
The realistic donor model (D85) and donor sex / demo seed (D86): contact
fatigue, new donors, the production contact cooldown, NBTC's 120-day
rule for women from the database, and the demo's critical cells.
"""

import os
from datetime import datetime, timedelta

import pytest

from agents.donor_module import (
    add_donors,
    alert_donors,
    arriving_donors,
    eligible_candidates,
    ensure_donor_pool,
    fatigued_rate,
    required_interval_days,
)
from bloodbank import catalog
from bloodbank.forecast import par_level
from bloodbank.stock import stock_status
from db.database import SessionLocal
from db.models import Donor, DonorAlert

NOW = datetime(2026, 10, 30, 8)


@pytest.fixture
def clean_donors():
    def wipe():
        with SessionLocal() as db:
            db.query(DonorAlert).delete()
            db.query(Donor).delete()
            db.commit()
    wipe()
    yield
    wipe()


def _donor(donor_id, sex="M", days_since=200):
    return {"donor_id": donor_id, "name": donor_id, "phone": "+910000000000", "blood_type": "O+", "sex": sex,
            "last_donation_date": (NOW - timedelta(days=days_since)).strftime("%Y-%m-%d"),
            "alerts_sent": 0, "alerts_responded": 0, "simulated_response_rate": 0.3}


def test_fatigue_drops_with_recent_alerts_and_recovers():
    f = {"decay_per_alert": 0.7, "half_life_days": 14}
    assert fatigued_rate(0.4, [], f) == 0.4
    assert fatigued_rate(0.4, [0], f) == pytest.approx(0.28)
    assert fatigued_rate(0.4, [14], f) == pytest.approx(0.4 * 0.7 ** 0.5)     # half the effect after a half-life
    assert fatigued_rate(0.4, [0, 0], f) == pytest.approx(0.4 * 0.49)
    assert fatigued_rate(0.4, [0], False) == 0.4                               # "no fatigue" reproduces D53


def test_arriving_donors_have_no_history_and_are_eligible():
    new = arriving_donors(3, seed=1, today=NOW, prefix="N007")
    assert [d["donor_id"] for d in new] == ["N00700", "N00701", "N00702"]
    assert all(d["alerts_sent"] == d["alerts_responded"] == 0 for d in new)
    last = datetime.strptime(new[0]["last_donation_date"], "%Y-%m-%d")
    assert (NOW - last).days >= required_interval_days("PRBC", None, "F")


def test_bandit_rule_keeps_thompson_within_seed_noise():
    saved = os.environ["DATABASE_URL"]  # the evaluation package repoints it; restore ours
    import evaluation.bandit_regret as b
    os.environ["DATABASE_URL"] = saved
    row = lambda mean, std=20.0: {"donations": {"mean": mean, "std": std}}  # noqa: E731
    rows = {"random": row(300), "oracle": row(600), "greedy+cooldown7": row(476), "thompson": row(450),
            "thompson+cooldown7": row(464)}
    assert b.choose_policy(rows)["chosen"] == "thompson+cooldown7"           # 12 < 20: kept
    rows["greedy+cooldown7"] = row(500)
    assert b.choose_policy(rows)["chosen"] == "greedy+cooldown7"             # 36 > 20: switch


def test_cooldown_skips_recently_alerted_donors(clean_donors, monkeypatch):
    monkeypatch.setattr(catalog, "CONTACT_COOLDOWN_DAYS", 14)
    with SessionLocal() as db:
        add_donors(db, [_donor("a"), _donor("b")])
        db.add(DonorAlert(donor_id="a", blood_type="O+", component="PRBC", method="log",
                          sent_at=NOW - timedelta(days=5), responded=False))
        db.commit()
        assert [d["donor_id"] for d in eligible_candidates(db, "O+", "PRBC", NOW)] == ["b"]
        assert len(eligible_candidates(db, "O+", "PRBC", NOW + timedelta(days=10))) == 2
        monkeypatch.setattr(catalog, "CONTACT_COOLDOWN_DAYS", 0)
        assert len(eligible_candidates(db, "O+", "PRBC", NOW)) == 2


def test_women_in_the_database_wait_120_days(clean_donors, monkeypatch):
    monkeypatch.setattr(catalog, "CONTACT_COOLDOWN_DAYS", 0)
    with SessionLocal() as db:
        add_donors(db, [_donor("man", "M", 100), _donor("woman", "F", 100), _donor("unknown", None, 100)])
        db.commit()
        assert sorted(d["donor_id"] for d in eligible_candidates(db, "O+", "PRBC", NOW)) == ["man", "unknown"]
        assert alert_donors(db, "O+", "PRBC", 5, NOW)["eligible_count"] == 2


def test_seeded_pool_stores_sex(clean_donors):
    with SessionLocal() as db:
        ensure_donor_pool(db, today=NOW)
        db.commit()
        sexes = {s for (s,) in db.query(Donor.sex)}
    assert sexes == {"M", "F"}


def test_demo_scenarios_are_critical():
    from scripts.seed import SCENARIOS
    for (group, component), count in SCENARIOS.items():
        assert stock_status(count, par_level(group, component)) == "Critical", (group, component)
