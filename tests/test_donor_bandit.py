import random
from datetime import datetime, timedelta

from collections import Counter

import pytest

from agents.donor_module import (
    MIN_DONATION_INTERVAL_DAYS,
    RESPONSE_RATE_PRIOR,
    filter_eligible_donors,
    is_eligible,
    required_interval_days,
    select_top_donors,
    simulate_donors,
    update_donor_response,
)

TODAY = datetime(2026, 10, 13)


def _donor(donor_id, blood_type="O-", days_since=200, sent=0, responded=0):
    return {
        "donor_id": donor_id, "name": donor_id, "phone": "x", "blood_type": blood_type,
        "last_donation_date": (TODAY - timedelta(days=days_since)).strftime("%Y-%m-%d"),
        "alerts_sent": sent, "alerts_responded": responded,
    }


def test_eligibility_requires_matching_type():
    assert not is_eligible(_donor("a", blood_type="A+"), "O-", today=TODAY)


def test_eligibility_enforces_donation_interval():
    assert not is_eligible(_donor("a", days_since=MIN_DONATION_INTERVAL_DAYS - 1), "O-", today=TODAY)
    assert is_eligible(_donor("a", days_since=MIN_DONATION_INTERVAL_DAYS), "O-", today=TODAY)


@pytest.mark.parametrize("component, last, sex, days", [
    ("PRBC", "whole_blood", "M", 90),       # NBTC 2017: whole blood, men
    ("PRBC", "whole_blood", "F", 120),      # women
    ("PRBC", None, None, 90),               # unknown history/sex: the men's interval
    ("FFP", "apheresis", "F", 28),          # whole blood after platelet apheresis
    ("Platelets", "apheresis", "M", 4),     # 48 h and at most 2 a week
    ("Platelets", "whole_blood", "M", 28),
])
def test_required_interval_follows_nbtc(component, last, sex, days):
    assert required_interval_days(component, last, sex) == days


def test_women_wait_120_days_for_whole_blood():
    woman = {**_donor("w", days_since=100), "sex": "F"}
    assert not is_eligible(woman, "O-", today=TODAY)
    assert is_eligible({**woman, "sex": "M"}, "O-", today=TODAY)


def test_simulated_pool_matches_population_mix_and_lower_response():
    pool = simulate_donors(n=4000, seed=7)
    types = Counter(d["blood_type"] for d in pool)
    assert 0.33 < types["O+"] / 4000 < 0.39 and types["O-"] / 4000 < 0.04   # not 12.5% each
    mean_rate = sum(d["simulated_response_rate"] for d in pool) / 4000
    assert abs(mean_rate - RESPONSE_RATE_PRIOR[0] / sum(RESPONSE_RATE_PRIOR)) < 0.02
    assert {d["sex"] for d in pool} == {"F", "M"}


def test_filter_keeps_only_eligible():
    donors = [_donor("ok"), _donor("recent", days_since=10), _donor("wrong", blood_type="B+")]
    assert [d["donor_id"] for d in filter_eligible_donors(donors, "O-")] == ["ok"]


def test_select_top_donors_returns_k_from_the_pool():
    pool = [_donor(f"d{i}") for i in range(10)]
    chosen = select_top_donors(pool, k=3)
    assert len(chosen) == 3
    assert all(d in pool for d in chosen)


def test_thompson_sampling_prefers_reliable_responders():
    # Exploitation: a donor who answered 19/20 alerts should be picked
    # far more often than one who answered 1/20.
    random.seed(0)
    reliable = _donor("reliable", sent=20, responded=19)
    unreliable = _donor("unreliable", sent=20, responded=1)
    wins = sum(select_top_donors([reliable, unreliable], k=1)[0] is reliable for _ in range(500))
    assert wins > 475


def test_untested_donors_still_get_explored():
    # Exploration: a never-contacted donor (uniform prior) must still win
    # sometimes against a mediocre track record.
    random.seed(0)
    veteran = _donor("veteran", sent=20, responded=10)
    newcomer = _donor("new")
    wins = sum(select_top_donors([veteran, newcomer], k=1)[0] is newcomer for _ in range(500))
    assert 100 < wins < 400


def test_update_donor_response_records_outcome():
    donors = [_donor("a", sent=2, responded=1)]
    update_donor_response(donors, "a", responded=True)
    assert (donors[0]["alerts_sent"], donors[0]["alerts_responded"]) == (3, 2)
    update_donor_response(donors, "a", responded=False)
    assert (donors[0]["alerts_sent"], donors[0]["alerts_responded"]) == (4, 2)
