"""
HemoSmart - Donor Module (Multi-Armed Bandit Donor Selection)
--------------------------------------------------------------------
Implements the Donor Module as specified in the project report
(Section 1.5, point 6 and Section 4.3.6):

  "A Multi-Armed Bandit model selects the most eligible and responsive
   donors based on donation history and response behavior."

WHY A MULTI-ARMED BANDIT, NOT FULL RL:
A full RL agent (e.g. deep Q-learning) needs a large volume of
interaction data to learn a good policy -- far more than a real donor
pool or a class project could realistically generate. A Multi-Armed
Bandit is a simpler, well-established RL technique suited to EXACTLY
this problem: repeatedly choosing which donor(s) to contact ("arms"),
observing a reward (did they respond?), and updating beliefs about
each donor's response likelihood over time. It converges with far
less data and is easy to verify is behaving correctly, which matters
for a system involving real people being contacted.

ALGORITHM: Thompson Sampling (Beta-Bernoulli bandit)
  - Each donor has a Beta(alpha, beta) distribution representing our
    belief about their probability of responding to an alert.
  - alpha = 1 + number of times they responded
  - beta  = 1 + number of times they did NOT respond
  - To choose who to alert, we SAMPLE a value from each eligible
    donor's Beta distribution and pick the highest sample(s).
  - This naturally balances exploration (untested donors have wide,
    uncertain distributions and can still win) and exploitation
    (donors with a strong track record tend to score higher on
    average) without needing to hand-tune an explore/exploit ratio.

WHAT'S SIMULATED VS REAL:
  - Donor records (name, phone, blood type, donation history) are
    simulated -- a real deployment would pull this from the hospital's
    donor database (Person C's PostgreSQL layer).
  - Sending the actual SMS/WhatsApp alert is a print/log stub here --
    real dispatch needs a paid API (Twilio, WhatsApp Business API) per
    the report's MCP notification service design (Section 3.8.5). This
    keeps the project's zero-cost constraint intact while the selection
    LOGIC itself is fully real and testable.
  - The "response" (did the donor respond) is simulated for testing,
    since we have no real alert history yet. In production, this would
    be recorded from actual donor replies and fed back to update the
    bandit -- exactly as this module is designed to support.
"""

import json
import os
import random
from datetime import datetime, timedelta

DONOR_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "donors.json")
MIN_DONATION_INTERVAL_DAYS = 90  # standard eligibility gap between donations
BLOOD_TYPES = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]


# ----------------------------------------------------------------------
# Donor data simulation (stands in for Person C's PostgreSQL donor table)
# ----------------------------------------------------------------------
def simulate_donors(n=50, seed=42):
    """
    Generates a realistic simulated donor pool: blood type, last donation
    date, and prior alert/response history (so the bandit has something
    to learn from immediately, rather than starting completely blank).
    """
    random.seed(seed)
    donors = []
    today = datetime(2026, 8, 29)

    for i in range(n):
        blood_type = random.choice(BLOOD_TYPES)
        days_since_donation = random.randint(0, 300)
        last_donation = (today - timedelta(days=days_since_donation)).strftime("%Y-%m-%d")

        # Simulate prior history: some donors are naturally more
        # responsive than others (mirrors real-world variation), so the
        # bandit has a genuine pattern to discover, not just noise.
        true_response_rate = random.betavariate(2, 2)  # varies 0-1, centered ~0.5
        prior_alerts = random.randint(0, 8)
        prior_responses = sum(
            1 for _ in range(prior_alerts) if random.random() < true_response_rate
        )

        donors.append({
            "donor_id": f"D{i+1:04d}",
            "name": f"Donor {i+1}",
            "phone": f"+91XXXXX{1000+i:05d}",  # masked for privacy in logs
            "blood_type": blood_type,
            "last_donation_date": last_donation,
            "alerts_sent": prior_alerts,
            "alerts_responded": prior_responses,
        })

    return donors


def load_donors():
    if os.path.exists(DONOR_FILE):
        with open(DONOR_FILE, "r") as f:
            return json.load(f)
    donors = simulate_donors()
    save_donors(donors)
    return donors


def save_donors(donors):
    with open(DONOR_FILE, "w") as f:
        json.dump(donors, f, indent=2)


# ----------------------------------------------------------------------
# Eligibility filtering
# ----------------------------------------------------------------------
def is_eligible(donor: dict, blood_type: str, today: datetime = None) -> bool:
    """A donor is eligible if their blood type matches and enough time
    has passed since their last donation."""
    if today is None:
        today = datetime(2026, 8, 29)

    if donor["blood_type"] != blood_type:
        return False

    last_donation = datetime.strptime(donor["last_donation_date"], "%Y-%m-%d")
    days_since = (today - last_donation).days
    return days_since >= MIN_DONATION_INTERVAL_DAYS


def filter_eligible_donors(donors: list, blood_type: str) -> list:
    return [d for d in donors if is_eligible(d, blood_type)]


# ----------------------------------------------------------------------
# Multi-Armed Bandit (Thompson Sampling)
# ----------------------------------------------------------------------
def thompson_sample_score(donor: dict) -> float:
    """
    Samples a value from this donor's Beta(alpha, beta) distribution.
    alpha/beta start at 1 (uniform prior) and are shifted by their
    actual alert/response history -- donors with more responses relative
    to alerts sent will tend to sample higher, but the randomness means
    a rarely-contacted donor can still win occasionally (exploration).
    """
    alpha = 1 + donor["alerts_responded"]
    beta = 1 + (donor["alerts_sent"] - donor["alerts_responded"])
    return random.betavariate(alpha, beta)


def select_top_donors(eligible_donors: list, k: int = 3) -> list:
    """Ranks eligible donors by a fresh Thompson sample each call and
    returns the top k. Re-sampling each call is intentional -- it's
    what makes this a bandit rather than a static leaderboard."""
    scored = [(d, thompson_sample_score(d)) for d in eligible_donors]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _score in scored[:k]]


def update_donor_response(donors: list, donor_id: str, responded: bool) -> list:
    """
    Feeds a real outcome back into the bandit -- call this once a
    donor's actual response (or non-response) is known, so future
    selections improve over time. This is the "learning" part of the
    Multi-Armed Bandit.
    """
    for d in donors:
        if d["donor_id"] == donor_id:
            d["alerts_sent"] += 1
            if responded:
                d["alerts_responded"] += 1
            break
    return donors


# ----------------------------------------------------------------------
# Public entry point: intelligent, personalized donor alerting
# ----------------------------------------------------------------------
def send_intelligent_donor_alerts(blood_type: str, k: int = 3) -> dict:
    """
    Full pipeline: load donors -> filter eligible -> bandit-select top k
    -> "send" personalized alerts (logged/simulated) -> return summary.
    """
    donors = load_donors()
    eligible = filter_eligible_donors(donors, blood_type)

    if not eligible:
        return {
            "blood_type": blood_type,
            "alerted_donors": [],
            "message": f"No eligible donors found for blood type {blood_type}.",
        }

    selected = select_top_donors(eligible, k=min(k, len(eligible)))

    alerted = []
    for donor in selected:
        message = (
            f"[DONOR ALERT] To {donor['name']} ({donor['phone']}), "
            f"blood type {blood_type}: Your donation is urgently needed. "
            f"Please visit the nearest blood bank."
        )
        print(message)
        alerted.append({
            "donor_id": donor["donor_id"],
            "name": donor["name"],
            "phone": donor["phone"],
            "prior_response_rate": (
                round(donor["alerts_responded"] / donor["alerts_sent"], 2)
                if donor["alerts_sent"] > 0 else None
            ),
        })

    save_donors(donors)

    return {
        "blood_type": blood_type,
        "eligible_count": len(eligible),
        "alerted_donors": alerted,
        "message": f"Alerted {len(alerted)} donor(s) for blood type {blood_type}.",
    }
