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
  - Responses: in live use a Donor Coordinator records each reply
    (record_donor_response). With HEMOSMART_SIMULATE_DONOR_RESPONSES on,
    the autonomous loop instead samples replies from each synthetic
    donor's hidden `simulated_response_rate` -- a value the bandit never
    reads, so it still has to learn who responds, from outcomes alone.

THE FEEDBACK LOOP:
  alert sent        -> alerts_sent += 1              (Beta beta grows)
  donor responds    -> alerts_responded += 1         (Beta alpha grows)
                    -> donation: new units enter stock, eligibility clock resets
  no reply in 48h   -> counted as a non-response
"""

import os
import random
import sys
from datetime import date, datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bloodbank.catalog import BLOOD_GROUPS, DONATION_YIELD
from db.database import SessionLocal
from db.models import Donor as DonorRow
from db.models import DonorAlert
from schema.privacy import mask_identifier

MIN_DONATION_INTERVAL_DAYS = 90   # whole blood after whole blood (red cells take ~3 months to recover)
APHERESIS_INTERVAL_DAYS = 28      # any donation after apheresis, and apheresis itself -- configurable policy
RESPONSE_WINDOW_HOURS = 48        # no reply by then counts as "didn't respond"


def _utcnow() -> datetime:
    from bloodbank.stock import utcnow
    return utcnow()


# ----------------------------------------------------------------------
# Donor data simulation (stands in for a hospital's donor registry)
# ----------------------------------------------------------------------
def simulate_donors(n=400, seed=42, today: datetime = None):
    """
    Generates a realistic simulated donor pool: blood type, last donation
    date, prior alert/response history (so the bandit has something to
    learn from immediately), and each donor's hidden true response rate.
    """
    rng = random.Random(seed)  # own RNG: seeding must not reset the bandit's global sampler
    donors = []
    today = today or datetime.now()

    for i in range(n):
        blood_type = rng.choice(BLOOD_GROUPS)
        days_since_donation = rng.randint(0, 300)
        last_donation = (today - timedelta(days=days_since_donation)).strftime("%Y-%m-%d")

        # Some donors are naturally more responsive than others (mirrors
        # real-world variation), so the bandit has a genuine pattern to
        # discover, not just noise.
        true_response_rate = rng.betavariate(2, 2)  # varies 0-1, centered ~0.5
        prior_alerts = rng.randint(0, 8)
        prior_responses = sum(
            1 for _ in range(prior_alerts) if rng.random() < true_response_rate
        )

        donors.append({
            "donor_id": f"D{i+1:04d}",
            "name": f"Donor {i+1}",
            "phone": f"+91XXXXX{1000+i:05d}",  # masked for privacy in logs
            "blood_type": blood_type,
            "last_donation_date": last_donation,
            "alerts_sent": prior_alerts,
            "alerts_responded": prior_responses,
            "simulated_response_rate": round(true_response_rate, 4),
        })

    return donors


def _row_to_dict(row: DonorRow) -> dict:
    return {
        "donor_id": row.donor_id,
        "name": row.name,
        "phone": row.phone,
        "blood_type": row.blood_type,
        "last_donation_date": row.last_donation_date.strftime("%Y-%m-%d"),
        "alerts_sent": row.alerts_sent,
        "alerts_responded": row.alerts_responded,
        "last_donation_type": row.last_donation_type,
    }


def ensure_donor_pool(db, today: datetime = None) -> None:
    """Seeds the simulated donor pool the first time the table is empty."""
    if db.query(DonorRow).first():
        return
    for d in simulate_donors(today=today):
        db.add(DonorRow(
            donor_id=d["donor_id"], name=d["name"], phone=d["phone"],
            blood_type=d["blood_type"],
            last_donation_date=datetime.strptime(d["last_donation_date"], "%Y-%m-%d").date(),
            alerts_sent=d["alerts_sent"], alerts_responded=d["alerts_responded"],
            simulated_response_rate=d["simulated_response_rate"],
        ))
    db.flush()


def load_donors():
    """Reads the donor pool from Postgres, seeding it on first use."""
    with SessionLocal() as session:
        ensure_donor_pool(session)
        session.commit()
        return [_row_to_dict(r) for r in session.query(DonorRow).all()]


# ----------------------------------------------------------------------
# Eligibility filtering
# ----------------------------------------------------------------------
def required_interval_days(component: str, last_donation_type: str = None) -> int:
    """
    Days a donor must wait since their last donation before donating for
    `component` (whole blood for PRBC/FFP, apheresis for platelets).
    The long whole-blood gap exists because whole blood removes red
    cells; apheresis returns them to the donor, so after an apheresis
    donation -- and for apheresis itself -- the shorter interval applies.
    Treating apheresis as blocking whole blood for 90 days starved red
    cell outreach in simulation (docs/DESIGN_DECISIONS.md D47).
    """
    if component == "Platelets" or last_donation_type == "apheresis":
        return APHERESIS_INTERVAL_DAYS
    return MIN_DONATION_INTERVAL_DAYS


def is_eligible(donor: dict, blood_type: str, today: datetime = None, component: str = "PRBC") -> bool:
    """A donor is eligible if their blood type matches and enough time has
    passed since their last donation (see required_interval_days)."""
    if today is None:
        today = datetime.now()

    if donor["blood_type"] != blood_type:
        return False

    interval = required_interval_days(component, donor.get("last_donation_type"))
    last_donation = datetime.strptime(donor["last_donation_date"], "%Y-%m-%d")
    return (today - last_donation).days >= interval


def filter_eligible_donors(donors: list, blood_type: str, today: datetime = None, component: str = "PRBC") -> list:
    return [d for d in donors if is_eligible(d, blood_type, today, component)]


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


def posterior_mean(donor: dict) -> float:
    """Expected response probability under the donor's Beta posterior."""
    return (1 + donor["alerts_responded"]) / (2 + donor["alerts_sent"])


def select_top_donors(eligible_donors: list, k: int = 3) -> list:
    """Ranks eligible donors by a fresh Thompson sample each call and
    returns the top k. Re-sampling each call is intentional -- it's
    what makes this a bandit rather than a static leaderboard."""
    scored = [(d, thompson_sample_score(d)) for d in eligible_donors]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _score in scored[:k]]


def update_donor_response(donors: list, donor_id: str, responded: bool) -> list:
    """
    In-memory form of the bandit update (one alert, one outcome) -- the
    DB-backed path below applies the same update in two halves: the
    alert at send time, the outcome when it's known.
    """
    for d in donors:
        if d["donor_id"] == donor_id:
            d["alerts_sent"] += 1
            if responded:
                d["alerts_responded"] += 1
            break
    return donors


# ----------------------------------------------------------------------
# Alerting (DB-backed)
# ----------------------------------------------------------------------
def pending_alerts(db, blood_type: str = None, component: str = None) -> list:
    q = db.query(DonorAlert).filter(DonorAlert.responded.is_(None))
    if blood_type:
        q = q.filter(DonorAlert.blood_type == blood_type)
    if component:
        q = q.filter(DonorAlert.component == component)
    return q.all()


def eligible_candidates(db, blood_type: str, component: str, now: datetime) -> list:
    """Eligible donors of this type with no alert already awaiting a reply
    -- nobody gets a second message while the first is unanswered."""
    ensure_donor_pool(db)
    busy = {a.donor_id for a in pending_alerts(db)}
    rows = db.query(DonorRow).filter(DonorRow.blood_type == blood_type).all()
    donors = [_row_to_dict(r) for r in rows if r.donor_id not in busy]
    return filter_eligible_donors(donors, blood_type, now, component)


def alert_donors(db, blood_type: str, component: str, k: int, now: datetime, cycle_id: int = None,
                 selector=None) -> dict:
    """Selects up to k donors (Thompson-Sampling bandit by default; the
    evaluation passes a random selector as the no-learning baseline) and
    records each alert. The caller owns the transaction."""
    selector = selector or select_top_donors
    eligible = eligible_candidates(db, blood_type, component, now)
    if not eligible:
        return {
            "blood_type": blood_type, "component": component, "eligible_count": 0,
            "alerted_donors": [],
            "message": f"No eligible donors found for {blood_type} {component}.",
        }

    selected = selector(eligible, k=min(k, len(eligible)))
    donation = "platelet apheresis" if component == "Platelets" else "whole blood"
    rows = {r.donor_id: r for r in db.query(DonorRow).filter(
        DonorRow.donor_id.in_([d["donor_id"] for d in selected])
    )}

    alerted = []
    for donor in selected:
        print(
            f"[DONOR ALERT] To {donor['name']} ({mask_identifier(donor['phone'])}), blood type {blood_type}: "
            f"a {donation} donation is urgently needed. Please visit the nearest blood bank."
        )
        rows[donor["donor_id"]].alerts_sent += 1  # bandit: one more trial
        alert = DonorAlert(
            donor_id=donor["donor_id"], blood_type=blood_type, component=component,
            method="log", sent_at=now, cycle_id=cycle_id,
        )
        db.add(alert)
        db.flush()
        alerted.append({
            "alert_id": alert.id,
            "donor_id": donor["donor_id"],
            "name": donor["name"],
            "phone": mask_identifier(donor["phone"]),
            "expected_response_rate": round(posterior_mean(donor), 2),
        })

    return {
        "blood_type": blood_type,
        "component": component,
        "eligible_count": len(eligible),
        "alerted_donors": alerted,
        "message": f"Alerted {len(alerted)} donor(s) for {blood_type} {component} ({donation}).",
    }


def send_intelligent_donor_alerts(blood_type: str, component: str = "PRBC", k: int = 3) -> dict:
    """Standalone entry point (REST, chat, MCP): one transaction, real time."""
    with SessionLocal() as db:
        result = alert_donors(db, blood_type, component, k, _utcnow())
        db.commit()
        return result


def record_donor_response(db, alert_id: int, responded: bool, now: datetime) -> dict:
    """
    Feeds a real outcome back into the bandit -- the "learning" half of
    the Multi-Armed Bandit. A donor who responds also donates: the
    donation's units enter stock (DONATION_YIELD) and their eligibility
    clock restarts.
    """
    from bloodbank.stock import receive_units

    alert = db.query(DonorAlert).filter(DonorAlert.id == alert_id).first()
    if alert is None:
        raise LookupError(f"No donor alert {alert_id}")
    if alert.responded is not None:
        raise ValueError(f"Alert {alert_id} already resolved")

    alert.responded, alert.responded_at = responded, now
    units = []
    if responded:
        donor = db.query(DonorRow).filter(DonorRow.donor_id == alert.donor_id).first()
        donor.alerts_responded += 1
        donor.last_donation_date = now.date()
        donor.last_donation_type = "apheresis" if alert.component == "Platelets" else "whole_blood"
        for component in DONATION_YIELD[alert.component or "PRBC"]:
            units += receive_units(db, alert.blood_type, component, 1, now,
                                   source="donation", donor_id=donor.donor_id)
    # Sessions don't autoflush: without this, a later pending_alerts()
    # query in the same transaction would still see this alert as pending.
    db.flush()

    return {
        "alert_id": alert_id, "donor_id": alert.donor_id, "responded": responded,
        "units_received": [u.unit_id for u in units],
    }


def expire_unanswered_alerts(db, now: datetime) -> list:
    """No reply within RESPONSE_WINDOW_HOURS counts as a non-response."""
    cutoff = now - timedelta(hours=RESPONSE_WINDOW_HOURS)
    stale = [a for a in pending_alerts(db) if a.sent_at <= cutoff]
    return [record_donor_response(db, a.id, False, now) for a in stale]


def simulate_responses(db, now: datetime, rng: random.Random, current_cycle_id: int = None) -> list:
    """SIMULATION ONLY: resolves alerts from earlier cycles by sampling each
    donor's hidden simulated_response_rate."""
    outcomes = []
    for alert in pending_alerts(db):
        if current_cycle_id is not None and alert.cycle_id == current_cycle_id:
            continue
        donor = db.query(DonorRow).filter(DonorRow.donor_id == alert.donor_id).first()
        rate = donor.simulated_response_rate if donor.simulated_response_rate is not None else 0.5
        outcomes.append(record_donor_response(db, alert.id, rng.random() < rate, now))
    return outcomes
