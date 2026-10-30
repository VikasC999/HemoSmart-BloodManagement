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
from bloodbank import catalog
from bloodbank.catalog import DONATION_YIELD
from data.demand_process import GROUP_SHARE
from db.database import SessionLocal
from db.models import Donor as DonorRow
from db.models import DonorAlert
from integrations.notify import get_notifier
from schema.privacy import mask_identifier

# Donation intervals, NBTC donor selection guidelines 2017 (docs/DESIGN_DECISIONS.md D83)
MIN_DONATION_INTERVAL_DAYS = 90         # whole blood after whole blood, men (and donors of unknown sex)
FEMALE_WHOLE_BLOOD_INTERVAL_DAYS = 120  # whole blood after whole blood, women
APHERESIS_INTERVAL_DAYS = 4             # apheresis after apheresis: NBTC says 48 h and at most 2 a week;
                                        # 4 whole days is the shortest gap that keeps both
APHERESIS_TO_WHOLE_BLOOD_DAYS = 28      # whole blood after platelet apheresis (NBTC)
WHOLE_BLOOD_TO_APHERESIS_DAYS = 28      # apheresis after whole blood: not in the sources, kept from D47
RESPONSE_WINDOW_HOURS = 48              # no reply by then counts as "didn't respond"

# Simulated donors only. Hidden reply rates ~ Beta(1.5, 5), mean 0.23: an assumption anchored to the
# UCI donors' 23.8% monthly donation rate (evaluation/results/donor_real_data.md), bracketed by 8%
# re-donation after recalling lapsed donors (Ou-Yang et al. 2020). Was Beta(2, 2), mean 0.5.
RESPONSE_RATE_PRIOR = (1.5, 5.0)
# Not sourced: the research found no Indian donor sex mix. An assumption, stated as one.
FEMALE_DONOR_SHARE = 0.10

# Simulated donors only: contact fatigue (docs/DESIGN_DECISIONS.md D85). Each earlier alert multiplies
# a donor's hidden reply rate by `decay_per_alert`, and its effect halves every `half_life_days`. No
# study measuring alert frequency against response was found, so both values are assumptions.
# None switches fatigue off (the D53/D80 results).
DONOR_FATIGUE = {"decay_per_alert": 0.7, "half_life_days": 14}
# Simulated donors only: new donors (no history) joining the pool, per year. An assumption (25% of
# the 400-donor pool); real registries also lose donors, which isn't modelled.
NEW_DONORS_PER_YEAR = 100


def fatigued_rate(base: float, alert_ages_days: list, fatigue: dict = None) -> float:
    """A donor's reply probability given the ages (days) of their earlier alerts."""
    fatigue = DONOR_FATIGUE if fatigue is None else fatigue
    if not fatigue:
        return base
    load = sum(0.5 ** (age / fatigue["half_life_days"]) for age in alert_ages_days)
    return base * fatigue["decay_per_alert"] ** load


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
        # Population ABO/RhD mix (Agrawal et al. 2014), not uniform: O- is ~2% of donors, not 12.5%.
        blood_type = rng.choices(list(GROUP_SHARE), weights=list(GROUP_SHARE.values()))[0]
        sex = "F" if rng.random() < FEMALE_DONOR_SHARE else "M"
        days_since_donation = rng.randint(0, 300)
        last_donation = (today - timedelta(days=days_since_donation)).strftime("%Y-%m-%d")

        # Some donors are naturally more responsive than others (mirrors
        # real-world variation), so the bandit has a genuine pattern to
        # discover, not just noise.
        true_response_rate = rng.betavariate(*RESPONSE_RATE_PRIOR)
        prior_alerts = rng.randint(0, 8)
        prior_responses = sum(
            1 for _ in range(prior_alerts) if rng.random() < true_response_rate
        )

        donors.append({
            "donor_id": f"D{i+1:04d}",
            "name": f"Donor {i+1}",
            "phone": f"+91XXXXX{1000+i:05d}",  # masked for privacy in logs
            "blood_type": blood_type,
            "sex": sex,
            "last_donation_date": last_donation,
            "alerts_sent": prior_alerts,
            "alerts_responded": prior_responses,
            "simulated_response_rate": round(true_response_rate, 4),
        })

    return donors


def arriving_donors(n: int, seed: int, today: datetime, prefix: str) -> list:
    """New registrations: like simulate_donors(), but no alert history and
    eligible at once. donor_id is f"{prefix}{i:02d}"."""
    out = simulate_donors(n=n, seed=seed, today=today)
    for i, d in enumerate(out):
        d.update(donor_id=f"{prefix}{i:02d}", name=f"New donor {prefix}{i:02d}", alerts_sent=0, alerts_responded=0,
                 last_donation_date=(today - timedelta(days=365)).strftime("%Y-%m-%d"))
    return out


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
        "sex": row.sex,
    }


def ensure_donor_pool(db, today: datetime = None) -> None:
    """Seeds the simulated donor pool the first time the table is empty."""
    if db.query(DonorRow).first():
        return
    add_donors(db, simulate_donors(today=today))


def add_donors(db, donors: list) -> None:
    for d in donors:
        db.add(DonorRow(
            donor_id=d["donor_id"], name=d["name"], phone=d["phone"],
            blood_type=d["blood_type"], sex=d.get("sex"),
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
def required_interval_days(component: str, last_donation_type: str = None, sex: str = None) -> int:
    """
    Days a donor must wait since their last donation before donating for
    `component` (whole blood for PRBC/FFP, apheresis for platelets).
    The long whole-blood gap exists because whole blood removes red
    cells; apheresis returns them to the donor, so after an apheresis
    donation the shorter interval applies (docs/DESIGN_DECISIONS.md D47,
    D83). An unknown last donation type counts as whole blood; an
    unknown sex (rows seeded before the sex column) gets the men's 90 days.
    """
    if component == "Platelets":
        return APHERESIS_INTERVAL_DAYS if last_donation_type == "apheresis" else WHOLE_BLOOD_TO_APHERESIS_DAYS
    if last_donation_type == "apheresis":
        return APHERESIS_TO_WHOLE_BLOOD_DAYS
    return FEMALE_WHOLE_BLOOD_INTERVAL_DAYS if sex == "F" else MIN_DONATION_INTERVAL_DAYS


def is_eligible(donor: dict, blood_type: str, today: datetime = None, component: str = "PRBC") -> bool:
    """A donor is eligible if their blood type matches and enough time has
    passed since their last donation (see required_interval_days)."""
    if today is None:
        today = datetime.now()

    if donor["blood_type"] != blood_type:
        return False

    interval = required_interval_days(component, donor.get("last_donation_type"), donor.get("sex"))
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
    -- nobody gets a second message while the first is unanswered -- and
    none alerted within catalog.CONTACT_COOLDOWN_DAYS (D85)."""
    ensure_donor_pool(db)
    busy = {a.donor_id for a in pending_alerts(db)}
    if catalog.CONTACT_COOLDOWN_DAYS:
        cutoff = now - timedelta(days=catalog.CONTACT_COOLDOWN_DAYS)
        busy |= {d for (d,) in db.query(DonorAlert.donor_id).filter(DonorAlert.sent_at > cutoff)}
    rows = db.query(DonorRow).filter(DonorRow.blood_type == blood_type).all()
    donors = [_row_to_dict(r) for r in rows if r.donor_id not in busy]
    return filter_eligible_donors(donors, blood_type, now, component)


def alert_donors(db, blood_type: str, component: str, k: int, now: datetime, cycle_id: int = None,
                 selector=None, notifier=None) -> dict:
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

    notifier = notifier or get_notifier()
    alerted = []
    for donor in selected:
        method = notifier.method
        try:
            notifier.send(donor["phone"], donor["name"],
                          f"Blood type {blood_type}: a {donation} donation is urgently needed. "
                          f"Please visit the nearest blood bank.")
        except Exception as exc:
            # The alert is still recorded, so the coordinator sees who was meant to be contacted.
            print(f"[DONOR ALERT] {notifier.method} delivery to {mask_identifier(donor['phone'])} failed "
                  f"({exc.__class__.__name__}); recorded as log only.")
            method = "log"
        rows[donor["donor_id"]].alerts_sent += 1  # bandit: one more trial
        alert = DonorAlert(
            donor_id=donor["donor_id"], blood_type=blood_type, component=component,
            method=method, sent_at=now, cycle_id=cycle_id,
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
        rate = (donor.simulated_response_rate if donor.simulated_response_rate is not None
                else RESPONSE_RATE_PRIOR[0] / sum(RESPONSE_RATE_PRIOR))
        if DONOR_FATIGUE:
            earlier = db.query(DonorAlert.sent_at).filter(
                DonorAlert.donor_id == donor.donor_id, DonorAlert.sent_at < alert.sent_at,
                DonorAlert.sent_at >= alert.sent_at - timedelta(days=6 * DONOR_FATIGUE["half_life_days"]))
            rate = fatigued_rate(rate, [(alert.sent_at - t).total_seconds() / 86400 for (t,) in earlier])
        outcomes.append(record_donor_response(db, alert.id, rng.random() < rate, now))
    return outcomes
