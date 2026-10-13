"""
Seed / reset script -- puts the database into a known, demo-ready state.

Run from the repo root:

    python -m scripts.seed                 # idempotent: only adds what's missing
    python -m scripts.seed --reset --yes   # WIPES all data, then re-seeds

It reuses the app's own code paths rather than inserting rows by hand:
donors come from the same loader that auto-seeds on first request
(agents/donor_module.py), blood units are received through
bloodbank/stock.py, and the sample patients go through the real models
and persist_prediction(), so the seeded state is exactly what the app
itself would have produced.

The inventory is generated relative to NOW (unit ages spread across each
component's shelf life) with a few deliberate situations for the
autonomous loop to find on its first cycle: a critical O- red cell
shortage, thin A+ platelet cover, and an O+/B+ platelet surplus close to
expiry.

Points at whatever DATABASE_URL is set to (backend/.env is loaded, same
as the API) -- including a deployed Neon database. --reset therefore
prints the target host and refuses to run without --yes.
"""

import argparse
import os
import random
import sys
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend", ".env"))

import pandas as pd

from agents.donor_module import load_donors
from agents.tools import predict_transfusion
from backend.services.auth import hash_password
from backend.services.bootstrap import ensure_admin_user
from backend.services.persistence import persist_prediction
from bloodbank.catalog import BLOOD_GROUPS, COMPONENTS, SHELF_LIFE_DAYS
from bloodbank.forecast import par_level
from bloodbank.stock import receive_units, utcnow
from db.database import SessionLocal, engine
from db.models import (
    AgentCycle,
    AuditLog,
    BloodUnit,
    Donor,
    DonorAlert,
    ForecastRun,
    Patient,
    Prediction,
    Transfusion,
    User,
)
from schema.patient_schema import PatientRecord

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# One account per non-Admin role; Admin comes from ADMIN_EMAIL/ADMIN_PASSWORD.
DEMO_USERS = {
    "manager@hemosmart.app": "Blood Bank Manager",
    "staff@hemosmart.app": "Hospital Staff",
    "coordinator@hemosmart.app": "Donor Coordinator",
    "auditor@hemosmart.app": "Auditor",
}
DEFAULT_DEMO_PASSWORD = "demo-pass-123"

N_SAMPLE_PATIENTS = 8  # half transfusion-needed, half not, so both outcomes show


def _wipe() -> None:
    """Deletes every row, children before parents (FK order)."""
    with SessionLocal() as db:
        for model in (DonorAlert, BloodUnit, Transfusion, Prediction, AgentCycle, AuditLog,
                      ForecastRun, Patient, Donor, User):
            db.query(model).delete()
        db.commit()


def _seed_demo_users() -> None:
    password = os.environ.get("DEMO_PASSWORD", DEFAULT_DEMO_PASSWORD)
    with SessionLocal() as db:
        for email, role in DEMO_USERS.items():
            if not db.query(User).filter(User.email == email).first():
                db.add(User(email=email, hashed_password=hash_password(password), role=role))
        db.commit()


# Stock as a multiple of par, per component, before the scenarios below
STOCK_FACTOR = (0.9, 1.6)
SCENARIOS = {
    ("O-", "PRBC"): 1,        # critical: one unit left
    ("A+", "Platelets"): 1,   # thin cover going into the week
}
NEAR_EXPIRY_SURPLUS = {("O+", "Platelets"): 5, ("B+", "Platelets"): 4}  # ~4.3 days old


def _seed_inventory(now=None, rng: random.Random = None) -> int:
    """Also used by evaluation/simulate.py to give every policy the same
    starting stock, dated relative to the simulation's start."""
    with SessionLocal() as db:
        if db.query(BloodUnit).count():
            return 0

        rng = rng or random.Random(2026)
        now = now or utcnow()
        added = 0
        for group in BLOOD_GROUPS:
            for component in COMPONENTS:
                count = SCENARIOS.get((group, component))
                if count is None:
                    count = round(par_level(group, component) * rng.uniform(*STOCK_FACTOR))
                shelf = SHELF_LIFE_DAYS[component]
                for _ in range(count):
                    # Ages across the first ~90% of shelf life -- nothing already expired
                    age = timedelta(days=rng.uniform(0, shelf * 0.9))
                    receive_units(db, group, component, 1, now - age, source="seed")
                    added += 1
        for (group, component), count in NEAR_EXPIRY_SURPLUS.items():
            receive_units(db, group, component, count, now - timedelta(days=4.3), source="seed")
            added += count
        db.commit()
        return added


GROUP_WEIGHTS = (0.21, 0.01, 0.31, 0.015, 0.36, 0.02, 0.07, 0.005)  # same order as BLOOD_GROUPS


def _seed_sample_patients() -> int:
    with SessionLocal() as db:
        if db.query(Patient).count():
            return 0

    df = pd.read_csv(os.path.join(REPO_ROOT, "data", "patient_data.csv"))
    half = N_SAMPLE_PATIENTS // 2
    sample = pd.concat([
        df[df.transfusion_needed == 1].sample(half, random_state=7),
        df[df.transfusion_needed == 0].sample(half, random_state=7),
    ])

    rng = random.Random(7)
    with SessionLocal() as db:
        for row in sample.itertuples():
            record = PatientRecord(
                blood_group=rng.choices(BLOOD_GROUPS, weights=GROUP_WEIGHTS)[0],
                hemoglobin=row.hemoglobin,
                platelets=int(row.platelets),
                INR=row.INR,
                age=int(row.age),
                surgery_type=row.surgery_type,
                source_format="seed",
                source_hospital="Demo Hospital",
            )
            persist_prediction(db, record, predict_transfusion(record))
    return len(sample)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset", action="store_true", help="wipe ALL data before seeding")
    parser.add_argument("--yes", action="store_true", help="confirm --reset")
    args = parser.parse_args()

    target = engine.url
    print(f"Target database: {target.host or 'local socket'}/{target.database}")

    if args.reset:
        if not args.yes:
            sys.exit("--reset deletes every row in every table. Re-run with --yes to confirm.")
        _wipe()
        print("Wiped all tables.")

    ensure_admin_user()
    _seed_demo_users()
    load_donors()       # seeds the simulated donor pool if the table is empty
    n_units = _seed_inventory()
    n_patients = _seed_sample_patients()

    with SessionLocal() as db:
        counts = {
            "users": db.query(User).count(),
            "donors": db.query(Donor).count(),
            "blood units": db.query(BloodUnit).count(),
            "patients": db.query(Patient).count(),
            "predictions": db.query(Prediction).count(),
        }
    print(f"Seeded (added {n_units} blood units, {n_patients} sample patients this run). Current row counts:")
    for name, n in counts.items():
        print(f"  {name}: {n}")

    password = os.environ.get("DEMO_PASSWORD", DEFAULT_DEMO_PASSWORD)
    print(f"\nDemo logins (password: {password}):")
    print(f"  Admin: {os.environ.get('ADMIN_EMAIL', 'admin@hemosmart.local')} (password from ADMIN_PASSWORD)")
    for email, role in DEMO_USERS.items():
        print(f"  {role}: {email}")

    is_local = target.host in (None, "", "localhost", "127.0.0.1")
    if not is_local and password == DEFAULT_DEMO_PASSWORD:
        print("\nWARNING: this is a remote database and DEMO_PASSWORD is the public default. "
              "Set DEMO_PASSWORD before sharing the URL.")


if __name__ == "__main__":
    main()
