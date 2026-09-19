"""
Seed / reset script -- puts the database into a known, demo-ready state.

Run from the repo root:

    python -m scripts.seed                 # idempotent: only adds what's missing
    python -m scripts.seed --reset --yes   # WIPES all data, then re-seeds

It reuses the app's own code paths rather than inserting rows by hand:
donors/inventory come from the same loaders that auto-seed on first
request (agents/donor_module.py, agents/tools.py), and the sample
patients go through the real XGBoost model and persist_prediction(), so
the seeded state is exactly what the app itself would have produced.

Points at whatever DATABASE_URL is set to (backend/.env is loaded, same
as the API) -- including a deployed Neon database. --reset therefore
prints the target host and refuses to run without --yes.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend", ".env"))

import pandas as pd

from agents.donor_module import load_donors
from agents.tools import _load_inventory, predict_transfusion
from backend.services.auth import hash_password
from backend.services.bootstrap import ensure_admin_user
from backend.services.persistence import persist_prediction
from db.database import SessionLocal, engine
from db.models import (
    AuditLog,
    Donor,
    DonorAlert,
    ForecastRun,
    InventoryItem,
    Patient,
    Prediction,
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
        for model in (DonorAlert, Prediction, AuditLog, ForecastRun, Patient, Donor, InventoryItem, User):
            db.query(model).delete()
        db.commit()


def _seed_demo_users() -> None:
    password = os.environ.get("DEMO_PASSWORD", DEFAULT_DEMO_PASSWORD)
    with SessionLocal() as db:
        for email, role in DEMO_USERS.items():
            if not db.query(User).filter(User.email == email).first():
                db.add(User(email=email, hashed_password=hash_password(password), role=role))
        db.commit()


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

    with SessionLocal() as db:
        for row in sample.itertuples():
            record = PatientRecord(
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
    _load_inventory()   # seeds DEFAULT_INVENTORY if the table is empty
    n_patients = _seed_sample_patients()

    with SessionLocal() as db:
        counts = {
            "users": db.query(User).count(),
            "donors": db.query(Donor).count(),
            "inventory rows": db.query(InventoryItem).count(),
            "patients": db.query(Patient).count(),
            "predictions": db.query(Prediction).count(),
        }
    print("Seeded (added " + str(n_patients) + " sample patients this run). Current row counts:")
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
