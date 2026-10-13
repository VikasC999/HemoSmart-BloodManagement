"""
HemoSmart - Re-score Existing Patients
------------------------------------------
After the models are retrained, stored predictions still come from the
old models. This adds a NEW prediction from the current models to each
patient. Nothing is deleted or overwritten: the old predictions stay in
the patient's history, and the newest one becomes the "latest" the
dashboard shows. A non-destructive alternative to `seed --reset`.

By default only patients whose latest prediction predates per-component
models (no `components`) are re-scored; --all re-scores everyone.

Run (from the repo root; targets DATABASE_URL like the app):
    python -m scripts.rescore            # stale predictions only
    python -m scripts.rescore --all      # every patient
    python -m scripts.rescore --dry-run  # show what would change
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(REPO_ROOT, "backend", ".env"))

from agents.tools import predict_transfusion  # noqa: E402
from db.database import DATABASE_URL, SessionLocal  # noqa: E402
from db.models import Patient, Prediction  # noqa: E402
from schema.patient_schema import PatientRecord  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="re-score every patient, not just stale ones")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Database: {DATABASE_URL.split('@')[-1]}")
    added = skipped = failed = 0
    with SessionLocal() as db:
        for p in db.query(Patient).order_by(Patient.id).all():
            latest = p.predictions[-1] if p.predictions else None
            if not args.all and latest is not None and latest.components is not None:
                skipped += 1
                continue
            try:
                record = PatientRecord(
                    hemoglobin=p.hemoglobin, platelets=p.platelets, INR=p.inr, age=p.age,
                    surgery_type=p.surgery_type, blood_group=p.blood_group,
                    source_format=p.source_format or "unknown", source_hospital=p.source_hospital or "unknown",
                )
            except Exception as exc:  # a row the current schema rejects -- report, don't guess
                print(f"  patient {p.id}: skipped, invalid under current schema ({exc.__class__.__name__})")
                failed += 1
                continue
            result = predict_transfusion(record)
            before = f"{'yes' if latest and latest.transfusion_needed else 'no'}" if latest else "—"
            needed = ", ".join(c["component"] for c in result["components"] if c["needed"]) or "none"
            print(f"  patient {p.id}: was {before} -> now {'yes' if result['transfusion_needed'] else 'no'} "
                  f"({needed}, {result['urgency']})")
            if not args.dry_run:
                db.add(Prediction(
                    patient_id=p.id, transfusion_needed=result["transfusion_needed"],
                    confidence=result["confidence"], components=result["components"],
                    urgency=result["urgency"], shap=result["shap"],
                ))
            added += 1
        if not args.dry_run:
            db.commit()
    verb = "Would add" if args.dry_run else "Added"
    print(f"{verb} {added} prediction(s); {skipped} already current; {failed} skipped as invalid.")


if __name__ == "__main__":
    main()
