"""
HemoSmart - Response-Time Measurement (NFR-01)
--------------------------------------------------
The report's performance requirements, measured end to end through the
real FastAPI app (auth, validation, model, database write, audit log),
in-process with FastAPI's TestClient -- so network time between a
browser and the server is NOT included.

  NFR-01 target                          what is timed
  prediction            < 2 s            POST /api/predict
  RAG explanation       < 5 s            POST /api/explain (FAISS + 1 Groq LLM call)
  7-day forecast        < 10 s           GET /api/forecast (Prophet + gap projection), uncached
  dashboard load        < 3 s            not measured here (browser); see the walkthrough notes

Run (from the repo root):
    python -m evaluation.latency            # 30 predictions, 10 forecasts, 5 explanations
    python -m evaluation.latency --no-llm   # skip the explanation (no Groq calls)

Writes evaluation/results/latency.md and .json.
"""

import argparse
import json
import os
import statistics
import time
from datetime import datetime

import evaluation  # noqa: F401  -- throwaway database first

os.environ["HEMOSMART_AUTONOMY_INTERVAL_MINUTES"] = "0"  # no background cycles while timing

from dotenv import load_dotenv  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(REPO_ROOT, "backend", ".env"))  # GROQ_API_KEY; doesn't override DATABASE_URL

from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402
from backend.services.auth import create_access_token, hash_password  # noqa: E402
from db.database import SessionLocal  # noqa: E402
from db.models import User  # noqa: E402
from scripts.seed import _seed_inventory  # noqa: E402

RESULTS_DIR = os.path.join(REPO_ROOT, "evaluation", "results")
PATIENTS = [
    dict(hemoglobin=6.8, platelets=210_000, INR=1.1, age=64, surgery_type="General", blood_group="O+"),
    dict(hemoglobin=12.9, platelets=32_000, INR=1.0, age=41, surgery_type="Orthopedic", blood_group="A+"),
    dict(hemoglobin=7.2, platelets=45_000, INR=1.9, age=35, surgery_type="Emergency", blood_group="O-"),
    dict(hemoglobin=13.4, platelets=240_000, INR=1.0, age=50, surgery_type="Cardiac", blood_group="B+"),
]
TARGETS = {"prediction": 2.0, "explanation": 5.0, "forecast": 10.0}


def _token() -> str:
    email = "latency@hemosmart.app"
    with SessionLocal() as db:
        if not db.query(User).filter(User.email == email).first():
            db.add(User(email=email, hashed_password=hash_password("x"), role="Blood Bank Manager"))
            db.commit()
    return create_access_token(email, "Blood Bank Manager")


def _time(fn, n: int) -> list:
    out = []
    for i in range(n):
        t = time.perf_counter()
        r = fn(i)
        out.append(time.perf_counter() - t)
        assert r.status_code == 200, f"{r.status_code}: {r.text[:200]}"
    return out


def _summary(samples: list, target: float) -> dict:
    s = sorted(samples)
    p95 = s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]
    return {"n": len(s), "median_s": round(statistics.median(s), 3), "p95_s": round(p95, 3),
            "max_s": round(s[-1], 3), "target_s": target, "meets_target": p95 < target}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-llm", action="store_true", help="skip the RAG explanation (no Groq calls)")
    parser.add_argument("--predictions", type=int, default=30)
    parser.add_argument("--forecasts", type=int, default=10)
    parser.add_argument("--explanations", type=int, default=5)
    args = parser.parse_args()

    _seed_inventory()  # realistic stock for the gap projection
    headers = {"Authorization": f"Bearer {_token()}"}
    results = {}
    with TestClient(app) as client:  # runs the lifespan: models and RAG loaded once, like production
        client.post("/api/predict", json=PATIENTS[0], headers=headers)  # warm-up, not timed

        results["prediction"] = _summary(_time(
            lambda i: client.post("/api/predict", json=PATIENTS[i % len(PATIENTS)], headers=headers),
            args.predictions), TARGETS["prediction"])
        print("prediction", results["prediction"], flush=True)

        # A distinct horizon per call (7, 8, ... days) so the Redis forecast
        # cache, if running, never serves a repeat: every call runs Prophet.
        results["forecast"] = _summary(_time(
            lambda i: client.get(f"/api/forecast?days={min(7 + i, 30)}", headers=headers),
            args.forecasts), TARGETS["forecast"])
        print("forecast", results["forecast"], flush=True)

        if not args.no_llm:
            def explain(i):
                patient = PATIENTS[i % len(PATIENTS)]
                return client.post("/api/explain", json={"patient": patient, "prediction": True}, headers=headers)
            samples = []
            for i in range(args.explanations):
                samples += _time(lambda _: explain(i), 1)
                time.sleep(8)  # stay under Groq's free-tier tokens/minute, so we time the call, not a rate-limit wait
            results["explanation"] = _summary(samples, TARGETS["explanation"])
            print("explanation", results["explanation"], flush=True)

    meta = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "llm": "openai/gpt-oss-20b via Groq" if not args.no_llm else None}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "latency.json"), "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    labels = {"prediction": "Prediction (POST /api/predict)",
              "forecast": "Forecast + gap, uncached, 7–16 days (GET /api/forecast)",
              "explanation": "RAG explanation (POST /api/explain, 1 LLM call)"}
    lines = [
        "# HemoSmart — Response Times (NFR-01)",
        "",
        f"In-process through the FastAPI app (auth, validation, models, DB write, audit log); browser↔server "
        f"network time not included. Generated by `python -m evaluation.latency` on {meta['generated_at']}.",
        "",
        "| Operation | n | Median | p95 | Max | NFR-01 target | Met (p95) |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, r in results.items():
        lines.append(f"| {labels[key]} | {r['n']} | {r['median_s']:.2f} s | {r['p95_s']:.2f} s | {r['max_s']:.2f} s | "
                     f"< {r['target_s']:.0f} s | {'yes' if r['meets_target'] else 'NO'} |")
    lines += ["", "The explanation time is dominated by the Groq API round trip. With few samples the p95 is the "
                  "slowest call, and on Groq's free tier an occasional call waits several seconds in their queue or "
                  "on a rate limit -- so the median is the typical experience and the p95 is the free-tier tail. "
                  "Dashboard load (< 3 s) is a browser measurement and isn't covered here."]
    with open(os.path.join(RESULTS_DIR, "latency.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
