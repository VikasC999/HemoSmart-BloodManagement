"""
Full-case processing by the CrewAI agent pipeline (agents/crew_agents.py):
a clinician uploads one CBC report PDF and six LLM agents -- Extraction,
Prediction, RAG, Inventory, Action, Orchestrator -- work it end to end,
each checked against an independently computed value.

    POST /api/cases          upload a PDF, start a job      -> {job_id}
    GET  /api/cases/{job_id}  poll per-agent progress/result

Why a background job: ~15-20 LLM calls make a run take tens of seconds
(docs/DESIGN_DECISIONS.md D40), too long for one HTTP request. Why one
run at a time: Groq's free tier caps calls per minute.

Jobs live in memory -- a restart loses in-flight job status, though a
finished case's patient + prediction are already saved to the database.
"""

import json
import os
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from agents import telemetry
from agents.tools import predict_transfusion
from backend.dependencies.rbac import require_permission
from backend.services.persistence import persist_prediction
from db.database import SessionLocal
from db.models import User
from rag.llm_client import llm_call_count
from schema.patient_schema import PatientRecord

router = APIRouter()
case_roles = require_permission("cases:run")

AGENTS = ("Extraction", "Prediction", "RAG", "Inventory", "Action", "Orchestrator")
DISPATCH_ROLES = {"Admin", "Blood Bank Manager"}  # may contact donors (same as /api/donors/alert)
MAX_JOBS_KEPT = 20
LAYER = "case_pipeline"

_jobs: "OrderedDict[str, dict]" = OrderedDict()
_run_lock = threading.Lock()


def _new_job(user: User, size_bytes: int) -> dict:
    # No filename: CBC files are often named after the patient (D72).
    job = {
        "job_id": uuid.uuid4().hex[:12],
        "status": "queued",            # queued | running | done | failed
        "size_bytes": size_bytes,
        "requested_by": user.email,
        "dispatch_allowed": user.role in DISPATCH_ROLES,
        "steps": [{"agent": a, "status": "pending", "detail": None} for a in AGENTS],
        "result": None,
        "error": None,
        "llm_calls": None,
        "seconds": None,
    }
    _jobs[job["job_id"]] = job
    while len(_jobs) > MAX_JOBS_KEPT:
        _jobs.popitem(last=False)
    return job


def _run(job: dict, pdf_path: str, user_id: int, confirmed: dict = None) -> None:
    started, calls_before = time.time(), llm_call_count()
    run_id = f"case:{job['job_id']}"
    timers = {}
    recommended = 0

    def progress(agent, status, detail):
        for step in job["steps"]:
            if step["agent"] == agent:
                step["status"], step["detail"] = status, detail
        # The agent monitor's view of the same progress (D71); status only, never `detail`.
        if status == "running":
            timers[agent] = telemetry.start_step(run_id, LAYER, agent, agent, hook_llm=True)
        elif agent in timers:
            timers.pop(agent).finish(status, outcome=status)

    telemetry.emit(run_id, LAYER, telemetry.RUN, None, "run_started", dispatch_allowed=job["dispatch_allowed"])
    try:
        job["status"] = "running"
        from agents.crew_agents import run_full_pipeline  # heavy: imports CrewAI
        with telemetry.run_context(run_id, LAYER):
            summary = run_full_pipeline(pdf_path, progress=progress, allow_dispatch=job["dispatch_allowed"],
                                        confirmed=confirmed)
        recommended = len(summary.get("action", {}).get("recommended", []))

        # Save the case like any other intake path, with the full
        # component-level prediction, so it appears in patient history.
        record = PatientRecord(**summary.pop("patient_record"))
        prediction = predict_transfusion(record)
        with SessionLocal() as db:
            ids = persist_prediction(db, record, prediction, created_by=user_id)
        job["result"] = {**summary, "prediction": prediction, **ids}
        job["status"] = "done"
    except Exception as exc:
        job["status"], job["error"] = "failed", f"{exc.__class__.__name__}: {exc}"
        for timer in timers.values():  # the step that raised
            timer.finish("error", exc_class=exc.__class__.__name__)
        timers.clear()
    finally:
        job["seconds"] = round(time.time() - started, 1)
        job["llm_calls"] = llm_call_count() - calls_before
        telemetry.emit(run_id, LAYER, telemetry.RUN, None, "run_finished",
                       status="done" if job["status"] == "done" else "failed",
                       duration_ms=int(job["seconds"] * 1000), llm_calls=job["llm_calls"],
                       dispatch_allowed=job["dispatch_allowed"], recommended_alerts=recommended,
                       exc_class=job["error"].split(":")[0] if job["error"] else None)
        os.unlink(pdf_path)
        _run_lock.release()


@router.post("/api/cases")
async def start_case(
    request: Request, file: UploadFile = File(...), values: Optional[str] = Form(None),
    user: User = Depends(case_roles),
):
    """`values`: optional JSON of the clinician-confirmed lab values from
    the upload page's confirm step (FR-03). Without it, the report must
    state every model input itself."""
    confirmed = _confirmed_values(values) if values else None
    if not _run_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="An agent case is already running -- try again shortly.")
    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
        job = _new_job(user, len(content))
        threading.Thread(target=_run, args=(job, tmp.name, user.id, confirmed), daemon=True).start()
    except Exception:
        _run_lock.release()
        raise

    request.state.audit_action = "agent_case_started"
    request.state.audit_resource_type = "agent_case"
    request.state.audit_resource_id = job["job_id"]
    request.state.audit_details = {"size_bytes": job["size_bytes"], "dispatch_allowed": job["dispatch_allowed"]}
    return {"job_id": job["job_id"], "status": job["status"]}


def _confirmed_values(values: str) -> dict:
    try:
        raw = json.loads(values)
        record = PatientRecord(
            hemoglobin=raw.get("hemoglobin"), platelets=raw.get("platelets"), INR=raw.get("INR"),
            age=raw.get("age"), surgery_type=raw.get("surgery_type"), blood_group=raw.get("blood_group"),
        )
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(status_code=422, detail="The confirmed values aren't valid -- check each field.")
    return {"hemoglobin": record.hemoglobin, "platelets": record.platelets, "INR": record.INR,
            "age": record.age, "surgery_type": record.surgery_type, "blood_group": record.blood_group}


@router.get("/api/cases/{job_id}", dependencies=[Depends(case_roles)])
def case_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such case job (jobs are kept in memory).")
    return job
