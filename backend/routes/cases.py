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

import os
import tempfile
import threading
import time
import uuid
from collections import OrderedDict

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from agents.tools import predict_transfusion
from backend.dependencies.rbac import require_role
from backend.services.persistence import persist_prediction
from db.database import SessionLocal
from db.models import User
from rag.llm_client import llm_call_count
from schema.patient_schema import PatientRecord

router = APIRouter()
case_roles = require_role("Hospital Staff", "Blood Bank Manager")

AGENTS = ("Extraction", "Prediction", "RAG", "Inventory", "Action", "Orchestrator")
DISPATCH_ROLES = {"Admin", "Blood Bank Manager"}  # may contact donors (same as /api/donors/alert)
MAX_JOBS_KEPT = 20

_jobs: "OrderedDict[str, dict]" = OrderedDict()
_run_lock = threading.Lock()


def _new_job(user: User, filename: str) -> dict:
    job = {
        "job_id": uuid.uuid4().hex[:12],
        "status": "queued",            # queued | running | done | failed
        "filename": filename,
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


def _run(job: dict, pdf_path: str, user_id: int) -> None:
    started, calls_before = time.time(), llm_call_count()

    def progress(agent, status, detail):
        for step in job["steps"]:
            if step["agent"] == agent:
                step["status"], step["detail"] = status, detail

    try:
        job["status"] = "running"
        from agents.crew_agents import run_full_pipeline  # heavy: imports CrewAI
        summary = run_full_pipeline(pdf_path, progress=progress, allow_dispatch=job["dispatch_allowed"])

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
    finally:
        job["seconds"] = round(time.time() - started, 1)
        job["llm_calls"] = llm_call_count() - calls_before
        os.unlink(pdf_path)
        _run_lock.release()


@router.post("/api/cases")
async def start_case(request: Request, file: UploadFile = File(...), user: User = Depends(case_roles)):
    if not _run_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="An agent case is already running -- try again shortly.")
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(await file.read())
        job = _new_job(user, file.filename)
        threading.Thread(target=_run, args=(job, tmp.name, user.id), daemon=True).start()
    except Exception:
        _run_lock.release()
        raise

    request.state.audit_action = "agent_case_started"
    request.state.audit_resource_type = "agent_case"
    request.state.audit_resource_id = job["job_id"]
    request.state.audit_details = {"filename": file.filename, "dispatch_allowed": job["dispatch_allowed"]}
    return {"job_id": job["job_id"], "status": job["status"]}


@router.get("/api/cases/{job_id}", dependencies=[Depends(case_roles)])
def case_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such case job (jobs are kept in memory).")
    return job
