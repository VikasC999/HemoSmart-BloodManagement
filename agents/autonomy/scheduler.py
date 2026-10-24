"""
Runs the autonomous loop on a timer inside the API process -- what makes
it autonomous rather than a button: nobody has to remember to check for
expiring units or a coming shortage.

HEMOSMART_AUTONOMY_INTERVAL_MINUTES sets the period (default 60; 0
disables). The first cycle runs one interval after startup, not at
startup, so booting the API stays fast.
"""

import asyncio
import os

from starlette.concurrency import run_in_threadpool

from agents import telemetry
from agents.autonomy.orchestrator import CycleAlreadyRunning, run_cycle


def interval_minutes() -> float:
    return float(os.environ.get("HEMOSMART_AUTONOMY_INTERVAL_MINUTES", "60"))


async def autonomy_loop(get_retriever, get_llm) -> None:
    minutes = interval_minutes()
    heartbeat("started", minutes)
    while True:
        await asyncio.sleep(minutes * 60)
        try:
            cycle = await run_in_threadpool(
                run_cycle, trigger="scheduler", retriever=get_retriever(), llm=get_llm(),
            )
            print(f"[autonomy] Cycle {cycle['id']} {cycle['status']}: {cycle['summary']}")
            heartbeat(cycle["status"], minutes)
        except CycleAlreadyRunning:
            print("[autonomy] Previous cycle still running -- skipping this tick.")
            heartbeat("busy", minutes)
        except Exception as exc:  # never let the loop die
            print(f"[autonomy] Cycle failed: {exc.__class__.__name__}: {exc}")
            heartbeat("error", minutes, exc_class=exc.__class__.__name__)
        await run_in_threadpool(telemetry.prune)


def heartbeat(status: str, minutes: float, **data) -> None:
    """Proof of life for the agent monitor: it alerts when a tick is
    overdue (D75)."""
    telemetry.emit("scheduler", "scheduler", "Scheduler", "tick", "heartbeat", status=status,
                   interval_min=minutes, **data)


def start(app) -> asyncio.Task | None:
    if interval_minutes() <= 0:
        print("[autonomy] Scheduler disabled (HEMOSMART_AUTONOMY_INTERVAL_MINUTES=0).")
        return None
    generator = lambda: getattr(app.state, "explanation_generator", None)
    return asyncio.create_task(autonomy_loop(
        get_retriever=lambda: generator().retriever if generator() else None,
        get_llm=lambda: generator().llm if generator() else None,
    ))
