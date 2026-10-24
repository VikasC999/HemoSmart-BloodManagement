"""
HemoSmart - Agent Monitor (D75)
----------------------------------
Turns agent_events rows (agents/telemetry.py) into what the Agent
monitor page shows: one card per agent, open runs, alerts, and the
"Needs attention" list.

Status is derived from events plus the clock, never stored: a process
that dies mid-step shows up as "stuck", not as a frozen "working". The
derivation functions are pure (events in, dicts out, `now` passed in)
so the rules are unit-tested with a fixed clock. The thresholds below
are the tuning knobs.
"""

import math
import os
from collections import defaultdict
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from agents.autonomy.orchestrator import ORPHAN_AFTER
from agents.telemetry import RUN
from db.models import AgentCycle, AgentEvent, AuditLog, DonorAlert

STUCK_AFTER = {"supply_loop": timedelta(minutes=5),
               "case_pipeline": timedelta(minutes=6)}  # 4 rate-limit retries x 65 s, plus the calls
WAIT_GRACE = timedelta(seconds=5)          # an llm_wait counts as "waiting" until its retry is due
WAIT_DEFAULT = timedelta(seconds=70)       # ...or this long, if it didn't say
MISSED_TICK_GRACE = timedelta(minutes=5)
REPEATED_FALLBACKS = 3                     # same agent fell back in this many runs in a row
DONOR_REPLY_ATTENTION = timedelta(hours=24)
HISTORY = timedelta(days=7)                # events the overview reads

LAYERS = {
    "supply_loop": {"title": "Supply loop", "steps": 7,
                    "agents": ("Donor", "Inventory", "Forecast", "Planner", "Logistics", "Explanation")},
    "case_pipeline": {"title": "Case pipeline", "steps": 6,
                      "agents": ("Extraction", "Prediction", "RAG", "Inventory", "Action", "Orchestrator")},
}
# Rule-based Python agents vs agents whose output an LLM writes. The
# supply loop's Explanation agent writes its brief with RAG + an LLM.
LLM_AGENTS = {("supply_loop", "Explanation")} | {("case_pipeline", a) for a in LAYERS["case_pipeline"]["agents"]}

OK_STATUSES = {"ok", "verified"}
FALLBACK_STATUSES = {"fell_back", "skipped"}


def llm_daily_limit() -> int:
    """Groq free-tier requests per day for the configured model."""
    return int(os.environ.get("HEMOSMART_LLM_DAILY_LIMIT", "1000"))


def event_dict(row: AgentEvent) -> dict:
    return {"id": row.id, "run_id": row.run_id, "layer": row.layer, "agent": row.agent, "step": row.step,
            "type": row.type, "status": row.status, "ts": row.ts, "duration_ms": row.duration_ms,
            "llm_calls": row.llm_calls, "data": row.data or {}}


def _iso(ts):
    return ts.isoformat() if ts else None


def _percentile(values, pct):
    """Nearest-rank percentile; None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(pct / 100 * len(ordered)) - 1)]


# ----------------------------------------------------------------------
# Pure derivation
# ----------------------------------------------------------------------
def summarise_runs(events: list) -> dict:
    """{run_id: run summary} for every run with a run_started event."""
    runs = {}
    for e in sorted(events, key=lambda e: (e["ts"], e["id"] or 0)):
        if e["layer"] not in LAYERS:
            continue
        run = runs.get(e["run_id"])
        if e["type"] == "run_started":
            runs[e["run_id"]] = run = {
                "run_id": e["run_id"], "layer": e["layer"], "started_at": e["ts"], "finished_at": None,
                "status": "running", "open": True, "steps_done": 0, "steps_total": LAYERS[e["layer"]]["steps"],
                "llm_calls": 0, "duration_ms": None, "trigger": e["data"].get("trigger"),
                "dispatch_allowed": e["data"].get("dispatch_allowed"), "recommended_alerts": 0,
                "waits": 0, "wait_s": 0,
            }
            continue
        if run is None:
            continue
        if e["type"] == "step_finished":
            run["steps_done"] += 1
            run["llm_calls"] += e["llm_calls"] or 0
        elif e["type"] == "llm_wait":
            run["waits"] += 1
            run["wait_s"] += e["data"].get("wait_s") or 0
        elif e["type"] == "run_finished":
            run.update(status=e["status"] or "ok", open=False, finished_at=e["ts"],
                       recommended_alerts=e["data"].get("recommended_alerts") or 0)
            if e["llm_calls"] is not None:
                run["llm_calls"] = e["llm_calls"]
            run["duration_ms"] = e["duration_ms"] if e["duration_ms"] is not None else \
                int((e["ts"] - run["started_at"]).total_seconds() * 1000)
    return runs


def _open_step(agent_events: list, runs: dict):
    """The agent's step that started and hasn't finished, in a run
    that is still open -- or None."""
    finished = {(e["run_id"], e["step"]) for e in agent_events if e["type"] == "step_finished"}
    for e in reversed(agent_events):
        if e["type"] == "step_started":
            run = runs.get(e["run_id"])
            if (e["run_id"], e["step"]) not in finished and run and run["open"]:
                return e
            return None
    return None


def agent_card(layer: str, agent: str, events: list, runs: dict, now: datetime) -> dict:
    """One agent's card. `events` is this agent's step_* / llm_wait
    events, oldest first."""
    card = {"layer": layer, "agent": agent, "kind": "llm" if (layer, agent) in LLM_AGENTS else "rule",
            "status": "idle", "current": None, "retry_in_s": None, "last_activity": None,
            "counts_24h": {"ok": 0, "fallback": 0, "error": 0}, "p50_ms": None, "p95_ms": None}
    if not events:
        return card
    last = events[-1]
    card["last_activity"] = _iso(last["ts"])

    finished = [e for e in events if e["type"] == "step_finished"]
    for e in finished:
        if e["ts"] >= now - timedelta(hours=24):
            bucket = "ok" if e["status"] in OK_STATUSES else "fallback" if e["status"] in FALLBACK_STATUSES else "error"
            card["counts_24h"][bucket] += 1
    durations = [e["duration_ms"] for e in finished if e["duration_ms"] is not None and e["status"] != "skipped"]
    card["p50_ms"], card["p95_ms"] = _percentile(durations, 50), _percentile(durations, 95)

    open_step = _open_step(events, runs)
    if open_step:
        run = runs[open_step["run_id"]]
        card["current"] = {"run_id": open_step["run_id"], "step": open_step["step"],
                           "started_at": _iso(open_step["ts"]),
                           "done": run["steps_done"], "total": run["steps_total"]}
        if last["type"] == "llm_wait":
            wait = timedelta(seconds=last["data"]["wait_s"]) if last["data"].get("wait_s") else WAIT_DEFAULT
            due = last["ts"] + wait
            if now < due + WAIT_GRACE:
                card["status"] = "waiting"
                card["retry_in_s"] = max(0, int((due - now).total_seconds()))
                return card
        card["status"] = "working"
        return card

    last_step = next((e for e in reversed(events) if e["type"] in ("step_started", "step_finished")), None)
    if last_step is None:
        return card
    if last_step["type"] == "step_started":  # its run closed under it: the process died
        card["status"] = "failed"
    elif last_step["status"] in FALLBACK_STATUSES:
        card["status"] = "degraded"
    elif last_step["status"] == "error":
        card["status"] = "failed"
    return card


def build_cards(events: list, runs: dict, now: datetime) -> list:
    by_agent = defaultdict(list)
    for e in sorted(events, key=lambda e: (e["ts"], e["id"] or 0)):
        if e["type"] in ("step_started", "step_finished", "llm_wait"):
            by_agent[(e["layer"], e["agent"])].append(e)
    return [agent_card(layer, agent, by_agent.get((layer, agent), []), runs, now)
            for layer, spec in LAYERS.items() for agent in spec["agents"]]


def scheduler_state(events: list, interval_min: float, now: datetime) -> dict:
    beats = [e for e in events if e["type"] == "heartbeat"]
    last = max(beats, key=lambda e: (e["ts"], e["id"] or 0)) if beats else None
    state = {"interval_min": interval_min, "last_heartbeat": _iso(last["ts"]) if last else None,
             "last_status": last["status"] if last else None, "next_due": None, "state": "unknown"}
    if interval_min <= 0:
        state["state"] = "off"
        return state
    if last is None:
        return state
    due = last["ts"] + timedelta(minutes=interval_min)
    state["next_due"] = _iso(due)
    state["state"] = "late" if now > due + MISSED_TICK_GRACE else "ok"
    return state


def build_alerts(cards: list, scheduler: dict, orphaned_cycles: list, events: list, now: datetime) -> list:
    alerts = []
    for card in cards:
        current = card["current"]
        if current:
            started = datetime.fromisoformat(current["started_at"])
            limit = STUCK_AFTER[card["layer"]]
            if now - started > limit:
                minutes = int((now - started).total_seconds() // 60)
                alerts.append({"kind": "stuck", "severity": "high",
                               "title": f"{card['agent']} agent may be stuck",
                               "detail": f"{_run_label(current['run_id'])}: \"{current['step']}\" has run for "
                                         f"{minutes} min (limit {int(limit.total_seconds() // 60)} min).",
                               "link": "agentMonitor", "run_id": current["run_id"]})
    for cycle in orphaned_cycles:
        alerts.append({"kind": "orphaned_cycle", "severity": "high",
                       "title": f"Cycle #{cycle['id']} never finished",
                       "detail": f"Still marked running since {cycle['started_at']}. It is marked failed at the "
                                 f"next restart.", "link": "agents", "run_id": f"cycle:{cycle['id']}"})
    if scheduler["state"] == "late":
        alerts.append({"kind": "missed_tick", "severity": "high", "title": "The scheduler missed a tick",
                       "detail": f"Last heartbeat {scheduler['last_heartbeat']}; one was due every "
                                 f"{scheduler['interval_min']:g} min.", "link": "agentMonitor"})

    # Repeated fallback: the same agent's last N finished steps, in N different runs, all fell back.
    by_agent = defaultdict(list)
    for e in sorted(events, key=lambda e: (e["ts"], e["id"] or 0)):
        if e["type"] == "step_finished" and e["layer"] in LAYERS:
            by_agent[(e["layer"], e["agent"])].append(e)
    for (layer, agent), finished in by_agent.items():
        latest_per_run = {}
        for e in finished:
            latest_per_run[e["run_id"]] = e
        recent = sorted(latest_per_run.values(), key=lambda e: e["ts"])[-REPEATED_FALLBACKS:]
        if len(recent) == REPEATED_FALLBACKS and all(e["status"] == "fell_back" for e in recent):
            alerts.append({"kind": "repeated_fallback", "severity": "moderate",
                           "title": f"{agent} agent fell back {REPEATED_FALLBACKS} runs in a row",
                           "detail": f"{LAYERS[layer]['title']}: its own answer was replaced by the verified "
                                     f"value each time. Check the LLM provider.",
                           "link": "agentMonitor"})
    return alerts


def _run_label(run_id: str) -> str:
    kind, _, ref = run_id.partition(":")
    return f"Cycle #{ref}" if kind == "cycle" else f"Case {ref[:6]}…" if kind == "case" else run_id


def run_timeline(events: list) -> list:
    """One entry per step of a run: start, end, duration, status, and
    its rate-limit waits -- the timeline drawer's bars."""
    steps, open_by_key = [], {}
    for e in sorted(events, key=lambda e: (e["ts"], e["id"] or 0)):
        key = (e["agent"], e["step"])
        if e["type"] == "step_started":
            open_by_key[key] = entry = {"agent": e["agent"], "step": e["step"], "started_at": _iso(e["ts"]),
                                        "finished_at": None, "duration_ms": None, "status": "running",
                                        "llm_calls": None, "waits": []}
            steps.append(entry)
        elif e["type"] == "llm_wait":
            entry = open_by_key.get(key) or next((s for s in reversed(steps) if s["agent"] == e["agent"]), None)
            if entry:
                entry["waits"].append({"at": _iso(e["ts"]), "seconds": e["data"].get("wait_s")})
        elif e["type"] == "step_finished":
            entry = open_by_key.pop(key, None)
            if entry is None:  # a skipped step never started
                entry = {"agent": e["agent"], "step": e["step"], "started_at": _iso(e["ts"]), "waits": []}
                steps.append(entry)
            entry.update(finished_at=_iso(e["ts"]), duration_ms=e["duration_ms"], status=e["status"],
                         llm_calls=e["llm_calls"])
    return steps


# ----------------------------------------------------------------------
# Database reads
# ----------------------------------------------------------------------
def load_events(db: Session, since: datetime) -> list:
    rows = db.query(AgentEvent).filter(AgentEvent.ts >= since).order_by(AgentEvent.ts, AgentEvent.id).all()
    return [event_dict(r) for r in rows]


def last_heartbeat(db: Session) -> list:
    """The latest heartbeat, as a one-item event list (or empty)."""
    row = (db.query(AgentEvent).filter(AgentEvent.type == "heartbeat")
           .order_by(AgentEvent.ts.desc(), AgentEvent.id.desc()).first())
    return [event_dict(row)] if row else []


def orphaned_cycles(db: Session, now: datetime) -> list:
    rows = (db.query(AgentCycle).filter(AgentCycle.status == "running", AgentCycle.mode != "simulation",
                                        AgentCycle.started_at < now - ORPHAN_AFTER).all())
    return [{"id": c.id, "started_at": _iso(c.started_at)} for c in rows]


def assistant_card(db: Session, now: datetime) -> dict:
    """The assistant has no events of its own: its card comes from the
    chat_query audit rows (D68), which hold no message text."""
    rows = (db.query(AuditLog).filter(AuditLog.action == "chat_query", AuditLog.timestamp >= now - timedelta(hours=24))
            .order_by(AuditLog.timestamp).all())
    details = [r.details or {} for r in rows]
    outcomes = defaultdict(int)
    for d in details:
        outcomes[d.get("outcome") or "answered"] += 1
    last = rows[-1] if rows else None
    return {"layer": "assistant", "agent": "Assistant", "kind": "llm",
            "status": "degraded" if last and (last.details or {}).get("degraded") else "idle",
            "last_activity": _iso(last.timestamp) if last else None,
            "messages_24h": len(rows), "outcomes_24h": dict(outcomes),
            "llm_used_24h": sum(1 for d in details if d.get("llm_used")),
            "degraded_24h": sum(1 for d in details if d.get("degraded"))}


def attention_items(db: Session, runs: dict, alerts: list, now: datetime) -> list:
    """What needs a person, each linking to the page where it's acted on.
    The monitor never acts itself (D74)."""
    items = [dict(a) for a in alerts]
    day_ago = now - timedelta(hours=24)

    live = db.query(AgentCycle).filter(AgentCycle.mode != "simulation")
    for cycle in live.filter(AgentCycle.status == "degraded", AgentCycle.started_at >= day_ago) \
            .order_by(AgentCycle.id.desc()).limit(5):
        agents = sorted({e["agent"] for e in (cycle.events or []) if e.get("kind") in ("error", "fallback")})
        items.append({"kind": "degraded_cycle", "severity": "moderate", "title": f"Cycle #{cycle.id} ran degraded",
                      "detail": f"Fell back or failed: {', '.join(agents) or 'see the cycle log'}.",
                      "link": "agents", "run_id": f"cycle:{cycle.id}"})

    latest = live.filter(AgentCycle.finished_at.isnot(None)).order_by(AgentCycle.id.desc()).first()
    escalations = (latest.summary or {}).get("escalations", 0) if latest else 0
    if escalations:
        items.append({"kind": "escalation", "severity": "moderate",
                      "title": f"{escalations} supply gap{'s' if escalations != 1 else ''} escalated to the regional bank",
                      "detail": f"From cycle #{latest.id}: donors and partner hospitals couldn't cover "
                                f"{'them' if escalations != 1 else 'it'}.", "link": "agents",
                      "run_id": f"cycle:{latest.id}"})

    for run in runs.values():
        if (run["layer"] == "case_pipeline" and not run["open"] and run["dispatch_allowed"] is False
                and run["recommended_alerts"] and run["finished_at"] >= day_ago):
            n = run["recommended_alerts"]
            items.append({"kind": "recommended_alerts", "severity": "info",
                          "title": f"{_run_label(run['run_id'])} recommended donor alerts",
                          "detail": f"{n} blood type{'s' if n != 1 else ''} to alert. A Blood Bank Manager must "
                                    f"send them.", "link": "donorOutreach", "run_id": run["run_id"]})

    unanswered = (db.query(DonorAlert).filter(DonorAlert.responded.is_(None),
                                              DonorAlert.sent_at < now - DONOR_REPLY_ATTENTION).count())
    if unanswered:
        items.append({"kind": "unanswered_alerts", "severity": "info",
                      "title": f"{unanswered} donor alert{'s' if unanswered != 1 else ''} unanswered for over 24 h",
                      "detail": "Record replies you've heard back, or let the next cycle expire them at 48 h.",
                      "link": "donorResponses"})
    return items


def overview(db: Session, now: datetime, interval_min: float) -> dict:
    events = load_events(db, now - HISTORY)
    runs = summarise_runs(events)
    cards = build_cards(events, runs, now)
    scheduler = scheduler_state(last_heartbeat(db), interval_min, now)
    alerts = build_alerts(cards, scheduler, orphaned_cycles(db, now), events, now)

    week = db.query(AgentCycle).filter(AgentCycle.mode != "simulation", AgentCycle.started_at >= now - HISTORY)
    cycles_7d = {status: week.filter(AgentCycle.status == status).count() for status in ("ok", "degraded", "failed")}
    case_steps = [e for e in events if e["layer"] == "case_pipeline" and e["type"] == "step_finished"]
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    assistant = assistant_card(db, now)
    llm_today = sum(e["llm_calls"] or 0 for e in events if e["type"] == "step_finished" and e["ts"] >= midnight)

    open_runs = [_run_out(r) for r in sorted(runs.values(), key=lambda r: r["started_at"], reverse=True) if r["open"]]
    return {
        "now": _iso(now),
        "scheduler": scheduler,
        "has_open_run": bool(open_runs),
        "open_runs": open_runs,
        "layers": [{"layer": layer, "title": spec["title"], "steps": spec["steps"],
                    "cards": [c for c in cards if c["layer"] == layer],
                    "last_run": _latest_run(runs, layer),
                    "runs_24h": sum(1 for r in runs.values()
                                    if r["layer"] == layer and r["started_at"] >= now - timedelta(hours=24))}
                   for layer, spec in LAYERS.items()],
        "assistant": assistant,
        "alerts": alerts,
        "attention": attention_items(db, runs, alerts, now),
        "metrics": {
            "cycles_7d": cycles_7d,
            "case_steps_7d": {"verified": sum(1 for e in case_steps if e["status"] == "verified"),
                              "total": len(case_steps)},
            "llm_calls_today": llm_today,  # agent steps; the assistant's are in its card
            "llm_daily_limit": llm_daily_limit(),
        },
    }


def _latest_run(runs: dict, layer: str):
    mine = [r for r in runs.values() if r["layer"] == layer]
    return _run_out(max(mine, key=lambda r: r["started_at"])) if mine else None


def _run_out(run: dict) -> dict:
    return {**run, "label": _run_label(run["run_id"]), "started_at": _iso(run["started_at"]),
            "finished_at": _iso(run["finished_at"])}


def recent_runs(db: Session, now: datetime, layer: str = None, limit: int = 20) -> list:
    query = db.query(AgentEvent.run_id).filter(AgentEvent.type == "run_started")
    if layer:
        query = query.filter(AgentEvent.layer == layer)
    ids = [r for (r,) in query.order_by(AgentEvent.ts.desc(), AgentEvent.id.desc()).limit(limit)]
    if not ids:
        return []
    rows = db.query(AgentEvent).filter(AgentEvent.run_id.in_(ids)).all()
    runs = summarise_runs([event_dict(r) for r in rows])
    return [_run_out(r) for r in sorted(runs.values(), key=lambda r: r["started_at"], reverse=True)]


def run_detail(db: Session, run_id: str):
    rows = db.query(AgentEvent).filter(AgentEvent.run_id == run_id).order_by(AgentEvent.ts, AgentEvent.id).all()
    events = [event_dict(r) for r in rows]
    runs = summarise_runs(events)
    if run_id not in runs:
        return None
    kind, _, ref = run_id.partition(":")
    return {"run": _run_out(runs[run_id]), "steps": run_timeline(events),
            "cycle_id": int(ref) if kind == "cycle" and ref.isdigit() else None}


def metrics(db: Session, now: datetime, days: int) -> dict:
    events = load_events(db, now - timedelta(days=days))
    per_agent = defaultdict(lambda: {"steps": 0, "ok": 0, "verified": 0, "fell_back": 0, "skipped": 0, "error": 0,
                                     "llm_calls": 0, "waits": 0, "wait_s": 0, "_durations": []})
    for e in events:
        if e["layer"] not in LAYERS or e["agent"] == RUN:
            continue
        m = per_agent[(e["layer"], e["agent"])]
        if e["type"] == "step_finished":
            m["steps"] += 1
            if e["status"] in m:
                m[e["status"]] += 1
            m["llm_calls"] += e["llm_calls"] or 0
            if e["duration_ms"] is not None and e["status"] != "skipped":
                m["_durations"].append(e["duration_ms"])
        elif e["type"] == "llm_wait":
            m["waits"] += 1
            m["wait_s"] += e["data"].get("wait_s") or 0
    agents = []
    for (layer, agent), m in sorted(per_agent.items()):
        durations = m.pop("_durations")
        judged = m["verified"] + m["fell_back"] + m["error"]
        agents.append({"layer": layer, "agent": agent, **m,
                       "p50_ms": _percentile(durations, 50), "p95_ms": _percentile(durations, 95),
                       "verified_rate": round(m["verified"] / judged, 3) if layer == "case_pipeline" and judged else None})
    runs = summarise_runs(events)
    by_layer = defaultdict(lambda: defaultdict(int))
    for r in runs.values():
        by_layer[r["layer"]][r["status"]] += 1
    return {"days": days, "agents": agents, "runs": {k: dict(v) for k, v in by_layer.items()}}
