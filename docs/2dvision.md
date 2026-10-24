# 2D Vision: Agent Monitor for HemoSmart

A handoff for building a live "agents at work" monitor: a flat (2D) take on the [Agents Office](https://github.com/ajsahni/agents-office) idea, fitted to HemoSmart's agents. Planned on 2026-10-24. **Built on 2026-10-24** (minimum version, steps 1–10; not committed). The decisions are D71–D75 in [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md), with all of the §11 defaults; no Groq budget was spent on the evidence table. Orphaned-cycle recovery (extra c) was built too.

Read this top to bottom before starting. Line numbers were correct on 2026-10-24 and will drift, so search for the named function if one doesn't match.

---

## 1. The idea in one paragraph

Every agent run reports small, structured, patient-data-free events (step started, step finished, waiting on the LLM, heartbeat). They are stored in one new table. A new "Agent monitor" page polls them and shows:
- which agents are working, waiting, degraded, failed or idle;
- how long each step takes;
- how often the LLM agents' answers check out;
- what needs a person.

It sits *next to* the agents, not inside them: about 10 lines of hooks per agent layer, and no agent logic changes.

## 2. Why it's worth building (and what it is not)

The existing **Agent activity** page ([AutonomyCard.jsx](../frontend/src/components/AutonomyCard.jsx)) already shows *what the supply loop decided* in its latest cycle. The monitor only earns its place if it shows what that page can't:

| New evidence | Why it matters for the viva |
|---|---|
| Liveness: scheduler heartbeat, stuck and orphaned runs | "Autonomous" is only credible if you can show it's still running |
| Time per agent step | Shows where the time goes (LLM calls, rate-limit waits) |
| CrewAI verified vs fell-back rate per agent, over many runs | Hard evidence for "agents recover from failure". Today this is thrown away (in-memory, last 20 jobs) |
| A "Needs attention" queue | The human-in-the-loop story, in one place |

**The best live demo is a CrewAI patient case.** It runs about 1 minute, with 16–30 LLM requests and Groq rate-limit waits of 10–65 s, so the cards visibly move working → waiting → verified. The hourly supply loop finishes in 1–2 s (plus the LLM brief), so it is only briefly "live".

## 3. What maps from the reference, and what doesn't

| Agents Office idea | HemoSmart decision |
|---|---|
| Card per agent: status, current task, last activity | ✅ Build. Group by layer: rule-based supply-loop agents vs LLM case-pipeline agents. LLM agents get the yellow AI marker (D64), so the two-layer design is visible. |
| doing / next / done counts | ✅ As a step tracker: both layers are fixed sequential pipelines. |
| Run timeline | ✅ One horizontal bar per step; a dashed segment for rate-limit waits. |
| Metrics | ✅ Scoped down (§9). LLM calls against the Groq free-tier limit, not $. |
| Alerts and approvals | ✅ The most useful part. It **links** to existing pages to act; it never acts itself. |
| 3D isometric office | ❌ Clashes with D64 (calm clinical UI; severity shown by icon and word, never colour alone), poor accessibility and mobile, heavy dependency, no evidence value. |
| $ cost | ❌ Groq free tier. The real constraint is the ~8k tokens-per-minute cap. |
| WebSockets / SSE push | ❌ for the first version. 2 s polling looks live at this event rate (§7). |
| Langfuse / OpenTelemetry | ❌ Needs self-hosting (deployment is out of scope) and would send traces outside, against D56/D68. Name it in the report as the production path behind the same `emit()` seam. |

## 4. Current state: what exists, and the gaps

| Source | What it has | Gap |
|---|---|---|
| `AgentCycle` row ([db/models.py](../db/models.py), ~L120–134) | `started_at`, `finished_at`, `status` (ok/degraded), `summary` counts, `events` | `events` is written only at the end ([orchestrator.py](../agents/autonomy/orchestrator.py) ~L126), so nothing is visible mid-cycle. Events have no timestamps or durations. A crashed cycle stays `running` forever. |
| Blackboard events ([blackboard.py](../agents/autonomy/blackboard.py) ~L50–56) | agent, kind (observe/decide/act/explain/fallback/error), message, data | A reasoning log written for people to read, not telemetry. Free text and donor IDs: **don't copy it into the monitor store.** |
| Case jobs ([backend/routes/cases.py](../backend/routes/cases.py) ~L40–62) | per-agent pending/running/verified/fell_back, `llm_calls`, `seconds` | In-memory `OrderedDict`, `MAX_JOBS_KEPT = 20`, lost on restart. No per-step timing. |
| Audit rows ([middleware/audit.py](../backend/middleware/audit.py), [routes/chat.py](../backend/routes/chat.py)) | `chat_query` details (intents, decisions, outcome, `llm_used`, `degraded`, D68); `autonomy_cycle_run`; `agent_case_started` | Enough for assistant metrics as-is. **Leak:** `agent_case_started` stores the uploaded **filename** (cases.py ~L111); see §5. |
| `llm_call_count()` ([rag/llm_client.py](../rag/llm_client.py) ~L36) | Process-wide counter, also hooks LiteLLM (used by CrewAI) | Not per agent, and concurrent requests get counted. Acceptable, because cycles and cases each run under a lock. No token counts. |

## 5. Two existing problems to fix along the way

1. **Patient name can leak through the audit log.** [cases.py](../backend/routes/cases.py) ~L111 writes `{"filename": file.filename, ...}` into `audit_details`. CBC filenames are often patient names (`Ravi_Kumar_CBC.pdf`), which contradicts D56 (no patient identifiers stored). Fix: drop the filename, or store only the extension and size.
2. **Orphaned cycles.** If the process dies mid-cycle, the `AgentCycle` row stays `status="running"` forever. Fix: at startup, mark `running` rows older than 10 minutes as `failed`, and flag them in the monitor.

Also decide this: **CrewAI's anonymous telemetry is not disabled** anywhere in the repo (no `CREWAI_DISABLE_TELEMETRY` / `OTEL_SDK_DISABLED`). Turning it off keeps the data-minimisation story consistent.

## 6. Design

### 6.1 Event table (proposed D71)

A new append-only table. `AgentCycle.events` stays unchanged: it is the domain reasoning log that AutonomyCard and the assistant read. `create_all` creates the new table with no migration ([db/database.py](../db/database.py) ~L31).

```
AgentEvent(
  id,
  run_id: str,        # "cycle:{AgentCycle.id}" | "case:{job_id}"
  layer: str,         # "supply_loop" | "case_pipeline"
  agent: str,         # "Forecast", "Extraction", "Scheduler", ...
  step: str,          # short step label
  type: str,          # run_started | step_started | step_finished | llm_wait | run_finished | heartbeat
  status: str|None,   # ok | verified | fell_back | skipped | error | degraded
  ts: DateTime (indexed),
  duration_ms: int|None,
  llm_calls: int|None,
  tokens: int|None,   # optional, see §9
  data: JSON          # allow-listed keys only, see 6.2
)
```

New module `agents/telemetry.py`:
- `emit(run_id, layer, agent, step, type, status=None, **safe)` writes one row.
- `step(run_id, layer, agent, step)` is a context manager that records start, end, duration and the LLM-call delta.
- Both **never raise** (same rule as the audit middleware) and use their **own** `SessionLocal`, so a telemetry failure can't roll back agent work.
- `emit()` returns immediately when `run_id is None`, `mode == "simulation"`, or `HEMOSMART_AGENT_EVENTS=0`. [evaluation/simulate.py](../evaluation/simulate.py) runs thousands of cycles and must write **zero** events.
- `prune(days)` runs at each scheduler tick. Default 14 days (`HEMOSMART_AGENT_EVENTS_DAYS`). Expect about 500 rows/day.

### 6.2 Redaction: an allow-list, not a deny-list (proposed D72)

`emit()` keeps only these `data` keys and silently drops the rest:
- counts: `expired_units`, `donors_alerted`, `transferred_units`, `escalations`, `llm_calls`, `units`;
- `blood_group`, `component`;
- `step`, `outcome` (verified / fell_back);
- `exc_class`: the exception **class name only**.

Never store:
- lab values;
- the PDF filename or path;
- chat text or the cycle brief;
- donor IDs or phones;
- exception **messages**. Pydantic errors echo input values such as hemoglobin.

### 6.3 Where events come from

| Layer | Hook | Size |
|---|---|---|
| **Supply loop** | Around the step loop in [orchestrator.py](../agents/autonomy/orchestrator.py) (~L106–120): `step_started` before each `step(bb, db)`; `step_finished` after the commit or in the `except`; the "skipped" branch (~L108) → `skipped`. `run_started` / `run_finished` around ~L86 and ~L128. | ~10 lines; no agent code touched |
| **Scheduler** | [scheduler.py](../agents/autonomy/scheduler.py) (~L23–35): `heartbeat` once at startup and on every tick, including the "previous still running" and exception branches. | ~3 lines |
| **Case pipeline** | Extend the existing `progress()` callback in [cases.py](../backend/routes/cases.py) (~L65–92, wired at [crew_agents.py](../agents/crew_agents.py) ~L436): "running" → `step_started`; verified / fell_back → `step_finished` with duration. In `_run_isolated_task`, emit `llm_wait` just before `time.sleep(wait)` (~L394). | ~10 lines |
| **Assistant** | No new events: derive its card from the existing `chat_query` audit rows (D68). | 0 |
| **MCP** | Skip (low value). Optional later: a card fed by audit rows. | — |

Don't use CrewAI's internal event bus for this. CrewAI is pinned at 1.15.21 and its internals are already patched once (D41).

### 6.4 Status model (proposed D75)

Each agent's status is derived from its latest events. No state is stored.

| Status | Rule |
|---|---|
| working | a `step_started` with no matching `step_finished` |
| waiting | latest event is `llm_wait` within the last 70 s (a queued case job shows "Queued") |
| degraded | last `step_finished` was `fell_back` or `skipped` |
| failed | last `step_finished` was `error` |
| idle | otherwise |

**Alerts:**
- **Stuck:** `working` longer than 5 min (loop) or 6 min (case step: 4 retries × 65 s max wait, plus the calls).
- **Orphaned cycle:** an `AgentCycle` still `running` after 10 min.
- **Scheduler missed a tick:** now is past `last_heartbeat + interval + 5 min`. Interval 0 shows "Scheduler off", not an alert.
- **Repeated fallback:** the same agent `fell_back` in 3 runs in a row.

**Current task:** the open step's label plus the run ID, e.g. "Case 3f2a… · Action agent (2 of 6 done)". Never show the filename or any values.

### 6.5 Live updates: polling (proposed D73)

- Poll `GET /api/agents/overview` **every 2 s while a run is open, every 20 s when idle**, and pause while the tab is hidden.
- This is the same pattern UploadCard already uses for case status ([UploadCard.jsx](../frontend/src/components/UploadCard.jsx) ~L226–245). It works with the existing Bearer-token `fetch` in [api.js](../frontend/src/api.js) and adds no new auth or CORS surface.

**Why not SSE or WebSockets (yet):**
- Events are seconds apart, so 2 s polling looks live.
- Events come from a `threading.Thread` (cases.py ~L103) and the threadpool (autonomy.py ~L27). A push bus would need thread-to-event-loop plumbing, and Redis pub/sub for more than one worker.
- The browser's `EventSource` can't send an `Authorization` header. You'd need a short-lived stream ticket in the URL (which the audit middleware would then log) or a new npm dependency.
- The audit middleware would write one row per *connection*, so SSE routes would need excluding from it.

**Optional extra:** `sse_starlette` is already installed as a transitive dependency. If you want to say "push" in the viva, add one SSE endpoint with `POST /api/agents/stream-ticket` (a 60 s JWT, `scope=agents:stream`), and keep polling as the fallback.

### 6.6 API and permission (proposed D74)

New router `backend/routes/agents_monitor.py`, all `GET`, sync, cheap queries:

| Endpoint | Returns |
|---|---|
| `GET /api/agents/overview` | Per-agent cards (layer, kind rule-based/LLM, status, current task, last activity, 24 h ok/fallback/error counts, p50 step time), scheduler heartbeat, open runs, alerts, metric strip |
| `GET /api/agents/runs?layer=&limit=` | Recent runs: status, duration, LLM calls |
| `GET /api/agents/runs/{run_id}` | Ordered timeline events. For `cycle:*`, a link to that cycle's reasoning log |
| `GET /api/agents/metrics?days=7` | Aggregates from §9 |

**Permission:** add `"agents:monitor": {Blood Bank Manager, Auditor}` (Admin implicit) to [schema/permissions.py](../schema/permissions.py), with `LABELS["agents:monitor"] = "agent monitoring"`. Guard every route with `require_permission("agents:monitor")`.
- **Don't** add it to `CHAT_TOPICS`: the assistant doesn't answer monitoring questions.
- **Parity test:** add the new router module to `_route_capabilities()` in [tests/test_chat_rbac.py](../tests/test_chat_rbac.py) (~L128–133; the module list is hard-coded), plus an assert that `("GET", "/api/agents/overview") == "agents:monitor"`.

### 6.7 Frontend

**Section:** new `agentMonitor` in [navigation.jsx](../frontend/src/navigation.jsx).
- Lucide `Activity` icon, layout `wide`, title "Agent monitor", description "Are the agents running, how long each step takes, and what needs a person."
- Placed after "Agent activity" for Blood Bank Manager and Auditor, and in Admin's "Oversight" group.
- Cross-link it with Agent activity (decisions vs operations).
- Use the **hemosmart-frontend-design** skill when building it.

> **As built (owner's request, 2026-10-24):** the page is a flat 2D "agent office" modelled on the Agents Office screenshot. Rooms for the supply loop, case pipeline, assistant, scheduler and partner hospital each have a plaque (agent count, two metrics, doing / next / done) and a desk per agent showing icon + word status. Dotted SVG corridors lead to a "Blood bank data" hub and animate while a room's run is open. A right-hand **Task status** panel filters runs (All / In progress / Waiting / Done / Needs attention) with progress bars; clicking a run opens the timeline drawer, and clicking a desk shows that agent's details. On phones the rooms stack and the corridors are hidden. The list below was the original card layout.

**Layout, top to bottom (original plan):**
1. **Health strip:**
   - scheduler heartbeat ("Last tick 12 min ago · next in 48 min", or a stale warning);
   - cycles OK vs degraded over 7 days;
   - case verified rate;
   - LLM calls today against the free-tier limit;
   - units transferred and expired over 30 days (reuse `api.wastageMetrics`, as AutonomyCard does).
2. **Agent cards, grouped by layer:**
   - **Supply loop, rule-based:** Donor, Inventory, Forecast, Planner, Logistics, Explanation.
   - **Case pipeline, LLM:** Extraction, Prediction, RAG, Inventory, Action, Orchestrator.
   - **Assistant:** one card from the audit rows.

   Each card shows name, kind tag, status badge (icon + word), current task, last activity (`UpdatedAt`), small counts and p50 time. An open run shows a done / current / next step tracker. LLM cards get the 3px yellow AI border (D64); rule-based cards don't.
3. **Needs attention:** link out to act, never act here. Items:
   - degraded cycles, with the fallback reason category;
   - escalations to the regional bank (`summary.escalations > 0`);
   - case runs whose donor alerts were only *recommended* (`dispatch_allowed = False`, needs a manager);
   - donor alerts with no reply after 24 h (→ Donor responses);
   - stuck or orphaned runs;
   - a missed scheduler tick;
   - repeated fallbacks.
4. **Recent runs table:** clicking a row opens a timeline drawer with one bar per step, a dashed segment for `llm_wait`, and status as icon + word. The drawer is full-screen on mobile.

**Not shown:** chat "yes" confirmations. They are per user and per session, and live in Redis (D67, D70), so there is no global list. Say why in D75.

**States:** use the existing primitives in [components/ui](../frontend/src/components/ui/index.jsx):
- `SkeletonLines` / `SkeletonTable` while loading, with no placeholder numbers (D64);
- `EmptyState` "No agent runs recorded yet" (with the Run button if `canRun`);
- `StaleNotice` when polling fails for longer than 2× the interval;
- `ErrorState` with retry.

**Accessibility and theming:**
- Severity is never shown by colour alone.
- The step tracker and timeline also render as an `<ol>` of text.
- Status changes are announced through a throttled polite `aria-live` region.
- Use only [tokens.css](../frontend/src/styles/tokens.css) variables, so dark mode works with no new colours.
- On mobile: one column, with the attention panel above the cards.

## 7. Metrics

| Available now (no instrumentation) | Needs `AgentEvent` |
|---|---|
| cycle duration; ok vs degraded rate | per-agent step latency p50 / p95, both layers |
| per-cycle counts: donors alerted, transferred, expired, escalations, errors | per-agent LLM-call delta (caveat: process-wide counter) |
| `wastage_stats`: issued, expired, transferred, wastage rate | CrewAI verified vs fell-back per agent, kept across runs and restarts |
| assistant outcomes, `llm_used`, degraded rate (audit rows) | rate-limit waits (count, seconds); waiting vs working time |
| pending donor alerts (`DonorAlert.responded IS NULL`) | |

**Optional: token counts.** Read `usage.total_tokens` in the LiteLLM success callback ([rag/llm_client.py](../rag/llm_client.py) ~L30–48) and from `response.usage` in `GroqClient.generate`. Show tokens per minute against the 8k cap.

**Optional: evidence script.** `evaluation/agent_reliability.py` turns `AgentEvent` rows into `evaluation/results/agent_reliability.md` (per-agent verified rate, step p50 / p95, waits). It needs about 10 real case runs, which spends Groq free-tier budget.

## 8. Build order

Implement as the minimum version (steps 1–10), then the extras. Total effort for the minimum version is about 2–3 days.

| # | Step | Files | Effort | Risk |
|---|---|---|---|---|
| 1 | Write D71–D75 in DESIGN_DECISIONS (owner approves) | docs/DESIGN_DECISIONS.md | 0.5 h | none |
| 2 | `AgentEvent` model + `agents/telemetry.py` (`emit`, `step`, allow-list, off-switch, `prune`) | db/models.py, agents/telemetry.py (new) | 2 h | low |
| 3 | Supply-loop hooks + scheduler heartbeat | agents/autonomy/orchestrator.py, agents/autonomy/scheduler.py | 1.5 h | low |
| 4 | Case-pipeline hooks (`progress`, `llm_wait`); **remove filename from audit details** | backend/routes/cases.py, agents/crew_agents.py | 1.5 h | low–medium (live CrewAI runs cost Groq budget) |
| 5 | Status derivation + alerts as pure, unit-testable functions | backend/services/agent_monitor.py (new) | 3 h | medium (threshold tuning) |
| 6 | Monitor router + `agents:monitor` capability; include the router in main.py | backend/routes/agents_monitor.py (new), schema/permissions.py, backend/main.py | 2 h | low |
| 7 | Tests (§9) | tests/test_agent_monitor.py (new), tests/test_chat_rbac.py | 3 h | low |
| 8 | Frontend page: health strip, cards, attention panel, polling | frontend/src/components/AgentMonitor.jsx (new), navigation.jsx, api.js, index.css | 5–6 h | medium (design review) |
| 9 | Timeline drawer | AgentMonitor.jsx or AgentRunTimeline.jsx | 2–3 h | low |
| 10 | Demo step + report alignment rows | docs/DEMO_SCRIPT.md, docs/REPORT_ALIGNMENT.md | 1 h | none |

**Optional extras, most valuable first:**
- **(a)** the `agent_reliability.py` evidence table;
- **(b)** token capture;
- **(c)** startup recovery for orphaned cycles (§5);
- **(d)** the SSE stream with a ticket;
- **(e)** an MCP card from audit rows.

**Don't build:**
- the 3D office, a $ cost view, or WebSockets;
- Langfuse or OpenTelemetry exporters;
- copying every blackboard log line into the new table;
- per-message chat events;
- a global list of chat confirmations;
- a chat intent for monitoring;
- approval *buttons* inside the monitor. Link to the existing pages instead, so each action and its permission check lives in one place.

## 9. Tests (`tests/test_agent_monitor.py`, SQLite, reuse conftest.py)

- **Supply-loop emission:**
  - one `run_cycle` with fakes writes `run_started`, 7× `step_started` / `step_finished`, and `run_finished`, all with `run_id = "cycle:{id}"`;
  - a forced Forecast failure gives `status="error"` and a degraded cycle;
  - a Planner failure gives `skipped` on the dependent steps.
- **Case emission:** a fake `run_full_pipeline` that calls `progress` produces step events. A simulated rate limit, with `time.sleep` monkeypatched, emits `llm_wait`.
- **Redaction:**
  - passing lab values, a filename, a donor ID, a phone and an exception message to `emit()` leaves none of them in the row;
  - a case uploaded as `Ravi_Kumar_CBC.pdf` leaves the name in no event and no audit row.
- **Permissions:** for 5 roles × 4 endpoints, Blood Bank Manager, Auditor and Admin get 200; Hospital Staff and Donor Coordinator get 403 with the shared `denial()` text; no token gets 401. The route/matrix parity test still passes.
- **Status and alerts:** with a frozen clock, fabricated events yield each status. A stale open step, an orphaned cycle and a missed heartbeat each raise an alert; interval 0 shows "off".
- **Simulation:** `run_cycle(mode="simulation")` × 50 writes 0 rows, and so does a run with `HEMOSMART_AGENT_EVENTS=0`.
- **Retention:** `prune` removes only rows older than the window.
- **SSE (only if built):** the ticket works only on the stream endpoint and expires after 60 s; a stream without a ticket gets 401.

## 10. Viva value

**New DEMO_SCRIPT step "3b. Agent monitor (1.5 min)"**, after the Blood Bank Manager section:
1. Open the monitor as the manager in one tab.
2. In another tab, staff uploads `adapters/sample_cbc_report.pdf` → **Run AI agents**.
3. Watch the case cards move Extraction working → verified → Prediction …, including "waiting: rate-limited by Groq, retry in 12 s".
4. Click **Run agent cycle** and watch the loop's steps finish in seconds.
5. Point at **Needs attention** (an escalation, a degraded cycle).
6. Log in as the Auditor and show the same monitor, read-only.

**Examiner Q&A to add:**
- *How do you know an agent is stuck?* The heartbeat, an open step past its threshold, and the orphaned-cycle check.
- *Why not 3D, push or Langfuse?* §3 and §6.5.
- *Does monitoring leak patient data?* No. The payload is allow-listed, and a test proves a patient-named file leaves no trace.
- *How often do your LLM agents fail?* The verified-rate metric, and the evidence table if it was run.

**Report alignment:**
- Add rows under Chapter 4.3.8 (Agent Orchestration Module): monitoring (heartbeat, stuck detection, per-step telemetry) and failure recovery shown live.
- Keep the honest wording for 4.5 #2 ("independent, sequentially coordinated"). The timeline makes the sequential order visible.

## 11. Open decisions (recommended default in brackets)

1. New `agents:monitor` permission, or reuse `autonomy:read`? [new]
2. Should Hospital Staff see case cards for their own runs? [no; they already see progress on the upload page]
3. Separate "Agent monitor" page, or tabs inside "Agent activity"? [separate page]
4. Polling, or SSE so the viva can say "push"? [polling]
5. Event retention? [14 days]
6. Stuck thresholds? [5 min loop, 6 min case step]
7. Fix the filename-in-audit leak? [yes, in step 4]
8. Disable CrewAI's anonymous telemetry? [yes]
9. Spend Groq budget on about 10 real case runs for the evidence table? [only if you want the table]
10. Mark orphaned "running" cycles failed at startup? [yes]
11. Age after which an unanswered donor alert needs attention? [24 h]

## 12. Ground rules carried over from the project

- **North star:** a wastage-reducing autonomous system using RAG and multi-agents. Judge every addition against it.
- **Synthetic data only.** Deployment is out of scope.
- **Commits:** don't commit until the owner says so.
- **Design decisions:** record each one in [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md). This work continues from D70, using D71–D75.
- **Permissions:** every new route uses `require_permission(...)` from the shared matrix in [schema/permissions.py](../schema/permissions.py) (D65). The parity test fails otherwise.
- **Tests:** the full suite (236 tests on 2026-10-24) must stay green: `.venv/bin/python -m pytest -q`.
