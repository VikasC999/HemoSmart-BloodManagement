import { useCallback, useEffect, useRef, useState } from "react";
import {
  Activity, ArrowLeft, ArrowRight, Circle, CircleAlert, CircleCheck, Database, Hourglass, Play, RotateCw, Sparkles,
  TriangleAlert,
} from "lucide-react";
import { api } from "../api";
import AgentRunTimeline from "./AgentRunTimeline";
import { EmptyState, ErrorState, SeverityBadge, Skeleton, StaleNotice, UpdatedAt } from "./ui";
import { formatAgo, parseTime, runLabel, TRIGGER_LABEL } from "./ui/format";

// The Agent monitor as a flat (2D) office floor: one room per agent
// layer, one desk per agent, corridors to the shared blood-bank data in
// the middle, and a task panel on the right. Read-only (D74).

// Polling, not push (D73): fast while a run is open, slow when idle,
// paused while the tab is hidden.
const FAST_MS = 2000;
const IDLE_MS = 20000;

// Status: icon + word, never colour alone (D64).
const STATUS = {
  working: { icon: Activity, label: "Working" },
  waiting: { icon: Hourglass, label: "Waiting" },
  degraded: { icon: TriangleAlert, label: "Degraded" },
  failed: { icon: CircleAlert, label: "Failed" },
  idle: { icon: Circle, label: "Idle" },
};

const ms = (v) => (v == null ? null : v < 1000 ? `${v} ms` : `${(v / 1000).toFixed(v < 10000 ? 1 : 0)} s`);
const pct = (n, d) => (d ? `${Math.round((n / d) * 100)}%` : "None yet");
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

function usePolling(load, fast) {
  const [hidden, setHidden] = useState(() => document.hidden);
  useEffect(() => {
    const onChange = () => setHidden(document.hidden);
    document.addEventListener("visibilitychange", onChange);
    return () => document.removeEventListener("visibilitychange", onChange);
  }, []);
  useEffect(() => {
    if (hidden) return undefined;
    load();
    const id = setInterval(load, fast ? FAST_MS : IDLE_MS);
    return () => clearInterval(id);
  }, [hidden, fast, load]);
}

// Announces status changes politely, at most once every few seconds,
// so a screen reader isn't flooded during a fast run.
function useAnnouncer() {
  const [message, setMessage] = useState("");
  const last = useRef(0);
  const queued = useRef([]);
  const announce = useCallback((lines) => {
    queued.current.push(...lines);
    const now = Date.now();
    if (now - last.current < 5000 || !queued.current.length) return;
    last.current = now;
    setMessage(queued.current.slice(-3).join(". "));
    queued.current = [];
  }, []);
  return [message, announce];
}

function useClock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return now;
}

// doing / next / done for a room: the open run if there is one,
// otherwise how the last run ended.
function roomCounts(layer, openRun) {
  const run = openRun || layer.last_run;
  if (!run) return { doing: 0, next: 0, done: 0 };
  if (!run.open) return { doing: 0, next: 0, done: run.steps_done };
  return { doing: 1, next: Math.max(0, run.steps_total - run.steps_done - 1), done: run.steps_done };
}

// ---------------------------------------------------------------------
// Office floor
// ---------------------------------------------------------------------
function Desk({ name, status, note, selected, onClick, kind }) {
  const s = STATUS[status] || STATUS.idle;
  const Icon = s.icon;
  return (
    <li>
      <button className={`desk s-${status} ${selected ? "selected" : ""} ${kind === "llm" ? "llm" : ""}`}
              onClick={onClick} aria-pressed={selected}
              aria-label={`${name} agent: ${s.label}${note ? `, ${note}` : ""}`}>
        <span className="desk-top" aria-hidden="true"><span className="desk-screen" /></span>
        <span className="desk-avatar" aria-hidden="true">
          {kind === "llm" ? <Sparkles size={13} /> : name.slice(0, 2)}
        </span>
        <span className="desk-name">{name}</span>
        <span className="desk-state"><Icon size={12} aria-hidden="true" />{s.label}</span>
        {note && <span className="desk-note">{note}</span>}
      </button>
    </li>
  );
}

function Room({ id, title, tone, count, unit = "agents", stats = [], counts, active, roomRef, children }) {
  return (
    <section className={`room room-${id} tone-${tone} ${active ? "active" : ""}`} aria-label={title} ref={roomRef}>
      <div className="room-plaque">
        <p className="room-title"><span className="room-dot" aria-hidden="true" />{title}</p>
        <p className="room-count">{count}<span>{unit}</span></p>
        {stats.length > 0 && (
          <dl className="room-stats">
            {stats.map(([label, value]) => (
              <div key={label}><dt>{label}</dt><dd>{value}</dd></div>
            ))}
          </dl>
        )}
        {counts && (
          <p className="room-flow">
            <span>Doing <b>{counts.doing}</b></span><span>Next <b>{counts.next}</b></span><span>Done <b>{counts.done}</b></span>
          </p>
        )}
      </div>
      <ul className="desks">{children}</ul>
    </section>
  );
}

// Dotted corridors from the shared data hub to each room, measured from
// the laid-out rooms so they follow any screen size. Decorative only.
// A passive effect, not a layout one: the floor's ref is attached after
// this child's layout effects run, so it would still be null there.
function Corridors({ floor, hub, rooms, active, version }) {
  const [lines, setLines] = useState([]);
  useEffect(() => {
    const measure = () => {
      if (!floor.current || !hub.current) return;
      const base = floor.current.getBoundingClientRect();
      const h = hub.current.getBoundingClientRect();
      const hx = h.left + h.width / 2 - base.left;
      const hy = h.top + h.height / 2 - base.top;
      setLines(Object.entries(rooms).flatMap(([id, ref]) => {
        if (!ref.current) return [];
        const r = ref.current.getBoundingClientRect();
        // Aim at the nearest point of the room's edge.
        const x = Math.min(Math.max(hx, r.left - base.left), r.right - base.left);
        const y = Math.min(Math.max(hy, r.top - base.top), r.bottom - base.top);
        return [{ id, x1: hx, y1: hy, x2: x, y2: y }];
      }));
    };
    measure();
    if (!floor.current || typeof ResizeObserver === "undefined") return undefined;
    const observer = new ResizeObserver(measure);
    observer.observe(floor.current);
    return () => observer.disconnect();
  }, [floor, hub, rooms, version]);
  return (
    <svg className="corridors" aria-hidden="true">
      {lines.map((l) => (
        <line key={l.id} x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2} className={active[l.id] ? "active" : ""} />
      ))}
    </svg>
  );
}

function Floor({ data, wastage, selected, onSelect }) {
  const floor = useRef(null);
  const hub = useRef(null);
  const rooms = useRef({
    supply_loop: { current: null }, case_pipeline: { current: null },
    scheduler: { current: null }, partner: { current: null }, assistant: { current: null },
  }).current;

  const layer = (id) => data.layers.find((l) => l.layer === id);
  const openRun = (id) => data.open_runs.find((r) => r.layer === id);
  const supply = layer("supply_loop");
  const cases = layer("case_pipeline");
  const { cycles_7d: c, case_steps_7d: steps, llm_calls_today: calls, llm_daily_limit: limit } = data.metrics;
  const sched = data.scheduler;
  const a = data.assistant;
  const logistics = supply.cards.find((x) => x.agent === "Logistics");
  const active = {
    supply_loop: Boolean(openRun("supply_loop")), case_pipeline: Boolean(openRun("case_pipeline")),
    partner: logistics?.status === "working", scheduler: false, assistant: false,
  };

  const deskNote = (card) => (card.status === "waiting" && card.retry_in_s != null
    ? `retry in ${card.retry_in_s} s` : card.current ? card.current.step : null);

  const agentDesks = (l) => l.cards.map((card) => (
    <Desk key={card.agent} name={card.agent} status={card.status} kind={card.kind} note={deskNote(card)}
          selected={selected === `${card.layer}/${card.agent}`} onClick={() => onSelect(`${card.layer}/${card.agent}`)} />
  ));

  return (
    <div className="office-floor" ref={floor}>
      <Corridors floor={floor} hub={hub} rooms={rooms} active={active} version={data.now} />

      <Room id="supply" tone="info" title="Supply loop" count={supply.cards.length} roomRef={rooms.supply_loop}
            active={active.supply_loop} counts={roomCounts(supply, openRun("supply_loop"))}
            stats={[["Cycles OK · 7 d", `${c.ok} / ${c.ok + c.degraded + c.failed}`],
              ["Expired · 30 d", wastage ? wastage.totals.expired : "…"]]}>
        {agentDesks(supply)}
      </Room>

      <div className="office-hub" ref={hub}>
        <Database size={18} aria-hidden="true" />
        <p className="hub-title">Blood bank data</p>
        <p className="hub-sub">Stock · donors · forecast · WHO handbook</p>
      </div>

      <Room id="case" tone="ai" title="Case pipeline" count={cases.cards.length} roomRef={rooms.case_pipeline}
            active={active.case_pipeline} counts={roomCounts(cases, openRun("case_pipeline"))}
            stats={[["Answers verified · 7 d", pct(steps.verified, steps.total)],
              ["LLM calls today", `${calls} / ${limit.toLocaleString()}`]]}>
        {agentDesks(cases)}
      </Room>

      <Room id="scheduler" tone="neutral" title="Scheduler" count={sched.interval_min > 0 ? sched.interval_min : "Off"}
            unit={sched.interval_min > 0 ? "min interval" : ""} roomRef={rooms.scheduler}
            stats={[["Last tick", sched.last_heartbeat ? formatAgo(sched.last_heartbeat) : "None yet"]]}>
        <Desk name="Scheduler" kind="rule"
              status={sched.state === "late" ? "failed" : "idle"}
              note={sched.state === "off" ? "cycles run by hand"
                : sched.state === "late" ? "missed a tick"
                  : sched.next_due ? `next ${formatIn(sched.next_due)}` : "waiting for first tick"}
              selected={selected === "scheduler"} onClick={() => onSelect("scheduler")} />
      </Room>

      <Room id="partner" tone="neutral" title="Partner hospital" count={wastage ? wastage.totals.transferred : "…"}
            unit="units sent · 30 d" roomRef={rooms.partner}>
        <Desk name="Mock hospital" kind="rule" status={active.partner ? "working" : "idle"}
              note="via MCP adapter" selected={selected === "partner"} onClick={() => onSelect("partner")} />
      </Room>

      <Room id="assistant" tone="ai" title="Assistant" count={a.messages_24h} unit="messages · 24 h"
            roomRef={rooms.assistant}
            stats={[["Refused", (a.outcomes_24h.denied || 0) + (a.outcomes_24h.partial || 0)], ["Used the LLM", a.llm_used_24h]]}>
        <Desk name="Assistant" kind="llm" status={a.status}
              note={a.degraded_24h ? `${a.degraded_24h} in limited mode` : null}
              selected={selected === "assistant"} onClick={() => onSelect("assistant")} />
      </Room>
    </div>
  );
}

function formatIn(iso) {
  const minutes = Math.round((parseTime(iso).getTime() - Date.now()) / 60000);
  return minutes <= 0 ? "due now" : `in ${minutes} min`;
}

// ---------------------------------------------------------------------
// Task panel
// ---------------------------------------------------------------------
function runState(run, waitingRuns) {
  if (!run.open) return "done";
  return waitingRuns.has(run.run_id) ? "waiting" : "progress";
}

const RUN_RESULT = {
  ok: ["low", "Completed"], done: ["low", "Completed"], degraded: ["moderate", "With fallbacks"],
  failed: ["high", "Failed"],
};

function TaskRow({ run, state, currentStep, onOpen }) {
  const progress = Math.round((run.steps_done / run.steps_total) * 100);
  const [level, result] = RUN_RESULT[run.status] || ["info", run.status];
  return (
    <li>
      <button className={`task ${state}`} onClick={() => onOpen(run.run_id)}>
        <span className="task-pct">{progress}%</span>
        <span className="task-body">
          <span className="task-title">{run.label}</span>
          <span className="task-sub">
            {state === "done" ? <SeverityBadge level={level} label={result} />
              : <>{state === "waiting" ? "Waiting on the LLM rate limit" : currentStep || "Starting"}</>}
            {" · "}{run.layer === "case_pipeline" ? "Case pipeline" : `Supply loop${run.trigger ? `, ${TRIGGER_LABEL[run.trigger] || run.trigger}` : ""}`}
          </span>
          <span className="task-bar" aria-hidden="true"><span style={{ width: `${progress}%` }} /></span>
        </span>
        <span className="task-time">{formatAgo(run.started_at)}</span>
      </button>
    </li>
  );
}

function AgentDetail({ id, data, wastage, onBack }) {
  let title, rows = [], card = null;
  if (id === "scheduler") {
    const s = data.scheduler;
    title = "Scheduler";
    rows = [["State", { off: "Off: cycles run by hand", late: "Late: missed a tick", ok: "Running", unknown: "No heartbeat yet" }[s.state]],
      ["Interval", s.interval_min > 0 ? `${s.interval_min} min` : "None"],
      ["Last tick", s.last_heartbeat ? formatAgo(s.last_heartbeat) : "None yet"],
      ["Next tick", s.next_due ? formatIn(s.next_due) : "None"]];
  } else if (id === "partner") {
    title = "Partner hospital";
    rows = [["What it is", "A mock partner hospital behind the MCP adapter; the Logistics agent offers it near-expiry surplus."],
      ["Units sent · 30 d", wastage ? wastage.totals.transferred : "Loading"]];
  } else if (id === "assistant") {
    const a = data.assistant;
    title = "Assistant";
    rows = [["Messages · 24 h", a.messages_24h], ["Answered", a.outcomes_24h.answered || 0],
      ["Partly refused", a.outcomes_24h.partial || 0], ["Refused", a.outcomes_24h.denied || 0],
      ["Used the LLM", a.llm_used_24h], ["Limited mode", a.degraded_24h],
      ["Last message", a.last_activity ? formatAgo(a.last_activity) : "None yet"]];
  } else {
    const [layer, agent] = id.split("/");
    card = data.layers.find((l) => l.layer === layer)?.cards.find((c) => c.agent === agent);
    if (!card) return null;
    title = `${agent} agent`;
    const n = card.counts_24h;
    rows = [["Layer", layer === "case_pipeline" ? "Case pipeline" : "Supply loop"],
      ["Kind", card.kind === "llm" ? "LLM agent" : "Rule-based"],
      ["Now", card.current ? `${runLabel(card.current.run_id)} · ${card.current.step} (${card.current.done} of ${card.current.total} done)` : "No step in progress"],
      ["24 h", `${n.ok} ok · ${n.fallback} fallback · ${plural(n.error, "error")}`],
      ["Typical step", ms(card.p50_ms) || "No data"], ["Slow step (p95)", ms(card.p95_ms) || "No data"],
      ["Last active", card.last_activity ? formatAgo(card.last_activity) : "Hasn't run yet"]];
  }
  const status = card ? card.status : null;
  return (
    <div className="agent-detail">
      <button className="tool" onClick={onBack}><ArrowLeft size={15} aria-hidden="true" /> All tasks</button>
      <h3>{title}</h3>
      {status && <StatusLine status={status} retryIn={card.retry_in_s} />}
      <dl className="detail-list">
        {rows.map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}
      </dl>
    </div>
  );
}

function StatusLine({ status, retryIn }) {
  const s = STATUS[status] || STATUS.idle;
  const Icon = s.icon;
  return (
    <p className={`status-line s-${status}`}>
      <Icon size={14} aria-hidden="true" /> {status === "waiting" && retryIn != null ? `Waiting · retry in ${retryIn} s` : s.label}
    </p>
  );
}

function TaskPanel({ data, runs, runsError, onRetryRuns, links, onOpenRun, selected, onSelect, wastage }) {
  const [filter, setFilter] = useState("all");
  const waitingRuns = new Set(data.layers.flatMap((l) => l.cards)
    .filter((c) => c.status === "waiting" && c.current).map((c) => c.current.run_id));
  // Open runs from the overview are fresher than the runs list.
  const byId = new Map((runs || []).map((r) => [r.run_id, r]));
  data.open_runs.forEach((r) => byId.set(r.run_id, r));
  const all = [...byId.values()].sort((x, y) => parseTime(y.started_at) - parseTime(x.started_at));
  const currentStep = (run) => data.layers.flatMap((l) => l.cards)
    .find((c) => c.current?.run_id === run.run_id && c.status === "working")?.agent;

  const counts = { all: all.length, progress: 0, waiting: 0, done: 0 };
  all.forEach((r) => { counts[runState(r, waitingRuns)] += 1; });
  const shown = filter === "attention" ? [] : all.filter((r) => filter === "all" || runState(r, waitingRuns) === filter);

  if (selected) {
    return (
      <aside className="task-panel" aria-label="Agent details">
        <AgentDetail id={selected} data={data} wastage={wastage} onBack={() => onSelect(null)} />
      </aside>
    );
  }

  const chips = [["all", "All"], ["progress", "In progress"], ["waiting", "Waiting"], ["done", "Done"]];
  return (
    <aside className="task-panel" aria-label="Task status">
      <h2 className="panel-title">Task status</h2>
      <div className="task-chips" role="group" aria-label="Filter tasks">
        {chips.map(([id, label]) => (
          <button key={id} className={`task-chip ${filter === id ? "on" : ""} ${id === "waiting" ? "waiting" : ""}`}
                  aria-pressed={filter === id} onClick={() => setFilter(id)}>
            {label} <b>{counts[id]}</b>
          </button>
        ))}
        <button className={`task-chip attention ${filter === "attention" ? "on" : ""}`}
                aria-pressed={filter === "attention"} onClick={() => setFilter("attention")}>
          Needs attention <b>{data.attention.length}</b>
        </button>
      </div>

      {filter === "attention" ? (
        data.attention.length === 0 ? (
          <EmptyState icon={CircleCheck} title="Nothing needs a person right now">
            No cycle or case is waiting on a decision.
          </EmptyState>
        ) : (
          <ul className="attention-stack">
            {data.attention.map((item, i) => {
              const target = links[item.link];
              return (
                <li key={`${item.kind}-${i}`} className="attention-note">
                  <SeverityBadge level={item.severity} />
                  <p><b>{item.title}.</b> {item.detail}</p>
                  {item.link === "agentMonitor" && item.run_id ? (
                    <button className="tool" onClick={() => onOpenRun(item.run_id)}>
                      Timeline <ArrowRight size={14} aria-hidden="true" />
                    </button>
                  ) : target && (
                    <a className="attention-link" href={`#${item.link}`}>{target} <ArrowRight size={14} aria-hidden="true" /></a>
                  )}
                </li>
              );
            })}
          </ul>
        )
      ) : runsError ? (
        <ErrorState what="Couldn't load recent runs." error={runsError} onRetry={onRetryRuns}
                    stillValid="The office floor is still current." />
      ) : runs == null ? (
        <div role="status" aria-label="Loading tasks" className="task-list">
          {Array.from({ length: 4 }, (_, i) => <Skeleton key={i} height={56} />)}
        </div>
      ) : shown.length === 0 ? (
        <EmptyState title={all.length ? "No tasks with this status" : "No agent runs recorded yet"}>
          {all.length ? "Choose another filter." : "Runs appear here when a cycle runs or a CBC report goes through the AI agents."}
        </EmptyState>
      ) : (
        <ul className="task-list">
          {shown.map((r) => (
            <TaskRow key={r.run_id} run={r} state={runState(r, waitingRuns)} currentStep={currentStep(r)} onOpen={onOpenRun} />
          ))}
        </ul>
      )}
    </aside>
  );
}

// ---------------------------------------------------------------------
// The page
// ---------------------------------------------------------------------
// `links` maps the section ids this role can open to their labels, for
// the attention items' cross-links.
export default function AgentMonitor({ canRun = false, links = {} }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [lastOk, setLastOk] = useState(null);
  const [runs, setRuns] = useState(null);
  const [runsError, setRunsError] = useState(null);
  const [wastage, setWastage] = useState(null);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState(null);
  const [openRun, setOpenRun] = useState(null);
  const [selected, setSelected] = useState(null);
  const [message, announce] = useAnnouncer();
  const clock = useClock();
  const previous = useRef({});
  const runsKey = useRef(null);

  const loadRuns = useCallback(async () => {
    setRunsError(null);
    try {
      setRuns(await api.agentRuns(12));
    } catch (e) {
      setRunsError(e);
    }
  }, []);

  const loadWastage = useCallback(() => api.wastageMetrics(30).then(setWastage).catch(() => setWastage(null)), []);

  const load = useCallback(async () => {
    try {
      const d = await api.agentOverview();
      setData(d);
      setError(null);
      setLastOk(Date.now());
      const changes = [];
      for (const layer of d.layers) {
        for (const card of layer.cards) {
          const key = `${card.layer}/${card.agent}`;
          if (previous.current[key] && previous.current[key] !== card.status) {
            changes.push(`${card.agent} agent ${STATUS[card.status]?.label.toLowerCase() || card.status}`);
          }
          previous.current[key] = card.status;
        }
      }
      if (changes.length) announce(changes);
      // Reload the task list when a run starts or finishes.
      const key = d.open_runs.map((r) => r.run_id).join(",");
      if (key !== runsKey.current) {
        runsKey.current = key;
        loadRuns();
      }
    } catch (e) {
      setError(e);
    }
  }, [announce, loadRuns]);

  useEffect(() => { loadWastage(); }, [loadWastage]);
  usePolling(load, running || Boolean(data?.has_open_run));

  const runNow = async () => {
    setRunning(true);
    setRunError(null);
    try {
      await api.runAutonomyCycle();
      loadWastage();
    } catch (e) {
      setRunError(e);
    } finally {
      setRunning(false);
      load();
    }
  };

  const interval = running || data?.has_open_run ? FAST_MS : IDLE_MS;
  const stale = error && lastOk && Date.now() - lastOk > 2 * interval;
  const sched = data?.scheduler;

  return (
    <div className="monitor">
      <div className="office-bar">
        <p className="office-name">Agent office</p>
        <p className="office-chip">
          <span className={`live-dot ${sched?.state === "late" ? "late" : ""}`} aria-hidden="true" />
          {!sched ? "Connecting" : sched.state === "off" ? "Scheduler off" : sched.state === "late" ? "Scheduler late" : `Scheduler every ${sched.interval_min} min`}
        </p>
        <p className="office-chip muted-chip">Works with: Groq LLM · WHO handbook (RAG) · partner hospital (mock MCP)</p>
        <div className="office-actions">
          <time className="office-clock" dateTime={clock.toISOString()}>
            {clock.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
          </time>
          <button className="tool" onClick={load} aria-label="Refresh"><RotateCw size={15} aria-hidden="true" /></button>
          {canRun && (
            <button onClick={runNow} disabled={running || data?.has_open_run}>
              <Play size={15} aria-hidden="true" /> {running ? "Running agents…" : "Run agent cycle"}
            </button>
          )}
        </div>
      </div>
      <p className="visually-hidden" aria-live="polite">{message}</p>

      {runError && <ErrorState what="The agent cycle didn't run." error={runError} />}
      {error && !data && (
        <ErrorState what="Couldn't load the agent monitor." error={error} onRetry={load}
                    stillValid={/404/.test(String(error?.message)) ? "The API server may be older than this page; restart it to pick up the monitor routes." : undefined} />
      )}
      {stale && (
        <StaleNotice>
          Lost contact with the server {formatAgo(new Date(lastOk).toISOString())}. The office below may be out of date.
        </StaleNotice>
      )}

      {!data && !error && (
        <div className="office" role="status" aria-label="Loading the agent office">
          <div className="office-floor loading"><Skeleton height={260} /><Skeleton height={260} /></div>
          <aside className="task-panel"><Skeleton height={200} /></aside>
        </div>
      )}

      {data && (
        <div className="office">
          <div className="office-main">
            <Floor data={data} wastage={wastage} selected={selected} onSelect={setSelected} />
            <p className="office-legend">
              <span><span className="legend-desk llm" aria-hidden="true" /> LLM agent</span>
              <span><span className="legend-desk" aria-hidden="true" /> Rule-based agent</span>
              <span><Activity size={12} aria-hidden="true" /> Working</span>
              <span><Hourglass size={12} aria-hidden="true" /> Waiting on the LLM rate limit</span>
              <span><TriangleAlert size={12} aria-hidden="true" /> Degraded</span>
              <span><CircleAlert size={12} aria-hidden="true" /> Failed</span>
              <span className="legend-note">Select a desk for details. For what each cycle decided, see{" "}
                {links.agents ? <a href="#agents">{links.agents}</a> : "Agent activity"}.</span>
            </p>
          </div>
          <TaskPanel data={data} runs={runs} runsError={runsError} onRetryRuns={loadRuns} links={links}
                     onOpenRun={setOpenRun} selected={selected} onSelect={setSelected} wastage={wastage} />
        </div>
      )}

      {data && <UpdatedAt iso={data.now} />}

      {openRun && (
        <AgentRunTimeline runId={openRun} onClose={() => setOpenRun(null)}
                          live={Boolean(data?.open_runs.some((r) => r.run_id === openRun))}
                          cycleLink={links.agents} />
      )}
    </div>
  );
}
