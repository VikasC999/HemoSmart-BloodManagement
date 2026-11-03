import { useEffect, useState } from "react";
import { Play, RotateCw } from "lucide-react";
import { api } from "../api";
import { COMPONENTS } from "../constants";
import { AIPanel, EmptyState, ErrorState, SeverityBadge, SkeletonLines, UpdatedAt } from "./ui";
import { TRIGGER_LABEL, tidyText } from "./ui/format";

// Tone by meaning, not by "non-zero": risk counts use risk colours,
// good outcomes use success, activity stays neutral. Yellow is brand/AI.
const SUMMARY_LABELS = [
  ["expired_units", "expired", "risk"],
  ["shortages", "projected shortages", "risk"],
  ["escalations", "escalations", "risk"],
  ["outreach_needs", "outreach needs", "neutral"],
  ["donors_alerted", "donors alerted", "neutral"],
  ["donations", "donations", "good"],
  ["transferred_units", "transferred", "good"],
];

function EventRow({ event }) {
  const citations = event.data?.citations;
  return (
    <li className={`event kind-${event.kind}`}>
      <span className="event-agent">{event.agent}</span>
      <span className="event-kind">{event.kind}</span>
      <span className="event-msg">
        {tidyText(event.message)}
        {citations && (
          <details>
            <summary>WHO handbook</summary>
            {citations.map((c) => <p key={c.ref} className="muted">p.{c.page}: {c.excerpt}</p>)}
          </details>
        )}
      </span>
    </li>
  );
}

// The autonomous wastage-reduction loop: what the agents saw, decided and
// did in the latest cycle, and unit outcomes over the last 30 days.
// `canRun` (Blood Bank Manager) adds the manual trigger; the scheduler
// runs it on its own either way.
export default function AutonomyCard({ canRun = false }) {
  const [status, setStatus] = useState(null);
  const [cycle, setCycle] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState(null);
  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setError(null);
    setLoading(true);
    try {
      const [s, m] = await Promise.all([api.autonomyStatus(), api.wastageMetrics(30)]);
      setStatus(s);
      setCycle(s.last_cycle);
      setMetrics(m);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const runNow = async () => {
    setRunning(true);
    setError(null);
    try {
      setCycle(await api.runAutonomyCycle());
      setMetrics(await api.wastageMetrics(30));
    } catch (e) {
      setError(e);
    } finally {
      setRunning(false);
    }
  };

  return (
    <section className="card">
      <div className="card-head">
        <h2>Latest cycle</h2>
        <div className="actions" style={{ margin: 0 }}>
          {status && (
            <span className="muted">
              {status.scheduler_interval_minutes > 0
                ? `Runs every ${status.scheduler_interval_minutes} min`
                : "Scheduler off · runs when you start it"}
            </span>
          )}
          <button className="tool" onClick={load} disabled={loading}>
            <RotateCw size={15} className={loading ? "spin" : ""} aria-hidden="true" /> Refresh
          </button>
          {canRun && (
            <button onClick={runNow} disabled={running}>
              <Play size={15} aria-hidden="true" /> {running ? "Running agents…" : "Run agent cycle"}
            </button>
          )}
        </div>
      </div>
      {error && (
        <ErrorState what={running ? "The agent cycle didn't run." : "Couldn't load agent activity."}
                    error={error} onRetry={load} />
      )}

      {loading && !metrics ? <SkeletonLines lines={3} label="Loading outcomes" /> : metrics && (
        <>
          <dl className="outcomes autonomy-outcomes">
            <div><dt>Issued</dt><dd>{metrics.totals.issued}</dd></div>
            <div><dt>Expired</dt><dd className={metrics.totals.expired > 0 ? "risk" : ""}>{metrics.totals.expired}</dd></div>
            <div><dt>Redistributed</dt><dd>{metrics.totals.transferred}</dd></div>
            <div><dt>Wastage rate</dt><dd>{metrics.wastage_rate == null ? "None used" : `${(metrics.wastage_rate * 100).toFixed(1)}%`}</dd></div>
          </dl>
          <p className="metric-note muted">
            Last 30 days. Expired by component: {COMPONENTS.map((c) => `${c} ${metrics.by_status.expired[c]}`).join(", ")}
          </p>
        </>
      )}

      {!loading && !error && !cycle && (
        <EmptyState title="No cycles have run yet">
          {canRun ? "Run a cycle to check stock against the forecast and act on any gaps." : "The first cycle will run on the scheduler."}
        </EmptyState>
      )}
      {cycle && (
        <>
          <p className="cycle-meta">
            <strong>Cycle #{cycle.id}</strong>
            <span className="muted"> · {TRIGGER_LABEL[cycle.trigger] || cycle.trigger} · </span>
            <UpdatedAt iso={cycle.started_at} prefix="Ran" />
            {cycle.status === "ok"
              ? <SeverityBadge level="low" label="Completed" />
              : <SeverityBadge level="moderate" label={`Completed with fallbacks (${cycle.status})`} />}
          </p>
          {/* only what happened: a row of zeros is noise */}
          <div className="chips">
            {SUMMARY_LABELS.filter(([key]) => cycle.summary[key]).map(([key, label, tone]) => (
              <span key={key} className={`chip ${tone}`}>{cycle.summary[key]} {label}</span>
            ))}
            {!SUMMARY_LABELS.some(([key]) => cycle.summary[key]) && <span className="muted">Nothing needed action this cycle.</span>}
          </div>
          {cycle.brief && (
            <AIPanel label="HemoSmart AI · cycle brief">
              <p className="cycle-brief">{tidyText(cycle.brief)}</p>
            </AIPanel>
          )}
          <details className="agent-log" open>
            <summary>Step-by-step log · {cycle.events.length} events</summary>
            <ol className="events">
              {cycle.events.map((e, i) => <EventRow key={i} event={e} />)}
            </ol>
          </details>
        </>
      )}
    </section>
  );
}
