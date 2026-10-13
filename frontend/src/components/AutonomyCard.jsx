import { useEffect, useState } from "react";
import { Play, RotateCw } from "lucide-react";
import { api } from "../api";
import { COMPONENTS } from "../constants";
import { AIPanel, EmptyState, ErrorState, SeverityBadge, SkeletonLines, UpdatedAt } from "./ui";

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
        {event.message}
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
                : "Scheduler off · runs on demand"}
            </span>
          )}
          <button className="secondary" onClick={load} disabled={loading}>
            <RotateCw size={15} aria-hidden="true" /> Refresh
          </button>
          {canRun && (
            <button onClick={runNow} disabled={running}>
              <Play size={15} aria-hidden="true" /> {running ? "Running…" : "Run cycle now"}
            </button>
          )}
        </div>
      </div>
      {error && (
        <ErrorState what={running ? "The agent cycle didn't run." : "Couldn't load agent activity."}
                    error={error} onRetry={load} />
      )}

      {loading && !metrics ? <SkeletonLines lines={3} label="Loading outcomes" /> : metrics && (
        <div className="metrics">
          <div className="metric"><span>{metrics.totals.issued}</span>issued</div>
          <div className={`metric ${metrics.totals.expired > 0 ? "bad" : ""}`}><span>{metrics.totals.expired}</span>expired</div>
          <div className="metric"><span>{metrics.totals.transferred}</span>redistributed</div>
          <div className="metric">
            <span>{metrics.wastage_rate == null ? "n/a" : `${(metrics.wastage_rate * 100).toFixed(1)}%`}</span>
            wastage rate
          </div>
          <div className="metric-note muted">
            Last 30 days · expired by component: {COMPONENTS.map((c) => `${c} ${metrics.by_status.expired[c]}`).join(", ")}
          </div>
        </div>
      )}

      {!loading && !error && !cycle && (
        <EmptyState title="No cycles have run yet">
          {canRun ? "Run a cycle to check stock against the forecast." : "The scheduler will run the first cycle."}
        </EmptyState>
      )}
      {cycle && (
        <>
          <p className="cycle-meta">
            <strong>Cycle #{cycle.id}</strong>
            <span className="muted"> · {cycle.trigger} · </span>
            <UpdatedAt iso={cycle.started_at} prefix="Ran" />
            {cycle.status === "ok"
              ? <SeverityBadge level="low" label="Completed" />
              : <SeverityBadge level="moderate" label={`Completed with fallbacks (${cycle.status})`} />}
          </p>
          <div className="chips">
            {SUMMARY_LABELS.filter(([key]) => key in cycle.summary).map(([key, label, tone]) => (
              <span key={key} className={`chip ${cycle.summary[key] ? tone : ""}`}>
                {cycle.summary[key]} {label}
              </span>
            ))}
          </div>
          {cycle.brief && (
            <AIPanel label="HemoSmart AI · cycle brief">
              <p className="cycle-brief">{cycle.brief}</p>
            </AIPanel>
          )}
          <details className="agent-log" open>
            <summary>Agent log · {cycle.events.length} events</summary>
            <ol className="events">
              {cycle.events.map((e, i) => <EventRow key={i} event={e} />)}
            </ol>
          </details>
        </>
      )}
    </section>
  );
}
