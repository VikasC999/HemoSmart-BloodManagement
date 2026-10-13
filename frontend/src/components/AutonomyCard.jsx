import { useEffect, useState } from "react";
import { api } from "../api";
import { COMPONENTS } from "../constants";

const SUMMARY_LABELS = [
  ["expired_units", "expired"],
  ["shortages", "projected shortages"],
  ["outreach_needs", "outreach needs"],
  ["donors_alerted", "donors alerted"],
  ["donations", "donations"],
  ["transferred_units", "transferred"],
  ["escalations", "escalations"],
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

  const load = async () => {
    setError(null);
    try {
      const [s, m] = await Promise.all([api.autonomyStatus(), api.wastageMetrics(30)]);
      setStatus(s);
      setCycle(s.last_cycle);
      setMetrics(m);
    } catch (e) {
      setError(e.message);
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
      setError(e.message);
    } finally {
      setRunning(false);
    }
  };

  return (
    <section className="card card-wide">
      <h2>Autonomous Agents</h2>
      <div className="actions">
        {canRun && <button onClick={runNow} disabled={running}>{running ? "Running…" : "Run cycle now"}</button>}
        <button className="secondary" onClick={load}>Refresh</button>
        {status && (
          <span className="muted">
            {status.scheduler_interval_minutes > 0
              ? `Runs every ${status.scheduler_interval_minutes} min`
              : "Scheduler off"}
          </span>
        )}
      </div>
      {error && <p className="error">{error}</p>}

      {metrics && (
        <div className="metrics">
          <div className="metric"><span>{metrics.totals.issued}</span>issued</div>
          <div className="metric bad"><span>{metrics.totals.expired}</span>expired</div>
          <div className="metric good"><span>{metrics.totals.transferred}</span>redistributed</div>
          <div className="metric">
            <span>{metrics.wastage_rate == null ? "—" : `${(metrics.wastage_rate * 100).toFixed(1)}%`}</span>
            wastage rate
          </div>
          <div className="metric-note muted">
            last 30 days · expired by component: {COMPONENTS.map((c) => `${c} ${metrics.by_status.expired[c]}`).join(", ")}
          </div>
        </div>
      )}

      {!cycle && <p className="muted">No cycles have run yet.</p>}
      {cycle && (
        <>
          <p>
            <strong>Cycle #{cycle.id}</strong> · {cycle.trigger} · {cycle.clock.slice(0, 16).replace("T", " ")} UTC ·{" "}
            <span className={`badge ${cycle.status === "ok" ? "ok" : "near"}`}>{cycle.status}</span>
          </p>
          <div className="chips">
            {SUMMARY_LABELS.filter(([key]) => key in cycle.summary).map(([key, label]) => (
              <span key={key} className={`chip ${cycle.summary[key] ? "on" : ""}`}>
                {cycle.summary[key]} {label}
              </span>
            ))}
          </div>
          {cycle.brief && <p className="brief">{cycle.brief}</p>}
          <details open>
            <summary>Agent log ({cycle.events.length} events)</summary>
            <ol className="events">
              {cycle.events.map((e, i) => <EventRow key={i} event={e} />)}
            </ol>
          </details>
        </>
      )}
    </section>
  );
}
