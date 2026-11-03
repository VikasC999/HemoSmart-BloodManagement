import { useEffect, useState } from "react";
import { RotateCw } from "lucide-react";
import { api } from "../api";
import { BloodGroup, EmptyState, ErrorState, SeverityBadge, SkeletonLines } from "./ui";
import { formatAgo } from "./ui/format";

// One alert per row: group, component, donor, and whether they replied.
export function AlertList({ alerts, columns = false }) {
  if (alerts.length === 0) {
    return <EmptyState title="No donor alerts yet">Alerts sent by you or the agents appear here.</EmptyState>;
  }
  return (
    <ul className={`plain-list ${columns ? "alert-columns" : ""}`}>
      {alerts.map((a) => (
        <li key={a.id}>
          <span><BloodGroup g={a.blood_type} /> {a.component || ""} · {a.donor_id}</span>
          <span className="list-meta">
            {a.responded == null ? <SeverityBadge level="info" label="Pending" />
              : a.responded ? <SeverityBadge level="low" label="Donated" />
                : <span className="muted">No reply</span>}
            <span className="muted">{a.cycle_id ? "auto · " : ""}{formatAgo(a.sent_at)}</span>
          </span>
        </li>
      ))}
    </ul>
  );
}

// Beside the outreach form: what was already sent, so nobody alerts the same group twice.
export default function RecentDonorAlerts({ limit = 10 }) {
  const [alerts, setAlerts] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      setAlerts(await api.recentDonorAlerts(limit));
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  };
  useEffect(() => { load(); }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <section className="card" aria-labelledby="recent-alerts-title">
      <div className="card-head">
        <div>
          <h2 id="recent-alerts-title">Recent donor alerts</h2>
          <p className="card-sub">Sent by you or the agents, newest first</p>
        </div>
        <button className="tool" onClick={load} disabled={loading}>
          <RotateCw size={15} className={loading ? "spin" : ""} aria-hidden="true" /> Refresh
        </button>
      </div>
      {error && !alerts ? <ErrorState what="Couldn't load donor alerts." error={error} onRetry={load} />
        : !alerts ? <SkeletonLines lines={5} /> : <AlertList alerts={alerts} />}
    </section>
  );
}
