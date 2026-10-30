import { useEffect, useState } from "react";
import { RotateCw, ShieldCheck } from "lucide-react";
import { api } from "../api";
import { EmptyState, ErrorState, SkeletonTable } from "./ui";

// Details are stored as JSON; show them as readable key: value pairs.
function Details({ details }) {
  if (!details) return <span className="muted" aria-label="None">—</span>;
  // false/null/empty fields ("injection suspected: false") are noise; only what happened is shown.
  const entries = Object.entries(details).filter(([, v]) => v !== false && v !== null && v !== "" &&
                                                         !(Array.isArray(v) && v.length === 0));
  return (
    <dl className="details-list">
      {entries.map(([k, v]) => (
        <div key={k}>
          <dt>{k.replace(/_/g, " ")}</dt>
          <dd>{Array.isArray(v) ? v.join(", ") : typeof v === "object" ? JSON.stringify(v) : String(v)}</dd>
        </div>
      ))}
    </dl>
  );
}

export default function AuditLogCard() {
  const [entries, setEntries] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [showReads, setShowReads] = useState(false);

  const load = async () => {
    setError(null);
    setLoading(true);
    try {
      setEntries(await api.auditLog(200));
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  // Page loads (GET) are most of the log; actions, chat decisions and refusals are what an auditor reviews.
  const shown = entries && (showReads ? entries : entries.filter((e) => !e.action.startsWith("GET "))).slice(0, 50);

  return (
    <section className="card" aria-labelledby="audit-title">
      <div className="card-head">
        <div>
          <h2 id="audit-title">Recent activity</h2>
          {entries && (
            <p className="card-sub">
              {shown.length} most recent {showReads ? "API calls" : "actions"}, newest first · times in UTC
            </p>
          )}
        </div>
        <label className="inline audit-toggle">
          <input type="checkbox" checked={showReads} onChange={(e) => setShowReads(e.target.checked)} />
          Include page loads
        </label>
        <button className="tool" onClick={load} disabled={loading}>
          <RotateCw size={15} className={loading ? "spin" : ""} aria-hidden="true" /> Refresh
        </button>
      </div>
      {error && <ErrorState what="Couldn't load the audit log." error={error} onRetry={load} />}
      {loading && !entries && <SkeletonTable rows={8} cols={5} label="Loading audit log" />}
      {shown && shown.length === 0 && (
        <EmptyState icon={ShieldCheck} title={showReads ? "No activity recorded yet" : "No actions recorded yet"}>
          {showReads ? "Every API call is recorded here." : "Only page loads so far. Turn on Include page loads to see them."}
        </EmptyState>
      )}
      {shown && shown.length > 0 && (
        <div className="table-scroll">
          <table className="audit-table">
            <thead>
              <tr><th>Time (UTC)</th><th>User</th><th>Action</th><th>Resource</th><th>Details</th></tr>
            </thead>
            <tbody>
              {shown.map((e) => (
                <tr key={e.id}>
                  <td className="nowrap tabular">{String(e.timestamp).slice(0, 19).replace("T", " ")}</td>
                  <td className="nowrap">{e.user_email || (e.user_id != null ? `#${e.user_id}` : <span className="muted">System</span>)}</td>
                  <td className="nowrap">{e.action.replace(/_/g, " ")}</td>
                  <td className="nowrap">{e.resource_type ? `${e.resource_type}${e.resource_id ? ` #${String(e.resource_id).slice(0, 8)}` : ""}` : <span className="muted" aria-label="None">—</span>}</td>
                  <td className="details-cell"><Details details={e.details} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
