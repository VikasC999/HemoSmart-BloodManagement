import { useEffect, useState } from "react";
import { RotateCw, ShieldCheck } from "lucide-react";
import { api } from "../api";
import { EmptyState, ErrorState, SkeletonTable } from "./ui";

// Details are stored as JSON; show them as readable key: value pairs.
function Details({ details }) {
  if (!details) return <span className="muted">None</span>;
  const entries = Object.entries(details);
  return (
    <dl className="details-list">
      {entries.map(([k, v]) => (
        <div key={k}>
          <dt>{k.replace(/_/g, " ")}</dt>
          <dd>{typeof v === "object" && v !== null ? JSON.stringify(v) : String(v)}</dd>
        </div>
      ))}
    </dl>
  );
}

export default function AuditLogCard() {
  const [entries, setEntries] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setError(null);
    setLoading(true);
    try {
      setEntries(await api.auditLog(30));
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  return (
    <section className="card" aria-labelledby="audit-title">
      <div className="card-head">
        <div>
          <h2 id="audit-title">Recent activity</h2>
          {entries && <p className="card-sub">The 30 most recent API calls, newest first · times in UTC</p>}
        </div>
        <button className="tool" onClick={load} disabled={loading}>
          <RotateCw size={15} className={loading ? "spin" : ""} aria-hidden="true" /> Refresh
        </button>
      </div>
      {error && <ErrorState what="Couldn't load the audit log." error={error} onRetry={load} />}
      {loading && !entries && <SkeletonTable rows={8} cols={5} label="Loading audit log" />}
      {entries && entries.length === 0 && (
        <EmptyState icon={ShieldCheck} title="No activity recorded yet">Every API call is recorded here.</EmptyState>
      )}
      {entries && entries.length > 0 && (
        <div className="table-scroll">
          <table className="audit-table">
            <thead>
              <tr><th>Time (UTC)</th><th>User</th><th>Action</th><th>Resource</th><th>Details</th></tr>
            </thead>
            <tbody>
              {entries.map((e) => (
                <tr key={e.id}>
                  <td className="nowrap tabular">{String(e.timestamp).slice(0, 19).replace("T", " ")}</td>
                  <td>{e.user_id != null ? `#${e.user_id}` : <span className="muted">System</span>}</td>
                  <td className="nowrap">{e.action.replace(/_/g, " ")}</td>
                  <td className="nowrap">{e.resource_type ? `${e.resource_type}${e.resource_id ? ` #${e.resource_id}` : ""}` : <span className="muted">None</span>}</td>
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
