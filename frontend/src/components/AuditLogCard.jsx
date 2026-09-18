import { useState } from "react";
import { api } from "../api";

export default function AuditLogCard() {
  const [entries, setEntries] = useState(null);
  const [error, setError] = useState(null);

  const load = async () => {
    setError(null);
    try {
      setEntries(await api.auditLog(30));
    } catch (e) {
      setError(e.message);
    }
  };

  return (
    <section className="card">
      <h2>Audit Log</h2>
      <button onClick={load}>Refresh</button>
      {error && <p className="error">{error}</p>}
      {entries && (
        <div className="table-scroll">
          <table>
            <thead>
              <tr><th>Time</th><th>User</th><th>Action</th><th>Resource</th><th>Details</th></tr>
            </thead>
            <tbody>
              {entries.map((e) => (
                <tr key={e.id}>
                  <td>{e.timestamp}</td>
                  <td>{e.user_id ?? "—"}</td>
                  <td>{e.action}</td>
                  <td>{e.resource_type ? `${e.resource_type}${e.resource_id ? `:${e.resource_id}` : ""}` : "—"}</td>
                  <td className="details-cell">{e.details ? JSON.stringify(e.details) : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
