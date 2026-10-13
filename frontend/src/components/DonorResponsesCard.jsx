import { useEffect, useState } from "react";
import { api } from "../api";

// The bandit's learning signal: a coordinator records each donor's real
// reply. "Donated" also adds the donation's units to stock. (In a
// synthetic run with simulated replies, the autonomous loop resolves
// these itself.)
export default function DonorResponsesCard() {
  const [pending, setPending] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(null);
  const [last, setLast] = useState(null);

  const load = async () => {
    setError(null);
    try {
      setPending(await api.recentDonorAlerts(25, true));
    } catch (e) {
      setError(e.message);
    }
  };

  useEffect(() => { load(); }, []);

  const record = async (alert, responded) => {
    setBusy(alert.id);
    setError(null);
    try {
      const res = await api.recordDonorResponse(alert.id, responded);
      setLast(responded
        ? `${alert.donor_id} donated — ${res.units_received.length} unit(s) added to stock.`
        : `${alert.donor_id} recorded as no response.`);
      await load();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(null);
    }
  };

  return (
    <section className="card">
      <h2>Donor Responses</h2>
      <button className="secondary" onClick={load}>Refresh</button>
      {error && <p className="error">{error}</p>}
      {last && <p className="result">{last}</p>}
      {pending && (pending.length === 0 ? <p className="muted">No alerts awaiting a reply.</p> : (
        <table>
          <thead><tr><th>Donor</th><th>Need</th><th>Sent</th><th></th></tr></thead>
          <tbody>
            {pending.map((a) => (
              <tr key={a.id}>
                <td>{a.donor_id}</td>
                <td>{a.blood_type} {a.component || "PRBC"}{a.cycle_id && <span className="badge info">auto</span>}</td>
                <td>{a.sent_at.slice(5, 16).replace("T", " ")}</td>
                <td className="nowrap">
                  <button onClick={() => record(a, true)} disabled={busy === a.id}>Donated</button>{" "}
                  <button className="secondary" onClick={() => record(a, false)} disabled={busy === a.id}>No reply</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ))}
    </section>
  );
}
