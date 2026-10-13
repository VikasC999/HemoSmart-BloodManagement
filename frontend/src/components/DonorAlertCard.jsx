import { useState } from "react";
import { api } from "../api";
import { BLOOD_TYPES, COMPONENTS } from "../constants";

// Manual outreach. The autonomous loop already alerts donors for
// projected shortages; this is for situations it can't see coming.
export default function DonorAlertCard() {
  const [bloodType, setBloodType] = useState("O-");
  const [component, setComponent] = useState("PRBC");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const send = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await api.alertDonors(bloodType, component));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="card">
      <h2>Donor Alert</h2>
      <p className="hint">Donors are chosen by Thompson sampling over their reply history.</p>
      <div className="actions">
        <select value={bloodType} onChange={(e) => setBloodType(e.target.value)}>
          {BLOOD_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
        </select>
        <select value={component} onChange={(e) => setComponent(e.target.value)}>
          {COMPONENTS.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
        <button onClick={send} disabled={loading}>Send Alert</button>
      </div>
      {error && <p className="error">{error}</p>}
      {result && (
        <div className="result">
          <p>{result.message}</p>
          {result.alerted_donors.length > 0 && (
            <ul className="alert-list">
              {result.alerted_donors.map((d) => (
                <li key={d.alert_id} className="alert-list-item">
                  {d.name} ({d.donor_id}) — expected reply {Math.round(d.expected_response_rate * 100)}%
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}
