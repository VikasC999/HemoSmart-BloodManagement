import { useState } from "react";
import { Send } from "lucide-react";
import { api } from "../api";
import { BLOOD_TYPES, COMPONENTS } from "../constants";
import { ErrorState } from "./ui";

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
      setError(e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="card" aria-labelledby="outreach-title">
      <h2 id="outreach-title">Alert donors</h2>
      <p className="hint">
        HemoSmart picks the eligible donors most likely to reply, learned from their reply history.
        The agents already do this for projected shortages; use this for needs they can't see coming.
      </p>
      <form className="outreach-form" onSubmit={(e) => { e.preventDefault(); send(); }}>
        <label>
          Blood group
          <select value={bloodType} onChange={(e) => setBloodType(e.target.value)}>
            {BLOOD_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </label>
        <label>
          Component needed
          <select value={component} onChange={(e) => setComponent(e.target.value)}>
            {COMPONENTS.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </label>
        <button type="submit" disabled={loading}>
          <Send size={16} aria-hidden="true" /> {loading ? "Sending…" : "Send alerts"}
        </button>
      </form>
      {error && <ErrorState what="Alerts weren't sent." error={error} onRetry={send} />}
      {result && (
        <div className="outreach-result" role="status">
          <p className="section-title">{result.message}</p>
          {result.alerted_donors.length > 0 && (
            <ul className="plain-list">
              {result.alerted_donors.map((d) => (
                <li key={d.alert_id}>
                  <span>{d.name} <span className="muted">· {d.donor_id}</span></span>
                  <span className="muted tabular">likely to reply: {Math.round(d.expected_response_rate * 100)}%</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}
