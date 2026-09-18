import { useState } from "react";
import { api } from "../api";
import { BLOOD_TYPES } from "../constants";

export default function DonorAlertCard() {
  const [bloodType, setBloodType] = useState("O-");
  const [message, setMessage] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const send = async () => {
    setLoading(true);
    setError(null);
    setMessage(null);
    try {
      const res = await api.alertDonors(bloodType);
      setMessage(res.message);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="card">
      <h2>Donor Alert</h2>
      <div className="actions">
        <select value={bloodType} onChange={(e) => setBloodType(e.target.value)}>
          {BLOOD_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
        </select>
        <button onClick={send} disabled={loading}>Send Alert</button>
      </div>
      {error && <p className="error">{error}</p>}
      {message && <p className="result">{message}</p>}
    </section>
  );
}
