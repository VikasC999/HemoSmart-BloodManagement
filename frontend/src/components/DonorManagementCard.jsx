import { useState } from "react";
import { api } from "../api";
import { BLOOD_TYPES } from "../constants";

export default function DonorManagementCard() {
  const [bloodType, setBloodType] = useState("");
  const [donors, setDonors] = useState(null);
  const [error, setError] = useState(null);

  const load = async () => {
    setError(null);
    try {
      setDonors(await api.donors(bloodType || undefined));
    } catch (e) {
      setError(e.message);
    }
  };

  return (
    <section className="card">
      <h2>Donor Management</h2>
      <div className="actions">
        <select value={bloodType} onChange={(e) => setBloodType(e.target.value)}>
          <option value="">All blood types</option>
          {BLOOD_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
        </select>
        <button onClick={load}>Load</button>
      </div>
      {error && <p className="error">{error}</p>}
      {donors && (
        <table>
          <thead>
            <tr><th>ID</th><th>Name</th><th>Type</th><th>Last Donation</th><th>Response Rate</th></tr>
          </thead>
          <tbody>
            {donors.map((d) => (
              <tr key={d.donor_id}>
                <td>{d.donor_id}</td>
                <td>{d.name}</td>
                <td>{d.blood_type}</td>
                <td>{d.last_donation_date}</td>
                <td>{d.alerts_sent > 0 ? `${Math.round((d.alerts_responded / d.alerts_sent) * 100)}%` : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
