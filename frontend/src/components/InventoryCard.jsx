import { useEffect, useState } from "react";
import { api } from "../api";
import { BLOOD_TYPES, COMPONENTS } from "../constants";

// Unit-level stock: every cell is units available for one group x
// component, coloured by status against its par level, with how many of
// them are near expiry. `editable` (Blood Bank Manager) adds receiving
// stock -- the backend 403s anyone else, this just hides the control.
export default function InventoryCard({ editable = false }) {
  const [inventory, setInventory] = useState(null);
  const [receive, setReceive] = useState({ group: "O+", component: "PRBC", count: 5, daysOld: 0 });
  const [message, setMessage] = useState(null);
  const [error, setError] = useState(null);
  const [saving, setSaving] = useState(false);

  const load = async () => {
    setError(null);
    try {
      setInventory(await api.inventory());
    } catch (e) {
      setError(e.message);
    }
  };

  useEffect(() => { load(); }, []);

  const submitReceive = async () => {
    setSaving(true);
    setError(null);
    setMessage(null);
    try {
      const collected = new Date(Date.now() - receive.daysOld * 86400000).toISOString();
      const res = await api.receiveUnits(receive.group, receive.component, receive.count, collected);
      setMessage(`Received ${res.received.length} ${receive.group} ${receive.component} unit(s).`);
      await load();
    } catch (e) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <section className="card">
      <h2>Blood Inventory</h2>
      <div className="actions">
        <button onClick={load}>Refresh</button>
        {inventory && (
          <span className="muted">
            {inventory.total_available} units · {inventory.near_expiry_total} near expiry
          </span>
        )}
      </div>
      {error && <p className="error">{error}</p>}
      {inventory && (
        <table className="stock-grid">
          <thead>
            <tr><th>Group</th>{COMPONENTS.map((c) => <th key={c}>{c}</th>)}</tr>
          </thead>
          <tbody>
            {BLOOD_TYPES.map((g) => (
              <tr key={g}>
                <td><strong>{g}</strong></td>
                {COMPONENTS.map((c) => {
                  const cell = inventory.inventory[g][c];
                  return (
                    <td key={c} className={`stock-cell ${cell.status.toLowerCase()}`}
                        title={`${cell.status} — par ${cell.par}`}>
                      {cell.available}
                      {cell.near_expiry > 0 && <span className="near-count"> ⏳{cell.near_expiry}</span>}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p className="hint legend">
        <span className="stock-cell sufficient">Sufficient</span>
        <span className="stock-cell low">Low</span>
        <span className="stock-cell critical">Critical</span> vs par · ⏳ near expiry
      </p>

      {editable && (
        <div className="receive">
          <h4>Receive units</h4>
          <div className="actions">
            <select value={receive.group} onChange={(e) => setReceive({ ...receive, group: e.target.value })}>
              {BLOOD_TYPES.map((t) => <option key={t}>{t}</option>)}
            </select>
            <select value={receive.component} onChange={(e) => setReceive({ ...receive, component: e.target.value })}>
              {COMPONENTS.map((c) => <option key={c}>{c}</option>)}
            </select>
            <label className="inline">
              count
              <input type="number" min={1} max={200} value={receive.count} className="inline-edit-input"
                     onChange={(e) => setReceive({ ...receive, count: parseInt(e.target.value, 10) || 1 })} />
            </label>
            <label className="inline">
              days old
              <input type="number" min={0} value={receive.daysOld} className="inline-edit-input"
                     onChange={(e) => setReceive({ ...receive, daysOld: parseFloat(e.target.value) || 0 })} />
            </label>
            <button onClick={submitReceive} disabled={saving}>Receive</button>
          </div>
          {message && <p className="result">{message}</p>}
        </div>
      )}
    </section>
  );
}
