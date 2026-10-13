import { useEffect, useState } from "react";
import { Hourglass, OctagonAlert, PackagePlus, RotateCw, TriangleAlert } from "lucide-react";
import { api } from "../api";
import { BLOOD_TYPES, COMPONENTS } from "../constants";
import { ErrorState, SkeletonTable, STOCK_LABEL, UpdatedAt } from "./ui";

const STATUS_ICON = { Critical: OctagonAlert, Low: TriangleAlert };

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
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setError(null);
    setLoading(true);
    try {
      setInventory(await api.inventory());
    } catch (e) {
      setError({ what: "Couldn't load inventory.", error: e });
    } finally {
      setLoading(false);
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
      setError({ what: "Units weren't received.", error: e });
    } finally {
      setSaving(false);
    }
  };

  return (
    <section className="card" aria-labelledby="inventory-title">
      <div className="card-head">
        <div>
          <h2 id="inventory-title">Blood inventory</h2>
          {inventory && (
            <p className="card-sub">
              {inventory.total_available} units available · {inventory.near_expiry_total} near expiry ·{" "}
              <UpdatedAt iso={inventory.as_of} />
            </p>
          )}
        </div>
        <button className="secondary" onClick={load} disabled={loading}>
          <RotateCw size={15} aria-hidden="true" /> Refresh
        </button>
      </div>
      {error && <ErrorState what={error.what} error={error.error} onRetry={load} />}
      {loading && !inventory ? <SkeletonTable rows={8} cols={4} label="Loading inventory" /> : inventory && (
        <div className="table-scroll">
          <table className="stock-grid">
            <caption className="visually-hidden">
              Units available by blood group and component, with status against par level
            </caption>
            <thead>
              <tr><th scope="col">Group</th>{COMPONENTS.map((c) => <th key={c} scope="col">{c}</th>)}</tr>
            </thead>
            <tbody>
              {BLOOD_TYPES.map((g) => (
                <tr key={g}>
                  <th scope="row">{g}</th>
                  {COMPONENTS.map((c) => {
                    const cell = inventory.inventory[g][c];
                    const Icon = STATUS_ICON[cell.status];
                    const label = `${g} ${c}: ${cell.available} units, ${STOCK_LABEL[cell.status] || cell.status} (par ${Math.round(cell.par)})`
                      + (cell.near_expiry ? `, ${cell.near_expiry} near expiry` : "");
                    return (
                      <td key={c}>
                        <span className={`stock-cell ${cell.status.toLowerCase()}`} title={label} aria-label={label}>
                          {Icon && <Icon size={13} aria-hidden="true" />}
                          <span className="stock-count">{cell.available}</span>
                          {cell.near_expiry > 0 && (
                            <span className="near-count" aria-hidden="true">
                              <Hourglass size={11} />{cell.near_expiry}
                            </span>
                          )}
                        </span>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <ul className="legend" aria-label="Legend">
        <li><span className="stock-cell sufficient legend-key">Sufficient</span></li>
        <li><span className="stock-cell low legend-key"><TriangleAlert size={13} aria-hidden="true" /> Low stock</span></li>
        <li><span className="stock-cell critical legend-key"><OctagonAlert size={13} aria-hidden="true" /> Critical</span></li>
        <li className="muted">against par level ·</li>
        <li className="muted legend-near"><Hourglass size={12} aria-hidden="true" /> units near expiry</li>
      </ul>

      {editable && (
        <form className="receive" onSubmit={(e) => { e.preventDefault(); submitReceive(); }}>
          <h3 className="section-title">Receive units</h3>
          <div className="receive-grid">
            <label>
              Blood group
              <select value={receive.group} onChange={(e) => setReceive({ ...receive, group: e.target.value })}>
                {BLOOD_TYPES.map((t) => <option key={t}>{t}</option>)}
              </select>
            </label>
            <label>
              Component
              <select value={receive.component} onChange={(e) => setReceive({ ...receive, component: e.target.value })}>
                {COMPONENTS.map((c) => <option key={c}>{c}</option>)}
              </select>
            </label>
            <label>
              Units
              <input type="number" min={1} max={200} value={receive.count}
                     onChange={(e) => setReceive({ ...receive, count: parseInt(e.target.value, 10) || 1 })} />
            </label>
            <label>
              Collected (days ago)
              <input type="number" min={0} value={receive.daysOld}
                     onChange={(e) => setReceive({ ...receive, daysOld: parseFloat(e.target.value) || 0 })} />
            </label>
          </div>
          <div className="actions">
            <button type="submit" disabled={saving}>
              <PackagePlus size={16} aria-hidden="true" /> {saving ? "Receiving…" : "Receive units"}
            </button>
          </div>
          {message && <p className="state empty" role="status">{message}</p>}
        </form>
      )}
    </section>
  );
}
