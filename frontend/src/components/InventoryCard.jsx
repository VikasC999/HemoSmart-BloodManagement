import { useState } from "react";
import { api } from "../api";

// `editable` gates the PATCH-based edit controls -- Hospital Staff has
// read-only inventory access per RBAC (the backend would reject a PATCH
// from them with a 403 anyway; hiding the controls here just means they
// never see a control that would fail).
export default function InventoryCard({ editable = false }) {
  const [inventory, setInventory] = useState(null);
  const [error, setError] = useState(null);
  const [editingType, setEditingType] = useState(null);
  const [editValue, setEditValue] = useState("");
  const [saving, setSaving] = useState(false);

  const load = async () => {
    setError(null);
    try {
      const result = await api.inventory();
      setInventory(result);
    } catch (e) {
      setError(e.message);
    }
  };

  const startEdit = (bloodType, currentUnits) => {
    setEditingType(bloodType);
    setEditValue(String(currentUnits));
  };

  const saveEdit = async (bloodType) => {
    const units = parseInt(editValue, 10);
    if (Number.isNaN(units) || units < 0) return;
    setSaving(true);
    setError(null);
    try {
      await api.updateInventory(bloodType, units);
      setEditingType(null);
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
      <button onClick={load}>Refresh</button>
      {error && <p className="error">{error}</p>}
      {inventory && (
        <table>
          <thead>
            <tr>
              <th>Type</th><th>Units</th><th>Status</th>{editable && <th></th>}
            </tr>
          </thead>
          <tbody>
            {Object.entries(inventory.inventory).map(([bt, units]) => (
              <tr key={bt}>
                <td>{bt}</td>
                <td>
                  {editable && editingType === bt ? (
                    <input
                      type="number" min={0} value={editValue}
                      onChange={(e) => setEditValue(e.target.value)}
                      className="inline-edit-input"
                    />
                  ) : units}
                </td>
                <td>{inventory.low_stock_types[bt] !== undefined ? "LOW" : "OK"}</td>
                {editable && (
                  <td>
                    {editingType === bt ? (
                      <button onClick={() => saveEdit(bt)} disabled={saving}>Save</button>
                    ) : (
                      <button onClick={() => startEdit(bt, units)}>Edit</button>
                    )}
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
