import { useEffect, useState } from "react";
import { api } from "../api";

// Glanceable summary replacing the old pattern of manually clicking
// Inventory -> Refresh and Donor Alert -> Send just to see what's
// already happened. Loads once on mount.
export default function AlertsPanel() {
  const [lowStock, setLowStock] = useState(null);
  const [recentAlerts, setRecentAlerts] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    (async () => {
      try {
        const [inv, alerts] = await Promise.all([
          api.inventory(),
          api.recentDonorAlerts(5),
        ]);
        setLowStock(inv.low_stock_types);
        setRecentAlerts(alerts);
      } catch (e) {
        setError(e.message);
      }
    })();
  }, []);

  return (
    <section className="card alerts-panel">
      <h2>Alerts</h2>
      {error && <p className="error">{error}</p>}
      <div className="alerts-columns">
        <div>
          <h4>Low Stock</h4>
          {lowStock && (
            Object.keys(lowStock).length === 0
              ? <p className="stock-ok">No shortages right now.</p>
              : (
                <ul className="alert-list">
                  {Object.entries(lowStock).map(([bt, units]) => (
                    <li key={bt} className="alert-list-item low">{bt}: {units} units</li>
                  ))}
                </ul>
              )
          )}
        </div>
        <div>
          <h4>Recent Donor Alerts</h4>
          {recentAlerts && (
            recentAlerts.length === 0
              ? <p>No alerts sent yet.</p>
              : (
                <ul className="alert-list">
                  {recentAlerts.map((a, i) => (
                    <li key={i} className="alert-list-item">{a.blood_type} — donor {a.donor_id}</li>
                  ))}
                </ul>
              )
          )}
        </div>
      </div>
    </section>
  );
}
