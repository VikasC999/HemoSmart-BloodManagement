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
        setLowStock(inv.low_stock);
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
          <h4>Below Par</h4>
          {lowStock && (
            lowStock.length === 0
              ? <p className="stock-ok">Every group and component is at par.</p>
              : (
                <ul className="alert-list">
                  {lowStock.map((i) => (
                    <li key={i.blood_group + i.component}
                        className={`alert-list-item ${i.status === "Critical" ? "low" : "warn"}`}>
                      {i.blood_group} {i.component}: {i.available}/{i.par} ({i.status})
                    </li>
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
                    <li key={i} className="alert-list-item">
                      {a.blood_type} {a.component || ""} — {a.donor_id}{" "}
                      <span className="muted">
                        {a.responded == null ? "pending" : a.responded ? "donated" : "no reply"}
                        {a.cycle_id ? " · auto" : ""}
                      </span>
                    </li>
                  ))}
                </ul>
              )
          )}
        </div>
      </div>
    </section>
  );
}
