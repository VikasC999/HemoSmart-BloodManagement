import { useState } from "react";
import {
  Area, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "../api";
import { COMPONENTS } from "../constants";

const COLORS = { PRBC: "#b91c1c", Platelets: "#d97706", FFP: "#2563eb" };

// Demand forecast per component, and the demand-supply gap it implies
// against live stock: where demand will go unmet, and which units will
// expire before anyone needs them.
export default function ForecastChart() {
  const [days, setDays] = useState(7);
  const [outlook, setOutlook] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      setOutlook(await api.outlook(days));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const chartData = outlook
    ? outlook.component_forecast[COMPONENTS[0]].map((d, i) => ({
        date: d.date.slice(5),
        ...Object.fromEntries(COMPONENTS.flatMap((c) => {
          const f = outlook.component_forecast[c][i];
          // Prophet's 80% interval; evaluation/forecast_backtest.py measures how often
          // actual demand really falls inside it (coverage).
          return [[c, f.units], [`${c} band`, [f.lower, f.upper]]];
        })),
      }))
    : [];

  const tooltipValue = (v, name) =>
    Array.isArray(v) ? [`${v[0]}–${v[1]} units`, `${name.replace(" band", "")} 80% range`] : [`${v} units`, name];

  return (
    <section className="card">
      <h2>Demand Forecast &amp; Supply Gap</h2>
      <div className="actions">
        <input type="number" value={days} min={1} max={30} className="inline-edit-input"
               onChange={(e) => setDays(parseInt(e.target.value, 10) || 7)} />
        <span className="muted">days</span>
        <button onClick={run} disabled={loading}>Forecast</button>
      </div>
      {error && <p className="error">{error}</p>}
      {outlook && (
        <>
          <p className="muted">{outlook.model_used}</p>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={220}>
              <ComposedChart data={chartData} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={tooltipValue} />
                <Legend verticalAlign="top" height={24} wrapperStyle={{ fontSize: 12 }} />
                {COMPONENTS.map((c) => (
                  <Area key={`${c}-band`} type="monotone" dataKey={`${c} band`} stroke="none"
                        fill={COLORS[c]} fillOpacity={0.12} legendType="none" isAnimationActive={false} />
                ))}
                {COMPONENTS.map((c) => (
                  <Line key={c} type="monotone" dataKey={c} stroke={COLORS[c]} strokeWidth={2} dot={false} />
                ))}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <p className="hint">
            Shaded: Prophet&apos;s 80% range. In backtests actual demand fell inside it 77–79% of days,
            but only 68% for platelets during the dengue spike, when it runs too narrow.
          </p>

          <h4>Projected shortages</h4>
          {outlook.shortages.length === 0 ? <p className="stock-ok">None within {outlook.horizon_days} days.</p> : (
            <div className="table-scroll"><table>
              <thead><tr><th>Group</th><th>Component</th><th>In stock</th><th>Demand</th><th>Short</th><th>From</th></tr></thead>
              <tbody>
                {outlook.shortages.map((s) => (
                  <tr key={s.blood_group + s.component} className="row-critical">
                    <td>{s.blood_group}</td><td>{s.component}</td><td>{s.supply_now}</td>
                    <td>{s.forecast_demand}</td><td><strong>{s.projected_shortage}</strong></td>
                    <td>{s.first_shortage_date}</td>
                  </tr>
                ))}
              </tbody>
            </table></div>
          )}

          <h4>Projected wastage</h4>
          {outlook.wastage_risks.length === 0 ? <p className="stock-ok">No units projected to expire unused.</p> : (
            <div className="table-scroll"><table>
              <thead><tr><th>Group</th><th>Component</th><th>In stock</th><th>Demand</th><th>Will expire unused</th></tr></thead>
              <tbody>
                {outlook.wastage_risks.map((w) => (
                  <tr key={w.blood_group + w.component} className="row-low">
                    <td>{w.blood_group}</td><td>{w.component}</td><td>{w.supply_now}</td>
                    <td>{w.forecast_demand}</td><td><strong>{w.projected_wastage}</strong></td>
                  </tr>
                ))}
              </tbody>
            </table></div>
          )}
        </>
      )}
    </section>
  );
}
