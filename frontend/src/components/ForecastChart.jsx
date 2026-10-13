import { useEffect, useState } from "react";
import { RotateCw } from "lucide-react";
import {
  Area, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "../api";
import { COMPONENTS } from "../constants";
import { EmptyState, ErrorState, SeverityBadge, Skeleton, UpdatedAt } from "./ui";

const HORIZONS = [7, 14, 30];

// Series identity from the token sheet (works in light and dark).
const COLORS = { PRBC: "var(--chart-prbc)", Platelets: "var(--chart-platelets)", FFP: "var(--chart-ffp)" };
const TOOLTIP_STYLE = {
  background: "var(--color-raised)",
  border: "1px solid var(--color-border)",
  borderRadius: 8,
  color: "var(--color-text)",
  fontSize: 13,
};

// Demand forecast per component, and the demand-supply gap it implies
// against live stock: where demand will go unmet, and which units will
// expire before anyone needs them.
export default function ForecastChart() {
  const [days, setDays] = useState(7);
  const [outlook, setOutlook] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  const run = async (horizon = days) => {
    setLoading(true);
    setError(null);
    try {
      setOutlook(await api.outlook(horizon));
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  };

  // The gap is the main reason to open this page: load it on arrival.
  useEffect(() => { run(7); }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  const changeHorizon = (h) => {
    setDays(h);
    run(h);
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

  const totals = outlook && COMPONENTS.map((c) => ({
    component: c,
    units: Math.round(outlook.component_forecast[c].reduce((sum, d) => sum + d.units, 0)),
  }));
  const summary = totals && `Forecast demand over the next ${outlook.horizon_days} days: `
    + totals.map((t) => `${t.component} ${t.units} units`).join(", ")
    + `. ${outlook.shortages.length} projected shortages, ${outlook.wastage_risks.length} groups with units at risk of expiring.`;

  return (
    <section className="card" aria-labelledby="forecast-title">
      <div className="card-head">
        <div>
          <h2 id="forecast-title">Demand forecast &amp; supply gap</h2>
          {outlook && (
            <p className="card-sub">
              Units per day, all blood groups · next {outlook.horizon_days} days · <UpdatedAt iso={outlook.as_of} prefix="Forecast" />
            </p>
          )}
        </div>
        <div className="segmented" role="radiogroup" aria-label="Forecast horizon">
          {HORIZONS.map((h) => (
            <button key={h} role="radio" aria-checked={days === h} disabled={loading}
                    className={`segment ${days === h ? "active" : ""}`} onClick={() => changeHorizon(h)}>
              {h} days
            </button>
          ))}
        </div>
      </div>

      {error && (
        <ErrorState what="Couldn't load the forecast." error={error} onRetry={() => run(days)}
                    stillValid="The inventory figures are unaffected." />
      )}
      {loading && !outlook && (
        <div role="status" aria-label="Loading forecast">
          <Skeleton height={240} style={{ borderRadius: "var(--radius-md)" }} />
        </div>
      )}
      {outlook && (
        <>
          <figure className="chart-figure" aria-label={summary}>
            <div className={`chart-wrap ${loading ? "is-refreshing" : ""}`}>
              <ResponsiveContainer width="100%" height={240}>
                <ComposedChart data={chartData} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
                  <CartesianGrid stroke="var(--color-gridline)" vertical={false} />
                  <XAxis dataKey="date" tick={{ fontSize: 12, fill: "var(--color-text-secondary)" }}
                         stroke="var(--color-border)" minTickGap={16} />
                  <YAxis tick={{ fontSize: 12, fill: "var(--color-text-secondary)" }} stroke="var(--color-border)"
                         width={44} />
                  <Tooltip formatter={tooltipValue} contentStyle={TOOLTIP_STYLE} labelStyle={{ color: "var(--color-text)" }} />
                  <Legend verticalAlign="top" height={28} iconType="plainline"
                          wrapperStyle={{ fontSize: 12, color: "var(--color-text-secondary)" }} />
                  {COMPONENTS.map((c) => (
                    <Area key={`${c}-band`} type="monotone" dataKey={`${c} band`} stroke="none"
                          fill={COLORS[c]} fillOpacity={0.14} legendType="none" isAnimationActive={false} />
                  ))}
                  {COMPONENTS.map((c) => (
                    <Line key={c} type="monotone" dataKey={c} stroke={COLORS[c]} strokeWidth={2} dot={false}
                          strokeDasharray="6 4" isAnimationActive={false} />
                  ))}
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            <figcaption className="hint">
              Dashed lines are model forecasts ({outlook.model_used}); the shaded band is its 80% range. In backtests,
              actual demand fell inside it on 77–79% of days, but on only 68% for platelets during the dengue
              spike, when the range runs too narrow.
            </figcaption>
          </figure>

          <h3 className="section-title">Projected shortages</h3>
          {outlook.shortages.length === 0 ? (
            <EmptyState title={`No shortages projected in the next ${outlook.horizon_days} days`}>
              Forecast demand is covered by current stock plus routine collection.
            </EmptyState>
          ) : (
            <div className="table-scroll"><table>
              <thead><tr>
                <th>Need</th><th>Risk</th><th className="num">Short</th><th>From</th>
                <th className="num">In stock</th><th className="num">Demand</th>
              </tr></thead>
              <tbody>
                {outlook.shortages.map((s) => (
                  <tr key={s.blood_group + s.component}>
                    <td className="nowrap"><strong>{s.blood_group}</strong> {s.component}</td>
                    <td><SeverityBadge level="high" /></td>
                    <td className="num"><strong>{s.projected_shortage}</strong></td>
                    <td className="nowrap tabular">{s.first_shortage_date}</td>
                    <td className="num">{s.supply_now}</td><td className="num">{s.forecast_demand}</td>
                  </tr>
                ))}
              </tbody>
            </table></div>
          )}

          <h3 className="section-title">Units at risk of expiring</h3>
          {outlook.wastage_risks.length === 0 ? (
            <EmptyState title="No units projected to expire unused">
              Every unit in stock is expected to be issued before it expires.
            </EmptyState>
          ) : (
            <div className="table-scroll"><table>
              <thead><tr>
                <th>Units</th><th>Risk</th><th className="num">May expire</th>
                <th className="num">In stock</th><th className="num">Demand</th>
              </tr></thead>
              <tbody>
                {outlook.wastage_risks.map((w) => (
                  <tr key={w.blood_group + w.component}>
                    <td className="nowrap"><strong>{w.blood_group}</strong> {w.component}</td>
                    <td><SeverityBadge level="moderate" label="Wastage" /></td>
                    <td className="num"><strong>{w.projected_wastage}</strong></td>
                    <td className="num">{w.supply_now}</td><td className="num">{w.forecast_demand}</td>
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
