import { useState } from "react";
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "../api";

export default function ForecastChart() {
  const [days, setDays] = useState(7);
  const [forecast, setForecast] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.forecast(days);
      if (result.error) setError(result.error);
      else setForecast(result);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="card">
      <h2>Blood Demand Forecast</h2>
      <div className="actions">
        <input type="number" value={days} min={1} max={30}
               onChange={(e) => setDays(parseInt(e.target.value, 10))} />
        <button onClick={run} disabled={loading}>Forecast</button>
      </div>
      {error && <p className="error">{error}</p>}
      {forecast && (
        <div className="result">
          <p>
            Model: <strong>{forecast.model_used}</strong> — avg {forecast.average_daily_demand} units/day
            {forecast.cached && <span className="pill-cached"> cached</span>}
          </p>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={forecast.predicted_units_per_day} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => [`${v} units`, "Predicted demand"]} />
                <Line type="monotone" dataKey="predicted_units" stroke="#b91c1c" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </section>
  );
}
