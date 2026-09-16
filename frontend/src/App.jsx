import { useState } from "react";
import { api } from "./api";

const SURGERY_TYPES = ["Cardiac", "Orthopedic", "General", "Emergency"];
const BLOOD_TYPES = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"];

function PredictionCard() {
  const [form, setForm] = useState({
    hemoglobin: 7.2,
    platelets: 65000,
    INR: 1.9,
    age: 58,
    surgery_type: "Emergency",
  });
  const [result, setResult] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const update = (field, value) => setForm((f) => ({ ...f, [field]: value }));

  const runPredict = async () => {
    setLoading(true);
    setError(null);
    setExplanation(null);
    try {
      const prediction = await api.predict(form);
      setResult(prediction);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const runExplain = async () => {
    if (!result) return;
    setLoading(true);
    setError(null);
    try {
      const exp = await api.explain(form, result.transfusion_needed);
      setExplanation(exp);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="card">
      <h2>Transfusion Prediction</h2>
      <div className="form-grid">
        <label>
          Hemoglobin (g/dL)
          <input type="number" step="0.1" value={form.hemoglobin}
                 onChange={(e) => update("hemoglobin", parseFloat(e.target.value))} />
        </label>
        <label>
          Platelets (/µL)
          <input type="number" value={form.platelets}
                 onChange={(e) => update("platelets", parseInt(e.target.value, 10))} />
        </label>
        <label>
          INR
          <input type="number" step="0.1" value={form.INR}
                 onChange={(e) => update("INR", parseFloat(e.target.value))} />
        </label>
        <label>
          Age
          <input type="number" value={form.age}
                 onChange={(e) => update("age", parseInt(e.target.value, 10))} />
        </label>
        <label>
          Surgery Type
          <select value={form.surgery_type}
                  onChange={(e) => update("surgery_type", e.target.value)}>
            {SURGERY_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </label>
      </div>
      <div className="actions">
        <button onClick={runPredict} disabled={loading}>Predict</button>
        <button onClick={runExplain} disabled={loading || !result}>Explain</button>
      </div>
      {error && <p className="error">{error}</p>}
      {result && (
        <div className="result">
          <strong>{result.transfusion_needed ? "Transfusion needed" : "No transfusion needed"}</strong>
          {" "}({(result.confidence * 100).toFixed(1)}% confidence)
        </div>
      )}
      {explanation && (
        <div className="explanation">
          <h4>Threshold facts</h4>
          <pre>{explanation.threshold_summary}</pre>
          <h4>Clinical explanation</h4>
          <p>{explanation.llm_explanation}</p>
        </div>
      )}
    </section>
  );
}

function ForecastCard() {
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
          <p>Model: <strong>{forecast.model_used}</strong> — avg {forecast.average_daily_demand} units/day</p>
          <ul>
            {forecast.predicted_units_per_day.map((d) => (
              <li key={d.date}>{d.date}: {d.predicted_units} units</li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

function ChatCard() {
  const [sessionId, setSessionId] = useState(null);
  const [message, setMessage] = useState("");
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);

  const send = async () => {
    if (!message.trim()) return;
    const userMessage = message;
    setHistory((h) => [...h, { role: "user", text: userMessage }]);
    setMessage("");
    setLoading(true);
    try {
      const res = await api.chat(sessionId, userMessage);
      setSessionId(res.session_id);
      setHistory((h) => [...h, { role: "assistant", text: res.reply }]);
    } catch (e) {
      setHistory((h) => [...h, { role: "assistant", text: `Error: ${e.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="card">
      <h2>Chat Assistant</h2>
      <div className="chat-log">
        {history.map((m, i) => (
          <div key={i} className={`chat-msg ${m.role}`}><strong>{m.role}:</strong> {m.text}</div>
        ))}
      </div>
      <div className="actions">
        <input
          type="text"
          placeholder="e.g. hemoglobin 7.2, platelets 65000, INR 1.9, age 58, Emergency surgery"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
        />
        <button onClick={send} disabled={loading}>Send</button>
      </div>
    </section>
  );
}

function InventoryCard() {
  const [inventory, setInventory] = useState(null);
  const [error, setError] = useState(null);

  const load = async () => {
    setError(null);
    try {
      const result = await api.inventory();
      setInventory(result);
    } catch (e) {
      setError(e.message);
    }
  };

  return (
    <section className="card">
      <h2>Blood Inventory</h2>
      <button onClick={load}>Refresh</button>
      {error && <p className="error">{error}</p>}
      {inventory && (
        <table>
          <thead><tr><th>Type</th><th>Units</th><th>Status</th></tr></thead>
          <tbody>
            {Object.entries(inventory.inventory).map(([bt, units]) => (
              <tr key={bt}>
                <td>{bt}</td>
                <td>{units}</td>
                <td>{inventory.low_stock_types[bt] !== undefined ? "LOW" : "OK"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

function DonorAlertCard() {
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

export default function App() {
  return (
    <div className="app">
      <header>
        <h1>HemoSmart</h1>
        <p>AI-assisted blood transfusion prediction &amp; supply coordination</p>
      </header>
      <main className="grid">
        <PredictionCard />
        <ForecastCard />
        <ChatCard />
        <InventoryCard />
        <DonorAlertCard />
      </main>
    </div>
  );
}
