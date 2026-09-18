import { useState } from "react";
import { api } from "../api";
import { BLOOD_TYPES, SURGERY_TYPES } from "../constants";

// `canAlertDonors` reflects RBAC, not just UI preference: Hospital
// Staff can predict but cannot call /api/donors/alert (403), so their
// version of this card surfaces the low-stock finding without a live
// button -- Blood Bank Manager's does, since they hold both
// permissions. This is the "connected predict -> inventory ->
// donor-alert" flow, scoped honestly to who's actually allowed to act.
export default function PredictionCard({ canAlertDonors = false }) {
  const [form, setForm] = useState({
    hemoglobin: 7.2,
    platelets: 65000,
    INR: 1.9,
    age: 58,
    surgery_type: "Emergency",
  });
  const [bloodType, setBloodType] = useState("O-");
  const [result, setResult] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [stockCheck, setStockCheck] = useState(null);
  const [alertMessage, setAlertMessage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const update = (field, value) => setForm((f) => ({ ...f, [field]: value }));

  const runPredict = async () => {
    setLoading(true);
    setError(null);
    setExplanation(null);
    setStockCheck(null);
    setAlertMessage(null);
    try {
      const prediction = await api.predict(form);
      setResult(prediction);

      // Connected flow: a positive prediction triggers an inventory
      // check for the selected blood type, right here, instead of
      // requiring a trip to a separate Inventory panel.
      if (prediction.transfusion_needed) {
        const stock = await api.inventory(bloodType);
        setStockCheck(stock);
      }
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

  const runAlertDonors = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.alertDonors(bloodType);
      setAlertMessage(res.message);
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
        <label>
          Blood Type
          <select value={bloodType} onChange={(e) => setBloodType(e.target.value)}>
            {BLOOD_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
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

      {stockCheck && (
        stockCheck.low_stock ? (
          <div className="stock-alert">
            <p>
              <strong>{bloodType}</strong> stock is low ({stockCheck.units} units) —
              {canAlertDonors ? " ready to notify eligible donors." : " a Blood Bank Manager needs to notify donors."}
            </p>
            {canAlertDonors ? (
              <button onClick={runAlertDonors} disabled={loading}>Alert Donors for {bloodType}</button>
            ) : (
              <button disabled title="Requires Blood Bank Manager or Donor Coordinator access">
                Alert Donors for {bloodType}
              </button>
            )}
            {alertMessage && <p className="result">{alertMessage}</p>}
          </div>
        ) : (
          <p className="stock-ok">{bloodType} stock is sufficient ({stockCheck.units} units) — no donor alert needed.</p>
        )
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
