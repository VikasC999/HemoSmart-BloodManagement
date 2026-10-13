import { useState } from "react";
import { api } from "../api";
import { BLOOD_TYPES, SURGERY_TYPES } from "../constants";

const FEATURE_LABELS = {
  hemoglobin: "Hemoglobin", platelets: "Platelets", INR: "INR", age: "Age", surgery_type: "Surgery type",
};

// SHAP for one component: each feature's push toward (+) or away from (-)
// "needs this component", in log-odds. Exact TreeSHAP from the model.
function ShapBars({ component, contributions }) {
  const entries = Object.entries(contributions).filter(([k]) => k !== "_base");
  const max = Math.max(...entries.map(([, v]) => Math.abs(v)), 1e-6);
  return (
    <div className="shap">
      <div className="shap-title">Why {component}? (SHAP)</div>
      {entries
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
        .map(([feature, value]) => (
          <div key={feature} className="shap-row">
            <span className="shap-label">{FEATURE_LABELS[feature] || feature}</span>
            <span className="shap-track">
              <span
                className={`shap-bar ${value >= 0 ? "pos" : "neg"}`}
                style={{ width: `${(Math.abs(value) / max) * 100}%` }}
              />
            </span>
            <span className="shap-value">{value >= 0 ? "+" : ""}{value.toFixed(2)}</span>
          </div>
        ))}
    </div>
  );
}

function Citations({ citations }) {
  if (!citations?.length) return null;
  return (
    <details className="citations">
      <summary>Sources ({citations.length})</summary>
      <ol>
        {citations.map((c) => (
          <li key={c.ref} value={c.ref}>
            <strong>{c.source}, p.{c.page}</strong> — {c.excerpt}
          </li>
        ))}
      </ol>
    </details>
  );
}

// `canAlertDonors` reflects RBAC: Hospital Staff can predict and reserve
// blood but cannot call /api/donors/alert (403), so their card surfaces a
// shortfall without a live alert button; Blood Bank Manager's card has it.
export default function PredictionCard({ canAlertDonors = false }) {
  const [form, setForm] = useState({
    hemoglobin: 7.2,
    platelets: 65000,
    INR: 1.9,
    age: 58,
    surgery_type: "Emergency",
    blood_group: "O+",
  });
  const [result, setResult] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [transfusion, setTransfusion] = useState(null);
  const [alertMessage, setAlertMessage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const update = (field, value) => setForm((f) => ({ ...f, [field]: value }));

  const run = async (fn) => {
    setLoading(true);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const runPredict = () => run(async () => {
    setExplanation(null);
    setTransfusion(null);
    setAlertMessage(null);
    setResult(await api.predict(form));
  });

  const runExplain = () => run(async () => {
    setExplanation(await api.explain(form, result.transfusion_needed, result.prediction_id));
  });

  const reserve = () => run(async () => {
    setTransfusion(await api.requestTransfusion(result.patient_id, result.prediction_id, form.blood_group));
  });

  const finish = (confirm) => run(async () => {
    setTransfusion(await (confirm ? api.confirmTransfusion : api.cancelTransfusion)(transfusion.id));
  });

  const alertFor = (component) => run(async () => {
    const res = await api.alertDonors(form.blood_group, component);
    setAlertMessage(res.message);
  });

  const needed = result?.components.filter((c) => c.needed) ?? [];
  const shortfall = transfusion ? Object.entries(transfusion.shortfall) : [];

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
          <select value={form.surgery_type} onChange={(e) => update("surgery_type", e.target.value)}>
            {SURGERY_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </label>
        <label>
          Patient Blood Group
          <select value={form.blood_group} onChange={(e) => update("blood_group", e.target.value)}>
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
          <div className="result-head">
            <strong>{result.transfusion_needed ? "Transfusion needed" : "No transfusion needed"}</strong>
            <span className={`badge urgency-${result.urgency.toLowerCase()}`}>{result.urgency}</span>
          </div>
          <div className="muted">P(transfusion) {(result.confidence * 100).toFixed(1)}%</div>
          {needed.length > 0 && (
            <ul className="component-list">
              {needed.map((c) => (
                <li key={c.component}>
                  <strong>{c.component}</strong> × {c.units} unit{c.units > 1 ? "s" : ""}
                  <span className="muted"> ({(c.probability * 100).toFixed(0)}%)</span>
                </li>
              ))}
            </ul>
          )}
          {needed.map((c) => (
            <ShapBars key={c.component} component={c.component} contributions={result.shap[c.component]} />
          ))}
          {needed.length > 0 && !transfusion && (
            <div className="actions">
              <button onClick={reserve} disabled={loading}>Reserve blood (FEFO)</button>
            </div>
          )}
        </div>
      )}

      {transfusion && (
        <div className={shortfall.length ? "stock-alert" : "stock-ok"}>
          <p>
            Transfusion #{transfusion.id} — <strong>{transfusion.status}</strong>
            {" "}for a {transfusion.recipient_group} patient
          </p>
          <div className="table-scroll">
            <table>
              <thead><tr><th>Unit</th><th>Group</th><th>Component</th><th>Expires</th><th></th></tr></thead>
              <tbody>
                {transfusion.units.map((u) => (
                  <tr key={u.unit_id}>
                    <td>{u.unit_id}</td>
                    <td>{u.blood_group}</td>
                    <td>{u.component}</td>
                    <td>{u.expires_at.slice(0, 16).replace("T", " ")}</td>
                    <td>
                      {u.near_expiry && <span className="badge near">near expiry</span>}
                      {u.cross_group && <span className="badge info">compatible</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {shortfall.length > 0 && (
            <div className="shortfall">
              {shortfall.map(([component, n]) => (
                <p key={component}>
                  Short <strong>{n} {component}</strong> for {transfusion.recipient_group} —{" "}
                  {canAlertDonors ? (
                    <button onClick={() => alertFor(component)} disabled={loading}>
                      Alert {form.blood_group} donors
                    </button>
                  ) : "a Blood Bank Manager needs to source more."}
                </p>
              ))}
              {alertMessage && <p className="result">{alertMessage}</p>}
            </div>
          )}
          {transfusion.status === "reserved" && (
            <div className="actions">
              <button onClick={() => finish(true)} disabled={loading}>Confirm transfused</button>
              <button className="secondary" onClick={() => finish(false)} disabled={loading}>Cancel & release</button>
            </div>
          )}
        </div>
      )}

      {explanation && (
        <div className="explanation">
          <h4>Threshold facts</h4>
          <pre>{explanation.threshold_summary}</pre>
          <h4>Clinical explanation</h4>
          <p>{explanation.llm_explanation}</p>
          <Citations citations={explanation.citations} />
        </div>
      )}
    </section>
  );
}
