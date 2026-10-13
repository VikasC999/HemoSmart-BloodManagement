import { useState } from "react";
import { BookOpen, ClipboardList, PackageCheck, Sparkles, Stethoscope } from "lucide-react";
import { api } from "../api";
import { BLOOD_TYPES, SURGERY_TYPES } from "../constants";
import { formatPercent } from "./ui/format";
import {
  AIPanel, EmptyState, ErrorState, SeverityBadge, SkeletonLines, urgencySeverity,
} from "./ui";

const FEATURE_LABELS = {
  hemoglobin: "Hemoglobin", platelets: "Platelets", INR: "INR", age: "Age", surgery_type: "Surgery type",
};

const NUMERIC_FIELDS = [
  { key: "hemoglobin", label: "Hemoglobin", unit: "g/dL", step: "0.1", parse: parseFloat },
  { key: "platelets", label: "Platelets", unit: "/µL", step: "1000", parse: (v) => parseInt(v, 10) },
  { key: "INR", label: "INR", unit: "ratio", step: "0.1", parse: parseFloat },
  { key: "age", label: "Age", unit: "years", step: "1", parse: (v) => parseInt(v, 10) },
];

// SHAP for one component: each value's push toward (+) or away from (−)
// "needs this component", in log-odds. Exact TreeSHAP from the model.
function ShapBars({ component, contributions }) {
  const entries = Object.entries(contributions).filter(([k]) => k !== "_base");
  const max = Math.max(...entries.map(([, v]) => Math.abs(v)), 1e-6);
  return (
    <div className="shap">
      <div className="shap-title">{component}</div>
      {entries
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
        .map(([feature, value]) => (
          <div key={feature} className="shap-row">
            <span className="shap-label">{FEATURE_LABELS[feature] || feature}</span>
            <span className="shap-track" aria-hidden="true">
              <span className={`shap-bar ${value >= 0 ? "pos" : "neg"}`}
                    style={{ width: `${(Math.abs(value) / max) * 100}%` }} />
            </span>
            <span className="shap-value">
              {value >= 0 ? "+" : "−"}{Math.abs(value).toFixed(2)}
              <span className="visually-hidden">{value >= 0 ? " raises" : " lowers"}</span>
            </span>
          </div>
        ))}
    </div>
  );
}

function Citations({ citations }) {
  if (!citations?.length) return null;
  return (
    <details className="citations">
      <summary><BookOpen size={14} aria-hidden="true" /> WHO handbook sources ({citations.length})</summary>
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

const STEP_LABEL = {
  predict: "The prediction didn't run.",
  explain: "The AI explanation couldn't be written.",
  reserve: "Blood couldn't be reserved.",
  finish: "The transfusion couldn't be updated.",
  alert: "Donors couldn't be alerted.",
};

// `canAlertDonors` reflects RBAC: Hospital Staff can predict and reserve
// blood but cannot call /api/donors/alert (403), so their view surfaces a
// shortfall without a live alert button; Blood Bank Manager's has it.
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
  const [receivedAt, setReceivedAt] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [transfusion, setTransfusion] = useState(null);
  const [alertMessage, setAlertMessage] = useState(null);
  const [busy, setBusy] = useState(null); // which step is running
  const [error, setError] = useState(null); // { step, error }

  const update = (field, value) => setForm((f) => ({ ...f, [field]: value }));
  const invalid = NUMERIC_FIELDS.filter((f) => Number.isNaN(form[f.key]) || form[f.key] == null);

  const run = async (step, fn) => {
    setBusy(step);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError({ step, error: e });
    } finally {
      setBusy(null);
    }
  };

  const runPredict = () => run("predict", async () => {
    setExplanation(null);
    setTransfusion(null);
    setAlertMessage(null);
    setResult(await api.predict(form));
    setReceivedAt(new Date());
  });

  const runExplain = () => run("explain", async () => {
    setExplanation(await api.explain(form, result.transfusion_needed, result.prediction_id));
  });

  const reserve = () => run("reserve", async () => {
    setTransfusion(await api.requestTransfusion(result.patient_id, result.prediction_id, form.blood_group));
  });

  const finish = (confirm) => run("finish", async () => {
    setTransfusion(await (confirm ? api.confirmTransfusion : api.cancelTransfusion)(transfusion.id));
  });

  const alertFor = (component) => run("alert", async () => {
    const res = await api.alertDonors(form.blood_group, component);
    setAlertMessage(res.message);
  });

  const needed = result?.components.filter((c) => c.needed) ?? [];
  const shortfall = transfusion ? Object.entries(transfusion.shortfall) : [];
  const loading = busy !== null;
  const level = result ? urgencySeverity(result.urgency) : null;

  return (
    <div className="predict-layout">
      {/* 1. Data: what was measured */}
      <section className="card predict-input" aria-labelledby="patient-data-title">
        <div className="card-head">
          <h2 id="patient-data-title">Patient data</h2>
        </div>
        <p className="hint">Enter values as measured. Nothing is saved until you run the prediction.</p>
        <div className="form-grid">
          {NUMERIC_FIELDS.map((f) => {
            const bad = invalid.includes(f);
            return (
              <label key={f.key}>
                {f.label}
                <span className="input-unit">
                  <input type="number" step={f.step} value={Number.isNaN(form[f.key]) ? "" : form[f.key]}
                         aria-invalid={bad ? "true" : undefined}
                         aria-describedby={bad ? `${f.key}-err` : undefined}
                         onChange={(e) => update(f.key, f.parse(e.target.value))} />
                  <span className="unit" aria-hidden="true">{f.unit}</span>
                </span>
                {bad && <span id={`${f.key}-err`} className="field-error">Enter a value.</span>}
              </label>
            );
          })}
          <label>
            Surgery type
            <select value={form.surgery_type} onChange={(e) => update("surgery_type", e.target.value)}>
              {SURGERY_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </label>
          <label>
            Blood group
            <select value={form.blood_group} onChange={(e) => update("blood_group", e.target.value)}>
              {BLOOD_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </label>
        </div>
        <div className="actions">
          <button onClick={runPredict} disabled={loading || invalid.length > 0}>
            <Stethoscope size={16} aria-hidden="true" />
            {busy === "predict" ? "Predicting…" : result ? "Predict again" : "Predict"}
          </button>
        </div>
        {error?.step === "predict" && <ErrorState what={STEP_LABEL.predict} error={error.error} onRetry={runPredict} />}
      </section>

      <div className="predict-results">
        {!result && error?.step !== "predict" && (
          <EmptyState icon={ClipboardList} title="No prediction yet">
            Enter the patient's lab values and choose Predict. The result shows which blood components they're likely to need, how many units, and why.
          </EmptyState>
        )}

        {result && (
          <>
            {/* 2. Model prediction */}
            <section className="card" aria-labelledby="prediction-title">
              <div className="card-head">
                <h2 id="prediction-title">Model prediction</h2>
                <span className="muted">
                  Patient #{result.patient_id ?? "—"}
                  {receivedAt && ` · ${receivedAt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`}
                </span>
              </div>
              <div className="prediction-headline">
                <p className="prediction-verdict">
                  {result.transfusion_needed ? "Transfusion likely needed" : "Transfusion not indicated"}
                </p>
                <SeverityBadge level={level} label={`${result.urgency} urgency`} />
              </div>
              <p className="prediction-prob">
                Model probability <strong className="tabular">{formatPercent(result.confidence)}</strong>
                <span className="muted"> · XGBoost, one model per blood component</span>
              </p>

              {needed.length > 0 && (
                <div className="table-scroll">
                  <table className="components-table">
                    <thead>
                      <tr><th>Component</th><th className="num">Units</th><th className="num">Probability</th></tr>
                    </thead>
                    <tbody>
                      {needed.map((c) => (
                        <tr key={c.component}>
                          <td>{c.component}</td>
                          <td className="num">{c.units}</td>
                          <td className="num">{formatPercent(c.probability)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              <p className="decision-support">Decision support. Review before acting.</p>

              {/* 3. Key factors */}
              {needed.length > 0 && (
                <details className="factors" open>
                  <summary>What drove this prediction</summary>
                  <p className="hint">
                    How much each value pushed the model toward (+) or away from (−) each component.
                  </p>
                  <div className="shap-grid">
                    {needed.map((c) => (
                      <ShapBars key={c.component} component={c.component} contributions={result.shap[c.component]} />
                    ))}
                  </div>
                </details>
              )}
            </section>

            {/* 4. AI explanation (+ the rule-based checks it is grounded on) */}
            {!explanation && busy !== "explain" && (
              <section className="ai-panel explain-cta" aria-labelledby="explain-title">
                <div className="explain-copy">
                  <h2 id="explain-title" className="ai-label">
                    <Sparkles size={14} aria-hidden="true" /> HemoSmart AI explanation
                  </h2>
                  <p className="muted">
                    A plain-language reading of this result, checked against protocol thresholds and cited to the
                    WHO handbook. Takes a few seconds.
                  </p>
                </div>
                <button className="secondary" onClick={runExplain} disabled={loading}>
                  <Sparkles size={16} aria-hidden="true" /> Explain this prediction
                </button>
              </section>
            )}
            {busy === "explain" && (
              <AIPanel label="HemoSmart AI explanation" decisionSupport={false}>
                <SkeletonLines lines={4} label="Writing the explanation" />
              </AIPanel>
            )}
            {error?.step === "explain" && (
              <ErrorState what={STEP_LABEL.explain} error={error.error} onRetry={runExplain}
                          stillValid="The model prediction above is unaffected." />
            )}
            {explanation && (
              <>
                <section className="card protocol-checks" aria-labelledby="protocol-title">
                  <div className="card-head">
                    <h2 id="protocol-title">Protocol checks</h2>
                    <span className="muted">Rule-based, not AI</span>
                  </div>
                  <pre>{explanation.threshold_summary}</pre>
                </section>
                <AIPanel label="HemoSmart AI explanation">
                  <p>{explanation.llm_explanation}</p>
                  <Citations citations={explanation.citations} />
                </AIPanel>
              </>
            )}

            {/* 5. Recommendation: optional next step */}
            {needed.length > 0 && !transfusion && (
              <section className="card next-step" aria-labelledby="next-step-title">
                <div className="card-head">
                  <h2 id="next-step-title">Suggested next step</h2>
                  <span className="muted">Optional</span>
                </div>
                <p className="hint">
                  Reserve {needed.map((c) => `${c.units} ${c.component}`).join(", ")} for a {form.blood_group} patient.
                  Units closest to expiry are allocated first (FEFO), so less stock expires unused.
                </p>
                <div className="actions">
                  <button onClick={reserve} disabled={loading}>
                    <PackageCheck size={16} aria-hidden="true" /> {busy === "reserve" ? "Reserving…" : "Reserve blood"}
                  </button>
                </div>
                {error?.step === "reserve" && <ErrorState what={STEP_LABEL.reserve} error={error.error} onRetry={reserve} />}
              </section>
            )}

            {transfusion && (
              <section className="card" aria-labelledby="transfusion-title">
                <div className="card-head">
                  <h2 id="transfusion-title">Transfusion #{transfusion.id}</h2>
                  <SeverityBadge
                    level={transfusion.status === "issued" ? "low" : transfusion.status === "cancelled" ? "info" : "info"}
                    label={{ reserved: "Reserved", issued: "Transfused", cancelled: "Cancelled" }[transfusion.status] || transfusion.status} />
                </div>
                <p className="hint">For a {transfusion.recipient_group} patient.</p>
                {transfusion.units.length > 0 && (
                  <div className="table-scroll">
                    <table>
                      <thead><tr><th>Unit</th><th>Group</th><th>Component</th><th>Expires</th><th>Note</th></tr></thead>
                      <tbody>
                        {transfusion.units.map((u) => (
                          <tr key={u.unit_id}>
                            <td className="nowrap">{u.unit_id}</td>
                            <td>{u.blood_group}</td>
                            <td>{u.component}</td>
                            <td className="nowrap tabular">{u.expires_at.slice(0, 16).replace("T", " ")}</td>
                            <td>
                              {u.near_expiry && <SeverityBadge level="moderate" label="Near expiry" />}
                              {u.cross_group && <span className="badge info">Compatible group</span>}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
                {shortfall.length > 0 && (
                  <ul className="attention-list shortfall">
                    {shortfall.map(([component, n]) => (
                      <li key={component} className="attention-item">
                        <SeverityBadge level="high" />
                        <p>Short {n} {component} for this {transfusion.recipient_group} patient.</p>
                        {canAlertDonors ? (
                          <button className="secondary" onClick={() => alertFor(component)} disabled={loading}>
                            Alert {form.blood_group} donors
                          </button>
                        ) : <span className="muted">Ask a Blood Bank Manager to source more.</span>}
                      </li>
                    ))}
                  </ul>
                )}
                {alertMessage && <p className="state empty" role="status">{alertMessage}</p>}
                {error?.step === "alert" && <ErrorState what={STEP_LABEL.alert} error={error.error} />}
                {transfusion.status === "reserved" && (
                  <div className="actions">
                    <button onClick={() => finish(true)} disabled={loading}>Confirm transfused</button>
                    <button className="secondary" onClick={() => finish(false)} disabled={loading}>Cancel and release units</button>
                  </div>
                )}
                {error?.step === "finish" && <ErrorState what={STEP_LABEL.finish} error={error.error} />}
              </section>
            )}
          </>
        )}
      </div>
    </div>
  );
}
