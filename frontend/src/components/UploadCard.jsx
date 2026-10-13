import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { BLOOD_TYPES, SURGERY_TYPES } from "../constants";

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Not going through api.js's request() helper -- file uploads need
// multipart/form-data, not the JSON Content-Type that helper always
// sets, so this builds the fetch call directly (still attaching the
// same bearer token from localStorage).
async function uploadFile(path, file) {
  const token = (() => {
    try { return localStorage.getItem("hemosmart_token"); } catch { return null; }
  })();
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: formData,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json();
}

const STEP_ICON = { pending: "○", running: "…", verified: "✓", fell_back: "↺" };

function PredictionLine({ r, prefix = "" }) {
  return (
    <p>
      {prefix}
      <strong>{r.transfusion_needed ? "Transfusion needed" : "No transfusion needed"}</strong>
      {" "}({(r.confidence * 100).toFixed(1)}%)
      {r.transfusion_needed && (
        <>
          {" — "}
          {r.components.filter((c) => c.needed).map((c) => `${c.component}×${c.units}`).join(", ")}
          <span className={`badge urgency-${r.urgency.toLowerCase()}`}>{r.urgency}</span>
        </>
      )}
      {r.blood_group && <span className="muted"> · {r.blood_group}</span>}
    </p>
  );
}

// The CrewAI pipeline's run, agent by agent. "verified" = the LLM agent's
// own answer matched an independent check and was used; "fell back" = it
// failed or disagreed, so the checked value was used instead.
function AgentCase({ job }) {
  const r = job.result;
  const action = r?.action;
  return (
    <div className="result">
      <ol className="events">
        {job.steps.map((s) => (
          <li key={s.agent} className={`event step-${s.status}`}>
            <span className="event-agent">{STEP_ICON[s.status]} {s.agent}</span>
            <span className="event-kind">{s.status.replace("_", " ")}</span>
            <span className="event-msg muted">{s.detail ? JSON.stringify(s.detail) : ""}</span>
          </li>
        ))}
      </ol>
      {job.status === "failed" && <p className="error">{job.error}</p>}
      {r && (
        <>
          <PredictionLine r={r.prediction} />
          <p className="brief">{r.orchestrator.case_summary}</p>
          <h4>Explanation (RAG agent)</h4>
          <p className="pre-wrap">{r.explanation.delivered_explanation}</p>
          {action && (
            <p>
              {action.dispatched
                ? `Donor alerts sent: ${(action.actually_alerted || []).join(", ") || "none needed"}`
                : `Donor alerts recommended (needs a Blood Bank Manager): ${(action.recommended || []).join(", ") || "none"}`}
            </p>
          )}
        </>
      )}
      {job.seconds != null && (
        <p className="muted">{job.llm_calls} LLM calls · {job.seconds}s · patient #{r?.patient_id ?? "—"}</p>
      )}
    </div>
  );
}

// FR-03: values extracted from a PDF are shown for the clinician to check
// and correct BEFORE anything is predicted or saved.
const FIELDS = [
  { key: "hemoglobin", label: "Hemoglobin (g/dL)", step: "0.1" },
  { key: "platelets", label: "Platelets (/µL)", step: "1000" },
  { key: "INR", label: "INR", step: "0.1" },
  { key: "age", label: "Age (years)", step: "1" },
];

function ConfirmExtracted({ values, onChange, onConfirm, onCancel, busy }) {
  return (
    <div className="result">
      <p><strong>Extracted from the PDF — check before predicting.</strong></p>
      <div className="form-grid">
        {FIELDS.map((f) => (
          <label key={f.key}>
            {f.label}
            <input type="number" step={f.step} value={values[f.key] ?? ""}
                   onChange={(e) => onChange({ ...values, [f.key]: e.target.value })} />
          </label>
        ))}
        <label>
          Surgery type
          <select value={values.surgery_type}
                  onChange={(e) => onChange({ ...values, surgery_type: e.target.value })}>
            {SURGERY_TYPES.map((t) => <option key={t}>{t}</option>)}
          </select>
        </label>
        <label>
          Blood group
          <select value={values.blood_group || ""}
                  onChange={(e) => onChange({ ...values, blood_group: e.target.value || null })}>
            <option value="">Unknown</option>
            {BLOOD_TYPES.map((g) => <option key={g}>{g}</option>)}
          </select>
        </label>
      </div>
      <div className="actions">
        <button onClick={onConfirm} disabled={busy}>{busy ? "Predicting…" : "Confirm & predict"}</button>
        <button className="secondary" onClick={onCancel} disabled={busy}>Discard</button>
      </div>
    </div>
  );
}

export default function UploadCard() {
  const [file, setFile] = useState(null);
  const [kind, setKind] = useState("pdf");
  const [results, setResults] = useState(null);
  const [job, setJob] = useState(null);
  const [extracted, setExtracted] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const poll = useRef(null);

  useEffect(() => () => clearInterval(poll.current), []);

  const watch = (jobId) => {
    clearInterval(poll.current);
    poll.current = setInterval(async () => {
      try {
        const j = await api.caseStatus(jobId);
        setJob(j);
        if (j.status === "done" || j.status === "failed") {
          clearInterval(poll.current);
          setLoading(false);
        }
      } catch (e) {
        clearInterval(poll.current);
        setError(e.message);
        setLoading(false);
      }
    }, 1500);
  };

  const submit = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResults(null);
    setJob(null);
    setExtracted(null);
    try {
      if (kind === "pdf") {
        setExtracted(await uploadFile("/api/predict/pdf/extract", file));
        setLoading(false);
        return;
      }
      if (kind === "case") {
        const { job_id } = await uploadFile("/api/cases", file);
        watch(job_id);
        return; // stays loading until the job finishes
      }
      const data = await uploadFile(`/api/predict/${kind}`, file);
      setResults(Array.isArray(data) ? data : [data]);
    } catch (e) {
      setError(e.message);
    }
    setLoading(false);
  };

  const confirmExtracted = async () => {
    setLoading(true);
    setError(null);
    try {
      const v = extracted;
      const r = await api.predict({
        ...v,
        hemoglobin: Number(v.hemoglobin),
        platelets: Math.round(Number(v.platelets)),
        INR: Number(v.INR),
        age: Math.round(Number(v.age)),
        source_format: "pdf",
      });
      setResults([r]);
      setExtracted(null);
    } catch (e) {
      setError(e.message);
    }
    setLoading(false);
  };

  return (
    <section className="card">
      <h2>Upload Lab Report</h2>
      <p className="hint">
        PDF: one patient's CBC report — values are extracted for you to confirm, then predicted.
        CSV: a batch export, one prediction per row.
        Full case: six AI agents (CrewAI) process the PDF end to end — slower (~15–20 LLM calls).
      </p>
      <div className="actions">
        <select value={kind} onChange={(e) => setKind(e.target.value)}>
          <option value="pdf">PDF report</option>
          <option value="csv">CSV export</option>
          <option value="case">Full case (AI agents)</option>
        </select>
        <input
          type="file"
          accept={kind === "csv" ? ".csv" : ".pdf"}
          onChange={(e) => setFile(e.target.files[0] || null)}
        />
        <button onClick={submit} disabled={loading || !file}>
          {kind === "case" ? (loading ? "Agents working…" : "Run agents")
            : kind === "pdf" ? (loading && !extracted ? "Extracting…" : "Extract values")
            : "Upload & Predict"}
        </button>
      </div>
      {error && <p className="error">{error}</p>}
      {extracted && (
        <ConfirmExtracted values={extracted} onChange={setExtracted} onConfirm={confirmExtracted}
                          onCancel={() => setExtracted(null)} busy={loading} />
      )}
      {job && <AgentCase job={job} />}
      {results && (
        <div className="result">
          {results.map((r, i) => (
            <PredictionLine key={i} r={r} prefix={results.length > 1 ? `Row ${i + 1}: ` : ""} />
          ))}
        </div>
      )}
    </section>
  );
}
