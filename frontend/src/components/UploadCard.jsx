import { useEffect, useRef, useState } from "react";
import {
  ArrowRight, Bot, Circle, CircleCheck, FileText, LoaderCircle, RotateCcw, Sheet, SquarePen, Stethoscope,
  UploadCloud, X,
} from "lucide-react";
import { api } from "../api";
import { BLOOD_TYPES, SURGERY_TYPES } from "../constants";
import { AIPanel, EmptyState, ErrorState, SeverityBadge, urgencySeverity } from "./ui";

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

const MODES = {
  pdf: {
    label: "PDF report", icon: FileText, accept: ".pdf", action: "Extract values",
    help: "One patient's CBC report. HemoSmart AI reads the values; you check them before anything is predicted.",
  },
  csv: {
    label: "CSV batch", icon: Sheet, accept: ".csv", action: "Predict all rows",
    help: "A hospital export with one patient per row. Each valid row gets its own prediction.",
  },
  case: {
    label: "Full case (AI agents)", icon: Bot, accept: ".pdf", action: "Run AI agents",
    help: "Six AI agents process the PDF end to end, each checked against an independent calculation. About a minute.",
  },
};

const formatSize = (bytes) => (bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / 1048576).toFixed(1)} MB`);

function useElapsed(running) {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    if (!running) return undefined;
    setSeconds(0);
    const started = Date.now();
    const t = setInterval(() => setSeconds(Math.round((Date.now() - started) / 1000)), 500);
    return () => clearInterval(t);
  }, [running]);
  return seconds;
}

// ✓ done → ● active → ○ to do, with elapsed time on the active step
// "waiting" = it's the user's turn (no spinner: nothing is processing)
const STEP_ICON = { done: CircleCheck, active: LoaderCircle, waiting: SquarePen, todo: Circle, fell_back: RotateCcw };
function ProcessingSteps({ steps, elapsed }) {
  return (
    <ol className="steps" aria-label="Processing steps">
      {steps.map((s) => {
        const Icon = STEP_ICON[s.state];
        return (
          <li key={s.label} className={`step step-${s.state}`}
              aria-current={s.state === "active" || s.state === "waiting" ? "step" : undefined}>
            <Icon size={16} className={s.state === "active" ? "spin" : ""} aria-hidden="true" />
            <span>{s.label}</span>
            {s.state === "active" && elapsed != null && <span className="muted tabular"> · {elapsed}s</span>}
            {s.note && <span className="step-note">{s.note}</span>}
          </li>
        );
      })}
    </ol>
  );
}

function ResultSummary({ r, label }) {
  const needed = r.components.filter((c) => c.needed);
  return (
    <div className="result-summary">
      {label && <span className="muted result-row-label">{label}</span>}
      <strong>{r.transfusion_needed ? "Transfusion likely needed" : "Transfusion not indicated"}</strong>
      <SeverityBadge level={urgencySeverity(r.urgency)} label={`${r.urgency} urgency`} />
      <span className="muted">
        {needed.length ? needed.map((c) => `${c.units} ${c.component}`).join(", ") : "No components"}
        {" · "}model probability {(r.confidence * 100).toFixed(0)}%
        {r.blood_group ? ` · ${r.blood_group}` : ""}
      </span>
    </div>
  );
}

// The CrewAI pipeline's run, agent by agent. "verified" = the LLM agent's
// own answer matched an independent check and was used; "fell back" = it
// failed or disagreed, so the checked value was used instead.
const AGENT_STATE = { pending: "todo", running: "active", verified: "done", fell_back: "fell_back" };
const describeDetail = (detail) => (detail && typeof detail === "object"
  ? Object.entries(detail).map(([k, v]) => `${k.replace(/_/g, " ")}: ${Array.isArray(v) ? v.join(", ") : v}`).join(" · ")
  : detail || null);

function AgentCase({ job, elapsed }) {
  const r = job.result;
  const action = r?.action;
  const running = job.status === "queued" || job.status === "running";
  return (
    <div className="upload-section">
      <ProcessingSteps
        elapsed={running ? elapsed : null}
        steps={job.steps.map((s) => ({
          label: `${s.agent} agent${s.status === "fell_back" ? " · used the verified value" : ""}`,
          state: AGENT_STATE[s.status] || "todo",
          note: describeDetail(s.detail),
        }))} />
      {job.status === "failed" && <ErrorState what="The agent pipeline stopped." error={job.error} />}
      {r && (
        <>
          <ResultSummary r={r.prediction} />
          <AIPanel label="HemoSmart AI · case summary">
            <p>{r.orchestrator.case_summary}</p>
            <p className="muted" style={{ marginTop: 8 }}>{r.explanation.delivered_explanation}</p>
          </AIPanel>
          {action && (
            <p className="hint">
              {action.dispatched
                ? `Donor alerts sent: ${(action.actually_alerted || []).join(", ") || "none needed"}.`
                : `Donor alerts recommended (needs a Blood Bank Manager): ${(action.recommended || []).join(", ") || "none"}.`}
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
  { key: "hemoglobin", label: "Hemoglobin", unit: "g/dL", step: "0.1" },
  { key: "platelets", label: "Platelets", unit: "/µL", step: "1000" },
  { key: "INR", label: "INR", unit: "ratio", step: "0.1" },
  { key: "age", label: "Age", unit: "years", step: "1" },
];

function ConfirmExtracted({ fileName, values, onChange, onConfirm, onCancel, busy }) {
  const missing = FIELDS.filter((f) => values[f.key] === "" || values[f.key] == null || Number.isNaN(Number(values[f.key])));
  return (
    <div className="upload-section">
      <AIPanel label="Extracted by HemoSmart AI" decisionSupport={false}>
        <p>Check each value against <strong>{fileName}</strong> and correct anything that's wrong.
          Nothing is predicted or saved until you confirm.</p>
      </AIPanel>
      <div className="table-scroll">
        <table className="extracted-table">
          <thead><tr><th>Field</th><th>Value</th><th>Unit</th></tr></thead>
          <tbody>
            {FIELDS.map((f) => {
              const bad = missing.includes(f);
              return (
                <tr key={f.key} className={bad ? "row-missing" : ""}>
                  <th scope="row"><label htmlFor={`x-${f.key}`}>{f.label}</label></th>
                  <td>
                    <input id={`x-${f.key}`} type="number" step={f.step} value={values[f.key] ?? ""}
                           aria-invalid={bad ? "true" : undefined}
                           aria-describedby={bad ? `x-${f.key}-err` : undefined}
                           onChange={(e) => onChange({ ...values, [f.key]: e.target.value })} />
                    {bad && <span id={`x-${f.key}-err`} className="field-error">Required. Not found in the report: enter it.</span>}
                  </td>
                  <td className="muted">{f.unit}</td>
                </tr>
              );
            })}
            <tr>
              <th scope="row"><label htmlFor="x-surgery">Surgery type</label></th>
              <td>
                <select id="x-surgery" value={values.surgery_type}
                        onChange={(e) => onChange({ ...values, surgery_type: e.target.value })}>
                  {SURGERY_TYPES.map((t) => <option key={t}>{t}</option>)}
                </select>
              </td>
              <td />
            </tr>
            <tr>
              <th scope="row"><label htmlFor="x-group">Blood group</label></th>
              <td>
                <select id="x-group" value={values.blood_group || ""}
                        onChange={(e) => onChange({ ...values, blood_group: e.target.value || null })}>
                  <option value="">Unknown</option>
                  {BLOOD_TYPES.map((g) => <option key={g}>{g}</option>)}
                </select>
              </td>
              <td />
            </tr>
          </tbody>
        </table>
      </div>
      <div className="actions">
        <button onClick={onConfirm} disabled={busy || missing.length > 0}>
          <Stethoscope size={16} aria-hidden="true" /> {busy ? "Predicting…" : "Confirm and predict"}
        </button>
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
  const [error, setError] = useState(null); // { what, error }
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const poll = useRef(null);
  const input = useRef(null);
  const elapsed = useElapsed(loading);

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
        setError({ what: "Lost track of the agent run.", error: e });
        setLoading(false);
      }
    }, 1500);
  };

  const reset = () => {
    setResults(null);
    setJob(null);
    setExtracted(null);
    setError(null);
  };

  const choose = (f) => {
    if (!f) return;
    reset();
    setFile(f);
  };

  const submit = async () => {
    if (!file) return;
    setLoading(true);
    reset();
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
      setError({
        what: kind === "pdf" ? "Couldn't read this PDF." : kind === "csv" ? "Couldn't process this CSV." : "The agent run didn't start.",
        error: e,
      });
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
      setError({ what: "The prediction didn't run.", error: e });
    }
    setLoading(false);
  };

  const mode = MODES[kind];
  const pdfSteps = [
    { label: "Read file", state: "done" },
    { label: "Extract CBC values with HemoSmart AI", state: loading && !extracted && !results ? "active" : extracted || results ? "done" : "todo" },
    { label: "Your check", state: extracted ? (loading ? "done" : "waiting") : results ? "done" : "todo" },
    { label: "Prediction", state: results ? "done" : loading && extracted ? "active" : "todo" },
  ];
  const showPdfSteps = kind === "pdf" && (loading || extracted || results);

  return (
    <section className="card upload-card" aria-labelledby="upload-title">
      <h2 id="upload-title" className="visually-hidden">Upload a report</h2>
      <div className="segmented" role="radiogroup" aria-label="What are you uploading?">
        {Object.entries(MODES).map(([key, m]) => (
          <button key={key} role="radio" aria-checked={kind === key} disabled={loading}
                  className={`segment ${kind === key ? "active" : ""}`}
                  onClick={() => { setKind(key); setFile(null); reset(); }}>
            <m.icon size={16} aria-hidden="true" /> {m.label}
          </button>
        ))}
      </div>
      <p className="hint">{mode.help}</p>

      {!file ? (
        <div className={`dropzone ${dragging ? "dragging" : ""}`}
             onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
             onDragLeave={() => setDragging(false)}
             onDrop={(e) => { e.preventDefault(); setDragging(false); choose(e.dataTransfer.files[0]); }}>
          <UploadCloud size={28} aria-hidden="true" />
          <p className="dropzone-title">Drag and drop a {mode.accept.slice(1).toUpperCase()} file here</p>
          <p className="muted">or</p>
          <button className="secondary" onClick={() => input.current?.click()}>Browse files</button>
          <p className="muted">{mode.accept.slice(1).toUpperCase()} only</p>
          <input ref={input} type="file" accept={mode.accept} className="visually-hidden" tabIndex={-1}
                 onChange={(e) => choose(e.target.files[0] || null)} />
        </div>
      ) : (
        <div className="file-chip">
          <mode.icon size={18} aria-hidden="true" />
          <span className="file-name">{file.name}</span>
          <span className="muted">{formatSize(file.size)}</span>
          <button className="ghost icon-button" onClick={() => { setFile(null); reset(); }}
                  disabled={loading} aria-label={`Remove ${file.name}`}>
            <X size={16} />
          </button>
        </div>
      )}

      {file && !extracted && !(job && loading) && (
        <div className="actions">
          <button onClick={submit} disabled={loading}>
            {loading ? (kind === "case" ? "Starting agents…" : "Working…") : mode.action}
          </button>
        </div>
      )}

      {showPdfSteps && <ProcessingSteps steps={pdfSteps} elapsed={loading ? elapsed : null} />}

      {error && (
        <ErrorState what={error.what} error={error.error} onRetry={file ? submit : undefined}
                    action={kind === "pdf" && (
                      <a className="text-link" href="#predict">Enter values by hand <ArrowRight size={14} aria-hidden="true" /></a>
                    )} />
      )}

      {extracted && (
        <ConfirmExtracted fileName={file?.name} values={extracted} onChange={setExtracted}
                          onConfirm={confirmExtracted} onCancel={() => setExtracted(null)} busy={loading} />
      )}

      {job && <AgentCase job={job} elapsed={elapsed} />}

      {results && (
        <div className="upload-section">
          {results.length === 0 ? (
            <EmptyState title="No rows could be predicted">Check that the CSV has the expected columns.</EmptyState>
          ) : (
            <>
              <h3 className="section-title">
                {results.length === 1 ? "Prediction" : `${results.length} predictions`}
              </h3>
              <ul className="plain-list result-list">
                {results.map((r, i) => (
                  <li key={i}><ResultSummary r={r} label={results.length > 1 ? `Row ${i + 1}` : null} /></li>
                ))}
              </ul>
              <p className="decision-support">
                Decision support. Review before acting. Saved to <a className="text-link" href="#patients">Patients</a>.
              </p>
            </>
          )}
        </div>
      )}
    </section>
  );
}
