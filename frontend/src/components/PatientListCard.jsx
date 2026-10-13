import { Fragment, useEffect, useState } from "react";
import { ChevronDown, ChevronRight, RotateCw, Sparkles, Users } from "lucide-react";
import { api } from "../api";
import { EmptyState, ErrorState, SeverityBadge, SkeletonLines, SkeletonTable, urgencySeverity } from "./ui";
import { formatAgo } from "./ui/format";

function ResultSummary({ prediction }) {
  if (!prediction.transfusion_needed) return <span className="muted">Not indicated</span>;
  const needed = (prediction.components || []).filter((c) => c.needed);
  return (
    <span className="result-inline">
      <span>{needed.length ? needed.map((c) => `${c.units} ${c.component}`).join(", ") : "Transfusion needed"}</span>
      {prediction.urgency && (
        <SeverityBadge level={urgencySeverity(prediction.urgency)} label={prediction.urgency} />
      )}
    </span>
  );
}

const SOURCE_LABEL = { manual: "Manual", pdf: "PDF", csv: "CSV", seed: "Sample", pdf_agents: "AI agents", mcp: "Hospital sync" };

export default function PatientListCard() {
  const [patients, setPatients] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState(null);
  const [history, setHistory] = useState(null);
  const [historyError, setHistoryError] = useState(null);

  const load = async () => {
    setError(null);
    setLoading(true);
    try {
      setPatients(await api.patients(20));
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const toggleHistory = async (id) => {
    if (expandedId === id) {
      setExpandedId(null);
      setHistory(null);
      return;
    }
    setExpandedId(id);
    setHistory(null);
    setHistoryError(null);
    try {
      setHistory(await api.patientHistory(id));
    } catch (e) {
      setHistoryError(e);
    }
  };

  return (
    <section className="card" aria-labelledby="patients-title">
      <div className="card-head">
        <div>
          <h2 id="patients-title">Recent patients</h2>
          {patients && <p className="card-sub">The 20 most recent, newest first</p>}
        </div>
        <button className="secondary" onClick={load} disabled={loading}>
          <RotateCw size={15} aria-hidden="true" /> Refresh
        </button>
      </div>
      {error && <ErrorState what="Couldn't load patients." error={error} onRetry={load} />}
      {loading && !patients && <SkeletonTable rows={6} cols={5} label="Loading patients" />}
      {patients && patients.length === 0 && (
        <EmptyState icon={Users} title="No patients yet">
          Patients appear here after a prediction from manual entry, a CBC upload or a CSV batch.
        </EmptyState>
      )}
      {patients && patients.length > 0 && (
        <div className="table-scroll">
          <table className="patients-table">
            <thead>
              <tr>
                <th>Patient</th><th>Group</th><th>Surgery</th>
                <th className="num">Hb</th><th className="num">Platelets</th><th className="num">INR</th><th className="num">Age</th>
                <th>Latest prediction</th><th><span className="visually-hidden">History</span></th>
              </tr>
            </thead>
            <tbody>
              {patients.map((p) => (
                <Fragment key={p.id}>
                  <tr className={expandedId === p.id ? "is-expanded" : ""}>
                    <td className="nowrap">
                      <strong>#{p.id}</strong>
                      <span className="muted cell-sub">{SOURCE_LABEL[p.source_format] || p.source_format} · {formatAgo(p.created_at)}</span>
                    </td>
                    <td>{p.blood_group || <span className="muted">Unknown</span>}</td>
                    <td>{p.surgery_type}</td>
                    <td className="num">{p.hemoglobin}</td>
                    <td className="num">{p.platelets.toLocaleString()}</td>
                    <td className="num">{p.inr}</td>
                    <td className="num">{p.age}</td>
                    <td>{p.latest_prediction ? <ResultSummary prediction={p.latest_prediction} /> : <span className="muted">None</span>}</td>
                    <td>
                      <button className="ghost icon-button" onClick={() => toggleHistory(p.id)}
                              aria-expanded={expandedId === p.id}
                              aria-label={`${expandedId === p.id ? "Hide" : "Show"} prediction history for patient ${p.id}`}>
                        {expandedId === p.id ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
                      </button>
                    </td>
                  </tr>
                  {expandedId === p.id && (
                    <tr className="history-row">
                      <td colSpan={9}>
                        {historyError ? (
                          <ErrorState what="Couldn't load this patient's history." error={historyError}
                                      onRetry={() => { setExpandedId(null); toggleHistory(p.id); }} />
                        ) : !history ? <SkeletonLines lines={2} label="Loading history" /> : history.length === 0 ? (
                          <span className="muted">No prediction history.</span>
                        ) : (
                          <ol className="history-list">
                            {history.map((h) => (
                              <li key={h.id}>
                                <span className="muted tabular nowrap">{h.created_at.slice(0, 16).replace("T", " ")}</span>
                                <ResultSummary prediction={h} />
                                {h.explanation && (
                                  <p className="history-ai">
                                    <span className="ai-label"><Sparkles size={13} aria-hidden="true" /> HemoSmart AI</span>
                                    {h.explanation}
                                  </p>
                                )}
                              </li>
                            ))}
                          </ol>
                        )}
                      </td>
                    </tr>
                  )}
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
