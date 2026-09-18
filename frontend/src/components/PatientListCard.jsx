import { Fragment, useState } from "react";
import { api } from "../api";

export default function PatientListCard() {
  const [patients, setPatients] = useState(null);
  const [error, setError] = useState(null);
  const [expandedId, setExpandedId] = useState(null);
  const [history, setHistory] = useState(null);

  const load = async () => {
    setError(null);
    try {
      setPatients(await api.patients(20));
    } catch (e) {
      setError(e.message);
    }
  };

  const toggleHistory = async (id) => {
    if (expandedId === id) {
      setExpandedId(null);
      setHistory(null);
      return;
    }
    setExpandedId(id);
    try {
      setHistory(await api.patientHistory(id));
    } catch (e) {
      setError(e.message);
    }
  };

  return (
    <section className="card">
      <h2>Patient History</h2>
      <button onClick={load}>Refresh</button>
      {error && <p className="error">{error}</p>}
      {patients && (
        <table>
          <thead>
            <tr><th>ID</th><th>Surgery</th><th>Hb / Plt / INR / Age</th><th>Latest Result</th><th></th></tr>
          </thead>
          <tbody>
            {patients.map((p) => (
              <Fragment key={p.id}>
                <tr>
                  <td>{p.id}</td>
                  <td>{p.surgery_type}</td>
                  <td>{p.hemoglobin} / {p.platelets} / {p.inr} / {p.age}</td>
                  <td>
                    {p.latest_prediction ? (
                      p.latest_prediction.transfusion_needed
                        ? `Needed (${(p.latest_prediction.confidence * 100).toFixed(0)}%)`
                        : "Not needed"
                    ) : "—"}
                  </td>
                  <td><button onClick={() => toggleHistory(p.id)}>{expandedId === p.id ? "Hide" : "History"}</button></td>
                </tr>
                {expandedId === p.id && history && (
                  <tr>
                    <td colSpan={5} className="history-row">
                      {history.length === 0 ? "No prediction history." : (
                        <ul>
                          {history.map((h) => (
                            <li key={h.id}>
                              {h.created_at}: {h.transfusion_needed ? "Needed" : "Not needed"} ({(h.confidence * 100).toFixed(1)}%)
                            </li>
                          ))}
                        </ul>
                      )}
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
