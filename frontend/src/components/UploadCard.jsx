import { useState } from "react";

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

export default function UploadCard() {
  const [file, setFile] = useState(null);
  const [kind, setKind] = useState("pdf");
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const submit = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const data = await uploadFile(`/api/predict/${kind}`, file);
      setResults(Array.isArray(data) ? data : [data]);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="card">
      <h2>Upload Lab Report</h2>
      <p className="hint">PDF: one patient's CBC report. CSV: a batch export, one prediction per row.</p>
      <div className="actions">
        <select value={kind} onChange={(e) => setKind(e.target.value)}>
          <option value="pdf">PDF report</option>
          <option value="csv">CSV export</option>
        </select>
        <input
          type="file"
          accept={kind === "pdf" ? ".pdf" : ".csv"}
          onChange={(e) => setFile(e.target.files[0] || null)}
        />
        <button onClick={submit} disabled={loading || !file}>Upload &amp; Predict</button>
      </div>
      {error && <p className="error">{error}</p>}
      {results && (
        <div className="result">
          {results.map((r, i) => (
            <p key={i}>
              {results.length > 1 ? `Row ${i + 1}: ` : ""}
              <strong>{r.transfusion_needed ? "Transfusion needed" : "No transfusion needed"}</strong>
              {" "}({(r.confidence * 100).toFixed(1)}%)
            </p>
          ))}
        </div>
      )}
    </section>
  );
}
