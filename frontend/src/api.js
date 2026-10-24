const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function getToken() {
  try {
    return localStorage.getItem("hemosmart_token");
  } catch {
    return null; // private browsing / blocked storage -- fall through to unauthenticated
  }
}

export function setToken(token) {
  try {
    if (token) localStorage.setItem("hemosmart_token", token);
    else localStorage.removeItem("hemosmart_token");
  } catch {
    // per-viewer convenience only; app still works this request, just
    // won't remember the session on reload
  }
}

async function request(path, options = {}) {
  const token = getToken();
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    ...options,
  });
  if (res.status === 401) {
    setToken(null);
    window.location.reload(); // bounce back to the login screen
  }
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json();
}

export const auth = {
  login: async (email, password) => {
    const form = new URLSearchParams();
    form.set("username", email);
    form.set("password", password);
    const res = await fetch(`${BASE_URL}/api/auth/login`, { method: "POST", body: form });
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`${res.status} ${res.statusText}: ${body}`);
    }
    const data = await res.json();
    setToken(data.access_token);
    return data;
  },
  logout: () => setToken(null),
  isLoggedIn: () => Boolean(getToken()),
  me: () => request("/api/auth/me"),
};

const qs = (params) => {
  const entries = Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== "");
  return entries.length ? `?${new URLSearchParams(entries)}` : "";
};

export const api = {
  predict: (patient) =>
    request("/api/predict", { method: "POST", body: JSON.stringify(patient) }),

  explain: (patient, prediction, predictionId) =>
    request("/api/explain", {
      method: "POST",
      body: JSON.stringify({ patient, prediction, prediction_id: predictionId }),
    }),

  // Demand forecast + supply gap (shortages and wastage risks)
  outlook: (days = 7) => request(`/api/forecast?days=${days}`),

  chat: (sessionId, message) =>
    request("/api/chat", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, message }),
    }),
  // What the logged-in role can ask the assistant about
  chatCapabilities: () => request("/api/chat/capabilities"),

  // Inventory (unit level)
  inventory: (bloodGroup, component) => request(`/api/inventory${qs({ blood_group: bloodGroup, component })}`),
  inventoryUnits: (params = {}) => request(`/api/inventory/units${qs(params)}`),
  receiveUnits: (bloodGroup, component, count, collectedAt) =>
    request("/api/inventory/units", {
      method: "POST",
      body: JSON.stringify({ blood_group: bloodGroup, component, count, collected_at: collectedAt }),
    }),

  // Transfusion workflow
  requestTransfusion: (patientId, predictionId, bloodGroup) =>
    request("/api/transfusions", {
      method: "POST",
      body: JSON.stringify({ patient_id: patientId, prediction_id: predictionId, blood_group: bloodGroup }),
    }),
  confirmTransfusion: (id) => request(`/api/transfusions/${id}/confirm`, { method: "POST" }),
  cancelTransfusion: (id) => request(`/api/transfusions/${id}/cancel`, { method: "POST" }),

  // Donors
  donors: (bloodType) => request(`/api/donors${qs({ blood_type: bloodType })}`),
  alertDonors: (bloodType, component = "PRBC") =>
    request("/api/donors/alert", {
      method: "POST",
      body: JSON.stringify({ blood_type: bloodType, component }),
    }),
  recentDonorAlerts: (limit = 10, pendingOnly = false) =>
    request(`/api/donors/alerts/recent${qs({ limit, pending_only: pendingOnly || undefined })}`),
  recordDonorResponse: (alertId, responded) =>
    request(`/api/donors/alerts/${alertId}/response`, {
      method: "POST",
      body: JSON.stringify({ responded }),
    }),

  // Autonomous loop
  autonomyStatus: () => request("/api/autonomy/status"),
  autonomyCycles: (limit = 5) => request(`/api/autonomy/cycles?limit=${limit}`),
  runAutonomyCycle: () => request("/api/autonomy/run", { method: "POST" }),
  wastageMetrics: (days = 30) => request(`/api/autonomy/metrics?days=${days}`),

  // Agent monitor (read-only operations view, D71-D75)
  agentOverview: () => request("/api/agents/overview"),
  agentRuns: (limit = 10) => request(`/api/agents/runs?limit=${limit}`),
  agentRun: (runId) => request(`/api/agents/runs/${encodeURIComponent(runId)}`),

  // CrewAI full-case job (started via multipart upload in UploadCard)
  caseStatus: (jobId) => request(`/api/cases/${jobId}`),

  patients: (limit = 20) => request(`/api/patients?limit=${limit}`),

  patientHistory: (id) => request(`/api/patients/${id}/predictions`),

  auditLog: (limit = 50) => request(`/api/audit-log?limit=${limit}`),
};
