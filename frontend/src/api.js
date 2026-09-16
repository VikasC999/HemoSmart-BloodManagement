const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json();
}

export const api = {
  predict: (patient) =>
    request("/api/predict", { method: "POST", body: JSON.stringify(patient) }),

  explain: (patient, prediction) =>
    request("/api/explain", {
      method: "POST",
      body: JSON.stringify({ patient, prediction }),
    }),

  forecast: (days = 7) => request(`/api/forecast?days=${days}`),

  chat: (sessionId, message) =>
    request("/api/chat", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, message }),
    }),

  alertDonors: (bloodType) =>
    request("/api/donors/alert", {
      method: "POST",
      body: JSON.stringify({ blood_type: bloodType }),
    }),

  inventory: (bloodType) =>
    request(`/api/inventory${bloodType ? `?blood_type=${bloodType}` : ""}`),
};
