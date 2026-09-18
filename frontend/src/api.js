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
