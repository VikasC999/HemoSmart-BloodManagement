// Formatting helpers for the UI layer (no API or data-shape changes).

// The API helper throws `${status} ${statusText}: ${body}`; turn that
// into plain words for the user, keeping the status as a support ref.
export function describeError(error) {
  if (!error) return { message: "Something went wrong.", ref: null };
  const raw = typeof error === "string" ? error : error.message || String(error);
  const match = raw.match(/^(\d{3})\s[^:]*:\s?([\s\S]*)$/);
  if (!match) {
    if (/Failed to fetch|NetworkError|Load failed/i.test(raw)) {
      return { message: "Can't reach the HemoSmart server. Check your connection.", ref: null };
    }
    return { message: raw, ref: null };
  }
  const [, status, body] = match;
  let detail = body;
  try {
    const parsed = JSON.parse(body);
    detail = typeof parsed.detail === "string" ? parsed.detail : null;
  } catch {
    // body wasn't JSON; keep it
  }
  const byStatus = {
    400: "The request wasn't valid.",
    403: "Your role doesn't have access to this.",
    404: "That record wasn't found.",
    409: "Another job is already running.",
    422: "Some values weren't accepted.",
    500: "The server hit an error.",
    502: "The server is unavailable.",
    503: "The server is busy or unavailable.",
  };
  const base = byStatus[status] || "The request failed.";
  // Our backend's `detail` is already written for people; prefer it.
  const message = detail && status !== "500" ? detail : base;
  return { message, ref: `HTTP ${status}` };
}

export function formatAgo(iso) {
  const seconds = Math.round((Date.now() - new Date(iso).getTime()) / 1000);
  if (Number.isNaN(seconds)) return "";
  if (seconds < 45) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  const days = Math.round(hours / 24);
  return `${days} d ago`;
}

// How an agent cycle was started, in words.
export const TRIGGER_LABEL = { manual: "run by hand", scheduler: "scheduled", simulation: "simulation" };

// Probabilities as whole percentages; never "100.0%" false precision.
// Values that round to 0 or 100 but aren't exactly that say so.
export function formatPercent(p) {
  const pct = p * 100;
  if (pct > 99.5 && pct < 100) return ">99%";
  if (pct > 0 && pct < 0.5) return "<1%";
  return `${Math.round(pct)}%`;
}
