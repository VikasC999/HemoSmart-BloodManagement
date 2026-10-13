// Shared HemoSmart UI primitives. Every screen uses these so severity,
// AI content and loading/error/stale/empty states look the same
// everywhere (HemoSmart design skill §10, §12, §13).
import {
  CircleAlert, CircleCheck, Clock, Inbox, OctagonAlert, RotateCw, Sparkles, TriangleAlert, Info,
} from "lucide-react";
import { describeError, formatAgo } from "./format";

// ---------------------------------------------------------------------
// Severity: colour + icon + word, never colour alone
// ---------------------------------------------------------------------
const SEVERITY = {
  low: { icon: CircleCheck, label: "Low" },
  moderate: { icon: TriangleAlert, label: "Moderate" },
  high: { icon: CircleAlert, label: "High" },
  critical: { icon: OctagonAlert, label: "Critical" },
  info: { icon: Info, label: "Info" },
};

// The backend says "Low" for two opposite things, so each vocabulary
// gets its own mapper onto the four HemoSmart levels.

// Transfusion urgency: Critical / Moderate / Low (= low risk)
export function urgencySeverity(urgency) {
  return { Critical: "critical", Moderate: "moderate", Low: "low" }[urgency] || "info";
}

// Stock status vs par: Critical / Low (= running low, a warning) / Sufficient
export function stockSeverity(status) {
  return { Critical: "critical", Low: "moderate", Sufficient: "low" }[status] || "info";
}

export const STOCK_LABEL = { Critical: "Critical", Low: "Low stock", Sufficient: "Sufficient" };

export function SeverityBadge({ level, label, className = "" }) {
  const s = SEVERITY[level] || SEVERITY.info;
  const Icon = s.icon;
  return (
    <span className={`badge ${level} ${className}`}>
      <Icon size={13} aria-hidden="true" />
      {label || s.label}
    </span>
  );
}

// ---------------------------------------------------------------------
// Loading: skeletons shaped like the content. Never placeholder numbers.
// ---------------------------------------------------------------------
export function Skeleton({ width = "100%", height = 14, style, className = "" }) {
  return <span className={`skeleton ${className}`} style={{ width, height, ...style }} aria-hidden="true" />;
}

export function SkeletonLines({ lines = 3, label = "Loading" }) {
  return (
    <div role="status" aria-label={label} style={{ display: "grid", gap: 8 }}>
      {Array.from({ length: lines }, (_, i) => (
        <Skeleton key={i} width={`${92 - i * 14}%`} />
      ))}
    </div>
  );
}

export function SkeletonTable({ rows = 5, cols = 4, label = "Loading table" }) {
  return (
    <div role="status" aria-label={label} style={{ display: "grid", gap: 10, marginTop: 8 }}>
      {Array.from({ length: rows }, (_, r) => (
        <div key={r} style={{ display: "grid", gridTemplateColumns: `repeat(${cols}, 1fr)`, gap: 12 }}>
          {Array.from({ length: cols }, (_, c) => <Skeleton key={c} height={18} />)}
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------
// Empty, error, stale
// ---------------------------------------------------------------------
export function EmptyState({ icon: Icon = Inbox, title, children, action }) {
  return (
    <div className="state empty">
      <Icon size={18} aria-hidden="true" />
      <div>
        <p className="state-title">{title}</p>
        {children && <p className="state-body">{children}</p>}
        {action && <div className="actions">{action}</div>}
      </div>
    </div>
  );
}

// Shown where the failure happened. Says what failed, what is still
// valid, and what to do next. `blocking` (critical styling) only for
// failures that stop a clinical task.
export function ErrorState({ what, error, stillValid, onRetry, action, blocking = false }) {
  const { message, ref } = describeError(error);
  return (
    <div className={`state ${blocking ? "blocking" : "state-error"}`} role="alert">
      <CircleAlert size={18} aria-hidden="true" />
      <div>
        <p className="state-title">{what}</p>
        <p className="state-body">
          {message}
          {stillValid ? ` ${stillValid}` : ""}
        </p>
        {ref && <p className="ref">Reference: {ref}</p>}
        {(onRetry || action) && (
          <div className="actions">
            {onRetry && (
              <button className="secondary" onClick={onRetry}>
                <RotateCw size={15} aria-hidden="true" /> Retry
              </button>
            )}
            {action}
          </div>
        )}
      </div>
    </div>
  );
}

export function StaleNotice({ children }) {
  return (
    <div className="state stale" role="status">
      <Clock size={18} aria-hidden="true" />
      <p className="state-body" style={{ color: "var(--color-text)" }}>{children}</p>
    </div>
  );
}

// "Updated 4 min ago". `staleAfterMinutes` is opt-in: the refresh
// interval is a product decision, so screens don't invent one.
export function UpdatedAt({ iso, staleAfterMinutes, prefix = "Updated" }) {
  if (!iso) return null;
  const ageMin = (Date.now() - new Date(iso).getTime()) / 60000;
  const stale = staleAfterMinutes != null && ageMin > staleAfterMinutes;
  return (
    <span className={`updated ${stale ? "is-stale" : ""}`} title={new Date(iso).toLocaleString()}>
      <Clock size={12} aria-hidden="true" />
      {prefix} {formatAgo(iso)}
    </span>
  );
}

// ---------------------------------------------------------------------
// AI content: one consistent treatment, plus the decision-support line
// ---------------------------------------------------------------------
export function AIPanel({ label = "HemoSmart AI", children, decisionSupport = true, className = "" }) {
  return (
    <div className={`ai-panel ${className}`}>
      <p className="ai-label"><Sparkles size={14} aria-hidden="true" /> {label}</p>
      <div className="ai-body">{children}</div>
      {decisionSupport && <p className="decision-support">Decision support. Review before acting.</p>}
    </div>
  );
}
