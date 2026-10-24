import { useCallback, useEffect, useRef, useState } from "react";
import { CircleAlert, CircleCheck, CircleDot, Hourglass, MinusCircle, TriangleAlert, X } from "lucide-react";
import { api } from "../api";
import { ErrorState, SkeletonLines } from "./ui";
import { parseTime, runLabel } from "./ui/format";

// Status: icon + word, never colour alone.
const STEP = {
  ok: { icon: CircleCheck, label: "OK" },
  verified: { icon: CircleCheck, label: "Verified" },
  fell_back: { icon: TriangleAlert, label: "Fell back to the verified value" },
  skipped: { icon: MinusCircle, label: "Skipped" },
  error: { icon: CircleAlert, label: "Error" },
  running: { icon: CircleDot, label: "Running" },
};

const t = (iso) => parseTime(iso).getTime();
const secs = (msValue) => (msValue == null ? "" : msValue < 1000 ? `${msValue} ms` : `${(msValue / 1000).toFixed(1)} s`);

// One run's steps as horizontal bars on a shared time axis, with a
// dashed segment for each LLM rate-limit wait. The same facts are
// listed as text below the bars, so the chart is never the only source.
export default function AgentRunTimeline({ runId, onClose, live = false, cycleLink }) {
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState(null);
  const panel = useRef(null);
  const opener = useRef(document.activeElement);

  const load = useCallback(async () => {
    try {
      setDetail(await api.agentRun(runId));
      setError(null);
    } catch (e) {
      setError(e);
    }
  }, [runId]);

  useEffect(() => {
    load();
    if (!live) return undefined;
    const id = setInterval(load, 2000);
    return () => clearInterval(id);
  }, [load, live]);

  // Dialog behaviour: Escape closes, focus moves in and back out, no
  // page scroll behind it.
  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    panel.current?.focus();
    const back = opener.current;
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
      back?.focus?.();
    };
  }, [onClose]);

  const run = detail?.run;
  const start = run ? t(run.started_at) : 0;
  const end = run
    ? Math.max(run.finished_at ? t(run.finished_at) : Date.now(),
      ...detail.steps.map((s) => (s.finished_at ? t(s.finished_at) : Date.now())))
    : 0;
  const span = Math.max(1, end - start);
  const pos = (ms) => `${Math.min(100, Math.max(0, ((ms - start) / span) * 100))}%`;
  const width = (from, to) => `${Math.max(0.6, ((to - from) / span) * 100)}%`;

  return (
    <div className="run-drawer-scrim" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <aside className="run-drawer" role="dialog" aria-modal="true" aria-labelledby="run-drawer-title"
             tabIndex={-1} ref={panel}>
        <div className="run-drawer-head">
          <div>
            <h2 id="run-drawer-title">{runLabel(runId)} timeline</h2>
            {run && (
              <p className="card-sub">
                {run.layer === "case_pipeline" ? "Case pipeline" : "Supply loop"} · {run.steps_done} of {run.steps_total} steps
                {run.duration_ms != null && ` · ${secs(run.duration_ms)}`} · {run.llm_calls} LLM call{run.llm_calls === 1 ? "" : "s"}
                {run.waits > 0 && ` · ${run.waits} rate-limit wait${run.waits === 1 ? "" : "s"} (${run.wait_s} s)`}
              </p>
            )}
          </div>
          <button className="ghost icon-button" onClick={onClose} aria-label="Close timeline"><X size={18} /></button>
        </div>

        {error && <ErrorState what="Couldn't load this run." error={error} onRetry={load} />}
        {!detail && !error && <SkeletonLines lines={5} label="Loading timeline" />}

        {detail && (
          <>
            <div className="timeline" aria-hidden="true">
              {detail.steps.map((s, i) => {
                const from = t(s.started_at);
                const to = s.finished_at ? t(s.finished_at) : Date.now();
                return (
                  <div className="timeline-row" key={i}>
                    <span className="timeline-label">{s.agent}</span>
                    <div className="timeline-track">
                      <span className={`timeline-bar status-${s.status}`} style={{ left: pos(from), width: width(from, to) }} />
                      {s.waits.map((w, j) => (
                        <span key={j} className="timeline-wait"
                              style={{ left: pos(t(w.at)), width: width(t(w.at), t(w.at) + (w.seconds || 0) * 1000) }} />
                      ))}
                    </div>
                  </div>
                );
              })}
              <p className="timeline-legend muted">
                <span className="legend-swatch solid" /> step <span className="legend-swatch dashed" /> waiting on the LLM rate limit
              </p>
            </div>

            <ol className="timeline-text">
              {detail.steps.map((s, i) => {
                const st = STEP[s.status] || STEP.running;
                const Icon = st.icon;
                return (
                  <li key={i} className={`timeline-item status-${s.status}`}>
                    <Icon size={15} aria-hidden="true" />
                    <span>
                      <strong>{s.agent}</strong>{s.step && s.step !== s.agent ? ` · ${s.step}` : ""}: {st.label}
                      {s.duration_ms != null && ` in ${secs(s.duration_ms)}`}
                      {s.llm_calls ? ` · ${s.llm_calls} LLM call${s.llm_calls === 1 ? "" : "s"}` : ""}
                      {s.waits.length > 0 && (
                        <span className="muted"> · <Hourglass size={12} aria-hidden="true" /> waited
                          {" "}{s.waits.map((w) => `${w.seconds} s`).join(", ")} for the rate limit</span>
                      )}
                    </span>
                  </li>
                );
              })}
            </ol>

            {detail.cycle_id != null && cycleLink && (
              <p className="muted">
                What this cycle observed and decided is in <a href="#agents" onClick={onClose}>{cycleLink}</a>.
              </p>
            )}
          </>
        )}
      </aside>
    </div>
  );
}
