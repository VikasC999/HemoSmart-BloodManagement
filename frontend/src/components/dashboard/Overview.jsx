import { useCallback, useEffect, useState } from "react";
import { ArrowRight, CircleCheck, Play, RotateCw } from "lucide-react";
import { api } from "../../api";
import { COMPONENTS } from "../../constants";
import { PageHeader } from "../layout/AppShell";
import {
  AIPanel, EmptyState, ErrorState, SeverityBadge, Skeleton, SkeletonLines, UpdatedAt, stockSeverity,
} from "../ui";
import { TRIGGER_LABEL, formatAgo } from "../ui/format";

// One loader per block, so a failed forecast doesn't blank the inventory.
function useLoad(fn) {
  const [state, setState] = useState({ data: null, error: null, loading: true });
  const load = useCallback(async () => {
    setState((s) => ({ ...s, error: null, loading: true }));
    try {
      setState({ data: await fn(), error: null, loading: false });
    } catch (error) {
      setState((s) => ({ data: s.data, error, loading: false }));
    }
  }, [fn]);
  useEffect(() => { load(); }, [load]);
  return { ...state, reload: load };
}

const loadInventory = () => api.inventory();
const loadOutlook = () => api.outlook(7);
const loadAgents = () => Promise.all([api.autonomyStatus(), api.wastageMetrics(30)]);
const loadAlerts = () => api.recentDonorAlerts(6);

const SEVERITY_ORDER = { critical: 0, high: 1, moderate: 2 };

// What needs attention, worst first. Every item says what is wrong and
// where to act (HemoSmart §12: icon + label + sentence).
function attentionItems(inventory, outlook) {
  const items = [];
  for (const s of inventory?.low_stock || []) {
    if (s.status === "Critical") {
      items.push({
        level: "critical", key: `crit-${s.blood_group}-${s.component}`,
        subject: `${s.blood_group} ${s.component}`,
        text: `${s.available} unit${s.available === 1 ? "" : "s"} left against a par of ${Math.round(s.par)}.`,
        action: { href: "#donorOutreach", label: "Alert donors" },
      });
    }
  }
  for (const s of outlook?.shortages || []) {
    items.push({
      level: "high", key: `short-${s.blood_group}-${s.component}`,
      subject: `${s.blood_group} ${s.component}`,
      text: `Forecast to run ${s.projected_shortage} unit${s.projected_shortage === 1 ? "" : "s"} short from ${s.first_shortage_date}.`,
      action: { href: "#donorOutreach", label: "Alert donors" },
    });
  }
  for (const s of inventory?.low_stock || []) {
    if (s.status === "Low") {
      items.push({
        level: "moderate", key: `low-${s.blood_group}-${s.component}`,
        subject: `${s.blood_group} ${s.component}`,
        text: `Below par: ${s.available} of ${Math.round(s.par)} units.`,
        action: { href: "#inventory", label: "Inventory" },
      });
    }
  }
  for (const w of outlook?.wastage_risks || []) {
    items.push({
      level: "moderate", key: `waste-${w.blood_group}-${w.component}`,
      subject: `${w.blood_group} ${w.component}`,
      text: `${w.projected_wastage} unit${w.projected_wastage === 1 ? "" : "s"} may expire unused within 7 days.`,
      action: { href: "#inventory", label: "Inventory" },
    });
  }
  return items.sort((a, b) => SEVERITY_ORDER[a.level] - SEVERITY_ORDER[b.level]);
}

function StatusCell({ label, loading, value, unit, context, updated, children }) {
  return (
    <div className="status-cell">
      <p className="status-label">{label}</p>
      {loading ? (
        <><Skeleton width={72} height={28} /><Skeleton width="80%" height={12} style={{ marginTop: 8 }} /></>
      ) : (
        <>
          <p className="status-value">{value}{unit && <span className="status-unit"> {unit}</span>}</p>
          {context && <p className="status-context">{context}</p>}
          {children}
          {updated}
        </>
      )}
    </div>
  );
}

function sumForecast(outlook, key = "units") {
  return COMPONENTS.reduce(
    (acc, c) => acc + (outlook.component_forecast[c] || []).reduce((s, d) => s + d[key], 0), 0,
  );
}

// [summary key, singular, plural, tone]
const CYCLE_FACTS = [
  ["shortages", "projected shortage", "projected shortages", "risk"],
  ["donors_alerted", "donor alerted", "donors alerted", "neutral"],
  ["donations", "donation", "donations", "good"],
  ["transferred_units", "unit redistributed", "units redistributed", "good"],
  ["expired_units", "unit expired", "units expired", "risk"],
  ["escalations", "escalation", "escalations", "risk"],
];

export default function Overview({ canRun = false }) {
  const inventory = useLoad(loadInventory);
  const outlook = useLoad(loadOutlook);
  const agents = useLoad(loadAgents);
  const alerts = useLoad(loadAlerts);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState(null);

  const runCycle = async () => {
    setRunning(true);
    setRunError(null);
    try {
      await api.runAutonomyCycle();
      await Promise.all([inventory.reload(), outlook.reload(), agents.reload(), alerts.reload()]);
    } catch (e) {
      setRunError(e);
    } finally {
      setRunning(false);
    }
  };

  const inv = inventory.data;
  const out = outlook.data;
  const [status, metrics] = agents.data || [];
  const cycle = status?.last_cycle;
  const critical = (inv?.low_stock || []).filter((s) => s.status === "Critical");
  const items = attentionItems(inv, out);
  const attentionLoading = (inventory.loading && !inv) || (outlook.loading && !out);

  return (
    <>
      <PageHeader
        title="Supply overview"
        description="Today's stock, the week ahead, and what the agents have already handled."
        actions={canRun && (
          <button onClick={runCycle} disabled={running}>
            {running ? <RotateCw size={16} className="spin" aria-hidden="true" /> : <Play size={16} aria-hidden="true" />}
            {running ? "Running agents…" : "Run agent cycle"}
          </button>
        )}
      />
      {runError && <ErrorState what="The agent cycle didn't run." error={runError}
                               stillValid="The figures below are from the last successful load." />}

      <div className="overview">
        {/* 1. What is happening */}
        <section className="status-strip" aria-label="Current status">
          {inventory.error && !inv ? (
            <div className="status-error">
              <ErrorState what="Couldn't load inventory." error={inventory.error} onRetry={inventory.reload} />
            </div>
          ) : (
            <>
              <StatusCell label="Units in stock" loading={!inv} value={inv?.total_available} unit="units"
                          context={inv && `${inv.near_expiry_total} near expiry`}
                          updated={<UpdatedAt iso={inv?.as_of} />} />
              <StatusCell label="Critical stock" loading={!inv} value={critical.length}
                          unit={critical.length === 1 ? "group" : "groups"}
                          context={inv && (critical.length
                            ? critical.map((s) => `${s.blood_group} ${s.component}`).join(", ")
                            : "None below half of par")}>
                {inv && critical.length > 0 && <SeverityBadge level="critical" className="status-badge" />}
              </StatusCell>
            </>
          )}
          {outlook.error && !out ? (
            <div className="status-error">
              <ErrorState what="Couldn't load the forecast." error={outlook.error} onRetry={outlook.reload}
                          stillValid="Current inventory is still up to date." />
            </div>
          ) : (
            <>
              <StatusCell label="Projected shortages · 7 days" loading={!out} value={out?.shortages.length}
                          unit={out?.shortages.length === 1 ? "group" : "groups"}
                          context={out && (out.shortages.length
                            ? `${out.shortages.reduce((s, x) => s + x.projected_shortage, 0)} units short in total`
                            : "None projected")}>
                {out && out.shortages.length > 0 && <SeverityBadge level="high" className="status-badge" />}
              </StatusCell>
              <StatusCell label="Predicted demand · 7 days" loading={!out}
                          value={out && Math.round(sumForecast(out))} unit="units"
                          context={out && `80% range ${Math.round(sumForecast(out, "lower"))}–${Math.round(sumForecast(out, "upper"))} · model forecast`}
                          updated={<UpdatedAt iso={out?.as_of} prefix="Forecast" />} />
            </>
          )}
        </section>

        <div className="overview-main">
          {/* 2. What needs attention */}
          <section className="card" aria-labelledby="attention-title">
            <div className="card-head">
              <h2 id="attention-title">Needs attention</h2>
              {!attentionLoading && items.length > 0 && (
                <span className="muted">
                  {items.length} {items.length === 1 ? "item" : "items"}{outlook.error || inventory.error ? " · incomplete" : ""}
                </span>
              )}
            </div>
            {attentionLoading ? <SkeletonLines lines={4} label="Loading alerts" /> : items.length === 0 ? (
              inventory.error || outlook.error ? (
                <p className="muted">Can't check everything right now. See the errors above.</p>
              ) : (
                <EmptyState icon={CircleCheck} title="Nothing needs attention">
                  Every group is at par and no shortages or expiries are projected for the next 7 days.
                </EmptyState>
              )
            ) : (
              <>
              {(outlook.error || inventory.error) && (
                <p className="partial-note" role="status">
                  {outlook.error
                    ? "Forecast unavailable: projected shortages and expiries aren't included in this list."
                    : "Inventory unavailable: stock alerts aren't included in this list."}
                </p>
              )}
              <ul className="attention-list">
                {items.map((it) => (
                  <li key={it.key} className={`attention-item level-${it.level}`}>
                    <SeverityBadge level={it.level} className="attention-badge" />
                    <p><strong className="attention-subject">{it.subject}</strong> {it.text}</p>
                    <a className="text-link quiet" href={it.action.href}
                       aria-label={`${it.action.label}: ${it.subject}`}>
                      {it.action.label} <ArrowRight size={14} aria-hidden="true" />
                    </a>
                  </li>
                ))}
              </ul>
              </>
            )}
          </section>

          {/* 3. What the AI did and says */}
          <section aria-label="Latest agent cycle" className="overview-agent">
            {agents.error && !agents.data ? (
              <ErrorState what="Couldn't load agent activity." error={agents.error} onRetry={agents.reload} />
            ) : !agents.data ? (
              <div className="ai-panel"><SkeletonLines lines={5} label="Loading agent activity" /></div>
            ) : !cycle ? (
              <EmptyState title="The agents haven't run yet">
                {canRun ? "Run a cycle to check stock against the forecast and act on any gaps." : "The first cycle will run on the scheduler."}
              </EmptyState>
            ) : (
              <AIPanel label="HemoSmart AI · latest cycle">
                <p className="cycle-meta">
                  <UpdatedAt iso={cycle.started_at} prefix="Ran" />
                  <span className="muted"> · {TRIGGER_LABEL[cycle.trigger] || cycle.trigger}</span>
                  {cycle.status !== "ok" && <SeverityBadge level="moderate" label={`Cycle ${cycle.status}`} />}
                </p>
                {cycle.brief && <p className="cycle-brief">{cycle.brief}</p>}
                <ul className="cycle-facts">
                  {CYCLE_FACTS.filter(([k]) => cycle.summary[k] > 0).map(([k, one, many, tone]) => (
                    <li key={k} className={`fact ${tone}`}>
                      <span className="tabular">{cycle.summary[k]}</span> {cycle.summary[k] === 1 ? one : many}
                    </li>
                  ))}
                </ul>
                <a className="text-link" href="#agents">See what each agent did <ArrowRight size={14} aria-hidden="true" /></a>
              </AIPanel>
            )}
          </section>
        </div>

        <div className="overview-lower">
          {/* 4. Outcomes */}
          <section className="card" aria-labelledby="outcomes-title">
            <div className="card-head">
              <h2 id="outcomes-title">Unit outcomes · last 30 days</h2>
            </div>
            {!metrics ? (agents.error ? <p className="muted">Outcomes are unavailable until agent activity loads.</p> : <SkeletonLines lines={2} />) : (
              <dl className="outcomes">
                <div><dt>Issued</dt><dd>{metrics.totals.issued}</dd></div>
                <div><dt>Expired</dt><dd className={metrics.totals.expired > 0 ? "risk" : ""}>{metrics.totals.expired}</dd></div>
                <div><dt>Redistributed</dt><dd>{metrics.totals.transferred}</dd></div>
                <div>
                  <dt>Wastage rate</dt>
                  <dd>{metrics.wastage_rate == null ? "No units used" : `${(metrics.wastage_rate * 100).toFixed(1)}%`}</dd>
                </div>
              </dl>
            )}
          </section>

          <section className="card" aria-labelledby="donor-alerts-title">
            <div className="card-head">
              <h2 id="donor-alerts-title">Recent donor alerts</h2>
              <a className="text-link" href="#donorOutreach">Donor outreach <ArrowRight size={14} aria-hidden="true" /></a>
            </div>
            {alerts.error && !alerts.data ? (
              <ErrorState what="Couldn't load donor alerts." error={alerts.error} onRetry={alerts.reload} />
            ) : !alerts.data ? <SkeletonLines lines={4} /> : alerts.data.length === 0 ? (
              <EmptyState title="No donor alerts yet">Alerts sent by you or the agents appear here.</EmptyState>
            ) : (
              <ul className="plain-list">
                {alerts.data.map((a) => (
                  <li key={a.id}>
                    <span><strong>{a.blood_type}</strong> {a.component || ""} · {a.donor_id}</span>
                    <span className="list-meta">
                      {a.responded == null ? <SeverityBadge level="info" label="Pending" />
                        : a.responded ? <SeverityBadge level="low" label="Donated" />
                          : <span className="muted">No reply</span>}
                      <span className="muted">{a.cycle_id ? "auto · " : ""}{formatAgo(a.sent_at)}</span>
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>
      </div>
    </>
  );
}
