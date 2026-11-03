import { useCallback, useEffect, useState } from "react";
import { ArrowRight, CircleCheck, Play, RotateCw, TriangleAlert } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../../api";
import { COMPONENTS } from "../../constants";
import { PageHeader } from "../layout/AppShell";
import { AlertList } from "../RecentDonorAlerts";
import {
  AIPanel, BloodGroup, EmptyState, ErrorState, SeverityBadge, Skeleton, SkeletonLines, UpdatedAt, stockSeverity,
} from "../ui";
import { TRIGGER_LABEL, formatAgo, formatDay, groupLabel, tidyText } from "../ui/format";

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

const shortDate = (iso) => formatDay(iso);

// What the latest cycle actually did, step by step: only steps with an
// outcome, or that had to fall back. The full log lives on Agent activity.
const plural = (n, one, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;
const STEPS = [
  ["Donor replies", "Donor", (s) => s.donor_replies && `${plural(s.donor_replies, "reply", "replies")} · ${s.donations} donated`],
  ["Inventory", "Inventory", (s, ev) => s.expired_units && `${expiredLabel(ev) || s.expired_units} expired`],
  ["Forecast", "Forecast", () => null],
  ["Planner", "Planner", (s) => (s.shortages || s.wastage_risk_units)
    && [s.shortages && plural(s.shortages, "shortage"), s.wastage_risk_units && `${s.wastage_risk_units} at risk of expiry`].filter(Boolean).join(" · ")],
  ["Donor outreach", "Donor", (s) => (s.donors_alerted || s.escalations)
    && [s.donors_alerted && `${s.donors_alerted} alerted`, s.escalations && `${s.escalations} escalated to the regional bank`].filter(Boolean).join(" · ")],
  ["Logistics", "Logistics", (s) => s.transferred_units && `${s.transferred_units} sent to partner`],
  ["Explanation", "Explanation", () => null],
];
// "1 O+ PRBC": which units expired, from the Inventory agent's own record.
function expiredLabel(events) {
  const by = events.find((e) => e.agent === "Inventory" && e.kind === "act" && e.data?.by_group_component)?.data.by_group_component;
  return by && Object.entries(by).map(([k, n]) => `${n} ${k}`).join(", ");
}
function agentSteps(cycle) {
  const s = cycle.summary || {};
  const troubled = new Set((cycle.events || []).filter((e) => e.kind === "fallback" || e.kind === "error").map((e) => e.agent));
  return STEPS
    .map(([name, agent, result]) => ({ name, result: result(s, cycle.events || []) || null, ok: !troubled.has(agent) }))
    .map((st) => (st.ok ? st : { ...st, result: st.result || "fell back" }))
    .filter((st) => st.result);
}

// The handbook pages behind the brief's [n] markers, so they point somewhere.
function cycleSources(events = []) {
  return events.filter((e) => e.agent === "Explanation" && e.data?.citations?.length).flatMap((e) => {
    const topic = (e.message.match(/^Grounded (.+?) in the WHO handbook/) || [])[1]?.replace(/\s*\(.*\)$/, "");
    return e.data.citations.map((c) => ({ ref: c.ref, page: c.page, topic }));
  });
}

// What needs attention, worst first. Every item says what is wrong and
// where to act (HemoSmart §12: icon + label + sentence).
function attentionItems(inventory, outlook) {
  const items = [];
  for (const s of inventory?.low_stock || []) {
    if (s.status === "Critical") {
      items.push({
        level: "critical", key: `crit-${s.blood_group}-${s.component}`,
        subject: `${groupLabel(s.blood_group)} ${s.component}`, group: s.blood_group, component: s.component,
        text: `${s.available} unit${s.available === 1 ? "" : "s"} left against a par of ${Math.ceil(s.par)}.`,
        action: { href: "#donorOutreach", label: "Alert donors" },
      });
    }
  }
  for (const s of outlook?.shortages || []) {
    items.push({
      level: "high", key: `short-${s.blood_group}-${s.component}`,
      subject: `${groupLabel(s.blood_group)} ${s.component}`, group: s.blood_group, component: s.component,
      text: `Forecast to run ${s.projected_shortage} unit${s.projected_shortage === 1 ? "" : "s"} short from ${shortDate(s.first_shortage_date)}.`,
      action: { href: "#donorOutreach", label: "Alert donors" },
    });
  }
  for (const s of inventory?.low_stock || []) {
    if (s.status === "Low") {
      items.push({
        level: "moderate", key: `low-${s.blood_group}-${s.component}`,
        subject: `${groupLabel(s.blood_group)} ${s.component}`, group: s.blood_group, component: s.component,
        text: `Below par: ${s.available} of ${Math.ceil(s.par)} units.`,
        action: { href: "#inventory", label: "Inventory" },
      });
    }
  }
  for (const w of outlook?.wastage_risks || []) {
    items.push({
      level: "moderate", key: `waste-${w.blood_group}-${w.component}`,
      subject: `${groupLabel(w.blood_group)} ${w.component}`, group: w.blood_group, component: w.component,
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

// The page's opening sentence: the whole state in words, before any number.
function lede(inv, out, cycle) {
  const parts = [];
  if (inv) {
    const crit = inv.low_stock.filter((s) => s.status === "Critical").map((s) => `${groupLabel(s.blood_group)} ${s.component}`);
    parts.push(crit.length === 0 ? "Every group is above half of par"
      : crit.length === 1 ? `${crit[0]} is critical` : `${crit.length} groups are critical`);
  }
  if (out) {
    const n = out.shortages.length;
    const first = out.shortages[0];
    parts.push(n === 0 ? "nothing is forecast to run short this week"
      : n === 1 ? `${groupLabel(first.blood_group)} ${first.component} is forecast to run short from ${shortDate(first.first_shortage_date)}`
        : `${n} groups are forecast to run short this week`);
  }
  let text = parts.length ? `${parts.join(", and ")}.` : "";
  const s = cycle?.summary;
  if (s) {
    const did = [s.donors_alerted && `alerted ${plural(s.donors_alerted, "donor")}`,
      s.transferred_units && `sent ${plural(s.transferred_units, "near-expiry unit")} to a partner hospital`].filter(Boolean);
    text += did.length ? ` On their last cycle the agents ${did.join(" and ")}.` : " The agents' last cycle alerted no donors and moved no units.";
  }
  return text.charAt(0).toUpperCase() + text.slice(1);
}

const SERIES = { PRBC: "var(--chart-prbc)", Platelets: "var(--chart-platelets)", FFP: "var(--chart-ffp)" };
const TOOLTIP_STYLE = {
  background: "var(--color-raised)", border: "1px solid var(--color-border)",
  borderRadius: 8, color: "var(--color-text)", fontSize: 13,
};
const dayLabel = (iso) => new Date(`${iso}T00:00:00`).toLocaleDateString("en-GB", { weekday: "short", day: "numeric" });


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
        description={lede(inv, out, cycle) || "Today's stock, the week ahead, and what the agents have already handled."}
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
                            ? critical.map((s) => `${groupLabel(s.blood_group)} ${s.component}`).join(", ")
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
                          updated={<UpdatedAt iso={out?.as_of} prefix="Forecast" />}
                          unit={out?.shortages.length === 1 ? "group" : "groups"}
                          context={out && (out.shortages.length
                            ? ((n) => `${n} ${n === 1 ? "unit" : "units"} short in total`)(out.shortages.reduce((s, x) => s + x.projected_shortage, 0))
                            : "None projected")}>
                {out && out.shortages.length > 0 && <SeverityBadge level="high" className="status-badge" />}
              </StatusCell>
            </>
          )}
        </section>

        <div className="overview-main">
          {/* 2. What needs attention */}
          <section className="card attention-card" aria-labelledby="attention-title">
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
              {/* Worst first; the card is as tall as the AI panel beside it and the rest scroll. */}
              <ul className="attention-list attention-scroll" tabIndex={0} aria-label="Items needing attention">
                {items.map((it) => (
                  <li key={it.key} className={`attention-item level-${it.level}`}>
                    <SeverityBadge level={it.level} className="attention-badge" />
                    <p><span className="attention-subject"><BloodGroup g={it.group} /> {it.component}</span> {it.text}</p>
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
                <a className="text-link ai-log-link" href="#agents">Full agent log <ArrowRight size={14} aria-hidden="true" /></a>
                <p className="cycle-meta">
                  <UpdatedAt iso={cycle.started_at} prefix="Ran" />
                  <span className="muted"> · {TRIGGER_LABEL[cycle.trigger] || cycle.trigger}</span>
                  {cycle.status !== "ok" && <SeverityBadge level="moderate" label={`Cycle ${cycle.status}`} />}
                </p>
                {cycle.brief && <p className="cycle-brief">{tidyText(cycle.brief)}</p>}
                <ol className="agent-steps" aria-label="Steps of the latest cycle">
                  {agentSteps(cycle).map((st) => (
                    <li key={st.name} className={st.ok ? "" : "problem"}>
                      {st.ok ? <CircleCheck size={14} aria-label="Done" /> : <TriangleAlert size={14} aria-label="Fell back" />}
                      <span className="agent-step-name">{st.name}</span>
                      <span className="agent-step-text">{st.result}</span>
                    </li>
                  ))}
                </ol>
                {cycleSources(cycle.events).length > 0 && (
                  <ol className="cycle-sources" aria-label="Sources">
                    {cycleSources(cycle.events).map((src) => (
                      <li key={src.ref}><span>[{src.ref}]</span> WHO handbook, p.{src.page}{src.topic ? ` · ${src.topic}` : ""}</li>
                    ))}
                  </ol>
                )}
              </AIPanel>
            )}
          </section>
        </div>

        {/* 4. The week ahead */}
        {out && (
          <section className="card week-card" aria-labelledby="week-title">
            <div className="card-head">
              <div>
                <h2 id="week-title">The week ahead</h2>
                <p className="card-sub">
                  {Math.round(sumForecast(out))} units forecast to be needed over 7 days
                  · 80% range {Math.round(sumForecast(out, "lower"))}–{Math.round(sumForecast(out, "upper"))}
                </p>
              </div>
              <a className="text-link" href="#inventory">Inventory &amp; demand <ArrowRight size={14} aria-hidden="true" /></a>
            </div>
            <div className="week-chart" role="img"
                 aria-label={`Forecast units needed per day for the next 7 days, by component: ${Math.round(sumForecast(out))} in total.`}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={out.component_forecast[COMPONENTS[0]].map((d, i) => ({
                  day: dayLabel(d.date),
                  ...Object.fromEntries(COMPONENTS.map((c) => [c, Math.round(out.component_forecast[c][i].units)])),
                }))} margin={{ top: 4, right: 4, bottom: 0, left: -20 }} barCategoryGap="28%">
                  <CartesianGrid vertical={false} stroke="var(--color-gridline)" />
                  <XAxis dataKey="day" tickLine={false} axisLine={false} tick={{ fill: "var(--color-text-secondary)", fontSize: 12 }} />
                  <YAxis tickLine={false} axisLine={false} allowDecimals={false} tick={{ fill: "var(--color-text-secondary)", fontSize: 12 }} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "var(--color-beige)", opacity: 0.5 }} formatter={(v) => `${v} units`} />
                  <Legend iconType="square" iconSize={10} wrapperStyle={{ fontSize: 13, color: "var(--color-text-secondary)" }} />
                  {COMPONENTS.map((c, i) => (
                    <Bar key={c} dataKey={c} stackId="d" fill={SERIES[c]} radius={i === COMPONENTS.length - 1 ? [4, 4, 0, 0] : 0} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>
        )}

        {/* 5. The last 30 days */}
        <div className="overview-lower">
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
            ) : !alerts.data ? <SkeletonLines lines={4} /> : <AlertList alerts={alerts.data} columns />}
          </section>
        </div>
      </div>
    </>
  );
}
