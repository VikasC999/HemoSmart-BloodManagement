import { useEffect, useState } from "react";
import { Check, Inbox, RotateCw, X } from "lucide-react";
import { api } from "../api";
import { EmptyState, ErrorState, SkeletonTable } from "./ui";
import { formatAgo } from "./ui/format";

// The bandit's learning signal: a coordinator records each donor's real
// reply. "Donated" also adds the donation's units to stock. (In a
// synthetic run with simulated replies, the autonomous loop resolves
// these itself.)
export default function DonorResponsesCard() {
  const [pending, setPending] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(null);
  const [last, setLast] = useState(null);

  const load = async () => {
    setError(null);
    setLoading(true);
    try {
      setPending(await api.recentDonorAlerts(25, true));
    } catch (e) {
      setError({ what: "Couldn't load pending alerts.", error: e });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const record = async (alert, responded) => {
    setBusy(alert.id);
    setError(null);
    try {
      const res = await api.recordDonorResponse(alert.id, responded);
      setLast(responded
        ? `${alert.donor_id} donated: ${res.units_received.length} unit${res.units_received.length === 1 ? "" : "s"} added to stock.`
        : `${alert.donor_id} recorded as no reply.`);
      await load();
    } catch (e) {
      setError({ what: `Couldn't record ${alert.donor_id}'s reply.`, error: e });
    } finally {
      setBusy(null);
    }
  };

  return (
    <section className="card" aria-labelledby="responses-title">
      <div className="card-head">
        <div>
          <h2 id="responses-title">Awaiting a reply</h2>
          {pending && <p className="card-sub">{pending.length} alert{pending.length === 1 ? "" : "s"} · oldest replies expire after 48 h</p>}
        </div>
        <button className="secondary" onClick={load} disabled={loading}>
          <RotateCw size={15} aria-hidden="true" /> Refresh
        </button>
      </div>
      {error && <ErrorState what={error.what} error={error.error} onRetry={load} />}
      {last && <p className="state empty" role="status">{last}</p>}
      {loading && !pending && <SkeletonTable rows={5} cols={4} label="Loading pending alerts" />}
      {pending && pending.length === 0 && (
        <EmptyState icon={Inbox} title="No alerts awaiting a reply">
          New alerts from you or the agents appear here until the donor replies.
        </EmptyState>
      )}
      {pending && pending.length > 0 && (
        <div className="table-scroll">
          <table>
            <thead><tr><th>Donor</th><th>Need</th><th>Sent</th><th>Record reply</th></tr></thead>
            <tbody>
              {pending.map((a) => (
                <tr key={a.id}>
                  <td className="nowrap">{a.donor_id}</td>
                  <td className="nowrap">
                    {a.blood_type} {a.component || "PRBC"}
                    {a.cycle_id && <span className="badge info">Agent</span>}
                  </td>
                  <td className="nowrap muted">{formatAgo(a.sent_at)}</td>
                  <td className="nowrap">
                    <div className="row-actions">
                      <button className="secondary" onClick={() => record(a, true)} disabled={busy === a.id}
                              aria-label={`${a.donor_id} donated`}>
                        <Check size={15} aria-hidden="true" /> Donated
                      </button>
                      <button className="ghost" onClick={() => record(a, false)} disabled={busy === a.id}
                              aria-label={`${a.donor_id} did not reply`}>
                        <X size={15} aria-hidden="true" /> No reply
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
