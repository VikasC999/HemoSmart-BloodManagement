import { useEffect, useState } from "react";
import { HeartHandshake, RotateCw } from "lucide-react";
import { api } from "../api";
import { BLOOD_TYPES } from "../constants";
import { EmptyState, ErrorState, SkeletonTable } from "./ui";

export default function DonorManagementCard() {
  const [bloodType, setBloodType] = useState("");
  const [donors, setDonors] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  const load = async (type = bloodType) => {
    setError(null);
    setLoading(true);
    try {
      setDonors(await api.donors(type || undefined));
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(""); }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  const filter = (type) => {
    setBloodType(type);
    load(type);
  };

  return (
    <section className="card" aria-labelledby="donors-title">
      <div className="card-head">
        <div>
          <h2 id="donors-title">Donor pool</h2>
          {donors && <p className="card-sub">{donors.length} donors{bloodType ? ` · ${bloodType}` : ""}</p>}
        </div>
        <div className="actions" style={{ margin: 0 }}>
          <label className="inline">
            <span className="visually-hidden">Blood type</span>
            <select value={bloodType} onChange={(e) => filter(e.target.value)}>
              <option value="">All blood types</option>
              {BLOOD_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </label>
          <button className="secondary" onClick={() => load()} disabled={loading}>
            <RotateCw size={15} aria-hidden="true" /> Refresh
          </button>
        </div>
      </div>
      <p className="hint">
        Learned reply rate: what the donor-selection model currently believes about each donor, from their
        reply history. Phone numbers are never shown here.
      </p>
      {error && <ErrorState what="Couldn't load donors." error={error} onRetry={() => load()} />}
      {loading && !donors && <SkeletonTable rows={8} cols={5} label="Loading donors" />}
      {donors && donors.length === 0 && (
        bloodType ? (
          <EmptyState icon={HeartHandshake} title={`No ${bloodType} donors`}
                      action={<button className="secondary" onClick={() => filter("")}>Clear filter</button>}>
            Try another blood type.
          </EmptyState>
        ) : <EmptyState icon={HeartHandshake} title="No donors registered yet" />
      )}
      {donors && donors.length > 0 && (
        <div className="table-scroll table-tall">
          <table>
            <thead>
              <tr>
                <th>Donor</th><th>Name</th><th>Group</th><th>Last donation</th>
                <th className="num">Replies</th><th className="num">Learned reply rate</th>
              </tr>
            </thead>
            <tbody>
              {donors.map((d) => (
                <tr key={d.donor_id}>
                  <td className="nowrap">{d.donor_id}</td>
                  <td>{d.name}</td>
                  <td>{d.blood_type}</td>
                  <td className="nowrap tabular">{d.last_donation_date}</td>
                  <td className="num">{d.alerts_responded} of {d.alerts_sent}</td>
                  <td className="num">
                    <span className="rate">
                      <span className="rate-track" aria-hidden="true">
                        <span className="rate-bar" style={{ width: `${Math.round(d.expected_response_rate * 100)}%` }} />
                      </span>
                      {Math.round(d.expected_response_rate * 100)}%
                    </span>
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
