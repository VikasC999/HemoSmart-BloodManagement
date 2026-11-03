import { useEffect, useState } from "react";
import { ArrowRight } from "lucide-react";
import { api } from "../api";
import { BloodGroup, EmptyState, ErrorState, SkeletonLines } from "./ui";
import { formatAgo } from "./ui/format";

const SOURCE_LABEL = { pdf: "PDF report", csv: "CSV batch", pdf_agents: "Full case (AI agents)" };

// Beside the upload form: the patients that came in from files, so staff can
// see their last upload landed.
export default function RecentUploads() {
  const [patients, setPatients] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.patients(40).then(setPatients).catch(setError);
  }, []);

  const uploaded = (patients || []).filter((p) => SOURCE_LABEL[p.source_format]).slice(0, 8);
  return (
    <section className="card" aria-labelledby="recent-uploads-title">
      <div className="card-head">
        <div>
          <h2 id="recent-uploads-title">Recent uploads</h2>
          <p className="card-sub">Patients added from files, newest first</p>
        </div>
        <a className="text-link" href="#patients">Patients <ArrowRight size={14} aria-hidden="true" /></a>
      </div>
      {error ? <ErrorState what="Couldn't load recent uploads." error={error} />
        : !patients ? <SkeletonLines lines={5} />
          : uploaded.length === 0 ? <EmptyState title="No uploads yet">Patients from PDF, CSV or full-case uploads appear here.</EmptyState>
            : (
              <ul className="plain-list">
                {uploaded.map((p) => (
                  <li key={p.id}>
                    <span>
                      <strong>#{p.id}</strong>{" "}
                      {p.blood_group ? <BloodGroup g={p.blood_group} /> : <span className="muted">Group unknown</span>}
                    </span>
                    <span className="list-meta muted">{SOURCE_LABEL[p.source_format]} · {formatAgo(p.created_at)}</span>
                  </li>
                ))}
              </ul>
            )}
    </section>
  );
}
