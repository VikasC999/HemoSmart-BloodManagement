import AuditLogCard from "../components/AuditLogCard";
import PatientListCard from "../components/PatientListCard";

export default function AuditorDashboard() {
  return (
    <main className="grid">
      <AuditLogCard />
      <PatientListCard />
    </main>
  );
}
