import AuditLogCard from "../components/AuditLogCard";
import AutonomyCard from "../components/AutonomyCard";
import PatientListCard from "../components/PatientListCard";

export default function AuditorDashboard() {
  return (
    <main className="grid">
      <AutonomyCard canRun={false} />
      <AuditLogCard />
      <PatientListCard />
    </main>
  );
}
