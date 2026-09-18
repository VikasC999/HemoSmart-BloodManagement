import ChatCard from "../components/ChatCard";
import PatientListCard from "../components/PatientListCard";
import PredictionCard from "../components/PredictionCard";
import UploadCard from "../components/UploadCard";

export default function HospitalStaffDashboard() {
  return (
    <main className="grid">
      <PredictionCard canAlertDonors={false} />
      <UploadCard />
      <ChatCard />
      <PatientListCard />
    </main>
  );
}
