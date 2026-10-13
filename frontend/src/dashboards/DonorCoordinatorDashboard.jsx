import DonorAlertCard from "../components/DonorAlertCard";
import DonorManagementCard from "../components/DonorManagementCard";
import DonorResponsesCard from "../components/DonorResponsesCard";

export default function DonorCoordinatorDashboard() {
  return (
    <main className="grid">
      <DonorResponsesCard />
      <DonorAlertCard />
      <DonorManagementCard />
    </main>
  );
}
