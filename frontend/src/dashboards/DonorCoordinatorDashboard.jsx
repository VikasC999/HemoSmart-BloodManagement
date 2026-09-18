import DonorAlertCard from "../components/DonorAlertCard";
import DonorManagementCard from "../components/DonorManagementCard";

export default function DonorCoordinatorDashboard() {
  return (
    <main className="grid">
      <DonorAlertCard />
      <DonorManagementCard />
    </main>
  );
}
