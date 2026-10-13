import AlertsPanel from "../components/AlertsPanel";
import AutonomyCard from "../components/AutonomyCard";
import DonorAlertCard from "../components/DonorAlertCard";
import ForecastChart from "../components/ForecastChart";
import InventoryCard from "../components/InventoryCard";
import PatientListCard from "../components/PatientListCard";
import PredictionCard from "../components/PredictionCard";

export default function BloodBankManagerDashboard() {
  return (
    <main className="grid">
      <AutonomyCard canRun={true} />
      <AlertsPanel />
      <InventoryCard editable={true} />
      <ForecastChart />
      <PredictionCard canAlertDonors={true} />
      <DonorAlertCard />
      <PatientListCard />
    </main>
  );
}
