import AlertsPanel from "../components/AlertsPanel";
import DonorAlertCard from "../components/DonorAlertCard";
import ForecastChart from "../components/ForecastChart";
import InventoryCard from "../components/InventoryCard";
import PatientListCard from "../components/PatientListCard";
import PredictionCard from "../components/PredictionCard";

export default function BloodBankManagerDashboard() {
  return (
    <main className="grid">
      <AlertsPanel />
      <PredictionCard canAlertDonors={true} />
      <InventoryCard editable={true} />
      <ForecastChart />
      <DonorAlertCard />
      <PatientListCard />
    </main>
  );
}
