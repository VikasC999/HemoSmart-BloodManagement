import { useState } from "react";
import AuditorDashboard from "./AuditorDashboard";
import BloodBankManagerDashboard from "./BloodBankManagerDashboard";
import DonorCoordinatorDashboard from "./DonorCoordinatorDashboard";
import HospitalStaffDashboard from "./HospitalStaffDashboard";

// Admin has full access, so rather than one giant page combining every
// panel, it switches between the same role dashboards everyone else
// sees -- a direct way to demonstrate that role separation is real,
// not just a login-screen label.
const TABS = {
  "Hospital Staff": HospitalStaffDashboard,
  "Blood Bank Manager": BloodBankManagerDashboard,
  "Donor Coordinator": DonorCoordinatorDashboard,
  "Auditor": AuditorDashboard,
};

export default function AdminDashboard() {
  const [tab, setTab] = useState("Blood Bank Manager");
  const ActiveDashboard = TABS[tab];

  return (
    <div>
      <div className="admin-tabs">
        {Object.keys(TABS).map((t) => (
          <button
            key={t}
            className={t === tab ? "tab active" : "tab"}
            onClick={() => setTab(t)}
          >
            {t}
          </button>
        ))}
      </div>
      <ActiveDashboard />
    </div>
  );
}
