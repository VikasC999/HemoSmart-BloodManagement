import { useEffect, useState } from "react";
import { auth } from "./api";
import LoginScreen from "./components/LoginScreen";
import AdminDashboard from "./dashboards/AdminDashboard";
import AuditorDashboard from "./dashboards/AuditorDashboard";
import BloodBankManagerDashboard from "./dashboards/BloodBankManagerDashboard";
import DonorCoordinatorDashboard from "./dashboards/DonorCoordinatorDashboard";
import HospitalStaffDashboard from "./dashboards/HospitalStaffDashboard";

const DASHBOARDS = {
  "Admin": AdminDashboard,
  "Blood Bank Manager": BloodBankManagerDashboard,
  "Hospital Staff": HospitalStaffDashboard,
  "Donor Coordinator": DonorCoordinatorDashboard,
  "Auditor": AuditorDashboard,
};

export default function App() {
  const [loggedIn, setLoggedIn] = useState(auth.isLoggedIn());
  const [user, setUser] = useState(null);

  useEffect(() => {
    if (loggedIn) {
      auth.me().then(setUser).catch(() => {});
    }
  }, [loggedIn]);

  if (!loggedIn) {
    return <LoginScreen onLoggedIn={() => setLoggedIn(true)} />;
  }

  const logout = () => {
    auth.logout();
    setLoggedIn(false);
    setUser(null);
  };

  const RoleDashboard = user ? DASHBOARDS[user.role] : null;

  return (
    <div className="app">
      <header className="header-row">
        <div>
          <h1>HemoSmart</h1>
          <p>AI-assisted blood transfusion prediction &amp; supply coordination</p>
        </div>
        <div className="account">
          {user && <span className="account-info">{user.email} · {user.role}</span>}
          <button onClick={logout}>Sign Out</button>
        </div>
      </header>
      {RoleDashboard ? <RoleDashboard /> : <p>Loading your dashboard…</p>}
    </div>
  );
}
