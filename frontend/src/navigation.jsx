// Role-specific sections for the app shell. Each section renders the
// same components (with the same props) the old stacked role dashboards
// did; only the page they appear on changed. The backend still enforces
// access on every call -- this decides what each role is shown.
import {
  Activity, Bot, ClipboardList, Droplets, FileUp, HeartHandshake, LayoutDashboard, MessageSquareText, Send,
  ShieldCheck, Stethoscope, Users,
} from "lucide-react";
import AgentMonitor from "./components/AgentMonitor";
import AuditLogCard from "./components/AuditLogCard";
import AutonomyCard from "./components/AutonomyCard";
import ChatCard from "./components/ChatCard";
import Overview from "./components/dashboard/Overview";
import DonorAlertCard from "./components/DonorAlertCard";
import DonorManagementCard from "./components/DonorManagementCard";
import DonorResponsesCard from "./components/DonorResponsesCard";
import ForecastChart from "./components/ForecastChart";
import InventoryCard from "./components/InventoryCard";
import PatientListCard from "./components/PatientListCard";
import PredictionCard from "./components/PredictionCard";
import UploadCard from "./components/UploadCard";

// layout: "stack" = full-width blocks in order; "split" = two columns on
// wide screens; "focus" = one narrow working column (forms).
const SECTIONS = {
  overview: {
    label: "Overview", icon: LayoutDashboard, layout: "wide", ownHeader: true,
    title: "Supply overview",
    render: ({ canRun }) => <Overview canRun={canRun} />,
  },
  inventory: {
    label: "Inventory & demand", icon: Droplets, layout: "split",
    title: "Inventory & demand",
    description: "Stock by blood group and component, and where the forecast says it will run short or expire unused.",
    render: ({ canEdit }) => <><InventoryCard editable={canEdit} /><ForecastChart /></>,
  },
  predict: {
    label: "Transfusion prediction", icon: Stethoscope, layout: "wide",
    title: "Transfusion prediction",
    description: "Enter a patient's lab values to see which blood components they're likely to need, and why.",
    render: ({ canAlertDonors }) => <PredictionCard canAlertDonors={canAlertDonors} />,
  },
  upload: {
    label: "CBC upload", icon: FileUp, layout: "focus",
    title: "CBC report upload",
    description: "Upload a CBC report or a batch export. You check the extracted values before anything is predicted.",
    render: () => <UploadCard />,
  },
  assistant: {
    label: "AI assistant", icon: MessageSquareText, layout: "focus",
    title: "AI assistant",
    description: "Ask in plain language about the data your role can see.",
    render: () => <ChatCard />,
  },
  stock: {
    label: "Inventory", icon: Droplets, layout: "stack",
    title: "Blood inventory",
    description: "Stock by blood group and component, to coordinate donor calls against what's running low.",
    render: () => <InventoryCard editable={false} />,
  },
  patients: {
    label: "Patients", icon: Users, layout: "stack",
    title: "Patients",
    description: "Recent patients and every prediction made for them.",
    render: () => <PatientListCard />,
  },
  donorResponses: {
    label: "Donor responses", icon: ClipboardList, layout: "stack",
    title: "Donor responses",
    description: "Donors who were alerted and haven't replied yet. Each reply you record teaches donor selection who responds.",
    render: () => <DonorResponsesCard />,
  },
  donorOutreach: {
    label: "Donor outreach", icon: Send, layout: "focus",
    title: "Donor outreach",
    description: "Alert the donors most likely to respond for a blood group and component.",
    render: () => <DonorAlertCard />,
  },
  donors: {
    label: "Donors", icon: HeartHandshake, layout: "stack",
    title: "Donors",
    description: "Everyone in the donor pool, and how often each donor replies to alerts.",
    render: () => <DonorManagementCard />,
  },
  agents: {
    label: "Agent activity", icon: Bot, layout: "stack",
    title: "Agent activity",
    description: "What the autonomous agents observed, decided and did in each cycle.",
    render: ({ canRun }) => (
      <>
        <p className="muted cross-link">Is each agent running, and how long do its steps take? See <a href="#agentMonitor">Agent monitor</a>.</p>
        <AutonomyCard canRun={canRun} />
      </>
    ),
  },
  agentMonitor: {
    label: "Agent monitor", icon: Activity, layout: "wide",
    title: "Agent monitor",
    description: "Are the agents running, how long each step takes, and what needs a person.",
    render: ({ canRun, links }) => <AgentMonitor canRun={canRun} links={links} />,
  },
  audit: {
    label: "Audit log", icon: ShieldCheck, layout: "stack",
    title: "Audit log",
    description: "Every API call, with who made it and when.",
    render: () => <AuditLogCard />,
  },
};

const item = (id, props = {}) => ({ id, ...SECTIONS[id], props });

// The pages an Agent monitor attention item can link to, per role:
// only sections that role actually has.
const MONITOR_LINKS = {
  "Blood Bank Manager": { agents: "Agent activity", donorOutreach: "Donor outreach" },
  "Auditor": { agents: "Agent activity" },
  "Admin": { agents: "Agent activity", donorOutreach: "Donor outreach", donorResponses: "Donor responses" },
};

// Groups only matter for Admin, who sees every area.
export const NAVIGATION = {
  "Blood Bank Manager": [{
    items: [
      item("overview", { canRun: true }),
      item("inventory", { canEdit: true }),
      item("agents", { canRun: true }),
      item("agentMonitor", { canRun: true, links: MONITOR_LINKS["Blood Bank Manager"] }),
      item("predict", { canAlertDonors: true }),
      item("donorOutreach"),
      item("patients"),
      item("assistant"),
    ],
  }],
  "Hospital Staff": [{
    items: [
      item("predict", { canAlertDonors: false }),
      item("upload"),
      item("patients"),
      item("assistant"),
    ],
  }],
  "Donor Coordinator": [{
    items: [item("donorResponses"), item("donorOutreach"), item("donors"), item("stock"), item("assistant")],
  }],
  "Auditor": [{
    items: [
      item("agents", { canRun: false }), item("agentMonitor", { canRun: false, links: MONITOR_LINKS.Auditor }),
      item("audit"), item("patients"), item("assistant"),
    ],
  }],
  "Admin": [
    { label: "Blood bank", items: [item("overview", { canRun: true }), item("inventory", { canEdit: true }), item("agents", { canRun: true })] },
    { label: "Clinical", items: [item("predict", { canAlertDonors: true }), item("upload"), item("patients")] },
    { label: "Donors", items: [item("donorResponses"), item("donorOutreach"), item("donors")] },
    { label: "Oversight", items: [item("agentMonitor", { canRun: true, links: MONITOR_LINKS.Admin }), item("audit")] },
    { label: "Assistant", items: [item("assistant")] },
  ],
};

export function sectionsFor(role) {
  return (NAVIGATION[role] || []).flatMap((g) => g.items);
}
