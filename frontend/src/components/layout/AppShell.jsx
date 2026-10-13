import { useEffect, useRef, useState } from "react";
import { LogOut, Menu, PanelLeftClose, PanelLeftOpen, X } from "lucide-react";
import { LogoMark, Wordmark } from "./Logo";
import ThemeToggle from "./ThemeToggle";

export function BrandMark({ size = 28 }) {
  return (
    <span className="brand">
      <LogoMark size={size} />
      <Wordmark />
    </span>
  );
}

const initials = (email = "") => email.split("@")[0].split(/[._-]/).map((p) => p[0] || "").join("").slice(0, 2).toUpperCase();

function NavList({ groups, activeId, onSelect, collapsed }) {
  return (
    <nav aria-label="Main" className="nav">
      {groups.map((group, gi) => (
        <div className="nav-group" key={group.label || gi}>
          {group.label && !collapsed && <p className="nav-group-label">{group.label}</p>}
          <ul>
            {group.items.map(({ id, label, icon: Icon }) => (
              <li key={id}>
                <a href={`#${id}`}
                   className={`nav-item ${id === activeId ? "active" : ""}`}
                   aria-current={id === activeId ? "page" : undefined}
                   title={collapsed ? label : undefined}
                   onClick={() => onSelect?.()}>
                  <Icon size={18} aria-hidden="true" />
                  <span className="nav-label">{label}</span>
                </a>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </nav>
  );
}

function SidebarFooter({ user, onLogout, collapsed }) {
  return (
    <div className="sidebar-footer">
      {user && !collapsed && (
        <div className="account-card">
          <span className="avatar" aria-hidden="true">{initials(user.email)}</span>
          <div className="account-text">
            <p className="account-role">{user.role}</p>
            <p className="account-email" title={user.email}>{user.email}</p>
          </div>
        </div>
      )}
      <ThemeToggle compact={collapsed} />
      <button className="ghost nav-tool" onClick={onLogout} aria-label="Sign out" title="Sign out">
        <LogOut size={18} aria-hidden="true" />
        {!collapsed && <span className="nav-label">Sign out</span>}
      </button>
    </div>
  );
}

const RAIL_KEY = "hemosmart_rail";

export default function AppShell({ user, groups, activeId, onLogout, children }) {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [railCollapsed, setRailCollapsed] = useState(() => {
    try {
      return localStorage.getItem(RAIL_KEY) === "1";
    } catch {
      return false;
    }
  });
  const menuButton = useRef(null);
  const drawer = useRef(null);

  useEffect(() => {
    try {
      localStorage.setItem(RAIL_KEY, railCollapsed ? "1" : "0");
    } catch {
      // per-viewer convenience only
    }
  }, [railCollapsed]);

  // Drawer: Escape closes, focus moves in on open and back on close,
  // page behind doesn't scroll.
  useEffect(() => {
    if (!drawerOpen) return undefined;
    const onKey = (e) => e.key === "Escape" && setDrawerOpen(false);
    document.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    drawer.current?.querySelector("a, button")?.focus();
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
      menuButton.current?.focus();
    };
  }, [drawerOpen]);

  return (
    <div className={`shell ${railCollapsed ? "rail-collapsed" : ""}`}>
      <a className="skip-link" href="#main">Skip to content</a>

      {/* Desktop / tablet sidebar */}
      <aside className="sidebar" aria-label="Sidebar">
        <div className="sidebar-head">
          <BrandMark />
          <button className="ghost icon-button rail-toggle"
                  onClick={() => setRailCollapsed((v) => !v)}
                  aria-label={railCollapsed ? "Expand sidebar" : "Collapse sidebar"}
                  aria-expanded={!railCollapsed}>
            {railCollapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
          </button>
        </div>
        <NavList groups={groups} activeId={activeId} collapsed={railCollapsed} />
        <SidebarFooter user={user} onLogout={onLogout} collapsed={railCollapsed} />
      </aside>

      {/* Mobile top bar + drawer */}
      <header className="topbar">
        <button ref={menuButton} className="ghost icon-button" onClick={() => setDrawerOpen(true)}
                aria-label="Open menu" aria-expanded={drawerOpen} aria-controls="mobile-drawer">
          <Menu size={20} />
        </button>
        <BrandMark size={26} />
      </header>
      <div className={`drawer-scrim ${drawerOpen ? "open" : ""}`} onClick={() => setDrawerOpen(false)}
           aria-hidden="true" />
      <aside id="mobile-drawer" ref={drawer} className={`drawer ${drawerOpen ? "open" : ""}`}
             aria-label="Menu" aria-hidden={!drawerOpen} inert={drawerOpen ? undefined : ""}>
        <div className="sidebar-head">
          <BrandMark />
          <button className="ghost icon-button" onClick={() => setDrawerOpen(false)} aria-label="Close menu">
            <X size={20} />
          </button>
        </div>
        <NavList groups={groups} activeId={activeId} onSelect={() => setDrawerOpen(false)} />
        <SidebarFooter user={user} onLogout={onLogout} />
      </aside>

      <main id="main" className="main" tabIndex={-1}>{children}</main>
    </div>
  );
}

export function PageHeader({ title, description, actions }) {
  return (
    <div className="page-header">
      <div>
        <h1>{title}</h1>
        {description && <p>{description}</p>}
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </div>
  );
}
