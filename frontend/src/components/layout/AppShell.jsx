import { Fragment, useEffect, useRef, useState } from "react";
import { ChevronDown, LogOut } from "lucide-react";
import { LogoMark, Wordmark } from "./Logo";
import ThemeChoice from "./ThemeToggle";

export function BrandMark({ size = 28 }) {
  return (
    <span className="brand">
      <LogoMark size={size} />
      <Wordmark />
    </span>
  );
}

// "Blood Bank Manager" -> "BM": the role is what the menu is about.
const initials = (role = "") => {
  const w = role.trim().split(/\s+/);
  return (w.length > 1 ? w[0][0] + w[w.length - 1][0] : role.slice(0, 2)).toUpperCase();
};

// Top right: who is signed in. Opens the account menu: who is signed in,
// the theme, and sign out.
function AccountMenu({ user, onLogout }) {
  const [open, setOpen] = useState(false);
  const root = useRef(null);
  const trigger = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => {
      if (e.key === "Escape") {
        setOpen(false);
        trigger.current?.focus();
      }
    };
    const onDown = (e) => !root.current?.contains(e.target) && setOpen(false);
    document.addEventListener("keydown", onKey);
    document.addEventListener("pointerdown", onDown);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("pointerdown", onDown);
    };
  }, [open]);

  return (
    <div className="account-menu" ref={root}>
      <button ref={trigger} type="button" className="account-trigger" onClick={() => setOpen((v) => !v)}
              aria-expanded={open} aria-controls="account-panel"
              aria-label={user ? `Account: ${user.role}. Theme and sign out` : "Account"}>
        <span className="avatar" aria-hidden="true">{user ? initials(user.role) : ""}</span>
        {user && <span className="account-trigger-role">{user.role}</span>}
        <ChevronDown size={16} aria-hidden="true" className="account-chevron" />
      </button>
      {open && (
        <div id="account-panel" className="account-panel">
          {user && (
            <div className="account-who">
              <p className="account-role">{user.role}</p>
              <p className="account-email" title={user.email}>{user.email}</p>
            </div>
          )}
          <p className="account-section">Theme</p>
          <ThemeChoice />
          <button type="button" className="ghost account-signout" onClick={onLogout}>
            <LogOut size={16} aria-hidden="true" />Sign out
          </button>
        </div>
      )}
    </div>
  );
}

// Magnification like a macOS dock: items near the cursor grow, the rest stay put.
const BASE = 44;
const GROW = 0.42;
const REACH = 120;

function Dock({ groups, activeId }) {
  const ref = useRef(null);
  const frame = useRef(0);

  const still = () => window.matchMedia("(prefers-reduced-motion: reduce), (pointer: coarse)").matches;
  const reset = () => {
    cancelAnimationFrame(frame.current);
    ref.current?.querySelectorAll(".dock-item").forEach((el) => el.style.removeProperty("--s"));
  };
  const magnify = (e) => {
    if (still()) return;
    const x = e.clientX;
    cancelAnimationFrame(frame.current);
    frame.current = requestAnimationFrame(() => {
      const items = [...ref.current.querySelectorAll(".dock-item")];
      const centers = items.map((el) => { const r = el.getBoundingClientRect(); return r.left + r.width / 2; });
      items.forEach((el, i) => {
        const d = Math.abs(x - centers[i]);
        el.style.setProperty("--s", d < REACH ? 1 + GROW * Math.cos((d / REACH) * (Math.PI / 2)) : 1);
      });
    });
  };
  useEffect(() => () => cancelAnimationFrame(frame.current), []);

  return (
    <nav className="dock" aria-label="Main" ref={ref} onPointerMove={magnify} onPointerLeave={reset}
         style={{ "--dock-base": `${BASE}px` }}>
      {groups.map((group, gi) => (
        <Fragment key={group.label || gi}>
          {gi > 0 && <span className="dock-sep" aria-hidden="true" />}
          {group.items.map(({ id, label, icon: Icon }) => (
            <a key={id} href={`#${id}`} className={`dock-item ${id === activeId ? "active" : ""}`}
               aria-current={id === activeId ? "page" : undefined} aria-label={label} data-label={label}>
              <Icon aria-hidden="true" />
            </a>
          ))}
        </Fragment>
      ))}
    </nav>
  );
}

export default function AppShell({ user, groups, activeId, onLogout, children }) {
  return (
    <div className="shell">
      <a className="skip-link" href="#main">Skip to content</a>
      <header className="appbar">
        <div className="appbar-inner">
          <BrandMark size={32} />
          <AccountMenu user={user} onLogout={onLogout} />
        </div>
      </header>
      <main id="main" className="main" tabIndex={-1}>{children}</main>
      {groups.length > 0 && <Dock groups={groups} activeId={activeId} />}
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
