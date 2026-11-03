import { useEffect, useState } from "react";
import { Monitor, Moon, Sun } from "lucide-react";

// System by default; an explicit choice is stored per viewer. Dark mode
// matters here: clinical staff often work night shifts in dim wards.
const KEY = "hemosmart_theme";
const OPTIONS = [
  { id: "system", label: "System", icon: Monitor },
  { id: "light", label: "Light", icon: Sun },
  { id: "dark", label: "Dark", icon: Moon },
];

export function readTheme() {
  try {
    return localStorage.getItem(KEY) || "system";
  } catch {
    return "system";
  }
}

export function applyTheme(theme) {
  const root = document.documentElement;
  if (theme === "system") delete root.dataset.theme;
  else root.dataset.theme = theme;
}

export default function ThemeChoice() {
  const [theme, setTheme] = useState(readTheme);

  useEffect(() => {
    applyTheme(theme);
    try {
      localStorage.setItem(KEY, theme);
    } catch {
      // preference just won't persist
    }
  }, [theme]);

  return (
    <div className="theme-choice" role="group" aria-label="Theme">
      {OPTIONS.map(({ id, label, icon: Icon }) => (
        <button key={id} type="button" className={`theme-option ${theme === id ? "active" : ""}`}
                aria-pressed={theme === id} onClick={() => setTheme(id)}>
          <Icon size={15} aria-hidden="true" />{label}
        </button>
      ))}
    </div>
  );
}
