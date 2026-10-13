import { useEffect, useState } from "react";
import { Monitor, Moon, Sun } from "lucide-react";

// System by default; an explicit choice is stored per viewer. Dark mode
// matters here: clinical staff often work night shifts in dim wards.
const KEY = "hemosmart_theme";
const ORDER = ["system", "light", "dark"];
const NAME = { system: "System", light: "Light", dark: "Dark" };
const META = {
  system: { icon: Monitor, label: "Theme: system" },
  light: { icon: Sun, label: "Theme: light" },
  dark: { icon: Moon, label: "Theme: dark" },
};

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

export default function ThemeToggle({ compact = false }) {
  const [theme, setTheme] = useState(readTheme);

  useEffect(() => {
    applyTheme(theme);
    try {
      localStorage.setItem(KEY, theme);
    } catch {
      // preference just won't persist
    }
  }, [theme]);

  const next = ORDER[(ORDER.indexOf(theme) + 1) % ORDER.length];
  const { icon: Icon, label } = META[theme];
  return (
    <button className="ghost nav-tool" onClick={() => setTheme(next)}
            aria-label={`${label}. Switch to ${next}.`} title={`${label} — switch to ${next}`}>
      <Icon size={18} aria-hidden="true" />
      {!compact && <span className="nav-label">{NAME[theme]} theme</span>}
    </button>
  );
}
