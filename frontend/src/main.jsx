import React from "react";
import ReactDOM from "react-dom/client";
import "@fontsource-variable/inter";
import App from "./App.jsx";
import { applyTheme, readTheme } from "./components/layout/ThemeToggle";
import "./index.css";

// Apply the saved theme before the first render (no light-mode flash).
applyTheme(readTheme());

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
