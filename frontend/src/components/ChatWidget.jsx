import { useEffect, useRef, useState } from "react";
import { X } from "lucide-react";
import ChatCard from "./ChatCard";
import { BuddyMark } from "./layout/Logo";

// HemoBuddy lives in a corner panel on every page instead of its own
// section, so a question never costs the page you were on. It stays
// mounted while closed, so the conversation survives navigation.
export default function ChatWidget({ role }) {
  const [open, setOpen] = useState(false);
  const [started, setStarted] = useState(false);
  const launcher = useRef(null);
  const panel = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => e.key === "Escape" && setOpen(false);
    document.addEventListener("keydown", onKey);
    panel.current?.querySelector("input")?.focus();
    return () => {
      document.removeEventListener("keydown", onKey);
      launcher.current?.focus();
    };
  }, [open]);

  const toggle = () => {
    setStarted(true);
    setOpen((v) => !v);
  };

  return (
    <>
      <section ref={panel} id="assistant-panel" className={`chat-widget ${open ? "open" : ""}`}
               role="dialog" aria-modal="false" aria-labelledby="assistant-title"
               aria-hidden={!open} inert={open ? undefined : ""}>
        <header className="chat-widget-head">
          <BuddyMark size={36} />
          <div>
            <h2 id="assistant-title">HemoBuddy</h2>
            <p><span className="online-dot" aria-hidden="true" />{role ? `Here for you as ${role}` : "Online"}</p>
          </div>
          <button className="ghost icon-button" onClick={() => setOpen(false)} aria-label="Close HemoBuddy">
            <X size={18} aria-hidden="true" />
          </button>
        </header>
        {started && <ChatCard />}
      </section>
      <button ref={launcher} className={`chat-launcher ${open ? "is-open" : ""}`} onClick={toggle}
              aria-expanded={open} aria-controls="assistant-panel"
              aria-label={open ? "Close HemoBuddy" : "Open HemoBuddy, the HemoSmart assistant"}>
        {open ? <X size={20} aria-hidden="true" /> : <BuddyMark size={30} />}
        {!open && <span>Ask HemoBuddy</span>}
      </button>
    </>
  );
}
