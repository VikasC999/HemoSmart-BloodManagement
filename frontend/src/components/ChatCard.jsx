import { useEffect, useRef, useState } from "react";
import { MessageSquareText, Send, Sparkles } from "lucide-react";
import { api } from "../api";
import { EmptyState, ErrorState, SkeletonLines } from "./ui";

const EXAMPLES = [
  "Hemoglobin 7.2, platelets 65000, INR 1.9, age 58, Emergency surgery",
  "How much O- blood do we have?",
  "What's the platelet demand forecast for next week?",
];

export default function ChatCard() {
  const [sessionId, setSessionId] = useState(null);
  const [message, setMessage] = useState("");
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const log = useRef(null);

  useEffect(() => {
    log.current?.scrollTo({ top: log.current.scrollHeight, behavior: "smooth" });
  }, [history, loading]);

  const send = async (text = message) => {
    if (!text.trim() || loading) return;
    setHistory((h) => [...h, { role: "user", text }]);
    setMessage("");
    setError(null);
    setLoading(true);
    try {
      const res = await api.chat(sessionId, text);
      setSessionId(res.session_id);
      setHistory((h) => [...h, { role: "assistant", text: res.reply }]);
    } catch (e) {
      setError({ error: e, text });
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="card chat-card" aria-labelledby="chat-title">
      <h2 id="chat-title" className="visually-hidden">Conversation</h2>
      <div className="chat-log" ref={log} aria-live="polite">
        {history.length === 0 && !loading && (
          <EmptyState icon={MessageSquareText} title="Ask about a patient, stock or the forecast">
            Answers come from the same models and data as the rest of HemoSmart.
          </EmptyState>
        )}
        {history.map((m, i) => (
          m.role === "user" ? (
            <div key={i} className="chat-msg user"><span className="visually-hidden">You: </span>{m.text}</div>
          ) : (
            <div key={i} className="chat-msg assistant">
              <p className="ai-label"><Sparkles size={13} aria-hidden="true" /> HemoSmart AI</p>
              <p className="chat-text">{m.text}</p>
            </div>
          )
        ))}
        {loading && (
          <div className="chat-msg assistant" aria-label="HemoSmart AI is answering">
            <p className="ai-label"><Sparkles size={13} aria-hidden="true" /> HemoSmart AI</p>
            <SkeletonLines lines={2} label="Writing a reply" />
          </div>
        )}
      </div>
      {error && (
        <ErrorState what="No reply this time." error={error.error} onRetry={() => {
          setHistory((h) => h.slice(0, -1));
          send(error.text);
        }} />
      )}
      {history.length === 0 && (
        <div className="chat-examples" aria-label="Example questions">
          {EXAMPLES.map((ex) => (
            <button key={ex} className="secondary chip-button" onClick={() => send(ex)}>{ex}</button>
          ))}
        </div>
      )}
      <form className="chat-input" onSubmit={(e) => { e.preventDefault(); send(); }}>
        <label htmlFor="chat-message" className="visually-hidden">Message</label>
        <input id="chat-message" type="text" placeholder="Type a question or a patient's lab values"
               value={message} onChange={(e) => setMessage(e.target.value)} autoComplete="off" />
        <button type="submit" disabled={loading || !message.trim()} aria-label="Send">
          <Send size={16} aria-hidden="true" /> <span className="send-label">Send</span>
        </button>
      </form>
      <p className="decision-support">Decision support. Review before acting.</p>
    </section>
  );
}
