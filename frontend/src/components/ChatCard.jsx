import { useEffect, useRef, useState } from "react";
import { Lock, MessageSquareText, Send, Sparkles } from "lucide-react";
import { api } from "../api";
import { EmptyState, ErrorState, SkeletonLines } from "./ui";

// The assistant, for every role. What a role can ask is decided by the
// backend before any data is read (D66); refused parts of a reply come
// from code, not the AI, so they don't wear the AI label.
export default function ChatCard() {
  const [sessionId, setSessionId] = useState(null);
  const [message, setMessage] = useState("");
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [caps, setCaps] = useState(null);
  const log = useRef(null);

  useEffect(() => {
    api.chatCapabilities().then(setCaps).catch(() => setCaps(null));
  }, []);

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
      setHistory((h) => [...h, { role: "assistant", ...res }]);
    } catch (e) {
      setError({ error: e, text });
    } finally {
      setLoading(false);
    }
  };

  const topics = caps?.can_ask_about || [];

  return (
    <section className="card chat-card" aria-labelledby="chat-title">
      <h2 id="chat-title" className="visually-hidden">Conversation</h2>
      <div className="chat-log" ref={log} aria-live="polite">
        {history.length === 0 && !loading && (
          <EmptyState icon={MessageSquareText} title="Ask about the data your role can see">
            {topics.length > 0
              ? <>As {caps.role}, you can ask about {topics.join(", ")}.</>
              : "Answers come from the same models and data as the rest of HemoSmart."}
          </EmptyState>
        )}
        {history.map((m, i) => (
          m.role === "user" ? (
            <div key={i} className="chat-msg user"><span className="visually-hidden">You: </span>{m.text}</div>
          ) : (
            <AssistantMessage key={i} reply={m} />
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
        <ErrorState what="HemoSmart AI didn't reply." error={error.error} onRetry={() => {
          setHistory((h) => h.slice(0, -1));
          send(error.text);
        }} />
      )}
      {history.length === 0 && caps?.suggestions?.length > 0 && (
        <div className="chat-examples" aria-label="Example questions">
          {caps.suggestions.map((ex) => (
            <button key={ex} className="secondary chip-button" onClick={() => send(ex)}>{ex}</button>
          ))}
        </div>
      )}
      <form className="chat-input" onSubmit={(e) => { e.preventDefault(); send(); }}>
        <label htmlFor="chat-message" className="visually-hidden">Message</label>
        <input id="chat-message" type="text" placeholder="Ask a question"
               value={message} onChange={(e) => setMessage(e.target.value)} autoComplete="off" />
        <button type="submit" disabled={loading || !message.trim()} aria-label="Send">
          <Send size={16} aria-hidden="true" /> <span className="send-label">Send</span>
        </button>
      </form>
      <p className="decision-support">Decision support. Review before acting.</p>
    </section>
  );
}

function AssistantMessage({ reply }) {
  const parts = reply.parts?.length ? reply.parts : [{ allowed: true, text: reply.reply }];
  const answered = parts.filter((p) => p.allowed);
  const refused = parts.filter((p) => !p.allowed);
  return (
    <>
      {answered.length > 0 && (
        <div className="chat-msg assistant">
          <p className="ai-label"><Sparkles size={13} aria-hidden="true" /> HemoSmart AI</p>
          {reply.degraded && <p className="chat-degraded">Limited mode: the AI model is unavailable.</p>}
          {answered.map((p, i) => <p key={i} className="chat-text">{p.text}</p>)}
        </div>
      )}
      {refused.length > 0 && (
        <div className="chat-msg denied" role="note">
          <p className="denied-label"><Lock size={13} aria-hidden="true" /> Not authorised</p>
          {refused.map((p, i) => <p key={i} className="chat-text">{p.text}</p>)}
          {reply.can_ask_about?.length > 0 && (
            <p className="chat-guide">As {reply.role}, you can ask about: {reply.can_ask_about.join(", ")}.</p>
          )}
        </div>
      )}
    </>
  );
}
