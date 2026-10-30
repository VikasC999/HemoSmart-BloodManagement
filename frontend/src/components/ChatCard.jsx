import { useEffect, useRef, useState } from "react";
import { ArrowUp, Lock } from "lucide-react";
import { api } from "../api";
import { BuddyMark } from "./layout/Logo";
import { ErrorState } from "./ui";

// HemoBuddy, the assistant, for every role. What a role can ask is decided by the
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
    setHistory((h) => [...h, { kind: "user", text }]);
    setMessage("");
    setError(null);
    setLoading(true);
    try {
      const res = await api.chat(sessionId, text);
      setSessionId(res.session_id);
      // `kind`, not `role`: the reply carries the user's role and would overwrite it.
      setHistory((h) => [...h, { kind: "assistant", ...res }]);
    } catch (e) {
      setError({ error: e, text });
    } finally {
      setLoading(false);
    }
  };

  const topics = caps?.can_ask_about || [];
  const last = history[history.length - 1];
  const lastReply = last?.kind === "assistant" ? last : null;

  return (
    <div className="chat-card">
      <div className="chat-log" ref={log} aria-live="polite" aria-label="Conversation">
        {history.length === 0 && !loading && (
          <div className="chat-greeting">
            <BuddyMark size={52} />
            <p className="chat-greeting-title">Hi, I'm HemoBuddy. How can I help?</p>
            <p>
              {topics.length > 0
                ? <>Ask me about {topics.slice(0, 3).join(", ")}{topics.length > 3 ? " and more" : ""}. I only show what your role can see.</>
                : "Answers come from the same models and data as the rest of HemoSmart."}
            </p>
          </div>
        )}
        {history.map((m, i) => (
          m.kind === "user" ? (
            <div key={i} className="chat-msg user"><span className="visually-hidden">You: </span>{m.text}</div>
          ) : (
            <AssistantMessage key={i} reply={m} />
          )
        ))}
        {loading && (
          <div className="chat-row">
            <BuddyMark size={26} />
            <div className="chat-msg assistant typing" role="status" aria-label="HemoBuddy is typing">
              <span /><span /><span />
            </div>
          </div>
        )}
        {!loading && lastReply?.follow_ups?.length > 0 && (
          <div className="chat-followups" aria-label="Suggested follow-ups">
            {lastReply.follow_ups.map((q) => (
              <button key={q} className="chip-button" onClick={() => send(q)}>{q}</button>
            ))}
          </div>
        )}
      </div>
      {error && (
        <ErrorState what="HemoBuddy didn't reply." error={error.error} onRetry={() => {
          setHistory((h) => h.slice(0, -1));
          send(error.text);
        }} />
      )}
      {history.length === 0 && caps?.suggestions?.length > 0 && (
        <div className="chat-examples" aria-label="Example questions">
          <p className="chat-examples-label">Try asking</p>
          {caps.suggestions.map((ex) => (
            <button key={ex} className="chip-button" onClick={() => send(ex)}>{ex}</button>
          ))}
        </div>
      )}
      <div className="chat-compose">
        <form className="chat-input" onSubmit={(e) => { e.preventDefault(); send(); }}>
          <label htmlFor="chat-message" className="visually-hidden">Message</label>
          <input id="chat-message" type="text" placeholder="Type your question…"
                 value={message} onChange={(e) => setMessage(e.target.value)} autoComplete="off" />
          <button type="submit" className="send-button" disabled={loading || !message.trim()} aria-label="Send">
            <ArrowUp size={18} aria-hidden="true" />
          </button>
        </form>
        <p className="decision-support">Decision support. Review before acting.</p>
      </div>
    </div>
  );
}

function AssistantMessage({ reply }) {
  const parts = reply.parts?.length ? reply.parts : [{ allowed: true, text: reply.reply }];
  const answered = parts.filter((p) => p.allowed);
  const refused = parts.filter((p) => !p.allowed);
  return (
    <>
      {answered.length > 0 && (
        <div className="chat-row">
          <BuddyMark size={26} />
          <div className="chat-msg assistant">
            <span className="visually-hidden">HemoBuddy: </span>
            {reply.degraded && <p className="chat-degraded">Limited mode: the AI model is unavailable.</p>}
            {answered.map((p, i) => <ChatText key={i} text={p.text} />)}
          </div>
        </div>
      )}
      {refused.length > 0 && (
        <div className="chat-msg denied" role="note">
          <p className="denied-label"><Lock size={13} aria-hidden="true" /> Not authorised</p>
          {refused.map((p, i) => <p key={i} className="chat-text">{p.text.replace(/^Not authorised:\s*/, "")}</p>)}
          {reply.can_ask_about?.length > 0 && (
            <p className="chat-guide">As {reply.role}, you can ask about: {reply.can_ask_about.join(", ")}.</p>
          )}
        </div>
      )}
    </>
  );
}

// Replies are plain text; indented "label: value" lines (stock per
// component, demand per day) render as a compact table so they don't wrap.
function ChatText({ text }) {
  const blocks = [];
  for (const line of text.split("\n")) {
    const row = line.match(/^\s{2,}(.+?):\s+(.+)$/);
    const prev = blocks[blocks.length - 1];
    if (row) {
      if (prev?.rows) prev.rows.push(row.slice(1));
      else blocks.push({ rows: [row.slice(1)] });
    } else if (line.trim()) {
      blocks.push({ line });
    }
  }
  return blocks.map((b, i) => (b.rows ? (
    <dl key={i} className="chat-table">
      {b.rows.map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}
    </dl>
  ) : <p key={i} className="chat-text">{b.line}</p>));
}
