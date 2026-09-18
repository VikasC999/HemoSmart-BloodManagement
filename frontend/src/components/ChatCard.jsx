import { useState } from "react";
import { api } from "../api";

export default function ChatCard() {
  const [sessionId, setSessionId] = useState(null);
  const [message, setMessage] = useState("");
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);

  const send = async () => {
    if (!message.trim()) return;
    const userMessage = message;
    setHistory((h) => [...h, { role: "user", text: userMessage }]);
    setMessage("");
    setLoading(true);
    try {
      const res = await api.chat(sessionId, userMessage);
      setSessionId(res.session_id);
      setHistory((h) => [...h, { role: "assistant", text: res.reply }]);
    } catch (e) {
      setHistory((h) => [...h, { role: "assistant", text: `Error: ${e.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="card">
      <h2>Chat Assistant</h2>
      <div className="chat-log">
        {history.map((m, i) => (
          <div key={i} className={`chat-msg ${m.role}`}><strong>{m.role}:</strong> {m.text}</div>
        ))}
      </div>
      <div className="actions">
        <input
          type="text"
          placeholder="e.g. hemoglobin 7.2, platelets 65000, INR 1.9, age 58, Emergency surgery"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
        />
        <button onClick={send} disabled={loading}>Send</button>
      </div>
    </section>
  );
}
